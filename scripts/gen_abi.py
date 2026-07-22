#!/usr/bin/env python3
"""Regenerate all shader ABI outputs from their .abi schemas.

Pass --check to verify the committed outputs are current instead of
rewriting them (exits nonzero on drift). Library outputs ship with the
package; test outputs are generated from test-owned schemas under test/abi/.

Portable replacement for the old gen_abi.sh — runs anywhere python3 and c3c
do, including native Windows (no bash required).
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_ABI_START = "// BEGIN GENERATED SHADER ABI - do not edit."
PUBLIC_ABI_END = "// END GENERATED SHADER ABI"


def generator_binary(tool_dir):
    exe = tool_dir / "build" / "gen_shader_abi.exe"
    return exe if exe.exists() else tool_dir / "build" / "gen_shader_abi"


def build_generator():
    tool_dir = ROOT / "tools" / "gen_shader_abi"
    c3c = shutil.which(os.environ.get("C3C", "c3c"))
    if c3c is None:
        sys.exit("gen_abi: c3c not found (set C3C or add it to PATH)")
    subprocess.run(
        [c3c, "build", "gen_shader_abi", "--path", str(tool_dir)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return generator_binary(tool_dir)


def gen(gen_bin, check, module, c3_out, glsl_out, schemas):
    Path(glsl_out).parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(gen_bin), "--module", module, "--c3-out", str(c3_out), "--glsl-out", str(glsl_out)]
    if check:
        cmd.append("--check")
    cmd += [str(s) for s in schemas]
    subprocess.run(cmd, check=True)


def extract_braced_declaration(source, start):
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                if end < len(source) and source[end] == ";":
                    end += 1
                if end < len(source) and source[end] == "\n":
                    end += 1
                return end
    raise ValueError("unterminated generated C3 declaration")


def split_gpu_c3(source):
    lines = source.splitlines(keepends=True)
    if len(lines) < 2 or lines[1] != "module gpu;\n":
        raise ValueError("unexpected generated gpu module header")

    header = lines[0]
    body = "".join(lines[2:])
    member_start = body.index("struct RootAbiMemberSpec @private {")
    spec_start = body.index("struct RootAbiSpec @private {", member_start)
    spec_end = extract_braced_declaration(body, spec_start)
    private_types = body[member_start:spec_end]

    constants_start = body.index("const RootAbiSpec ", spec_end)
    private_constants = body[constants_start:]
    public = (body[:member_start] + body[spec_end:constants_start]).strip("\n") + "\n"
    private = (
        header
        + "module gpu::internal @private;\n\n"
        + "import gpu;\n"
        + private_types
        + "\n"
        + private_constants
    )
    return public, private


def sync_file(path, expected, check):
    path = Path(path)
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing == expected:
        return True
    if check:
        print(f"stale: '{path}' does not match the schema; regenerate with scripts/gen_abi.py", file=sys.stderr)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")
    return True


def sync_marked_block(path, start_marker, end_marker, content, check):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError(f"expected one generated ABI marker pair in '{path}'")
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    expected_block = f"{start_marker}\n{content.rstrip()}\n{end_marker}"
    expected = source[:start] + expected_block + source[end:]
    if expected == source:
        return True
    if check:
        print(f"stale: generated ABI block in '{path}' does not match the schema; regenerate with scripts/gen_abi.py", file=sys.stderr)
        return False
    path.write_text(expected, encoding="utf-8")
    return True


def gen_gpu(gen_bin, check):
    with tempfile.TemporaryDirectory(prefix="gpu-c3-abi-") as directory:
        temporary = Path(directory)
        c3_out = temporary / "shader_abi.c3"
        glsl_out = temporary / "shader_abi.glsl"
        gen(
            gen_bin,
            False,
            "gpu",
            c3_out,
            glsl_out,
            sorted((ROOT / "abi").glob("*.abi")),
        )
        public, private = split_gpu_c3(c3_out.read_text(encoding="utf-8"))
        clean = sync_marked_block(
            ROOT / "gpu" / "gpu.c3i",
            PUBLIC_ABI_START,
            PUBLIC_ABI_END,
            public,
            check,
        )
        clean &= sync_file(
            ROOT / "gpu" / "internal" / "shader_abi.c3",
            private,
            check,
        )
        clean &= sync_file(
            ROOT / "include" / "shaders" / "generated" / "shader_abi.glsl",
            glsl_out.read_text(encoding="utf-8"),
            check,
        )
        if not clean:
            raise subprocess.CalledProcessError(1, "gen_shader_abi")


def main():
    check = "--check" in sys.argv[1:]
    gen_bin = build_generator()

    gen_gpu(gen_bin, check)

    for name in ["root_pointer", "bindless", "offscreen", "depth", "indirect", "generated_work"]:
        gen(gen_bin, check, "gpu_test",
            ROOT / "test" / "src" / f"{name}_abi.c3",
            ROOT / "test" / "shaders" / "generated" / f"{name}_abi.glsl",
            [ROOT / "test" / "abi" / f"{name}.abi"])

    gen(gen_bin, check, "gpu_bench",
        ROOT / "test" / "src" / "overlap_abi.c3",
        ROOT / "test" / "shaders" / "generated" / "overlap_abi.glsl",
        [ROOT / "test" / "abi" / "overlap.abi"])


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
