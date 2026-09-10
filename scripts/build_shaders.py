#!/usr/bin/env python3
"""Regenerate shader ABI outputs and compile every repository shader.

Builds tools/gpu_shaders, regenerates the library ABI (gpu/gpu.c3i marked
block, gpu/internal/shader_abi.c3, include/shaders/generated/shader_abi.glsl)
and the test ABI outputs, compiles test and example GLSL, and assembles the
SPIR-V assembly fixtures. Pass --check to verify committed ABI outputs instead
of rewriting them (exits nonzero on drift); shaders compile in both modes.

Set C3C, GLSLC, or SPIRV_AS to point at specific tool binaries.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL_DIR = ROOT / "tools" / "gpu_shaders"
PUBLIC_ABI_START = "// BEGIN GENERATED SHADER ABI - do not edit."
PUBLIC_ABI_END = "// END GENERATED SHADER ABI"


def tool_binary():
    exe = TOOL_DIR / "build" / "gpu_shaders.exe"
    return exe if exe.exists() else TOOL_DIR / "build" / "gpu_shaders"


def build_tool():
    c3c = shutil.which(os.environ.get("C3C", "c3c"))
    if c3c is None:
        sys.exit("build_shaders: c3c not found (set C3C or add it to PATH)")
    subprocess.run(
        [c3c, "build", "gpu_shaders", "--path", str(TOOL_DIR)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return tool_binary()


def run_tool(tool, *args, check=False):
    command = [str(tool), *map(str, args)]
    if check:
        command.append("--check")
    subprocess.run(command, check=True, cwd=ROOT)


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
        print(f"stale: '{path}' does not match the schema; regenerate with scripts/build_shaders.py", file=sys.stderr)
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
        print(f"stale: generated ABI block in '{path}' does not match the schema; regenerate with scripts/build_shaders.py", file=sys.stderr)
        return False
    path.write_text(expected, encoding="utf-8")
    return True


def gen_core(tool, check):
    """Library ABI: file mode into a temp dir, then split into the shipped files."""
    with tempfile.TemporaryDirectory(prefix="gpu-c3-abi-") as directory:
        temporary = Path(directory)
        c3_out = temporary / "shader_abi.c3"
        glsl_out = temporary / "shader_abi.glsl"
        run_tool(
            tool,
            "--module", "gpu",
            "--c3-out", c3_out,
            "--glsl-out", glsl_out,
            *sorted((ROOT / "abi").glob("*.abi")),
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
            raise subprocess.CalledProcessError(1, "gpu_shaders")


def gen_tests(tool, check):
    run_tool(
        tool,
        "--abi-dir", "test/abi",
        "--module", "gpu_test",
        "--c3-out", "test/src",
        "--glsl-out", "test/shaders/generated",
        "--shader-dir", "test/shaders",
        "--spv-out", "test/src/shaders",
        check=check,
    )
    run_tool(
        tool,
        "--abi-dir", "test/abi/bench",
        "--module", "gpu_bench",
        "--c3-out", "test/src",
        "--glsl-out", "test/shaders/generated",
        check=check,
    )


def build_example(tool):
    run_tool(tool, "--shader-dir", "examples/getting_started/shaders")


def assemble_fixtures():
    sources = sorted((ROOT / "test" / "shaders").glob("*.spvasm"))
    if not sources:
        return
    spirv_as = shutil.which(os.environ.get("SPIRV_AS", "spirv-as"))
    if spirv_as is None:
        sys.exit("build_shaders: spirv-as not found (set SPIRV_AS or add it to PATH)")
    out_dir = ROOT / "test" / "src" / "shaders"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        out = out_dir / (src.stem + ".spv")
        subprocess.run(
            [spirv_as, "--target-env", "vulkan1.3", str(src), "-o", str(out)],
            check=True,
        )
        print(f"assembled {out}")


def main():
    check = "--check" in sys.argv[1:]
    tool = build_tool()
    gen_core(tool, check)
    gen_tests(tool, check)
    build_example(tool)
    assemble_fixtures()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
