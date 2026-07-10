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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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


def main():
    check = "--check" in sys.argv[1:]
    gen_bin = build_generator()

    gen(gen_bin, check, "gpu",
        ROOT / "shader_abi.c3",
        ROOT / "include" / "shaders" / "generated" / "shader_abi.glsl",
        sorted((ROOT / "abi").glob("*.abi")))

    for name in ["root_pointer", "bindless", "offscreen", "depth", "indirect"]:
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
