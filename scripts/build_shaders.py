#!/usr/bin/env python3
"""Compile repository GLSL and assemble SPIR-V test fixtures.

.spv is gitignored; .glsl is the source of truth. Run after editing a shader
or regenerating ABI includes (scripts/gen_abi.py). Set GLSLC or SPIRV_AS to
point at specific tool binaries.

Portable replacement for the old build_shaders.sh — runs anywhere python3
and glslc do, including native Windows (no bash required).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = {
    ".comp": "compute",
    ".vert": "vertex",
    ".frag": "fragment",
    ".rgen": "rgen",
    ".rmiss": "rmiss",
    ".rchit": "rchit",
    ".rahit": "rahit",
    ".rint": "rint",
    ".rcall": "rcall",
}
SHADER_TREES = (
    (ROOT / "test" / "shaders", ROOT / "test" / "src" / "shaders"),
    (
        ROOT / "examples" / "getting_started" / "shaders",
        ROOT / "examples" / "getting_started" / "shaders",
    ),
)


def main():
    glslc = shutil.which(os.environ.get("GLSLC", "glslc"))
    if glslc is None:
        sys.exit("build_shaders: glslc not found (set GLSLC or add it to PATH)")
    include_dir = ROOT / "include" / "shaders"
    for source_dir, out_dir in SHADER_TREES:
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(source_dir.glob("*.glsl")):
            stage = STAGES.get(Path(src.stem).suffix)
            if stage is None:
                print(
                    f"build_shaders: unknown shader stage for {src}",
                    file=sys.stderr,
                )
                sys.exit(1)
            out = out_dir / (src.stem + ".spv")
            command = [
                glslc,
                f"-fshader-stage={stage}",
                "--target-env=vulkan1.3",
                "-I",
                str(include_dir),
                str(src),
                "-o",
                str(out),
            ]
            subprocess.run(command, check=True)
            print(f"built {out}")

    assembly_sources = sorted((ROOT / "test" / "shaders").glob("*.spvasm"))
    if not assembly_sources:
        return
    spirv_as = shutil.which(os.environ.get("SPIRV_AS", "spirv-as"))
    if spirv_as is None:
        sys.exit("build_shaders: spirv-as not found (set SPIRV_AS or add it to PATH)")
    out_dir = ROOT / "test" / "src" / "shaders"
    for src in assembly_sources:
        out = out_dir / (src.stem + ".spv")
        subprocess.run(
            [spirv_as, "--target-env", "vulkan1.3", str(src), "-o", str(out)],
            check=True,
        )
        print(f"assembled {out}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
