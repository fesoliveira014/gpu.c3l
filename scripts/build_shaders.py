#!/usr/bin/env python3
"""Compile test GLSL shaders to SPIR-V into test/src/shaders/.

.spv is gitignored; .glsl under test/shaders/ is the source of truth. Run
after editing a shader or regenerating ABI includes (scripts/gen_abi.py).
Set GLSLC to point at a specific glslc binary.

Portable replacement for the old build_shaders.sh — runs anywhere python3
and glslc do, including native Windows (no bash required).
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = {".comp": "compute", ".vert": "vertex", ".frag": "fragment"}


def main():
    glslc = os.environ.get("GLSLC", "glslc")
    include_dir = ROOT / "include" / "shaders"
    out_dir = ROOT / "test" / "src" / "shaders"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted((ROOT / "test" / "shaders").glob("*.glsl")):
        stage = STAGES.get(Path(src.stem).suffix)
        if stage is None:
            print(f"build_shaders: unknown shader stage for {src}", file=sys.stderr)
            sys.exit(1)
        out = out_dir / (src.stem + ".spv")
        subprocess.run(
            [glslc, f"-fshader-stage={stage}", "--target-env=vulkan1.3",
             "-I", str(include_dir), str(src), "-o", str(out)],
            check=True,
        )
        print(f"built {out}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
