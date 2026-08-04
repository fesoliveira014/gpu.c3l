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
}
RAY_TRACING_STAGES = {
    ".rgen": "rgen",
    ".rmiss": "rmiss",
    ".rchit": "rchit",
    ".rahit": "rahit",
    ".rint": "rint",
    ".rcall": "rcall",
}
REQUIRED_RAY_TRACING_FIXTURES = (
    "ray_stage.rgen",
    "ray_stage.rmiss",
    "ray_stage.rchit",
    "ray_stage.rahit",
    "ray_stage.rint",
    "ray_stage.rcall",
    "ray_trace_functional.rgen",
    "ray_trace_functional_primary.rmiss",
    "ray_trace_functional_secondary.rmiss",
    "ray_trace_functional_triangle.rchit",
    "ray_trace_functional_procedural.rint",
    "ray_trace_functional_procedural.rchit",
    "ray_trace_functional.rcall",
)
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
    glslang = shutil.which(
        os.environ.get("GLSLANG_VALIDATOR", "glslangValidator")
    )
    include_dir = ROOT / "include" / "shaders"
    for source_dir, out_dir in SHADER_TREES:
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(source_dir.glob("*.glsl")):
            stage = STAGES.get(Path(src.stem).suffix)
            ray_stage = RAY_TRACING_STAGES.get(Path(src.stem).suffix)
            if stage is None and ray_stage is None:
                print(
                    f"build_shaders: unknown shader stage for {src}",
                    file=sys.stderr,
                )
                sys.exit(1)
            out = out_dir / (src.stem + ".spv")
            if ray_stage is not None:
                if glslang is None:
                    sys.exit(
                        "build_shaders: glslangValidator not found "
                        "(set GLSLANG_VALIDATOR or add it to PATH)"
                    )
                command = [
                    glslang,
                    "-V",
                    "--target-env",
                    "vulkan1.3",
                    f"-I{include_dir}",
                    "-S",
                    ray_stage,
                    str(src),
                    "-o",
                    str(out),
                ]
            else:
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

    fixture_output = ROOT / "test" / "src" / "shaders"
    missing_fixtures = [
        name for name in REQUIRED_RAY_TRACING_FIXTURES
        if not (fixture_output / f"{name}.spv").is_file()
    ]
    if missing_fixtures:
        sys.exit(
            "build_shaders: missing required ray-tracing outputs: "
            + ", ".join(missing_fixtures)
        )

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
