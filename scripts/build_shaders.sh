#!/usr/bin/env sh
# Compile test GLSL shaders to SPIR-V into test/src/shaders/. .spv is
# gitignored; .glsl under test/shaders/ is the source of truth. Run after
# editing a shader or regenerating ABI includes (scripts/gen_abi.sh).
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GLSLC="${GLSLC:-glslc}"
INC="$ROOT/include/shaders"
OUT="$ROOT/test/src/shaders"

mkdir -p "$OUT"

for src in "$ROOT"/test/shaders/*.glsl; do
    base="$(basename "$src" .glsl)"
    # Stage is the middle extension (foo.comp.glsl -> compute); glslc cannot
    # infer it from a .glsl suffix.
    case "$src" in
        *.comp.glsl) stage=compute ;;
        *.vert.glsl) stage=vertex ;;
        *.frag.glsl) stage=fragment ;;
        *) echo "build_shaders: unknown shader stage for $src" >&2; exit 1 ;;
    esac
    "$GLSLC" -fshader-stage="$stage" --target-env=vulkan1.3 -I "$INC" "$src" -o "$OUT/$base.spv"
    echo "built $OUT/$base.spv"
done
