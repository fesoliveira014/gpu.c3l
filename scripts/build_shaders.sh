#!/usr/bin/env sh
# Compile sample GLSL shaders to SPIR-V and place the .spv next to each source
# and into the test's local shader dir. .spv is gitignored; .glsl is the source
# of truth. Run this before building the sample or the vk_root_pointer test.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GLSLC="${GLSLC:-glslc}"
INC="$ROOT/include/shaders"

mkdir -p "$ROOT/test/src/shaders"

compile() {
    src="$1"
    out="$2"
    # Stage is the middle extension (foo.comp.glsl -> compute); glslc cannot infer
    # it from a .glsl suffix.
    case "$src" in
        *.comp.glsl) stage=compute ;;
        *.vert.glsl) stage=vertex ;;
        *.frag.glsl) stage=fragment ;;
        *) echo "build_shaders: unknown shader stage for $src" >&2; exit 1 ;;
    esac
    "$GLSLC" -fshader-stage="$stage" --target-env=vulkan1.3 -I "$INC" "$src" -o "$out"
    echo "built $out"
}

RP="$ROOT/samples/root_pointer_compute/shaders/root_pointer.comp"
compile "$RP.glsl" "$RP.spv"
cp "$RP.spv" "$ROOT/test/src/shaders/root_pointer.comp.spv"
echo "copied root_pointer.comp.spv -> test/src/shaders/"

BT="$ROOT/samples/bindless_texture_compute/shaders"
for name in heap_write heap_sample; do
    compile "$BT/$name.comp.glsl" "$BT/$name.comp.spv"
    cp "$BT/$name.comp.spv" "$ROOT/test/src/shaders/$name.comp.spv"
    echo "copied $name.comp.spv -> test/src/shaders/"
done

# Reflection-validation fixtures: intentionally convention-violating shaders.
for name in bad_set bad_binding; do
    compile "$ROOT/test/shaders/$name.comp.glsl" "$ROOT/test/src/shaders/$name.comp.spv"
done

OT="$ROOT/samples/offscreen_triangle/shaders"
for name in offscreen.vert offscreen.frag; do
    compile "$OT/$name.glsl" "$OT/$name.spv"
    cp "$OT/$name.spv" "$ROOT/test/src/shaders/$name.spv"
    echo "copied $name.spv -> test/src/shaders/"
done
