#!/usr/bin/env sh
# Regenerate all shader ABI outputs from their .abi schemas. Pass --check to
# verify the committed outputs are current instead of rewriting them (exits
# nonzero on drift). Schemas shared by a sample and its test generate one GLSL
# file and one C3 file per consuming module.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECK="${1:-}"

c3c build gen_shader_abi --path "$ROOT/tools/gen_shader_abi" >/dev/null
GEN="$ROOT/tools/gen_shader_abi/build/gen_shader_abi"

# gen <module> <c3-out> <glsl-out> <schemas...>
gen() {
    module="$1"; c3_out="$2"; glsl_out="$3"
    shift 3
    mkdir -p "$(dirname "$glsl_out")"
    # shellcheck disable=SC2086
    "$GEN" --module "$module" --c3-out "$c3_out" --glsl-out "$glsl_out" $CHECK "$@"
}

gen gpu "$ROOT/shader_abi.c3" "$ROOT/include/shaders/generated/shader_abi.glsl" \
    "$ROOT"/abi/*.abi

RPC="$ROOT/samples/root_pointer_compute"
gen root_pointer_compute "$RPC/shader_abi.c3" "$RPC/shaders/generated/root_pointer_abi.glsl" \
    "$RPC/abi/root_pointer.abi"
gen gpu_test "$ROOT/test/src/root_pointer_abi.c3" "$RPC/shaders/generated/root_pointer_abi.glsl" \
    "$RPC/abi/root_pointer.abi"

GDD="$ROOT/samples/gpu_driven_draw_sdl"
gen gpu_driven_draw_sdl "$GDD/shader_abi.c3" "$GDD/shaders/generated/gpu_driven_abi.glsl" \
    "$GDD/abi/gpu_driven.abi"

BTC="$ROOT/samples/bindless_texture_compute"
gen bindless_texture_compute "$BTC/shader_abi.c3" "$BTC/shaders/generated/bindless_abi.glsl" \
    "$BTC/abi/bindless.abi"
gen gpu_test "$ROOT/test/src/bindless_abi.c3" "$BTC/shaders/generated/bindless_abi.glsl" \
    "$BTC/abi/bindless.abi"

OT="$ROOT/samples/offscreen_triangle"
gen offscreen_triangle "$OT/shader_abi.c3" "$OT/shaders/generated/offscreen_abi.glsl" \
    "$OT/abi/offscreen.abi"
gen hello_triangle_sdl "$ROOT/samples/hello_triangle_sdl/shader_abi.c3" "$OT/shaders/generated/offscreen_abi.glsl" \
    "$OT/abi/offscreen.abi"
gen gpu_test "$ROOT/test/src/offscreen_abi.c3" "$OT/shaders/generated/offscreen_abi.glsl" \
    "$OT/abi/offscreen.abi"

gen gpu_test "$ROOT/test/src/depth_abi.c3" "$ROOT/test/shaders/generated/depth_abi.glsl" \
    "$ROOT/test/abi/depth.abi"
gen gpu_test "$ROOT/test/src/indirect_abi.c3" "$ROOT/test/shaders/generated/indirect_abi.glsl" \
    "$ROOT/test/abi/indirect.abi"
