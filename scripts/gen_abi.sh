#!/usr/bin/env sh
# Regenerate all shader ABI outputs from their .abi schemas. Pass --check to
# verify the committed outputs are current instead of rewriting them (exits
# nonzero on drift). Library outputs ship with the package; test outputs are
# generated from test-owned schemas under test/abi/.
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

for name in root_pointer bindless offscreen depth indirect; do
    gen gpu_test "$ROOT/test/src/${name}_abi.c3" "$ROOT/test/shaders/generated/${name}_abi.glsl" \
        "$ROOT/test/abi/$name.abi"
done
