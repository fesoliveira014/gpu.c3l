module gpu::internal @private;

import std::hash::fnv64a;

const uint SPIRV_MAGIC @private           = 0x07230203;
const usz SPIRV_HEADER_SIZE @private      = 20;
const usz SHADER_ENTRY_POINT_CAP @private = 64;

enum ShaderRole : int {
    COMPUTE,
    VERTEX,
    FRAGMENT,
}

fn ulong shader_identity_hash(char[] spirv) @private {
    ulong digest = fnv64a::hash(spirv);
    return digest == 0 ? 1 : digest;
}
