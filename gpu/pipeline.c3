module gpu;

import std::hash::fnv64a;

const uint SPIRV_MAGIC @private           = 0x07230203;
const usz SPIRV_HEADER_SIZE @private      = 20;
const usz SHADER_ENTRY_POINT_CAP @private = 64;



fn bool shader_stage_valid(ShaderStage stage) @private {
    switch (stage) {
        case ShaderStage.COMPUTE:
        case ShaderStage.VERTEX:
        case ShaderStage.FRAGMENT:
            return true;
        default:
            return false;
    }
}

fn ulong shader_code_digest(char[] spirv) @private {
    ulong digest = fnv64a::hash(spirv);
    return digest == 0 ? 1 : digest;
}
