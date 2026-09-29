module gpu::internal @private;

import std::hash::fnv64a;

const uint SPIRV_MAGIC @private                    = 0x07230203;
const usz SPIRV_HEADER_SIZE @private               = 20;
const usz SHADER_ENTRY_POINT_CAP @private          = 64;
const uint SPIRV_OPCODE_MASK @private              = 0xFFFF;
const uint SPIRV_WORD_COUNT_SHIFT @private         = 16;
const uint SPIRV_OP_CAPABILITY @private            = 17;
const uint SPIRV_CAPABILITY_CLIP_DISTANCE @private = 32;

enum ShaderRole : int {
    COMPUTE,
    VERTEX,
    FRAGMENT,
    RAY_GENERATION,
    MISS,
    CLOSEST_HIT,
    ANY_HIT,
    INTERSECTION,
    CALLABLE,
    TASK,
    MESH,
}

fn ulong shader_identity_hash(char[] spirv) @private {
    ulong digest = fnv64a::hash(spirv);
    return digest == 0 ? 1 : digest;
}

<* Whether the module's leading capability instructions name the capability. Pure. *>
fn bool spirv_declares_capability(uint[] words, uint capability) @private {
    usz index = SPIRV_HEADER_SIZE / uint::size;
    while (index + 1 < words.len) {
        uint opcode = words[index] & SPIRV_OPCODE_MASK;
        uint word_count = words[index] >> SPIRV_WORD_COUNT_SHIFT;
        if (opcode != SPIRV_OP_CAPABILITY || word_count < 2) return false;
        if (words[index + 1] == capability) return true;
        index += word_count;
    }
    return false;
}
