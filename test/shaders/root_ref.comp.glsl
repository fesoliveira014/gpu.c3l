#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = 1) in;
layout(buffer_reference, scalar) readonly buffer RootReference {
    uint64_t input_gpu;
    uint64_t output_gpu;
};
layout(buffer_reference, scalar) readonly buffer InputValue {
    uint value;
};
layout(buffer_reference, scalar) writeonly buffer OutputValue {
    uint value;
};
layout(push_constant) uniform Push {
    RootReference root_gpu;
} pc;

void main() {
    if (uint64_t(pc.root_gpu) == 0ul) return;
    OutputValue(pc.root_gpu.output_gpu).value =
        InputValue(pc.root_gpu.input_gpu).value ^ 0xa5a55a5au;
}
