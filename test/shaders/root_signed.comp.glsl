#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = 1) in;

layout(buffer_reference, scalar) writeonly buffer OutputValue {
    uint value;
};
layout(push_constant) uniform Push {
    int64_t root_gpu;
} pc;

void main() {
    if (pc.root_gpu == 0l) return;
    OutputValue(uint64_t(pc.root_gpu)).value = 0xc3a5f17eu;
}
