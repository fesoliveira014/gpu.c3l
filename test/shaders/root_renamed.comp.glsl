#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 1) in;

layout(push_constant) uniform Push {
    uint64_t unrelated_member_name;
} unrelated_block_name;

void main() {
    if (unrelated_block_name.unrelated_member_name == 1ul) memoryBarrier();
}
