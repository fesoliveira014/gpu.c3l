#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 1) in;

layout(push_constant) uniform Push {
    u64vec2 root_gpu;
} pc;

void main() {
    if (pc.root_gpu.x == 1) memoryBarrier();
}
