#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require

layout(local_size_x = 1) in;

layout(push_constant) uniform Push {
    float64_t root_gpu;
} pc;

void main() {
    if (pc.root_gpu == 1.0lf) memoryBarrier();
}
