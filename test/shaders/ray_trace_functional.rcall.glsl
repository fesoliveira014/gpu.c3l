#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#include "ray_tracing.glsl"

layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

layout(location = 1) callableDataInEXT uint callable_value;

void main() {
    callable_value = 0x10u;
}
