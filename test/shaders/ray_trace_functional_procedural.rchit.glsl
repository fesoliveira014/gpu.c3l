#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#include "ray_tracing.glsl"

layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

layout(location = 0) rayPayloadInEXT uint payload;
hitAttributeEXT vec2 attributes;

void main() {
    payload = (payload & 0x10u) | 2u;
}
