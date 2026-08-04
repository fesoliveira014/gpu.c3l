#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#include "ray_tracing.glsl"

layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

hitAttributeEXT vec2 attributes;

void main() {
    attributes = vec2(0.25, 0.75);
    reportIntersectionEXT(1.5, 0);
}
