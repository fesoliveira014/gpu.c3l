#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#include "ray_tracing.glsl"

layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

layout(location = 0) rayPayloadEXT vec3 payload;

void main() {
    if (pc.root_gpu == 0ul) return;
    traceRayEXT(
        GPU_ACCELERATION_STRUCTURE(1u),
        gl_RayFlagsOpaqueEXT,
        0xff,
        0,
        0,
        0,
        vec3(0.0),
        0.0,
        vec3(0.0, 0.0, 1.0),
        1.0,
        0);
}
