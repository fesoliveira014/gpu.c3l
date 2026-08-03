#version 460
#extension GL_GOOGLE_include_directive : require
#include "ray_tracing.glsl"

layout(push_constant) uniform Push {
    uint root_gpu;
} pc;

void main() {
    if (pc.root_gpu == 0u) return;
}
