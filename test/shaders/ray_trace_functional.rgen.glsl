#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#include "ray_tracing.glsl"

layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer TraceRoot {
    uint acceleration_structure_index;
    uint _pad0;
    uint64_t output_gpu;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer TraceOutput {
    uint values[];
};

layout(location = 0) rayPayloadEXT uint payload;
layout(location = 1) callableDataEXT uint callable_value;

void main() {
    TraceRoot root = TraceRoot(pc.root_gpu);
    uint ray = gl_LaunchIDEXT.x;
    float x = ray == 0 ? 0.5 : ray == 1 ? 2.25 : 4.0;
    uint hit_record = ray == 0 ? 1 : 0;
    uint miss_record = ray == 3 ? 1 : 0;

    callable_value = 0;
    executeCallableEXT(0, 1);
    payload = callable_value;
    traceRayEXT(
        GPU_ACCELERATION_STRUCTURE(root.acceleration_structure_index),
        gl_RayFlagsOpaqueEXT,
        0xff,
        hit_record,
        0,
        miss_record,
        vec3(x, 0.25, -1.0),
        0.0,
        vec3(0.0, 0.0, 1.0),
        100.0,
        0);
    TraceOutput(root.output_gpu).values[ray] = payload;
}
