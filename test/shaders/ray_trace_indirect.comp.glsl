#version 460
#include "generated/shader_abi.glsl"

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer TraceArgs {
    TraceRaysIndirectCommand command;
};

layout(buffer_reference, std430) writeonly buffer TraceArgs2 {
    TraceRaysIndirectCommand2 command2;
};

layout(buffer_reference, std430) writeonly buffer IndirectBuildRange {
    AccelerationStructureIndirectBuildRange range;
};

layout(buffer_reference, std430) readonly buffer TraceArgsRoot {
    uint64_t arguments_gpu;
    uint64_t arguments2_gpu;
    uint64_t ray_generation_record_address;
    uint64_t ray_generation_record_size;
    uint64_t miss_table_address;
    uint64_t miss_table_size;
    uint64_t miss_table_stride;
    uint64_t hit_table_address;
    uint64_t hit_table_size;
    uint64_t hit_table_stride;
    uint64_t callable_table_address;
    uint64_t callable_table_size;
    uint64_t callable_table_stride;
    uint64_t acceleration_structure_range_gpu;
};

layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    TraceArgsRoot root = TraceArgsRoot(pc.root_gpu);
    TraceArgs(root.arguments_gpu).command =
        TraceRaysIndirectCommand(5u, 1u, 1u);
    TraceArgs2(root.arguments2_gpu).command2 = TraceRaysIndirectCommand2(
        root.ray_generation_record_address,
        root.ray_generation_record_size,
        root.miss_table_address,
        root.miss_table_size,
        root.miss_table_stride,
        root.hit_table_address,
        root.hit_table_size,
        root.hit_table_stride,
        root.callable_table_address,
        root.callable_table_size,
        root.callable_table_stride,
        4u,
        1u,
        1u,
        0u
    );
    IndirectBuildRange(root.acceleration_structure_range_gpu).range =
        AccelerationStructureIndirectBuildRange(1u, 64u, 0u, 0u);
}
