#version 460
#include "generated/shader_abi.glsl"
#include "generated/generated_work_abi.glsl"

layout(local_size_x = 1) in;

layout(buffer_reference, std430) writeonly buffer DrawRecords { GeneratedDrawRecord records[]; };
layout(buffer_reference, std430) writeonly buffer IndexedRecords { GeneratedDrawIndexedRecord records[]; };
layout(buffer_reference, std430) writeonly buffer DispatchRecords { GeneratedDispatchRecord records[]; };
layout(buffer_reference, std430) writeonly buffer CountBuffer { uint value; };
layout(push_constant) uniform Push {
    uint64_t root_gpu;
} pc;

void main() {
    GeneratedWorkBuildRoot root = GeneratedWorkBuildRoot(pc.root_gpu);
    DrawRecords(root.draw_records_gpu).records[0] = GeneratedDrawRecord(
        root.vertex_root_gpu,
        root.fragment_root_gpu,
        DrawIndirectCommand(root.vertex_count, 1u, 0u, 0u));
    IndexedRecords(root.indexed_records_gpu).records[0] = GeneratedDrawIndexedRecord(
        root.vertex_root_gpu,
        root.fragment_root_gpu,
        DrawIndexedIndirectCommand(root.index_count, 1u, 0u, 0, 0u),
        0u);
    DispatchRecords(root.dispatch_records_gpu).records[0] = GeneratedDispatchRecord(
        root.dispatch_root_gpu,
        DispatchIndirectCommand(root.dispatch_x, 1u, 1u),
        0u);
    CountBuffer(root.count_gpu).value = root.count_value;
}
