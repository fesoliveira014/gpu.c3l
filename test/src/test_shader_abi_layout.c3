module gpu_test;

import gpu;

// Hand-computed std430 layouts for the generated ABI structs. The generated
// gpu/internal/shader_abi.c3 already carries compile-time asserts; these
// runtime checks pin the same values as an executable acceptance record.

fn void shader_visible_scalar_widths() @test {
    assert(GpuAddress::size == ulong::size);
    assert(GpuAddress::size == 8);
    assert(TextureIndex::size == uint::size);
    assert(SamplerIndex::size == uint::size);
    assert(AccelerationStructureIndex::size == uint::size);
}

fn void acceleration_structure_instance_layout() @test {
    assert(AccelerationStructureInstance::size == 64);
    assert($reflect(AccelerationStructureInstance.transform_row_0).offset == 0);
    assert($reflect(AccelerationStructureInstance.transform_row_1).offset == 16);
    assert($reflect(AccelerationStructureInstance.transform_row_2).offset == 32);
    assert($reflect(AccelerationStructureInstance.custom_index_and_mask).offset == 48);
    assert($reflect(AccelerationStructureInstance.record_offset_and_flags).offset == 52);
    assert($reflect(AccelerationStructureInstance.acceleration_structure).offset == 56);
}

fn void root_push_layout() @test {
    assert(RootPush::size == 8);
    assert($reflect(RootPush.root_gpu).offset == 0);
}

fn void ray_tracing_shader_binding_table_layout() @test {
    assert($reflect(RayTracingShaderBindingTableRegion.span).offset == 0);
    assert($reflect(RayTracingShaderBindingTableRegion.stride).offset
        == GpuSpan::size);
    assert($reflect(RayTracingShaderBindingTable.ray_generation).offset == 0);
    assert($reflect(RayTracingShaderBindingTable.miss).offset
        == RayTracingShaderBindingTableRegion::size);
    assert($reflect(RayTracingShaderBindingTable.hit).offset
        == 2 * RayTracingShaderBindingTableRegion::size);
    assert($reflect(RayTracingShaderBindingTable.callable).offset
        == 3 * RayTracingShaderBindingTableRegion::size);
}

fn void graphics_root_push_layout() @test {
    assert(GraphicsRootPush::size == 16);
    assert($reflect(GraphicsRootPush.vertex_root_gpu).offset == 0);
    assert($reflect(GraphicsRootPush.fragment_root_gpu).offset == 8);
}

fn void indirect_command_layouts() @test {
    assert(DrawIndirectCommand::size == 16);
    assert($reflect(DrawIndirectCommand.first_vertex).offset == 8);
    assert(DrawIndexedIndirectCommand::size == 20);
    assert($reflect(DrawIndexedIndirectCommand.vertex_offset).offset == 12);
    assert(DispatchIndirectCommand::size == 12);
    assert($reflect(DispatchIndirectCommand.z).offset == 8);
}

fn void generated_work_record_layouts() @test {
    assert(GeneratedDrawRecord::size == 32);
    assert($reflect(GeneratedDrawRecord.vertex_root_gpu).offset == 0);
    assert($reflect(GeneratedDrawRecord.fragment_root_gpu).offset == 8);
    assert($reflect(GeneratedDrawRecord.arguments).offset == 16);

    assert(GeneratedDrawIndexedRecord::size == 40);
    assert($reflect(GeneratedDrawIndexedRecord.vertex_root_gpu).offset == 0);
    assert($reflect(GeneratedDrawIndexedRecord.fragment_root_gpu).offset == 8);
    assert($reflect(GeneratedDrawIndexedRecord.arguments).offset == 16);
    assert($reflect(GeneratedDrawIndexedRecord._pad0).offset == 36);

    assert(GeneratedDispatchRecord::size == 24);
    assert($reflect(GeneratedDispatchRecord.root_gpu).offset == 0);
    assert($reflect(GeneratedDispatchRecord.arguments).offset == 8);
    assert($reflect(GeneratedDispatchRecord._pad0).offset == 20);
}

fn void vector_alias_layouts() @test {
    assert(Vec2f::size == 8);
    assert(Vec4f::size == 16);
    assert(Vec4u::size == 16);
    Vec4f a = { 1, 2, 3, 4 };
    Vec4f b = a * 2.0f + (Vec4f){ 1, 1, 1, 1 };
    assert(b[3] == 9.0f);
}

fn void push_structs_are_std430_tight() @test {
    // A GraphicsRootPush written to memory must be two packed device
    // addresses: byte-identical to what the shader-side struct reads.
    GraphicsRootPush push = {
        .vertex_root_gpu = (GpuAddress)0x1111_2222_3333_4444,
        .fragment_root_gpu = (GpuAddress)0x5555_6666_7777_8888,
    };
    ulong* raw = (ulong*)&push;
    assert(raw[0] == 0x1111_2222_3333_4444);
    assert(raw[1] == 0x5555_6666_7777_8888);
}
