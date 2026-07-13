module gpu_test;

import gpu::compat;

// Hand-computed std430 layouts for the generated ABI structs. The generated
// gpu/shader_abi.c3 already carries compile-time asserts; these runtime checks
// pin the same values as an executable acceptance record.

fn void root_push_layout() @test {
    assert(RootPush::size == 8);
    assert($reflect(RootPush.root_gpu).offset == 0);
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
