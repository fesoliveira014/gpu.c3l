module gpu_test;

import gpu;
import std::core::mem;

// Headless (lavapipe): U16 and U32 indexed draws with equivalent indices must
// produce identical pixels, at zero and nonzero index-span offsets.

const char[*] U16V_VERT_SPIRV = $embed("shaders/depth.vert.spv");
const char[*] U16V_FRAG_SPIRV = $embed("shaders/depth.frag.spv");

const uint U16V_RT = 16;
const usz U16V_BYTES = U16V_RT * U16V_RT * 4;
const uint U16V_INDEX_COUNT = 6;
const uint U16V_OFFSET_ELEMENTS = 6;

fn gpu::Device u16v_device() {
    gpu::RuntimeDesc desc = {
        .backend = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_u16_verify",
    };
    return create_test_device(&desc)!!;
}

fn OwnedSpan u16v_upload(gpu::Device* device, usz size, ZString debug_name) {
    return allocate_owned_span(
        device:       device,
        size:         size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   debug_name,
    )!!;
}

fn void u16v_draw(
    gpu::Device* device,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress verts,
    gpu::GpuAddress color,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
    gpu::GpuSpan* rb_span,
) {
    gpu::TextureDesc rt_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = U16V_RT,
        .height    = U16V_RT,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    gpu::TextureHandle target = gpu::create_texture(device, &rt_desc)!!;
    defer gpu::destroy_texture(device, target)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = U16V_RT, .height = U16V_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    bind_test_graphics_pipeline(&cmd, pipeline);
    gpu::cmd_draw_indexed(
        commands:       &cmd,
        vertex_root:    verts,
        fragment_root:  color,
        index_span:     index_span,
        index_count:    U16V_INDEX_COUNT,
        instance_count: 1,
        index_type:     index_type,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::TextureBarrier to_src = {
        .texture = target,
        .before  = gpu::TextureUse.COLOR_ATTACHMENT,
        .after   = gpu::TextureUse.TRANSFER_SOURCE,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = *rb_span };
    gpu::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&cmd, &host_read)!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(device, gpu::QueueKind.GRAPHICS)!!;
    gpu::invalidate_mapped_span(device, *rb_span)!!;
}

fn void u16_and_u32_indexed_draws_match() @test {
    gpu::Device device = u16v_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = U16V_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = U16V_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode vert = gpu::prepare_shader_code(&vert_sd)!!;
    gpu::ShaderCode frag = gpu::prepare_shader_code(&frag_sd)!!;
    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc pd = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = formats[..],
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(&device, &pd)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    OwnedSpan vbuf = u16v_upload(&device, 4 * 16, "vertices");
    defer free_owned_span(&device, &vbuf)!!;
    gpu::GpuSpan vspan = vbuf.span;
    float[16] verts = {
        -0.8f, -0.8f, 0.5f, 0.0f,
         0.8f, -0.8f, 0.5f, 0.0f,
        -0.8f,  0.8f, 0.5f, 0.0f,
         0.8f,  0.8f, 0.5f, 0.0f,
    };
    mem::copy(test_span_mapping(&device, vspan).ptr, &verts, 16 * 4);

    OwnedSpan cbuf = u16v_upload(&device, 16, "color");
    defer free_owned_span(&device, &cbuf)!!;
    gpu::GpuSpan cspan = cbuf.span;
    float[4] color = { 0.0f, 1.0f, 0.0f, 1.0f };
    mem::copy(test_span_mapping(&device, cspan).ptr, &color, 16);

    ushort[6] idx16 = { 0, 1, 2, 2, 1, 3 };
    uint[6] idx32 = { 0, 1, 2, 2, 1, 3 };

    OwnedSpan ib16 = u16v_upload(&device, U16V_INDEX_COUNT * 2, "indices_u16");
    defer free_owned_span(&device, &ib16)!!;
    gpu::GpuSpan span16 = ib16.span;
    mem::copy(test_span_mapping(&device, span16).ptr, &idx16, U16V_INDEX_COUNT * 2);

    OwnedSpan ib32 = u16v_upload(&device, U16V_INDEX_COUNT * 4, "indices_u32");
    defer free_owned_span(&device, &ib32)!!;
    gpu::GpuSpan span32 = ib32.span;
    mem::copy(test_span_mapping(&device, span32).ptr, &idx32, U16V_INDEX_COUNT * 4);

    usz u16_off_bytes = (U16V_OFFSET_ELEMENTS + U16V_INDEX_COUNT) * 2;
    OwnedSpan ib16_off = u16v_upload(&device, u16_off_bytes, "indices_u16_offset");
    defer free_owned_span(&device, &ib16_off)!!;
    gpu::GpuSpan span16_off = ib16_off.span;
    ushort* p16_off = (ushort*)test_span_mapping(&device, span16_off).ptr;
    for (uint i = 0; i < U16V_OFFSET_ELEMENTS; i++) p16_off[i] = 0;
    for (uint i = 0; i < U16V_INDEX_COUNT; i++) p16_off[U16V_OFFSET_ELEMENTS + i] = idx16[i];

    usz u32_off_bytes = (U16V_OFFSET_ELEMENTS + U16V_INDEX_COUNT) * 4;
    OwnedSpan ib32_off = u16v_upload(&device, u32_off_bytes, "indices_u32_offset");
    defer free_owned_span(&device, &ib32_off)!!;
    gpu::GpuSpan span32_off = ib32_off.span;
    uint* p32_off = (uint*)test_span_mapping(&device, span32_off).ptr;
    for (uint i = 0; i < U16V_OFFSET_ELEMENTS; i++) p32_off[i] = 0;
    for (uint i = 0; i < U16V_INDEX_COUNT; i++) p32_off[U16V_OFFSET_ELEMENTS + i] = idx32[i];

    gpu::flush_mapped_span(&device, vbuf.span)!!;
    gpu::flush_mapped_span(&device, cbuf.span)!!;
    gpu::flush_mapped_span(&device, ib16.span)!!;
    gpu::flush_mapped_span(&device, ib32.span)!!;
    gpu::flush_mapped_span(&device, ib16_off.span)!!;
    gpu::flush_mapped_span(&device, ib32_off.span)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         U16V_BYTES * 4,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "indexed_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;
    gpu::GpuSpan rb_all = readback.span;
    gpu::GpuSpan rb_u16 = rb_all.checked_subspan(0, U16V_BYTES)!!;
    gpu::GpuSpan rb_u32 = rb_all.checked_subspan(U16V_BYTES, U16V_BYTES)!!;
    gpu::GpuSpan rb_u16_off = rb_all.checked_subspan(U16V_BYTES * 2, U16V_BYTES)!!;
    gpu::GpuSpan rb_u32_off = rb_all.checked_subspan(U16V_BYTES * 3, U16V_BYTES)!!;
    u16v_draw(
        device:     &device,
        pipeline:   pipeline,
        verts:      test_span_address(&device, vspan),
        color:      test_span_address(&device, cspan),
        index_span: span16.checked_subspan(0, U16V_INDEX_COUNT * 2)!!,
        index_type: gpu::IndexType.U16,
        rb_span:    &rb_u16,
    );
    u16v_draw(
        device:     &device,
        pipeline:   pipeline,
        verts:      test_span_address(&device, vspan),
        color:      test_span_address(&device, cspan),
        index_span: span32.checked_subspan(0, U16V_INDEX_COUNT * 4)!!,
        index_type: gpu::IndexType.U32,
        rb_span:    &rb_u32,
    );
    u16v_draw(
        device:     &device,
        pipeline:   pipeline,
        verts:      test_span_address(&device, vspan),
        color:      test_span_address(&device, cspan),
        index_span: span16_off.checked_subspan(U16V_OFFSET_ELEMENTS * 2, U16V_INDEX_COUNT * 2)!!,
        index_type: gpu::IndexType.U16,
        rb_span:    &rb_u16_off,
    );
    u16v_draw(
        device:     &device,
        pipeline:   pipeline,
        verts:      test_span_address(&device, vspan),
        color:      test_span_address(&device, cspan),
        index_span: span32_off.checked_subspan(U16V_OFFSET_ELEMENTS * 4, U16V_INDEX_COUNT * 4)!!,
        index_type: gpu::IndexType.U32,
        rb_span:    &rb_u32_off,
    );
    char* a = (char*)test_span_mapping(&device, rb_u16).ptr;
    char* b = (char*)test_span_mapping(&device, rb_u32).ptr;
    char* c = (char*)test_span_mapping(&device, rb_u16_off).ptr;
    char* d = (char*)test_span_mapping(&device, rb_u32_off).ptr;
    uint diff_zero = 0;
    uint diff_off = 0;
    uint diff_u16_offsets = 0;
    uint diff_u32_offsets = 0;
    for (usz i = 0; i < U16V_BYTES; i++) {
        if (a[i] != b[i]) diff_zero++;
        if (c[i] != d[i]) diff_off++;
        if (a[i] != c[i]) diff_u16_offsets++;
        if (b[i] != d[i]) diff_u32_offsets++;
    }
    usz center_green = ((U16V_RT / 2) * U16V_RT + U16V_RT / 2) * 4 + 1;
    usz corner_green = 1;
    assert(b[center_green] != 0, "the zero-offset U32 draw must cover the center pixel");
    assert(b[corner_green] == 0, "the zero-offset U32 draw must leave the corner clear");
    assert(d[center_green] != 0, "the nonzero-offset U32 draw must cover the center pixel");
    assert(d[corner_green] == 0, "the nonzero-offset U32 draw must leave the corner clear");
    assert(diff_zero == 0, "U16 and U32 draws at offset 0 must match pixel-for-pixel");
    assert(diff_off == 0, "U16 and U32 draws at a nonzero index offset must match pixel-for-pixel");
    assert(diff_u16_offsets == 0, "U16 draws at zero and nonzero offsets must match pixel-for-pixel");
    assert(diff_u32_offsets == 0, "U32 draws at zero and nonzero offsets must match pixel-for-pixel");
}
