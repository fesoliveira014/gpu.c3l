module gpu_test;

import gpu::compat;
import std::core::mem;

// Headless (lavapipe): U16 and U32 indexed draws with equivalent indices must
// produce identical pixels, at zero and nonzero index-span offsets.

const char[*] U16V_VERT_SPIRV = $embed("shaders/depth.vert.spv");
const char[*] U16V_FRAG_SPIRV = $embed("shaders/depth.frag.spv");

const uint U16V_RT = 16;
const usz U16V_BYTES = U16V_RT * U16V_RT * 4;
const uint U16V_INDEX_COUNT = 6;
const uint U16V_OFFSET_ELEMENTS = 6;

fn gpu::compat::Device u16v_device() {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_u16_verify",
    };
    return gpu::compat::create_device(&desc)!!;
}

fn gpu::compat::BufferHandle u16v_persistent(gpu::compat::Device* device, usz size, gpu::compat::BufferUsage usage) {
    gpu::compat::BufferDesc desc = { .size = size, .usage = usage, .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD };
    return gpu::compat::create_buffer(device, &desc)!!;
}

fn void u16v_draw(gpu::compat::Device* device, gpu::compat::PipelineHandle pipeline, gpu::compat::GpuAddress verts, gpu::compat::GpuAddress color, gpu::compat::GpuSpan index_span, gpu::compat::IndexType index_type, gpu::compat::GpuSpan* rb_span) {
    gpu::compat::TextureDesc rt_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = U16V_RT,
        .height    = U16V_RT,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    gpu::compat::TextureHandle target = gpu::compat::create_texture(device, &rt_desc)!!;
    defer gpu::compat::destroy_texture(device, target)!!;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::compat::Stage.HOST,
        .after_stage   = gpu::compat::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::compat::Hazard.HOST_WRITE,
        .after_hazard  = gpu::compat::Hazard.COLOR_WRITE,
        .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
        .new_layout    = gpu::compat::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::compat::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
        .clear = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
    }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = U16V_RT, .height = U16V_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::compat::cmd_draw_indexed(
        commands:       &cmd,
        pipeline:       pipeline,
        vertex_root:    verts,
        fragment_root:  color,
        index_span:     index_span,
        index_count:    U16V_INDEX_COUNT,
        instance_count: 1,
        index_type:     index_type,
    )!!;
    gpu::compat::cmd_end_render_pass(&cmd)!!;

    gpu::compat::TextureBarrier to_src = {
        .texture       = target,
        .before_stage  = gpu::compat::Stage.COLOR_ATTACHMENT,
        .after_stage   = gpu::compat::Stage.TRANSFER,
        .before_hazard = gpu::compat::Hazard.COLOR_WRITE,
        .after_hazard  = gpu::compat::Hazard.TRANSFER_READ,
        .old_layout    = gpu::compat::TextureLayout.COLOR_ATTACHMENT,
        .new_layout    = gpu::compat::TextureLayout.TRANSFER_SRC,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_src)!!;
    gpu::compat::TextureBufferCopyDesc copy = { .texture = target, .dst = *rb_span };
    gpu::compat::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
    gpu::compat::BufferBarrier host_read = {
        .buffer = rb_span.buffer, .offset = rb_span.offset, .size = rb_span.size,
        .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(device, &submit)!!;
    gpu::compat::wait_queue_idle(device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::invalidate_buffer(device, rb_span.buffer, rb_span.offset, rb_span.size)!!;
}

fn void u16_and_u32_indexed_draws_match() @test {
    gpu::compat::Device device = u16v_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::ShaderDesc vert_sd = { .stage = gpu::compat::ShaderStage.VERTEX, .spirv = U16V_VERT_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderDesc frag_sd = { .stage = gpu::compat::ShaderStage.FRAGMENT, .spirv = U16V_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle vert = gpu::compat::create_shader(&device, &vert_sd)!!;
    defer gpu::compat::destroy_shader(&device, vert)!!;
    gpu::compat::ShaderHandle frag = gpu::compat::create_shader(&device, &frag_sd)!!;
    defer gpu::compat::destroy_shader(&device, frag)!!;
    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc pd = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .topology        = gpu::compat::PrimitiveTopology.TRIANGLES,
        .color_formats   = formats[..],
    };
    gpu::compat::PipelineHandle pipeline = gpu::compat::create_graphics_pipeline(&device, &pd)!!;
    defer gpu::compat::destroy_pipeline(&device, pipeline)!!;

    gpu::compat::BufferHandle vbuf = u16v_persistent(&device, 4 * 16, { .storage, .addressable });
    defer gpu::compat::destroy_buffer(&device, vbuf)!!;
    gpu::compat::GpuSpan vspan = gpu::compat::get_buffer_span(&device, vbuf)!!;
    float[16] verts = {
        -0.8f, -0.8f, 0.5f, 0.0f,
         0.8f, -0.8f, 0.5f, 0.0f,
        -0.8f,  0.8f, 0.5f, 0.0f,
         0.8f,  0.8f, 0.5f, 0.0f,
    };
    mem::copy(vspan.cpu, &verts, 16 * 4);

    gpu::compat::BufferHandle cbuf = u16v_persistent(&device, 16, { .storage, .addressable });
    defer gpu::compat::destroy_buffer(&device, cbuf)!!;
    gpu::compat::GpuSpan cspan = gpu::compat::get_buffer_span(&device, cbuf)!!;
    float[4] color = { 0.0f, 1.0f, 0.0f, 1.0f };
    mem::copy(cspan.cpu, &color, 16);

    ushort[6] idx16 = { 0, 1, 2, 2, 1, 3 };
    uint[6] idx32 = { 0, 1, 2, 2, 1, 3 };

    gpu::compat::BufferHandle ib16 = u16v_persistent(&device, U16V_INDEX_COUNT * 2, { .index });
    defer gpu::compat::destroy_buffer(&device, ib16)!!;
    gpu::compat::GpuSpan span16 = gpu::compat::get_buffer_span(&device, ib16)!!;
    mem::copy(span16.cpu, &idx16, U16V_INDEX_COUNT * 2);

    gpu::compat::BufferHandle ib32 = u16v_persistent(&device, U16V_INDEX_COUNT * 4, { .index });
    defer gpu::compat::destroy_buffer(&device, ib32)!!;
    gpu::compat::GpuSpan span32 = gpu::compat::get_buffer_span(&device, ib32)!!;
    mem::copy(span32.cpu, &idx32, U16V_INDEX_COUNT * 4);

    usz u16_off_bytes = (U16V_OFFSET_ELEMENTS + U16V_INDEX_COUNT) * 2;
    gpu::compat::BufferHandle ib16_off = u16v_persistent(&device, u16_off_bytes, { .index });
    defer gpu::compat::destroy_buffer(&device, ib16_off)!!;
    gpu::compat::GpuSpan span16_off = gpu::compat::get_buffer_span(&device, ib16_off)!!;
    ushort* p16_off = (ushort*)span16_off.cpu;
    for (uint i = 0; i < U16V_OFFSET_ELEMENTS; i++) p16_off[i] = 0;
    for (uint i = 0; i < U16V_INDEX_COUNT; i++) p16_off[U16V_OFFSET_ELEMENTS + i] = idx16[i];

    usz u32_off_bytes = (U16V_OFFSET_ELEMENTS + U16V_INDEX_COUNT) * 4;
    gpu::compat::BufferHandle ib32_off = u16v_persistent(&device, u32_off_bytes, { .index });
    defer gpu::compat::destroy_buffer(&device, ib32_off)!!;
    gpu::compat::GpuSpan span32_off = gpu::compat::get_buffer_span(&device, ib32_off)!!;
    uint* p32_off = (uint*)span32_off.cpu;
    for (uint i = 0; i < U16V_OFFSET_ELEMENTS; i++) p32_off[i] = 0;
    for (uint i = 0; i < U16V_INDEX_COUNT; i++) p32_off[U16V_OFFSET_ELEMENTS + i] = idx32[i];

    gpu::compat::flush_buffer(&device, vbuf, 0, 0)!!;
    gpu::compat::flush_buffer(&device, cbuf, 0, 0)!!;
    gpu::compat::flush_buffer(&device, ib16, 0, 0)!!;
    gpu::compat::flush_buffer(&device, ib32, 0, 0)!!;
    gpu::compat::flush_buffer(&device, ib16_off, 0, 0)!!;
    gpu::compat::flush_buffer(&device, ib32_off, 0, 0)!!;

    gpu::compat::BufferHandle rb = gpu::compat::create_buffer(&device, &&(gpu::compat::BufferDesc){ .size = U16V_BYTES * 4, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.READBACK })!!;
    defer gpu::compat::destroy_buffer(&device, rb)!!;
    gpu::compat::GpuSpan rb_all = gpu::compat::get_buffer_span(&device, rb)!!;
    gpu::compat::GpuSpan rb_u16 = rb_all.checked_subspan(0, U16V_BYTES)!!;
    gpu::compat::GpuSpan rb_u32 = rb_all.checked_subspan(U16V_BYTES, U16V_BYTES)!!;
    gpu::compat::GpuSpan rb_u16_off = rb_all.checked_subspan(U16V_BYTES * 2, U16V_BYTES)!!;
    gpu::compat::GpuSpan rb_u32_off = rb_all.checked_subspan(U16V_BYTES * 3, U16V_BYTES)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    u16v_draw(
        device:     &device,
        pipeline:   pipeline,
        verts:      vspan.gpu,
        color:      cspan.gpu,
        index_span: span16.checked_subspan(0, U16V_INDEX_COUNT * 2)!!,
        index_type: gpu::compat::IndexType.U16,
        rb_span:    &rb_u16,
    );
    u16v_draw(
        device:     &device,
        pipeline:   pipeline,
        verts:      vspan.gpu,
        color:      cspan.gpu,
        index_span: span32.checked_subspan(0, U16V_INDEX_COUNT * 4)!!,
        index_type: gpu::compat::IndexType.U32,
        rb_span:    &rb_u32,
    );
    u16v_draw(
        device:     &device,
        pipeline:   pipeline,
        verts:      vspan.gpu,
        color:      cspan.gpu,
        index_span: span16_off.checked_subspan(U16V_OFFSET_ELEMENTS * 2, U16V_INDEX_COUNT * 2)!!,
        index_type: gpu::compat::IndexType.U16,
        rb_span:    &rb_u16_off,
    );
    u16v_draw(
        device:     &device,
        pipeline:   pipeline,
        verts:      vspan.gpu,
        color:      cspan.gpu,
        index_span: span32_off.checked_subspan(U16V_OFFSET_ELEMENTS * 4, U16V_INDEX_COUNT * 4)!!,
        index_type: gpu::compat::IndexType.U32,
        rb_span:    &rb_u32_off,
    );
    gpu::compat::end_frame(&frame_token)!!;

    char* a = (char*)rb_u16.cpu;
    char* b = (char*)rb_u32.cpu;
    char* c = (char*)rb_u16_off.cpu;
    char* d = (char*)rb_u32_off.cpu;
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
