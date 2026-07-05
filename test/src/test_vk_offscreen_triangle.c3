module gpu_test;

import gpu;

// Headless (lavapipe): offscreen graphics. Clear, textured triangle through
// both root pointers + the heap from the fragment stage, render-to-mip, and
// render-pass state machine enforcement.

const char[*] OFFSCREEN_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] OFFSCREEN_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");

const uint RT_SIZE = 32;
const usz RT_BYTES = RT_SIZE * RT_SIZE * 4;

fn gpu::Device make_offscreen_test_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_offscreen_test",
    };
    return gpu::create_device(&desc)!!;
}

fn uint unorm_byte(float value) {
    return (uint)(value * 255.0f + 0.5f);
}

fn gpu::TextureHandle make_render_target(gpu::Device* device, uint mips = 1) {
    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = RT_SIZE,
        .height     = RT_SIZE,
        .mip_levels = mips,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .color_attach, .transfer_src },
        .debug_name = "render_target",
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void offscreen_clear_works() @test {
    gpu::Device device = make_offscreen_test_device();
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::BufferDesc rb_desc = {
        .size = RT_BYTES, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK,
    };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, readback)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

    gpu::ColorTargetDesc[1] colors = {{
        .texture = target,
        .load_op = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 64.0f / 255.0f, 128.0f / 255.0f, 192.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::TextureBarrier to_src = {
        .texture       = target,
        .before_stage  = gpu::Stage.COLOR_ATTACHMENT,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.COLOR_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::GpuSpan rb_span = gpu::get_buffer_span(&device, readback)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = rb_span };
    gpu::cmd_copy_texture_to_buffer(&cmd, &copy)!!;

    gpu::BufferBarrier host_read = {
        .buffer = readback, .offset = 0, .size = RT_BYTES,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!!;

    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::invalidate_buffer(&device, readback, 0, 0)!!;

    char* pixels = (char*)rb_span.cpu;
    for (uint i = 0; i < RT_SIZE * RT_SIZE; i++) {
        assert(pixels[i * 4] == (char)64, "cleared red");
        assert(pixels[i * 4 + 1] == (char)128, "cleared green");
        assert(pixels[i * 4 + 2] == (char)192, "cleared blue");
        assert(pixels[i * 4 + 3] == (char)255, "cleared alpha");
    }
    gpu::end_frame(&device)!!;
}

fn void offscreen_textured_triangle() @test {
    gpu::Device device = make_offscreen_test_device();
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    // Flat-color pattern texture, uploaded through the M11 staging path.
    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle pattern = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, pattern)!!;
    gpu::TextureIndex tex_index = gpu::create_texture_descriptor(&device, pattern, null)!!;
    defer gpu::destroy_texture_descriptor(&device, tex_index)!!;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex smp_index = gpu::create_sampler(&device, &sampler_desc)!!;
    defer gpu::destroy_sampler(&device, smp_index)!!;

    gpu::BufferDesc staging_desc = {
        .size = 4 * 4 * 4, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.STAGING,
    };
    gpu::BufferHandle staging = gpu::create_buffer(&device, &staging_desc)!!;
    defer gpu::destroy_buffer(&device, staging)!!;
    gpu::GpuSpan staging_span = gpu::get_buffer_span(&device, staging)!!;
    char* texels = (char*)staging_span.cpu;
    for (uint i = 0; i < 16; i++) {
        texels[i * 4]     = (char)100;
        texels[i * 4 + 1] = (char)150;
        texels[i * 4 + 2] = (char)200;
        texels[i * 4 + 3] = (char)255;
    }
    gpu::flush_buffer(&device, staging, 0, 0)!!;

    gpu::BufferDesc rb_desc = {
        .size = RT_BYTES, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK,
    };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, readback)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = OFFSCREEN_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle vert = gpu::create_shader(&device, &vert_sd)!!;
    defer gpu::destroy_shader(&device, vert)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = OFFSCREEN_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle frag = gpu::create_shader(&device, &frag_sd)!!;
    defer gpu::destroy_shader(&device, frag)!!;

    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = color_formats[..],
        .debug_name      = "offscreen_triangle",
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(&device, &pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::begin_frame(&device)!!;

    // Vertex data: xy = NDC position, zw = uv.
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!!;
    float* verts = (float*)verts_span.cpu;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = vertex_data[i];

    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, FragRoot::size, 16)!!;
    FragRoot* frag_root = (FragRoot*)frag_span.cpu;
    frag_root.texture_index = tex_index;
    frag_root.sampler_index = smp_index;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::TextureBarrier pattern_to_dst = {
        .texture       = pattern,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.TRANSFER_DST,
    };
    gpu::cmd_texture_barrier(&cmd, &pattern_to_dst)!!;
    gpu::BufferTextureCopyDesc upload = { .src = staging_span, .texture = pattern };
    gpu::cmd_copy_buffer_to_texture(&cmd, &upload)!!;
    gpu::TextureBarrier pattern_to_read = {
        .texture       = pattern,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = gpu::Stage.FRAGMENT_SHADER,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
        .old_layout    = gpu::TextureLayout.TRANSFER_DST,
        .new_layout    = gpu::TextureLayout.SHADER_READ,
    };
    gpu::cmd_texture_barrier(&cmd, &pattern_to_read)!!;

    gpu::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

    gpu::ColorTargetDesc[1] colors = {{
        .texture = target,
        .load_op = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        pipeline:       pipeline,
        vertex_root:    verts_span.gpu,
        fragment_root:  frag_span.gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::TextureBarrier to_src = {
        .texture       = target,
        .before_stage  = gpu::Stage.COLOR_ATTACHMENT,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.COLOR_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::GpuSpan rb_span = gpu::get_buffer_span(&device, readback)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = rb_span };
    gpu::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
    gpu::BufferBarrier host_read = {
        .buffer = readback, .offset = 0, .size = RT_BYTES,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!!;

    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::invalidate_buffer(&device, readback, 0, 0)!!;

    char* pixels = (char*)rb_span.cpu;
    // Triangle interior: the center of the target sits well inside.
    char* center = &pixels[(RT_SIZE / 2 * RT_SIZE + RT_SIZE / 2) * 4];
    assert(center[0] == (char)100, "triangle interior red is the sampled texture");
    assert(center[1] == (char)150, "triangle interior green is the sampled texture");
    assert(center[2] == (char)200, "triangle interior blue is the sampled texture");
    // Corners: well outside the triangle, must be the clear color.
    char* corner = &pixels[0];
    assert(corner[0] == (char)8 && corner[1] == (char)16 && corner[2] == (char)32, "corner is the clear color");
    char* far_corner = &pixels[(RT_SIZE * RT_SIZE - 1) * 4];
    assert(far_corner[0] == (char)8, "opposite corner is the clear color");

    gpu::end_frame(&device)!!;
}

fn void offscreen_render_to_mip() @test {
    gpu::Device device = make_offscreen_test_device();
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device, mips: 2);
    defer gpu::destroy_texture(&device, target)!!;

    const uint MIP1 = RT_SIZE / 2;
    gpu::BufferDesc rb_desc = {
        .size = MIP1 * MIP1 * 4, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK,
    };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, readback)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

    gpu::ColorTargetDesc[1] colors = {{
        .texture = target,
        .mip_level = 1,
        .load_op = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 40.0f / 255.0f, 80.0f / 255.0f, 120.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = MIP1, .height = MIP1 };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::TextureBarrier to_src = {
        .texture       = target,
        .before_stage  = gpu::Stage.COLOR_ATTACHMENT,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.COLOR_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::GpuSpan rb_span = gpu::get_buffer_span(&device, readback)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = rb_span, .mip = 1 };
    gpu::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
    gpu::BufferBarrier host_read = {
        .buffer = readback, .offset = 0, .size = MIP1 * MIP1 * 4,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!!;

    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::invalidate_buffer(&device, readback, 0, 0)!!;

    char* pixels = (char*)rb_span.cpu;
    for (uint i = 0; i < MIP1 * MIP1; i++) {
        assert(pixels[i * 4] == (char)40 && pixels[i * 4 + 1] == (char)80 && pixels[i * 4 + 2] == (char)120,
            "mip 1 holds the rendered clear color");
    }
    gpu::end_frame(&device)!!;
}

fn void graphics_pipeline_depth_format_validation() @test {
    gpu::Device device = make_offscreen_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = OFFSCREEN_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle vert = gpu::create_shader(&device, &vert_sd)!!;
    defer gpu::destroy_shader(&device, vert)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = OFFSCREEN_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle frag = gpu::create_shader(&device, &frag_sd)!!;
    defer gpu::destroy_shader(&device, frag)!!;

    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc with_depth = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .color_formats   = color_formats[..],
        .depth_format    = gpu::Format.D32_FLOAT,
    };
    gpu::PipelineHandle depth_pipeline = gpu::create_graphics_pipeline(&device, &with_depth)!!;
    gpu::destroy_pipeline(&device, depth_pipeline)!!;

    gpu::GraphicsPipelineDesc packed_depth = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .color_formats   = color_formats[..],
        .depth_format    = gpu::Format.D24_UNORM_S8_UINT,
    };
    assert(@catch(gpu::create_graphics_pipeline(&device, &packed_depth)) == gpu::INVALID_ARGUMENT,
        "D24_UNORM_S8_UINT faults INVALID_ARGUMENT");

    gpu::GraphicsPipelineDesc no_attachments = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
    };
    assert(@catch(gpu::create_graphics_pipeline(&device, &no_attachments)) == gpu::INVALID_ARGUMENT,
        "zero color formats with no depth format faults INVALID_ARGUMENT");
}

fn void render_pass_state_machine() @test {
    gpu::Device device = make_offscreen_test_device();
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::BufferDesc buf_desc = {
        .size = 64, .usage = { .transfer_src, .transfer_dst }, .memory_kind = gpu::MemoryKind.STAGING,
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &buf_desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;

    // Draw outside a pass.
    fault draw_fault = @catch(gpu::cmd_draw(
        commands:       &cmd,
        pipeline:       gpu::PIPELINE_HANDLE_INVALID,
        vertex_root:    (gpu::GpuAddress)0,
        fragment_root:  (gpu::GpuAddress)0,
        vertex_count:   3,
        instance_count: 1,
    ));
    assert(draw_fault == gpu::COMMAND_RECORDING_ERROR, "draw outside a render pass faults COMMAND_RECORDING_ERROR");

    gpu::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    // Copies and barriers are illegal inside a pass.
    fault fill_fault = @catch(gpu::cmd_fill_buffer(
        commands: &cmd,
        buffer:   buffer,
        offset:   0,
        size:     64,
        value:    0,
    ));
    assert(fill_fault == gpu::COMMAND_RECORDING_ERROR, "buffer fill inside a render pass faults COMMAND_RECORDING_ERROR");

    fault barrier_fault = @catch(gpu::cmd_texture_barrier(&cmd, &to_attach));
    assert(barrier_fault == gpu::COMMAND_RECORDING_ERROR, "texture barrier inside a render pass faults COMMAND_RECORDING_ERROR");

    // Unclosed pass rejected at end_commands.
    fault end_fault = @catch(gpu::end_commands(&device, &cmd));
    assert(end_fault == gpu::COMMAND_RECORDING_ERROR, "ending a command list with an open render pass faults COMMAND_RECORDING_ERROR");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}
