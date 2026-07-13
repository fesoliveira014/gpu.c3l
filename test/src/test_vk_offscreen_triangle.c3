module gpu_test;

import gpu::compat;

// Headless (lavapipe): offscreen graphics. Clear, textured triangle through
// both root pointers + the heap from the fragment stage, render-to-mip, and
// render-pass state machine enforcement.

const char[*] OFFSCREEN_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] OFFSCREEN_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");
const char[*] DYNAMIC_RECT_VERT_SPIRV = $embed("shaders/depth.vert.spv");
const char[*] DYNAMIC_RECT_FRAG_SPIRV = $embed("shaders/depth.frag.spv");

const uint RT_SIZE = 32;
const usz RT_BYTES = RT_SIZE * RT_SIZE * 4;

fn gpu::compat::Device make_offscreen_test_device() {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_offscreen_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

fn uint unorm_byte(float value) {
    return (uint)(value * 255.0f + 0.5f);
}

fn gpu::compat::TextureHandle make_render_target(gpu::compat::Device* device, uint mips = 1) {
    gpu::compat::TextureDesc desc = {
        .dimension  = gpu::compat::TextureDimension.TEX_2D,
        .width      = RT_SIZE,
        .height     = RT_SIZE,
        .mip_levels = mips,
        .format     = gpu::compat::Format.RGBA8_UNORM,
        .usage      = { .color_attach, .transfer_src },
        .debug_name = "render_target",
    };
    return gpu::compat::create_texture(device, &desc)!!;
}

fn void offscreen_clear_works() @test {
    gpu::compat::Device device = make_offscreen_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle target = make_render_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::BufferDesc rb_desc = {
        .size = RT_BYTES, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.READBACK,
    };
    gpu::compat::BufferHandle readback = gpu::compat::create_buffer(&device, &rb_desc)!!;
    defer gpu::compat::destroy_buffer(&device, readback)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;

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
        .texture = target,
        .load_op = gpu::compat::LoadOp.CLEAR,
        .store_op = gpu::compat::StoreOp.STORE,
        .clear = { .rgba = { 64.0f / 255.0f, 128.0f / 255.0f, 192.0f / 255.0f, 1.0f } },
    }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
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

    gpu::compat::GpuSpan rb_span = gpu::compat::get_buffer_span(&device, readback)!!;
    gpu::compat::TextureBufferCopyDesc copy = { .texture = target, .dst = rb_span };
    gpu::compat::cmd_copy_texture_to_buffer(&cmd, &copy)!!;

    gpu::compat::BufferBarrier host_read = {
        .buffer = readback, .offset = 0, .size = RT_BYTES,
        .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::invalidate_buffer(&device, readback, 0, 0)!!;

    char* pixels = (char*)rb_span.cpu;
    for (uint i = 0; i < RT_SIZE * RT_SIZE; i++) {
        assert(pixels[i * 4] == (char)64, "cleared red");
        assert(pixels[i * 4 + 1] == (char)128, "cleared green");
        assert(pixels[i * 4 + 2] == (char)192, "cleared blue");
        assert(pixels[i * 4 + 3] == (char)255, "cleared alpha");
    }
    gpu::compat::end_frame(&frame_token)!!;
}

fn void offscreen_textured_triangle() @test {
    gpu::compat::Device device = make_offscreen_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle target = make_render_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    // Flat-color pattern texture, uploaded through the staging path.
    gpu::compat::TextureDesc tex_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::compat::TextureHandle pattern = gpu::compat::create_texture(&device, &tex_desc)!!;
    defer gpu::compat::destroy_texture(&device, pattern)!!;
    gpu::compat::TextureIndex tex_index = gpu::compat::create_texture_descriptor(&device, pattern, null)!!;
    defer gpu::compat::destroy_texture_descriptor(&device, tex_index)!!;
    gpu::compat::SamplerDesc sampler_desc = {
        .min_filter = gpu::compat::Filter.NEAREST,
        .mag_filter = gpu::compat::Filter.NEAREST,
        .address_u  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::compat::SamplerIndex smp_index = gpu::compat::create_sampler(&device, &sampler_desc)!!;
    defer gpu::compat::destroy_sampler(&device, smp_index)!!;

    gpu::compat::BufferDesc staging_desc = {
        .size = 4 * 4 * 4, .usage = { .transfer_src }, .memory_kind = gpu::compat::MemoryKind.STAGING,
    };
    gpu::compat::BufferHandle staging = gpu::compat::create_buffer(&device, &staging_desc)!!;
    defer gpu::compat::destroy_buffer(&device, staging)!!;
    gpu::compat::GpuSpan staging_span = gpu::compat::get_buffer_span(&device, staging)!!;
    char* texels = (char*)staging_span.cpu;
    for (uint i = 0; i < 16; i++) {
        texels[i * 4]     = (char)100;
        texels[i * 4 + 1] = (char)150;
        texels[i * 4 + 2] = (char)200;
        texels[i * 4 + 3] = (char)255;
    }
    gpu::compat::flush_buffer(&device, staging, 0, 0)!!;

    gpu::compat::BufferDesc rb_desc = {
        .size = RT_BYTES, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.READBACK,
    };
    gpu::compat::BufferHandle readback = gpu::compat::create_buffer(&device, &rb_desc)!!;
    defer gpu::compat::destroy_buffer(&device, readback)!!;

    gpu::compat::ShaderDesc vert_sd = { .stage = gpu::compat::ShaderStage.VERTEX, .spirv = OFFSCREEN_VERT_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle vert = gpu::compat::create_shader(&device, &vert_sd)!!;
    defer gpu::compat::destroy_shader(&device, vert)!!;
    gpu::compat::ShaderDesc frag_sd = { .stage = gpu::compat::ShaderStage.FRAGMENT, .spirv = OFFSCREEN_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle frag = gpu::compat::create_shader(&device, &frag_sd)!!;
    defer gpu::compat::destroy_shader(&device, frag)!!;

    gpu::compat::Format[1] color_formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .topology        = gpu::compat::PrimitiveTopology.TRIANGLES,
        .color_formats   = color_formats[..],
        .debug_name      = "offscreen_triangle",
    };
    gpu::compat::PipelineHandle pipeline = gpu::compat::create_graphics_pipeline(&device, &pipe_desc)!!;
    defer gpu::compat::destroy_pipeline(&device, pipeline)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;

    // Vertex data: xy = NDC position, zw = uv.
    gpu::compat::GpuSpan verts_span = gpu::compat::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
    float* verts = (float*)verts_span.cpu;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = vertex_data[i];

    gpu::compat::GpuSpan frag_span = gpu::compat::alloc_frame_span(&frame_token, FragRoot::size, 16)!!;
    FragRoot* frag_root = (FragRoot*)frag_span.cpu;
    frag_root.texture_index = tex_index;
    frag_root.sampler_index = smp_index;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::TextureBarrier pattern_to_dst = {
        .texture       = pattern,
        .before_stage  = gpu::compat::Stage.HOST,
        .after_stage   = gpu::compat::Stage.TRANSFER,
        .before_hazard = gpu::compat::Hazard.HOST_WRITE,
        .after_hazard  = gpu::compat::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
        .new_layout    = gpu::compat::TextureLayout.TRANSFER_DST,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &pattern_to_dst)!!;
    gpu::compat::BufferTextureCopyDesc upload = { .src = staging_span, .texture = pattern };
    gpu::compat::cmd_copy_buffer_to_texture(&cmd, &upload)!!;
    gpu::compat::TextureBarrier pattern_to_read = {
        .texture       = pattern,
        .before_stage  = gpu::compat::Stage.TRANSFER,
        .after_stage   = gpu::compat::Stage.FRAGMENT_SHADER,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::compat::Hazard.SHADER_READ,
        .old_layout    = gpu::compat::TextureLayout.TRANSFER_DST,
        .new_layout    = gpu::compat::TextureLayout.SHADER_READ,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &pattern_to_read)!!;

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
        .texture = target,
        .load_op = gpu::compat::LoadOp.CLEAR,
        .store_op = gpu::compat::StoreOp.STORE,
        .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::compat::cmd_draw(
        commands:       &cmd,
        pipeline:       pipeline,
        vertex_root:    verts_span.gpu,
        fragment_root:  frag_span.gpu,
        vertex_count:   3,
        instance_count: 1,
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

    gpu::compat::GpuSpan rb_span = gpu::compat::get_buffer_span(&device, readback)!!;
    gpu::compat::TextureBufferCopyDesc copy = { .texture = target, .dst = rb_span };
    gpu::compat::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
    gpu::compat::BufferBarrier host_read = {
        .buffer = readback, .offset = 0, .size = RT_BYTES,
        .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::invalidate_buffer(&device, readback, 0, 0)!!;

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

    gpu::compat::end_frame(&frame_token)!!;
}

fn void dynamic_viewport_scissor_pixels() @test {
    gpu::compat::Device device = make_offscreen_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::ShaderDesc vert_desc = {
        .stage = gpu::compat::ShaderStage.VERTEX, .spirv = DYNAMIC_RECT_VERT_SPIRV[..], .entry_point = "main",
    };
    gpu::compat::ShaderHandle vert = gpu::compat::create_shader(&device, &vert_desc)!!;
    defer gpu::compat::destroy_shader(&device, vert)!!;
    gpu::compat::ShaderDesc frag_desc = {
        .stage = gpu::compat::ShaderStage.FRAGMENT, .spirv = DYNAMIC_RECT_FRAG_SPIRV[..], .entry_point = "main",
    };
    gpu::compat::ShaderHandle frag = gpu::compat::create_shader(&device, &frag_desc)!!;
    defer gpu::compat::destroy_shader(&device, frag)!!;
    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .topology        = gpu::compat::PrimitiveTopology.TRIANGLES,
        .color_formats   = formats[..],
    };
    gpu::compat::PipelineHandle pipeline = gpu::compat::create_graphics_pipeline(&device, &pipeline_desc)!!;
    defer gpu::compat::destroy_pipeline(&device, pipeline)!!;

    gpu::compat::TextureHandle[3] targets;
    for (uint i = 0; i < targets.len; i++) targets[i] = make_render_target(&device);
    defer for (uint i = 0; i < targets.len; i++) gpu::compat::destroy_texture(&device, targets[i])!!;

    gpu::compat::BufferDesc readback_desc = {
        .size = RT_BYTES * targets.len,
        .usage = { .transfer_dst },
        .memory_kind = gpu::compat::MemoryKind.READBACK,
    };
    gpu::compat::BufferHandle readback = gpu::compat::create_buffer(&device, &readback_desc)!!;
    defer gpu::compat::destroy_buffer(&device, readback)!!;
    gpu::compat::GpuSpan readback_all = gpu::compat::get_buffer_span(&device, readback)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan vertices = gpu::compat::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
    float[12] vertex_data = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         3.0f, -1.0f, 0.0f, 0.0f,
        -1.0f,  3.0f, 0.0f, 0.0f,
    };
    for (uint i = 0; i < vertex_data.len; i++) ((float*)vertices.cpu)[i] = vertex_data[i];

    gpu::compat::GpuSpan color = gpu::compat::alloc_frame_span(&frame_token, 16, 16)!!;
    float[4] green = { 0.0f, 1.0f, 0.0f, 1.0f };
    for (uint i = 0; i < green.len; i++) ((float*)color.cpu)[i] = green[i];

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    for (uint i = 0; i < targets.len; i++) {
        gpu::compat::TextureBarrier to_attach = {
            .texture       = targets[i],
            .before_stage  = gpu::compat::Stage.HOST,
            .after_stage   = gpu::compat::Stage.COLOR_ATTACHMENT,
            .before_hazard = gpu::compat::Hazard.HOST_WRITE,
            .after_hazard  = gpu::compat::Hazard.COLOR_WRITE,
            .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
            .new_layout    = gpu::compat::TextureLayout.COLOR_ATTACHMENT,
        };
        gpu::compat::cmd_texture_barrier(&cmd, &to_attach)!!;

        gpu::compat::ColorTargetDesc[1] colors = {{
            .texture = targets[i],
            .load_op = gpu::compat::LoadOp.CLEAR,
            .store_op = gpu::compat::StoreOp.STORE,
            .clear = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
        }};
        gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
        gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
        if (i == 0) {
            gpu::compat::Viewport viewport = {
                .width = (float)RT_SIZE / 2.0f, .height = (float)RT_SIZE, .max_depth = 1.0f,
            };
            gpu::compat::cmd_set_viewport(&cmd, &viewport)!!;
            gpu::compat::Viewport rejected = viewport;
            rejected.width = (float)RT_SIZE + 1.0f;
            assert(@catch(gpu::compat::cmd_set_viewport(&cmd, &rejected)) == gpu::compat::INVALID_ARGUMENT,
                "a rejected viewport does not replace the valid viewport");
        } else if (i == 1) {
            gpu::compat::ScissorRect scissor = {
                .width = (int)RT_SIZE / 2, .height = (int)RT_SIZE,
            };
            gpu::compat::cmd_set_scissor(&cmd, &scissor)!!;
            gpu::compat::ScissorRect rejected = scissor;
            rejected.x = (int)RT_SIZE;
            assert(@catch(gpu::compat::cmd_set_scissor(&cmd, &rejected)) == gpu::compat::INVALID_ARGUMENT,
                "a rejected scissor does not replace the valid scissor");
        }
        gpu::compat::cmd_draw(
            commands:       &cmd,
            pipeline:       pipeline,
            vertex_root:    vertices.gpu,
            fragment_root:  color.gpu,
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::compat::cmd_end_render_pass(&cmd)!!;

        gpu::compat::TextureBarrier to_src = {
            .texture       = targets[i],
            .before_stage  = gpu::compat::Stage.COLOR_ATTACHMENT,
            .after_stage   = gpu::compat::Stage.TRANSFER,
            .before_hazard = gpu::compat::Hazard.COLOR_WRITE,
            .after_hazard  = gpu::compat::Hazard.TRANSFER_READ,
            .old_layout    = gpu::compat::TextureLayout.COLOR_ATTACHMENT,
            .new_layout    = gpu::compat::TextureLayout.TRANSFER_SRC,
        };
        gpu::compat::cmd_texture_barrier(&cmd, &to_src)!!;

        gpu::compat::GpuSpan region = readback_all.checked_subspan(RT_BYTES * i, RT_BYTES)!!;
        gpu::compat::TextureBufferCopyDesc copy = { .texture = targets[i], .dst = region };
        gpu::compat::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
        gpu::compat::BufferBarrier host_read = {
            .buffer = readback, .offset = region.offset, .size = region.size,
            .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
            .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
        };
        gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;
    }

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::invalidate_buffer(&device, readback, 0, 0)!!;

    for (uint i = 0; i < 2; i++) {
        char* pixels = (char*)readback_all.cpu + RT_BYTES * i;
        usz left_green = ((RT_SIZE / 2) * RT_SIZE + RT_SIZE / 4) * 4 + 1;
        usz right_green = ((RT_SIZE / 2) * RT_SIZE + RT_SIZE * 3 / 4) * 4 + 1;
        assert(pixels[left_green] != 0, "the clipped draw covers the left half");
        assert(pixels[right_green] == 0, "the clipped draw leaves the right half clear");
    }
    char* reset_pixels = (char*)readback_all.cpu + RT_BYTES * 2;
    usz reset_right_green = ((RT_SIZE / 2) * RT_SIZE + RT_SIZE * 3 / 4) * 4 + 1;
    assert(reset_pixels[reset_right_green] != 0, "the next pass restores full viewport and scissor defaults");

    gpu::compat::end_frame(&frame_token)!!;
}

fn void offscreen_render_to_mip() @test {
    gpu::compat::Device device = make_offscreen_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle target = make_render_target(&device, mips: 2);
    defer gpu::compat::destroy_texture(&device, target)!!;

    const uint MIP1 = RT_SIZE / 2;
    gpu::compat::BufferDesc rb_desc = {
        .size = MIP1 * MIP1 * 4, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.READBACK,
    };
    gpu::compat::BufferHandle readback = gpu::compat::create_buffer(&device, &rb_desc)!!;
    defer gpu::compat::destroy_buffer(&device, readback)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;

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
        .texture = target,
        .mip_level = 1,
        .load_op = gpu::compat::LoadOp.CLEAR,
        .store_op = gpu::compat::StoreOp.STORE,
        .clear = { .rgba = { 40.0f / 255.0f, 80.0f / 255.0f, 120.0f / 255.0f, 1.0f } },
    }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = MIP1, .height = MIP1 };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
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

    gpu::compat::GpuSpan rb_span = gpu::compat::get_buffer_span(&device, readback)!!;
    gpu::compat::TextureBufferCopyDesc copy = { .texture = target, .dst = rb_span, .mip = 1 };
    gpu::compat::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
    gpu::compat::BufferBarrier host_read = {
        .buffer = readback, .offset = 0, .size = MIP1 * MIP1 * 4,
        .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::invalidate_buffer(&device, readback, 0, 0)!!;

    char* pixels = (char*)rb_span.cpu;
    for (uint i = 0; i < MIP1 * MIP1; i++) {
        assert(pixels[i * 4] == (char)40 && pixels[i * 4 + 1] == (char)80 && pixels[i * 4 + 2] == (char)120,
            "mip 1 holds the rendered clear color");
    }
    gpu::compat::end_frame(&frame_token)!!;
}

fn void graphics_pipeline_depth_format_validation() @test {
    gpu::compat::Device device = make_offscreen_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::ShaderDesc vert_sd = { .stage = gpu::compat::ShaderStage.VERTEX, .spirv = OFFSCREEN_VERT_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle vert = gpu::compat::create_shader(&device, &vert_sd)!!;
    defer gpu::compat::destroy_shader(&device, vert)!!;
    gpu::compat::ShaderDesc frag_sd = { .stage = gpu::compat::ShaderStage.FRAGMENT, .spirv = OFFSCREEN_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle frag = gpu::compat::create_shader(&device, &frag_sd)!!;
    defer gpu::compat::destroy_shader(&device, frag)!!;

    gpu::compat::Format[1] color_formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc with_depth = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .color_formats   = color_formats[..],
        .depth_format    = gpu::compat::Format.D32_FLOAT,
    };
    gpu::compat::PipelineHandle depth_pipeline = gpu::compat::create_graphics_pipeline(&device, &with_depth)!!;
    gpu::compat::destroy_pipeline(&device, depth_pipeline)!!;

    gpu::compat::GraphicsPipelineDesc packed_depth = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .color_formats   = color_formats[..],
        .depth_format    = gpu::compat::Format.D24_UNORM_S8_UINT,
    };
    assert(@catch(gpu::compat::create_graphics_pipeline(&device, &packed_depth)) == gpu::compat::INVALID_ARGUMENT,
        "D24_UNORM_S8_UINT faults INVALID_ARGUMENT");

    gpu::compat::GraphicsPipelineDesc no_attachments = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
    };
    assert(@catch(gpu::compat::create_graphics_pipeline(&device, &no_attachments)) == gpu::compat::INVALID_ARGUMENT,
        "zero color formats with no depth format faults INVALID_ARGUMENT");
}

fn void dynamic_viewport_scissor_validation() @test {
    gpu::compat::Device device = make_offscreen_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle target = make_render_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::Viewport full_viewport = {
        .width     = (float)RT_SIZE,
        .height    = (float)RT_SIZE,
        .min_depth = 0.0f,
        .max_depth = 1.0f,
    };
    gpu::compat::ScissorRect full_scissor = {
        .width  = (int)RT_SIZE,
        .height = (int)RT_SIZE,
    };
    assert(@catch(gpu::compat::cmd_set_viewport(&cmd, &full_viewport)) == gpu::compat::COMMAND_RECORDING_ERROR,
        "viewport outside a render pass faults COMMAND_RECORDING_ERROR");
    assert(@catch(gpu::compat::cmd_set_scissor(&cmd, &full_scissor)) == gpu::compat::COMMAND_RECORDING_ERROR,
        "scissor outside a render pass faults COMMAND_RECORDING_ERROR");

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
    }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;

    assert(@catch(gpu::compat::cmd_set_viewport(&cmd, null)) == gpu::compat::INVALID_ARGUMENT,
        "null viewport faults INVALID_ARGUMENT");
    assert(@catch(gpu::compat::cmd_set_scissor(&cmd, null)) == gpu::compat::INVALID_ARGUMENT,
        "null scissor faults INVALID_ARGUMENT");

    gpu::compat::Viewport[13] invalid_viewports = {
        { .x = float::nan, .width = 1.0f, .height = 1.0f, .max_depth = 1.0f },
        { .y = float::inf, .width = 1.0f, .height = 1.0f, .max_depth = 1.0f },
        { .width = float::nan, .height = 1.0f, .max_depth = 1.0f },
        { .width = 1.0f, .height = -float::inf, .max_depth = 1.0f },
        { .width = 1.0f, .height = 1.0f, .min_depth = float::nan, .max_depth = 1.0f },
        { .width = 1.0f, .height = 1.0f, .max_depth = float::inf },
        { .x = -1.0f, .width = 1.0f, .height = 1.0f, .max_depth = 1.0f },
        { .width = 0.0f, .height = 1.0f, .max_depth = 1.0f },
        { .width = 1.0f, .height = -1.0f, .max_depth = 1.0f },
        { .x = (float)RT_SIZE - 1.0f, .width = 2.0f, .height = 1.0f, .max_depth = 1.0f },
        { .y = (float)RT_SIZE - 1.0f, .width = 1.0f, .height = 2.0f, .max_depth = 1.0f },
        { .width = 1.0f, .height = 1.0f, .min_depth = -0.1f, .max_depth = 1.0f },
        { .width = 1.0f, .height = 1.0f, .max_depth = 1.1f },
    };
    foreach (&viewport : invalid_viewports) {
        assert(@catch(gpu::compat::cmd_set_viewport(&cmd, viewport)) == gpu::compat::INVALID_ARGUMENT,
            "invalid viewport faults INVALID_ARGUMENT");
    }

    gpu::compat::Viewport reversed_depth = full_viewport;
    reversed_depth.min_depth = 1.0f;
    reversed_depth.max_depth = 0.0f;
    gpu::compat::cmd_set_viewport(&cmd, &reversed_depth)!!;

    gpu::compat::ScissorRect[8] invalid_scissors = {
        { .x = -1, .width = 1, .height = 1 },
        { .y = -1, .width = 1, .height = 1 },
        { .width = -1, .height = 1 },
        { .width = 1, .height = -1 },
        { .x = (int)RT_SIZE - 1, .width = 2, .height = 1 },
        { .y = (int)RT_SIZE - 1, .width = 1, .height = 2 },
        { .x = int::max, .width = 1, .height = 1 },
        { .y = int::max, .width = 1, .height = 1 },
    };
    foreach (&scissor : invalid_scissors) {
        assert(@catch(gpu::compat::cmd_set_scissor(&cmd, scissor)) == gpu::compat::INVALID_ARGUMENT,
            "invalid scissor faults INVALID_ARGUMENT");
    }

    gpu::compat::ScissorRect empty_scissor = {};
    gpu::compat::cmd_set_scissor(&cmd, &empty_scissor)!!;

    gpu::compat::cmd_end_render_pass(&cmd)!!;
    assert(@catch(gpu::compat::cmd_set_viewport(&cmd, &full_viewport)) == gpu::compat::COMMAND_RECORDING_ERROR,
        "viewport after a render pass faults COMMAND_RECORDING_ERROR");
    assert(@catch(gpu::compat::cmd_set_scissor(&cmd, &full_scissor)) == gpu::compat::COMMAND_RECORDING_ERROR,
        "scissor after a render pass faults COMMAND_RECORDING_ERROR");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void render_pass_state_machine() @test {
    gpu::compat::Device device = make_offscreen_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle target = make_render_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::BufferDesc buf_desc = {
        .size = 64, .usage = { .transfer_src, .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.STAGING,
    };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &buf_desc)!!;
    defer gpu::compat::destroy_buffer(&device, buffer)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    // Draw outside a pass.
    fault draw_fault = @catch(gpu::compat::cmd_draw(
        commands:       &cmd,
        pipeline:       gpu::compat::PIPELINE_HANDLE_INVALID,
        vertex_root:    (gpu::compat::GpuAddress)0,
        fragment_root:  (gpu::compat::GpuAddress)0,
        vertex_count:   3,
        instance_count: 1,
    ));
    assert(draw_fault == gpu::compat::COMMAND_RECORDING_ERROR, "draw outside a render pass faults COMMAND_RECORDING_ERROR");

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
    }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;

    // Copies and barriers are illegal inside a pass.
    fault fill_fault = @catch(gpu::compat::cmd_fill_buffer(
        commands: &cmd,
        buffer:   buffer,
        offset:   0,
        size:     64,
        value:    0,
    ));
    assert(fill_fault == gpu::compat::COMMAND_RECORDING_ERROR, "buffer fill inside a render pass faults COMMAND_RECORDING_ERROR");

    fault barrier_fault = @catch(gpu::compat::cmd_texture_barrier(&cmd, &to_attach));
    assert(barrier_fault == gpu::compat::COMMAND_RECORDING_ERROR, "texture barrier inside a render pass faults COMMAND_RECORDING_ERROR");

    // Unclosed pass rejected at end_commands.
    fault end_fault = @catch(gpu::compat::end_commands(&cmd));
    assert(end_fault == gpu::compat::COMMAND_RECORDING_ERROR, "ending a command list with an open render pass faults COMMAND_RECORDING_ERROR");

    gpu::compat::cmd_end_render_pass(&cmd)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}
