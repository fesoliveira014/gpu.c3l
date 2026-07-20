module gpu_test;

import gpu;

// Headless (lavapipe): offscreen graphics. Clear, textured triangle through
// both root pointers + the heap from the fragment stage, render-to-mip, and
// render-pass state machine enforcement.

const char[*] OFFSCREEN_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] OFFSCREEN_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");
const char[*] DYNAMIC_RECT_VERT_SPIRV = $embed("shaders/depth.vert.spv");
const char[*] DYNAMIC_RECT_FRAG_SPIRV = $embed("shaders/depth.frag.spv");

const uint RT_SIZE = 32;
const usz RT_BYTES = RT_SIZE * RT_SIZE * 4;

fn gpu::Device make_offscreen_test_device() {
    gpu::RuntimeDesc desc = {
        .backend = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_offscreen_test",
    };
    return create_test_device(&desc)!!;
}

fn uint unorm_byte(float value) {
    return (uint)(value * 255.0f + 0.5f);
}

fn gpu::TextureHandle make_render_target(gpu::Device* device, uint mips = 1) {
    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
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
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "offscreen_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.COLOR_ATTACHMENT,
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
        .texture = target,
        .before  = gpu::TextureUse.COLOR_ATTACHMENT,
        .after   = gpu::TextureUse.TRANSFER_SOURCE,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::GpuSpan rb_span = readback.span;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = rb_span };
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
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
    gpu::invalidate_mapped_span(&device, readback.span)!!;

    char* pixels = (char*)test_span_mapping(&device, rb_span).ptr;
    for (uint i = 0; i < RT_SIZE * RT_SIZE; i++) {
        assert(pixels[i * 4] == (char)64, "cleared red");
        assert(pixels[i * 4 + 1] == (char)128, "cleared green");
        assert(pixels[i * 4 + 2] == (char)192, "cleared blue");
        assert(pixels[i * 4 + 3] == (char)255, "cleared alpha");
    }
}

fn void offscreen_textured_triangle() @test {
    gpu::Device device = make_offscreen_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    // Flat-color pattern texture, uploaded through the staging path.
    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle pattern = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, pattern)!!;
    gpu::TextureView tex_index_view = gpu::create_texture_view(&device, pattern, null)!!;
    gpu::TextureIndex tex_index = tex_index_view.index;
    defer gpu::destroy_texture_view(&device, tex_index_view)!!;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::Sampler sampler = gpu::intern_sampler(&device, &sampler_desc)!!;
    gpu::SamplerIndex smp_index = gpu::publish_sampler(&device, sampler)!!;
    OwnedSpan staging = allocate_owned_span(
        device:       &device,
        size:         4 * 4 * 4,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "texture_staging",
    )!!;
    defer free_owned_span(&device, &staging)!!;
    gpu::GpuSpan staging_span = staging.span;
    char* texels = (char*)test_span_mapping(&device, staging_span).ptr;
    for (uint i = 0; i < 16; i++) {
        texels[i * 4]     = (char)100;
        texels[i * 4 + 1] = (char)150;
        texels[i * 4 + 2] = (char)200;
        texels[i * 4 + 3] = (char)255;
    }
    gpu::flush_mapped_span(&device, staging.span)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "offscreen_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = OFFSCREEN_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode vert = gpu::prepare_shader_code(&vert_sd)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = OFFSCREEN_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode frag = gpu::prepare_shader_code(&frag_sd)!!;

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
    // Vertex data: xy = NDC position, zw = uv.
    OwnedSpan verts_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "offscreen_vertices",
    )!!;
    defer free_owned_span(&device, &verts_owned)!!;
    gpu::GpuSpan verts_span = verts_owned.span;
    float* verts = (float*)test_span_mapping(&device, verts_span).ptr;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = vertex_data[i];

    OwnedSpan frag_owned = allocate_owned_span(
        device:       &device,
        size:         FragRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "offscreen_fragment_root",
    )!!;
    defer free_owned_span(&device, &frag_owned)!!;
    gpu::GpuSpan frag_span = frag_owned.span;
    FragRoot* frag_root = (FragRoot*)test_span_mapping(&device, frag_span).ptr;
    frag_root.texture_index = tex_index;
    frag_root.sampler_index = smp_index;
    gpu::flush_mapped_span(&device, verts_span)!!;
    gpu::flush_mapped_span(&device, frag_span)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureBarrier pattern_to_dst = {
        .texture = pattern,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.TRANSFER_DESTINATION,
    };
    gpu::cmd_texture_barrier(&cmd, &pattern_to_dst)!!;
    gpu::BufferTextureCopyDesc upload = { .src = staging_span, .texture = pattern };
    gpu::cmd_copy_buffer_to_texture(&cmd, &upload)!!;
    gpu::TextureBarrier pattern_to_read = {
        .texture = pattern,
        .before  = gpu::TextureUse.TRANSFER_DESTINATION,
        .after   = gpu::TextureUse.SAMPLED_FRAGMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &pattern_to_read)!!;

    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.COLOR_ATTACHMENT,
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
    bind_test_graphics_pipeline(&cmd, pipeline);
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, verts_span),
        fragment_root:  test_span_address(&device, frag_span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::TextureBarrier to_src = {
        .texture = target,
        .before  = gpu::TextureUse.COLOR_ATTACHMENT,
        .after   = gpu::TextureUse.TRANSFER_SOURCE,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::GpuSpan rb_span = readback.span;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = rb_span };
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
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
    gpu::invalidate_mapped_span(&device, readback.span)!!;

    char* pixels = (char*)test_span_mapping(&device, rb_span).ptr;
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
}

fn void dynamic_viewport_scissor_pixels() @test {
    gpu::Device device = make_offscreen_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc vert_desc = {
        .stage = gpu::ShaderStage.VERTEX, .spirv = DYNAMIC_RECT_VERT_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderCode vert = gpu::prepare_shader_code(&vert_desc)!!;
    gpu::ShaderDesc frag_desc = {
        .stage = gpu::ShaderStage.FRAGMENT, .spirv = DYNAMIC_RECT_FRAG_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderCode frag = gpu::prepare_shader_code(&frag_desc)!!;
    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = formats[..],
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::TextureHandle[3] targets;
    for (uint i = 0; i < targets.len; i++) targets[i] = make_render_target(&device);
    defer for (uint i = 0; i < targets.len; i++) gpu::destroy_texture(&device, targets[i])!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         RT_BYTES * targets.len,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "dynamic_state_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;
    gpu::GpuSpan readback_all = readback.span;
    OwnedSpan vertices_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "dynamic_vertices",
    )!!;
    defer free_owned_span(&device, &vertices_owned)!!;
    gpu::GpuSpan vertices = vertices_owned.span;
    float[12] vertex_data = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         3.0f, -1.0f, 0.0f, 0.0f,
        -1.0f,  3.0f, 0.0f, 0.0f,
    };
    for (uint i = 0; i < vertex_data.len; i++) ((float*)test_span_mapping(&device, vertices).ptr)[i] = vertex_data[i];

    OwnedSpan color_owned = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "dynamic_color_root",
    )!!;
    defer free_owned_span(&device, &color_owned)!!;
    gpu::GpuSpan color = color_owned.span;
    float[4] green = { 0.0f, 1.0f, 0.0f, 1.0f };
    for (uint i = 0; i < green.len; i++) ((float*)test_span_mapping(&device, color).ptr)[i] = green[i];
    gpu::flush_mapped_span(&device, vertices)!!;
    gpu::flush_mapped_span(&device, color)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    for (uint i = 0; i < targets.len; i++) {
        gpu::TextureBarrier to_attach = {
            .texture = targets[i],
            .before  = gpu::TextureUse.UNDEFINED,
            .after   = gpu::TextureUse.COLOR_ATTACHMENT,
        };
        gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

        gpu::ColorTargetDesc[1] colors = {{
            .texture = targets[i],
            .load_op = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
        }};
        gpu::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
        gpu::cmd_begin_render_pass(&cmd, &pass)!!;
        if (i == 0) {
            gpu::Viewport viewport = {
                .width = (float)RT_SIZE / 2.0f, .height = (float)RT_SIZE, .max_depth = 1.0f,
            };
            gpu::cmd_set_viewport(&cmd, &viewport)!!;
            gpu::Viewport rejected = viewport;
            rejected.width = (float)RT_SIZE + 1.0f;
            assert(@catch(gpu::cmd_set_viewport(&cmd, &rejected)) == gpu::INVALID_ARGUMENT,
                "a rejected viewport does not replace the valid viewport");
        } else if (i == 1) {
            gpu::ScissorRect scissor = {
                .width = (int)RT_SIZE / 2, .height = (int)RT_SIZE,
            };
            gpu::cmd_set_scissor(&cmd, &scissor)!!;
            gpu::ScissorRect rejected = scissor;
            rejected.x = (int)RT_SIZE;
            assert(@catch(gpu::cmd_set_scissor(&cmd, &rejected)) == gpu::INVALID_ARGUMENT,
                "a rejected scissor does not replace the valid scissor");
        }
        bind_test_graphics_pipeline(&cmd, pipeline);
        gpu::cmd_draw(
            commands:       &cmd,
            vertex_root:    test_span_address(&device, vertices),
            fragment_root:  test_span_address(&device, color),
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::cmd_end_render_pass(&cmd)!!;

        gpu::TextureBarrier to_src = {
            .texture = targets[i],
            .before  = gpu::TextureUse.COLOR_ATTACHMENT,
            .after   = gpu::TextureUse.TRANSFER_SOURCE,
        };
        gpu::cmd_texture_barrier(&cmd, &to_src)!!;

        gpu::GpuSpan region = readback_all.checked_subspan(RT_BYTES * i, RT_BYTES)!!;
        gpu::TextureBufferCopyDesc copy = { .texture = targets[i], .dst = region };
        gpu::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
        gpu::Barrier host_read = {
            .before = { .transfer = true },
            .after = { .host = true },
        };
        gpu::cmd_barrier(&cmd, &host_read)!!;
    }

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
    gpu::invalidate_mapped_span(&device, readback.span)!!;

    for (uint i = 0; i < 2; i++) {
        char* pixels = (char*)test_span_mapping(&device, readback_all).ptr + RT_BYTES * i;
        usz left_green = ((RT_SIZE / 2) * RT_SIZE + RT_SIZE / 4) * 4 + 1;
        usz right_green = ((RT_SIZE / 2) * RT_SIZE + RT_SIZE * 3 / 4) * 4 + 1;
        assert(pixels[left_green] != 0, "the clipped draw covers the left half");
        assert(pixels[right_green] == 0, "the clipped draw leaves the right half clear");
    }
    char* reset_pixels = (char*)test_span_mapping(&device, readback_all).ptr + RT_BYTES * 2;
    usz reset_right_green = ((RT_SIZE / 2) * RT_SIZE + RT_SIZE * 3 / 4) * 4 + 1;
    assert(reset_pixels[reset_right_green] != 0, "the next pass restores full viewport and scissor defaults");
}

fn void offscreen_render_to_mip() @test {
    gpu::Device device = make_offscreen_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device, mips: 2);
    defer gpu::destroy_texture(&device, target)!!;

    const uint MIP1 = RT_SIZE / 2;
    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         MIP1 * MIP1 * 4,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "mip_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.COLOR_ATTACHMENT,
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
        .texture = target,
        .before  = gpu::TextureUse.COLOR_ATTACHMENT,
        .after   = gpu::TextureUse.TRANSFER_SOURCE,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!!;

    gpu::GpuSpan rb_span = readback.span;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = rb_span, .mip = 1 };
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
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
    gpu::invalidate_mapped_span(&device, readback.span)!!;

    char* pixels = (char*)test_span_mapping(&device, rb_span).ptr;
    for (uint i = 0; i < MIP1 * MIP1; i++) {
        assert(pixels[i * 4] == (char)40 && pixels[i * 4 + 1] == (char)80 && pixels[i * 4 + 2] == (char)120,
            "mip 1 holds the rendered clear color");
    }
}

fn void graphics_pipeline_depth_format_validation() @test {
    gpu::Device device = make_offscreen_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = OFFSCREEN_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode vert = gpu::prepare_shader_code(&vert_sd)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = OFFSCREEN_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode frag = gpu::prepare_shader_code(&frag_sd)!!;

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

fn void dynamic_viewport_scissor_validation() @test {
    gpu::Device device = make_offscreen_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::Viewport full_viewport = {
        .width     = (float)RT_SIZE,
        .height    = (float)RT_SIZE,
        .min_depth = 0.0f,
        .max_depth = 1.0f,
    };
    gpu::ScissorRect full_scissor = {
        .width  = (int)RT_SIZE,
        .height = (int)RT_SIZE,
    };
    assert(@catch(gpu::cmd_set_viewport(&cmd, &full_viewport)) == gpu::COMMAND_RECORDING_ERROR,
        "viewport outside a render pass faults COMMAND_RECORDING_ERROR");
    assert(@catch(gpu::cmd_set_scissor(&cmd, &full_scissor)) == gpu::COMMAND_RECORDING_ERROR,
        "scissor outside a render pass faults COMMAND_RECORDING_ERROR");

    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    assert(@catch(gpu::cmd_set_viewport(&cmd, null)) == gpu::INVALID_ARGUMENT,
        "null viewport faults INVALID_ARGUMENT");
    assert(@catch(gpu::cmd_set_scissor(&cmd, null)) == gpu::INVALID_ARGUMENT,
        "null scissor faults INVALID_ARGUMENT");

    gpu::Viewport[13] invalid_viewports = {
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
        assert(@catch(gpu::cmd_set_viewport(&cmd, viewport)) == gpu::INVALID_ARGUMENT,
            "invalid viewport faults INVALID_ARGUMENT");
    }

    gpu::Viewport reversed_depth = full_viewport;
    reversed_depth.min_depth = 1.0f;
    reversed_depth.max_depth = 0.0f;
    gpu::cmd_set_viewport(&cmd, &reversed_depth)!!;

    gpu::ScissorRect[8] invalid_scissors = {
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
        assert(@catch(gpu::cmd_set_scissor(&cmd, scissor)) == gpu::INVALID_ARGUMENT,
            "invalid scissor faults INVALID_ARGUMENT");
    }

    gpu::ScissorRect empty_scissor = {};
    gpu::cmd_set_scissor(&cmd, &empty_scissor)!!;

    gpu::cmd_end_render_pass(&cmd)!!;
    assert(@catch(gpu::cmd_set_viewport(&cmd, &full_viewport)) == gpu::COMMAND_RECORDING_ERROR,
        "viewport after a render pass faults COMMAND_RECORDING_ERROR");
    assert(@catch(gpu::cmd_set_scissor(&cmd, &full_scissor)) == gpu::COMMAND_RECORDING_ERROR,
        "scissor after a render pass faults COMMAND_RECORDING_ERROR");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void render_pass_state_machine() @test {
    gpu::Device device = make_offscreen_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    OwnedSpan data = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "render_pass_data",
    )!!;
    defer free_owned_span(&device, &data)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

    // Draw outside a pass.
    fault draw_fault = @catch(gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    (gpu::GpuAddress)0,
        fragment_root:  (gpu::GpuAddress)0,
        vertex_count:   3,
        instance_count: 1,
    ));
    assert(draw_fault == gpu::COMMAND_RECORDING_ERROR, "draw outside a render pass faults COMMAND_RECORDING_ERROR");

    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.COLOR_ATTACHMENT,
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
        dst:      data.span,
        value:    0,
    ));
    assert(fill_fault == gpu::COMMAND_RECORDING_ERROR, "buffer fill inside a render pass faults COMMAND_RECORDING_ERROR");

    fault barrier_fault = @catch(gpu::cmd_texture_barrier(&cmd, &to_attach));
    assert(barrier_fault == gpu::COMMAND_RECORDING_ERROR, "texture barrier inside a render pass faults COMMAND_RECORDING_ERROR");

    // Unclosed pass rejected at end_commands.
    fault end_fault = @catch(gpu::end_commands(&cmd));
    assert(end_fault == gpu::COMMAND_RECORDING_ERROR, "ending a command list with an open render pass faults COMMAND_RECORDING_ERROR");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}
