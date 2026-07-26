module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk;
import vk;
import std::io;

// Headless (lavapipe): offscreen graphics. Clear, textured triangle through
// both root pointers + the heap from the fragment stage, render-to-mip, and
// render-pass state machine enforcement.

const char[*] OFFSCREEN_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] OFFSCREEN_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");
const char[*] DYNAMIC_RECT_VERT_SPIRV = $embed("shaders/depth.vert.spv");
const char[*] DYNAMIC_RECT_FRAG_SPIRV = $embed("shaders/depth.frag.spv");

const uint RT_SIZE = 32;
const usz RT_BYTES = RT_SIZE * RT_SIZE * 4;
const uint COLOR_READBACK_TOLERANCE = 1;

struct VulkanValidationCapture {
    uint errors;
}

fn void capture_vulkan_validation_error(
    gpu::DebugMessage* message,
    void* user_data,
) {
    if (message.category.backend
        && message.category.validation
        && message.severity == gpu::DebugMessageSeverity.ERROR) {
        VulkanValidationCapture* capture =
            (VulkanValidationCapture*)user_data;
        capture.errors++;
    }
}

fn gpu::Device make_offscreen_test_device(
    VulkanValidationCapture* capture = null,
) {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_offscreen_test",
        .debug_callback           = capture == null
            ? null
            : &capture_vulkan_validation_error,
        .debug_user_data          = capture,
    };
    return create_test_device(&desc)!!;
}

fn uint unorm_byte(float value) {
    return (uint)(value * 255.0f + 0.5f);
}

fn gpu::TextureHandle make_render_target(gpu::Device* device, uint mips = 1) {
    gpu::TextureDesc desc = {
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
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "offscreen_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view,
        .load_op = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 64.0f / 255.0f, 128.0f / 255.0f, 192.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::TextureBarrier to_src = {
        .texture = target,
        .before  = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
        .after   = {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
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

fn void repeated_render_pass_retains_precreated_attachment() @test {
    gpu::Device device = make_offscreen_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(&commands, &to_attach)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .view     = target_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = RT_SIZE,
        .height = RT_SIZE,
    };
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_end_render_pass(&commands)!!;
    assert(@catch(gpu::destroy_attachment_view(&device, target_view))
        == gpu::RESOURCE_IN_USE,
        "a recorded attachment cannot be destroyed before command retirement");
}

fn void offscreen_textured_triangle() @test {
    gpu::Device device = make_offscreen_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    // Flat-color pattern texture, uploaded through the staging path.
    gpu::TextureDesc tex_desc = {
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
    gpu::SamplerIndex smp_index = gpu::intern_sampler(&device, &sampler_desc)!!;
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

    gpu::ShaderDesc vert_sd = { .spirv = OFFSCREEN_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc vert = vert_sd;
    gpu::ShaderDesc frag_sd = { .spirv = OFFSCREEN_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc frag = frag_sd;

    gpu::Format[1] pipeline_color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .color_formats   = pipeline_color_formats[..],
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

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureBarrier pattern_to_dst = {
        .texture = pattern,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    };
    gpu::cmd_texture_barrier(&cmd, &pattern_to_dst)!!;
    gpu::BufferTextureCopyDesc upload = { .src = staging_span, .texture = pattern };
    gpu::cmd_copy_buffer_to_texture(&cmd, &upload)!!;
    gpu::TextureBarrier pattern_to_read = {
        .texture = pattern,
        .before  = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
        .after   = gpu::sampled_at({ .fragment_shader }),
    };
    gpu::cmd_texture_barrier(&cmd, &pattern_to_read)!!;

    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view,
        .load_op = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(RT_SIZE, RT_SIZE)!!;
    gpu::ColorTargetState[1] color_targets = {{
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    graphics_state.color = { .targets = color_targets[..] };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
    gpu::cmd_set_graphics_state(&cmd, &graphics_state)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    (gpu::GpuAddress)0,
        fragment_root:  test_span_address(&device, frag_span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, verts_span),
        fragment_root:  (gpu::GpuAddress)0,
        vertex_count:   3,
        instance_count: 1,
    )!!;
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
        .before  = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
        .after   = {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
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
        .spirv = DYNAMIC_RECT_VERT_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderDesc vert = vert_desc;
    gpu::ShaderDesc frag_desc = {
        .spirv = DYNAMIC_RECT_FRAG_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderDesc frag = frag_desc;
    gpu::Format[1] pipeline_color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .color_formats   = pipeline_color_formats[..],
        .depth_format    = gpu::Format.D32_FLOAT,
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::TextureHandle[5] targets;
    for (uint i = 0; i < targets.len; i++) targets[i] = make_render_target(&device);
    defer for (uint i = 0; i < targets.len; i++) gpu::destroy_texture(&device, targets[i])!!;
    gpu::AttachmentViewHandle[5] target_views;
    for (uint i = 0; i < target_views.len; i++) {
        target_views[i] = create_test_attachment_view(
            &device,
            targets[i],
        )!!;
    }
    defer for (uint i = 0; i < target_views.len; i++) {
        destroy_test_attachment_view(&device, &target_views[i])!!;
    }
    gpu::TextureDesc depth_desc = {
        .access = { .graphics, .transfer },
        .width  = RT_SIZE,
        .height = RT_SIZE,
        .format = gpu::Format.D32_FLOAT,
        .usage  = { .depth_attach },
    };
    gpu::TextureHandle depth = gpu::create_texture(&device, &depth_desc)!!;
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::AttachmentViewHandle depth_view = create_test_attachment_view(
        &device,
        depth,
    )!!;
    defer destroy_test_attachment_view(&device, &depth_view)!!;

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

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    record_depth_to_attach(&cmd, depth);
    for (uint i = 0; i < targets.len; i++) {
        gpu::TextureBarrier to_attach = {
            .texture = targets[i],
            .before  = { .layout = gpu::TextureLayout.UNDEFINED },
            .after   = {
                .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
                .stages = { .color_output },
                .access = { .read, .write },
            },
        };
        gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

        gpu::ColorTargetDesc[1] colors = {{
            .view = target_views[i],
            .load_op = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
        }};
        gpu::DepthTargetDesc depth_target = {
            .view     = depth_view,
            .load_op  = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear    = { .depth = 1.0f },
        };
        gpu::RenderPassDesc pass = {
            .colors = colors[..],
            .depth  = &depth_target,
            .width  = RT_SIZE,
            .height = RT_SIZE,
        };
        gpu::GraphicsState graphics_state =
            gpu::render_geometry_state(RT_SIZE, RT_SIZE)!!;
        gpu::ColorTargetState[1] color_targets = {{
            .write_mask = gpu::COLOR_WRITE_ALL,
        }};
        graphics_state.color = { .targets = color_targets[..] };
        if (i == 0) {
            graphics_state.viewport = {
                .width = (float)RT_SIZE / 2.0f, .height = (float)RT_SIZE, .max_depth = 1.0f,
            };
        }
        if (i == 0) {
            gpu::cmd_begin_render_pass(&cmd, &pass)!!;
            gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
            gpu::cmd_set_graphics_state(&cmd, &graphics_state)!!;
        } else {
            if (i == 2) {
                gpu::ScissorRect half_scissor =
                    graphics_state.scissor;
                half_scissor.width = RT_SIZE / 2;
                gpu::cmd_set_viewport(
                    &cmd,
                    &graphics_state.viewport,
                )!!;
                gpu::cmd_set_scissor(&cmd, &half_scissor)!!;
            } else if (i == 3) {
                gpu::cmd_set_graphics_state(&cmd, &graphics_state)!!;
            } else if (i == 4) {
                gpu::GraphicsState invalid = graphics_state;
                invalid.viewport.width = (float)RT_SIZE / 2.0f;
                invalid.scissor.width = RT_SIZE / 2;
                invalid.raster.cull_mode = gpu::CullMode.BACK;
                invalid.depth = {
                    .test_enable = true,
                    .compare     = gpu::CompareOp.NEVER,
                };
                gpu::ColorTargetState[1] invalid_color_targets = {{
                    .write_mask = gpu::COLOR_WRITE_ALL,
                }};
                uint invalid_write_mask = 1u << 8;
                mem::copy(
                    &invalid_color_targets[0].write_mask,
                    &invalid_write_mask,
                    uint::size,
                );
                invalid.color.targets = invalid_color_targets[..];
                assert(@catch(gpu::cmd_set_graphics_state(
                    &cmd,
                    &invalid,
                )) == gpu::INVALID_ARGUMENT,
                    "late color validation rejects the complete packet");
            }
            gpu::cmd_begin_render_pass(&cmd, &pass)!!;
        }
        if (i == 0) {
            gpu::Viewport rejected = graphics_state.viewport;
            rejected.width =
                (float)state.viewport_limits.max_width * 2.0f;
            assert(@catch(gpu::cmd_set_viewport(&cmd, &rejected)) == gpu::INVALID_ARGUMENT,
                "a rejected viewport does not replace the valid viewport");
        } else if (i == 2) {
            gpu::ScissorRect rejected = graphics_state.scissor;
            rejected.x = int::max;
            rejected.width = 1;
            assert(@catch(gpu::cmd_set_scissor(&cmd, &rejected)) == gpu::INVALID_ARGUMENT,
                "a rejected scissor does not replace the valid scissor");
        }
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
            .before  = {
                .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
                .stages = { .color_output },
                .access = { .read, .write },
            },
            .after   = {
                .layout = gpu::TextureLayout.TRANSFER_SOURCE,
                .stages = { .transfer },
                .access = { .read },
            },
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

    usz left_green =
        ((RT_SIZE / 2) * RT_SIZE + RT_SIZE / 4) * 4 + 1;
    usz right_green =
        ((RT_SIZE / 2) * RT_SIZE + RT_SIZE * 3 / 4) * 4 + 1;
    char* viewport_pixels =
        (char*)test_span_mapping(&device, readback_all).ptr;
    assert(viewport_pixels[left_green] != 0,
        "the half viewport covers the left half");
    assert(viewport_pixels[right_green] == 0,
        "the half viewport leaves the right half clear");
    char* persisted_pixels = viewport_pixels + RT_BYTES;
    assert(persisted_pixels[left_green] != 0,
        "the persisted viewport covers the left half");
    assert(persisted_pixels[right_green] == 0,
        "the persisted viewport leaves the right half clear");
    char* scissor_pixels = viewport_pixels + RT_BYTES * 2;
    assert(scissor_pixels[left_green] != 0,
        "the half scissor covers the left half");
    assert(scissor_pixels[right_green] == 0,
        "the half scissor leaves the right half clear");
    char* reset_pixels = viewport_pixels + RT_BYTES * 3;
    assert(reset_pixels[right_green] != 0,
        "an explicit complete replacement restores full raster coverage");
    char* atomic_pixels = viewport_pixels + RT_BYTES * 4;
    assert(atomic_pixels[left_green] != 0
        && atomic_pixels[right_green] != 0,
        "late color failure emits no viewport, scissor, raster, or depth state");
}

fn void repeated_complete_color_state_output_matches() @test {
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_color_output_test",
    };
    gpu::Device? maybe_device = create_test_device(&runtime_desc);
    if (catch err = maybe_device) {
        assert(err == gpu::UNSUPPORTED_FEATURE,
            "color output fails only when live support is absent");
        io::eprintfn("repeated_complete_color_state_output_matches: unsupported, skipping");
        return;
    }
    gpu::Device device = maybe_device;
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc vertex_desc = {
        .spirv       = DYNAMIC_RECT_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc fragment_desc = {
        .spirv       = DYNAMIC_RECT_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc vertex = vertex_desc;
    gpu::ShaderDesc fragment = fragment_desc;
    gpu::ColorWriteMask red_green = { .red, .green };
    gpu::ColorTargetState[5] packets = {
        gpu::color_blend_disabled(),
        gpu::alpha_blend(),
        gpu::premultiplied_alpha_blend(),
        gpu::additive_blend(),
        gpu::color_blend_disabled(red_green),
    };
    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader   = vertex,
        .fragment_shader = fragment,
        .color_formats   = color_formats[..],
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(
        &device,
        &pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::TextureHandle[10] targets;
    foreach (&target : targets) *target = make_render_target(&device);
    defer foreach (target : targets) {
        gpu::destroy_texture(&device, target)!!;
    }
    gpu::AttachmentViewHandle[10] views;
    foreach (i, &view : views) {
        *view = create_test_attachment_view(&device, targets[i])!!;
    }
    defer foreach (&view : views) {
        destroy_test_attachment_view(&device, view)!!;
    }
    OwnedSpan vertices = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "color_output_vertices",
    )!!;
    defer free_owned_span(&device, &vertices)!!;
    float[12] vertex_data = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         3.0f, -1.0f, 0.0f, 0.0f,
        -1.0f,  3.0f, 0.0f, 0.0f,
    };
    foreach (i, value : vertex_data) {
        ((float*)test_span_mapping(&device, vertices.span).ptr)[i] = value;
    }
    OwnedSpan color = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "color_output_root",
    )!!;
    defer free_owned_span(&device, &color)!!;
    float[4] source_color = { 0.5f, 0.5f, 0.5f, 0.5f };
    foreach (i, value : source_color) {
        ((float*)test_span_mapping(&device, color.span).ptr)[i] = value;
    }
    gpu::flush_mapped_span(&device, vertices.span)!!;
    gpu::flush_mapped_span(&device, color.span)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    foreach (i, target : targets) {
        usz state_index = (usz)i / 2;
        gpu::TextureBarrier to_attachment = {
            .texture = target,
            .before  = { .layout = gpu::TextureLayout.UNDEFINED },
            .after   = {
                .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
                .stages = { .color_output },
                .access = { .read, .write },
            },
        };
        gpu::cmd_texture_barrier(&commands, &to_attachment)!!;
        gpu::ColorTargetDesc[1] attachments = {{
            .view     = views[i],
            .load_op  = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear    = { .rgba = { 0.25f, 0.25f, 0.25f, 0.25f } },
        }};
        gpu::RenderPassDesc pass = {
            .colors = attachments[..],
            .width  = RT_SIZE,
            .height = RT_SIZE,
        };
        gpu::GraphicsState graphics =
            gpu::render_geometry_state(RT_SIZE, RT_SIZE)!!;
        gpu::ColorTargetState[1] color_targets = {
            packets[state_index],
        };
        graphics.color = { .targets = color_targets[..] };
        gpu::cmd_begin_render_pass(&commands, &pass)!!;
        gpu::cmd_bind_pipeline(&commands, pipeline)!!;
        gpu::cmd_set_graphics_state(&commands, &graphics)!!;
        gpu::cmd_draw(
            commands:       &commands,
            vertex_root:    test_span_address(&device, vertices.span),
            fragment_root:  test_span_address(&device, color.span),
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::cmd_end_render_pass(&commands)!!;
    }
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;

    char[RT_BYTES * 10] pixels;
    gpu::TextureState color_attachment = {
        .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
        .stages = { .color_output },
        .access = { .read, .write },
    };
    foreach (i, target : targets) {
        readback_test_texture_data(
            device:   &device,
            src:      target,
            mip:      0,
            out_data: pixels[RT_BYTES * (usz)i : RT_BYTES],
            from_use: color_attachment,
        )!!;
    }
    uint[20] expected = {
        128, 128, 128, 128,
         96,  96,  96, 160,
        160, 160, 160, 160,
        192, 192, 192, 192,
        128, 128,  64,  64,
    };
    usz center = ((RT_SIZE / 2) * RT_SIZE + RT_SIZE / 2) * 4;
    for (usz state_index = 0; state_index < packets.len; state_index++) {
        char[] first_pixels =
            pixels[RT_BYTES * state_index * 2 : RT_BYTES];
        char[] repeated_pixels =
            pixels[RT_BYTES * (state_index * 2 + 1) : RT_BYTES];
        assert(first_pixels == repeated_pixels,
            "repeated complete color state renders identical bytes");
        for (uint channel = 0; channel < 4; channel++) {
            uint actual = (uint)repeated_pixels[center + channel];
            uint expected_value = expected[state_index * 4 + channel];
            uint difference = actual >= expected_value
                ? actual - expected_value
                : expected_value - actual;
            assert(difference <= COLOR_READBACK_TOLERANCE,
                "color state %d channel %d readback %d is within one LSB of expected %d",
                state_index,
                channel,
                actual,
                expected_value);
        }
    }
}

fn void offscreen_render_to_mip() @test {
    gpu::Device device = make_offscreen_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device, mips: 2);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        device:    &device,
        texture:   target,
        mip_level: 1,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    const uint MIP1 = RT_SIZE / 2;
    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         MIP1 * MIP1 * 4,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "mip_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view,
        .load_op = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 40.0f / 255.0f, 80.0f / 255.0f, 120.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = MIP1, .height = MIP1 };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::TextureBarrier to_src = {
        .texture = target,
        .before  = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
        .after   = {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
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

    gpu::ShaderDesc vert_sd = { .spirv = OFFSCREEN_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc vert = vert_sd;
    gpu::ShaderDesc frag_sd = { .spirv = OFFSCREEN_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc frag = frag_sd;

    gpu::Format[1] pipeline_color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc with_depth = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .color_formats   = pipeline_color_formats[..],
        .depth_format    = gpu::Format.D32_FLOAT,
    };
    gpu::PipelineHandle depth_pipeline = gpu::create_graphics_pipeline(&device, &with_depth)!!;
    gpu::destroy_pipeline(&device, depth_pipeline)!!;

    gpu::GraphicsPipelineDesc color_as_depth = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .color_formats   = pipeline_color_formats[..],
        .depth_format    = gpu::Format.RGBA8_UNORM,
    };
    assert(@catch(gpu::create_graphics_pipeline(&device, &color_as_depth)) == gpu::INVALID_ARGUMENT,
        "a color format used as depth faults INVALID_ARGUMENT");

    gpu::GraphicsPipelineDesc no_attachments = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
    };
    assert(@catch(gpu::create_graphics_pipeline(&device, &no_attachments)) == gpu::INVALID_ARGUMENT,
        "zero color formats with no depth format faults INVALID_ARGUMENT");
}

fn void dynamic_viewport_scissor_validation() @test {
    VulkanValidationCapture validation_capture = {};
    gpu::Device device = make_offscreen_test_device(
        &validation_capture,
    );
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
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
    gpu::cmd_set_viewport(&cmd, &full_viewport)!!;
    gpu::cmd_set_scissor(&cmd, &full_scissor)!!;

    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    assert(@catch(gpu::cmd_set_viewport(&cmd, null)) == gpu::INVALID_ARGUMENT,
        "null viewport faults INVALID_ARGUMENT");
    assert(@catch(gpu::cmd_set_scissor(&cmd, null)) == gpu::INVALID_ARGUMENT,
        "null scissor faults INVALID_ARGUMENT");

    gpu::Viewport[10] invalid_viewports = {
        { .x = float::nan, .width = 1.0f, .height = 1.0f, .max_depth = 1.0f },
        { .y = float::inf, .width = 1.0f, .height = 1.0f, .max_depth = 1.0f },
        { .width = float::nan, .height = 1.0f, .max_depth = 1.0f },
        { .width = 1.0f, .height = -float::inf, .max_depth = 1.0f },
        { .width = 1.0f, .height = 1.0f, .min_depth = float::nan, .max_depth = 1.0f },
        { .width = 1.0f, .height = 1.0f, .max_depth = float::inf },
        { .width = 0.0f, .height = 1.0f, .max_depth = 1.0f },
        { .width = 1.0f, .height = 0.0f, .max_depth = 1.0f },
        { .width = 1.0f, .height = 1.0f, .min_depth = -0.1f, .max_depth = 1.0f },
        { .width = 1.0f, .height = 1.0f, .max_depth = 1.1f },
    };
    foreach (&viewport : invalid_viewports) {
        assert(@catch(gpu::cmd_set_viewport(&cmd, viewport)) == gpu::INVALID_ARGUMENT,
            "invalid viewport faults INVALID_ARGUMENT");
    }

    validation_capture.errors = 0;
    gpu::Viewport reversed_depth = full_viewport;
    reversed_depth.min_depth = 1.0f;
    reversed_depth.max_depth = 0.0f;
    gpu::cmd_set_viewport(&cmd, &reversed_depth)!!;

    gpu::Viewport negative_height = full_viewport;
    negative_height.y = (float)RT_SIZE;
    negative_height.height = -(float)RT_SIZE;
    gpu::cmd_set_viewport(&cmd, &negative_height)!!;

    gpu::Viewport negative_origin = full_viewport;
    negative_origin.x = -1.0f;
    gpu::cmd_set_viewport(&cmd, &negative_origin)!!;

    gpu::Viewport overscan = full_viewport;
    overscan.width = (float)RT_SIZE + 1.0f;
    gpu::cmd_set_viewport(&cmd, &overscan)!!;

    gpu::ScissorRect[6] invalid_scissors = {
        { .x = -1, .width = 1, .height = 1 },
        { .y = -1, .width = 1, .height = 1 },
        { .width = -1, .height = 1 },
        { .width = 1, .height = -1 },
        { .x = int::max, .width = 1, .height = 1 },
        { .y = int::max, .width = 1, .height = 1 },
    };
    foreach (&scissor : invalid_scissors) {
        assert(@catch(gpu::cmd_set_scissor(&cmd, scissor)) == gpu::INVALID_ARGUMENT,
            "invalid scissor faults INVALID_ARGUMENT");
    }

    gpu::ScissorRect empty_scissor = {};
    gpu::cmd_set_scissor(&cmd, &empty_scissor)!!;
    gpu::ScissorRect off_pass_scissor = {
        .x      = (int)RT_SIZE,
        .y      = (int)RT_SIZE,
        .width  = 1,
        .height = 1,
    };
    gpu::cmd_set_scissor(&cmd, &off_pass_scissor)!!;
    assert(validation_capture.errors == 0,
        "newly accepted dynamic states emit no Vulkan validation error");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::cmd_set_viewport(&cmd, &full_viewport)!!;
    gpu::cmd_set_scissor(&cmd, &full_scissor)!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void render_pass_state_machine() @test {
    gpu::Device device = make_offscreen_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle target = make_render_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    OwnedSpan data = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "render_pass_data",
    )!!;
    defer free_owned_span(&device, &data)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
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
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
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
