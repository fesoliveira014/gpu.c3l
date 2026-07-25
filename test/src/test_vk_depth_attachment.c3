module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;

// Headless (lavapipe): D32_FLOAT depth attachments. Depth-tested overlap in
// both draw orders, write/test toggles through separately bound dynamic state,
// one cache entry across depth variants, depth-only passes, sampled depth,
// and depth-value readback against exact NDC z.

const char[*] DEPTH_VERT_SPIRV   = $embed("shaders/depth.vert.spv");
const char[*] DEPTH_FRAG_SPIRV   = $embed("shaders/depth.frag.spv");
const char[*] DEPTH_SAMPLE_SPIRV = $embed("shaders/sample_depth.comp.spv");

const uint DP_RT = 16;
const usz DP_COLOR_BYTES = DP_RT * DP_RT * 4;
const usz DP_DEPTH_BYTES = DP_RT * DP_RT * 4;
const float NEAR_Z = 0.25f;
const float FAR_Z = 0.75f;

struct DepthCtx {
    gpu::ShaderCode vert;
    gpu::ShaderCode frag;
}

fn gpu::Device make_depth_device() {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_depth_test",
    };
    return create_test_device(&desc)!!;
}

fn DepthCtx make_depth_ctx() {
    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = DEPTH_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = DEPTH_FRAG_SPIRV[..], .entry_point = "main" };
    return {
        .vert = gpu::prepare_shader_code(&vert_sd)!!,
        .frag = gpu::prepare_shader_code(&frag_sd)!!,
    };
}

fn gpu::PipelineHandle make_depth_pipeline(gpu::Device* device, DepthCtx* ctx, bool with_color = true) {
    gpu::Format[1] pipeline_color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .color_formats   = with_color
            ? pipeline_color_formats[..]
            : (gpu::Format[]){},
        .depth_format    = gpu::Format.D32_FLOAT,
    };
    return gpu::create_graphics_pipeline(device, &desc)!!;
}

fn gpu::TextureHandle make_depth_texture(gpu::Device* device, bool sampled = false) {
    gpu::TextureDesc desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = DP_RT,
        .height    = DP_RT,
        .format    = gpu::Format.D32_FLOAT,
        .usage     = { .depth_attach, .transfer_src, .sampled = sampled },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::TextureHandle make_color_target(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = DP_RT,
        .height    = DP_RT,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn OwnedSpan write_tri(gpu::Device* device, float z) {
    OwnedSpan owned = allocate_owned_span(
        device:       device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "depth_vertices",
    )!!;
    float* verts = (float*)test_span_mapping(device, owned.span).ptr;
    float[12] data = {
        -0.8f, -0.8f, z, 0.0f,
         0.8f, -0.8f, z, 0.0f,
         0.0f,  0.8f, z, 0.0f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = data[i];
    gpu::flush_mapped_span(device, owned.span)!!;
    return owned;
}

fn OwnedSpan write_color(gpu::Device* device, float[4] rgba) {
    OwnedSpan owned = allocate_owned_span(
        device:       device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "depth_color_root",
    )!!;
    float* color = (float*)test_span_mapping(device, owned.span).ptr;
    for (uint i = 0; i < 4; i++) color[i] = rgba[i];
    gpu::flush_mapped_span(device, owned.span)!!;
    return owned;
}

fn void record_depth_to_attach(gpu::CommandList* cmd, gpu::TextureHandle depth) {
    gpu::TextureBarrier to_attach = gpu::texture_transition(
        texture: depth,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   {
            .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
            .stages = { .depth_output },
            .access = { .read, .write },
        },
    )!!;
    gpu::cmd_texture_barrier(cmd, &to_attach)!!;
}

fn void record_color_to_attach(gpu::CommandList* cmd, gpu::TextureHandle target) {
    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(cmd, &to_attach)!!;
}

fn void record_color_readback(gpu::CommandList* cmd, gpu::TextureHandle target, gpu::GpuSpan* rb_span) {
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
    gpu::cmd_texture_barrier(cmd, &to_src)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = *rb_span };
    gpu::cmd_copy_texture_to_buffer(cmd, &copy)!!;
    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(cmd, &host_read)!!;
}

fn void record_depth_readback(gpu::CommandList* cmd, gpu::TextureHandle depth, gpu::GpuSpan* rb_span) {
    gpu::TextureBarrier to_src = {
        .texture = depth,
        .before  = {
            .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
            .stages = { .depth_output },
            .access = { .read, .write },
        },
        .after   = {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
    };
    gpu::cmd_texture_barrier(cmd, &to_src)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = depth, .dst = *rb_span };
    gpu::cmd_copy_texture_to_buffer(cmd, &copy)!!;
    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(cmd, &host_read)!!;
}

fn OwnedSpan make_readback(gpu::Device* device, usz size) {
    return allocate_owned_span(
        device:       device,
        size:         size,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "depth_readback",
    )!!;
}

fn void submit_and_finish(gpu::Device* device, gpu::CommandList* commands) {
    gpu::ExecutableCommandList[1] lists = { gpu::end_commands(commands)!! };
    defer discard_test_executable(&lists[0])!!;
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
}

fn char* dp_center(gpu::Device* device, gpu::GpuSpan* rb_span) {
    char* pixels = (char*)test_span_mapping(device, *rb_span).ptr;
    return &pixels[(DP_RT / 2 * DP_RT + DP_RT / 2) * 4];
}

fn float* dp_center_depth(gpu::Device* device, gpu::GpuSpan* rb_span) {
    float* texels = (float*)test_span_mapping(device, *rb_span).ptr;
    return &texels[DP_RT / 2 * DP_RT + DP_RT / 2];
}

fn void depth_test_occludes_both_orders() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx();

    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle pipeline = make_depth_pipeline(&device, &ctx);
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;

    bool[2] near_first_options = { true, false };
    foreach (near_first : near_first_options) {
        gpu::TextureHandle target = make_color_target(&device);
        defer gpu::destroy_texture(&device, target)!!;
        gpu::AttachmentViewHandle target_view = create_test_attachment_view(
            &device,
            target,
        )!!;
        defer destroy_test_attachment_view(&device, &target_view)!!;
        gpu::TextureHandle depth = make_depth_texture(&device);
        defer gpu::destroy_texture(&device, depth)!!;
        gpu::AttachmentViewHandle depth_view = create_test_attachment_view(
            &device,
            depth,
        )!!;
        defer destroy_test_attachment_view(&device, &depth_view)!!;
        OwnedSpan color_rb = make_readback(&device, DP_COLOR_BYTES);
        defer free_owned_span(&device, &color_rb)!!;
        OwnedSpan depth_rb = make_readback(&device, DP_DEPTH_BYTES);
        defer free_owned_span(&device, &depth_rb)!!;
        gpu::GpuSpan color_rb_span = color_rb.span;
        gpu::GpuSpan depth_rb_span = depth_rb.span;
        OwnedSpan near_verts_owned = write_tri(&device, NEAR_Z);
        defer free_owned_span(&device, &near_verts_owned)!!;
        gpu::GpuSpan near_verts = near_verts_owned.span;
        OwnedSpan far_verts_owned = write_tri(&device, FAR_Z);
        defer free_owned_span(&device, &far_verts_owned)!!;
        gpu::GpuSpan far_verts = far_verts_owned.span;
        OwnedSpan red_owned = write_color(&device, { 1.0f, 0.0f, 0.0f, 1.0f });
        defer free_owned_span(&device, &red_owned)!!;
        gpu::GpuSpan red = red_owned.span;
        OwnedSpan green_owned = write_color(&device, { 0.0f, 1.0f, 0.0f, 1.0f });
        defer free_owned_span(&device, &green_owned)!!;
        gpu::GpuSpan green = green_owned.span;

        gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
        defer discard_test_commands(&cmd)!!;
        record_color_to_attach(&cmd, target);
        record_depth_to_attach(&cmd, depth);

        gpu::ColorTargetDesc[1] colors = {{
            .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
            .clear = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
        }};
        gpu::DepthTargetDesc depth_target = {
            .view = depth_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
            .clear = { .depth = 1.0f },
        };
        gpu::RenderPassDesc pass = { .colors = colors[..], .depth = &depth_target, .width = DP_RT, .height = DP_RT };
        gpu::GraphicsState graphics_state =
            gpu::render_geometry_state(DP_RT, DP_RT)!!;
        graphics_state.depth = tested;
        gpu::ColorTargetState[1] color_targets = {{
            .write_mask = gpu::COLOR_WRITE_ALL,
        }};
        graphics_state.color = { .targets = color_targets[..] };
        gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
        gpu::cmd_begin_render_pass_with_state(&cmd, &pass, &graphics_state)!!;

        gpu::GpuSpan first_verts = near_first ? near_verts : far_verts;
        gpu::GpuSpan first_color = near_first ? red : green;
        gpu::GpuSpan second_verts = near_first ? far_verts : near_verts;
        gpu::GpuSpan second_color = near_first ? green : red;
        gpu::cmd_draw(
            commands:       &cmd,
            vertex_root:    test_span_address(&device, first_verts),
            fragment_root:  test_span_address(&device, first_color),
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
        gpu::cmd_set_depth_state(&cmd, &tested)!!;
        gpu::cmd_draw(
            commands:       &cmd,
            vertex_root:    test_span_address(&device, second_verts),
            fragment_root:  test_span_address(&device, second_color),
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::cmd_end_render_pass(&cmd)!!;

        record_color_readback(&cmd, target, &color_rb_span);
        record_depth_readback(&cmd, depth, &depth_rb_span);
        submit_and_finish(&device, &cmd);

        gpu::invalidate_mapped_span(&device, color_rb.span)!!;
        char* center = dp_center(&device, &color_rb_span);
        assert(center[0] == (char)255 && center[1] == (char)0,
            "the near triangle wins regardless of draw order");

        gpu::invalidate_mapped_span(&device, depth_rb.span)!!;
        float* depth_center = dp_center_depth(&device, &depth_rb_span);
        assert(*depth_center == NEAR_Z, "depth buffer holds the near z");
        float* depth_corner = (float*)test_span_mapping(&device, depth_rb_span).ptr;
        assert(depth_corner[0] == 1.0f, "untouched pixels hold the clear depth");
    }
}

fn void depth_write_toggle_honored() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx();
    gpu::TextureHandle target = make_color_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    gpu::TextureHandle depth = make_depth_texture(&device);
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::AttachmentViewHandle depth_view = create_test_attachment_view(
        &device,
        depth,
    )!!;
    defer destroy_test_attachment_view(&device, &depth_view)!!;
    OwnedSpan color_rb = make_readback(&device, DP_COLOR_BYTES);
    defer free_owned_span(&device, &color_rb)!!;
    OwnedSpan depth_rb = make_readback(&device, DP_DEPTH_BYTES);
    defer free_owned_span(&device, &depth_rb)!!;
    gpu::GpuSpan color_rb_span = color_rb.span;
    gpu::GpuSpan depth_rb_span = depth_rb.span;

    gpu::DepthState no_write = { .test_enable = true, .write_enable = false, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle near_pipeline = make_depth_pipeline(&device, &ctx);
    defer gpu::destroy_pipeline(&device, near_pipeline)!!;
    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle far_pipeline = make_depth_pipeline(&device, &ctx);
    defer gpu::destroy_pipeline(&device, far_pipeline)!!;
    OwnedSpan near_verts_owned = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &near_verts_owned)!!;
    gpu::GpuSpan near_verts = near_verts_owned.span;
    OwnedSpan far_verts_owned = write_tri(&device, FAR_Z);
    defer free_owned_span(&device, &far_verts_owned)!!;
    gpu::GpuSpan far_verts = far_verts_owned.span;
    OwnedSpan red_owned = write_color(&device, { 1.0f, 0.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &red_owned)!!;
    gpu::GpuSpan red = red_owned.span;
    OwnedSpan green_owned = write_color(&device, { 0.0f, 1.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &green_owned)!!;
    gpu::GpuSpan green = green_owned.span;

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_color_to_attach(&cmd, target);
    record_depth_to_attach(&cmd, depth);
    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::DepthTargetDesc depth_target = {
        .view = depth_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::RenderPassDesc pass = { .colors = colors[..], .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(DP_RT, DP_RT)!!;
    graphics_state.depth = no_write;
    gpu::ColorTargetState[1] color_targets = {{
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    graphics_state.color = { .targets = color_targets[..] };
    gpu::cmd_bind_pipeline(&cmd, near_pipeline)!!;
    gpu::cmd_begin_render_pass_with_state(&cmd, &pass, &graphics_state)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, near_verts),
        fragment_root:  test_span_address(&device, red),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_bind_pipeline(&cmd, far_pipeline)!!;
    gpu::cmd_set_depth_state(&cmd, &tested)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, far_verts),
        fragment_root:  test_span_address(&device, green),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    record_color_readback(&cmd, target, &color_rb_span);
    record_depth_readback(&cmd, depth, &depth_rb_span);
    submit_and_finish(&device, &cmd);

    gpu::invalidate_mapped_span(&device, color_rb.span)!!;
    char* center = dp_center(&device, &color_rb_span);
    assert(center[1] == (char)255, "far triangle renders because the near draw wrote no depth");
    gpu::invalidate_mapped_span(&device, depth_rb.span)!!;
    assert(*dp_center_depth(&device, &depth_rb_span) == FAR_Z, "depth holds the far z, written by the tested draw");
}

fn void depth_state_variants_share_entry() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx();
    gpu::TextureHandle target = make_color_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    gpu::TextureHandle depth = make_depth_texture(&device);
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::AttachmentViewHandle depth_view = create_test_attachment_view(
        &device,
        depth,
    )!!;
    defer destroy_test_attachment_view(&device, &depth_view)!!;
    OwnedSpan color_rb = make_readback(&device, DP_COLOR_BYTES);
    defer free_owned_span(&device, &color_rb)!!;
    gpu::GpuSpan color_rb_span = color_rb.span;

    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle test_on = make_depth_pipeline(&device, &ctx);
    defer gpu::destroy_pipeline(&device, test_on)!!;
    gpu::DepthState untested = {};
    gpu::PipelineHandle test_off = make_depth_pipeline(&device, &ctx);
    defer gpu::destroy_pipeline(&device, test_off)!!;

    uint entries = vk::pipeline_cache_entry_count(gpu::internal::device_slots[device.index].data.state);
    assert(entries == 1, "depth-state variants share one cache entry");
    OwnedSpan near_verts_owned = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &near_verts_owned)!!;
    gpu::GpuSpan near_verts = near_verts_owned.span;
    OwnedSpan far_verts_owned = write_tri(&device, FAR_Z);
    defer free_owned_span(&device, &far_verts_owned)!!;
    gpu::GpuSpan far_verts = far_verts_owned.span;
    OwnedSpan red_owned = write_color(&device, { 1.0f, 0.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &red_owned)!!;
    gpu::GpuSpan red = red_owned.span;
    OwnedSpan green_owned = write_color(&device, { 0.0f, 1.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &green_owned)!!;
    gpu::GpuSpan green = green_owned.span;

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_color_to_attach(&cmd, target);
    record_depth_to_attach(&cmd, depth);
    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::DepthTargetDesc depth_target = {
        .view = depth_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::RenderPassDesc pass = { .colors = colors[..], .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(DP_RT, DP_RT)!!;
    graphics_state.depth = tested;
    gpu::ColorTargetState[1] color_targets = {{
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    graphics_state.color = { .targets = color_targets[..] };
    gpu::cmd_bind_pipeline(&cmd, test_on)!!;
    gpu::cmd_begin_render_pass_with_state(&cmd, &pass, &graphics_state)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, near_verts),
        fragment_root:  test_span_address(&device, red),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_bind_pipeline(&cmd, test_off)!!;
    gpu::cmd_set_depth_state(&cmd, &untested)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, far_verts),
        fragment_root:  test_span_address(&device, green),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    record_color_readback(&cmd, target, &color_rb_span);
    submit_and_finish(&device, &cmd);

    gpu::invalidate_mapped_span(&device, color_rb.span)!!;
    char* center = dp_center(&device, &color_rb_span);
    assert(center[1] == (char)255, "test-off alias of the same entry overwrites the near triangle");
}

fn void depth_only_pass_renders_footprint() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx();
    gpu::TextureHandle depth = make_depth_texture(&device);
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::AttachmentViewHandle depth_view = create_test_attachment_view(
        &device,
        depth,
    )!!;
    defer destroy_test_attachment_view(&device, &depth_view)!!;
    OwnedSpan depth_rb = make_readback(&device, DP_DEPTH_BYTES);
    defer free_owned_span(&device, &depth_rb)!!;
    gpu::GpuSpan depth_rb_span = depth_rb.span;

    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle pipeline = make_depth_pipeline(
        device:     &device,
        ctx:        &ctx,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    OwnedSpan near_verts_owned = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &near_verts_owned)!!;
    gpu::GpuSpan near_verts = near_verts_owned.span;
    OwnedSpan color_owned = write_color(&device, { 1.0f, 1.0f, 1.0f, 1.0f });
    defer free_owned_span(&device, &color_owned)!!;
    gpu::GpuSpan color = color_owned.span;

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_depth_to_attach(&cmd, depth);
    gpu::DepthTargetDesc depth_target = {
        .view = depth_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(DP_RT, DP_RT)!!;
    graphics_state.depth = tested;
    gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
    gpu::cmd_begin_render_pass_with_state(&cmd, &pass, &graphics_state)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, near_verts),
        fragment_root:  test_span_address(&device, color),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    record_depth_readback(&cmd, depth, &depth_rb_span);
    submit_and_finish(&device, &cmd);

    gpu::invalidate_mapped_span(&device, depth_rb.span)!!;
    assert(*dp_center_depth(&device, &depth_rb_span) == NEAR_Z, "depth-only pass writes the triangle footprint");
    float* corner = (float*)test_span_mapping(&device, depth_rb_span).ptr;
    assert(corner[0] == 1.0f, "outside the footprint the clear depth remains");
}

fn void sampled_depth_matches_readback() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx();
    gpu::TextureHandle depth = make_depth_texture(&device, sampled: true);
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::AttachmentViewHandle depth_view = create_test_attachment_view(
        &device,
        depth,
    )!!;
    defer destroy_test_attachment_view(&device, &depth_view)!!;

    gpu::TextureView tex_index_view = gpu::create_texture_view(&device, depth, null)!!;
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
    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DEPTH_SAMPLE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode comp = gpu::prepare_shader_code(&comp_sd)!!;
    gpu::ComputePipelineDesc comp_desc = { .shader = comp };
    gpu::PipelineHandle sample_pipeline = gpu::create_compute_pipeline(&device, &comp_desc)!!;
    defer gpu::destroy_pipeline(&device, sample_pipeline)!!;

    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle pipeline = make_depth_pipeline(
        device:     &device,
        ctx:        &ctx,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    OwnedSpan near_verts_owned = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &near_verts_owned)!!;
    gpu::GpuSpan near_verts = near_verts_owned.span;
    OwnedSpan color_owned = write_color(&device, { 1.0f, 1.0f, 1.0f, 1.0f });
    defer free_owned_span(&device, &color_owned)!!;
    gpu::GpuSpan color = color_owned.span;
    OwnedSpan out_owned = allocate_owned_span(
        device:       &device,
        size:         4,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics },
        debug_name:   "sampled_depth_output",
    )!!;
    defer free_owned_span(&device, &out_owned)!!;
    gpu::GpuSpan out_span = out_owned.span;
    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         SampleDepthRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "sampled_depth_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    gpu::GpuSpan root_span = root_owned.span;
    SampleDepthRoot* root = (SampleDepthRoot*)test_span_mapping(&device, root_span).ptr;
    root.out_gpu = test_span_address(&device, out_span);
    root.texture_index = tex_index;
    root.sampler_index = smp_index;
    root.u = 0.5f;
    root.v = 0.5f;
    gpu::flush_mapped_span(&device, root_span)!!;

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_depth_to_attach(&cmd, depth);
    gpu::DepthTargetDesc depth_target = {
        .view = depth_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(DP_RT, DP_RT)!!;
    graphics_state.depth = tested;
    gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
    gpu::cmd_begin_render_pass_with_state(&cmd, &pass, &graphics_state)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, near_verts),
        fragment_root:  test_span_address(&device, color),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::TextureBarrier to_read = {
        .texture = depth,
        .before  = {
            .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
            .stages = { .depth_output },
            .access = { .read, .write },
        },
        .after   = gpu::sampled_at({ .compute }),
    };
    gpu::cmd_texture_barrier(&cmd, &to_read)!!;
    bind_test_compute_pipeline(&cmd, sample_pipeline);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, root_span),
        groups:   { 1, 1, 1 },
    )!!;
    gpu::Barrier host_read = {
        .before = { .compute = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&cmd, &host_read)!!;
    submit_and_finish(&device, &cmd);
    gpu::invalidate_mapped_span(&device, out_span)!!;

    float* sampled = (float*)test_span_mapping(&device, out_span).ptr;
    assert(*sampled == NEAR_Z, "heap-sampled depth matches the rendered z");
}

fn void depth_nonzero_mip_layer_isolated() @test {
    const uint MIP_SIZE = DP_RT / 2;
    const usz MIP_BYTES = MIP_SIZE * MIP_SIZE * float::size;
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx();
    gpu::TextureDesc texture_desc = {
        .width        = DP_RT,
        .height       = DP_RT,
        .mip_levels   = 2,
        .array_layers = 2,
        .format       = gpu::Format.D32_FLOAT,
        .usage        = { .depth_attach, .transfer_src },
        .access       = { .graphics, .compute, .transfer },
    };
    gpu::TextureHandle depth = gpu::create_texture(&device, &texture_desc)!!;
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::AttachmentViewDesc target_desc = {
        .texture     = depth,
        .mip_level   = 1,
        .array_layer = 1,
    };
    gpu::AttachmentViewHandle target = gpu::create_attachment_view(
        &device,
        &target_desc,
    )!!;
    defer gpu::destroy_attachment_view(&device, target)!!;
    gpu::AttachmentViewDesc other_layer_desc = {
        .texture     = depth,
        .mip_level   = 1,
        .array_layer = 0,
    };
    gpu::AttachmentViewHandle other_layer = gpu::create_attachment_view(
        &device,
        &other_layer_desc,
    )!!;
    defer gpu::destroy_attachment_view(&device, other_layer)!!;
    gpu::AttachmentViewDesc other_mip_desc = {
        .texture     = depth,
        .mip_level   = 0,
        .array_layer = 1,
    };
    gpu::AttachmentViewHandle other_mip = gpu::create_attachment_view(
        &device,
        &other_mip_desc,
    )!!;
    defer gpu::destroy_attachment_view(&device, other_mip)!!;

    OwnedSpan target_readback = make_readback(&device, MIP_BYTES);
    defer free_owned_span(&device, &target_readback)!!;
    OwnedSpan layer_readback = make_readback(&device, MIP_BYTES);
    defer free_owned_span(&device, &layer_readback)!!;
    OwnedSpan mip_readback = make_readback(&device, DP_DEPTH_BYTES);
    defer free_owned_span(&device, &mip_readback)!!;
    gpu::PipelineHandle pipeline = make_depth_pipeline(
        device:     &device,
        ctx:        &ctx,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    OwnedSpan vertices = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &vertices)!!;
    OwnedSpan color = write_color(&device, { 1.0f, 1.0f, 1.0f, 1.0f });
    defer free_owned_span(&device, &color)!!;

    gpu::TextureState depth_state = {
        .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
        .stages = { .depth_output },
        .access = { .read, .write },
    };
    gpu::TextureViewDesc target_range = {
        .base_mip    = 1,
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };
    gpu::TextureViewDesc layer_range = {
        .base_mip    = 1,
        .mip_count   = 1,
        .base_layer  = 0,
        .layer_count = 1,
    };
    gpu::TextureViewDesc mip_range = {
        .base_mip    = 0,
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };
    gpu::Queue command_queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(
        &device,
        command_queue,
    )!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::TextureBarrier target_to_depth = gpu::texture_view_transition(
        texture: depth,
        view:    target_range,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   depth_state,
    )!!;
    gpu::TextureBarrier layer_to_depth = gpu::texture_view_transition(
        texture: depth,
        view:    layer_range,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   depth_state,
    )!!;
    gpu::TextureBarrier mip_to_depth = gpu::texture_view_transition(
        texture: depth,
        view:    mip_range,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   depth_state,
    )!!;
    gpu::cmd_texture_barrier(&commands, &target_to_depth)!!;
    gpu::cmd_texture_barrier(&commands, &layer_to_depth)!!;
    gpu::cmd_texture_barrier(&commands, &mip_to_depth)!!;

    gpu::DepthTargetDesc layer_target = {
        .view     = other_layer,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear    = { .depth = 0.5f },
    };
    gpu::RenderPassDesc layer_pass = {
        .depth  = &layer_target,
        .width  = MIP_SIZE,
        .height = MIP_SIZE,
    };
    gpu::cmd_begin_render_pass(&commands, &layer_pass)!!;
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::DepthTargetDesc mip_target = {
        .view     = other_mip,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear    = { .depth = 0.75f },
    };
    gpu::RenderPassDesc mip_pass = {
        .depth  = &mip_target,
        .width  = DP_RT,
        .height = DP_RT,
    };
    gpu::cmd_begin_render_pass(&commands, &mip_pass)!!;
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::DepthTargetDesc selected_target = {
        .view     = target,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear    = { .depth = 1.0f },
    };
    gpu::RenderPassDesc selected_pass = {
        .depth  = &selected_target,
        .width  = MIP_SIZE,
        .height = MIP_SIZE,
    };
    gpu::GraphicsState selected_graphics_state =
        gpu::render_geometry_state(MIP_SIZE, MIP_SIZE)!!;
    selected_graphics_state.depth = {
        .test_enable  = true,
        .write_enable = true,
        .compare      = gpu::CompareOp.LESS,
    };
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_begin_render_pass_with_state(
        &commands,
        &selected_pass,
        &selected_graphics_state,
    )!!;
    gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   test_span_address(&device, vertices.span),
        fragment_root: test_span_address(&device, color.span),
        vertex_count:  3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    gpu::TextureState transfer_state = {
        .layout = gpu::TextureLayout.TRANSFER_SOURCE,
        .stages = { .transfer },
        .access = { .read },
    };
    target_to_depth.before = depth_state;
    target_to_depth.after = transfer_state;
    layer_to_depth.before = depth_state;
    layer_to_depth.after = transfer_state;
    mip_to_depth.before = depth_state;
    mip_to_depth.after = transfer_state;
    gpu::cmd_texture_barrier(&commands, &target_to_depth)!!;
    gpu::cmd_texture_barrier(&commands, &layer_to_depth)!!;
    gpu::cmd_texture_barrier(&commands, &mip_to_depth)!!;
    gpu::TextureBufferCopyDesc target_copy = {
        .texture    = depth,
        .dst        = target_readback.span,
        .mip        = 1,
        .base_layer = 1,
    };
    gpu::cmd_copy_texture_to_buffer(&commands, &target_copy)!!;
    gpu::TextureBufferCopyDesc layer_copy = {
        .texture    = depth,
        .dst        = layer_readback.span,
        .mip        = 1,
        .base_layer = 0,
    };
    gpu::cmd_copy_texture_to_buffer(&commands, &layer_copy)!!;
    gpu::TextureBufferCopyDesc mip_copy = {
        .texture    = depth,
        .dst        = mip_readback.span,
        .mip        = 0,
        .base_layer = 1,
    };
    gpu::cmd_copy_texture_to_buffer(&commands, &mip_copy)!!;
    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after  = { .host = true },
    };
    gpu::cmd_barrier(&commands, &host_read)!!;
    submit_and_finish(&device, &commands);
    gpu::invalidate_mapped_span(&device, target_readback.span)!!;
    gpu::invalidate_mapped_span(&device, layer_readback.span)!!;
    gpu::invalidate_mapped_span(&device, mip_readback.span)!!;

    float* selected = (float*)test_span_mapping(
        &device,
        target_readback.span,
    ).ptr;
    assert(selected[MIP_SIZE / 2 * MIP_SIZE + MIP_SIZE / 2] == NEAR_Z,
        "draw writes the selected nonzero mip and layer");
    assert(selected[0] == 1.0f,
        "the selected subresource keeps its clear outside the draw");
    float* untouched_layer = (float*)test_span_mapping(
        &device,
        layer_readback.span,
    ).ptr;
    assert(untouched_layer[MIP_SIZE / 2 * MIP_SIZE + MIP_SIZE / 2] == 0.5f,
        "drawing one layer leaves the neighboring layer unchanged");
    float* untouched_mip = (float*)test_span_mapping(
        &device,
        mip_readback.span,
    ).ptr;
    assert(untouched_mip[DP_RT / 2 * DP_RT + DP_RT / 2] == 0.75f,
        "drawing one mip leaves the neighboring mip unchanged");
}

fn void depth_pass_validation_faults() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc sampled_only_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = DP_RT,
        .height    = DP_RT,
        .format    = gpu::Format.D32_FLOAT,
        .usage     = { .sampled },
    };
    gpu::TextureHandle sampled_only = gpu::create_texture(&device, &sampled_only_desc)!!;
    defer gpu::destroy_texture(&device, sampled_only)!!;

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::AttachmentViewDesc no_usage = { .texture = sampled_only };
    assert(@catch(gpu::create_attachment_view(&device, &no_usage))
        == gpu::INVALID_ARGUMENT,
        "attachment creation rejects a texture without attachment usage");

    gpu::RenderPassDesc empty_pass = { .width = DP_RT, .height = DP_RT };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(DP_RT, DP_RT)!!;
    assert(@catch(gpu::cmd_begin_render_pass_with_state(
        &cmd,
        &empty_pass,
        &graphics_state,
    )) == gpu::INVALID_ARGUMENT,
        "a pass with no attachments faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

const char[*] SHADOW_PROBE_SPIRV = $embed("shaders/shadow_probe.comp.spv");

fn void compare_sampler_exact_results() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx();
    gpu::TextureHandle depth = make_depth_texture(&device, sampled: true);
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::AttachmentViewHandle depth_view = create_test_attachment_view(
        &device,
        depth,
    )!!;
    defer destroy_test_attachment_view(&device, &depth_view)!!;

    gpu::TextureView tex_index_view = gpu::create_texture_view(&device, depth, null)!!;
    gpu::TextureIndex tex_index = tex_index_view.index;
    defer gpu::destroy_texture_view(&device, tex_index_view)!!;
    gpu::SamplerDesc shadow_desc = {
        .min_filter     = gpu::Filter.NEAREST,
        .mag_filter     = gpu::Filter.NEAREST,
        .address_u      = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v      = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w      = gpu::AddressMode.CLAMP_TO_EDGE,
        .compare_enable = true,
        .compare        = gpu::CompareOp.LESS,
    };
    gpu::SamplerIndex shadow_smp = gpu::intern_sampler(&device, &shadow_desc)!!;
    gpu::ShaderDesc probe_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = SHADOW_PROBE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode probe = gpu::prepare_shader_code(&probe_sd)!!;
    gpu::ComputePipelineDesc probe_desc = { .shader = probe };
    gpu::PipelineHandle probe_pipeline = gpu::create_compute_pipeline(&device, &probe_desc)!!;
    defer gpu::destroy_pipeline(&device, probe_pipeline)!!;

    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle pipeline = make_depth_pipeline(
        device:     &device,
        ctx:        &ctx,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    OwnedSpan near_verts_owned = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &near_verts_owned)!!;
    gpu::GpuSpan near_verts = near_verts_owned.span;
    OwnedSpan color_owned = write_color(&device, { 1.0f, 1.0f, 1.0f, 1.0f });
    defer free_owned_span(&device, &color_owned)!!;
    gpu::GpuSpan color = color_owned.span;
    OwnedSpan out_owned = allocate_owned_span(
        device:       &device,
        size:         8,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics },
        debug_name:   "shadow_probe_output",
    )!!;
    defer free_owned_span(&device, &out_owned)!!;
    gpu::GpuSpan out_span = out_owned.span;
    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         ShadowProbeRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "shadow_probe_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    gpu::GpuSpan root_span = root_owned.span;
    ShadowProbeRoot* root = (ShadowProbeRoot*)test_span_mapping(&device, root_span).ptr;
    root.out_gpu        = test_span_address(&device, out_span);
    root.depth_texture  = tex_index;
    root.shadow_sampler = shadow_smp;
    root.u = 0.5f;
    root.v = 0.5f;
    root.ref_lo = NEAR_Z - 0.1f;
    root.ref_hi = NEAR_Z + 0.1f;
    gpu::flush_mapped_span(&device, root_span)!!;

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_depth_to_attach(&cmd, depth);
    gpu::DepthTargetDesc depth_target = {
        .view = depth_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::GraphicsState graphics_state =
        gpu::render_geometry_state(DP_RT, DP_RT)!!;
    graphics_state.depth = tested;
    gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
    gpu::cmd_begin_render_pass_with_state(&cmd, &pass, &graphics_state)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, near_verts),
        fragment_root:  test_span_address(&device, color),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::TextureBarrier to_read = {
        .texture = depth,
        .before  = {
            .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
            .stages = { .depth_output },
            .access = { .read, .write },
        },
        .after   = gpu::sampled_at({ .compute }),
    };
    gpu::cmd_texture_barrier(&cmd, &to_read)!!;
    bind_test_compute_pipeline(&cmd, probe_pipeline);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     test_span_address(&device, root_span),
        groups:   { 1, 1, 1 },
    )!!;
    gpu::Barrier host_read = {
        .before = { .compute = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&cmd, &host_read)!!;
    submit_and_finish(&device, &cmd);
    gpu::invalidate_mapped_span(&device, out_span)!!;

    float* results = (float*)test_span_mapping(&device, out_span).ptr;
    assert(results[0] == 1.0f, "reference below stored depth passes LESS compare, got %f", results[0]);
    assert(results[1] == 0.0f, "reference above stored depth fails LESS compare, got %f", results[1]);
}

fn void depth_bias_shifts_stored_depth() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx();

    gpu::TextureHandle plain_depth = make_depth_texture(&device);
    defer gpu::destroy_texture(&device, plain_depth)!!;
    gpu::AttachmentViewHandle plain_view = create_test_attachment_view(
        &device,
        plain_depth,
    )!!;
    defer destroy_test_attachment_view(&device, &plain_view)!!;
    gpu::TextureHandle biased_depth = make_depth_texture(&device);
    defer gpu::destroy_texture(&device, biased_depth)!!;
    gpu::AttachmentViewHandle biased_view = create_test_attachment_view(
        &device,
        biased_depth,
    )!!;
    defer destroy_test_attachment_view(&device, &biased_view)!!;

    gpu::PipelineHandle pipeline = make_depth_pipeline(
        device:     &device,
        ctx:        &ctx,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    OwnedSpan readback = make_readback(&device, DP_DEPTH_BYTES * 2);
    defer free_owned_span(&device, &readback)!!;
    gpu::GpuSpan rb = readback.span;
    gpu::GpuSpan rb_plain = rb.checked_subspan(0, DP_DEPTH_BYTES)!!;
    gpu::GpuSpan rb_biased = rb.checked_subspan(DP_DEPTH_BYTES, DP_DEPTH_BYTES)!!;
    OwnedSpan verts_owned = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &verts_owned)!!;
    gpu::GpuSpan verts = verts_owned.span;
    OwnedSpan color_owned = write_color(&device, { 1.0f, 1.0f, 1.0f, 1.0f });
    defer free_owned_span(&device, &color_owned)!!;
    gpu::GpuSpan color = color_owned.span;

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::TextureHandle[2] targets = { plain_depth, biased_depth };
    gpu::AttachmentViewHandle[2] target_views = {
        plain_view,
        biased_view,
    };
    gpu::GpuSpan*[2] readbacks = { &rb_plain, &rb_biased };
    for (uint i = 0; i < 2; i++) {
        record_depth_to_attach(&cmd, targets[i]);
        gpu::DepthTargetDesc depth_target = {
            .view = target_views[i], .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
            .clear = { .depth = 1.0f },
        };
        gpu::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
        gpu::GraphicsState graphics_state =
            gpu::render_geometry_state(DP_RT, DP_RT)!!;
        graphics_state.depth = tested;
        if (i == 1) {
            graphics_state.raster = {
                .depth_bias_enable   = true,
                .depth_bias_constant = 4096.0f,
            };
        }
        gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
        gpu::cmd_begin_render_pass_with_state(&cmd, &pass, &graphics_state)!!;
        gpu::cmd_draw(
            commands:       &cmd,
            vertex_root:    test_span_address(&device, verts),
            fragment_root:  test_span_address(&device, color),
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::cmd_end_render_pass(&cmd)!!;
        record_depth_readback(&cmd, targets[i], readbacks[i]);
    }
    submit_and_finish(&device, &cmd);

    gpu::invalidate_mapped_span(&device, readback.span)!!;
    uint center = (DP_RT / 2 * DP_RT + DP_RT / 2);
    float plain_z = ((float*)test_span_mapping(&device, rb_plain).ptr)[center];
    float biased_z = ((float*)test_span_mapping(&device, rb_biased).ptr)[center];
    assert(plain_z == NEAR_Z, "unbiased depth stores the rendered z, got %f", plain_z);
    assert(biased_z > plain_z, "biased depth is pushed further, got %f vs %f", biased_z, plain_z);
}
