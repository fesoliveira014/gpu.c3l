module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;

// Headless (lavapipe): D32_FLOAT depth attachments. Depth-tested overlap in
// both draw orders, write/test toggles through dynamic state (one cache entry
// per immutable state), depth-only passes, sampled depth through the heap,
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
    gpu::ShaderHandle vert;
    gpu::ShaderHandle frag;
}

fn gpu::Device make_depth_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_depth_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
}

fn DepthCtx make_depth_ctx(gpu::Device* device) {
    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = DEPTH_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = DEPTH_FRAG_SPIRV[..], .entry_point = "main" };
    return {
        .vert = gpu::create_shader(device, &vert_sd)!!,
        .frag = gpu::create_shader(device, &frag_sd)!!,
    };
}

fn void destroy_depth_ctx(gpu::Device* device, DepthCtx* ctx) {
    gpu::destroy_shader(device, ctx.frag)!!;
    gpu::destroy_shader(device, ctx.vert)!!;
}

fn gpu::PipelineHandle make_depth_pipeline(gpu::Device* device, DepthCtx* ctx, gpu::DepthState depth, bool with_color = true) {
    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .depth           = depth,
        .color_formats   = with_color ? formats[..] : (gpu::Format[]){},
        .depth_format    = gpu::Format.D32_FLOAT,
    };
    return gpu::create_graphics_pipeline(device, &desc)!!;
}

fn gpu::TextureHandle make_depth_texture(gpu::Device* device, bool sampled = false) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
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
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = DP_RT,
        .height    = DP_RT,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::GpuSpan write_tri(gpu::Device* device, gpu::FrameToken* frame, float z) {
    gpu::GpuSpan span = gpu::alloc_frame_span(frame, 3 * 16, 16)!!;
    float* verts = (float*)test_span_mapping(device, span).ptr;
    float[12] data = {
        -0.8f, -0.8f, z, 0.0f,
         0.8f, -0.8f, z, 0.0f,
         0.0f,  0.8f, z, 0.0f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = data[i];
    return span;
}

fn gpu::GpuSpan write_color(gpu::Device* device, gpu::FrameToken* frame, float[4] rgba) {
    gpu::GpuSpan span = gpu::alloc_frame_span(frame, 16, 16)!!;
    float* color = (float*)test_span_mapping(device, span).ptr;
    for (uint i = 0; i < 4; i++) color[i] = rgba[i];
    return span;
}

fn void record_depth_to_attach(gpu::CommandList* cmd, gpu::TextureHandle depth) {
    gpu::TextureBarrier to_attach = gpu::texture_transition(
        texture: depth,
        before:  gpu::TextureUse.UNDEFINED,
        after:   gpu::TextureUse.DEPTH_ATTACHMENT,
    )!!;
    gpu::cmd_texture_barrier(cmd, &to_attach)!!;
}

fn void record_color_to_attach(gpu::CommandList* cmd, gpu::TextureHandle target) {
    gpu::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(cmd, &to_attach)!!;
}

fn void record_color_readback(gpu::CommandList* cmd, gpu::TextureHandle target, gpu::BufferHandle readback, gpu::GpuSpan* rb_span) {
    gpu::TextureBarrier to_src = {
        .texture       = target,
        .before_stage  = gpu::Stage.COLOR_ATTACHMENT,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.COLOR_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(cmd, &to_src)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = *rb_span };
    gpu::cmd_copy_texture_to_buffer(cmd, &copy)!!;
    gpu::BufferBarrier host_read = {
        .span = *rb_span,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(cmd, &host_read)!!;
}

fn void record_depth_readback(gpu::CommandList* cmd, gpu::TextureHandle depth, gpu::BufferHandle readback, gpu::GpuSpan* rb_span) {
    gpu::TextureBarrier to_src = {
        .texture       = depth,
        .before_stage  = gpu::Stage.DEPTH_STENCIL,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.DEPTH_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = gpu::TextureLayout.DEPTH_STENCIL,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(cmd, &to_src)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = depth, .dst = *rb_span };
    gpu::cmd_copy_texture_to_buffer(cmd, &copy)!!;
    gpu::BufferBarrier host_read = {
        .span = *rb_span,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(cmd, &host_read)!!;
}

fn gpu::BufferHandle make_readback(gpu::Device* device, usz size) {
    gpu::BufferDesc desc = {
        .size = size, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK, .access = { .graphics, .compute, .transfer },
    };
    return gpu::create_buffer(device, &desc)!!;
}

fn void submit_and_finish(gpu::Device* device, gpu::CommandList* commands) {
    gpu::ExecutableCommandList[1] lists = { gpu::end_commands(commands)!! };
    defer discard_test_executable(&lists[0])!!;
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(device, gpu::QueueKind.GRAPHICS)!!;
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
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);

    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle pipeline = make_depth_pipeline(&device, &ctx, tested);
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    bool[2] near_first_options = { true, false };
    foreach (near_first : near_first_options) {
        gpu::TextureHandle target = make_color_target(&device);
        defer gpu::destroy_texture(&device, target)!!;
        gpu::TextureHandle depth = make_depth_texture(&device);
        defer gpu::destroy_texture(&device, depth)!!;
        gpu::BufferHandle color_rb = make_readback(&device, DP_COLOR_BYTES);
        defer gpu::destroy_buffer(&device, color_rb)!!;
        gpu::BufferHandle depth_rb = make_readback(&device, DP_DEPTH_BYTES);
        defer gpu::destroy_buffer(&device, depth_rb)!!;
        gpu::GpuSpan color_rb_span = gpu::get_buffer_span(&device, color_rb)!!;
        gpu::GpuSpan depth_rb_span = gpu::get_buffer_span(&device, depth_rb)!!;

        gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
        gpu::GpuSpan near_verts = write_tri(&device, &frame_token, NEAR_Z);
        gpu::GpuSpan far_verts = write_tri(&device, &frame_token, FAR_Z);
        gpu::GpuSpan red = write_color(&device, &frame_token, { 1.0f, 0.0f, 0.0f, 1.0f });
        gpu::GpuSpan green = write_color(&device, &frame_token, { 0.0f, 1.0f, 0.0f, 1.0f });

        gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
        defer discard_test_commands(&cmd)!!;
        record_color_to_attach(&cmd, target);
        record_depth_to_attach(&cmd, depth);

        gpu::ColorTargetDesc[1] colors = {{
            .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
            .clear = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
        }};
        gpu::DepthTargetDesc depth_target = {
            .texture = depth, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
            .clear = { .depth = 1.0f },
        };
        gpu::RenderPassDesc pass = { .colors = colors[..], .depth = &depth_target, .width = DP_RT, .height = DP_RT };
        gpu::cmd_begin_render_pass(&cmd, &pass)!!;

        gpu::GpuSpan first_verts = near_first ? near_verts : far_verts;
        gpu::GpuSpan first_color = near_first ? red : green;
        gpu::GpuSpan second_verts = near_first ? far_verts : near_verts;
        gpu::GpuSpan second_color = near_first ? green : red;
        gpu::cmd_draw(
            commands:       &cmd,
            pipeline:       pipeline,
            vertex_root:    test_span_address(&device, first_verts),
            fragment_root:  test_span_address(&device, first_color),
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::cmd_draw(
            commands:       &cmd,
            pipeline:       pipeline,
            vertex_root:    test_span_address(&device, second_verts),
            fragment_root:  test_span_address(&device, second_color),
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::cmd_end_render_pass(&cmd)!!;

        record_color_readback(&cmd, target, color_rb, &color_rb_span);
        record_depth_readback(&cmd, depth, depth_rb, &depth_rb_span);
        submit_and_finish(&device, &cmd);

        gpu::invalidate_buffer(&device, color_rb, 0, 0)!!;
        char* center = dp_center(&device, &color_rb_span);
        assert(center[0] == (char)255 && center[1] == (char)0,
            "the near triangle wins regardless of draw order");

        gpu::invalidate_buffer(&device, depth_rb, 0, 0)!!;
        float* depth_center = dp_center_depth(&device, &depth_rb_span);
        assert(*depth_center == NEAR_Z, "depth buffer holds the near z");
        float* depth_corner = (float*)test_span_mapping(&device, depth_rb_span).ptr;
        assert(depth_corner[0] == 1.0f, "untouched pixels hold the clear depth");

        gpu::end_frame(&frame_token)!!;
    }
}

fn void depth_write_toggle_honored() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);
    gpu::TextureHandle target = make_color_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::TextureHandle depth = make_depth_texture(&device);
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::BufferHandle color_rb = make_readback(&device, DP_COLOR_BYTES);
    defer gpu::destroy_buffer(&device, color_rb)!!;
    gpu::BufferHandle depth_rb = make_readback(&device, DP_DEPTH_BYTES);
    defer gpu::destroy_buffer(&device, depth_rb)!!;
    gpu::GpuSpan color_rb_span = gpu::get_buffer_span(&device, color_rb)!!;
    gpu::GpuSpan depth_rb_span = gpu::get_buffer_span(&device, depth_rb)!!;

    gpu::DepthState no_write = { .test_enable = true, .write_enable = false, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle near_pipeline = make_depth_pipeline(&device, &ctx, no_write);
    defer gpu::destroy_pipeline(&device, near_pipeline)!!;
    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle far_pipeline = make_depth_pipeline(&device, &ctx, tested);
    defer gpu::destroy_pipeline(&device, far_pipeline)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::GpuSpan near_verts = write_tri(&device, &frame_token, NEAR_Z);
    gpu::GpuSpan far_verts = write_tri(&device, &frame_token, FAR_Z);
    gpu::GpuSpan red = write_color(&device, &frame_token, { 1.0f, 0.0f, 0.0f, 1.0f });
    gpu::GpuSpan green = write_color(&device, &frame_token, { 0.0f, 1.0f, 0.0f, 1.0f });

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_color_to_attach(&cmd, target);
    record_depth_to_attach(&cmd, depth);
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::DepthTargetDesc depth_target = {
        .texture = depth, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::RenderPassDesc pass = { .colors = colors[..], .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        pipeline:       near_pipeline,
        vertex_root:    test_span_address(&device, near_verts),
        fragment_root:  test_span_address(&device, red),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_draw(
        commands:       &cmd,
        pipeline:       far_pipeline,
        vertex_root:    test_span_address(&device, far_verts),
        fragment_root:  test_span_address(&device, green),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    record_color_readback(&cmd, target, color_rb, &color_rb_span);
    record_depth_readback(&cmd, depth, depth_rb, &depth_rb_span);
    submit_and_finish(&device, &cmd);

    gpu::invalidate_buffer(&device, color_rb, 0, 0)!!;
    char* center = dp_center(&device, &color_rb_span);
    assert(center[1] == (char)255, "far triangle renders because the near draw wrote no depth");
    gpu::invalidate_buffer(&device, depth_rb, 0, 0)!!;
    assert(*dp_center_depth(&device, &depth_rb_span) == FAR_Z, "depth holds the far z, written by the tested draw");
    gpu::end_frame(&frame_token)!!;
}

fn void depth_state_variants_share_entry() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);
    gpu::TextureHandle target = make_color_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::TextureHandle depth = make_depth_texture(&device);
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::BufferHandle color_rb = make_readback(&device, DP_COLOR_BYTES);
    defer gpu::destroy_buffer(&device, color_rb)!!;
    gpu::GpuSpan color_rb_span = gpu::get_buffer_span(&device, color_rb)!!;

    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle test_on = make_depth_pipeline(&device, &ctx, tested);
    defer gpu::destroy_pipeline(&device, test_on)!!;
    gpu::DepthState untested = {};
    gpu::PipelineHandle test_off = make_depth_pipeline(&device, &ctx, untested);
    defer gpu::destroy_pipeline(&device, test_off)!!;

    uint entries = vk::pipeline_cache_entry_count((vk::VkDeviceState*)gpu::device_backend_state(device));
    assert(entries == 1, "depth-state variants share one cache entry");

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::GpuSpan near_verts = write_tri(&device, &frame_token, NEAR_Z);
    gpu::GpuSpan far_verts = write_tri(&device, &frame_token, FAR_Z);
    gpu::GpuSpan red = write_color(&device, &frame_token, { 1.0f, 0.0f, 0.0f, 1.0f });
    gpu::GpuSpan green = write_color(&device, &frame_token, { 0.0f, 1.0f, 0.0f, 1.0f });

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_color_to_attach(&cmd, target);
    record_depth_to_attach(&cmd, depth);
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::DepthTargetDesc depth_target = {
        .texture = depth, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::RenderPassDesc pass = { .colors = colors[..], .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        pipeline:       test_on,
        vertex_root:    test_span_address(&device, near_verts),
        fragment_root:  test_span_address(&device, red),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_draw(
        commands:       &cmd,
        pipeline:       test_off,
        vertex_root:    test_span_address(&device, far_verts),
        fragment_root:  test_span_address(&device, green),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    record_color_readback(&cmd, target, color_rb, &color_rb_span);
    submit_and_finish(&device, &cmd);

    gpu::invalidate_buffer(&device, color_rb, 0, 0)!!;
    char* center = dp_center(&device, &color_rb_span);
    assert(center[1] == (char)255, "test-off alias of the same entry overwrites the near triangle");
    gpu::end_frame(&frame_token)!!;
}

fn void depth_only_pass_renders_footprint() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);
    gpu::TextureHandle depth = make_depth_texture(&device);
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::BufferHandle depth_rb = make_readback(&device, DP_DEPTH_BYTES);
    defer gpu::destroy_buffer(&device, depth_rb)!!;
    gpu::GpuSpan depth_rb_span = gpu::get_buffer_span(&device, depth_rb)!!;

    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle pipeline = make_depth_pipeline(
        device:     &device,
        ctx:        &ctx,
        depth:      tested,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::GpuSpan near_verts = write_tri(&device, &frame_token, NEAR_Z);
    gpu::GpuSpan color = write_color(&device, &frame_token, { 1.0f, 1.0f, 1.0f, 1.0f });

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_depth_to_attach(&cmd, depth);
    gpu::DepthTargetDesc depth_target = {
        .texture = depth, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        pipeline:       pipeline,
        vertex_root:    test_span_address(&device, near_verts),
        fragment_root:  test_span_address(&device, color),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    record_depth_readback(&cmd, depth, depth_rb, &depth_rb_span);
    submit_and_finish(&device, &cmd);

    gpu::invalidate_buffer(&device, depth_rb, 0, 0)!!;
    assert(*dp_center_depth(&device, &depth_rb_span) == NEAR_Z, "depth-only pass writes the triangle footprint");
    float* corner = (float*)test_span_mapping(&device, depth_rb_span).ptr;
    assert(corner[0] == 1.0f, "outside the footprint the clear depth remains");
    gpu::end_frame(&frame_token)!!;
}

fn void sampled_depth_matches_readback() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);
    gpu::TextureHandle depth = make_depth_texture(&device, sampled: true);
    defer gpu::destroy_texture(&device, depth)!!;

    gpu::TextureIndex tex_index = gpu::create_texture_descriptor(&device, depth, null)!!;
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

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DEPTH_SAMPLE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle comp = gpu::create_shader(&device, &comp_sd)!!;
    defer gpu::destroy_shader(&device, comp)!!;
    gpu::ComputePipelineDesc comp_desc = { .shader = comp, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle sample_pipeline = gpu::create_compute_pipeline(&device, &comp_desc)!!;
    defer gpu::destroy_pipeline(&device, sample_pipeline)!!;

    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle pipeline = make_depth_pipeline(
        device:     &device,
        ctx:        &ctx,
        depth:      tested,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::GpuSpan near_verts = write_tri(&device, &frame_token, NEAR_Z);
    gpu::GpuSpan color = write_color(&device, &frame_token, { 1.0f, 1.0f, 1.0f, 1.0f });
    gpu::GpuSpan out_span = gpu::alloc_frame_span(&frame_token, 4, 4)!!;
    vk::ResolvedSpan out_resolved = vk::resolve_span(
        (vk::VkDeviceState*)gpu::device_backend_state(device),
        &out_span,
    )!!;
    gpu::GpuSpan root_span = gpu::alloc_frame_span(&frame_token, SampleDepthRoot::size, 16)!!;
    SampleDepthRoot* root = (SampleDepthRoot*)test_span_mapping(&device, root_span).ptr;
    root.out_gpu = test_span_address(&device, out_span);
    root.texture_index = tex_index;
    root.sampler_index = smp_index;
    root.u = 0.5f;
    root.v = 0.5f;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_depth_to_attach(&cmd, depth);
    gpu::DepthTargetDesc depth_target = {
        .texture = depth, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        pipeline:       pipeline,
        vertex_root:    test_span_address(&device, near_verts),
        fragment_root:  test_span_address(&device, color),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::TextureBarrier to_read = {
        .texture       = depth,
        .before_stage  = gpu::Stage.DEPTH_STENCIL,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.DEPTH_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
        .old_layout    = gpu::TextureLayout.DEPTH_STENCIL,
        .new_layout    = gpu::TextureLayout.SHADER_READ,
    };
    gpu::cmd_texture_barrier(&cmd, &to_read)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: sample_pipeline,
        root:     test_span_address(&device, root_span),
        groups:   { 1, 1, 1 },
    )!!;
    gpu::BufferBarrier host_read = {
        .span = out_span,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!!;
    submit_and_finish(&device, &cmd);

    gpu::invalidate_buffer(&device, out_resolved.backing_buffer, 0, 0)!!;
    float* sampled = (float*)test_span_mapping(&device, out_span).ptr;
    assert(*sampled == NEAR_Z, "heap-sampled depth matches the rendered z");
    gpu::end_frame(&frame_token)!!;
}

fn void depth_pass_validation_faults() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc sampled_only_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = DP_RT,
        .height    = DP_RT,
        .format    = gpu::Format.D32_FLOAT,
        .usage     = { .sampled },
    };
    gpu::TextureHandle sampled_only = gpu::create_texture(&device, &sampled_only_desc)!!;
    defer gpu::destroy_texture(&device, sampled_only)!!;

    gpu::TextureHandle unbarriered = make_depth_texture(&device);
    defer gpu::destroy_texture(&device, unbarriered)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::DepthTargetDesc no_usage = { .texture = sampled_only, .load_op = gpu::LoadOp.CLEAR, .clear = { .depth = 1.0f } };
    gpu::RenderPassDesc no_usage_pass = { .depth = &no_usage, .width = DP_RT, .height = DP_RT };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &no_usage_pass)) == gpu::INVALID_ARGUMENT,
        "depth target without depth_attach usage faults INVALID_ARGUMENT");

    gpu::DepthTargetDesc wrong_layout = { .texture = unbarriered, .load_op = gpu::LoadOp.CLEAR, .clear = { .depth = 1.0f } };
    gpu::RenderPassDesc wrong_layout_pass = { .depth = &wrong_layout, .width = DP_RT, .height = DP_RT };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &wrong_layout_pass)) == gpu::INVALID_RESOURCE_STATE,
        "depth target outside the DEPTH_STENCIL layout faults INVALID_RESOURCE_STATE");

    gpu::RenderPassDesc empty_pass = { .width = DP_RT, .height = DP_RT };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &empty_pass)) == gpu::INVALID_ARGUMENT,
        "a pass with no attachments faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
    gpu::end_frame(&frame_token)!!;
}

const char[*] SHADOW_PROBE_SPIRV = $embed("shaders/shadow_probe.comp.spv");

fn void compare_sampler_exact_results() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);
    gpu::TextureHandle depth = make_depth_texture(&device, sampled: true);
    defer gpu::destroy_texture(&device, depth)!!;

    gpu::TextureIndex tex_index = gpu::create_texture_descriptor(&device, depth, null)!!;
    defer gpu::destroy_texture_descriptor(&device, tex_index)!!;
    gpu::SamplerDesc shadow_desc = {
        .min_filter     = gpu::Filter.NEAREST,
        .mag_filter     = gpu::Filter.NEAREST,
        .address_u      = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v      = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w      = gpu::AddressMode.CLAMP_TO_EDGE,
        .compare_enable = true,
        .compare        = gpu::CompareOp.LESS,
    };
    gpu::SamplerIndex shadow_smp = gpu::create_sampler(&device, &shadow_desc)!!;
    defer gpu::destroy_sampler(&device, shadow_smp)!!;

    gpu::ShaderDesc probe_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = SHADOW_PROBE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle probe = gpu::create_shader(&device, &probe_sd)!!;
    defer gpu::destroy_shader(&device, probe)!!;
    gpu::ComputePipelineDesc probe_desc = { .shader = probe, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle probe_pipeline = gpu::create_compute_pipeline(&device, &probe_desc)!!;
    defer gpu::destroy_pipeline(&device, probe_pipeline)!!;

    gpu::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle pipeline = make_depth_pipeline(
        device:     &device,
        ctx:        &ctx,
        depth:      tested,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::GpuSpan near_verts = write_tri(&device, &frame_token, NEAR_Z);
    gpu::GpuSpan color = write_color(&device, &frame_token, { 1.0f, 1.0f, 1.0f, 1.0f });
    gpu::GpuSpan out_span = gpu::alloc_frame_span(&frame_token, 8, 8)!!;
    vk::ResolvedSpan out_resolved = vk::resolve_span(
        (vk::VkDeviceState*)gpu::device_backend_state(device),
        &out_span,
    )!!;
    gpu::GpuSpan root_span = gpu::alloc_frame_span(&frame_token, ShadowProbeRoot::size, 16)!!;
    ShadowProbeRoot* root = (ShadowProbeRoot*)test_span_mapping(&device, root_span).ptr;
    root.out_gpu        = test_span_address(&device, out_span);
    root.depth_texture  = tex_index;
    root.shadow_sampler = shadow_smp;
    root.u = 0.5f;
    root.v = 0.5f;
    root.ref_lo = NEAR_Z - 0.1f;
    root.ref_hi = NEAR_Z + 0.1f;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_depth_to_attach(&cmd, depth);
    gpu::DepthTargetDesc depth_target = {
        .texture = depth, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        pipeline:       pipeline,
        vertex_root:    test_span_address(&device, near_verts),
        fragment_root:  test_span_address(&device, color),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::TextureBarrier to_read = {
        .texture       = depth,
        .before_stage  = gpu::Stage.DEPTH_STENCIL,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.DEPTH_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
        .old_layout    = gpu::TextureLayout.DEPTH_STENCIL,
        .new_layout    = gpu::TextureLayout.SHADER_READ,
    };
    gpu::cmd_texture_barrier(&cmd, &to_read)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: probe_pipeline,
        root:     test_span_address(&device, root_span),
        groups:   { 1, 1, 1 },
    )!!;
    gpu::BufferBarrier host_read = {
        .span = out_span,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!!;
    submit_and_finish(&device, &cmd);

    gpu::invalidate_buffer(&device, out_resolved.backing_buffer, 0, 0)!!;
    float* results = (float*)test_span_mapping(&device, out_span).ptr;
    assert(results[0] == 1.0f, "reference below stored depth passes LESS compare, got %f", results[0]);
    assert(results[1] == 0.0f, "reference above stored depth fails LESS compare, got %f", results[1]);
    gpu::end_frame(&frame_token)!!;
}

fn gpu::PipelineHandle make_biased_pipeline(gpu::Device* device, DepthCtx* ctx, float bias_constant) {
    gpu::GraphicsPipelineDesc desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .depth           = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS },
        .raster          = { .depth_bias_constant = bias_constant },
        .depth_format    = gpu::Format.D32_FLOAT,
    };
    return gpu::create_graphics_pipeline(device, &desc)!!;
}

fn void depth_bias_shifts_stored_depth() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);

    gpu::TextureHandle plain_depth = make_depth_texture(&device);
    defer gpu::destroy_texture(&device, plain_depth)!!;
    gpu::TextureHandle biased_depth = make_depth_texture(&device);
    defer gpu::destroy_texture(&device, biased_depth)!!;

    gpu::PipelineHandle plain = make_biased_pipeline(&device, &ctx, 0.0f);
    defer gpu::destroy_pipeline(&device, plain)!!;
    gpu::PipelineHandle biased = make_biased_pipeline(&device, &ctx, 4096.0f);
    defer gpu::destroy_pipeline(&device, biased)!!;

    gpu::BufferHandle readback = make_readback(&device, DP_DEPTH_BYTES * 2);
    defer gpu::destroy_buffer(&device, readback)!!;
    gpu::GpuSpan rb = gpu::get_buffer_span(&device, readback)!!;
    gpu::GpuSpan rb_plain = rb.checked_subspan(0, DP_DEPTH_BYTES)!!;
    gpu::GpuSpan rb_biased = rb.checked_subspan(DP_DEPTH_BYTES, DP_DEPTH_BYTES)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::GpuSpan verts = write_tri(&device, &frame_token, NEAR_Z);
    gpu::GpuSpan color = write_color(&device, &frame_token, { 1.0f, 1.0f, 1.0f, 1.0f });

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureHandle[2] targets = { plain_depth, biased_depth };
    gpu::PipelineHandle[2] pipelines = { plain, biased };
    gpu::GpuSpan*[2] readbacks = { &rb_plain, &rb_biased };
    for (uint i = 0; i < 2; i++) {
        record_depth_to_attach(&cmd, targets[i]);
        gpu::DepthTargetDesc depth_target = {
            .texture = targets[i], .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
            .clear = { .depth = 1.0f },
        };
        gpu::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
        gpu::cmd_begin_render_pass(&cmd, &pass)!!;
        gpu::cmd_draw(
            commands:       &cmd,
            pipeline:       pipelines[i],
            vertex_root:    test_span_address(&device, verts),
            fragment_root:  test_span_address(&device, color),
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::cmd_end_render_pass(&cmd)!!;
        record_depth_readback(&cmd, targets[i], readback, readbacks[i]);
    }
    submit_and_finish(&device, &cmd);

    gpu::invalidate_buffer(&device, readback, 0, 0)!!;
    uint center = (DP_RT / 2 * DP_RT + DP_RT / 2);
    float plain_z = ((float*)test_span_mapping(&device, rb_plain).ptr)[center];
    float biased_z = ((float*)test_span_mapping(&device, rb_biased).ptr)[center];
    assert(plain_z == NEAR_Z, "unbiased depth stores the rendered z, got %f", plain_z);
    assert(biased_z > plain_z, "biased depth is pushed further, got %f vs %f", biased_z, plain_z);
    gpu::end_frame(&frame_token)!!;
}
