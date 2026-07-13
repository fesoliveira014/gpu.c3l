module gpu_test;

import gpu::compat @public;
import gpu::compat::vk @public;
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
    gpu::compat::ShaderHandle vert;
    gpu::compat::ShaderHandle frag;
}

fn gpu::compat::Device make_depth_device() {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_depth_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

fn DepthCtx make_depth_ctx(gpu::compat::Device* device) {
    gpu::compat::ShaderDesc vert_sd = { .stage = gpu::compat::ShaderStage.VERTEX, .spirv = DEPTH_VERT_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderDesc frag_sd = { .stage = gpu::compat::ShaderStage.FRAGMENT, .spirv = DEPTH_FRAG_SPIRV[..], .entry_point = "main" };
    return {
        .vert = gpu::compat::create_shader(device, &vert_sd)!!,
        .frag = gpu::compat::create_shader(device, &frag_sd)!!,
    };
}

fn void destroy_depth_ctx(gpu::compat::Device* device, DepthCtx* ctx) {
    gpu::compat::destroy_shader(device, ctx.frag)!!;
    gpu::compat::destroy_shader(device, ctx.vert)!!;
}

fn gpu::compat::PipelineHandle make_depth_pipeline(gpu::compat::Device* device, DepthCtx* ctx, gpu::compat::DepthState depth, bool with_color = true) {
    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .topology        = gpu::compat::PrimitiveTopology.TRIANGLES,
        .depth           = depth,
        .color_formats   = with_color ? formats[..] : (gpu::compat::Format[]){},
        .depth_format    = gpu::compat::Format.D32_FLOAT,
    };
    return gpu::compat::create_graphics_pipeline(device, &desc)!!;
}

fn gpu::compat::TextureHandle make_depth_texture(gpu::compat::Device* device, bool sampled = false) {
    gpu::compat::TextureDesc desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = DP_RT,
        .height    = DP_RT,
        .format    = gpu::compat::Format.D32_FLOAT,
        .usage     = { .depth_attach, .transfer_src, .sampled = sampled },
    };
    return gpu::compat::create_texture(device, &desc)!!;
}

fn gpu::compat::TextureHandle make_color_target(gpu::compat::Device* device) {
    gpu::compat::TextureDesc desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = DP_RT,
        .height    = DP_RT,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    return gpu::compat::create_texture(device, &desc)!!;
}

fn gpu::compat::GpuSpan write_tri(gpu::compat::FrameToken* frame, float z) {
    gpu::compat::GpuSpan span = gpu::compat::alloc_frame_span(frame, 3 * 16, 16)!!;
    float* verts = (float*)span.cpu;
    float[12] data = {
        -0.8f, -0.8f, z, 0.0f,
         0.8f, -0.8f, z, 0.0f,
         0.0f,  0.8f, z, 0.0f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = data[i];
    return span;
}

fn gpu::compat::GpuSpan write_color(gpu::compat::FrameToken* frame, float[4] rgba) {
    gpu::compat::GpuSpan span = gpu::compat::alloc_frame_span(frame, 16, 16)!!;
    float* color = (float*)span.cpu;
    for (uint i = 0; i < 4; i++) color[i] = rgba[i];
    return span;
}

fn void record_depth_to_attach(gpu::compat::CommandList* cmd, gpu::compat::TextureHandle depth) {
    gpu::compat::TextureBarrier to_attach = gpu::compat::texture_transition(
        texture: depth,
        before:  gpu::compat::TextureUse.UNDEFINED,
        after:   gpu::compat::TextureUse.DEPTH_ATTACHMENT,
    )!!;
    gpu::compat::cmd_texture_barrier(cmd, &to_attach)!!;
}

fn void record_color_to_attach(gpu::compat::CommandList* cmd, gpu::compat::TextureHandle target) {
    gpu::compat::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::compat::Stage.HOST,
        .after_stage   = gpu::compat::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::compat::Hazard.HOST_WRITE,
        .after_hazard  = gpu::compat::Hazard.COLOR_WRITE,
        .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
        .new_layout    = gpu::compat::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::compat::cmd_texture_barrier(cmd, &to_attach)!!;
}

fn void record_color_readback(gpu::compat::CommandList* cmd, gpu::compat::TextureHandle target, gpu::compat::GpuSpan* rb_span) {
    gpu::compat::TextureBarrier to_src = {
        .texture       = target,
        .before_stage  = gpu::compat::Stage.COLOR_ATTACHMENT,
        .after_stage   = gpu::compat::Stage.TRANSFER,
        .before_hazard = gpu::compat::Hazard.COLOR_WRITE,
        .after_hazard  = gpu::compat::Hazard.TRANSFER_READ,
        .old_layout    = gpu::compat::TextureLayout.COLOR_ATTACHMENT,
        .new_layout    = gpu::compat::TextureLayout.TRANSFER_SRC,
    };
    gpu::compat::cmd_texture_barrier(cmd, &to_src)!!;
    gpu::compat::TextureBufferCopyDesc copy = { .texture = target, .dst = *rb_span };
    gpu::compat::cmd_copy_texture_to_buffer(cmd, &copy)!!;
    gpu::compat::BufferBarrier host_read = {
        .buffer = rb_span.buffer, .offset = rb_span.offset, .size = rb_span.size,
        .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(cmd, &host_read)!!;
}

fn void record_depth_readback(gpu::compat::CommandList* cmd, gpu::compat::TextureHandle depth, gpu::compat::GpuSpan* rb_span) {
    gpu::compat::TextureBarrier to_src = {
        .texture       = depth,
        .before_stage  = gpu::compat::Stage.DEPTH_STENCIL,
        .after_stage   = gpu::compat::Stage.TRANSFER,
        .before_hazard = gpu::compat::Hazard.DEPTH_WRITE,
        .after_hazard  = gpu::compat::Hazard.TRANSFER_READ,
        .old_layout    = gpu::compat::TextureLayout.DEPTH_STENCIL,
        .new_layout    = gpu::compat::TextureLayout.TRANSFER_SRC,
    };
    gpu::compat::cmd_texture_barrier(cmd, &to_src)!!;
    gpu::compat::TextureBufferCopyDesc copy = { .texture = depth, .dst = *rb_span };
    gpu::compat::cmd_copy_texture_to_buffer(cmd, &copy)!!;
    gpu::compat::BufferBarrier host_read = {
        .buffer = rb_span.buffer, .offset = rb_span.offset, .size = rb_span.size,
        .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(cmd, &host_read)!!;
}

fn gpu::compat::BufferHandle make_readback(gpu::compat::Device* device, usz size) {
    gpu::compat::BufferDesc desc = {
        .size = size, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.READBACK,
    };
    return gpu::compat::create_buffer(device, &desc)!!;
}

fn void submit_and_finish(gpu::compat::Device* device, gpu::compat::CommandList* cmd) {
    gpu::compat::end_commands(cmd)!!;
    gpu::compat::CommandList[1] lists = { *cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(device, &submit)!!;
    gpu::compat::wait_queue_idle(device, gpu::compat::QueueKind.GRAPHICS)!!;
}

fn char* dp_center(gpu::compat::GpuSpan* rb_span) {
    char* pixels = (char*)rb_span.cpu;
    return &pixels[(DP_RT / 2 * DP_RT + DP_RT / 2) * 4];
}

fn float* dp_center_depth(gpu::compat::GpuSpan* rb_span) {
    float* texels = (float*)rb_span.cpu;
    return &texels[DP_RT / 2 * DP_RT + DP_RT / 2];
}

fn void depth_test_occludes_both_orders() @test {
    gpu::compat::Device device = make_depth_device();
    defer gpu::compat::destroy_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);

    gpu::compat::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::compat::CompareOp.LESS };
    gpu::compat::PipelineHandle pipeline = make_depth_pipeline(&device, &ctx, tested);
    defer gpu::compat::destroy_pipeline(&device, pipeline)!!;

    bool[2] near_first_options = { true, false };
    foreach (near_first : near_first_options) {
        gpu::compat::TextureHandle target = make_color_target(&device);
        defer gpu::compat::destroy_texture(&device, target)!!;
        gpu::compat::TextureHandle depth = make_depth_texture(&device);
        defer gpu::compat::destroy_texture(&device, depth)!!;
        gpu::compat::BufferHandle color_rb = make_readback(&device, DP_COLOR_BYTES);
        defer gpu::compat::destroy_buffer(&device, color_rb)!!;
        gpu::compat::BufferHandle depth_rb = make_readback(&device, DP_DEPTH_BYTES);
        defer gpu::compat::destroy_buffer(&device, depth_rb)!!;
        gpu::compat::GpuSpan color_rb_span = gpu::compat::get_buffer_span(&device, color_rb)!!;
        gpu::compat::GpuSpan depth_rb_span = gpu::compat::get_buffer_span(&device, depth_rb)!!;

        gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
        gpu::compat::GpuSpan near_verts = write_tri(&frame_token, NEAR_Z);
        gpu::compat::GpuSpan far_verts = write_tri(&frame_token, FAR_Z);
        gpu::compat::GpuSpan red = write_color(&frame_token, { 1.0f, 0.0f, 0.0f, 1.0f });
        gpu::compat::GpuSpan green = write_color(&frame_token, { 0.0f, 1.0f, 0.0f, 1.0f });

        gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
        record_color_to_attach(&cmd, target);
        record_depth_to_attach(&cmd, depth);

        gpu::compat::ColorTargetDesc[1] colors = {{
            .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
            .clear = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
        }};
        gpu::compat::DepthTargetDesc depth_target = {
            .texture = depth, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
            .clear = { .depth = 1.0f },
        };
        gpu::compat::RenderPassDesc pass = { .colors = colors[..], .depth = &depth_target, .width = DP_RT, .height = DP_RT };
        gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;

        gpu::compat::GpuSpan first_verts = near_first ? near_verts : far_verts;
        gpu::compat::GpuSpan first_color = near_first ? red : green;
        gpu::compat::GpuSpan second_verts = near_first ? far_verts : near_verts;
        gpu::compat::GpuSpan second_color = near_first ? green : red;
        gpu::compat::cmd_draw(
            commands:       &cmd,
            pipeline:       pipeline,
            vertex_root:    first_verts.gpu,
            fragment_root:  first_color.gpu,
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::compat::cmd_draw(
            commands:       &cmd,
            pipeline:       pipeline,
            vertex_root:    second_verts.gpu,
            fragment_root:  second_color.gpu,
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::compat::cmd_end_render_pass(&cmd)!!;

        record_color_readback(&cmd, target, &color_rb_span);
        record_depth_readback(&cmd, depth, &depth_rb_span);
        submit_and_finish(&device, &cmd);

        gpu::compat::invalidate_buffer(&device, color_rb, 0, 0)!!;
        char* center = dp_center(&color_rb_span);
        assert(center[0] == (char)255 && center[1] == (char)0,
            "the near triangle wins regardless of draw order");

        gpu::compat::invalidate_buffer(&device, depth_rb, 0, 0)!!;
        float* depth_center = dp_center_depth(&depth_rb_span);
        assert(*depth_center == NEAR_Z, "depth buffer holds the near z");
        float* depth_corner = (float*)depth_rb_span.cpu;
        assert(depth_corner[0] == 1.0f, "untouched pixels hold the clear depth");

        gpu::compat::end_frame(&frame_token)!!;
    }
}

fn void depth_write_toggle_honored() @test {
    gpu::compat::Device device = make_depth_device();
    defer gpu::compat::destroy_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);
    gpu::compat::TextureHandle target = make_color_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;
    gpu::compat::TextureHandle depth = make_depth_texture(&device);
    defer gpu::compat::destroy_texture(&device, depth)!!;
    gpu::compat::BufferHandle color_rb = make_readback(&device, DP_COLOR_BYTES);
    defer gpu::compat::destroy_buffer(&device, color_rb)!!;
    gpu::compat::BufferHandle depth_rb = make_readback(&device, DP_DEPTH_BYTES);
    defer gpu::compat::destroy_buffer(&device, depth_rb)!!;
    gpu::compat::GpuSpan color_rb_span = gpu::compat::get_buffer_span(&device, color_rb)!!;
    gpu::compat::GpuSpan depth_rb_span = gpu::compat::get_buffer_span(&device, depth_rb)!!;

    gpu::compat::DepthState no_write = { .test_enable = true, .write_enable = false, .compare = gpu::compat::CompareOp.LESS };
    gpu::compat::PipelineHandle near_pipeline = make_depth_pipeline(&device, &ctx, no_write);
    defer gpu::compat::destroy_pipeline(&device, near_pipeline)!!;
    gpu::compat::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::compat::CompareOp.LESS };
    gpu::compat::PipelineHandle far_pipeline = make_depth_pipeline(&device, &ctx, tested);
    defer gpu::compat::destroy_pipeline(&device, far_pipeline)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan near_verts = write_tri(&frame_token, NEAR_Z);
    gpu::compat::GpuSpan far_verts = write_tri(&frame_token, FAR_Z);
    gpu::compat::GpuSpan red = write_color(&frame_token, { 1.0f, 0.0f, 0.0f, 1.0f });
    gpu::compat::GpuSpan green = write_color(&frame_token, { 0.0f, 1.0f, 0.0f, 1.0f });

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_color_to_attach(&cmd, target);
    record_depth_to_attach(&cmd, depth);
    gpu::compat::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
    }};
    gpu::compat::DepthTargetDesc depth_target = {
        .texture = depth, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::compat::cmd_draw(
        commands:       &cmd,
        pipeline:       near_pipeline,
        vertex_root:    near_verts.gpu,
        fragment_root:  red.gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::compat::cmd_draw(
        commands:       &cmd,
        pipeline:       far_pipeline,
        vertex_root:    far_verts.gpu,
        fragment_root:  green.gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::compat::cmd_end_render_pass(&cmd)!!;
    record_color_readback(&cmd, target, &color_rb_span);
    record_depth_readback(&cmd, depth, &depth_rb_span);
    submit_and_finish(&device, &cmd);

    gpu::compat::invalidate_buffer(&device, color_rb, 0, 0)!!;
    char* center = dp_center(&color_rb_span);
    assert(center[1] == (char)255, "far triangle renders because the near draw wrote no depth");
    gpu::compat::invalidate_buffer(&device, depth_rb, 0, 0)!!;
    assert(*dp_center_depth(&depth_rb_span) == FAR_Z, "depth holds the far z, written by the tested draw");
    gpu::compat::end_frame(&frame_token)!!;
}

fn void depth_state_variants_share_entry() @test {
    gpu::compat::Device device = make_depth_device();
    defer gpu::compat::destroy_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);
    gpu::compat::TextureHandle target = make_color_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;
    gpu::compat::TextureHandle depth = make_depth_texture(&device);
    defer gpu::compat::destroy_texture(&device, depth)!!;
    gpu::compat::BufferHandle color_rb = make_readback(&device, DP_COLOR_BYTES);
    defer gpu::compat::destroy_buffer(&device, color_rb)!!;
    gpu::compat::GpuSpan color_rb_span = gpu::compat::get_buffer_span(&device, color_rb)!!;

    gpu::compat::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::compat::CompareOp.LESS };
    gpu::compat::PipelineHandle test_on = make_depth_pipeline(&device, &ctx, tested);
    defer gpu::compat::destroy_pipeline(&device, test_on)!!;
    gpu::compat::DepthState untested = {};
    gpu::compat::PipelineHandle test_off = make_depth_pipeline(&device, &ctx, untested);
    defer gpu::compat::destroy_pipeline(&device, test_off)!!;

    uint entries = vk::pipeline_cache_entry_count((vk::VkDeviceState*)gpu::compat::device_backend_state(device));
    assert(entries == 1, "depth-state variants share one cache entry");

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan near_verts = write_tri(&frame_token, NEAR_Z);
    gpu::compat::GpuSpan far_verts = write_tri(&frame_token, FAR_Z);
    gpu::compat::GpuSpan red = write_color(&frame_token, { 1.0f, 0.0f, 0.0f, 1.0f });
    gpu::compat::GpuSpan green = write_color(&frame_token, { 0.0f, 1.0f, 0.0f, 1.0f });

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_color_to_attach(&cmd, target);
    record_depth_to_attach(&cmd, depth);
    gpu::compat::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
    }};
    gpu::compat::DepthTargetDesc depth_target = {
        .texture = depth, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::compat::cmd_draw(
        commands:       &cmd,
        pipeline:       test_on,
        vertex_root:    near_verts.gpu,
        fragment_root:  red.gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::compat::cmd_draw(
        commands:       &cmd,
        pipeline:       test_off,
        vertex_root:    far_verts.gpu,
        fragment_root:  green.gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::compat::cmd_end_render_pass(&cmd)!!;
    record_color_readback(&cmd, target, &color_rb_span);
    submit_and_finish(&device, &cmd);

    gpu::compat::invalidate_buffer(&device, color_rb, 0, 0)!!;
    char* center = dp_center(&color_rb_span);
    assert(center[1] == (char)255, "test-off alias of the same entry overwrites the near triangle");
    gpu::compat::end_frame(&frame_token)!!;
}

fn void depth_only_pass_renders_footprint() @test {
    gpu::compat::Device device = make_depth_device();
    defer gpu::compat::destroy_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);
    gpu::compat::TextureHandle depth = make_depth_texture(&device);
    defer gpu::compat::destroy_texture(&device, depth)!!;
    gpu::compat::BufferHandle depth_rb = make_readback(&device, DP_DEPTH_BYTES);
    defer gpu::compat::destroy_buffer(&device, depth_rb)!!;
    gpu::compat::GpuSpan depth_rb_span = gpu::compat::get_buffer_span(&device, depth_rb)!!;

    gpu::compat::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::compat::CompareOp.LESS };
    gpu::compat::PipelineHandle pipeline = make_depth_pipeline(
        device:     &device,
        ctx:        &ctx,
        depth:      tested,
        with_color: false,
    );
    defer gpu::compat::destroy_pipeline(&device, pipeline)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan near_verts = write_tri(&frame_token, NEAR_Z);
    gpu::compat::GpuSpan color = write_color(&frame_token, { 1.0f, 1.0f, 1.0f, 1.0f });

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_depth_to_attach(&cmd, depth);
    gpu::compat::DepthTargetDesc depth_target = {
        .texture = depth, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::compat::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::compat::cmd_draw(
        commands:       &cmd,
        pipeline:       pipeline,
        vertex_root:    near_verts.gpu,
        fragment_root:  color.gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::compat::cmd_end_render_pass(&cmd)!!;
    record_depth_readback(&cmd, depth, &depth_rb_span);
    submit_and_finish(&device, &cmd);

    gpu::compat::invalidate_buffer(&device, depth_rb, 0, 0)!!;
    assert(*dp_center_depth(&depth_rb_span) == NEAR_Z, "depth-only pass writes the triangle footprint");
    float* corner = (float*)depth_rb_span.cpu;
    assert(corner[0] == 1.0f, "outside the footprint the clear depth remains");
    gpu::compat::end_frame(&frame_token)!!;
}

fn void sampled_depth_matches_readback() @test {
    gpu::compat::Device device = make_depth_device();
    defer gpu::compat::destroy_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);
    gpu::compat::TextureHandle depth = make_depth_texture(&device, sampled: true);
    defer gpu::compat::destroy_texture(&device, depth)!!;

    gpu::compat::TextureIndex tex_index = gpu::compat::create_texture_descriptor(&device, depth, null)!!;
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

    gpu::compat::ShaderDesc comp_sd = { .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = DEPTH_SAMPLE_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle comp = gpu::compat::create_shader(&device, &comp_sd)!!;
    defer gpu::compat::destroy_shader(&device, comp)!!;
    gpu::compat::ComputePipelineDesc comp_desc = { .shader = comp, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle sample_pipeline = gpu::compat::create_compute_pipeline(&device, &comp_desc)!!;
    defer gpu::compat::destroy_pipeline(&device, sample_pipeline)!!;

    gpu::compat::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::compat::CompareOp.LESS };
    gpu::compat::PipelineHandle pipeline = make_depth_pipeline(
        device:     &device,
        ctx:        &ctx,
        depth:      tested,
        with_color: false,
    );
    defer gpu::compat::destroy_pipeline(&device, pipeline)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan near_verts = write_tri(&frame_token, NEAR_Z);
    gpu::compat::GpuSpan color = write_color(&frame_token, { 1.0f, 1.0f, 1.0f, 1.0f });
    gpu::compat::GpuSpan out_span = gpu::compat::alloc_frame_span(&frame_token, 4, 4)!!;
    gpu::compat::GpuSpan root_span = gpu::compat::alloc_frame_span(&frame_token, SampleDepthRoot::size, 16)!!;
    SampleDepthRoot* root = (SampleDepthRoot*)root_span.cpu;
    root.out_gpu = out_span.gpu;
    root.texture_index = tex_index;
    root.sampler_index = smp_index;
    root.u = 0.5f;
    root.v = 0.5f;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_depth_to_attach(&cmd, depth);
    gpu::compat::DepthTargetDesc depth_target = {
        .texture = depth, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::compat::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::compat::cmd_draw(
        commands:       &cmd,
        pipeline:       pipeline,
        vertex_root:    near_verts.gpu,
        fragment_root:  color.gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::compat::cmd_end_render_pass(&cmd)!!;

    gpu::compat::TextureBarrier to_read = {
        .texture       = depth,
        .before_stage  = gpu::compat::Stage.DEPTH_STENCIL,
        .after_stage   = gpu::compat::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::compat::Hazard.DEPTH_WRITE,
        .after_hazard  = gpu::compat::Hazard.SHADER_READ,
        .old_layout    = gpu::compat::TextureLayout.DEPTH_STENCIL,
        .new_layout    = gpu::compat::TextureLayout.SHADER_READ,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_read)!!;
    gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: sample_pipeline,
        root:     root_span.gpu,
        groups:   { 1, 1, 1 },
    )!!;
    gpu::compat::BufferBarrier host_read = {
        .buffer = out_span.buffer, .offset = out_span.offset, .size = out_span.size,
        .before_stage = gpu::compat::Stage.COMPUTE_SHADER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;
    submit_and_finish(&device, &cmd);

    gpu::compat::invalidate_buffer(&device, out_span.buffer, 0, 0)!!;
    float* sampled = (float*)out_span.cpu;
    assert(*sampled == NEAR_Z, "heap-sampled depth matches the rendered z");
    gpu::compat::end_frame(&frame_token)!!;
}

fn void depth_pass_validation_faults() @test {
    gpu::compat::Device device = make_depth_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::TextureDesc sampled_only_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = DP_RT,
        .height    = DP_RT,
        .format    = gpu::compat::Format.D32_FLOAT,
        .usage     = { .sampled },
    };
    gpu::compat::TextureHandle sampled_only = gpu::compat::create_texture(&device, &sampled_only_desc)!!;
    defer gpu::compat::destroy_texture(&device, sampled_only)!!;

    gpu::compat::TextureHandle unbarriered = make_depth_texture(&device);
    defer gpu::compat::destroy_texture(&device, unbarriered)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::DepthTargetDesc no_usage = { .texture = sampled_only, .load_op = gpu::compat::LoadOp.CLEAR, .clear = { .depth = 1.0f } };
    gpu::compat::RenderPassDesc no_usage_pass = { .depth = &no_usage, .width = DP_RT, .height = DP_RT };
    assert(@catch(gpu::compat::cmd_begin_render_pass(&cmd, &no_usage_pass)) == gpu::compat::INVALID_ARGUMENT,
        "depth target without depth_attach usage faults INVALID_ARGUMENT");

    gpu::compat::DepthTargetDesc wrong_layout = { .texture = unbarriered, .load_op = gpu::compat::LoadOp.CLEAR, .clear = { .depth = 1.0f } };
    gpu::compat::RenderPassDesc wrong_layout_pass = { .depth = &wrong_layout, .width = DP_RT, .height = DP_RT };
    assert(@catch(gpu::compat::cmd_begin_render_pass(&cmd, &wrong_layout_pass)) == gpu::compat::INVALID_RESOURCE_STATE,
        "depth target outside the DEPTH_STENCIL layout faults INVALID_RESOURCE_STATE");

    gpu::compat::RenderPassDesc empty_pass = { .width = DP_RT, .height = DP_RT };
    assert(@catch(gpu::compat::cmd_begin_render_pass(&cmd, &empty_pass)) == gpu::compat::INVALID_ARGUMENT,
        "a pass with no attachments faults INVALID_ARGUMENT");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

const char[*] SHADOW_PROBE_SPIRV = $embed("shaders/shadow_probe.comp.spv");

fn void compare_sampler_exact_results() @test {
    gpu::compat::Device device = make_depth_device();
    defer gpu::compat::destroy_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);
    gpu::compat::TextureHandle depth = make_depth_texture(&device, sampled: true);
    defer gpu::compat::destroy_texture(&device, depth)!!;

    gpu::compat::TextureIndex tex_index = gpu::compat::create_texture_descriptor(&device, depth, null)!!;
    defer gpu::compat::destroy_texture_descriptor(&device, tex_index)!!;
    gpu::compat::SamplerDesc shadow_desc = {
        .min_filter     = gpu::compat::Filter.NEAREST,
        .mag_filter     = gpu::compat::Filter.NEAREST,
        .address_u      = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_v      = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_w      = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .compare_enable = true,
        .compare        = gpu::compat::CompareOp.LESS,
    };
    gpu::compat::SamplerIndex shadow_smp = gpu::compat::create_sampler(&device, &shadow_desc)!!;
    defer gpu::compat::destroy_sampler(&device, shadow_smp)!!;

    gpu::compat::ShaderDesc probe_sd = { .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = SHADOW_PROBE_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle probe = gpu::compat::create_shader(&device, &probe_sd)!!;
    defer gpu::compat::destroy_shader(&device, probe)!!;
    gpu::compat::ComputePipelineDesc probe_desc = { .shader = probe, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle probe_pipeline = gpu::compat::create_compute_pipeline(&device, &probe_desc)!!;
    defer gpu::compat::destroy_pipeline(&device, probe_pipeline)!!;

    gpu::compat::DepthState tested = { .test_enable = true, .write_enable = true, .compare = gpu::compat::CompareOp.LESS };
    gpu::compat::PipelineHandle pipeline = make_depth_pipeline(
        device:     &device,
        ctx:        &ctx,
        depth:      tested,
        with_color: false,
    );
    defer gpu::compat::destroy_pipeline(&device, pipeline)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan near_verts = write_tri(&frame_token, NEAR_Z);
    gpu::compat::GpuSpan color = write_color(&frame_token, { 1.0f, 1.0f, 1.0f, 1.0f });
    gpu::compat::GpuSpan out_span = gpu::compat::alloc_frame_span(&frame_token, 8, 8)!!;
    gpu::compat::GpuSpan root_span = gpu::compat::alloc_frame_span(&frame_token, ShadowProbeRoot::size, 16)!!;
    ShadowProbeRoot* root = (ShadowProbeRoot*)root_span.cpu;
    root.out_gpu        = out_span.gpu;
    root.depth_texture  = tex_index;
    root.shadow_sampler = shadow_smp;
    root.u = 0.5f;
    root.v = 0.5f;
    root.ref_lo = NEAR_Z - 0.1f;
    root.ref_hi = NEAR_Z + 0.1f;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_depth_to_attach(&cmd, depth);
    gpu::compat::DepthTargetDesc depth_target = {
        .texture = depth, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
        .clear = { .depth = 1.0f },
    };
    gpu::compat::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::compat::cmd_draw(
        commands:       &cmd,
        pipeline:       pipeline,
        vertex_root:    near_verts.gpu,
        fragment_root:  color.gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::compat::cmd_end_render_pass(&cmd)!!;

    gpu::compat::TextureBarrier to_read = {
        .texture       = depth,
        .before_stage  = gpu::compat::Stage.DEPTH_STENCIL,
        .after_stage   = gpu::compat::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::compat::Hazard.DEPTH_WRITE,
        .after_hazard  = gpu::compat::Hazard.SHADER_READ,
        .old_layout    = gpu::compat::TextureLayout.DEPTH_STENCIL,
        .new_layout    = gpu::compat::TextureLayout.SHADER_READ,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_read)!!;
    gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: probe_pipeline,
        root:     root_span.gpu,
        groups:   { 1, 1, 1 },
    )!!;
    gpu::compat::BufferBarrier host_read = {
        .buffer = out_span.buffer, .offset = out_span.offset, .size = out_span.size,
        .before_stage = gpu::compat::Stage.COMPUTE_SHADER, .after_stage = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;
    submit_and_finish(&device, &cmd);

    gpu::compat::invalidate_buffer(&device, out_span.buffer, 0, 0)!!;
    float* results = (float*)out_span.cpu;
    assert(results[0] == 1.0f, "reference below stored depth passes LESS compare, got %f", results[0]);
    assert(results[1] == 0.0f, "reference above stored depth fails LESS compare, got %f", results[1]);
    gpu::compat::end_frame(&frame_token)!!;
}

fn gpu::compat::PipelineHandle make_biased_pipeline(gpu::compat::Device* device, DepthCtx* ctx, float bias_constant) {
    gpu::compat::GraphicsPipelineDesc desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .topology        = gpu::compat::PrimitiveTopology.TRIANGLES,
        .depth           = { .test_enable = true, .write_enable = true, .compare = gpu::compat::CompareOp.LESS },
        .raster          = { .depth_bias_constant = bias_constant },
        .depth_format    = gpu::compat::Format.D32_FLOAT,
    };
    return gpu::compat::create_graphics_pipeline(device, &desc)!!;
}

fn void depth_bias_shifts_stored_depth() @test {
    gpu::compat::Device device = make_depth_device();
    defer gpu::compat::destroy_device(&device)!!;
    DepthCtx ctx = make_depth_ctx(&device);
    defer destroy_depth_ctx(&device, &ctx);

    gpu::compat::TextureHandle plain_depth = make_depth_texture(&device);
    defer gpu::compat::destroy_texture(&device, plain_depth)!!;
    gpu::compat::TextureHandle biased_depth = make_depth_texture(&device);
    defer gpu::compat::destroy_texture(&device, biased_depth)!!;

    gpu::compat::PipelineHandle plain = make_biased_pipeline(&device, &ctx, 0.0f);
    defer gpu::compat::destroy_pipeline(&device, plain)!!;
    gpu::compat::PipelineHandle biased = make_biased_pipeline(&device, &ctx, 4096.0f);
    defer gpu::compat::destroy_pipeline(&device, biased)!!;

    gpu::compat::BufferHandle readback = make_readback(&device, DP_DEPTH_BYTES * 2);
    defer gpu::compat::destroy_buffer(&device, readback)!!;
    gpu::compat::GpuSpan rb = gpu::compat::get_buffer_span(&device, readback)!!;
    gpu::compat::GpuSpan rb_plain = { .buffer = rb.buffer, .offset = 0, .size = DP_DEPTH_BYTES, .cpu = rb.cpu, .gpu = rb.gpu };
    char* biased_cpu = (char*)rb.cpu + DP_DEPTH_BYTES;
    gpu::compat::GpuSpan rb_biased = { .buffer = rb.buffer, .offset = DP_DEPTH_BYTES, .size = DP_DEPTH_BYTES, .cpu = (void*)biased_cpu, .gpu = rb.gpu };

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan verts = write_tri(&frame_token, NEAR_Z);
    gpu::compat::GpuSpan color = write_color(&frame_token, { 1.0f, 1.0f, 1.0f, 1.0f });

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::TextureHandle[2] targets = { plain_depth, biased_depth };
    gpu::compat::PipelineHandle[2] pipelines = { plain, biased };
    gpu::compat::GpuSpan*[2] readbacks = { &rb_plain, &rb_biased };
    for (uint i = 0; i < 2; i++) {
        record_depth_to_attach(&cmd, targets[i]);
        gpu::compat::DepthTargetDesc depth_target = {
            .texture = targets[i], .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
            .clear = { .depth = 1.0f },
        };
        gpu::compat::RenderPassDesc pass = { .depth = &depth_target, .width = DP_RT, .height = DP_RT };
        gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
        gpu::compat::cmd_draw(
            commands:       &cmd,
            pipeline:       pipelines[i],
            vertex_root:    verts.gpu,
            fragment_root:  color.gpu,
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::compat::cmd_end_render_pass(&cmd)!!;
        record_depth_readback(&cmd, targets[i], readbacks[i]);
    }
    submit_and_finish(&device, &cmd);

    gpu::compat::invalidate_buffer(&device, readback, 0, 0)!!;
    uint center = (DP_RT / 2 * DP_RT + DP_RT / 2);
    float plain_z = ((float*)rb_plain.cpu)[center];
    float biased_z = ((float*)rb_biased.cpu)[center];
    assert(plain_z == NEAR_Z, "unbiased depth stores the rendered z, got %f", plain_z);
    assert(biased_z > plain_z, "biased depth is pushed further, got %f vs %f", biased_z, plain_z);
    gpu::compat::end_frame(&frame_token)!!;
}
