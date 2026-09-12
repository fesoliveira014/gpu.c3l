module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::io;

const char[*] FETCH_STENCIL_SPIRV = $embed("shaders/fetch_stencil.comp.spv");
const char[*] FETCH_STENCIL_FRAG_SPIRV = $embed("shaders/fetch_stencil.frag.spv");
const usz ST_STENCIL_BYTES = DP_RT * DP_RT;

fn gpu::PipelineHandle make_stencil_pipeline(gpu::Device* device, DepthCtx* ctx, gpu::Format format, bool with_color = true) {
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .color_formats   = with_color ? color_formats[..] : (gpu::Format[]){},
        .depth_format    = format,
    };
    return gpu::create_graphics_pipeline(device, &desc)!!;
}

fn gpu::TextureHandle make_stencil_texture(gpu::Device* device, gpu::Format format, bool sampled = false) {
    gpu::TextureDesc desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = DP_RT,
        .height    = DP_RT,
        .format    = format,
        .usage     = { .depth_attach, .transfer_src, .sampled = sampled },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::TextureDesc stencil_texture_desc(gpu::Format format) {
    return {
        .access = { .graphics, .compute, .transfer },
        .width  = DP_RT,
        .height = DP_RT,
        .format = format,
        .usage  = { .depth_attach, .transfer_src },
    };
}

fn void record_aspect_readback(
    gpu::CommandList* cmd,
    gpu::TextureHandle texture,
    gpu::GpuSpan* rb_span,
    gpu::TextureAspect aspect,
    gpu::TextureLayout before_layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
) {
    gpu::TextureBarrier to_src = {
        .texture = texture,
        .before  = {
            .layout = before_layout,
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
    gpu::TextureBufferCopyDesc copy = { .texture = texture, .dst = *rb_span, .aspect = aspect };
    gpu::cmd_copy_texture_to_buffer(cmd, &copy)!!;
    gpu::Barrier host_read = { .before = { .transfer }, .after = { .host } };
    gpu::cmd_barrier(cmd, &host_read)!!;
}

fn gpu::GraphicsState stencil_graphics_state(gpu::StencilFaceState face, bool color_on) {
    gpu::GraphicsState state = gpu::render_geometry_state(DP_RT, DP_RT)!!;
    state.stencil = { .test_enable = true, .front = face, .back = face };
    return state;
}

fn void stencil_mask_selects_fragments() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    gpu::Format format = gpu::get_device_caps(&device)!!.depth_stencil_format;
    assert(format != gpu::Format.UNDEFINED, "a conformant device exposes a combined depth-stencil format");
    DepthCtx ctx = make_depth_ctx();
    gpu::PipelineHandle pipeline = make_stencil_pipeline(&device, &ctx, format);
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;

    gpu::TextureHandle target = make_color_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(&device, target)!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    gpu::TextureHandle ds = make_stencil_texture(&device, format);
    defer gpu::destroy_texture(&device, ds)!!;
    gpu::AttachmentViewHandle ds_view = create_test_attachment_view(&device, ds)!!;
    defer destroy_test_attachment_view(&device, &ds_view)!!;
    OwnedSpan color_rb = make_readback(&device, DP_COLOR_BYTES);
    defer free_owned_span(&device, &color_rb)!!;
    OwnedSpan stencil_rb = make_readback(&device, ST_STENCIL_BYTES);
    defer free_owned_span(&device, &stencil_rb)!!;
    OwnedSpan depth_rb = make_readback(&device, DP_DEPTH_BYTES);
    defer free_owned_span(&device, &depth_rb)!!;
    OwnedSpan tri = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &tri)!!;
    OwnedSpan green_owned = write_color(&device, { 0.0f, 1.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &green_owned)!!;

    gpu::TextureDesc desc = stencil_texture_desc(format);
    assert(gpu::texture_mip_aspect_bytes(&desc, 0, gpu::TextureAspect.STENCIL)!! == ST_STENCIL_BYTES
        && gpu::texture_mip_aspect_bytes(&desc, 0, gpu::TextureAspect.DEPTH)!! == DP_DEPTH_BYTES
        && @catch(gpu::texture_mip_bytes(&desc, 0)) == gpu::INVALID_ARGUMENT,
        "aspect footprints are 1 and 4 bytes per texel and the combined footprint is undefined");

    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_color_to_attach(&cmd, target);
    record_depth_to_attach(&cmd, ds);

    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
    }};
    gpu::DepthTargetDesc ds_target = {
        .view = ds_view,
        .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .stencil_load_op = gpu::LoadOp.CLEAR, .stencil_store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f, .stencil = 0 },
    };
    gpu::RenderPassDesc pass = { .colors = colors[..], .depth = &ds_target, .width = DP_RT, .height = DP_RT };
    gpu::ColorTargetState[1] color_off = {{ .write_mask = {} }};
    gpu::ColorTargetState[1] color_on = {{ .write_mask = gpu::COLOR_WRITE_ALL }};

    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
    gpu::GraphicsState write_id = stencil_graphics_state(
        gpu::stencil_face(gpu::CompareOp.ALWAYS, gpu::StencilOp.REPLACE, 1),
        false,
    );
    write_id.depth = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    write_id.color = { .targets = color_off[..] };
    gpu::cmd_set_graphics_state(&cmd, &write_id)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, tri.span),
        fragment_root:  test_span_address(&device, green_owned.span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::ColorTargetDesc[1] keep_colors = {{
        .view = target_view, .load_op = gpu::LoadOp.LOAD, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::DepthTargetDesc keep_ds = { .view = ds_view };
    gpu::RenderPassDesc keep_pass = { .colors = keep_colors[..], .depth = &keep_ds, .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &keep_pass)!!;
    gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
    gpu::GraphicsState mask_other = stencil_graphics_state(
        gpu::stencil_face(gpu::CompareOp.EQUAL, gpu::StencilOp.KEEP, 2),
        true,
    );
    mask_other.color = { .targets = color_on[..] };
    gpu::cmd_set_graphics_state(&cmd, &mask_other)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, tri.span),
        fragment_root:  test_span_address(&device, green_owned.span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::GraphicsState mask_one = mask_other;
    mask_one.stencil.front.reference = 1;
    mask_one.stencil.back.reference = 1;
    gpu::cmd_set_graphics_state(&cmd, &mask_one)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, tri.span),
        fragment_root:  test_span_address(&device, green_owned.span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::GpuSpan color_rb_span = color_rb.span;
    gpu::GpuSpan stencil_rb_span = stencil_rb.span;
    gpu::GpuSpan depth_rb_span = depth_rb.span;
    record_color_readback(&cmd, target, &color_rb_span);
    record_aspect_readback(
        cmd:     &cmd,
        texture: ds,
        rb_span: &stencil_rb_span,
        aspect:  gpu::TextureAspect.STENCIL,
    );
    gpu::TextureBufferCopyDesc depth_copy = { .texture = ds, .dst = depth_rb_span, .aspect = gpu::TextureAspect.DEPTH };
    gpu::cmd_copy_texture_to_buffer(&cmd, &depth_copy)!!;
    gpu::TextureBufferCopyDesc ambiguous = { .texture = ds, .dst = stencil_rb_span };
    assert(@catch(gpu::cmd_copy_texture_to_buffer(&cmd, &ambiguous)) == gpu::INVALID_ARGUMENT,
        "a combined-format copy must select an aspect");
    gpu::Barrier host_read = { .before = { .transfer }, .after = { .host } };
    gpu::cmd_barrier(&cmd, &host_read)!!;
    submit_and_finish(&device, &cmd);

    gpu::invalidate_mapped_span(&device, color_rb.span)!!;
    char* center = dp_center(&device, &color_rb_span);
    assert(center[1] == (char)255, "the fragment whose stencil equals the reference is written");
    char* corner = (char*)test_span_mapping(&device, color_rb_span).ptr;
    assert(corner[1] == (char)0, "fragments outside the stencil mask stay clear");

    gpu::invalidate_mapped_span(&device, stencil_rb.span)!!;
    char* stencil = (char*)test_span_mapping(&device, stencil_rb_span).ptr;
    assert(stencil[DP_RT / 2 * DP_RT + DP_RT / 2] == 1 && stencil[0] == 0,
        "the stencil aspect reads back one byte per texel");
    gpu::invalidate_mapped_span(&device, depth_rb.span)!!;
    uint* depth_texels = (uint*)test_span_mapping(&device, depth_rb_span).ptr;
    uint center_depth = depth_texels[DP_RT / 2 * DP_RT + DP_RT / 2];
    if (format == gpu::Format.D24_UNORM_S8_UINT) {
        uint expected = (uint)(NEAR_Z * 16777215.0f + 0.5f);
        assert((center_depth & 0xFFFFFF) == expected, "the D24 depth aspect reads back as 24-bit unorm in four bytes");
    } else {
        assert(bitcast(center_depth, float) == NEAR_Z, "the D32 depth aspect reads back as float in four bytes");
    }
}

fn void depth_only_pass_preserves_stencil() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    gpu::Format format = gpu::get_device_caps(&device)!!.depth_stencil_format;
    DepthCtx ctx = make_depth_ctx();
    gpu::PipelineHandle pipeline = make_stencil_pipeline(
        device:     &device,
        ctx:        &ctx,
        format:     format,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::TextureHandle ds = make_stencil_texture(&device, format);
    defer gpu::destroy_texture(&device, ds)!!;
    gpu::AttachmentViewHandle ds_view = create_test_attachment_view(&device, ds)!!;
    defer destroy_test_attachment_view(&device, &ds_view)!!;
    OwnedSpan stencil_rb = make_readback(&device, ST_STENCIL_BYTES);
    defer free_owned_span(&device, &stencil_rb)!!;
    OwnedSpan tri = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &tri)!!;
    OwnedSpan color = write_color(&device, { 0.0f, 1.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &color)!!;

    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_depth_to_attach(&cmd, ds);
    gpu::DepthTargetDesc clear_both = {
        .view = ds_view,
        .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .stencil_load_op = gpu::LoadOp.CLEAR, .stencil_store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f, .stencil = 0 },
    };
    gpu::RenderPassDesc first = { .depth = &clear_both, .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &first)!!;
    gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
    gpu::GraphicsState write_id = stencil_graphics_state(
        gpu::stencil_face(gpu::CompareOp.ALWAYS, gpu::StencilOp.REPLACE, 7),
        false,
    );
    gpu::cmd_set_graphics_state(&cmd, &write_id)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, tri.span),
        fragment_root:  test_span_address(&device, color.span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::DepthTargetDesc depth_only = {
        .view = ds_view,
        .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 0.5f },
    };
    gpu::RenderPassDesc second = { .depth = &depth_only, .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &second)!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::GpuSpan stencil_rb_span = stencil_rb.span;
    record_aspect_readback(
        cmd:     &cmd,
        texture: ds,
        rb_span: &stencil_rb_span,
        aspect:  gpu::TextureAspect.STENCIL,
    );
    submit_and_finish(&device, &cmd);
    gpu::invalidate_mapped_span(&device, stencil_rb.span)!!;
    char* stencil = (char*)test_span_mapping(&device, stencil_rb_span).ptr;
    assert(stencil[DP_RT / 2 * DP_RT + DP_RT / 2] == 7 && stencil[0] == 0,
        "zero stencil ops load and store the stencil plane through a depth-only pass");
}

fn void stencil_aspect_samples_through_the_heap() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    gpu::Format format = gpu::get_device_caps(&device)!!.depth_stencil_format;
    DepthCtx ctx = make_depth_ctx();
    gpu::PipelineHandle pipeline = make_stencil_pipeline(
        device:     &device,
        ctx:        &ctx,
        format:     format,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::ShaderDesc fetch_shader = { .spirv = FETCH_STENCIL_SPIRV[..], .entry_point = "main" };
    gpu::ComputePipelineDesc fetch_desc = { .shader = fetch_shader };
    gpu::PipelineHandle fetch = gpu::create_compute_pipeline(&device, &fetch_desc)!!;
    defer gpu::destroy_pipeline(&device, fetch)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::TextureHandle ds = make_stencil_texture(&device, format, true);
    defer gpu::destroy_texture(&device, ds)!!;
    gpu::AttachmentViewHandle ds_view = create_test_attachment_view(&device, ds)!!;
    defer destroy_test_attachment_view(&device, &ds_view)!!;

    gpu::TextureViewDesc ambiguous = {};
    assert(@catch(gpu::create_texture_view(&device, ds, &ambiguous)) == gpu::INVALID_ARGUMENT,
        "a sampled view of a combined format must select an aspect");
    gpu::TextureViewDesc stencil_desc = { .aspect = gpu::TextureAspect.STENCIL };
    gpu::TextureView stencil_view = gpu::create_texture_view(&device, ds, &stencil_desc)!!;
    defer gpu::destroy_texture_view(&device, stencil_view)!!;
    gpu::TextureViewDesc depth_desc = { .aspect = gpu::TextureAspect.DEPTH };
    gpu::TextureView depth_view = gpu::create_texture_view(&device, ds, &depth_desc)!!;
    defer gpu::destroy_texture_view(&device, depth_view)!!;
    assert(stencil_view.index != depth_view.index, "each aspect publishes its own slot");

    OwnedSpan tri = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &tri)!!;
    OwnedSpan color = write_color(&device, { 0.0f, 1.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &color)!!;
    OwnedSpan output = make_readback(&device, 8);
    defer free_owned_span(&device, &output)!!;
    OwnedSpan roots = allocate_owned_span(
        device:       &device,
        size:         FetchStencilRoot::size * 2,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute },
        debug_name:   "fetch_stencil_roots",
    )!!;
    defer free_owned_span(&device, &roots)!!;
    FetchStencilRoot* root = (FetchStencilRoot*)test_span_mapping(&device, roots.span).ptr;
    root[0] = { .out_gpu = test_span_address(&device, output.span), .texture_index = stencil_view.index, .x = DP_RT / 2, .y = DP_RT / 2 };
    root[1] = { .out_gpu = test_span_address(&device, output.span.checked_subspan(4, 4)!!), .texture_index = stencil_view.index, .x = 0, .y = 0 };
    gpu::flush_mapped_span(&device, roots.span)!!;

    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_depth_to_attach(&cmd, ds);
    gpu::TextureViewDesc aspect_view = { .aspect = gpu::TextureAspect.DEPTH };
    gpu::TextureBarrier bad_barrier = gpu::texture_view_transition(
        texture: ds,
        view:    aspect_view,
        before:  { .layout = gpu::TextureLayout.DEPTH_ATTACHMENT, .stages = { .depth_output }, .access = { .read, .write } },
        after:   gpu::sampled_at({ .compute }),
    )!!;
    assert(@catch(gpu::cmd_texture_barrier(&cmd, &bad_barrier)) == gpu::INVALID_ARGUMENT,
        "a barrier view must not select an aspect");
    gpu::DepthTargetDesc ds_target = {
        .view = ds_view,
        .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .stencil_load_op = gpu::LoadOp.CLEAR, .stencil_store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f, .stencil = 0 },
    };
    gpu::RenderPassDesc pass = { .depth = &ds_target, .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
    gpu::GraphicsState write_id = stencil_graphics_state(
        gpu::stencil_face(gpu::CompareOp.ALWAYS, gpu::StencilOp.REPLACE, 3),
        false,
    );
    gpu::cmd_set_graphics_state(&cmd, &write_id)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, tri.span),
        fragment_root:  test_span_address(&device, color.span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::TextureBarrier to_sampled = gpu::texture_transition(
        texture: ds,
        before:  { .layout = gpu::TextureLayout.DEPTH_ATTACHMENT, .stages = { .depth_output }, .access = { .read, .write } },
        after:   gpu::sampled_at({ .compute }),
    )!!;
    gpu::cmd_texture_barrier(&cmd, &to_sampled)!!;
    bind_test_compute_pipeline(&cmd, fetch);
    for (uint i = 0; i < 2; i++) {
        gpu::cmd_dispatch(
            commands: &cmd,
            root:     test_span_address(&device, roots.span.checked_subspan(i * FetchStencilRoot::size, FetchStencilRoot::size)!!),
            groups:   { 1, 1, 1 },
        )!!;
    }
    gpu::Barrier host_read = { .before = { .compute }, .after = { .host } };
    gpu::cmd_barrier(&cmd, &host_read)!!;
    submit_and_finish(&device, &cmd);
    gpu::invalidate_mapped_span(&device, output.span)!!;
    uint* fetched = (uint*)test_span_mapping(&device, output.span).ptr;
    assert(fetched[0] == 3 && fetched[1] == 0, "the stencil aspect is fetched through the integer heap alias");
}

fn void stencil_only_format_when_supported() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureFormatSupport support = gpu::get_texture_format_support(&device, gpu::Format.S8_UINT)!!;
    if (!support.features.depth_attach) {
        io::eprintfn("S8_UINT attachments not supported on this adapter; skipping");
        return;
    }
    DepthCtx ctx = make_depth_ctx();
    gpu::PipelineHandle pipeline = make_stencil_pipeline(
        device:     &device,
        ctx:        &ctx,
        format:     gpu::Format.S8_UINT,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::TextureHandle stencil = make_stencil_texture(&device, gpu::Format.S8_UINT);
    defer gpu::destroy_texture(&device, stencil)!!;
    gpu::AttachmentViewHandle view = create_test_attachment_view(&device, stencil)!!;
    defer destroy_test_attachment_view(&device, &view)!!;
    OwnedSpan rb = make_readback(&device, ST_STENCIL_BYTES);
    defer free_owned_span(&device, &rb)!!;
    OwnedSpan tri = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &tri)!!;
    OwnedSpan color = write_color(&device, { 0.0f, 1.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &color)!!;
    gpu::TextureDesc desc = stencil_texture_desc(gpu::Format.S8_UINT);
    assert(gpu::texture_mip_bytes(&desc, 0)!! == ST_STENCIL_BYTES
        && @catch(gpu::texture_mip_aspect_bytes(&desc, 0, gpu::TextureAspect.DEPTH)) == gpu::INVALID_ARGUMENT,
        "S8_UINT has one byte per texel and no depth aspect");

    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_depth_to_attach(&cmd, stencil);
    gpu::DepthTargetDesc target = {
        .view = view,
        .stencil_load_op = gpu::LoadOp.CLEAR, .stencil_store_op = gpu::StoreOp.STORE,
        .clear = { .stencil = 0 },
    };
    gpu::RenderPassDesc pass = { .depth = &target, .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
    gpu::GraphicsState write_id = stencil_graphics_state(
        gpu::stencil_face(gpu::CompareOp.ALWAYS, gpu::StencilOp.REPLACE, 5),
        false,
    );
    gpu::cmd_set_graphics_state(&cmd, &write_id)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, tri.span),
        fragment_root:  test_span_address(&device, color.span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::GpuSpan rb_span = rb.span;
    record_aspect_readback(
        cmd:     &cmd,
        texture: stencil,
        rb_span: &rb_span,
        aspect:  gpu::TextureAspect.AUTO,
    );
    submit_and_finish(&device, &cmd);
    gpu::invalidate_mapped_span(&device, rb.span)!!;
    char* values = (char*)test_span_mapping(&device, rb_span).ptr;
    assert(values[DP_RT / 2 * DP_RT + DP_RT / 2] == 5 && values[0] == 0,
        "a stencil-only attachment writes and reads back");
}

fn void stencil_validation_faults() @test {
    stencil_validation_faults_under();
}

fn void stencil_validation_faults_under() {
    gpu::RuntimeDesc device_desc = {
        .enable_vulkan_validation = true,
        .application_name         = "gpu_stencil_validation_test",
    };
    gpu::Device device = create_test_device(&device_desc)!!;
    defer destroy_test_device(&device)!!;
    DepthCtx ctx = make_depth_ctx();
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc color_as_depth = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .color_formats   = color_formats[..],
        .depth_format    = gpu::Format.RGBA8_UNORM,
    };
    assert(@catch(gpu::create_graphics_pipeline(&device, &color_as_depth)) == gpu::INVALID_ARGUMENT,
        "a color format is not a depth-stencil format");
    gpu::Format[1] depth_as_color = { gpu::Format.D24_UNORM_S8_UINT };
    gpu::GraphicsPipelineDesc stencil_as_color = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .color_formats   = depth_as_color[..],
    };
    assert(@catch(gpu::create_graphics_pipeline(&device, &stencil_as_color)) == gpu::INVALID_ARGUMENT,
        "a depth-stencil format is not a color target");

    gpu::TextureHandle depth = make_depth_texture(&device, true);
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::TextureViewDesc stencil_of_depth = { .aspect = gpu::TextureAspect.STENCIL };
    assert(@catch(gpu::create_texture_view(&device, depth, &stencil_of_depth)) == gpu::INVALID_ARGUMENT,
        "D32_FLOAT has no stencil aspect");
    gpu::TextureDesc storage_stencil = stencil_texture_desc(gpu::Format.D24_UNORM_S8_UINT);
    storage_stencil.usage = { .storage };
    assert(!gpu::supports_texture_desc(&device, &storage_stencil)!!,
        "depth-stencil formats reject storage usage");

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::PipelineHandle pipeline = make_stencil_pipeline(&device, &ctx, gpu::Format.UNDEFINED);
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::TextureHandle target = make_color_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(&device, target)!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_color_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .view = target_view }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_bind_pipeline(&cmd, pipeline)!!;
    gpu::GraphicsState state = stencil_graphics_state(
        gpu::stencil_face(gpu::CompareOp.ALWAYS, gpu::StencilOp.KEEP, 0),
        true,
    );
    gpu::ColorTargetState[1] color_on = {{ .write_mask = gpu::COLOR_WRITE_ALL }};
    state.color = { .targets = color_on[..] };
    state.stencil.test_enable = false;
    *((int*)&state.stencil.back.compare) = 99;
    gpu::cmd_set_graphics_state(&cmd, &state)!!;
    gpu::cmd_end_render_pass(&cmd)!!;
}

struct StencilDiagnosticCapture {
    uint performance;
    uint faults;
    bool field_matches;
}

fn void stencil_aspect_fetches_in_fragment_stage() @test {
    gpu::Device device = make_depth_device();
    defer destroy_test_device(&device)!!;
    gpu::Format format = gpu::get_device_caps(&device)!!.depth_stencil_format;
    DepthCtx ctx = make_depth_ctx();
    gpu::PipelineHandle pipeline = make_stencil_pipeline(
        device:     &device,
        ctx:        &ctx,
        format:     format,
        with_color: false,
    );
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    gpu::ShaderDesc fetch_frag = { .spirv = FETCH_STENCIL_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::Format[1] uint_formats = { gpu::Format.R8_UINT };
    gpu::GraphicsPipelineDesc fetch_desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = fetch_frag,
        .color_formats   = uint_formats[..],
    };
    gpu::PipelineHandle fetch = gpu::create_graphics_pipeline(&device, &fetch_desc)!!;
    defer gpu::destroy_pipeline(&device, fetch)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::TextureHandle ds = make_stencil_texture(&device, format, true);
    defer gpu::destroy_texture(&device, ds)!!;
    gpu::AttachmentViewHandle ds_view = create_test_attachment_view(&device, ds)!!;
    defer destroy_test_attachment_view(&device, &ds_view)!!;
    gpu::TextureViewDesc stencil_desc = { .aspect = gpu::TextureAspect.STENCIL };
    gpu::TextureView stencil_view = gpu::create_texture_view(&device, ds, &stencil_desc)!!;
    defer gpu::destroy_texture_view(&device, stencil_view)!!;
    gpu::TextureDesc target_desc = {
        .access = { .graphics, .transfer },
        .width  = DP_RT,
        .height = DP_RT,
        .format = gpu::Format.R8_UINT,
        .usage  = { .color_attach, .transfer_src },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &target_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(&device, target)!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;

    OwnedSpan tri = write_tri(&device, NEAR_Z);
    defer free_owned_span(&device, &tri)!!;
    OwnedSpan color = write_color(&device, { 0.0f, 1.0f, 0.0f, 1.0f });
    defer free_owned_span(&device, &color)!!;
    OwnedSpan rb = make_readback(&device, DP_RT * DP_RT);
    defer free_owned_span(&device, &rb)!!;
    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         FetchStencilRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "fetch_stencil_fragment_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    FetchStencilRoot* root = (FetchStencilRoot*)test_span_mapping(&device, root_owned.span).ptr;
    *root = { .texture_index = stencil_view.index };
    gpu::flush_mapped_span(&device, root_owned.span)!!;

    gpu::CommandList write = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&write)!!;
    record_depth_to_attach(&write, ds);
    gpu::DepthTargetDesc ds_target = {
        .view = ds_view,
        .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .stencil_load_op = gpu::LoadOp.CLEAR, .stencil_store_op = gpu::StoreOp.STORE,
        .clear = { .depth = 1.0f, .stencil = 0 },
    };
    gpu::RenderPassDesc write_pass = { .depth = &ds_target, .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&write, &write_pass)!!;
    gpu::cmd_bind_pipeline(&write, pipeline)!!;
    gpu::GraphicsState write_id = stencil_graphics_state(
        gpu::stencil_face(gpu::CompareOp.ALWAYS, gpu::StencilOp.REPLACE, 5),
        false,
    );
    gpu::cmd_set_graphics_state(&write, &write_id)!!;
    gpu::cmd_draw(
        commands:       &write,
        vertex_root:    test_span_address(&device, tri.span),
        fragment_root:  test_span_address(&device, color.span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&write)!!;
    gpu::TextureBarrier to_sampled = gpu::texture_transition(
        texture: ds,
        before:  { .layout = gpu::TextureLayout.DEPTH_ATTACHMENT, .stages = { .depth_output }, .access = { .read, .write } },
        after:   gpu::sampled_at({ .fragment_shader }),
    )!!;
    gpu::cmd_texture_barrier(&write, &to_sampled)!!;
    submit_and_finish(&device, &write);

    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;
    record_color_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .view     = target_view,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear    = { .uint_rgba = { 0, 0, 0, 0 } },
    }};
    gpu::RenderPassDesc fetch_pass = { .colors = colors[..], .width = DP_RT, .height = DP_RT };
    gpu::cmd_begin_render_pass(&cmd, &fetch_pass)!!;
    gpu::cmd_bind_pipeline(&cmd, fetch)!!;
    gpu::GraphicsState fetch_state = gpu::render_geometry_state(DP_RT, DP_RT)!!;
    gpu::ColorTargetState[1] color_on = {{ .write_mask = gpu::COLOR_WRITE_ALL }};
    fetch_state.color = { .targets = color_on[..] };
    gpu::cmd_set_graphics_state(&cmd, &fetch_state)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, tri.span),
        fragment_root:  test_span_address(&device, root_owned.span),
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
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = rb.span };
    gpu::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
    gpu::Barrier host_read = { .before = { .transfer }, .after = { .host } };
    gpu::cmd_barrier(&cmd, &host_read)!!;
    submit_and_finish(&device, &cmd);
    gpu::invalidate_mapped_span(&device, rb.span)!!;
    char* fetched = (char*)test_span_mapping(&device, rb.span).ptr;
    assert(fetched[(DP_RT / 2) * DP_RT + DP_RT / 2] == (char)5, "the fragment stage fetches the stencil aspect into an R8_UINT target");
    assert(fetched[0] == (char)0, "pixels outside the triangle keep the clear value");
}
