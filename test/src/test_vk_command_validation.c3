module gpu_test;

import gpu;
import gpu::vk;

// Headless (lavapipe): recording/submit validation faults — mixed-queue
// submits, usage/alignment, pipeline-kind and shader-stage mismatches.

const char[*] VAL_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] VAL_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");
const char[*] VAL_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

const uint VAL_RT = 16;

fn gpu::Device make_val_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_command_validation_test",
    };
    return gpu::create_device(&desc)!!;
}

<* Textured draw context, same shape as the indirect-draw tests: offscreen
   shaders, one graphics pipeline, and a 1x1 white pattern feeding the
   fragment heap path. *>
struct ValDrawCtx {
    gpu::ShaderHandle   vert;
    gpu::ShaderHandle   frag;
    gpu::PipelineHandle graphics;
    gpu::TextureHandle  pattern;
    gpu::TextureIndex   tex_index;
    gpu::SamplerIndex   smp_index;
}

fn ValDrawCtx make_val_draw_ctx(gpu::Device* device) {
    ValDrawCtx ctx;
    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    ctx.vert = gpu::create_shader(device, &vert_sd)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = VAL_FRAG_SPIRV[..], .entry_point = "main" };
    ctx.frag = gpu::create_shader(device, &frag_sd)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = formats[..],
    };
    ctx.graphics = gpu::create_graphics_pipeline(device, &pipe_desc)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    ctx.pattern = gpu::create_texture(device, &tex_desc)!!;
    ctx.tex_index = gpu::create_texture_descriptor(device, ctx.pattern, null)!!;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    ctx.smp_index = gpu::create_sampler(device, &sampler_desc)!!;

    char[4] white = { (char)255, (char)255, (char)255, (char)255 };
    gpu::upload_texture_data(
        device:      device,
        dst:         ctx.pattern,
        mip:         0,
        data:        white[..],
        next_stage:  gpu::Stage.FRAGMENT_SHADER,
        next_hazard: gpu::Hazard.SHADER_READ,
        next_layout: gpu::TextureLayout.SHADER_READ,
    )!!;
    return ctx;
}

fn void destroy_val_draw_ctx(gpu::Device* device, ValDrawCtx* ctx) {
    gpu::destroy_sampler(device, ctx.smp_index)!!;
    gpu::destroy_texture_descriptor(device, ctx.tex_index)!!;
    gpu::destroy_texture(device, ctx.pattern)!!;
    gpu::destroy_pipeline(device, ctx.graphics)!!;
    gpu::destroy_shader(device, ctx.frag)!!;
    gpu::destroy_shader(device, ctx.vert)!!;
}

fn gpu::TextureHandle make_val_target(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = VAL_RT,
        .height    = VAL_RT,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void write_val_triangle_verts(gpu::GpuSpan* span) {
    float* verts = (float*)span.cpu;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = vertex_data[i];
}

fn void record_val_target_to_attach(gpu::CommandList* cmd, gpu::TextureHandle target) {
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

fn void mixed_queue_submit_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList gfx = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&device, &gfx)!!;
    gpu::CommandList xfer = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::end_commands(&device, &xfer)!!;

    gpu::CommandList[2] lists = { gfx, xfer };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    fault mixed = @catch(gpu::submit(&device, &submit));
    assert(mixed == gpu::INVALID_ARGUMENT, "mixed-queue submit faults INVALID_ARGUMENT before any queue work runs");
    gpu::end_frame(&device)!!;
}

fn void copy_without_transfer_usage_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc plain_desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle plain = gpu::create_buffer(&device, &plain_desc)!!;
    defer gpu::destroy_buffer(&device, plain)!!;
    gpu::BufferDesc src_desc = { .size = 64, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.STAGING };
    gpu::BufferHandle src = gpu::create_buffer(&device, &src_desc)!!;
    defer gpu::destroy_buffer(&device, src)!!;
    gpu::BufferDesc dst_desc = { .size = 64, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK };
    gpu::BufferHandle dst = gpu::create_buffer(&device, &dst_desc)!!;
    defer gpu::destroy_buffer(&device, dst)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;

    gpu::BufferCopyDesc no_src_usage_copy = { .src = plain, .dst = dst, .size = 64 };
    fault no_src_usage = @catch(gpu::cmd_copy_buffer(&cmd, &no_src_usage_copy));
    assert(no_src_usage == gpu::INVALID_ARGUMENT, "copy from a buffer without transfer_src faults INVALID_ARGUMENT");

    gpu::BufferCopyDesc no_dst_usage_copy = { .src = src, .dst = plain, .size = 64 };
    fault no_dst_usage = @catch(gpu::cmd_copy_buffer(&cmd, &no_dst_usage_copy));
    assert(no_dst_usage == gpu::INVALID_ARGUMENT, "copy to a buffer without transfer_dst faults INVALID_ARGUMENT");

    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}

fn void texture_copy_without_buffer_transfer_usage_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_src, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::BufferDesc plain_desc = { .size = 256, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle plain = gpu::create_buffer(&device, &plain_desc)!!;
    defer gpu::destroy_buffer(&device, plain)!!;
    gpu::GpuSpan plain_span = gpu::get_buffer_span(&device, plain)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;

    // Buffer usage is checked before the texture's tracked layout, so no
    // barrier is needed to reach either fault.
    gpu::BufferTextureCopyDesc up = { .src = plain_span, .texture = texture, .width = 4, .height = 4 };
    fault no_src_usage = @catch(gpu::cmd_copy_buffer_to_texture(&cmd, &up));
    assert(no_src_usage == gpu::INVALID_ARGUMENT, "buffer-to-texture copy without buffer transfer_src faults INVALID_ARGUMENT");

    gpu::TextureBufferCopyDesc down = { .texture = texture, .dst = plain_span, .width = 4, .height = 4 };
    fault no_dst_usage = @catch(gpu::cmd_copy_texture_to_buffer(&cmd, &down));
    assert(no_dst_usage == gpu::INVALID_ARGUMENT, "texture-to-buffer copy without buffer transfer_dst faults INVALID_ARGUMENT");

    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}

fn void fill_without_transfer_dst_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;
    fault no_usage = @catch(gpu::cmd_fill_buffer(&cmd, buffer, 0, 64, 0));
    assert(no_usage == gpu::INVALID_ARGUMENT, "fill on a buffer without transfer_dst faults INVALID_ARGUMENT");
    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}

fn void indexed_draw_without_index_usage_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    // Storage-only: no index usage.
    gpu::BufferDesc bad_index_desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle bad_index_buffer = gpu::create_buffer(&device, &bad_index_desc)!!;
    defer gpu::destroy_buffer(&device, bad_index_buffer)!!;
    gpu::GpuSpan bad_index_span = gpu::get_buffer_span(&device, bad_index_buffer)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!!;
    write_val_triangle_verts(&verts_span);
    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, 16, 16)!!;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    fault no_index_usage = @catch(gpu::cmd_draw_indexed(
        commands:      &cmd,
        pipeline:      ctx.graphics,
        vertex_root:   verts_span.gpu,
        fragment_root: frag_span.gpu,
        index_span:    bad_index_span,
        index_count:   3,
        instance_count: 1,
    ));
    assert(no_index_usage == gpu::INVALID_ARGUMENT, "indexed draw over a non-index buffer faults INVALID_ARGUMENT");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}

fn void indexed_draw_misaligned_offset_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!!;
    write_val_triangle_verts(&verts_span);
    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, 16, 16)!!;
    // Frame-arena buffers carry .index usage; a 2-byte offset still misaligns
    // the default U32 index type (needs 4-byte alignment).
    gpu::GpuSpan index_span = gpu::alloc_frame_span(&device, 16, 16)!!;
    gpu::GpuSpan misaligned = index_span.subspan(2, 8);

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    fault misaligned_fault = @catch(gpu::cmd_draw_indexed(
        commands:      &cmd,
        pipeline:      ctx.graphics,
        vertex_root:   verts_span.gpu,
        fragment_root: frag_span.gpu,
        index_span:    misaligned,
        index_count:   1,
        instance_count: 1,
    ));
    assert(misaligned_fault == gpu::INVALID_ARGUMENT, "misaligned index offset faults INVALID_ARGUMENT");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}

fn void dispatch_with_graphics_pipeline_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan root_span = gpu::alloc_frame_span(&device, 16, 16)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    fault wrong_kind = @catch(gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: ctx.graphics,
        root:     root_span.gpu,
        groups:   { 1, 1, 1 },
    ));
    assert(wrong_kind == gpu::INVALID_ARGUMENT, "dispatch with a graphics pipeline faults INVALID_ARGUMENT");
    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}

fn void draw_with_compute_pipeline_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = VAL_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle comp_shader = gpu::create_shader(&device, &comp_sd)!!;
    defer gpu::destroy_shader(&device, comp_shader)!!;
    gpu::ComputePipelineDesc comp_desc = { .shader = comp_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle compute = gpu::create_compute_pipeline(&device, &comp_desc)!!;
    defer gpu::destroy_pipeline(&device, compute)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    fault wrong_kind = @catch(gpu::cmd_draw(
        commands:      &cmd,
        pipeline:      compute,
        vertex_root:   (gpu::GpuAddress)0,
        fragment_root: (gpu::GpuAddress)0,
        vertex_count:  3,
        instance_count: 1,
    ));
    assert(wrong_kind == gpu::INVALID_ARGUMENT, "draw with a compute pipeline faults INVALID_ARGUMENT");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}

fn void compute_pipeline_from_vertex_shader_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle vert_shader = gpu::create_shader(&device, &vert_sd)!!;
    defer gpu::destroy_shader(&device, vert_shader)!!;

    gpu::ComputePipelineDesc desc = { .shader = vert_shader, .push_constant_size = gpu::RootPush::size };
    fault wrong_stage = @catch(gpu::create_compute_pipeline(&device, &desc));
    assert(wrong_stage == gpu::INVALID_ARGUMENT, "compute pipeline from a vertex-stage shader faults INVALID_ARGUMENT");
}

fn void graphics_pipeline_from_swapped_shader_stages_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle vert_shader = gpu::create_shader(&device, &vert_sd)!!;
    defer gpu::destroy_shader(&device, vert_shader)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = VAL_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle frag_shader = gpu::create_shader(&device, &frag_sd)!!;
    defer gpu::destroy_shader(&device, frag_shader)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc swapped = {
        .vertex_shader   = frag_shader,
        .fragment_shader = vert_shader,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = formats[..],
    };
    fault wrong_stage = @catch(gpu::create_graphics_pipeline(&device, &swapped));
    assert(wrong_stage == gpu::INVALID_ARGUMENT, "graphics pipeline with swapped vertex/fragment shaders faults INVALID_ARGUMENT");
}

fn void valid_copy_fill_and_draw_succeed() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::BufferDesc up_desc = { .size = 64, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.STAGING };
    gpu::BufferHandle upload = gpu::create_buffer(&device, &up_desc)!!;
    defer gpu::destroy_buffer(&device, upload)!!;
    gpu::BufferDesc rb_desc = { .size = 64, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, readback)!!;
    gpu::BufferDesc fill_desc = { .size = 64, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle fill_target = gpu::create_buffer(&device, &fill_desc)!!;
    defer gpu::destroy_buffer(&device, fill_target)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!!;
    write_val_triangle_verts(&verts_span);
    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, FragRoot::size, 16)!!;
    FragRoot* frag_root = (FragRoot*)frag_span.cpu;
    frag_root.texture_index = ctx.tex_index;
    frag_root.sampler_index = ctx.smp_index;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::BufferCopyDesc copy = { .src = upload, .dst = readback, .size = 64 };
    gpu::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::cmd_fill_buffer(&cmd, fill_target, 0, 64, 0)!!;

    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw(&cmd, ctx.graphics, verts_span.gpu, frag_span.gpu, 3, 1)!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&device)!!;
}
