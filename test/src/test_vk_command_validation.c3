module gpu_test;

import gpu::compat @public;
import gpu::compat::vk @public;

// Headless (lavapipe): recording/submit validation faults — mixed-queue
// submits, usage/alignment, pipeline-kind and shader-stage mismatches.

const char[*] VAL_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] VAL_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");
const char[*] VAL_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

const uint VAL_RT = 16;

struct SyncDiagnosticCapture {
    uint                   calls;
    bool                   has_fault;
    fault                  public_fault;
    bool                   public_contract;
    gpu::compat::DebugResourceKind resource_kind;
    bool                   has_public_identity;
    uint                   index;
    uint                   generation;
    ZString                expected_operation;
    ZString                expected_debug_name;
    ZString                expected_rejected_field;
    ZString                expected_invariant;
    bool                   operation_matches;
    bool                   debug_name_matches;
    bool                   rejected_field_matches;
    bool                   invariant_matches;
}

fn void capture_sync_diagnostic(gpu::compat::DebugMessage* message, void* user_data) {
    if (message.operation == null) return;
    SyncDiagnosticCapture* capture = (SyncDiagnosticCapture*)user_data;
    capture.calls++;
    capture.has_fault = message.has_fault;
    capture.public_fault = message.public_fault;
    capture.public_contract = message.category.public_contract;
    capture.resource_kind = message.resource.kind;
    capture.has_public_identity = message.resource.has_public_identity;
    capture.index = message.resource.index;
    capture.generation = message.resource.generation;
    capture.operation_matches = message.operation.str_view() == capture.expected_operation.str_view();
    capture.debug_name_matches = capture.expected_debug_name == null
        ? message.resource.debug_name == null
        : message.resource.debug_name != null
            && message.resource.debug_name.str_view() == capture.expected_debug_name.str_view();
    capture.rejected_field_matches = message.rejected_field != null
        && message.rejected_field.str_view() == capture.expected_rejected_field.str_view();
    capture.invariant_matches = message.invariant != null
        && message.invariant.str_view() == capture.expected_invariant.str_view();
}

fn gpu::compat::Device make_sync_diagnostic_device(SyncDiagnosticCapture* capture) {
    gpu::compat::DeviceDesc desc = {
        .backend              = gpu::compat::BackendKind.VULKAN,
        .enable_validation    = true,
        .frames_in_flight     = 2,
        .application_name     = "gpu_sync_diagnostic_test",
        .debug_callback       = &capture_sync_diagnostic,
        .debug_user_data      = capture,
    };
    return gpu::compat::create_device(&desc)!!;
}

fn gpu::compat::Device make_val_device() {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_command_validation_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

<* Textured draw context, same shape as the indirect-draw tests: offscreen
   shaders, one graphics pipeline, and a 1x1 white pattern feeding the
   fragment heap path. *>
struct ValDrawCtx {
    gpu::compat::ShaderHandle   vert;
    gpu::compat::ShaderHandle   frag;
    gpu::compat::PipelineHandle graphics;
    gpu::compat::TextureHandle  pattern;
    gpu::compat::TextureIndex   tex_index;
    gpu::compat::SamplerIndex   smp_index;
}

fn ValDrawCtx make_val_draw_ctx(gpu::compat::Device* device) {
    ValDrawCtx ctx;
    gpu::compat::ShaderDesc vert_sd = { .stage = gpu::compat::ShaderStage.VERTEX, .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    ctx.vert = gpu::compat::create_shader(device, &vert_sd)!!;
    gpu::compat::ShaderDesc frag_sd = { .stage = gpu::compat::ShaderStage.FRAGMENT, .spirv = VAL_FRAG_SPIRV[..], .entry_point = "main" };
    ctx.frag = gpu::compat::create_shader(device, &frag_sd)!!;

    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = ctx.vert,
        .fragment_shader = ctx.frag,
        .topology        = gpu::compat::PrimitiveTopology.TRIANGLES,
        .color_formats   = formats[..],
    };
    ctx.graphics = gpu::compat::create_graphics_pipeline(device, &pipe_desc)!!;

    gpu::compat::TextureDesc tex_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = 1,
        .height    = 1,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    ctx.pattern = gpu::compat::create_texture(device, &tex_desc)!!;
    ctx.tex_index = gpu::compat::create_texture_descriptor(device, ctx.pattern, null)!!;
    gpu::compat::SamplerDesc sampler_desc = {
        .min_filter = gpu::compat::Filter.NEAREST,
        .mag_filter = gpu::compat::Filter.NEAREST,
        .address_u  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
    };
    ctx.smp_index = gpu::compat::create_sampler(device, &sampler_desc)!!;

    char[4] white = { (char)255, (char)255, (char)255, (char)255 };
    gpu::compat::upload_texture_data(
        device:      device,
        dst:         ctx.pattern,
        mip:         0,
        data:        white[..],
        next_stage:  gpu::compat::Stage.FRAGMENT_SHADER,
        next_hazard: gpu::compat::Hazard.SHADER_READ,
        next_layout: gpu::compat::TextureLayout.SHADER_READ,
    )!!;
    return ctx;
}

fn void destroy_val_draw_ctx(gpu::compat::Device* device, ValDrawCtx* ctx) {
    gpu::compat::destroy_sampler(device, ctx.smp_index)!!;
    gpu::compat::destroy_texture_descriptor(device, ctx.tex_index)!!;
    gpu::compat::destroy_texture(device, ctx.pattern)!!;
    gpu::compat::destroy_pipeline(device, ctx.graphics)!!;
    gpu::compat::destroy_shader(device, ctx.frag)!!;
    gpu::compat::destroy_shader(device, ctx.vert)!!;
}

fn gpu::compat::TextureHandle make_val_target(gpu::compat::Device* device) {
    gpu::compat::TextureDesc desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = VAL_RT,
        .height    = VAL_RT,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    return gpu::compat::create_texture(device, &desc)!!;
}

fn gpu::compat::TextureHandle make_val_color_target(gpu::compat::Device* device, uint width, uint height, uint mip_levels) {
    gpu::compat::TextureDesc desc = {
        .dimension  = gpu::compat::TextureDimension.TEX_2D,
        .width      = width,
        .height     = height,
        .mip_levels = mip_levels,
        .format     = gpu::compat::Format.RGBA8_UNORM,
        .usage      = { .color_attach, .transfer_src },
    };
    return gpu::compat::create_texture(device, &desc)!!;
}

fn gpu::compat::TextureHandle make_val_depth_target(gpu::compat::Device* device, uint width, uint height) {
    gpu::compat::TextureDesc desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = width,
        .height    = height,
        .format    = gpu::compat::Format.D32_FLOAT,
        .usage     = { .depth_attach },
    };
    return gpu::compat::create_texture(device, &desc)!!;
}

fn gpu::compat::TextureHandle make_val_depth_target_mips(gpu::compat::Device* device, uint width, uint height, uint mip_levels) {
    gpu::compat::TextureDesc desc = {
        .dimension  = gpu::compat::TextureDimension.TEX_2D,
        .width      = width,
        .height     = height,
        .mip_levels = mip_levels,
        .format     = gpu::compat::Format.D32_FLOAT,
        .usage      = { .depth_attach },
    };
    return gpu::compat::create_texture(device, &desc)!!;
}

fn void record_val_depth_to_attach(gpu::compat::CommandList* cmd, gpu::compat::TextureHandle depth) {
    record_val_texture_transition(
        cmd:          cmd,
        texture:      depth,
        new_layout:   gpu::compat::TextureLayout.DEPTH_STENCIL,
        after_stage:  gpu::compat::Stage.DEPTH_STENCIL,
        after_hazard: gpu::compat::Hazard.DEPTH_WRITE,
    );
}

fn uint val_cached_view_count(gpu::compat::Device* device, gpu::compat::TextureHandle texture) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(*device);
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    uint count = 0;
    foreach (&cached : slot.views) if (cached.used) count++;
    return count;
}

fn bool val_has_cached_view(gpu::compat::Device* device, gpu::compat::TextureHandle texture, uint base_mip, uint mip_count) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(*device);
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    foreach (&cached : slot.views) {
        if (cached.used && cached.base_mip == base_mip && cached.mip_count == mip_count
            && cached.base_layer == 0 && cached.layer_count == 1) return true;
    }
    return false;
}

// Never actually dispatched, so the shader's ABI content is unused.
struct ValComputeCtx {
    gpu::compat::ShaderHandle   shader;
    gpu::compat::PipelineHandle pipeline;
}

fn ValComputeCtx make_val_compute_ctx(gpu::compat::Device* device) {
    ValComputeCtx ctx;
    gpu::compat::ShaderDesc comp_sd = { .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = VAL_COMP_SPIRV[..], .entry_point = "main" };
    ctx.shader = gpu::compat::create_shader(device, &comp_sd)!!;
    gpu::compat::ComputePipelineDesc comp_desc = { .shader = ctx.shader, .push_constant_size = gpu::compat::RootPush::size };
    ctx.pipeline = gpu::compat::create_compute_pipeline(device, &comp_desc)!!;
    return ctx;
}

fn void destroy_val_compute_ctx(gpu::compat::Device* device, ValComputeCtx* ctx) {
    gpu::compat::destroy_pipeline(device, ctx.pipeline)!!;
    gpu::compat::destroy_shader(device, ctx.shader)!!;
}

fn void record_val_texture_transition(gpu::compat::CommandList* cmd, gpu::compat::TextureHandle texture, gpu::compat::TextureLayout new_layout, gpu::compat::Stage after_stage, gpu::compat::Hazard after_hazard) {
    gpu::compat::TextureBarrier barrier = {
        .texture       = texture,
        .before_stage  = gpu::compat::Stage.HOST,
        .after_stage   = after_stage,
        .before_hazard = gpu::compat::Hazard.HOST_WRITE,
        .after_hazard  = after_hazard,
        .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
        .new_layout    = new_layout,
    };
    gpu::compat::cmd_texture_barrier(cmd, &barrier)!!;
}

fn void write_val_triangle_verts(gpu::compat::GpuSpan* span) {
    float* verts = (float*)span.cpu;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = vertex_data[i];
}

fn void record_val_target_to_attach(gpu::compat::CommandList* cmd, gpu::compat::TextureHandle target) {
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

fn void mixed_queue_submit_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList gfx = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_commands(&gfx)!!;
    gpu::compat::CommandList xfer = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.TRANSFER)!!;
    gpu::compat::end_commands(&xfer)!!;

    gpu::compat::CommandList[2] lists = { gfx, xfer };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    fault mixed = @catch(gpu::compat::submit(&device, &submit));
    assert(mixed == gpu::compat::INVALID_ARGUMENT, "mixed-queue submit faults INVALID_ARGUMENT before any queue work runs");
    gpu::compat::end_frame(&frame_token)!!;
}

fn void copy_without_transfer_usage_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc plain_desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle plain = gpu::compat::create_buffer(&device, &plain_desc)!!;
    defer gpu::compat::destroy_buffer(&device, plain)!!;
    gpu::compat::BufferDesc src_desc = { .size = 64, .usage = { .transfer_src }, .memory_kind = gpu::compat::MemoryKind.STAGING };
    gpu::compat::BufferHandle src = gpu::compat::create_buffer(&device, &src_desc)!!;
    defer gpu::compat::destroy_buffer(&device, src)!!;
    gpu::compat::BufferDesc dst_desc = { .size = 64, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.READBACK };
    gpu::compat::BufferHandle dst = gpu::compat::create_buffer(&device, &dst_desc)!!;
    defer gpu::compat::destroy_buffer(&device, dst)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.TRANSFER)!!;

    gpu::compat::BufferCopyDesc no_src_usage_copy = { .src = plain, .dst = dst, .size = 64 };
    fault no_src_usage = @catch(gpu::compat::cmd_copy_buffer(&cmd, &no_src_usage_copy));
    assert(no_src_usage == gpu::compat::INVALID_ARGUMENT, "copy from a buffer without transfer_src faults INVALID_ARGUMENT");

    gpu::compat::BufferCopyDesc no_dst_usage_copy = { .src = src, .dst = plain, .size = 64 };
    fault no_dst_usage = @catch(gpu::compat::cmd_copy_buffer(&cmd, &no_dst_usage_copy));
    assert(no_dst_usage == gpu::compat::INVALID_ARGUMENT, "copy to a buffer without transfer_dst faults INVALID_ARGUMENT");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void texture_copy_without_buffer_transfer_usage_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::TextureDesc tex_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_src, .transfer_dst },
    };
    gpu::compat::TextureHandle texture = gpu::compat::create_texture(&device, &tex_desc)!!;
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::BufferDesc plain_desc = { .size = 256, .usage = { .storage, .addressable }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle plain = gpu::compat::create_buffer(&device, &plain_desc)!!;
    defer gpu::compat::destroy_buffer(&device, plain)!!;
    gpu::compat::GpuSpan plain_span = gpu::compat::get_buffer_span(&device, plain)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.TRANSFER)!!;

    // Buffer usage is checked before the texture's tracked layout, so no
    // barrier is needed to reach either fault.
    gpu::compat::BufferTextureCopyDesc up = { .src = plain_span, .texture = texture, .width = 4, .height = 4 };
    fault no_src_usage = @catch(gpu::compat::cmd_copy_buffer_to_texture(&cmd, &up));
    assert(no_src_usage == gpu::compat::INVALID_ARGUMENT, "buffer-to-texture copy without buffer transfer_src faults INVALID_ARGUMENT");

    gpu::compat::TextureBufferCopyDesc down = { .texture = texture, .dst = plain_span, .width = 4, .height = 4 };
    fault no_dst_usage = @catch(gpu::compat::cmd_copy_texture_to_buffer(&cmd, &down));
    assert(no_dst_usage == gpu::compat::INVALID_ARGUMENT, "texture-to-buffer copy without buffer transfer_dst faults INVALID_ARGUMENT");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void fill_without_transfer_dst_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &desc)!!;
    defer gpu::compat::destroy_buffer(&device, buffer)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.TRANSFER)!!;
    fault no_usage = @catch(gpu::compat::cmd_fill_buffer(&cmd, buffer, 0, 64, 0));
    assert(no_usage == gpu::compat::INVALID_ARGUMENT, "fill on a buffer without transfer_dst faults INVALID_ARGUMENT");
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void indexed_draw_without_index_usage_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::compat::TextureHandle target = make_val_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    // Storage-only: no index usage.
    gpu::compat::BufferDesc bad_index_desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle bad_index_buffer = gpu::compat::create_buffer(&device, &bad_index_desc)!!;
    defer gpu::compat::destroy_buffer(&device, bad_index_buffer)!!;
    gpu::compat::GpuSpan bad_index_span = gpu::compat::get_buffer_span(&device, bad_index_buffer)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan verts_span = gpu::compat::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
    write_val_triangle_verts(&verts_span);
    gpu::compat::GpuSpan frag_span = gpu::compat::alloc_frame_span(&frame_token, 16, 16)!!;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::compat::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;

    fault no_index_usage = @catch(gpu::compat::cmd_draw_indexed(
        commands:      &cmd,
        pipeline:      ctx.graphics,
        vertex_root:   verts_span.gpu,
        fragment_root: frag_span.gpu,
        index_span:    bad_index_span,
        index_count:   3,
        instance_count: 1,
    ));
    assert(no_index_usage == gpu::compat::INVALID_ARGUMENT, "indexed draw over a non-index buffer faults INVALID_ARGUMENT");

    gpu::compat::cmd_end_render_pass(&cmd)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void indexed_draw_misaligned_offset_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::compat::TextureHandle target = make_val_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan verts_span = gpu::compat::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
    write_val_triangle_verts(&verts_span);
    gpu::compat::GpuSpan frag_span = gpu::compat::alloc_frame_span(&frame_token, 16, 16)!!;
    // Frame-arena buffers carry .index usage; a 2-byte offset still misaligns
    // the default U32 index type (needs 4-byte alignment).
    gpu::compat::GpuSpan index_span = gpu::compat::alloc_frame_span(&frame_token, 16, 16)!!;
    gpu::compat::GpuSpan misaligned = index_span.checked_subspan(2, 8)!!;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::compat::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;

    fault misaligned_fault = @catch(gpu::compat::cmd_draw_indexed(
        commands:      &cmd,
        pipeline:      ctx.graphics,
        vertex_root:   verts_span.gpu,
        fragment_root: frag_span.gpu,
        index_span:    misaligned,
        index_count:   1,
        instance_count: 1,
    ));
    assert(misaligned_fault == gpu::compat::INVALID_ARGUMENT, "misaligned index offset faults INVALID_ARGUMENT");

    gpu::compat::cmd_end_render_pass(&cmd)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void dispatch_with_graphics_pipeline_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan root_span = gpu::compat::alloc_frame_span(&frame_token, 16, 16)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    fault wrong_kind = @catch(gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: ctx.graphics,
        root:     root_span.gpu,
        groups:   { 1, 1, 1 },
    ));
    assert(wrong_kind == gpu::compat::INVALID_ARGUMENT, "dispatch with a graphics pipeline faults INVALID_ARGUMENT");
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void draw_with_compute_pipeline_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle target = make_val_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::ShaderDesc comp_sd = { .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = VAL_COMP_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle comp_shader = gpu::compat::create_shader(&device, &comp_sd)!!;
    defer gpu::compat::destroy_shader(&device, comp_shader)!!;
    gpu::compat::ComputePipelineDesc comp_desc = { .shader = comp_shader, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle compute = gpu::compat::create_compute_pipeline(&device, &comp_desc)!!;
    defer gpu::compat::destroy_pipeline(&device, compute)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::compat::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;

    fault wrong_kind = @catch(gpu::compat::cmd_draw(
        commands:      &cmd,
        pipeline:      compute,
        vertex_root:   (gpu::compat::GpuAddress)0,
        fragment_root: (gpu::compat::GpuAddress)0,
        vertex_count:  3,
        instance_count: 1,
    ));
    assert(wrong_kind == gpu::compat::INVALID_ARGUMENT, "draw with a compute pipeline faults INVALID_ARGUMENT");

    gpu::compat::cmd_end_render_pass(&cmd)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void compute_pipeline_from_vertex_shader_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::ShaderDesc vert_sd = { .stage = gpu::compat::ShaderStage.VERTEX, .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle vert_shader = gpu::compat::create_shader(&device, &vert_sd)!!;
    defer gpu::compat::destroy_shader(&device, vert_shader)!!;

    gpu::compat::ComputePipelineDesc desc = { .shader = vert_shader, .push_constant_size = gpu::compat::RootPush::size };
    fault wrong_stage = @catch(gpu::compat::create_compute_pipeline(&device, &desc));
    assert(wrong_stage == gpu::compat::INVALID_ARGUMENT, "compute pipeline from a vertex-stage shader faults INVALID_ARGUMENT");
}

fn void graphics_pipeline_from_swapped_shader_stages_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::ShaderDesc vert_sd = { .stage = gpu::compat::ShaderStage.VERTEX, .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle vert_shader = gpu::compat::create_shader(&device, &vert_sd)!!;
    defer gpu::compat::destroy_shader(&device, vert_shader)!!;
    gpu::compat::ShaderDesc frag_sd = { .stage = gpu::compat::ShaderStage.FRAGMENT, .spirv = VAL_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle frag_shader = gpu::compat::create_shader(&device, &frag_sd)!!;
    defer gpu::compat::destroy_shader(&device, frag_shader)!!;

    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc swapped = {
        .vertex_shader   = frag_shader,
        .fragment_shader = vert_shader,
        .topology        = gpu::compat::PrimitiveTopology.TRIANGLES,
        .color_formats   = formats[..],
    };
    fault wrong_stage = @catch(gpu::compat::create_graphics_pipeline(&device, &swapped));
    assert(wrong_stage == gpu::compat::INVALID_ARGUMENT, "graphics pipeline with swapped vertex/fragment shaders faults INVALID_ARGUMENT");
}

fn void valid_copy_fill_and_draw_succeed() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::compat::TextureHandle target = make_val_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::BufferDesc up_desc = { .size = 64, .usage = { .transfer_src }, .memory_kind = gpu::compat::MemoryKind.STAGING };
    gpu::compat::BufferHandle upload = gpu::compat::create_buffer(&device, &up_desc)!!;
    defer gpu::compat::destroy_buffer(&device, upload)!!;
    gpu::compat::BufferDesc rb_desc = { .size = 64, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.READBACK };
    gpu::compat::BufferHandle readback = gpu::compat::create_buffer(&device, &rb_desc)!!;
    defer gpu::compat::destroy_buffer(&device, readback)!!;
    gpu::compat::BufferDesc fill_desc = { .size = 64, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle fill_target = gpu::compat::create_buffer(&device, &fill_desc)!!;
    defer gpu::compat::destroy_buffer(&device, fill_target)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan verts_span = gpu::compat::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
    write_val_triangle_verts(&verts_span);
    gpu::compat::GpuSpan frag_span = gpu::compat::alloc_frame_span(&frame_token, FragRoot::size, 16)!!;
    FragRoot* frag_root = (FragRoot*)frag_span.cpu;
    frag_root.texture_index = ctx.tex_index;
    frag_root.sampler_index = ctx.smp_index;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::BufferCopyDesc copy = { .src = upload, .dst = readback, .size = 64 };
    gpu::compat::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::compat::cmd_fill_buffer(&cmd, fill_target, 0, 64, 0)!!;

    record_val_target_to_attach(&cmd, target);
    gpu::compat::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
        .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::compat::cmd_draw(&cmd, ctx.graphics, verts_span.gpu, frag_span.gpu, 3, 1)!!;
    gpu::compat::cmd_end_render_pass(&cmd)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

// Backing-range faults: subspan builds escaping spans without faulting;
// the fault lands at the consuming command.

fn void dispatch_indirect_args_span_beyond_buffer_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    ValComputeCtx ctx = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &ctx);

    gpu::compat::BufferDesc args_desc = { .size = 16, .usage = { .storage, .addressable, .indirect }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle args_buffer = gpu::compat::create_buffer(&device, &args_desc)!!;
    defer gpu::compat::destroy_buffer(&device, args_buffer)!!;
    gpu::compat::GpuSpan full_span = gpu::compat::get_buffer_span(&device, args_buffer)!!;
    gpu::compat::GpuSpan escaped = full_span.unchecked_subspan(8, gpu::compat::DispatchIndirectCommand::size);

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    fault beyond = @catch(gpu::compat::cmd_dispatch_indirect(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::compat::GpuAddress)0,
        args:     escaped,
    ));
    assert(beyond == gpu::compat::INVALID_ARGUMENT, "dispatch indirect args beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void dispatch_indirect_args_span_boundary_exact_succeeds() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    ValComputeCtx ctx = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &ctx);

    gpu::compat::BufferDesc args_desc = { .size = 16, .usage = { .storage, .addressable, .indirect }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle args_buffer = gpu::compat::create_buffer(&device, &args_desc)!!;
    defer gpu::compat::destroy_buffer(&device, args_buffer)!!;
    gpu::compat::GpuSpan full_span = gpu::compat::get_buffer_span(&device, args_buffer)!!;
    gpu::compat::GpuSpan boundary = full_span.checked_subspan(4, gpu::compat::DispatchIndirectCommand::size)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::cmd_dispatch_indirect(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::compat::GpuAddress)0,
        args:     boundary,
    )!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void draw_indexed_indirect_count_span_beyond_buffer_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    if (!gpu::compat::get_device_caps(&device)!!.draw_indirect_count) return;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::compat::TextureHandle target = make_val_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::BufferDesc args_desc = { .size = gpu::compat::DrawIndexedIndirectCommand::size * 2, .usage = { .storage, .addressable, .indirect }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle args_buffer = gpu::compat::create_buffer(&device, &args_desc)!!;
    defer gpu::compat::destroy_buffer(&device, args_buffer)!!;
    gpu::compat::GpuSpan args_span = gpu::compat::get_buffer_span(&device, args_buffer)!!;

    gpu::compat::BufferDesc count_desc = { .size = 8, .usage = { .storage, .addressable, .indirect }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle count_buffer = gpu::compat::create_buffer(&device, &count_desc)!!;
    defer gpu::compat::destroy_buffer(&device, count_buffer)!!;
    gpu::compat::GpuSpan full_count_span = gpu::compat::get_buffer_span(&device, count_buffer)!!;
    gpu::compat::GpuSpan escaped_count = full_count_span.unchecked_subspan(8, 4);

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::compat::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;

    gpu::compat::GpuSpan index_span_stub = {};
    fault beyond = @catch(gpu::compat::cmd_draw_indexed_indirect_count(
        commands:       &cmd,
        pipeline:       ctx.graphics,
        vertex_root:    (gpu::compat::GpuAddress)0,
        fragment_root:  (gpu::compat::GpuAddress)0,
        args:           args_span,
        count_span:     escaped_count,
        max_draw_count: 1,
        index_span:     index_span_stub,
    ));
    assert(beyond == gpu::compat::INVALID_ARGUMENT, "indirect count span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::compat::cmd_end_render_pass(&cmd)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void indirect_draw_count_limits_follow_command_contract() @test {
    SyncDiagnosticCapture capture = {};
    gpu::compat::Device device = make_sync_diagnostic_device(&capture);
    defer gpu::compat::destroy_device(&device)!!;
    if (!gpu::compat::get_device_caps(&device)!!.draw_indirect_count) return;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::compat::TextureHandle target = make_val_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::BufferDesc args_desc = {
        .size        = gpu::compat::DrawIndexedIndirectCommand::size * 2,
        .usage       = { .indirect },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
    };
    gpu::compat::BufferHandle args_buffer = gpu::compat::create_buffer(&device, &args_desc)!!;
    defer gpu::compat::destroy_buffer(&device, args_buffer)!!;
    gpu::compat::GpuSpan args_span = gpu::compat::get_buffer_span(&device, args_buffer)!!;

    gpu::compat::BufferDesc count_desc = {
        .size        = 4,
        .usage       = { .indirect },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
    };
    gpu::compat::BufferHandle count_buffer = gpu::compat::create_buffer(&device, &count_desc)!!;
    defer gpu::compat::destroy_buffer(&device, count_buffer)!!;
    gpu::compat::GpuSpan count_span = gpu::compat::get_buffer_span(&device, count_buffer)!!;

    gpu::compat::BufferDesc index_desc = {
        .size        = 4,
        .usage       = { .index },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
    };
    gpu::compat::BufferHandle index_buffer = gpu::compat::create_buffer(&device, &index_desc)!!;
    defer gpu::compat::destroy_buffer(&device, index_buffer)!!;
    gpu::compat::GpuSpan index_span = gpu::compat::get_buffer_span(&device, index_buffer)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::compat::ColorTargetDesc[1] colors = {{
        .texture  = target,
        .load_op  = gpu::compat::LoadOp.CLEAR,
        .store_op = gpu::compat::StoreOp.STORE,
    }};
    gpu::compat::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    uint original_limit = state.caps.max_draw_indirect_count;
    state.caps.max_draw_indirect_count = 1;
    defer state.caps.max_draw_indirect_count = original_limit;

    gpu::compat::cmd_draw_indirect(
        commands:      &cmd,
        pipeline:      ctx.graphics,
        vertex_root:   (gpu::compat::GpuAddress)0,
        fragment_root: (gpu::compat::GpuAddress)0,
        args:          args_span,
        draw_count:    1,
    )!!;

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_draw_indirect",
        debug_name:     null,
        rejected_field: "draw_count",
        invariant:      "count must not exceed DeviceCaps.max_draw_indirect_count",
    );
    fault draw_over = @catch(gpu::compat::cmd_draw_indirect(
        commands:      &cmd,
        pipeline:      ctx.graphics,
        vertex_root:   (gpu::compat::GpuAddress)0,
        fragment_root: (gpu::compat::GpuAddress)0,
        args:          args_span,
        draw_count:    2,
    ));
    assert(draw_over == gpu::compat::INVALID_ARGUMENT,
        "direct indirect draw count above the selected-device limit faults before recording");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      draw_over,
        resource_kind:       gpu::compat::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
    );

    gpu::compat::cmd_draw_indexed_indirect_count(
        commands:       &cmd,
        pipeline:       ctx.graphics,
        vertex_root:    (gpu::compat::GpuAddress)0,
        fragment_root:  (gpu::compat::GpuAddress)0,
        args:           args_span,
        count_span:     count_span,
        max_draw_count: 2,
        index_span:     index_span,
    )!!;

    gpu::compat::cmd_end_render_pass(&cmd)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void draw_indexed_index_span_beyond_buffer_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::compat::TextureHandle target = make_val_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::BufferDesc index_desc = { .size = 16, .usage = { .index }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle index_buffer = gpu::compat::create_buffer(&device, &index_desc)!!;
    defer gpu::compat::destroy_buffer(&device, index_buffer)!!;
    gpu::compat::GpuSpan full_index_span = gpu::compat::get_buffer_span(&device, index_buffer)!!;
    gpu::compat::GpuSpan escaped = full_index_span.unchecked_subspan(16, 4);

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::compat::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;

    fault beyond = @catch(gpu::compat::cmd_draw_indexed(
        commands:       &cmd,
        pipeline:       ctx.graphics,
        vertex_root:    (gpu::compat::GpuAddress)0,
        fragment_root:  (gpu::compat::GpuAddress)0,
        index_span:     escaped,
        index_count:    1,
        instance_count: 1,
    ));
    assert(beyond == gpu::compat::INVALID_ARGUMENT, "index span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::compat::cmd_end_render_pass(&cmd)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void draw_indexed_index_span_boundary_exact_succeeds() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::compat::TextureHandle target = make_val_target(&device);
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::BufferDesc index_desc = { .size = 16, .usage = { .index }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle index_buffer = gpu::compat::create_buffer(&device, &index_desc)!!;
    defer gpu::compat::destroy_buffer(&device, index_buffer)!!;
    gpu::compat::GpuSpan full_index_span = gpu::compat::get_buffer_span(&device, index_buffer)!!;
    gpu::compat::GpuSpan boundary = full_index_span.checked_subspan(12, 4)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::compat::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;

    gpu::compat::cmd_draw_indexed(
        commands:       &cmd,
        pipeline:       ctx.graphics,
        vertex_root:    (gpu::compat::GpuAddress)0,
        fragment_root:  (gpu::compat::GpuAddress)0,
        index_span:     boundary,
        index_count:    1,
        instance_count: 1,
    )!!;

    gpu::compat::cmd_end_render_pass(&cmd)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void copy_buffer_to_texture_span_beyond_buffer_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::TextureDesc tex_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::compat::TextureHandle texture = gpu::compat::create_texture(&device, &tex_desc)!!;
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::BufferDesc src_desc = { .size = 64, .usage = { .transfer_src }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle src_buffer = gpu::compat::create_buffer(&device, &src_desc)!!;
    defer gpu::compat::destroy_buffer(&device, src_buffer)!!;
    gpu::compat::GpuSpan full_span = gpu::compat::get_buffer_span(&device, src_buffer)!!;
    gpu::compat::GpuSpan escaped = full_span.unchecked_subspan(32, 64);

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.TRANSFER)!!;
    record_val_texture_transition(&cmd, texture, gpu::compat::TextureLayout.TRANSFER_DST, gpu::compat::Stage.TRANSFER, gpu::compat::Hazard.TRANSFER_WRITE);

    gpu::compat::BufferTextureCopyDesc up = { .src = escaped, .texture = texture, .width = 4, .height = 4 };
    fault beyond = @catch(gpu::compat::cmd_copy_buffer_to_texture(&cmd, &up));
    assert(beyond == gpu::compat::INVALID_ARGUMENT, "buffer-to-texture copy span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void copy_texture_to_buffer_span_beyond_buffer_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::TextureDesc tex_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .transfer_src },
    };
    gpu::compat::TextureHandle texture = gpu::compat::create_texture(&device, &tex_desc)!!;
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::BufferDesc dst_desc = { .size = 64, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle dst_buffer = gpu::compat::create_buffer(&device, &dst_desc)!!;
    defer gpu::compat::destroy_buffer(&device, dst_buffer)!!;
    gpu::compat::GpuSpan full_span = gpu::compat::get_buffer_span(&device, dst_buffer)!!;
    gpu::compat::GpuSpan escaped = full_span.unchecked_subspan(32, 64);

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.TRANSFER)!!;
    record_val_texture_transition(&cmd, texture, gpu::compat::TextureLayout.TRANSFER_SRC, gpu::compat::Stage.TRANSFER, gpu::compat::Hazard.TRANSFER_READ);

    gpu::compat::TextureBufferCopyDesc down = { .texture = texture, .dst = escaped, .width = 4, .height = 4 };
    fault beyond = @catch(gpu::compat::cmd_copy_texture_to_buffer(&cmd, &down));
    assert(beyond == gpu::compat::INVALID_ARGUMENT, "texture-to-buffer copy span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void texture_copy_span_boundary_exact_succeeds() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::TextureDesc tex_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::compat::TextureHandle texture = gpu::compat::create_texture(&device, &tex_desc)!!;
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::BufferDesc src_desc = { .size = 96, .usage = { .transfer_src }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle src_buffer = gpu::compat::create_buffer(&device, &src_desc)!!;
    defer gpu::compat::destroy_buffer(&device, src_buffer)!!;
    gpu::compat::GpuSpan full_span = gpu::compat::get_buffer_span(&device, src_buffer)!!;
    gpu::compat::GpuSpan boundary = full_span.checked_subspan(32, 64)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.TRANSFER)!!;
    record_val_texture_transition(&cmd, texture, gpu::compat::TextureLayout.TRANSFER_DST, gpu::compat::Stage.TRANSFER, gpu::compat::Hazard.TRANSFER_WRITE);

    gpu::compat::BufferTextureCopyDesc up = { .src = boundary, .texture = texture, .width = 4, .height = 4 };
    gpu::compat::cmd_copy_buffer_to_texture(&cmd, &up)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void buffer_barrier_range_beyond_buffer_faults() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &desc)!!;
    defer gpu::compat::destroy_buffer(&device, buffer)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::BufferBarrier barrier = {
        .buffer = buffer, .offset = 48, .size = 32,
        .before_stage = gpu::compat::Stage.COMPUTE_SHADER, .after_stage = gpu::compat::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE, .after_hazard = gpu::compat::Hazard.SHADER_READ,
    };
    fault beyond = @catch(gpu::compat::cmd_buffer_barrier(&cmd, &barrier));
    assert(beyond == gpu::compat::INVALID_ARGUMENT, "buffer barrier range beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void buffer_barrier_boundary_exact_succeeds() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &desc)!!;
    defer gpu::compat::destroy_buffer(&device, buffer)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::BufferBarrier barrier = {
        .buffer = buffer, .offset = 32, .size = 32,
        .before_stage = gpu::compat::Stage.COMPUTE_SHADER, .after_stage = gpu::compat::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE, .after_hazard = gpu::compat::Hazard.SHADER_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &barrier)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void buffer_barrier_zero_size_whole_buffer_succeeds() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &desc)!!;
    defer gpu::compat::destroy_buffer(&device, buffer)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::BufferBarrier barrier = {
        .buffer = buffer, .offset = 0, .size = 0,
        .before_stage = gpu::compat::Stage.COMPUTE_SHADER, .after_stage = gpu::compat::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE, .after_hazard = gpu::compat::Hazard.SHADER_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &barrier)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void render_pass_attachment_extents_validate_before_recording() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle large = make_val_color_target(
        device:     &device,
        width:      16,
        height:     16,
        mip_levels: 5,
    );
    defer gpu::compat::destroy_texture(&device, large)!!;
    gpu::compat::TextureHandle small = make_val_color_target(
        device:     &device,
        width:      8,
        height:     8,
        mip_levels: 1,
    );
    defer gpu::compat::destroy_texture(&device, small)!!;
    gpu::compat::TextureHandle unbarriered = make_val_color_target(
        device:     &device,
        width:      16,
        height:     16,
        mip_levels: 5,
    );
    defer gpu::compat::destroy_texture(&device, unbarriered)!!;
    gpu::compat::TextureHandle depth = make_val_depth_target(&device, 8, 8);
    defer gpu::compat::destroy_texture(&device, depth)!!;
    gpu::compat::TextureHandle mip_depth = make_val_depth_target_mips(
        device:     &device,
        width:      8,
        height:     8,
        mip_levels: 4,
    );
    defer gpu::compat::destroy_texture(&device, mip_depth)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, large);
    record_val_target_to_attach(&cmd, small);
    record_val_depth_to_attach(&cmd, depth);
    record_val_depth_to_attach(&cmd, mip_depth);

    gpu::compat::ColorTargetDesc[1] large_color = { { .texture = large } };
    gpu::compat::RenderPassDesc oversized = { .colors = large_color[..], .width = 17, .height = 16 };
    assert(@catch(gpu::compat::cmd_begin_render_pass(&cmd, &oversized)) == gpu::compat::INVALID_ARGUMENT,
        "render area wider than mip zero faults INVALID_ARGUMENT");
    assert(@catch(gpu::compat::cmd_end_render_pass(&cmd)) == gpu::compat::COMMAND_RECORDING_ERROR, "oversized color target records no pass");

    gpu::compat::ColorTargetDesc[2] mixed_colors = {
        { .texture = large },
        { .texture = small },
    };
    gpu::compat::RenderPassDesc mixed = { .colors = mixed_colors[..], .width = 16, .height = 16 };
    assert(@catch(gpu::compat::cmd_begin_render_pass(&cmd, &mixed)) == gpu::compat::INVALID_ARGUMENT,
        "one mixed-size attachment that cannot cover the pass faults INVALID_ARGUMENT");
    assert(@catch(gpu::compat::cmd_end_render_pass(&cmd)) == gpu::compat::COMMAND_RECORDING_ERROR, "mixed-size rejection records no pass");

    uint cached_before = val_cached_view_count(&device, large);
    gpu::compat::ColorTargetDesc[2] invalid_later = {
        { .texture = large, .mip_level = 2 },
        { .texture = unbarriered, .mip_level = 99 },
    };
    gpu::compat::RenderPassDesc invalid_later_pass = { .colors = invalid_later[..], .width = 4, .height = 4 };
    assert(@catch(gpu::compat::cmd_begin_render_pass(&cmd, &invalid_later_pass)) == gpu::compat::INVALID_ARGUMENT,
        "invalid later mip range faults before its invalid layout");
    assert(val_cached_view_count(&device, large) == cached_before,
        "later validation failure creates no cached view for an earlier target");
    assert(@catch(gpu::compat::cmd_end_render_pass(&cmd)) == gpu::compat::COMMAND_RECORDING_ERROR, "later attachment rejection records no pass");

    gpu::compat::ColorTargetDesc[1] mip_color = { { .texture = large, .mip_level = 1 } };
    gpu::compat::RenderPassDesc mip_oversized = { .colors = mip_color[..], .width = 9, .height = 8 };
    assert(@catch(gpu::compat::cmd_begin_render_pass(&cmd, &mip_oversized)) == gpu::compat::INVALID_ARGUMENT,
        "render area wider than nonzero mip faults INVALID_ARGUMENT");

    gpu::compat::RenderPassDesc mip_exact = { .colors = mip_color[..], .width = 8, .height = 8 };
    gpu::compat::cmd_begin_render_pass(&cmd, &mip_exact)!!;
    gpu::compat::cmd_end_render_pass(&cmd)!!;

    gpu::compat::DepthTargetDesc depth_target = { .texture = depth };
    gpu::compat::RenderPassDesc depth_oversized = { .depth = &depth_target, .width = 9, .height = 8 };
    assert(@catch(gpu::compat::cmd_begin_render_pass(&cmd, &depth_oversized)) == gpu::compat::INVALID_ARGUMENT,
        "render area wider than depth mip zero faults INVALID_ARGUMENT");
    assert(@catch(gpu::compat::cmd_end_render_pass(&cmd)) == gpu::compat::COMMAND_RECORDING_ERROR, "oversized depth target records no pass");

    gpu::compat::DepthTargetDesc mip_depth_target = { .texture = mip_depth };
    uint mip_depth_cached_before = val_cached_view_count(&device, mip_depth);
    gpu::compat::RenderPassDesc mip_depth_pass = { .depth = &mip_depth_target, .width = 8, .height = 8 };
    gpu::compat::cmd_begin_render_pass(&cmd, &mip_depth_pass)!!;
    assert(val_cached_view_count(&device, mip_depth) == mip_depth_cached_before + 1,
        "multi-mip depth attachment resolves a single-mip cached view");
    gpu::compat::cmd_end_render_pass(&cmd)!!;
    assert(val_has_cached_view(
        device:    &device,
        texture:   mip_depth,
        base_mip:  0,
        mip_count: 1,
    ), "multi-mip depth attachment view selects exactly mip zero and layer zero");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void render_pass_color_count_limits_validate_before_lookup() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::ColorTargetDesc[gpu::compat::MAX_COLOR_ATTACHMENTS + 1] too_many;
    gpu::compat::RenderPassDesc library_limit = { .colors = too_many[..], .width = 1, .height = 1 };
    assert(@catch(gpu::compat::cmd_begin_render_pass(&cmd, &library_limit)) == gpu::compat::INVALID_ARGUMENT,
        "color count above MAX_COLOR_ATTACHMENTS faults before invalid handle lookup");
    assert(@catch(gpu::compat::cmd_end_render_pass(&cmd)) == gpu::compat::COMMAND_RECORDING_ERROR, "library-limit rejection records no pass");

    uint device_limit = state.caps.max_color_attachments;
    state.caps.max_color_attachments = 1;
    gpu::compat::ColorTargetDesc[2] device_excess;
    gpu::compat::RenderPassDesc selected_limit = { .colors = device_excess[..], .width = 1, .height = 1 };
    assert(@catch(gpu::compat::cmd_begin_render_pass(&cmd, &selected_limit)) == gpu::compat::INVALID_ARGUMENT,
        "color count above selected device limit faults before invalid handle lookup");
    assert(@catch(gpu::compat::cmd_end_render_pass(&cmd)) == gpu::compat::COMMAND_RECORDING_ERROR, "device-limit rejection records no pass");
    state.caps.max_color_attachments = device_limit;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void selected_device_dispatch_limit_validates_before_recording() @test {
    SyncDiagnosticCapture capture = {};
    gpu::compat::Device device = make_sync_diagnostic_device(&capture);
    defer gpu::compat::destroy_device(&device)!!;
    ValComputeCtx ctx = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &ctx);

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    gpu::compat::DeviceCaps original_caps = state.caps;
    gpu::compat::DeviceCaps limited_caps = state.caps;
    limited_caps.max_compute_work_group_count = { .x = 1, .y = 1, .z = 1 };
    state.caps = limited_caps;
    defer state.caps = original_caps;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::compat::GpuAddress)0,
        groups:   {},
    )!!;
    gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::compat::GpuAddress)0,
        groups:   limited_caps.max_compute_work_group_count,
    )!!;

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_dispatch",
        debug_name:     null,
        rejected_field: "groups",
        invariant:      "group counts must not exceed DeviceCaps.max_compute_work_group_count",
    );
    fault dispatch_over = @catch(gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::compat::GpuAddress)0,
        groups:   { .x = 2, .y = 1, .z = 1 },
    ));
    assert(dispatch_over == gpu::compat::INVALID_ARGUMENT,
        "dispatch above the selected-device count limit faults before recording");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      dispatch_over,
        resource_kind:       gpu::compat::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
    );

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void render_pass_view_cache_rolls_back_on_later_resolution_fault() @test {
    gpu::compat::Device device = make_val_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle first = make_val_color_target(
        device:     &device,
        width:      16,
        height:     16,
        mip_levels: 5,
    );
    defer gpu::compat::destroy_texture(&device, first)!!;
    gpu::compat::TextureHandle saturated = make_val_color_target(
        device:     &device,
        width:      16,
        height:     16,
        mip_levels: 5,
    );
    defer gpu::compat::destroy_texture(&device, saturated)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    SyncDiagnosticCapture capture;
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_begin_render_pass",
        debug_name:     "(unnamed)",
        rejected_field: "texture view cache",
        invariant:      "texture must have free cached-view capacity",
    );
    state.debug_callback = &capture_sync_diagnostic;
    state.debug_user_data = &capture;
    vk::TextureSlot* saturated_slot = &state.textures.slots[saturated.index].value;
    foreach (&cached : saturated_slot.views) cached.used = true;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, first);
    record_val_target_to_attach(&cmd, saturated);

    uint first_cached_before = val_cached_view_count(&device, first);
    gpu::compat::ColorTargetDesc[2] colors = {
        { .texture = first, .mip_level = 1 },
        { .texture = saturated, .mip_level = 1 },
    };
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = 8, .height = 8 };
    fault view_fault = @catch(gpu::compat::cmd_begin_render_pass(&cmd, &pass));
    state.debug_callback = null;
    foreach (&cached : saturated_slot.views) cached.used = false;

    assert(view_fault == gpu::compat::SLOT_TABLE_FULL, "later saturated view cache faults SLOT_TABLE_FULL");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      view_fault,
        resource_kind:       gpu::compat::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               saturated.index,
        generation:          saturated.generation,
    );
    assert(val_cached_view_count(&device, first) == first_cached_before,
        "later view resolution fault rolls back the earlier cached view");
    assert(@catch(gpu::compat::cmd_end_render_pass(&cmd)) == gpu::compat::COMMAND_RECORDING_ERROR, "view resolution fault records no pass");

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void buffer_barrier_range_diagnostic_identifies_resource() @test {
    SyncDiagnosticCapture capture = {
        .expected_operation      = "cmd_buffer_barrier",
        .expected_debug_name     = "barrier_diagnostic_buffer",
        .expected_rejected_field = "BufferBarrier.size",
        .expected_invariant      = "offset and size must fit the buffer",
    };
    gpu::compat::Device device = make_sync_diagnostic_device(&capture);
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc desc = {
        .size        = 64,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::compat::MemoryKind.DEVICE,
        .debug_name  = "barrier_diagnostic_buffer",
    };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &desc)!!;
    defer gpu::compat::destroy_buffer(&device, buffer)!!;

    gpu::compat::FrameToken frame = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList commands = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::BufferBarrier barrier = {
        .buffer        = buffer,
        .offset        = 48,
        .size          = 32,
        .before_stage  = gpu::compat::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::compat::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::compat::Hazard.SHADER_READ,
    };
    fault with_callback = @catch(gpu::compat::cmd_buffer_barrier(&commands, &barrier));

    assert(with_callback == gpu::compat::INVALID_ARGUMENT, "barrier range keeps INVALID_ARGUMENT");
    assert(capture.calls == 1, "barrier range emits exactly one diagnostic");
    assert(capture.operation_matches, "operation is cmd_buffer_barrier");
    assert(capture.has_fault && capture.public_fault == with_callback,
        "diagnostic carries the unchanged barrier fault");
    assert(capture.public_contract, "barrier range is a public-contract diagnostic");
    assert(capture.resource_kind == gpu::compat::DebugResourceKind.BUFFER, "resource kind is buffer");
    assert(capture.has_public_identity, "live buffer identity is present");
    assert(capture.index == buffer.index, "buffer index is preserved");
    assert(capture.generation == buffer.generation, "buffer generation is preserved");
    assert(capture.debug_name_matches, "buffer debug name is preserved");
    assert(capture.rejected_field_matches, "barrier size field is identified");
    assert(capture.invariant_matches, "barrier range invariant is identified");

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    state.debug_callback = null;
    fault without_callback = @catch(gpu::compat::cmd_buffer_barrier(&commands, &barrier));
    assert(without_callback == with_callback, "callback presence does not alter the barrier fault");
    assert(capture.calls == 1, "null callback emits no additional diagnostic");

    state.debug_callback = &capture_sync_diagnostic;
    capture.expected_rejected_field = "BufferBarrier.offset";
    capture.expected_invariant = "whole-buffer barrier offset must lie inside the buffer";
    barrier.offset = 64;
    barrier.size = 0;
    fault whole_with_callback = @catch(gpu::compat::cmd_buffer_barrier(&commands, &barrier));

    assert(whole_with_callback == gpu::compat::INVALID_ARGUMENT, "whole-buffer offset keeps INVALID_ARGUMENT");
    assert(capture.calls == 2, "whole-buffer offset emits exactly one diagnostic");
    assert(capture.operation_matches, "whole-buffer operation is cmd_buffer_barrier");
    assert(capture.has_fault && capture.public_fault == whole_with_callback,
        "whole-buffer diagnostic carries the unchanged fault");
    assert(capture.resource_kind == gpu::compat::DebugResourceKind.BUFFER, "whole-buffer resource kind is buffer");
    assert(capture.has_public_identity && capture.index == buffer.index && capture.generation == buffer.generation,
        "whole-buffer diagnostic preserves buffer identity");
    assert(capture.debug_name_matches, "whole-buffer diagnostic preserves the debug name");
    assert(capture.rejected_field_matches, "whole-buffer offset field is identified");
    assert(capture.invariant_matches, "whole-buffer offset invariant is identified");

    state.debug_callback = null;
    fault whole_without_callback = @catch(gpu::compat::cmd_buffer_barrier(&commands, &barrier));
    assert(whole_without_callback == whole_with_callback, "whole-buffer callback presence does not alter the fault");
    assert(capture.calls == 2, "whole-buffer null callback emits no duplicate");

    gpu::compat::end_commands(&commands)!!;
    gpu::compat::end_frame(&frame)!!;
}

fn void reset_sync_diagnostic(
    SyncDiagnosticCapture* capture,
    ZString operation,
    ZString debug_name,
    ZString rejected_field,
    ZString invariant,
) {
    capture.calls = 0;
    capture.has_fault = false;
    capture.public_fault = {};
    capture.public_contract = false;
    capture.resource_kind = {};
    capture.has_public_identity = false;
    capture.index = 0;
    capture.generation = 0;
    capture.expected_operation = operation;
    capture.expected_debug_name = debug_name;
    capture.expected_rejected_field = rejected_field;
    capture.expected_invariant = invariant;
    capture.operation_matches = false;
    capture.debug_name_matches = false;
    capture.rejected_field_matches = false;
    capture.invariant_matches = false;
}

fn void assert_sync_contract_diagnostic(
    SyncDiagnosticCapture* capture,
    fault expected_fault,
    gpu::compat::DebugResourceKind resource_kind,
    bool has_public_identity,
    uint index,
    uint generation,
) {
    assert(capture.calls == 1, "rejection emits exactly one diagnostic");
    assert(capture.has_fault && capture.public_fault == expected_fault,
        "diagnostic preserves the public fault");
    assert(capture.public_contract, "rejection is a public-contract diagnostic");
    assert(capture.resource_kind == resource_kind, "resource kind is exact");
    assert(capture.has_public_identity == has_public_identity, "identity presence is exact");
    assert(capture.index == index && capture.generation == generation, "resource identity is exact");
    assert(capture.operation_matches, "operation is exact");
    assert(capture.debug_name_matches, "debug name is compared during callback");
    assert(capture.rejected_field_matches, "rejected field is exact");
    assert(capture.invariant_matches, "invariant is exact");
}

fn void tranche_b_command_and_render_diagnostics() @test {
    SyncDiagnosticCapture capture = {};
    gpu::compat::Device device = make_sync_diagnostic_device(&capture);
    defer gpu::compat::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);

    gpu::compat::BufferDesc plain_desc = {
        .size = 64, .usage = { .storage, .addressable },
        .memory_kind = gpu::compat::MemoryKind.DEVICE, .debug_name = "plain_buffer",
    };
    gpu::compat::BufferHandle plain = gpu::compat::create_buffer(&device, &plain_desc)!!;
    defer gpu::compat::destroy_buffer(&device, plain)!!;
    gpu::compat::BufferDesc dst_desc = {
        .size = 64, .usage = { .transfer_dst },
        .memory_kind = gpu::compat::MemoryKind.DEVICE, .debug_name = "fill_buffer",
    };
    gpu::compat::BufferHandle dst = gpu::compat::create_buffer(&device, &dst_desc)!!;
    defer gpu::compat::destroy_buffer(&device, dst)!!;

    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);

    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    gpu::compat::BufferDesc args_desc = {
        .size = 16, .usage = { .indirect },
        .memory_kind = gpu::compat::MemoryKind.DEVICE, .debug_name = "indirect_args",
    };
    gpu::compat::BufferHandle args_buffer = gpu::compat::create_buffer(&device, &args_desc)!!;
    defer gpu::compat::destroy_buffer(&device, args_buffer)!!;
    gpu::compat::GpuSpan args_full = gpu::compat::get_buffer_span(&device, args_buffer)!!;
    gpu::compat::GpuSpan escaped_args = args_full.unchecked_subspan(8, gpu::compat::DispatchIndirectCommand::size);
    gpu::compat::TextureDesc usage_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width = 8, .height = 8, .format = gpu::compat::Format.RGBA8_UNORM,
        .usage = { .sampled }, .debug_name = "usage_target",
    };
    gpu::compat::TextureHandle usage_target = gpu::compat::create_texture(&device, &usage_desc)!!;
    defer gpu::compat::destroy_texture(&device, usage_target)!!;
    gpu::compat::TextureDesc color_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width = 8, .height = 8, .format = gpu::compat::Format.RGBA8_UNORM,
        .usage = { .color_attach }, .debug_name = "color_target",
    };
    gpu::compat::TextureHandle color_target = gpu::compat::create_texture(&device, &color_desc)!!;
    defer gpu::compat::destroy_texture(&device, color_target)!!;

    gpu::compat::FrameToken frame = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan root = gpu::compat::alloc_frame_span(&frame, 16, 16)!!;
    gpu::compat::CommandList commands = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::BufferCopyDesc copy = { .src = plain, .dst = dst, .size = 16 };
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_copy_buffer",
        debug_name:     "plain_buffer",
        rejected_field: "BufferUsage.transfer_src",
        invariant:      "source buffer must include transfer_src usage",
    );
    fault copy_fault = @catch(gpu::compat::cmd_copy_buffer(&commands, &copy));
    assert(copy_fault == gpu::compat::INVALID_ARGUMENT, "copy usage fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      copy_fault,
        resource_kind:       gpu::compat::DebugResourceKind.BUFFER,
        has_public_identity: true,
        index:               plain.index,
        generation:          plain.generation,
    );
    state.debug_callback = null;
    assert(@catch(gpu::compat::cmd_copy_buffer(&commands, &copy)) == copy_fault,
        "copy usage fault is callback-independent");
    state.debug_callback = &capture_sync_diagnostic;

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_fill_buffer",
        debug_name:     "fill_buffer",
        rejected_field: "offset",
        invariant:      "offset and size must be 4-byte aligned",
    );
    fault fill_fault = @catch(gpu::compat::cmd_fill_buffer(
        commands: &commands,
        buffer:   dst,
        offset:   2,
        size:     4,
        value:    0,
    ));
    assert(fill_fault == gpu::compat::INVALID_ARGUMENT, "fill alignment fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      fill_fault,
        resource_kind:       gpu::compat::DebugResourceKind.BUFFER,
        has_public_identity: true,
        index:               dst.index,
        generation:          dst.generation,
    );
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_fill_buffer",
        debug_name:     "fill_buffer",
        rejected_field: "size",
        invariant:      "offset and size must be 4-byte aligned",
    );
    fault fill_size_fault = @catch(gpu::compat::cmd_fill_buffer(
        commands: &commands,
        buffer:   dst,
        offset:   0,
        size:     2,
        value:    0,
    ));
    assert(fill_size_fault == gpu::compat::INVALID_ARGUMENT, "fill size-only alignment fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      fill_size_fault,
        resource_kind:       gpu::compat::DebugResourceKind.BUFFER,
        has_public_identity: true,
        index:               dst.index,
        generation:          dst.generation,
    );

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_dispatch",
        debug_name:     null,
        rejected_field: "pipeline",
        invariant:      "pipeline kind must match the COMPUTE bind point",
    );
    fault pipeline_fault = @catch(gpu::compat::cmd_dispatch(
        commands:    &commands,
        pipeline:    draw.graphics,
        root:        root.gpu,
        groups:      { 1, 1, 1 },
    ));
    assert(pipeline_fault == gpu::compat::INVALID_ARGUMENT, "pipeline-kind fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      pipeline_fault,
        resource_kind:       gpu::compat::DebugResourceKind.PIPELINE,
        has_public_identity: true,
        index:               draw.graphics.index,
        generation:          draw.graphics.generation,
    );

    gpu::compat::ColorTargetDesc[1] usage_colors = {{ .texture = usage_target }};
    gpu::compat::RenderPassDesc usage_pass = { .colors = usage_colors[..], .width = 8, .height = 8 };
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_dispatch_indirect",
        debug_name:     "indirect_args",
        rejected_field: "args",
        invariant:      "span offset and size must fit the backing buffer",
    );
    fault span_fault = @catch(gpu::compat::cmd_dispatch_indirect(
        commands: &commands,
        pipeline: compute.pipeline,
        root:     root.gpu,
        args:     escaped_args,
    ));
    assert(span_fault == gpu::compat::INVALID_ARGUMENT, "shared span fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      span_fault,
        resource_kind:       gpu::compat::DebugResourceKind.BUFFER,
        has_public_identity: true,
        index:               args_buffer.index,
        generation:          args_buffer.generation,
    );
    state.debug_callback = null;
    fault span_without_callback = @catch(gpu::compat::cmd_dispatch_indirect(
        commands: &commands,
        pipeline: compute.pipeline,
        root:     root.gpu,
        args:     escaped_args,
    ));
    assert(span_without_callback == span_fault,
        "shared span fault is callback-independent");
    state.debug_callback = &capture_sync_diagnostic;
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_begin_render_pass",
        debug_name:     "usage_target",
        rejected_field: "TextureUsage.color_attach",
        invariant:      "color attachment must include color_attach usage",
    );
    fault usage_fault = @catch(gpu::compat::cmd_begin_render_pass(&commands, &usage_pass));
    assert(usage_fault == gpu::compat::INVALID_ARGUMENT, "attachment usage fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      usage_fault,
        resource_kind:       gpu::compat::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               usage_target.index,
        generation:          usage_target.generation,
    );

    gpu::compat::ColorTargetDesc[1] colors = {{ .texture = color_target }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = 8, .height = 8 };
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_begin_render_pass",
        debug_name:     "color_target",
        rejected_field: "ColorTargetDesc.texture",
        invariant:      "color attachment must be in COLOR_ATTACHMENT layout",
    );
    fault layout_fault = @catch(gpu::compat::cmd_begin_render_pass(&commands, &pass));
    assert(layout_fault == gpu::compat::INVALID_RESOURCE_STATE, "attachment layout fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      layout_fault,
        resource_kind:       gpu::compat::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               color_target.index,
        generation:          color_target.generation,
    );
    state.debug_callback = null;
    assert(@catch(gpu::compat::cmd_begin_render_pass(&commands, &pass)) == layout_fault,
        "attachment layout fault is callback-independent");
    state.debug_callback = &capture_sync_diagnostic;

    record_val_target_to_attach(&commands, color_target);
    gpu::compat::RenderPassDesc oversized = { .colors = colors[..], .width = 9, .height = 8 };
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_begin_render_pass",
        debug_name:     "color_target",
        rejected_field: "RenderPassDesc.width",
        invariant:      "render area must fit every color attachment mip",
    );
    fault extent_fault = @catch(gpu::compat::cmd_begin_render_pass(&commands, &oversized));
    assert(extent_fault == gpu::compat::INVALID_ARGUMENT, "attachment extent fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      extent_fault,
        resource_kind:       gpu::compat::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               color_target.index,
        generation:          color_target.generation,
    );

    gpu::compat::RenderPassDesc too_tall = { .colors = colors[..], .width = 8, .height = 9 };
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_begin_render_pass",
        debug_name:     "color_target",
        rejected_field: "RenderPassDesc.height",
        invariant:      "render area must fit every color attachment mip",
    );
    fault height_fault = @catch(gpu::compat::cmd_begin_render_pass(&commands, &too_tall));
    assert(height_fault == gpu::compat::INVALID_ARGUMENT, "attachment height-only extent fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      height_fault,
        resource_kind:       gpu::compat::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               color_target.index,
        generation:          color_target.generation,
    );
    gpu::compat::cmd_begin_render_pass(&commands, &pass)!!;
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_copy_buffer",
        debug_name:     null,
        rejected_field: "CommandList.state",
        invariant:      "command must be recording outside a render pass",
    );
    fault state_fault = @catch(gpu::compat::cmd_copy_buffer(&commands, &copy));
    assert(state_fault == gpu::compat::COMMAND_RECORDING_ERROR, "recording-state fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      state_fault,
        resource_kind:       gpu::compat::DebugResourceKind.COMMAND_LIST,
        has_public_identity: true,
        index:               commands.handle.index,
        generation:          commands.handle.generation,
    );
    state.debug_callback = null;
    assert(@catch(gpu::compat::cmd_copy_buffer(&commands, &copy)) == state_fault,
        "recording-state fault is callback-independent");
    state.debug_callback = &capture_sync_diagnostic;

    gpu::compat::cmd_end_render_pass(&commands)!!;
    gpu::compat::end_commands(&commands)!!;
    gpu::compat::end_frame(&frame)!!;
}
