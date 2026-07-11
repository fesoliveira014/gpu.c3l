module gpu_test;

import gpu;
import gpu::vk;

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
    gpu::DebugResourceKind resource_kind;
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

fn void capture_sync_diagnostic(gpu::DebugMessage* message, void* user_data) {
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
    capture.debug_name_matches = message.resource.debug_name != null
        && message.resource.debug_name.str_view() == capture.expected_debug_name.str_view();
    capture.rejected_field_matches = message.rejected_field != null
        && message.rejected_field.str_view() == capture.expected_rejected_field.str_view();
    capture.invariant_matches = message.invariant != null
        && message.invariant.str_view() == capture.expected_invariant.str_view();
}

fn gpu::Device make_sync_diagnostic_device(SyncDiagnosticCapture* capture) {
    gpu::DeviceDesc desc = {
        .backend              = gpu::BackendKind.VULKAN,
        .enable_validation    = true,
        .frames_in_flight     = 2,
        .application_name     = "gpu_sync_diagnostic_test",
        .debug_callback       = &capture_sync_diagnostic,
        .debug_user_data      = capture,
    };
    return gpu::create_device(&desc)!!;
}

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

fn gpu::TextureHandle make_val_color_target(gpu::Device* device, uint width, uint height, uint mip_levels) {
    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = width,
        .height     = height,
        .mip_levels = mip_levels,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .color_attach, .transfer_src },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::TextureHandle make_val_depth_target(gpu::Device* device, uint width, uint height) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = width,
        .height    = height,
        .format    = gpu::Format.D32_FLOAT,
        .usage     = { .depth_attach },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::TextureHandle make_val_depth_target_mips(gpu::Device* device, uint width, uint height, uint mip_levels) {
    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = width,
        .height     = height,
        .mip_levels = mip_levels,
        .format     = gpu::Format.D32_FLOAT,
        .usage      = { .depth_attach },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void record_val_depth_to_attach(gpu::CommandList* cmd, gpu::TextureHandle depth) {
    record_val_texture_transition(
        cmd:          cmd,
        texture:      depth,
        new_layout:   gpu::TextureLayout.DEPTH_STENCIL,
        after_stage:  gpu::Stage.DEPTH_STENCIL,
        after_hazard: gpu::Hazard.DEPTH_WRITE,
    );
}

fn uint val_cached_view_count(gpu::Device* device, gpu::TextureHandle texture) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    uint count = 0;
    foreach (&cached : slot.views) if (cached.used) count++;
    return count;
}

fn bool val_has_cached_view(gpu::Device* device, gpu::TextureHandle texture, uint base_mip, uint mip_count) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    foreach (&cached : slot.views) {
        if (cached.used && cached.base_mip == base_mip && cached.mip_count == mip_count
            && cached.base_layer == 0 && cached.layer_count == 1) return true;
    }
    return false;
}

// Never actually dispatched, so the shader's ABI content is unused.
struct ValComputeCtx {
    gpu::ShaderHandle   shader;
    gpu::PipelineHandle pipeline;
}

fn ValComputeCtx make_val_compute_ctx(gpu::Device* device) {
    ValComputeCtx ctx;
    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = VAL_COMP_SPIRV[..], .entry_point = "main" };
    ctx.shader = gpu::create_shader(device, &comp_sd)!!;
    gpu::ComputePipelineDesc comp_desc = { .shader = ctx.shader, .push_constant_size = gpu::RootPush::size };
    ctx.pipeline = gpu::create_compute_pipeline(device, &comp_desc)!!;
    return ctx;
}

fn void destroy_val_compute_ctx(gpu::Device* device, ValComputeCtx* ctx) {
    gpu::destroy_pipeline(device, ctx.pipeline)!!;
    gpu::destroy_shader(device, ctx.shader)!!;
}

fn void record_val_texture_transition(gpu::CommandList* cmd, gpu::TextureHandle texture, gpu::TextureLayout new_layout, gpu::Stage after_stage, gpu::Hazard after_hazard) {
    gpu::TextureBarrier barrier = {
        .texture       = texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = after_stage,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = after_hazard,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = new_layout,
    };
    gpu::cmd_texture_barrier(cmd, &barrier)!!;
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

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList gfx = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&gfx)!!;
    gpu::CommandList xfer = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::end_commands(&xfer)!!;

    gpu::CommandList[2] lists = { gfx, xfer };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    fault mixed = @catch(gpu::submit(&device, &submit));
    assert(mixed == gpu::INVALID_ARGUMENT, "mixed-queue submit faults INVALID_ARGUMENT before any queue work runs");
    gpu::end_frame(&frame_token)!!;
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

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;

    gpu::BufferCopyDesc no_src_usage_copy = { .src = plain, .dst = dst, .size = 64 };
    fault no_src_usage = @catch(gpu::cmd_copy_buffer(&cmd, &no_src_usage_copy));
    assert(no_src_usage == gpu::INVALID_ARGUMENT, "copy from a buffer without transfer_src faults INVALID_ARGUMENT");

    gpu::BufferCopyDesc no_dst_usage_copy = { .src = src, .dst = plain, .size = 64 };
    fault no_dst_usage = @catch(gpu::cmd_copy_buffer(&cmd, &no_dst_usage_copy));
    assert(no_dst_usage == gpu::INVALID_ARGUMENT, "copy to a buffer without transfer_dst faults INVALID_ARGUMENT");

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
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

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;

    // Buffer usage is checked before the texture's tracked layout, so no
    // barrier is needed to reach either fault.
    gpu::BufferTextureCopyDesc up = { .src = plain_span, .texture = texture, .width = 4, .height = 4 };
    fault no_src_usage = @catch(gpu::cmd_copy_buffer_to_texture(&cmd, &up));
    assert(no_src_usage == gpu::INVALID_ARGUMENT, "buffer-to-texture copy without buffer transfer_src faults INVALID_ARGUMENT");

    gpu::TextureBufferCopyDesc down = { .texture = texture, .dst = plain_span, .width = 4, .height = 4 };
    fault no_dst_usage = @catch(gpu::cmd_copy_texture_to_buffer(&cmd, &down));
    assert(no_dst_usage == gpu::INVALID_ARGUMENT, "texture-to-buffer copy without buffer transfer_dst faults INVALID_ARGUMENT");

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void fill_without_transfer_dst_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;
    fault no_usage = @catch(gpu::cmd_fill_buffer(&cmd, buffer, 0, 64, 0));
    assert(no_usage == gpu::INVALID_ARGUMENT, "fill on a buffer without transfer_dst faults INVALID_ARGUMENT");
    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
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

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
    write_val_triangle_verts(&verts_span);
    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&frame_token, 16, 16)!!;

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
    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void indexed_draw_misaligned_offset_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
    write_val_triangle_verts(&verts_span);
    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&frame_token, 16, 16)!!;
    // Frame-arena buffers carry .index usage; a 2-byte offset still misaligns
    // the default U32 index type (needs 4-byte alignment).
    gpu::GpuSpan index_span = gpu::alloc_frame_span(&frame_token, 16, 16)!!;
    gpu::GpuSpan misaligned = index_span.checked_subspan(2, 8)!!;

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
    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void dispatch_with_graphics_pipeline_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::GpuSpan root_span = gpu::alloc_frame_span(&frame_token, 16, 16)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    fault wrong_kind = @catch(gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: ctx.graphics,
        root:     root_span.gpu,
        groups:   { 1, 1, 1 },
    ));
    assert(wrong_kind == gpu::INVALID_ARGUMENT, "dispatch with a graphics pipeline faults INVALID_ARGUMENT");
    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
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

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
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
    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
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

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
    write_val_triangle_verts(&verts_span);
    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&frame_token, FragRoot::size, 16)!!;
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

    gpu::end_commands(&cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;
}

// Backing-range faults: subspan builds escaping spans without faulting;
// the fault lands at the consuming command.

fn void dispatch_indirect_args_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    ValComputeCtx ctx = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &ctx);

    gpu::BufferDesc args_desc = { .size = 16, .usage = { .storage, .addressable, .indirect }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle args_buffer = gpu::create_buffer(&device, &args_desc)!!;
    defer gpu::destroy_buffer(&device, args_buffer)!!;
    gpu::GpuSpan full_span = gpu::get_buffer_span(&device, args_buffer)!!;
    gpu::GpuSpan escaped = full_span.unchecked_subspan(8, gpu::DispatchIndirectCommand::size);

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    fault beyond = @catch(gpu::cmd_dispatch_indirect(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::GpuAddress)0,
        args:     escaped,
    ));
    assert(beyond == gpu::INVALID_ARGUMENT, "dispatch indirect args beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void dispatch_indirect_args_span_boundary_exact_succeeds() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    ValComputeCtx ctx = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &ctx);

    gpu::BufferDesc args_desc = { .size = 16, .usage = { .storage, .addressable, .indirect }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle args_buffer = gpu::create_buffer(&device, &args_desc)!!;
    defer gpu::destroy_buffer(&device, args_buffer)!!;
    gpu::GpuSpan full_span = gpu::get_buffer_span(&device, args_buffer)!!;
    gpu::GpuSpan boundary = full_span.checked_subspan(4, gpu::DispatchIndirectCommand::size)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::cmd_dispatch_indirect(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::GpuAddress)0,
        args:     boundary,
    )!!;

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void draw_indexed_indirect_count_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    if (!device.caps.draw_indirect_count) return;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::BufferDesc args_desc = { .size = gpu::DrawIndexedIndirectCommand::size, .usage = { .storage, .addressable, .indirect }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle args_buffer = gpu::create_buffer(&device, &args_desc)!!;
    defer gpu::destroy_buffer(&device, args_buffer)!!;
    gpu::GpuSpan args_span = gpu::get_buffer_span(&device, args_buffer)!!;

    gpu::BufferDesc count_desc = { .size = 8, .usage = { .storage, .addressable, .indirect }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle count_buffer = gpu::create_buffer(&device, &count_desc)!!;
    defer gpu::destroy_buffer(&device, count_buffer)!!;
    gpu::GpuSpan full_count_span = gpu::get_buffer_span(&device, count_buffer)!!;
    gpu::GpuSpan escaped_count = full_count_span.unchecked_subspan(8, 4);

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    gpu::GpuSpan index_span_stub = {};
    fault beyond = @catch(gpu::cmd_draw_indexed_indirect_count(
        commands:       &cmd,
        pipeline:       ctx.graphics,
        vertex_root:    (gpu::GpuAddress)0,
        fragment_root:  (gpu::GpuAddress)0,
        args:           args_span,
        count_span:     escaped_count,
        max_draw_count: 1,
        index_span:     index_span_stub,
    ));
    assert(beyond == gpu::INVALID_ARGUMENT, "indirect count span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void draw_indexed_index_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::BufferDesc index_desc = { .size = 16, .usage = { .index }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle index_buffer = gpu::create_buffer(&device, &index_desc)!!;
    defer gpu::destroy_buffer(&device, index_buffer)!!;
    gpu::GpuSpan full_index_span = gpu::get_buffer_span(&device, index_buffer)!!;
    gpu::GpuSpan escaped = full_index_span.unchecked_subspan(16, 4);

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    fault beyond = @catch(gpu::cmd_draw_indexed(
        commands:       &cmd,
        pipeline:       ctx.graphics,
        vertex_root:    (gpu::GpuAddress)0,
        fragment_root:  (gpu::GpuAddress)0,
        index_span:     escaped,
        index_count:    1,
        instance_count: 1,
    ));
    assert(beyond == gpu::INVALID_ARGUMENT, "index span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void draw_indexed_index_span_boundary_exact_succeeds() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::BufferDesc index_desc = { .size = 16, .usage = { .index }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle index_buffer = gpu::create_buffer(&device, &index_desc)!!;
    defer gpu::destroy_buffer(&device, index_buffer)!!;
    gpu::GpuSpan full_index_span = gpu::get_buffer_span(&device, index_buffer)!!;
    gpu::GpuSpan boundary = full_index_span.checked_subspan(12, 4)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    gpu::cmd_draw_indexed(
        commands:       &cmd,
        pipeline:       ctx.graphics,
        vertex_root:    (gpu::GpuAddress)0,
        fragment_root:  (gpu::GpuAddress)0,
        index_span:     boundary,
        index_count:    1,
        instance_count: 1,
    )!!;

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void copy_buffer_to_texture_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::BufferDesc src_desc = { .size = 64, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle src_buffer = gpu::create_buffer(&device, &src_desc)!!;
    defer gpu::destroy_buffer(&device, src_buffer)!!;
    gpu::GpuSpan full_span = gpu::get_buffer_span(&device, src_buffer)!!;
    gpu::GpuSpan escaped = full_span.unchecked_subspan(32, 64);

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;
    record_val_texture_transition(&cmd, texture, gpu::TextureLayout.TRANSFER_DST, gpu::Stage.TRANSFER, gpu::Hazard.TRANSFER_WRITE);

    gpu::BufferTextureCopyDesc up = { .src = escaped, .texture = texture, .width = 4, .height = 4 };
    fault beyond = @catch(gpu::cmd_copy_buffer_to_texture(&cmd, &up));
    assert(beyond == gpu::INVALID_ARGUMENT, "buffer-to-texture copy span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void copy_texture_to_buffer_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_src },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::BufferDesc dst_desc = { .size = 64, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle dst_buffer = gpu::create_buffer(&device, &dst_desc)!!;
    defer gpu::destroy_buffer(&device, dst_buffer)!!;
    gpu::GpuSpan full_span = gpu::get_buffer_span(&device, dst_buffer)!!;
    gpu::GpuSpan escaped = full_span.unchecked_subspan(32, 64);

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;
    record_val_texture_transition(&cmd, texture, gpu::TextureLayout.TRANSFER_SRC, gpu::Stage.TRANSFER, gpu::Hazard.TRANSFER_READ);

    gpu::TextureBufferCopyDesc down = { .texture = texture, .dst = escaped, .width = 4, .height = 4 };
    fault beyond = @catch(gpu::cmd_copy_texture_to_buffer(&cmd, &down));
    assert(beyond == gpu::INVALID_ARGUMENT, "texture-to-buffer copy span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void texture_copy_span_boundary_exact_succeeds() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::BufferDesc src_desc = { .size = 96, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle src_buffer = gpu::create_buffer(&device, &src_desc)!!;
    defer gpu::destroy_buffer(&device, src_buffer)!!;
    gpu::GpuSpan full_span = gpu::get_buffer_span(&device, src_buffer)!!;
    gpu::GpuSpan boundary = full_span.checked_subspan(32, 64)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;
    record_val_texture_transition(&cmd, texture, gpu::TextureLayout.TRANSFER_DST, gpu::Stage.TRANSFER, gpu::Hazard.TRANSFER_WRITE);

    gpu::BufferTextureCopyDesc up = { .src = boundary, .texture = texture, .width = 4, .height = 4 };
    gpu::cmd_copy_buffer_to_texture(&cmd, &up)!!;

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void buffer_barrier_range_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::BufferBarrier barrier = {
        .buffer = buffer, .offset = 48, .size = 32,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
    };
    fault beyond = @catch(gpu::cmd_buffer_barrier(&cmd, &barrier));
    assert(beyond == gpu::INVALID_ARGUMENT, "buffer barrier range beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void buffer_barrier_boundary_exact_succeeds() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::BufferBarrier barrier = {
        .buffer = buffer, .offset = 32, .size = 32,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &barrier)!!;

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void buffer_barrier_zero_size_whole_buffer_succeeds() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = { .size = 64, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::BufferBarrier barrier = {
        .buffer = buffer, .offset = 0, .size = 0,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &barrier)!!;

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void render_pass_attachment_extents_validate_before_recording() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle large = make_val_color_target(
        device:     &device,
        width:      16,
        height:     16,
        mip_levels: 5,
    );
    defer gpu::destroy_texture(&device, large)!!;
    gpu::TextureHandle small = make_val_color_target(
        device:     &device,
        width:      8,
        height:     8,
        mip_levels: 1,
    );
    defer gpu::destroy_texture(&device, small)!!;
    gpu::TextureHandle unbarriered = make_val_color_target(
        device:     &device,
        width:      16,
        height:     16,
        mip_levels: 5,
    );
    defer gpu::destroy_texture(&device, unbarriered)!!;
    gpu::TextureHandle depth = make_val_depth_target(&device, 8, 8);
    defer gpu::destroy_texture(&device, depth)!!;
    gpu::TextureHandle mip_depth = make_val_depth_target_mips(
        device:     &device,
        width:      8,
        height:     8,
        mip_levels: 4,
    );
    defer gpu::destroy_texture(&device, mip_depth)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, large);
    record_val_target_to_attach(&cmd, small);
    record_val_depth_to_attach(&cmd, depth);
    record_val_depth_to_attach(&cmd, mip_depth);

    gpu::ColorTargetDesc[1] large_color = { { .texture = large } };
    gpu::RenderPassDesc oversized = { .colors = large_color[..], .width = 17, .height = 16 };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &oversized)) == gpu::INVALID_ARGUMENT,
        "render area wider than mip zero faults INVALID_ARGUMENT");
    assert(@catch(gpu::cmd_end_render_pass(&cmd)) == gpu::COMMAND_RECORDING_ERROR, "oversized color target records no pass");

    gpu::ColorTargetDesc[2] mixed_colors = {
        { .texture = large },
        { .texture = small },
    };
    gpu::RenderPassDesc mixed = { .colors = mixed_colors[..], .width = 16, .height = 16 };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &mixed)) == gpu::INVALID_ARGUMENT,
        "one mixed-size attachment that cannot cover the pass faults INVALID_ARGUMENT");
    assert(@catch(gpu::cmd_end_render_pass(&cmd)) == gpu::COMMAND_RECORDING_ERROR, "mixed-size rejection records no pass");

    uint cached_before = val_cached_view_count(&device, large);
    gpu::ColorTargetDesc[2] invalid_later = {
        { .texture = large, .mip_level = 2 },
        { .texture = unbarriered, .mip_level = 99 },
    };
    gpu::RenderPassDesc invalid_later_pass = { .colors = invalid_later[..], .width = 4, .height = 4 };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &invalid_later_pass)) == gpu::INVALID_ARGUMENT,
        "invalid later mip range faults before its invalid layout");
    assert(val_cached_view_count(&device, large) == cached_before,
        "later validation failure creates no cached view for an earlier target");
    assert(@catch(gpu::cmd_end_render_pass(&cmd)) == gpu::COMMAND_RECORDING_ERROR, "later attachment rejection records no pass");

    gpu::ColorTargetDesc[1] mip_color = { { .texture = large, .mip_level = 1 } };
    gpu::RenderPassDesc mip_oversized = { .colors = mip_color[..], .width = 9, .height = 8 };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &mip_oversized)) == gpu::INVALID_ARGUMENT,
        "render area wider than nonzero mip faults INVALID_ARGUMENT");

    gpu::RenderPassDesc mip_exact = { .colors = mip_color[..], .width = 8, .height = 8 };
    gpu::cmd_begin_render_pass(&cmd, &mip_exact)!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::DepthTargetDesc depth_target = { .texture = depth };
    gpu::RenderPassDesc depth_oversized = { .depth = &depth_target, .width = 9, .height = 8 };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &depth_oversized)) == gpu::INVALID_ARGUMENT,
        "render area wider than depth mip zero faults INVALID_ARGUMENT");
    assert(@catch(gpu::cmd_end_render_pass(&cmd)) == gpu::COMMAND_RECORDING_ERROR, "oversized depth target records no pass");

    gpu::DepthTargetDesc mip_depth_target = { .texture = mip_depth };
    uint mip_depth_cached_before = val_cached_view_count(&device, mip_depth);
    gpu::RenderPassDesc mip_depth_pass = { .depth = &mip_depth_target, .width = 8, .height = 8 };
    gpu::cmd_begin_render_pass(&cmd, &mip_depth_pass)!!;
    assert(val_cached_view_count(&device, mip_depth) == mip_depth_cached_before + 1,
        "multi-mip depth attachment resolves a single-mip cached view");
    gpu::cmd_end_render_pass(&cmd)!!;
    assert(val_has_cached_view(
        device:    &device,
        texture:   mip_depth,
        base_mip:  0,
        mip_count: 1,
    ), "multi-mip depth attachment view selects exactly mip zero and layer zero");

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void render_pass_color_count_limits_validate_before_lookup() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::ColorTargetDesc[gpu::MAX_COLOR_ATTACHMENTS + 1] too_many;
    gpu::RenderPassDesc library_limit = { .colors = too_many[..], .width = 1, .height = 1 };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &library_limit)) == gpu::INVALID_ARGUMENT,
        "color count above MAX_COLOR_ATTACHMENTS faults before invalid handle lookup");
    assert(@catch(gpu::cmd_end_render_pass(&cmd)) == gpu::COMMAND_RECORDING_ERROR, "library-limit rejection records no pass");

    uint device_limit = state.caps.max_color_attachments;
    state.caps.max_color_attachments = 1;
    gpu::ColorTargetDesc[2] device_excess;
    gpu::RenderPassDesc selected_limit = { .colors = device_excess[..], .width = 1, .height = 1 };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &selected_limit)) == gpu::INVALID_ARGUMENT,
        "color count above selected device limit faults before invalid handle lookup");
    assert(@catch(gpu::cmd_end_render_pass(&cmd)) == gpu::COMMAND_RECORDING_ERROR, "device-limit rejection records no pass");
    state.caps.max_color_attachments = device_limit;

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void render_pass_view_cache_rolls_back_on_later_resolution_fault() @test {
    gpu::Device device = make_val_device();
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle first = make_val_color_target(
        device:     &device,
        width:      16,
        height:     16,
        mip_levels: 5,
    );
    defer gpu::destroy_texture(&device, first)!!;
    gpu::TextureHandle saturated = make_val_color_target(
        device:     &device,
        width:      16,
        height:     16,
        mip_levels: 5,
    );
    defer gpu::destroy_texture(&device, saturated)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    vk::TextureSlot* saturated_slot = &state.textures.slots[saturated.index].value;
    foreach (&cached : saturated_slot.views) cached.used = true;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_val_target_to_attach(&cmd, first);
    record_val_target_to_attach(&cmd, saturated);

    uint first_cached_before = val_cached_view_count(&device, first);
    gpu::ColorTargetDesc[2] colors = {
        { .texture = first, .mip_level = 1 },
        { .texture = saturated, .mip_level = 1 },
    };
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = 8, .height = 8 };
    fault view_fault = @catch(gpu::cmd_begin_render_pass(&cmd, &pass));
    foreach (&cached : saturated_slot.views) cached.used = false;

    assert(view_fault == gpu::SLOT_TABLE_FULL, "later saturated view cache faults SLOT_TABLE_FULL");
    assert(val_cached_view_count(&device, first) == first_cached_before,
        "later view resolution fault rolls back the earlier cached view");
    assert(@catch(gpu::cmd_end_render_pass(&cmd)) == gpu::COMMAND_RECORDING_ERROR, "view resolution fault records no pass");

    gpu::end_commands(&cmd)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void buffer_barrier_range_diagnostic_identifies_resource() @test {
    SyncDiagnosticCapture capture = {
        .expected_operation      = "cmd_buffer_barrier",
        .expected_debug_name     = "barrier_diagnostic_buffer",
        .expected_rejected_field = "BufferBarrier.size",
        .expected_invariant      = "offset and size must fit the buffer",
    };
    gpu::Device device = make_sync_diagnostic_device(&capture);
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = {
        .size        = 64,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "barrier_diagnostic_buffer",
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::BufferBarrier barrier = {
        .buffer        = buffer,
        .offset        = 48,
        .size          = 32,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
    };
    fault with_callback = @catch(gpu::cmd_buffer_barrier(&commands, &barrier));

    assert(with_callback == gpu::INVALID_ARGUMENT, "barrier range keeps INVALID_ARGUMENT");
    assert(capture.calls == 1, "barrier range emits exactly one diagnostic");
    assert(capture.operation_matches, "operation is cmd_buffer_barrier");
    assert(capture.has_fault && capture.public_fault == with_callback,
        "diagnostic carries the unchanged barrier fault");
    assert(capture.public_contract, "barrier range is a public-contract diagnostic");
    assert(capture.resource_kind == gpu::DebugResourceKind.BUFFER, "resource kind is buffer");
    assert(capture.has_public_identity, "live buffer identity is present");
    assert(capture.index == buffer.index, "buffer index is preserved");
    assert(capture.generation == buffer.generation, "buffer generation is preserved");
    assert(capture.debug_name_matches, "buffer debug name is preserved");
    assert(capture.rejected_field_matches, "barrier size field is identified");
    assert(capture.invariant_matches, "barrier range invariant is identified");

    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    state.debug_callback = null;
    fault without_callback = @catch(gpu::cmd_buffer_barrier(&commands, &barrier));
    assert(without_callback == with_callback, "callback presence does not alter the barrier fault");
    assert(capture.calls == 1, "null callback emits no additional diagnostic");

    state.debug_callback = &capture_sync_diagnostic;
    capture.expected_rejected_field = "BufferBarrier.offset";
    capture.expected_invariant = "whole-buffer barrier offset must lie inside the buffer";
    barrier.offset = 64;
    barrier.size = 0;
    fault whole_with_callback = @catch(gpu::cmd_buffer_barrier(&commands, &barrier));

    assert(whole_with_callback == gpu::INVALID_ARGUMENT, "whole-buffer offset keeps INVALID_ARGUMENT");
    assert(capture.calls == 2, "whole-buffer offset emits exactly one diagnostic");
    assert(capture.operation_matches, "whole-buffer operation is cmd_buffer_barrier");
    assert(capture.has_fault && capture.public_fault == whole_with_callback,
        "whole-buffer diagnostic carries the unchanged fault");
    assert(capture.resource_kind == gpu::DebugResourceKind.BUFFER, "whole-buffer resource kind is buffer");
    assert(capture.has_public_identity && capture.index == buffer.index && capture.generation == buffer.generation,
        "whole-buffer diagnostic preserves buffer identity");
    assert(capture.debug_name_matches, "whole-buffer diagnostic preserves the debug name");
    assert(capture.rejected_field_matches, "whole-buffer offset field is identified");
    assert(capture.invariant_matches, "whole-buffer offset invariant is identified");

    state.debug_callback = null;
    fault whole_without_callback = @catch(gpu::cmd_buffer_barrier(&commands, &barrier));
    assert(whole_without_callback == whole_with_callback, "whole-buffer callback presence does not alter the fault");
    assert(capture.calls == 2, "whole-buffer null callback emits no duplicate");

    gpu::end_commands(&commands)!!;
    gpu::end_frame(&frame)!!;
}
