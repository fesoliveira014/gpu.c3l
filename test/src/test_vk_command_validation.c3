module gpu_test;

import gpu @public;
import gpu::vk @public;
import std::io;

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
    capture.debug_name_matches = capture.expected_debug_name == null
        ? message.resource.debug_name == null
        : message.resource.debug_name != null
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
        .application_name     = "gpu_sync_diagnostic_test",
        .debug_callback       = &capture_sync_diagnostic,
        .debug_user_data      = capture,
    };
    return gpu::create_device_from_desc(&desc)!!;
}

fn gpu::Device make_val_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_command_validation_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
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
        .access    = { .graphics, .compute, .transfer },
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
    upload_test_texture_data(
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
        .access    = { .graphics, .compute, .transfer },
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
        .access    = { .graphics, .compute, .transfer },
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
        .access    = { .graphics, .compute, .transfer },
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
        .access    = { .graphics, .compute, .transfer },
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
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(*device);
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    uint count = 0;
    foreach (&cached : slot.views) if (cached.used) count++;
    return count;
}

fn bool val_has_cached_view(gpu::Device* device, gpu::TextureHandle texture, uint base_mip, uint mip_count) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(*device);
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

fn gpu::GpuAddress val_span_address(gpu::Device* device, gpu::GpuSpan span) {
    return gpu::get_span_address(device, span)!!;
}

fn void write_val_triangle_verts(gpu::Device* device, gpu::GpuSpan* span) {
    char[] mapping = gpu::get_span_mapping(device, *span)!!;
    float* verts = (float*)mapping.ptr;
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

fn void submit_requires_exact_recording_queue_token() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::CommandRecord* record = state.commands.get(executable.handle)!!;
    gpu::Queue recorded_queue = record.queue;
    record.queue.id++;

    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    fault mismatched = @catch(gpu::submit(queue, &submit));
    assert(mismatched == gpu::INVALID_ARGUMENT,
        "submission requires the exact queue token captured while recording");
    assert(lists[0].handle.is_valid(),
        "queue-token rejection preserves the executable token");

    record.queue = recorded_queue;
    gpu::CompletionPoint point = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(point, 1_000_000_000)!!;
}

fn void resource_access_intersects_recording_roles_before_mutation() @test {
    SyncDiagnosticCapture capture = {
        .expected_operation      = "",
        .expected_rejected_field = "",
        .expected_invariant      = "",
    };
    gpu::Device device = make_sync_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .compute },
        debug_name:   "compute_fill_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;

    gpu::TextureDesc texture_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
        .access    = { .compute },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &texture_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    OwnedSpan span_buffer = allocate_owned_span(
        device:       &device,
        size:         4,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "compute_copy_buffer",
    )!!;
    defer free_owned_span(&device, &span_buffer)!!;
    gpu::GpuSpan span = span_buffer.span;

    OwnedSpan compute_only_owned = allocate_owned_span(
        device:       &device,
        size:         8,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "compute_only_allocation",
    )!!;
    defer free_owned_span(&device, &compute_only_owned)!!;
    gpu::GpuSpan compute_only = compute_only_owned.span;

    OwnedSpan multi_access_owned = allocate_owned_span(
        device:       &device,
        size:         8,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute },
        debug_name:   "multi_access_allocation",
    )!!;
    defer free_owned_span(&device, &multi_access_owned)!!;
    gpu::GpuSpan multi_access = multi_access_owned.span;

    OwnedSpan stale_owned = allocate_owned_span(
        device:       &device,
        size:         8,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "stale_allocation",
    )!!;
    gpu::GpuSpan stale_alias = stale_owned.span;
    free_owned_span(&device, &stale_owned)!!;
    OwnedSpan replacement_owned = allocate_owned_span(
        device:       &device,
        size:         8,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "replacement_allocation",
    )!!;
    defer free_owned_span(&device, &replacement_owned)!!;
    assert(replacement_owned.span.index == stale_alias.index
        && replacement_owned.span.generation != stale_alias.generation,
        "allocation reuse advances the generation");

    gpu::TextureDesc copy_texture_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
        .access    = { .graphics },
    };
    gpu::TextureHandle copy_texture = gpu::create_texture(&device, &copy_texture_desc)!!;
    defer gpu::destroy_texture(&device, copy_texture)!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::ResolvedSpan buffer_span = vk::resolve_span(state, &span)!!;
    vk::ResolvedSpan compute_only_span = vk::resolve_span(state, &compute_only)!!;
    gpu::GpuSpan compute_only_tail = compute_only.checked_subspan(4, 4)!!;
    vk::ResolvedSpan compute_only_tail_span = vk::resolve_span(state, &compute_only_tail)!!;
    gpu::GpuSpan multi_access_head = multi_access.checked_subspan(0, 4)!!;
    vk::ResolvedSpan multi_access_span = vk::resolve_span(state, &multi_access_head)!!;
    assert(buffer_span.access.compute && !buffer_span.access.graphics,
        "allocation span access comes from its live owner");
    assert(compute_only_span.access.compute && !compute_only_span.access.graphics
        && compute_only_tail_span.access.compute && !compute_only_tail_span.access.graphics,
        "allocation subspans preserve immutable access");
    assert(multi_access_span.access.graphics && multi_access_span.access.compute,
        "allocation subspans preserve every immutable role");
    vk::CommandRecord* record = state.commands.get(commands.handle)!!;
    gpu::Queue recorded_queue = record.queue;
    record.queue.roles = { .graphics, .compute };

    gpu::cmd_fill_buffer(
        commands: &commands,
        dst:      buffer.span.checked_subspan(0, 4)!!,
        value:    0,
    )!!;
    record.queue.roles = { .graphics };
    gpu::TextureBarrier barrier = {
        .texture       = texture,
        .before_stage  = gpu::Stage.NONE,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.NONE,
        .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.TRANSFER_DST,
    };
    uint pending_before = record.pending_layout_count;
    assert(@catch(gpu::cmd_texture_barrier(&commands, &barrier))
        == gpu::INVALID_ARGUMENT,
        "compute-only texture rejects a disjoint recording-role token");
    assert(record.pending_layout_count == pending_before,
        "access rejection precedes pending-layout mutation");

    gpu::BufferTextureCopyDesc copy = {
        .src     = span,
        .texture = copy_texture,
    };
    assert(@catch(gpu::cmd_copy_buffer_to_texture(&commands, &copy))
        == gpu::INVALID_ARGUMENT,
        "span access, not only backing-buffer access, gates the recording role");

    copy.src = compute_only;
    assert(@catch(gpu::cmd_copy_buffer_to_texture(&commands, &copy))
        == gpu::INVALID_ARGUMENT,
        "compute-only allocation access rejects graphics recording");

    copy.src = compute_only_tail;
    assert(@catch(gpu::cmd_copy_buffer_to_texture(&commands, &copy))
        == gpu::INVALID_ARGUMENT,
        "an allocation subspan retains compute-only access");

    copy.src = multi_access;
    copy.src.offset += multi_access.size;
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_copy_buffer_to_texture",
        debug_name:     "multi_access_allocation",
        rejected_field: "GpuSpan",
        invariant:      "span range must fit its live owner",
    );
    assert(@catch(gpu::cmd_copy_buffer_to_texture(&commands, &copy))
        == gpu::INVALID_ARGUMENT,
        "allocation span bounds cannot escape their owner");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        index:               multi_access.index,
        generation:          multi_access.generation,
    );

    copy.src = stale_alias;
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_copy_buffer_to_texture",
        debug_name:     null,
        rejected_field: "GpuSpan.identity",
        invariant:      "must identify one live span owner",
    );
    assert(@catch(gpu::cmd_copy_buffer_to_texture(&commands, &copy))
        == gpu::INVALID_HANDLE,
        "a stale allocation generation cannot record commands");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_HANDLE,
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: false,
        index:               0,
        generation:          0,
    );
    record.queue = recorded_queue;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;

    defer discard_test_executable(&commands_executable)!!;
}

fn bool val_distinct_graphics_compute_families(gpu::Device* device) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(*device);
    if (state.graphics_family != state.compute_family) return true;
    io::eprintfn("access-domain rejection not applicable: graphics and compute share family %d", state.graphics_family);
    return false;
}

fn void allocation_span_outside_recording_access_rejects_copy_and_fill() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    if (!val_distinct_graphics_compute_families(&device)) return;

    OwnedSpan transfer_span = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics },
        debug_name:   "transfer_span",
    )!!;
    defer free_owned_span(&device, &transfer_span)!!;
    OwnedSpan restricted_owned = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "restricted_span",
    )!!;
    defer free_owned_span(&device, &restricted_owned)!!;
    gpu::GpuSpan restricted_span = restricted_owned.span;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;

    gpu::BufferCopyDesc restricted_source = {
        .src = restricted_span,
        .dst = transfer_span.span,
    };
    assert(@catch(gpu::cmd_copy_buffer(&commands, &restricted_source))
        == gpu::INVALID_ARGUMENT,
        "copy rejects a source span outside the recording queue access domain");

    gpu::BufferCopyDesc restricted_destination = {
        .src = transfer_span.span,
        .dst = restricted_span,
    };
    assert(@catch(gpu::cmd_copy_buffer(&commands, &restricted_destination))
        == gpu::INVALID_ARGUMENT,
        "copy rejects a destination span outside the recording queue access domain");

    assert(@catch(gpu::cmd_fill_buffer(&commands, restricted_span, 0))
        == gpu::INVALID_ARGUMENT,
        "fill rejects a span outside the recording queue access domain");
}

fn void render_pass_access_preflight_precedes_view_creation() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc admitted_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = 4,
        .height     = 4,
        .mip_levels = 2,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .color_attach },
        .access     = { .graphics },
    };
    gpu::TextureHandle admitted = gpu::create_texture(&device, &admitted_desc)!!;
    defer gpu::destroy_texture(&device, admitted)!!;

    gpu::TextureDesc rejected_desc = admitted_desc;
    rejected_desc.access = { .compute };
    gpu::TextureHandle rejected = gpu::create_texture(&device, &rejected_desc)!!;
    defer gpu::destroy_texture(&device, rejected)!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    record_val_target_to_attach(&commands, admitted);
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    gpu::Queue recorded_queue = record.queue;
    record.queue.roles = { .graphics };

    gpu::ColorTargetDesc[2] colors = {
        { .texture = admitted, .mip_level = 1 },
        { .texture = rejected, .mip_level = 1 },
    };
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = 2,
        .height = 2,
    };
    uint cached_before = val_cached_view_count(&device, admitted);
    assert(@catch(gpu::cmd_begin_render_pass(&commands, &pass))
        == gpu::INVALID_ARGUMENT,
        "a later attachment outside the recording role rejects the whole pass");
    assert(val_cached_view_count(&device, admitted) == cached_before,
        "attachment access preflight precedes view-cache mutation");
    assert(record.state == vk::CommandState.RECORDING,
        "attachment access rejection leaves command state reusable");

    record.queue = recorded_queue;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;

    defer discard_test_executable(&commands_executable)!!;
}

fn void fill_alignment_faults_report_the_rejected_component() @test {
    SyncDiagnosticCapture capture = {};
    gpu::Device device = make_sync_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         8,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "fill_alignment_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!,
    )!!;
    defer discard_test_commands(&commands)!!;

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_fill_buffer",
        debug_name:     "fill_alignment_buffer",
        rejected_field: "GpuSpan.offset",
        invariant:      "span offset must be 4-byte aligned",
    );
    gpu::GpuSpan misaligned_offset = buffer.span.checked_subspan(2, 4)!!;
    fault offset_fault = @catch(gpu::cmd_fill_buffer(
        &commands,
        misaligned_offset,
        0,
    ));
    assert(offset_fault == gpu::INVALID_ARGUMENT,
        "fill rejects a span with a misaligned offset");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      offset_fault,
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        index:               buffer.span.index,
        generation:          buffer.span.generation,
    );

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_fill_buffer",
        debug_name:     "fill_alignment_buffer",
        rejected_field: "GpuSpan.size",
        invariant:      "span size must be 4-byte aligned",
    );
    gpu::GpuSpan misaligned_size = buffer.span.checked_subspan(0, 2)!!;
    fault size_fault = @catch(gpu::cmd_fill_buffer(
        &commands,
        misaligned_size,
        0,
    ));
    assert(size_fault == gpu::INVALID_ARGUMENT,
        "fill rejects a span with a misaligned size");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      size_fault,
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        index:               buffer.span.index,
        generation:          buffer.span.generation,
    );

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void indexed_draw_misaligned_offset_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    OwnedSpan verts_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "misaligned_draw_vertices",
    )!!;
    defer free_owned_span(&device, &verts_owned)!!;
    gpu::GpuSpan verts_span = verts_owned.span;
    write_val_triangle_verts(&device, &verts_span);
    gpu::flush_mapped_span(&device, verts_span)!!;
    OwnedSpan frag_owned = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "misaligned_draw_fragment_root",
    )!!;
    defer free_owned_span(&device, &frag_owned)!!;
    gpu::GpuSpan frag_span = frag_owned.span;
    // Allocation buffers carry index usage; a 2-byte offset still misaligns
    // the default U32 index type (needs 4-byte alignment).
    OwnedSpan index_owned = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "misaligned_draw_indices",
    )!!;
    defer free_owned_span(&device, &index_owned)!!;
    gpu::GpuSpan index_span = index_owned.span;
    gpu::GpuSpan misaligned = index_span.checked_subspan(2, 8)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    fault misaligned_fault = @catch(gpu::cmd_draw_indexed(
        commands:      &cmd,
        pipeline:      ctx.graphics,
        vertex_root:   val_span_address(&device, verts_span),
        fragment_root: val_span_address(&device, frag_span),
        index_span:    misaligned,
        index_count:   1,
        instance_count: 1,
    ));
    assert(misaligned_fault == gpu::INVALID_ARGUMENT, "misaligned index offset faults INVALID_ARGUMENT");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void dispatch_with_graphics_pipeline_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "wrong_pipeline_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    gpu::GpuSpan root_span = root_owned.span;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    fault wrong_kind = @catch(gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: ctx.graphics,
        root:     val_span_address(&device, root_span),
        groups:   { 1, 1, 1 },
    ));
    assert(wrong_kind == gpu::INVALID_ARGUMENT, "dispatch with a graphics pipeline faults INVALID_ARGUMENT");
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void draw_with_compute_pipeline_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = VAL_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle comp_shader = gpu::create_shader(&device, &comp_sd)!!;
    defer gpu::destroy_shader(&device, comp_shader)!!;
    gpu::ComputePipelineDesc comp_desc = { .shader = comp_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle compute = gpu::create_compute_pipeline(&device, &comp_desc)!!;
    defer gpu::destroy_pipeline(&device, compute)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
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
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void compute_pipeline_from_vertex_shader_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle vert_shader = gpu::create_shader(&device, &vert_sd)!!;
    defer gpu::destroy_shader(&device, vert_shader)!!;

    gpu::ComputePipelineDesc desc = { .shader = vert_shader, .push_constant_size = gpu::RootPush::size };
    fault wrong_stage = @catch(gpu::create_compute_pipeline(&device, &desc));
    assert(wrong_stage == gpu::INVALID_ARGUMENT, "compute pipeline from a vertex-stage shader faults INVALID_ARGUMENT");
}

fn void graphics_pipeline_from_swapped_shader_stages_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

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
    defer destroy_test_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    OwnedSpan upload = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "valid_copy_src",
    )!!;
    defer free_owned_span(&device, &upload)!!;
    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "valid_copy_dst",
    )!!;
    defer free_owned_span(&device, &readback)!!;
    OwnedSpan fill_target = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "valid_fill_dst",
    )!!;
    defer free_owned_span(&device, &fill_target)!!;
    OwnedSpan verts_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "valid_draw_vertices",
    )!!;
    defer free_owned_span(&device, &verts_owned)!!;
    gpu::GpuSpan verts_span = verts_owned.span;
    write_val_triangle_verts(&device, &verts_span);
    gpu::flush_mapped_span(&device, verts_span)!!;
    OwnedSpan frag_owned = allocate_owned_span(
        device:       &device,
        size:         FragRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "valid_draw_fragment_root",
    )!!;
    defer free_owned_span(&device, &frag_owned)!!;
    gpu::GpuSpan frag_span = frag_owned.span;
    char[] frag_mapping = gpu::get_span_mapping(&device, frag_span)!!;
    FragRoot* frag_root = (FragRoot*)frag_mapping.ptr;
    frag_root.texture_index = ctx.tex_index;
    frag_root.sampler_index = ctx.smp_index;
    gpu::flush_mapped_span(&device, frag_span)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::BufferCopyDesc copy = {
        .src = upload.span,
        .dst = readback.span,
    };
    gpu::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::cmd_fill_buffer(&cmd, fill_target.span, 0)!!;

    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw(&cmd, ctx.graphics, val_span_address(&device, verts_span), val_span_address(&device, frag_span), 3, 1)!!;
    gpu::cmd_end_render_pass(&cmd)!!;

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
}

// Backing-range faults: subspan builds escaping spans without faulting;
// the fault lands at the consuming command.

fn void dispatch_indirect_args_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx ctx = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &ctx);

    OwnedSpan args_buffer = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "dispatch_indirect_args",
    )!!;
    defer free_owned_span(&device, &args_buffer)!!;
    gpu::GpuSpan full_span = args_buffer.span;
    gpu::GpuSpan escaped = full_span.unchecked_subspan(8, gpu::DispatchIndirectCommand::size);
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    fault beyond = @catch(gpu::cmd_dispatch_indirect(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::GpuAddress)0,
        args:     escaped,
    ));
    assert(beyond == gpu::INVALID_ARGUMENT, "dispatch indirect args beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void dispatch_indirect_args_span_boundary_exact_succeeds() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx ctx = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &ctx);

    OwnedSpan args_buffer = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "dispatch_indirect_args",
    )!!;
    defer free_owned_span(&device, &args_buffer)!!;
    gpu::GpuSpan full_span = args_buffer.span;
    gpu::GpuSpan boundary = full_span.checked_subspan(4, gpu::DispatchIndirectCommand::size)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::cmd_dispatch_indirect(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::GpuAddress)0,
        args:     boundary,
    )!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void draw_indexed_indirect_count_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    if (!gpu::get_device_caps(&device)!!.draw_indirect_count) return;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    OwnedSpan args_buffer = allocate_owned_span(
        device:       &device,
        size:         gpu::DrawIndexedIndirectCommand::size * 2,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "draw_indirect_args",
    )!!;
    defer free_owned_span(&device, &args_buffer)!!;
    gpu::GpuSpan args_span = args_buffer.span;

    OwnedSpan count_buffer = allocate_owned_span(
        device:       &device,
        size:         8,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "draw_indirect_count",
    )!!;
    defer free_owned_span(&device, &count_buffer)!!;
    gpu::GpuSpan full_count_span = count_buffer.span;
    gpu::GpuSpan escaped_count = full_count_span.unchecked_subspan(8, 4);
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
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
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void indirect_draw_count_limits_follow_command_contract() @test {
    SyncDiagnosticCapture capture = {};
    gpu::Device device = make_sync_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;
    if (!gpu::get_device_caps(&device)!!.draw_indirect_count) return;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    OwnedSpan args_buffer = allocate_owned_span(
        device:       &device,
        size:         gpu::DrawIndexedIndirectCommand::size * 2,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "draw_count_args",
    )!!;
    defer free_owned_span(&device, &args_buffer)!!;
    gpu::GpuSpan args_span = args_buffer.span;

    OwnedSpan count_buffer = allocate_owned_span(
        device:       &device,
        size:         4,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "draw_count_value",
    )!!;
    defer free_owned_span(&device, &count_buffer)!!;
    gpu::GpuSpan count_span = count_buffer.span;

    OwnedSpan index_buffer = allocate_owned_span(
        device:       &device,
        size:         4,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "draw_count_indices",
    )!!;
    defer free_owned_span(&device, &index_buffer)!!;
    gpu::GpuSpan index_span = index_buffer.span;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{
        .texture  = target,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = VAL_RT,
        .height = VAL_RT,
    };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    uint original_limit = state.caps.max_draw_indirect_count;
    state.caps.max_draw_indirect_count = 1;
    defer state.caps.max_draw_indirect_count = original_limit;

    gpu::cmd_draw_indirect(
        commands:      &cmd,
        pipeline:      ctx.graphics,
        vertex_root:   (gpu::GpuAddress)0,
        fragment_root: (gpu::GpuAddress)0,
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
    fault draw_over = @catch(gpu::cmd_draw_indirect(
        commands:      &cmd,
        pipeline:      ctx.graphics,
        vertex_root:   (gpu::GpuAddress)0,
        fragment_root: (gpu::GpuAddress)0,
        args:          args_span,
        draw_count:    2,
    ));
    assert(draw_over == gpu::INVALID_ARGUMENT,
        "direct indirect draw count above the selected-device limit faults before recording");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      draw_over,
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
    );

    gpu::cmd_draw_indexed_indirect_count(
        commands:       &cmd,
        pipeline:       ctx.graphics,
        vertex_root:    (gpu::GpuAddress)0,
        fragment_root:  (gpu::GpuAddress)0,
        args:           args_span,
        count_span:     count_span,
        max_draw_count: 2,
        index_span:     index_span,
    )!!;

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void draw_indexed_index_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    OwnedSpan index_buffer = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "indexed_draw_data",
    )!!;
    defer free_owned_span(&device, &index_buffer)!!;
    gpu::GpuSpan full_index_span = index_buffer.span;
    gpu::GpuSpan escaped = full_index_span.unchecked_subspan(16, 4);
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
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
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void draw_indexed_index_span_boundary_exact_succeeds() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValDrawCtx ctx = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &ctx);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    OwnedSpan index_buffer = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "indexed_draw_data",
    )!!;
    defer free_owned_span(&device, &index_buffer)!!;
    gpu::GpuSpan full_index_span = index_buffer.span;
    gpu::GpuSpan boundary = full_index_span.checked_subspan(12, 4)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
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
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void copy_buffer_to_texture_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    OwnedSpan src_buffer = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "texture_copy_src",
    )!!;
    defer free_owned_span(&device, &src_buffer)!!;
    gpu::GpuSpan full_span = src_buffer.span;
    gpu::GpuSpan escaped = full_span.unchecked_subspan(32, 64);
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_texture_transition(&cmd, texture, gpu::TextureLayout.TRANSFER_DST, gpu::Stage.TRANSFER, gpu::Hazard.TRANSFER_WRITE);

    gpu::BufferTextureCopyDesc up = { .src = escaped, .texture = texture, .width = 4, .height = 4 };
    fault beyond = @catch(gpu::cmd_copy_buffer_to_texture(&cmd, &up));
    assert(beyond == gpu::INVALID_ARGUMENT, "buffer-to-texture copy span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void copy_texture_to_buffer_span_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_src },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    OwnedSpan dst_buffer = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "texture_copy_dst",
    )!!;
    defer free_owned_span(&device, &dst_buffer)!!;
    gpu::GpuSpan full_span = dst_buffer.span;
    gpu::GpuSpan escaped = full_span.unchecked_subspan(32, 64);
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_texture_transition(&cmd, texture, gpu::TextureLayout.TRANSFER_SRC, gpu::Stage.TRANSFER, gpu::Hazard.TRANSFER_READ);

    gpu::TextureBufferCopyDesc down = { .texture = texture, .dst = escaped, .width = 4, .height = 4 };
    fault beyond = @catch(gpu::cmd_copy_texture_to_buffer(&cmd, &down));
    assert(beyond == gpu::INVALID_ARGUMENT, "texture-to-buffer copy span beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void texture_copy_span_boundary_exact_succeeds() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    OwnedSpan src_buffer = allocate_owned_span(
        device:       &device,
        size:         96,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "texture_copy_boundary_src",
    )!!;
    defer free_owned_span(&device, &src_buffer)!!;
    gpu::GpuSpan full_span = src_buffer.span;
    gpu::GpuSpan boundary = full_span.checked_subspan(32, 64)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_texture_transition(&cmd, texture, gpu::TextureLayout.TRANSFER_DST, gpu::Stage.TRANSFER, gpu::Hazard.TRANSFER_WRITE);

    gpu::BufferTextureCopyDesc up = { .src = boundary, .texture = texture, .width = 4, .height = 4 };
    gpu::cmd_copy_buffer_to_texture(&cmd, &up)!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void buffer_barrier_range_beyond_buffer_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "barrier_range_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::BufferBarrier barrier = {
        .span = buffer.span.unchecked_subspan(48, 32),
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
    };
    fault beyond = @catch(gpu::cmd_buffer_barrier(&cmd, &barrier));
    assert(beyond == gpu::INVALID_ARGUMENT, "buffer barrier range beyond the backing buffer faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void buffer_barrier_boundary_exact_succeeds() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "barrier_range_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::BufferBarrier barrier = {
        .span = buffer.span.checked_subspan(32, 32)!!,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &barrier)!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void buffer_barrier_zero_size_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "barrier_range_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::BufferBarrier barrier = {
        .span = buffer.span.unchecked_subspan(0, 0),
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
    };
    fault zero_size = @catch(gpu::cmd_buffer_barrier(&cmd, &barrier));
    assert(zero_size == gpu::INVALID_ARGUMENT, "zero-size barrier span faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void render_pass_attachment_extents_validate_before_recording() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
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
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
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

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void render_pass_color_count_limits_validate_before_lookup() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

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

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void selected_device_dispatch_limit_validates_before_recording() @test {
    SyncDiagnosticCapture capture = {};
    gpu::Device device = make_sync_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;
    ValComputeCtx ctx = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &ctx);

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    gpu::DeviceCaps original_caps = state.caps;
    gpu::DeviceCaps limited_caps = state.caps;
    limited_caps.max_compute_work_group_count = { .x = 1, .y = 1, .z = 1 };
    state.caps = limited_caps;
    defer state.caps = original_caps;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::GpuAddress)0,
        groups:   {},
    )!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::GpuAddress)0,
        groups:   limited_caps.max_compute_work_group_count,
    )!!;

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_dispatch",
        debug_name:     null,
        rejected_field: "groups",
        invariant:      "group counts must not exceed DeviceCaps.max_compute_work_group_count",
    );
    fault dispatch_over = @catch(gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: ctx.pipeline,
        root:     (gpu::GpuAddress)0,
        groups:   { .x = 2, .y = 1, .z = 1 },
    ));
    assert(dispatch_over == gpu::INVALID_ARGUMENT,
        "dispatch above the selected-device count limit faults before recording");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      dispatch_over,
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
    );

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void render_pass_view_cache_rolls_back_on_later_resolution_fault() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
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
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
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
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_target_to_attach(&cmd, first);
    record_val_target_to_attach(&cmd, saturated);

    uint first_cached_before = val_cached_view_count(&device, first);
    gpu::ColorTargetDesc[2] colors = {
        { .texture = first, .mip_level = 1 },
        { .texture = saturated, .mip_level = 1 },
    };
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = 8, .height = 8 };
    fault view_fault = @catch(gpu::cmd_begin_render_pass(&cmd, &pass));
    state.debug_callback = null;
    foreach (&cached : saturated_slot.views) cached.used = false;

    assert(view_fault == gpu::SLOT_TABLE_FULL, "later saturated view cache faults SLOT_TABLE_FULL");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      view_fault,
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               saturated.index,
        generation:          saturated.generation,
    );
    assert(val_cached_view_count(&device, first) == first_cached_before,
        "later view resolution fault rolls back the earlier cached view");
    assert(@catch(gpu::cmd_end_render_pass(&cmd)) == gpu::COMMAND_RECORDING_ERROR, "view resolution fault records no pass");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void buffer_barrier_range_diagnostic_identifies_resource() @test {
    SyncDiagnosticCapture capture = {
        .expected_operation      = "cmd_buffer_barrier",
        .expected_debug_name     = "barrier_diagnostic_buffer",
        .expected_rejected_field = "BufferBarrier.span",
        .expected_invariant      = "span range must fit its live owner",
    };
    gpu::Device device = make_sync_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "barrier_diagnostic_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::BufferBarrier barrier = {
        .span          = buffer.span.unchecked_subspan(48, 32),
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
    assert(capture.resource_kind == gpu::DebugResourceKind.ALLOCATION, "resource kind is allocation");
    assert(capture.has_public_identity, "live buffer identity is present");
    assert(capture.index == buffer.span.index, "allocation index is preserved");
    assert(capture.generation == buffer.span.generation, "allocation generation is preserved");
    assert(capture.debug_name_matches, "buffer debug name is preserved");
    assert(capture.rejected_field_matches, "barrier size field is identified");
    assert(capture.invariant_matches, "barrier range invariant is identified");

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    state.debug_callback = null;
    fault without_callback = @catch(gpu::cmd_buffer_barrier(&commands, &barrier));
    assert(without_callback == with_callback, "callback presence does not alter the barrier fault");
    assert(capture.calls == 1, "null callback emits no additional diagnostic");

    state.debug_callback = &capture_sync_diagnostic;
    capture.expected_rejected_field = "BufferBarrier.span";
    capture.expected_invariant = "span range must fit its live owner";
    barrier.span = buffer.span.unchecked_subspan(64, 0);
    fault zero_with_callback = @catch(gpu::cmd_buffer_barrier(&commands, &barrier));

    assert(zero_with_callback == gpu::INVALID_ARGUMENT, "zero-size span keeps INVALID_ARGUMENT");
    assert(capture.calls == 2, "zero-size span emits exactly one diagnostic");
    assert(capture.operation_matches, "zero-size span operation is cmd_buffer_barrier");
    assert(capture.has_fault && capture.public_fault == zero_with_callback,
        "zero-size span diagnostic carries the unchanged fault");
    assert(capture.resource_kind == gpu::DebugResourceKind.ALLOCATION, "zero-size span resource kind is allocation");
    assert(capture.has_public_identity
        && capture.index == buffer.span.index
        && capture.generation == buffer.span.generation,
        "zero-size span diagnostic preserves buffer identity");
    assert(capture.debug_name_matches, "zero-size span diagnostic preserves the debug name");
    assert(capture.rejected_field_matches, "zero-size span field is identified");
    assert(capture.invariant_matches, "zero-size span invariant is identified");

    state.debug_callback = null;
    fault zero_without_callback = @catch(gpu::cmd_buffer_barrier(&commands, &barrier));
    assert(zero_without_callback == zero_with_callback, "callback presence does not alter the zero-size span fault");
    assert(capture.calls == 2, "null callback emits no duplicate zero-size span diagnostic");

    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;

    defer discard_test_executable(&commands_executable)!!;
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
    gpu::DebugResourceKind resource_kind,
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

fn void buffer_to_texture_access_diagnostic_precedes_recording() @test {
    SyncDiagnosticCapture capture = {};
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_copy_buffer_to_texture",
        debug_name:     "copy_access_texture",
        rejected_field: "TextureDesc.access",
        invariant:      "access must include the operation queue role",
    );
    gpu::Device device = make_sync_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc texture_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access     = { .graphics },
        .width      = 1,
        .height     = 1,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .transfer_dst },
        .debug_name = "copy_access_texture",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &texture_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    OwnedSpan buffer = allocate_owned_span(
        device:       &device,
        size:         4,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .transfer },
        debug_name:   "copy_access_buffer",
    )!!;
    defer free_owned_span(&device, &buffer)!!;
    gpu::GpuSpan span = buffer.span;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    gpu::BufferTextureCopyDesc copy = {
        .src     = span,
        .texture = texture,
        .width   = 1,
        .height  = 1,
    };
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::CommandRecord* record = state.commands.get(commands.handle)!!;
    gpu::QueueRoles selected_roles = record.queue.roles;
    // Keep diagnostic precedence independent of physical queue aliasing.
    record.queue.roles = { .transfer };

    fault copy_fault = @catch(gpu::cmd_copy_buffer_to_texture(&commands, &copy));

    assert(copy_fault == gpu::INVALID_ARGUMENT, "copy keeps INVALID_ARGUMENT");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      copy_fault,
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               texture.index,
        generation:          texture.generation,
    );

    vk::TextureSlot* texture_slot = state.textures.get(texture)!!;
    gpu::QueueRoles texture_access = texture_slot.access;
    texture_slot.access = { .transfer };
    record_val_texture_transition(
        cmd:          &commands,
        texture:      texture,
        new_layout:   gpu::TextureLayout.TRANSFER_DST,
        after_stage:  gpu::Stage.TRANSFER,
        after_hazard: gpu::Hazard.TRANSFER_WRITE,
    );
    gpu::cmd_copy_buffer_to_texture(&commands, &copy)!!;
    assert(capture.calls == 1,
        "valid allocation copy emits no diagnostic after texture access is admitted");
    texture_slot.access = texture_access;
    record.queue.roles = selected_roles;

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void texture_copy_span_usage_diagnostics_are_semantic() @test {
    SyncDiagnosticCapture capture = {};
    gpu::Device device = make_sync_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc texture_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access     = { .transfer },
        .width      = 1,
        .height     = 1,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .transfer_src, .transfer_dst },
        .debug_name = "texture_copy_target",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &texture_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    OwnedSpan owned = allocate_owned_span(
        device:       &device,
        size:         4,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .transfer },
        debug_name:   "texture_copy_span",
    )!!;
    defer free_owned_span(&device, &owned)!!;
    gpu::GpuSpan span = owned.span;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::ResolvedSpan resolved = vk::resolve_span(state, &span)!!;
    bool transfer_src = resolved.slot.usage.transfer_src;
    bool transfer_dst = resolved.slot.usage.transfer_dst;
    defer {
        resolved.slot.usage.transfer_src = transfer_src;
        resolved.slot.usage.transfer_dst = transfer_dst;
    }
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!,
    )!!;
    defer discard_test_commands(&commands)!!;

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_copy_buffer_to_texture",
        debug_name:     "texture_copy_span",
        rejected_field: "GpuSpan",
        invariant:      "source span must support copy reads",
    );
    resolved.slot.usage.transfer_src = false;
    gpu::BufferTextureCopyDesc upload = {
        .src     = span,
        .texture = texture,
        .width   = 1,
        .height  = 1,
    };
    fault source_fault = @catch(gpu::cmd_copy_buffer_to_texture(&commands, &upload));
    resolved.slot.usage.transfer_src = transfer_src;

    assert(source_fault == gpu::INVALID_ARGUMENT, "source span rejection keeps INVALID_ARGUMENT");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      source_fault,
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        index:               span.index,
        generation:          span.generation,
    );

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_copy_texture_to_buffer",
        debug_name:     "texture_copy_span",
        rejected_field: "GpuSpan",
        invariant:      "destination span must support copy writes",
    );
    resolved.slot.usage.transfer_dst = false;
    gpu::TextureBufferCopyDesc readback = {
        .texture = texture,
        .dst     = span,
        .width   = 1,
        .height  = 1,
    };
    fault destination_fault = @catch(gpu::cmd_copy_texture_to_buffer(&commands, &readback));
    resolved.slot.usage.transfer_dst = transfer_dst;

    assert(destination_fault == gpu::INVALID_ARGUMENT,
        "destination span rejection keeps INVALID_ARGUMENT");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      destination_fault,
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        index:               span.index,
        generation:          span.generation,
    );

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}
fn void tranche_b_command_and_render_diagnostics() @test {
    SyncDiagnosticCapture capture = {};
    gpu::Device device = make_sync_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    OwnedSpan dst = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "fill_buffer",
    )!!;
    defer free_owned_span(&device, &dst)!!;

    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);

    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    OwnedSpan args_buffer = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "indirect_args",
    )!!;
    defer free_owned_span(&device, &args_buffer)!!;
    gpu::GpuSpan args_full = args_buffer.span;
    gpu::GpuSpan escaped_args = args_full.unchecked_subspan(8, gpu::DispatchIndirectCommand::size);
    gpu::TextureDesc usage_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width = 8, .height = 8, .format = gpu::Format.RGBA8_UNORM,
        .usage = { .sampled }, .debug_name = "usage_target",
    };
    gpu::TextureHandle usage_target = gpu::create_texture(&device, &usage_desc)!!;
    defer gpu::destroy_texture(&device, usage_target)!!;
    gpu::TextureDesc color_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width = 8, .height = 8, .format = gpu::Format.RGBA8_UNORM,
        .usage = { .color_attach }, .debug_name = "color_target",
    };
    gpu::TextureHandle color_target = gpu::create_texture(&device, &color_desc)!!;
    defer gpu::destroy_texture(&device, color_target)!!;
    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "diagnostic_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    gpu::GpuSpan root = root_owned.span;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_fill_buffer",
        debug_name:     "fill_buffer",
        rejected_field: "GpuSpan.offset",
        invariant:      "span offset must be 4-byte aligned",
    );
    fault fill_fault = @catch(gpu::cmd_fill_buffer(
        commands: &commands,
        dst:      dst.span.checked_subspan(2, 4)!!,
        value:    0,
    ));
    assert(fill_fault == gpu::INVALID_ARGUMENT, "fill alignment fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      fill_fault,
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        index:               dst.span.index,
        generation:          dst.span.generation,
    );
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_fill_buffer",
        debug_name:     "fill_buffer",
        rejected_field: "GpuSpan.size",
        invariant:      "span size must be 4-byte aligned",
    );
    fault fill_size_fault = @catch(gpu::cmd_fill_buffer(
        commands: &commands,
        dst:      dst.span.checked_subspan(0, 2)!!,
        value:    0,
    ));
    assert(fill_size_fault == gpu::INVALID_ARGUMENT, "fill size-only alignment fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      fill_size_fault,
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        index:               dst.span.index,
        generation:          dst.span.generation,
    );

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_dispatch",
        debug_name:     null,
        rejected_field: "pipeline",
        invariant:      "pipeline kind must match the COMPUTE bind point",
    );
    fault pipeline_fault = @catch(gpu::cmd_dispatch(
        commands:    &commands,
        pipeline:    draw.graphics,
        root:        val_span_address(&device, root),
        groups:      { 1, 1, 1 },
    ));
    assert(pipeline_fault == gpu::INVALID_ARGUMENT, "pipeline-kind fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      pipeline_fault,
        resource_kind:       gpu::DebugResourceKind.PIPELINE,
        has_public_identity: true,
        index:               draw.graphics.index,
        generation:          draw.graphics.generation,
    );

    gpu::ColorTargetDesc[1] usage_colors = {{ .texture = usage_target }};
    gpu::RenderPassDesc usage_pass = { .colors = usage_colors[..], .width = 8, .height = 8 };
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_dispatch_indirect",
        debug_name:     "indirect_args",
        rejected_field: "args",
        invariant:      "span range must fit its live owner",
    );
    fault span_fault = @catch(gpu::cmd_dispatch_indirect(
        commands: &commands,
        pipeline: compute.pipeline,
        root:     val_span_address(&device, root),
        args:     escaped_args,
    ));
    assert(span_fault == gpu::INVALID_ARGUMENT, "shared span fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      span_fault,
        resource_kind:       gpu::DebugResourceKind.ALLOCATION,
        has_public_identity: true,
        index:               args_buffer.span.index,
        generation:          args_buffer.span.generation,
    );
    state.debug_callback = null;
    fault span_without_callback = @catch(gpu::cmd_dispatch_indirect(
        commands: &commands,
        pipeline: compute.pipeline,
        root:     val_span_address(&device, root),
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
    fault usage_fault = @catch(gpu::cmd_begin_render_pass(&commands, &usage_pass));
    assert(usage_fault == gpu::INVALID_ARGUMENT, "attachment usage fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      usage_fault,
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               usage_target.index,
        generation:          usage_target.generation,
    );

    gpu::ColorTargetDesc[1] colors = {{ .texture = color_target }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = 8, .height = 8 };
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_begin_render_pass",
        debug_name:     "color_target",
        rejected_field: "ColorTargetDesc.texture",
        invariant:      "color attachment must be in COLOR_ATTACHMENT layout",
    );
    fault layout_fault = @catch(gpu::cmd_begin_render_pass(&commands, &pass));
    assert(layout_fault == gpu::INVALID_RESOURCE_STATE, "attachment layout fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      layout_fault,
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               color_target.index,
        generation:          color_target.generation,
    );
    state.debug_callback = null;
    assert(@catch(gpu::cmd_begin_render_pass(&commands, &pass)) == layout_fault,
        "attachment layout fault is callback-independent");
    state.debug_callback = &capture_sync_diagnostic;

    record_val_target_to_attach(&commands, color_target);
    gpu::RenderPassDesc oversized = { .colors = colors[..], .width = 9, .height = 8 };
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_begin_render_pass",
        debug_name:     "color_target",
        rejected_field: "RenderPassDesc.width",
        invariant:      "render area must fit every color attachment mip",
    );
    fault extent_fault = @catch(gpu::cmd_begin_render_pass(&commands, &oversized));
    assert(extent_fault == gpu::INVALID_ARGUMENT, "attachment extent fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      extent_fault,
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               color_target.index,
        generation:          color_target.generation,
    );

    gpu::RenderPassDesc too_tall = { .colors = colors[..], .width = 8, .height = 9 };
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_begin_render_pass",
        debug_name:     "color_target",
        rejected_field: "RenderPassDesc.height",
        invariant:      "render area must fit every color attachment mip",
    );
    fault height_fault = @catch(gpu::cmd_begin_render_pass(&commands, &too_tall));
    assert(height_fault == gpu::INVALID_ARGUMENT, "attachment height-only extent fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      height_fault,
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               color_target.index,
        generation:          color_target.generation,
    );
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::BufferCopyDesc state_copy = {
        .src = dst.span.checked_subspan(0, 16)!!,
        .dst = dst.span.checked_subspan(16, 16)!!,
    };
    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_copy_buffer",
        debug_name:     null,
        rejected_field: "CommandList.state",
        invariant:      "command must be recording outside a render pass",
    );
    fault state_fault = @catch(gpu::cmd_copy_buffer(&commands, &state_copy));
    assert(state_fault == gpu::COMMAND_RECORDING_ERROR, "recording-state fault is unchanged");
    assert_sync_contract_diagnostic(
        capture:             &capture,
        expected_fault:      state_fault,
        resource_kind:       gpu::DebugResourceKind.COMMAND_LIST,
        has_public_identity: true,
        index:               commands.handle.index,
        generation:          commands.handle.generation,
    );
    state.debug_callback = null;
    assert(@catch(gpu::cmd_copy_buffer(&commands, &state_copy)) == state_fault,
        "recording-state fault is callback-independent");
    state.debug_callback = &capture_sync_diagnostic;

    gpu::cmd_end_render_pass(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
}
