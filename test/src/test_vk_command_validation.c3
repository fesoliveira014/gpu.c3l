module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::core::mem;
import std::io;

// Headless (lavapipe): recording/submit validation faults — mixed-queue
// submits, usage/alignment, pipeline-kind and shader-stage mismatches.

const char[*] VAL_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] VAL_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");
const char[*] VAL_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

const uint VAL_RT = 16;
const vk::QueueFamilyFlags TRANSFER_ONLY_QUEUE_FLAGS =
    (vk::QueueFamilyFlags)(uint)vk::QueueFlagBits.TRANSFER_BIT;

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
   vertex/fragment code, one graphics pipeline, and a 1x1 white pattern feeding the
   fragment heap path. *>
struct ValDrawCtx {
    gpu::PipelineHandle graphics;
    gpu::TextureHandle  pattern;
    gpu::TextureView    tex_view;
    gpu::TextureIndex   tex_index;
    gpu::SamplerIndex   smp_index;
}

fn ValDrawCtx make_val_draw_ctx(gpu::Device* device) {
    ValDrawCtx ctx;
    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode vert = gpu::prepare_shader_code(&vert_sd)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = VAL_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode frag = gpu::prepare_shader_code(&frag_sd)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
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
    ctx.tex_view = gpu::create_texture_view(device, ctx.pattern, null)!!;
    ctx.tex_index = ctx.tex_view.index;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::Sampler sampler = gpu::intern_sampler(device, &sampler_desc)!!;
    ctx.smp_index = gpu::publish_sampler(device, sampler)!!;

    char[4] white = { (char)255, (char)255, (char)255, (char)255 };
    upload_test_texture_data(
        device:   device,
        dst:      ctx.pattern,
        mip:      0,
        data:     white[..],
        next_use: gpu::TextureUse.SAMPLED_FRAGMENT,
    )!!;
    return ctx;
}

fn void destroy_val_draw_ctx(gpu::Device* device, ValDrawCtx* ctx) {
    gpu::destroy_texture_view(device, ctx.tex_view)!!;
    gpu::destroy_texture(device, ctx.pattern)!!;
    gpu::destroy_pipeline(device, ctx.graphics)!!;
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
        commands: cmd,
        texture:  depth,
        after:    gpu::TextureUse.DEPTH_ATTACHMENT,
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
    gpu::PipelineHandle pipeline;
}

fn ValComputeCtx make_val_compute_ctx(gpu::Device* device) {
    ValComputeCtx ctx;
    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = VAL_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&comp_sd)!!;
    gpu::ComputePipelineDesc comp_desc = { .shader = shader, .push_constant_size = gpu::RootPush::size };
    ctx.pipeline = gpu::create_compute_pipeline(device, &comp_desc)!!;
    return ctx;
}

fn void destroy_val_compute_ctx(gpu::Device* device, ValComputeCtx* ctx) {
    gpu::destroy_pipeline(device, ctx.pipeline)!!;
}

fn void record_val_texture_transition(
    gpu::CommandList* commands,
    gpu::TextureHandle texture,
    gpu::TextureUse after,
) {
    gpu::TextureBarrier barrier = gpu::texture_transition(
        texture: texture,
        before:  gpu::TextureUse.UNDEFINED,
        after:   after,
    )!!;
    gpu::cmd_texture_barrier(commands, &barrier)!!;
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
        .texture = target,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.COLOR_ATTACHMENT,
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
        .texture = texture,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.TRANSFER_DESTINATION,
    };
    uint references_before = record.reference_count;
    assert(@catch(gpu::cmd_texture_barrier(&commands, &barrier))
        == gpu::INVALID_ARGUMENT,
        "compute-only texture rejects a disjoint recording-role token");
    assert(record.reference_count == references_before,
        "access rejection precedes validation-reference mutation");

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

    bind_test_graphics_pipeline(&cmd, ctx.graphics);
    fault misaligned_fault = @catch(gpu::cmd_draw_indexed(
        commands:      &cmd,
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
    bind_test_compute_pipeline(&cmd, ctx.graphics);
    fault wrong_kind = @catch(gpu::cmd_dispatch(
        commands: &cmd,
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
    gpu::ShaderCode comp_shader = gpu::prepare_shader_code(&comp_sd)!!;
    gpu::ComputePipelineDesc comp_desc = { .shader = comp_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle compute = gpu::create_compute_pipeline(&device, &comp_desc)!!;
    defer gpu::destroy_pipeline(&device, compute)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_val_target_to_attach(&cmd, target);
    gpu::ColorTargetDesc[1] colors = {{ .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = VAL_RT, .height = VAL_RT };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;

    bind_test_graphics_pipeline(&cmd, compute);
    fault wrong_kind = @catch(gpu::cmd_draw(
        commands:      &cmd,
        vertex_root:   (gpu::GpuAddress)1,
        fragment_root: (gpu::GpuAddress)1,
        vertex_count:  3,
        instance_count: 1,
    ));
    assert(wrong_kind == gpu::INVALID_ARGUMENT, "draw with a compute pipeline faults INVALID_ARGUMENT");

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void explicit_pipeline_binding_state_is_transactional() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;
    OwnedSpan root_owned = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute },
        debug_name:   "explicit_pipeline_root",
    )!!;
    defer free_owned_span(&device, &root_owned)!!;
    gpu::GpuAddress root = val_span_address(&device, root_owned.span);
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong native_pipelines_before = vk::pipeline_native_create_count(state);

    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;

    fault missing_pipeline = @catch(gpu::cmd_dispatch(
        commands: &commands,
        root:     root,
        groups:   { 1, 1, 1 },
    ));
    assert(missing_pipeline == gpu::COMMAND_RECORDING_ERROR,
        "dispatch without an active pipeline faults command state");

    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;
    fault wrong_compute_kind = @catch(gpu::cmd_dispatch(
        commands: &commands,
        root:     root,
        groups:   { 1, 1, 1 },
    ));
    assert(wrong_compute_kind == gpu::INVALID_ARGUMENT,
        "dispatch rejects an active graphics pipeline");

    gpu::cmd_bind_pipeline(&commands, compute.pipeline)!!;
    fault missing_compute_root = @catch(gpu::cmd_dispatch(
        commands: &commands,
        root:     (gpu::GpuAddress)0,
        groups:   { 1, 1, 1 },
    ));
    assert(missing_compute_root == gpu::INVALID_ARGUMENT,
        "dispatch rejects a missing compute root");
    gpu::cmd_dispatch(
        commands: &commands,
        root:     root,
        groups:   { 0, 0, 0 },
    )!!;

    ValComputeCtx stale = make_val_compute_ctx(&device);
    gpu::PipelineHandle stale_handle = stale.pipeline;
    gpu::destroy_pipeline(&device, stale_handle)!!;
    fault stale_bind = @catch(gpu::cmd_bind_pipeline(&commands, stale_handle));
    assert(stale_bind == gpu::INVALID_HANDLE, "binding a stale pipeline faults INVALID_HANDLE");
    gpu::cmd_dispatch(
        commands: &commands,
        root:     root,
        groups:   { 0, 0, 0 },
    )!!;

    record_val_target_to_attach(&commands, target);
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
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_bind_pipeline(&commands, draw.graphics)!!;

    fault missing_depth_state = @catch(gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   root,
        fragment_root: root,
        vertex_count:  0,
        instance_count: 0,
    ));
    assert(missing_depth_state == gpu::COMMAND_RECORDING_ERROR,
        "draw requires explicit depth state in each render pass");

    gpu::DepthState depth = {};
    gpu::cmd_set_depth_state(&commands, &depth)!!;
    fault missing_vertex_root = @catch(gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   (gpu::GpuAddress)0,
        fragment_root: root,
        vertex_count:  0,
        instance_count: 0,
    ));
    assert(missing_vertex_root == gpu::INVALID_ARGUMENT,
        "draw rejects a missing vertex root");
    fault missing_fragment_root = @catch(gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   root,
        fragment_root: (gpu::GpuAddress)0,
        vertex_count:  0,
        instance_count: 0,
    ));
    assert(missing_fragment_root == gpu::INVALID_ARGUMENT,
        "draw rejects a missing fragment root");

    gpu::DepthState invalid_depth = depth;
    int invalid_compare = 99;
    mem::copy(&invalid_depth.compare, &invalid_compare, int::size);
    fault invalid_depth_fault = @catch(gpu::cmd_set_depth_state(
        &commands,
        &invalid_depth,
    ));
    assert(invalid_depth_fault == gpu::INVALID_ARGUMENT,
        "invalid depth state faults before replacing valid command state");
    gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   root,
        fragment_root: root,
        vertex_count:  0,
        instance_count: 0,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    fault reset_depth_state = @catch(gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   root,
        fragment_root: root,
        vertex_count:  0,
        instance_count: 0,
    ));
    assert(reset_depth_state == gpu::COMMAND_RECORDING_ERROR,
        "render-pass begin resets explicit depth state");
    gpu::cmd_end_render_pass(&commands)!!;

    assert(vk::pipeline_native_create_count(state) == native_pipelines_before,
        "pipeline bind and execution perform no native compilation");
}
fn void compute_pipeline_from_vertex_shader_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode vert_shader = gpu::prepare_shader_code(&vert_sd)!!;

    gpu::ComputePipelineDesc desc = { .shader = vert_shader, .push_constant_size = gpu::RootPush::size };
    fault wrong_stage = @catch(gpu::create_compute_pipeline(&device, &desc));
    assert(wrong_stage == gpu::INVALID_ARGUMENT, "compute pipeline from a vertex-stage shader faults INVALID_ARGUMENT");
}

fn void graphics_pipeline_from_swapped_shader_stages_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = VAL_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode vert_shader = gpu::prepare_shader_code(&vert_sd)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = VAL_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode frag_shader = gpu::prepare_shader_code(&frag_sd)!!;

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
    bind_test_graphics_pipeline(&cmd, ctx.graphics);
    gpu::cmd_draw(&cmd, val_span_address(&device, verts_span), val_span_address(&device, frag_span), 3, 1)!!;
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
    bind_test_compute_pipeline(&cmd, ctx.pipeline);
    fault beyond = @catch(gpu::cmd_dispatch_indirect(
        commands: &cmd,
        root:     (gpu::GpuAddress)1,
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
    bind_test_compute_pipeline(&cmd, ctx.pipeline);
    gpu::cmd_dispatch_indirect(
        commands: &cmd,
        root:     (gpu::GpuAddress)1,
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
    bind_test_graphics_pipeline(&cmd, ctx.graphics);
    fault beyond = @catch(gpu::cmd_draw_indexed_indirect_count(
        commands:       &cmd,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
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

    bind_test_graphics_pipeline(&cmd, ctx.graphics);
    gpu::cmd_draw_indirect(
        commands:      &cmd,
        vertex_root:   (gpu::GpuAddress)1,
        fragment_root: (gpu::GpuAddress)1,
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
    bind_test_graphics_pipeline(&cmd, ctx.graphics);
    fault draw_over = @catch(gpu::cmd_draw_indirect(
        commands:      &cmd,
        vertex_root:   (gpu::GpuAddress)1,
        fragment_root: (gpu::GpuAddress)1,
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

    bind_test_graphics_pipeline(&cmd, ctx.graphics);
    gpu::cmd_draw_indexed_indirect_count(
        commands:       &cmd,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
        args:           args_span,
        count_span:     count_span,
        max_draw_count: 2,
        index_span:     index_span,
    )!!;

    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void generated_work_commands_require_capability() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);
    ValDrawCtx draw = make_val_draw_ctx(&device);
    defer destroy_val_draw_ctx(&device, &draw);
    gpu::TextureHandle target = make_val_target(&device);
    defer gpu::destroy_texture(&device, target)!!;

    OwnedSpan records = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "generated_records",
    )!!;
    defer free_owned_span(&device, &records)!!;
    OwnedSpan count = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "generated_count",
    )!!;
    defer free_owned_span(&device, &count)!!;
    OwnedSpan indices = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "generated_indices",
    )!!;
    defer free_owned_span(&device, &indices)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    bool original_enabled = state.generated_work.enabled;
    state.generated_work.enabled = false;
    defer state.generated_work.enabled = original_enabled;

    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    bind_test_compute_pipeline(&commands, compute.pipeline);
    fault dispatch_unsupported = @catch(gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           records.span,
        count_span:        count.span,
        max_dispatch_count: 1,
    ));
    assert(dispatch_unsupported == gpu::UNSUPPORTED_FEATURE,
        "generated dispatch requires the generated-work capability");

    record_val_target_to_attach(&commands, target);
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
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    bind_test_graphics_pipeline(&commands, draw.graphics);
    fault draw_unsupported = @catch(gpu::cmd_draw_generated(
        commands:      &commands,
        records:       records.span,
        count_span:    count.span,
        max_draw_count: 1,
    ));
    assert(draw_unsupported == gpu::UNSUPPORTED_FEATURE,
        "generated draw requires the generated-work capability");
    fault indexed_unsupported = @catch(gpu::cmd_draw_indexed_generated(
        commands:      &commands,
        records:       records.span,
        count_span:    count.span,
        max_draw_count: 1,
        index_span:    indices.span,
    ));
    assert(indexed_unsupported == gpu::UNSUPPORTED_FEATURE,
        "generated indexed draw requires the generated-work capability");

    state.generated_work.enabled = true;
    gpu::cmd_draw_generated(
        commands:      &commands,
        records:       {},
        count_span:    {},
        max_draw_count: 0,
    )!!;
    gpu::IndexType unknown_type;
    *(int*)&unknown_type = 99;
    fault invalid_zero_index_type = @catch(gpu::cmd_draw_indexed_generated(
        commands:      &commands,
        records:       {},
        count_span:    {},
        max_draw_count: 0,
        index_span:    {},
        index_type:    unknown_type,
    ));
    assert(invalid_zero_index_type == gpu::INVALID_ARGUMENT,
        "zero generated indexed draws still validate the index type");
    gpu::cmd_draw_indexed_generated(
        commands:      &commands,
        records:       {},
        count_span:    {},
        max_draw_count: 0,
        index_span:    {},
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void generated_dispatch_validates_zero_count_bounds_and_alignment() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    ValComputeCtx compute = make_val_compute_ctx(&device);
    defer destroy_val_compute_ctx(&device, &compute);

    OwnedSpan records = allocate_owned_span(
        device:       &device,
        size:         gpu::GeneratedDispatchRecord::size + 8,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "generated_dispatch_records",
    )!!;
    defer free_owned_span(&device, &records)!!;
    OwnedSpan count = allocate_owned_span(
        device:       &device,
        size:         8,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "generated_dispatch_count",
    )!!;
    defer free_owned_span(&device, &count)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    bool original_enabled = state.generated_work.enabled;
    uint original_limit = state.caps.max_generated_work_count;
    state.generated_work.enabled = true;
    state.caps.max_generated_work_count = 1;
    defer {
        state.generated_work.enabled = original_enabled;
        state.caps.max_generated_work_count = original_limit;
    }

    gpu::GpuSpan valid_records = records.span.checked_subspan(
        0,
        gpu::GeneratedDispatchRecord::size,
    )!!;
    gpu::GpuSpan misaligned_records = records.span.checked_subspan(
        4,
        gpu::GeneratedDispatchRecord::size,
    )!!;
    gpu::GpuSpan short_records = records.span.checked_subspan(
        0,
        gpu::GeneratedDispatchRecord::size - 1,
    )!!;
    gpu::GpuSpan valid_count = count.span.checked_subspan(0, 4)!!;
    gpu::GpuSpan misaligned_count = count.span.checked_subspan(2, 4)!!;
    gpu::GpuSpan short_count = count.span.checked_subspan(0, 2)!!;

    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    bind_test_compute_pipeline(&commands, compute.pipeline);

    gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           {},
        count_span:        {},
        max_dispatch_count: 0,
    )!!;
    fault over_limit = @catch(gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           valid_records,
        count_span:        valid_count,
        max_dispatch_count: 2,
    ));
    assert(over_limit == gpu::INVALID_ARGUMENT,
        "generated dispatch rejects counts above the selected-device limit");
    fault record_alignment = @catch(gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           misaligned_records,
        count_span:        valid_count,
        max_dispatch_count: 1,
    ));
    assert(record_alignment == gpu::INVALID_ARGUMENT,
        "generated records require eight-byte alignment");
    fault record_size = @catch(gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           short_records,
        count_span:        valid_count,
        max_dispatch_count: 1,
    ));
    assert(record_size == gpu::INVALID_ARGUMENT,
        "generated record span must hold the declared maximum count");
    fault count_alignment = @catch(gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           valid_records,
        count_span:        misaligned_count,
        max_dispatch_count: 1,
    ));
    assert(count_alignment == gpu::INVALID_ARGUMENT,
        "generated count span requires four-byte alignment");
    fault count_size = @catch(gpu::cmd_dispatch_generated(
        commands:          &commands,
        records:           valid_records,
        count_span:        short_count,
        max_dispatch_count: 1,
    ));
    assert(count_size == gpu::INVALID_ARGUMENT,
        "generated count span must hold one uint");

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
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

    bind_test_graphics_pipeline(&cmd, ctx.graphics);
    fault beyond = @catch(gpu::cmd_draw_indexed(
        commands:       &cmd,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
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

    bind_test_graphics_pipeline(&cmd, ctx.graphics);
    gpu::cmd_draw_indexed(
        commands:       &cmd,
        vertex_root:    (gpu::GpuAddress)1,
        fragment_root:  (gpu::GpuAddress)1,
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
    record_val_texture_transition(&cmd, texture, gpu::TextureUse.TRANSFER_DESTINATION);

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
    record_val_texture_transition(&cmd, texture, gpu::TextureUse.TRANSFER_SOURCE);

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
    record_val_texture_transition(&cmd, texture, gpu::TextureUse.TRANSFER_DESTINATION);

    gpu::BufferTextureCopyDesc up = { .src = boundary, .texture = texture, .width = 4, .height = 4 };
    gpu::cmd_copy_buffer_to_texture(&cmd, &up)!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void barrier_empty_before_faults_without_poisoning_recording() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    gpu::Barrier barrier = {
        .after = { .compute = true },
    };

    fault rejected = @catch(gpu::cmd_barrier(&commands, &barrier));
    assert(rejected == gpu::INVALID_ARGUMENT,
        "an empty producer stage mask faults INVALID_ARGUMENT");

    barrier.before.compute = true;
    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void barrier_combined_stage_masks_succeed() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    gpu::Barrier barrier = {
        .before = {
            .compute = true,
            .color_output = true,
        },
        .after = {
            .vertex_shader = true,
            .fragment_shader = true,
        },
    };

    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void barrier_contradictory_mask_faults() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    gpu::Barrier barrier = {
        .before = {
            .all = true,
            .compute = true,
        },
        .after = { .compute = true },
    };

    fault rejected = @catch(gpu::cmd_barrier(&commands, &barrier));
    assert(rejected == gpu::INVALID_ARGUMENT,
        "all combined with a specific stage faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void barrier_descriptor_hazard_requires_shader_consumer() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    gpu::Barrier barrier = {
        .before = { .transfer = true },
        .after = { .transfer = true },
        .hazards = { .descriptors = true },
    };

    fault rejected = @catch(gpu::cmd_barrier(&commands, &barrier));
    assert(rejected == gpu::INVALID_ARGUMENT,
        "descriptor hazards without a shader consumer fault INVALID_ARGUMENT");

    barrier.after = { .fragment_shader = true };
    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void barrier_queue_stage_mismatch_faults_without_poisoning_recording() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state_ptr(&device);
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, record.queue.id)!!;
    vk::QueueFamilyFlags selected_flags = completion.queue_flags;
    defer completion.queue_flags = selected_flags;
    completion.queue_flags = {};
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .transfer = true },
    };

    fault rejected = @catch(gpu::cmd_barrier(&commands, &barrier));
    assert(rejected == gpu::INVALID_ARGUMENT,
        "queue-incompatible stages fault INVALID_ARGUMENT");

    completion.queue_flags = selected_flags;
    barrier.before = { .host = true };
    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void texture_barrier_queue_stage_mismatch_faults_without_poisoning_recording() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .transfer },
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state_ptr(&device);
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, record.queue.id)!!;
    vk::QueueFamilyFlags selected_flags = completion.queue_flags;
    defer completion.queue_flags = selected_flags;
    completion.queue_flags = TRANSFER_ONLY_QUEUE_FLAGS;
    gpu::TextureBarrier barrier = gpu::texture_transition(
        texture: texture,
        before:  gpu::TextureUse.COLOR_ATTACHMENT,
        after:   gpu::TextureUse.TRANSFER_DESTINATION,
    )!!;
    uint references_before = record.reference_count;

    assert(@catch(gpu::cmd_texture_barrier(&commands, &barrier))
        == gpu::INVALID_ARGUMENT,
        "queue-incompatible before use faults INVALID_ARGUMENT");
    assert(record.reference_count == references_before,
        "before-use rejection retains no texture reference");

    barrier.before = gpu::TextureUse.UNDEFINED;
    barrier.after = gpu::TextureUse.COLOR_ATTACHMENT;
    assert(@catch(gpu::cmd_texture_barrier(&commands, &barrier))
        == gpu::INVALID_ARGUMENT,
        "queue-incompatible after use faults INVALID_ARGUMENT");
    assert(record.reference_count == references_before,
        "after-use rejection retains no texture reference");

    completion.queue_flags = selected_flags;
    barrier.after = gpu::TextureUse.TRANSFER_DESTINATION;
    gpu::cmd_texture_barrier(&commands, &barrier)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void barrier_hazard_matrix_submits() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    gpu::Barrier[] barriers = {
        {
            .before = { .host = true },
            .after = { .transfer = true },
        },
        {
            .before = { .transfer = true },
            .after = { .compute = true },
        },
        {
            .before = { .compute = true },
            .after = { .compute = true },
            .hazards = { .draw_arguments = true },
        },
        {
            .before = { .compute = true },
            .after = { .fragment_shader = true },
            .hazards = { .descriptors = true },
        },
        {
            .before = { .compute = true },
            .after = { .compute = true },
            .hazards = { .depth_stencil = true },
        },
        {
            .before = {
                .color_output = true,
                .depth_output = true,
            },
            .after = { .fragment_shader = true },
        },
        {
            .before = { .color_output = true },
            .after = { .present = true },
        },
    };
    foreach (&barrier : barriers) {
        gpu::cmd_barrier(&commands, barrier)!!;
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
    gpu::TextureHandle invalid_range = make_val_color_target(
        device:     &device,
        width:      16,
        height:     16,
        mip_levels: 5,
    );
    defer gpu::destroy_texture(&device, invalid_range)!!;
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
        { .texture = invalid_range, .mip_level = 99 },
    };
    gpu::RenderPassDesc invalid_later_pass = { .colors = invalid_later[..], .width = 4, .height = 4 };
    assert(@catch(gpu::cmd_begin_render_pass(&cmd, &invalid_later_pass)) == gpu::INVALID_ARGUMENT,
        "invalid later mip range faults before recording");
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

    bind_test_compute_pipeline(&cmd, ctx.pipeline);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     (gpu::GpuAddress)1,
        groups:   {},
    )!!;
    bind_test_compute_pipeline(&cmd, ctx.pipeline);
    gpu::cmd_dispatch(
        commands: &cmd,
        root:     (gpu::GpuAddress)1,
        groups:   limited_caps.max_compute_work_group_count,
    )!!;

    reset_sync_diagnostic(
        capture:        &capture,
        operation:      "cmd_dispatch",
        debug_name:     null,
        rejected_field: "groups",
        invariant:      "group counts must not exceed DeviceCaps.max_compute_work_group_count",
    );
    bind_test_compute_pipeline(&cmd, ctx.pipeline);
    fault dispatch_over = @catch(gpu::cmd_dispatch(
        commands: &cmd,
        root:     (gpu::GpuAddress)1,
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

fn void render_pass_view_cache_grows_past_initial_capacity() @test {
    gpu::Device device = make_val_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_val_color_target(
        device:     &device,
        width:      512,
        height:     512,
        mip_levels: 10,
    );
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::CommandList cmd = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&cmd)!!;
    record_val_target_to_attach(&cmd, texture);

    for (uint mip = 0; mip <= vk::INITIAL_CACHED_VIEW_CAPACITY; mip++) {
        gpu::ColorTargetDesc[1] colors = {
            { .texture = texture, .mip_level = mip },
        };
        gpu::RenderPassDesc pass = {
            .colors = colors[..],
            .width  = 1,
            .height = 1,
        };
        gpu::cmd_begin_render_pass(&cmd, &pass)!!;
        gpu::cmd_end_render_pass(&cmd)!!;
    }

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(val_cached_view_count(&device, texture)
        == vk::INITIAL_CACHED_VIEW_CAPACITY + 1,
        "render-pass subresource views grow past the initial cache capacity");
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    assert(slot.views.len > vk::INITIAL_CACHED_VIEW_CAPACITY,
        "render-pass cache storage grows with distinct live native views");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
}

fn void barrier_mask_diagnostic_identifies_semantic_field() @test {
    SyncDiagnosticCapture capture = {
        .expected_operation      = "cmd_barrier",
        .expected_rejected_field = "Barrier.before",
        .expected_invariant      = "stage mask must be nonempty and noncontradictory",
    };
    gpu::Device device = make_sync_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    gpu::Barrier barrier = {
        .before = {
            .all = true,
            .compute = true,
        },
        .after = { .compute = true },
    };

    fault with_callback = @catch(gpu::cmd_barrier(&commands, &barrier));

    assert(with_callback == gpu::INVALID_ARGUMENT,
        "contradictory barrier masks keep INVALID_ARGUMENT");
    assert(capture.calls == 1, "barrier rejection emits exactly one diagnostic");
    assert(capture.operation_matches, "operation is cmd_barrier");
    assert(capture.has_fault && capture.public_fault == with_callback,
        "diagnostic carries the unchanged barrier fault");
    assert(capture.public_contract,
        "barrier rejection is a public-contract diagnostic");
    assert(capture.resource_kind == gpu::DebugResourceKind.NONE,
        "global barriers have no resource identity");
    assert(!capture.has_public_identity,
        "global barriers expose no public resource identity");
    assert(capture.debug_name_matches,
        "global barriers expose no resource debug name");
    assert(capture.rejected_field_matches,
        "the contradictory producer field is identified");
    assert(capture.invariant_matches,
        "the semantic stage-mask invariant is identified");

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    state.debug_callback = null;
    fault without_callback = @catch(gpu::cmd_barrier(&commands, &barrier));
    assert(without_callback == with_callback,
        "callback presence does not alter the barrier fault");
    assert(capture.calls == 1,
        "a null callback emits no additional diagnostic");

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
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
        commands: &commands,
        texture:  texture,
        after:    gpu::TextureUse.TRANSFER_DESTINATION,
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
    bind_test_compute_pipeline(&commands, draw.graphics);
    fault pipeline_fault = @catch(gpu::cmd_dispatch(
        commands:    &commands,
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
    bind_test_compute_pipeline(&commands, compute.pipeline);
    fault span_fault = @catch(gpu::cmd_dispatch_indirect(
        commands: &commands,
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
    bind_test_compute_pipeline(&commands, compute.pipeline);
    fault span_without_callback = @catch(gpu::cmd_dispatch_indirect(
        commands: &commands,
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
