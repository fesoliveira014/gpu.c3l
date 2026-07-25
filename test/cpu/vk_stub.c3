module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import std::atomic;
import std::core::mem;
import std::thread;

const uint  STUB_SAMPLER_CAPACITY       @private = 16;
const float STUB_SAMPLER_LOD_BIAS_LIMIT @private = 16.0f;
const float STUB_SAMPLER_ANISOTROPY_LIMIT @private = 16.0f;

struct VkRuntimeState @public {
    gpu::Runtime runtime;
    uint         adapter_count;
}

struct StubSamplerSlot {
    SamplerKey        key;
    gpu::SamplerIndex index;
}

struct VkDeviceState @public {
    gpu::Adapter         adapter;
    gpu::Surface         surface;
    gpu::Device          owner;
    gpu::GpuAllocation   allocation;
    gpu::AllocationDesc  allocation_desc;
    gpu::TextureHandle   texture;
    gpu::SwapchainHandle swapchain;
    char[128]            allocation_mapping;
    StubSamplerSlot[STUB_SAMPLER_CAPACITY] samplers;
    Atomic{bool}         sampler_lock;
    ulong                resource_owner;
    uint                 sampler_count;
    uint                 sampler_intern_attempts;
    uint                 sampler_native_creates;
    uint                 allocation_attempts;
    uint                 allocation_free_calls;
    uint                 allocation_flush_calls;
    uint                 allocation_invalidate_calls;
    uint                 allocation_visibility_entries;
    uint                 texture_support_queries;
    uint                 texture_requirements_calls;
    uint                 texture_create_attempts;
    uint                 texture_placed_calls;
    uint                 texture_dedicated_calls;
    uint                 root_diagnostic_calls;
    ZString              root_diagnostic_operation;
    fault                root_diagnostic_fault;
    gpu::DebugResourceKind root_diagnostic_resource_kind;
    ZString              root_diagnostic_rejected_field;
    ZString              root_diagnostic_invariant;
    Atomic{bool}         destroy_entered;
    Atomic{bool}         destroy_release;
    Atomic{bool}         check_entered;
    Atomic{bool}         check_release;
    Atomic{uint}         check_calls;
    Atomic{uint}         destroy_calls;
    Atomic{bool}         memory_stats_entered;
    Atomic{bool}         memory_stats_release;
    bool                 allocation_live;
    bool                 allocation_has_texture_placement;
    bool                 texture_live;
    bool                 swapchain_live;
    bool                 block_destroy;
    bool                 block_check;
    bool                 destroy_busy;
    bool                 publish_child_on_check;
    bool                 allocation_block_free;
    bool                 allocation_out_of_memory;
    bool                 texture_out_of_memory;
    bool                 block_memory_stats;
    bool                 free_on_destroy;
}

uint configured_adapter_count @private = 1;
bool configured_strict_support @private = true;
bool configured_resource_agnostic_texture_sync_support @private = true;
uint device_support_queries @private;
uint resource_agnostic_texture_sync_support_queries @private;
bool configured_dynamic_color_state_support @private = true;
uint dynamic_color_state_support_queries @private;
uint device_temporary_creates @private;
uint device_temporary_destroys @private;
bool fail_device_after_temporary_allocation @private;
gpu::Adapter last_device_adapter @private;
uint runtime_creates @private;
uint runtime_destroys @private;
bool probe_destroy_visibility @private;
fault destroy_visibility_fault @private;
uint surface_creates @private;
uint surface_destroys @private;
bool presentation_supported @private = true;
uint swapchain_create_attempts @private;
uint allocation_create_attempts @private;
uint allocation_free_attempts @private;
uint texture_destroy_attempts @private;
uint sampler_native_destroys @private;

fn void lock_stub_samplers(VkDeviceState* state) @private {
    while (state.sampler_lock.set(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
}

fn void unlock_stub_samplers(VkDeviceState* state) @private {
    (void)state.sampler_lock.clear(AtomicOrdering.RELEASE);
}

fn gpu::AttachmentViewHandle? vk_create_attachment_view(
    VkDeviceState* state,
    gpu::AttachmentViewDesc* desc,
) {
    (void)state;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_destroy_attachment_view(
    VkDeviceState* state,
    gpu::AttachmentViewHandle view,
) {
    (void)state;
    (void)view;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::TextureView? vk_create_texture_view(
    VkDeviceState* state,
    gpu::TextureHandle texture,
    gpu::TextureViewDesc* desc,
) {
    (void)state;
    (void)texture;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_destroy_texture_view(
    VkDeviceState* state,
    gpu::TextureView view,
) {
    (void)state;
    (void)view;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_create_texture_views(
    VkDeviceState* state,
    gpu::TextureViewCreateDesc[] descs,
    gpu::TextureView[] out_views,
) {
    (void)state;
    (void)descs;
    (void)out_views;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::SamplerIndex? vk_intern_sampler(
    VkDeviceState* state,
    gpu::SamplerDesc* desc,
) {
    if (state == null || desc == null) return gpu::INVALID_ARGUMENT~;
    lock_stub_samplers(state);
    defer unlock_stub_samplers(state);
    state.sampler_intern_attempts++;
    SamplerKey key = canonical_sampler_key(desc);
    for (uint i = 0; i < state.sampler_count; i++) {
        if (sampler_key_equal(&state.samplers[i].key, &key)) {
            return state.samplers[i].index;
        }
    }
    if (state.sampler_count >= STUB_SAMPLER_CAPACITY) {
        return gpu::SLOT_TABLE_FULL~;
    }
    uint slot = state.sampler_count;
    gpu::SamplerIndex index = { .value = slot + 1 };
    state.samplers[slot] = { .key = key, .index = index };
    state.sampler_count++;
    state.sampler_native_creates++;
    return index;
}

fn gpu::PipelineHandle? vk_create_compute_pipeline(
    VkDeviceState* state,
    gpu::ComputePipelineDesc* desc,
) {
    (void)state;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::PipelineHandle? vk_create_graphics_pipeline(
    VkDeviceState* state,
    gpu::GraphicsPipelineDesc* desc,
) {
    (void)state;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::PipelineHandle? vk_create_dynamic_graphics_pipeline(
    VkDeviceState* state,
    gpu::DynamicGraphicsPipelineDesc* desc,
) {
    (void)state;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_create_compute_pipelines(
    VkDeviceState* state,
    gpu::ComputePipelineDesc[] descs,
    gpu::PipelineHandle[] out_pipelines,
) {
    (void)state;
    (void)descs;
    (void)out_pipelines;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_create_graphics_pipelines(
    VkDeviceState* state,
    gpu::GraphicsPipelineDesc[] descs,
    gpu::PipelineHandle[] out_pipelines,
) {
    (void)state;
    (void)descs;
    (void)out_pipelines;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_create_dynamic_graphics_pipelines(
    VkDeviceState* state,
    gpu::DynamicGraphicsPipelineDesc[] descs,
    gpu::PipelineHandle[] out_pipelines,
) {
    (void)state;
    (void)descs;
    (void)out_pipelines;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_destroy_pipeline(
    VkDeviceState* state,
    gpu::PipelineHandle pipeline,
) {
    (void)state;
    (void)pipeline;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn usz? vk_get_pipeline_cache_size(VkDeviceState* state) {
    (void)state;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn usz? vk_get_pipeline_cache_data(
    VkDeviceState* state,
    char[] out,
) {
    (void)state;
    (void)out;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn uint stub_sampler_intern_attempt_count(gpu::Device* device) {
    VkDeviceState* state = gpu::internal::device_backend_state_ptr(device);
    lock_stub_samplers(state);
    defer unlock_stub_samplers(state);
    return state.sampler_intern_attempts;
}

fn uint stub_sampler_native_create_count(gpu::Device* device) {
    VkDeviceState* state = gpu::internal::device_backend_state_ptr(device);
    lock_stub_samplers(state);
    defer unlock_stub_samplers(state);
    return state.sampler_native_creates;
}

fn uint stub_sampler_native_destroy_count() => sampler_native_destroys;

fn void destroy_vk_runtime_state(VkRuntimeState* state) {
    if (probe_destroy_visibility) {
        destroy_visibility_fault = @catch(gpu::enumerate_adapters(&state.runtime));
        probe_destroy_visibility = false;
    }
    runtime_destroys++;
    mem::free(state);
}

fn gpu::AdapterInfo? vk_get_adapter_info(VkRuntimeState* state, uint index) {
    device_support_queries++;
    if (state == null || index >= state.adapter_count) return gpu::INVALID_HANDLE~;
    return {
        .name             = index == 0 ? "stub adapter 0" : "stub adapter 1",
        .device_class     = gpu::AdapterClass.SOFTWARE,
        .memory           = {
            .device_local_bytes = 1024,
            .host_visible_bytes = 1024,
        },
        .queues           = {
            .graphics_count = 1,
            .compute_count  = 1,
            .transfer_count = 1,
        },
        .strict_supported = configured_strict_support,
    };
}

fn gpu::AdapterDiagnostics? vk_get_adapter_diagnostics(
    VkRuntimeState* state,
    uint index,
) {
    if (state == null || index >= state.adapter_count) return gpu::INVALID_HANDLE~;
    return {
        .backend_name = "stub",
        .driver_name  = "stub driver",
    };
}

fn void*? vk_create_win32_surface(
    VkRuntimeState* state,
    void* instance,
    void* window,
) {
    if (state == null || instance == null || window == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    surface_creates++;
    return mem::new(char);
}

fn void*? vk_create_wayland_surface(
    VkRuntimeState* state,
    void* display,
    void* surface,
) {
    return vk_create_win32_surface(
        state,
        display,
        surface,
    );
}

fn void*? vk_create_x11_surface(
    VkRuntimeState* state,
    void* display,
    ulong window,
) {
    if (state == null || display == null || window == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    surface_creates++;
    return mem::new(char);
}

fn void vk_destroy_surface(VkRuntimeState* state, void* surface) {
    if (state == null || surface == null) return;
    surface_destroys++;
    mem::free(surface);
}

fn bool? vk_supports_presentation(
    VkRuntimeState* state,
    uint adapter_index,
    void* surface,
) {
    if (state == null || surface == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_HANDLE~;
    }
    return presentation_supported;
}

fn bool? vk_supports_queue_requirements(
    VkRuntimeState* state,
    uint adapter_index,
    gpu::QueueRequirements* requirements,
    void* backend_surface,
) {
    (void)backend_surface;
    if (state == null || requirements == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_HANDLE~;
    }
    gpu::internal::QueueSelection? selection = gpu::internal::queue_selection_for_requirements(
        requirements:               *requirements,
        compute_aliases_graphics:    true,
        transfer_aliases_graphics:   true,
        transfer_aliases_compute:    true,
    );
    if (catch selection_fault = selection) {
        if (selection_fault == gpu::UNSUPPORTED_FEATURE) return false;
        return selection_fault~;
    }
    return true;
}

fn bool? vk_supports_resource_agnostic_texture_sync(
    VkRuntimeState* state,
    uint adapter_index,
) {
    if (state == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_HANDLE~;
    }
    resource_agnostic_texture_sync_support_queries++;
    return configured_resource_agnostic_texture_sync_support;
}

fn bool? vk_supports_dynamic_color_state(
    VkRuntimeState* state,
    uint adapter_index,
) {
    if (state == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_HANDLE~;
    }
    dynamic_color_state_support_queries++;
    return configured_dynamic_color_state_support;
}

fn gpu::Runtime? create_vk_runtime(gpu::RuntimeDesc* desc) {
    VkRuntimeState* state = mem::new(VkRuntimeState);
    state.adapter_count = configured_adapter_count;
    runtime_creates++;
    defer catch destroy_vk_runtime_state(state);
    gpu::Runtime runtime = gpu::internal::register_runtime(
        state:         state,
        adapter_count: state.adapter_count,
    )!;
    state.runtime = runtime;
    return runtime;
}

fn void stub_set_adapter_count(uint count) {
    configured_adapter_count = count;
    configured_strict_support = true;
    configured_resource_agnostic_texture_sync_support = true;
    configured_dynamic_color_state_support = true;
}

fn void stub_set_strict_support(bool supported) => configured_strict_support = supported;

fn void stub_set_resource_agnostic_texture_sync_support(bool supported) {
    configured_resource_agnostic_texture_sync_support = supported;
}

fn void stub_set_dynamic_color_state_support(bool supported) {
    configured_dynamic_color_state_support = supported;
}

fn void stub_reset_device_counts() {
    device_support_queries = 0;
    resource_agnostic_texture_sync_support_queries = 0;
    dynamic_color_state_support_queries = 0;
    device_temporary_creates = 0;
    device_temporary_destroys = 0;
    fail_device_after_temporary_allocation = false;
    last_device_adapter = {};
}

fn uint stub_device_support_query_count() => device_support_queries;

fn uint stub_resource_agnostic_texture_sync_support_query_count() {
    return resource_agnostic_texture_sync_support_queries;
}

fn uint stub_dynamic_color_state_support_query_count() {
    return dynamic_color_state_support_queries;
}

fn uint stub_device_temporary_create_count() => device_temporary_creates;

fn uint stub_device_temporary_destroy_count() => device_temporary_destroys;

fn void stub_fail_device_after_temporary_allocation(bool fail) {
    fail_device_after_temporary_allocation = fail;
}

fn gpu::Adapter stub_last_device_adapter() => last_device_adapter;

fn gpu::Adapter? stub_device_adapter(gpu::Device* device) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    VkDeviceState* state = gpu::internal::device_backend_state_ptr(device);
    if (state == null) return gpu::INVALID_HANDLE~;
    return state.adapter;
}

fn void stub_probe_destroy_visibility() {
    probe_destroy_visibility = true;
}

fn fault stub_destroy_visibility_fault() => destroy_visibility_fault;

fn void stub_reset_runtime_counts() {
    runtime_creates = 0;
    runtime_destroys = 0;
}

fn uint stub_runtime_create_count() => runtime_creates;

fn uint stub_runtime_destroy_count() => runtime_destroys;

fn void stub_reset_surface_counts() {
    surface_creates = 0;
    surface_destroys = 0;
    presentation_supported = true;
    swapchain_create_attempts = 0;
}

fn uint stub_surface_create_count() => surface_creates;

fn uint stub_surface_destroy_count() => surface_destroys;

fn uint stub_swapchain_create_attempt_count() => swapchain_create_attempts;

fn void stub_reset_allocation_counts() {
    allocation_create_attempts = 0;
    allocation_free_attempts = 0;
}

fn uint stub_allocation_free_attempt_count() => allocation_free_attempts;

fn void stub_reset_texture_counts() {
    texture_destroy_attempts = 0;
}

fn uint stub_texture_destroy_attempt_count() => texture_destroy_attempts;

fn void stub_set_presentation_support(bool supported) {
    presentation_supported = supported;
}

fn void? vk_check_device_destroy(VkDeviceState* state) {
    (void)state.check_calls.add(1, AtomicOrdering.RELEASE);
    state.check_entered.store(true, AtomicOrdering.RELEASE);
    while (state.block_check
        && !state.check_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    if (state.publish_child_on_check) {
        (void)gpu::internal::device_slots[state.owner.index].live_children.add(
            1,
            AtomicOrdering.RELEASE,
        );
    }
    if (state.destroy_busy) return gpu::DEVICE_BUSY~;
}

fn void vk_destroy_device(VkDeviceState* state) {
    state.destroy_entered.store(true, AtomicOrdering.RELEASE);
    while (state.block_destroy
        && !state.destroy_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    if (state.swapchain_live) {
        gpu::internal::surface_release_swapchain(&state.surface)!!;
    }
    sampler_native_destroys += state.sampler_count;
    (void)state.destroy_calls.add(1, AtomicOrdering.RELEASE);
    if (state.free_on_destroy) {
        device_temporary_destroys++;
        mem::free(state);
    }
}

fn fault vk_report_root_contract_failure(
    VkDeviceState* state,
    ZString operation,
    fault public_fault,
    gpu::DebugResourceKind resource_kind,
    ZString rejected_field,
    ZString invariant,
) {
    state.root_diagnostic_calls++;
    state.root_diagnostic_operation = operation;
    state.root_diagnostic_fault = public_fault;
    state.root_diagnostic_resource_kind = resource_kind;
    state.root_diagnostic_rejected_field = rejected_field;
    state.root_diagnostic_invariant = invariant;
    return public_fault;
}

fn gpu::MemoryStats? vk_get_memory_stats(VkDeviceState* state) {
    state.memory_stats_entered.store(true, AtomicOrdering.RELEASE);
    while (state.block_memory_stats
        && !state.memory_stats_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    return {};
}

fn String? vk_build_memory_report(VkDeviceState* state, bool detailed) {
    (void)state;
    (void)detailed;
    return {};
}

fn gpu::GpuAllocation? vk_allocate_memory(
    VkDeviceState* state,
    gpu::AllocationDesc* desc,
) {
    allocation_create_attempts++;
    state.allocation_attempts++;
    if (state.allocation_out_of_memory) return gpu::OUT_OF_DEVICE_MEMORY~;
    if (state.allocation_live) return gpu::INVALID_RESOURCE_STATE~;
    state.allocation_desc = *desc;
    state.allocation = {
        .owner      = gpu::internal::resource_owner_tag(
            state.resource_owner,
            gpu::internal::ResourceOwnerKind.ALLOCATION,
        ),
        .index      = 0,
        .generation = 1,
    };
    state.allocation_live = true;
    return state.allocation;
}

fn void? vk_free_allocation(
    VkDeviceState* state,
    gpu::GpuAllocation allocation,
) {
    allocation_free_attempts++;
    state.allocation_free_calls++;
    if (!state.allocation_live) return gpu::INVALID_HANDLE~;
    if (state.allocation != allocation) return gpu::INVALID_HANDLE~;
    if (state.allocation_block_free
        || state.allocation_has_texture_placement) {
        return gpu::RESOURCE_IN_USE~;
    }
    state.allocation = {};
    state.allocation_desc = {};
    state.allocation_live = false;
}

fn gpu::AllocationInfo? vk_get_allocation_info(
    VkDeviceState* state,
    gpu::GpuAllocation allocation,
) {
    if (!state.allocation_live || allocation != state.allocation) {
        return gpu::INVALID_HANDLE~;
    }
    bool texture_memory =
        state.allocation_desc.memory_class == gpu::MemoryClass.TEXTURE;
    return {
        .size         = state.allocation_desc.size,
        .alignment    = state.allocation_desc.alignment,
        .memory_class = state.allocation_desc.memory_class,
        .access       = state.allocation_desc.access,
        .mapped       = state.allocation_desc.memory_class
            != gpu::MemoryClass.GPU_PRIVATE
            && !texture_memory,
        .coherent     = false,
        .addressable  = !texture_memory,
    };
}

fn gpu::GpuSpan? vk_get_allocation_span(
    VkDeviceState* state,
    gpu::GpuAllocation allocation,
) {
    if (!state.allocation_live || allocation != state.allocation) {
        return gpu::INVALID_HANDLE~;
    }
    if (state.allocation_desc.memory_class == gpu::MemoryClass.TEXTURE) {
        return gpu::INVALID_ARGUMENT~;
    }
    return {
        .owner      = allocation.owner,
        .index      = allocation.index,
        .generation = allocation.generation,
        .size       = state.allocation_desc.size,
    };
}

fn char[]? vk_get_span_mapping(VkDeviceState* state, gpu::GpuSpan span) {
    if (!state.allocation_live || span.owner != state.allocation.owner
        || span.index != state.allocation.index
        || span.generation != state.allocation.generation) {
        return gpu::INVALID_HANDLE~;
    }
    if (span.size == 0
        || state.allocation_desc.memory_class == gpu::MemoryClass.GPU_PRIVATE
        || state.allocation_desc.memory_class == gpu::MemoryClass.TEXTURE) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (!gpu::internal::range_fits(
        state.allocation_desc.size,
        span.offset,
        span.size,
    )) {
        return gpu::INVALID_ARGUMENT~;
    }
    return state.allocation_mapping[span.offset:span.size];
}

fn gpu::GpuAddress? vk_get_span_address(
    VkDeviceState* state,
    gpu::GpuSpan span,
) {
    if (!state.allocation_live || span.owner != state.allocation.owner
        || span.index != state.allocation.index
        || span.generation != state.allocation.generation) {
        return gpu::INVALID_HANDLE~;
    }
    if (span.size == 0
        || state.allocation_desc.memory_class == gpu::MemoryClass.TEXTURE
        || !gpu::internal::range_fits(
            state.allocation_desc.size,
            span.offset,
            span.size,
        )) {
        return gpu::INVALID_ARGUMENT~;
    }
    return (gpu::GpuAddress)(0x1000 + span.offset);
}

fn void? vk_flush_mapped_span(VkDeviceState* state, gpu::GpuSpan span) {
    state.allocation_visibility_entries++;
    (void)vk_get_span_mapping(state, span)!;
    state.allocation_flush_calls++;
}

fn void? vk_invalidate_mapped_span(VkDeviceState* state, gpu::GpuSpan span) {
    state.allocation_visibility_entries++;
    (void)vk_get_span_mapping(state, span)!;
    state.allocation_invalidate_calls++;
}

fn gpu::TextureCompatibility stub_texture_compatibility(ulong owner, uint value) {
    return (gpu::TextureCompatibility)(((uint128)owner << 64) | (uint128)value);
}

fn gpu::TextureFormatSupport? vk_get_texture_format_support(
    VkDeviceState* state,
    gpu::Format format,
) {
    (void)state;
    if (format == gpu::Format.UNDEFINED) return {};
    return {
        .features = {
            .sampled       = true,
            .storage       = true,
            .color_attach  = true,
            .depth_attach  = true,
            .transfer_src  = true,
            .transfer_dst  = true,
            .linear_filter = true,
        },
        .sample_counts = { .one = true },
    };
}

fn bool? vk_supports_texture_desc(
    VkDeviceState* state,
    gpu::TextureDesc* desc,
) {
    state.texture_support_queries++;
    return gpu::internal::texture_desc_valid(desc);
}

fn gpu::TextureRequirements? vk_get_texture_requirements(
    VkDeviceState* state,
    gpu::TextureDesc* desc,
) {
    state.texture_requirements_calls++;
    if (!gpu::internal::texture_desc_valid(desc)) {
        return gpu::INVALID_ARGUMENT~;
    }
    return {
        .size      = 64,
        .alignment = 64,
        .compatibility = stub_texture_compatibility(
            gpu::internal::resource_owner_tag(
                state.resource_owner,
                gpu::internal::ResourceOwnerKind.ALLOCATION,
            ),
            1,
        ),
    };
}

fn gpu::TextureHandle? vk_create_texture(
    VkDeviceState* state,
    gpu::TextureDesc* desc,
) {
    state.texture_create_attempts++;
    if (!gpu::internal::texture_desc_valid(desc)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (state.texture_out_of_memory) return gpu::OUT_OF_DEVICE_MEMORY~;
    if (state.texture_live) return gpu::INVALID_RESOURCE_STATE~;
    state.texture = {
        .owner      = gpu::internal::resource_owner_tag(state.resource_owner, gpu::internal::ResourceOwnerKind.TEXTURE),
        .index      = 0,
        .generation = 1,
    };
    state.texture_live = true;
    return state.texture;
}

fn gpu::TextureHandle? vk_create_placed_texture(
    VkDeviceState* state,
    gpu::TextureDesc* desc,
    gpu::GpuAllocation placement,
    usz offset,
) {
    state.texture_placed_calls++;
    gpu::TextureRequirements requirements =
        vk_get_texture_requirements(state, desc)!;
    if (!state.allocation_live
        || state.allocation != placement
        || state.allocation_desc.memory_class != gpu::MemoryClass.TEXTURE
        || state.allocation_desc.size < requirements.size
        || offset > state.allocation_desc.size - requirements.size) {
        return gpu::INVALID_ARGUMENT~;
    }
    gpu::TextureHandle texture = vk_create_texture(state, desc)!;
    state.allocation_has_texture_placement = true;
    return texture;
}

fn gpu::DedicatedTexture? vk_create_dedicated_texture(
    VkDeviceState* state,
    gpu::TextureDesc* desc,
    gpu::AllocationDesc* allocation_desc,
) {
    state.texture_dedicated_calls++;
    gpu::GpuAllocation allocation =
        vk_allocate_memory(state, allocation_desc)!;
    gpu::TextureHandle? texture = vk_create_placed_texture(
        state:     state,
        desc:      desc,
        placement: allocation,
        offset:    0,
    );
    if (catch err = texture) {
        vk_free_allocation(state, allocation)!!;
        return err~;
    }
    return {
        .texture    = texture,
        .allocation = allocation,
    };
}

fn void? vk_destroy_texture(
    VkDeviceState* state,
    gpu::TextureHandle texture,
) {
    texture_destroy_attempts++;
    if (!state.texture_live) return gpu::INVALID_HANDLE~;
    if (state.texture != texture) return gpu::INVALID_HANDLE~;
    state.texture = {};
    state.texture_live = false;
    state.allocation_has_texture_placement = false;
}

fn gpu::SwapchainHandle? create_stub_swapchain(
    gpu::Device* device,
    gpu::Surface* surface,
    gpu::SwapchainDesc* desc,
) {
    VkDeviceState* state = gpu::internal::device_backend_state_ptr(device);
    swapchain_create_attempts++;
    if (state == null || surface == null || desc == null) return gpu::INVALID_ARGUMENT~;
    if (state.swapchain_live) return gpu::INVALID_RESOURCE_STATE~;

    gpu::internal::surface_retain_swapchain(surface)!;
    state.surface = *surface;
    state.swapchain = {
        .owner      = gpu::internal::resource_owner_tag(state.resource_owner, gpu::internal::ResourceOwnerKind.SWAPCHAIN),
        .index      = 0,
        .generation = 1,
    };
    state.swapchain_live = true;
    return state.swapchain;
}

fn void? destroy_stub_swapchain(
    gpu::Device* device,
    gpu::SwapchainHandle swapchain,
) {
    VkDeviceState* state = gpu::internal::device_backend_state_ptr(device);
    if (state == null || !state.swapchain_live) return gpu::INVALID_HANDLE~;
    if (state.swapchain != swapchain) return gpu::INVALID_HANDLE~;

    gpu::internal::surface_release_swapchain(&state.surface)!;
    state.surface = {};
    state.swapchain = {};
    state.swapchain_live = false;
}

const gpu::internal::BackendVTable STUB_DEVICE_VTABLE @private = {
    .create_swapchain  = &create_stub_swapchain,
    .destroy_swapchain = &destroy_stub_swapchain,
};

fn gpu::Device? stub_create_strict_disabled_device() {
    VkDeviceState* state = mem::new(VkDeviceState);
    defer catch mem::free(state);
    state.free_on_destroy = true;
    state.resource_owner = gpu::internal::allocate_device_resource_owner()!;
    return gpu::internal::register_device(
        state:          state,
        caps:           { .max_sampler_lod_bias = STUB_SAMPLER_LOD_BIAS_LIMIT },
        vtable:         &STUB_DEVICE_VTABLE,
        resource_owner: state.resource_owner,
    );
}

fn gpu::Device? stub_create_anisotropy_disabled_device() {
    VkDeviceState* state = mem::new(VkDeviceState);
    defer catch mem::free(state);
    state.free_on_destroy = true;
    state.resource_owner = gpu::internal::allocate_device_resource_owner()!;
    return gpu::internal::register_device(
        state: state,
        caps: {
            .strict_enabled       = true,
            .max_sampler_lod_bias = STUB_SAMPLER_LOD_BIAS_LIMIT,
        },
        vtable:         &STUB_DEVICE_VTABLE,
        resource_owner: state.resource_owner,
    );
}

fn gpu::Device? create_vk_device(
    gpu::Adapter* adapter,
    gpu::DeviceRequest* request,
    gpu::Runtime* retained_runtime,
) {
    gpu::internal::validate_device_request(request)!;
    VkDeviceState* state = mem::new(VkDeviceState);
    state.free_on_destroy = true;
    state.resource_owner = gpu::internal::allocate_device_resource_owner()!;
    device_temporary_creates++;
    defer catch {
        device_temporary_destroys++;
        mem::free(state);
    }
    if (fail_device_after_temporary_allocation) return gpu::BACKEND_ERROR~;

    state.adapter = *adapter;
    last_device_adapter = *adapter;
    gpu::Surface presentation_surface = gpu::internal::requested_surface(*request);
    gpu::QueueRequirements requirements = gpu::internal::requested_queue_requirements(*request);
    gpu::internal::QueueSelection queues = gpu::internal::queue_selection_for_requirements(
        requirements:               requirements,
        compute_aliases_graphics:    true,
        transfer_aliases_graphics:   true,
        transfer_aliases_compute:    true,
    )!;
    gpu::DeviceCaps caps = {
        .strict_enabled       = true,
        .presentation_enabled = presentation_surface.is_valid(),
        .dynamic_color_state  =
            gpu::internal::requests_dynamic_color_state(*request),
        .max_sampler_lod_bias = STUB_SAMPLER_LOD_BIAS_LIMIT,
        .max_sampler_anisotropy = STUB_SAMPLER_ANISOTROPY_LIMIT,
    };
    return gpu::internal::register_device(
        state:                state,
        caps:                 caps,
        vtable:               &STUB_DEVICE_VTABLE,
        resource_owner:       state.resource_owner,
        retained_runtime:     retained_runtime,
        adapter:              adapter,
        presentation_surface: presentation_surface.is_valid()
            ? &presentation_surface
            : null,
        queues: queues,
        owner:  &state.owner,
    );
}
