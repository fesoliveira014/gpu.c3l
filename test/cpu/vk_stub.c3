module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import std::atomic;
import std::core::mem;
import std::thread;

const uint  STUB_SAMPLER_CAPACITY       @private = 16;
const float STUB_SAMPLER_LOD_BIAS_LIMIT @private = 16.0f;
const float STUB_SAMPLER_ANISOTROPY_LIMIT @private = 16.0f;
const uint STUB_COMMAND_CAPACITY @private = 64;
const uint STUB_ADAPTER_CAPACITY @private = 16;

struct CommandRecord @public {
    VkDeviceState*                   device_state;
    gpu::Device                      device;
    gpu::internal::CommandListHandle identity;
    gpu::internal::CommandState      state;
    gpu::Queue                       queue;
    char[]                           inline_root;
}

fn void? set_inline_root(CommandRecord* record, char[] inline_root) {
    if (inline_root.len % 4 != 0 || inline_root.len > gpu::INLINE_ROOT_CAPACITY) {
        return gpu::INVALID_ARGUMENT~;
    }
    record.inline_root = inline_root;
}

struct VkRuntimeState @public {
    gpu::Runtime runtime;
    uint         adapter_count;
    uint         acceleration_structure_heap_capacity;
}

fn uint vk_acceleration_structure_heap_capacity(VkRuntimeState* state) =>
    state == null ? 0 : state.acceleration_structure_heap_capacity;

fn gpu::AccelerationStructureRequirements?
vk_get_acceleration_structure_requirements(
    VkDeviceState* state,
    gpu::AccelerationStructureDesc* desc,
) {
    (void)state;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::AccelerationStructureHandle? vk_create_acceleration_structure(
    VkDeviceState* state,
    gpu::AccelerationStructureDesc* desc,
) {
    (void)state;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::AccelerationStructureHandle? vk_create_placed_acceleration_structure(
    VkDeviceState* state,
    gpu::AccelerationStructureDesc* desc,
    gpu::GpuAllocation allocation,
    usz offset,
) {
    (void)state;
    (void)desc;
    (void)allocation;
    (void)offset;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::DedicatedAccelerationStructure?
vk_create_dedicated_acceleration_structure(
    VkDeviceState* state,
    gpu::AccelerationStructureDesc* desc,
) {
    (void)state;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_destroy_acceleration_structure(
    VkDeviceState* state,
    gpu::AccelerationStructureHandle acceleration_structure,
) {
    (void)state;
    (void)acceleration_structure;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::GpuAddress? vk_get_acceleration_structure_address(
    VkDeviceState* state,
    gpu::AccelerationStructureHandle acceleration_structure,
) {
    if (state == null
        || !state.acceleration_structure_live
        || state.acceleration_structure != acceleration_structure) {
        return gpu::INVALID_HANDLE~;
    }
    if (state.acceleration_structure_kind
        != gpu::AccelerationStructureKind.BOTTOM_LEVEL) {
        return gpu::INVALID_ARGUMENT~;
    }
    return state.acceleration_structure_address;
}

fn gpu::AccelerationStructureView? vk_create_acceleration_structure_view(
    VkDeviceState* state,
    gpu::AccelerationStructureHandle acceleration_structure,
) {
    (void)state;
    (void)acceleration_structure;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_destroy_acceleration_structure_view(
    VkDeviceState* state,
    gpu::AccelerationStructureView view,
) {
    (void)state;
    (void)view;
    return gpu::UNSUPPORTED_FEATURE~;
}

struct StubSamplerSlot {
    SamplerKey        key;
    gpu::SamplerIndex index;
}

struct VkDeviceState @public {
    Atomic{bool} command_entered;
    Atomic{bool} command_release;
    uint pins_during_command;
    bool block_command;
    bool reject_command;
    gpu::Adapter         adapter;
    gpu::Surface         surface;
    gpu::Device          owner;
    gpu::Queue           allocator_queue;
    gpu::CommandAllocatorDesc allocator_desc;
    CommandRecord[STUB_COMMAND_CAPACITY] command_records;
    gpu::GpuAllocation   allocation;
    gpu::AllocationDesc  allocation_desc;
    gpu::TextureHandle   texture;
    gpu::AccelerationStructureHandle acceleration_structure;
    gpu::AccelerationStructureKind   acceleration_structure_kind;
    gpu::GpuAddress                  acceleration_structure_address;
    bool                             acceleration_structure_live;
    gpu::SwapchainHandle swapchain;
    gpu::SwapchainInfo   swapchain_info;
    char[128]            allocation_mapping;
    StubSamplerSlot[STUB_SAMPLER_CAPACITY] samplers;
    Atomic{bool}         sampler_lock;
    ulong                resource_owner;
    uint                 sampler_count;
    uint                 sampler_intern_attempts;
    uint                 sampler_native_creates;
    uint                 allocator_creates;
    uint                 allocator_destroy_calls;
    uint[STUB_COMMAND_CAPACITY] command_generations;
    uint                 next_command;
    uint                 completion_poll_calls;
    uint                 completion_wait_calls;
    uint                 completion_last_queue_id;
    uint                 pins_during_submit;
    ulong                completion_current;
    ulong                completion_last_sequence;
    ulong                completion_last_timeout_ns;
    ulong                completion_sequence;
    uint                 present_wait_pending_count;
    uint                 present_wait_calls;
    uint                 present_reset_calls;
    ulong                present_wait_last_timeout_ns;
    fault                present_wait_fault;
    fault                present_reset_fault;
    uint                 allocation_attempts;
    uint                 allocation_free_calls;
    uint                 allocation_flush_calls;
    uint                 allocation_invalidate_calls;
    uint                 allocation_visibility_entries;
    uint                 abandoned_allocation_reports;
    uint                 abandoned_allocation_index;
    uint                 abandoned_allocation_generation;
    uint                 texture_support_queries;
    uint                 texture_requirements_calls;
    uint                 texture_create_attempts;
    uint                 texture_placed_calls;
    uint                 texture_dedicated_calls;
    uint                 root_diagnostic_calls;
    ZString              root_diagnostic_operation;
    fault                root_diagnostic_fault;
    fault                completion_wait_fault;
    gpu::DebugResourceKind root_diagnostic_resource_kind;
    ZString              root_diagnostic_rejected_field;
    ZString              root_diagnostic_invariant;
    ZString              abandoned_allocation_debug_name;
    Atomic{bool}         destroy_entered;
    Atomic{bool}         destroy_release;
    Atomic{bool}         check_entered;
    Atomic{bool}         check_release;
    Atomic{uint}         check_calls;
    Atomic{uint}         destroy_calls;
    Atomic{bool}         discard_entered;
    Atomic{bool}         discard_release;
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
    bool                 mapping_unavailable;
    bool                 allocation_out_of_memory;
    bool                 texture_out_of_memory;
    bool                 block_memory_stats;
    bool                 free_on_destroy;
    bool                 reject_allocator_create;
    bool                 reject_allocator_destroy;
    bool                 reject_swapchain_destroy;
    bool                 return_invalid_allocator;
    bool                 return_foreign_allocator;
    bool[STUB_COMMAND_CAPACITY] command_used;
    bool                 reject_discard;
    bool                 reject_end;
    bool                 block_discard;
    bool                 reuse_command_index;
    bool                 submit_busy;
    bool                 lose_during_submit;
}

fn void stub_set_acceleration_structure(
    VkDeviceState* state,
    gpu::AccelerationStructureHandle acceleration_structure,
    gpu::AccelerationStructureKind kind,
    gpu::GpuAddress address,
) {
    state.acceleration_structure = acceleration_structure;
    state.acceleration_structure_kind = kind;
    state.acceleration_structure_address = address;
    state.acceleration_structure_live = true;
}

uint configured_adapter_count @private = 1;
gpu::AdapterClass[STUB_ADAPTER_CAPACITY] configured_adapter_classes @private;
bool[STUB_ADAPTER_CAPACITY] configured_adapter_class_overrides @private;
bool[STUB_ADAPTER_CAPACITY] configured_adapter_unsupported @private;
bool configured_allocator_create_failure @private;
bool configured_swapchain_create_failure @private;
bool configured_swapchain_info_override @private;
gpu::SwapchainInfo configured_swapchain_info @private;
fault configured_swapchain_info_fault @private;
bool configured_device_destroy_failure @private;
gpu::Device last_created_device @private;
uint allocator_destroy_attempts @private;
uint swapchain_destroy_attempts @private;
uint context_wait_calls @private;
bool configured_device_baseline_support @private = true;
bool configured_ray_query_support @private = true;
bool configured_ray_tracing_pipeline_support @private = true;
bool configured_ray_tracing_queue_support @private = true;
uint configured_ray_query_heap_limit @private = uint::max;
bool configured_sparse_texture_support @private;
bool configured_sparse_queue_support @private;
bool configured_single_queue_support @private = true;
bool configured_single_queue_presentation_support @private = true;
uint device_support_queries @private;
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

fn void? vk_begin_commands(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
    gpu::CommandList* out_commands,
) {
    if (state == null || allocator == null || out_commands == null) {
        return gpu::INVALID_HANDLE~;
    }
    gpu::Device device = allocator.device;
    gpu::internal::DeviceData* data =
        gpu::internal::resolve_pinned_device(&device);
    if (data == null || data.state != state) return gpu::INVALID_HANDLE~;

    state.next_command++;
    uint index = state.reuse_command_index ? 1 : state.next_command;
    if (index >= state.command_records.len) return gpu::SLOT_TABLE_FULL~;
    if (state.command_used[index]) return gpu::DEVICE_BUSY~;
    state.command_generations[index]++;
    if (state.command_generations[index] == 0) {
        state.command_generations[index] = 1;
    }
    gpu::internal::CommandListHandle identity = {
        .owner = gpu::internal::resource_owner_tag(
            data.resource_owner,
            gpu::internal::ResourceOwnerKind.COMMAND,
        ),
        .index = index,
        .generation = state.command_generations[index],
    };
    state.command_records[index] = {
        .device_state = state,
        .device       = device,
        .identity     = identity,
        .state        = gpu::internal::CommandState.RECORDING,
        .queue        = allocator.queue,
    };
    state.command_used[index] = true;
    gpu::internal::publish_command_list(
        &state.command_records[index],
        out_commands,
    );
}

fn void? vk_end_commands(
    VkDeviceState* state,
    CommandRecord* record,
) {
    if (state == null || record == null || record.device_state != state) {
        return gpu::INVALID_HANDLE~;
    }
    if (state.reject_end) return gpu::DEVICE_BUSY~;
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    record.state = gpu::internal::CommandState.EXECUTABLE;
}

fn void? vk_discard_commands(
    VkDeviceState* state,
    CommandRecord* record,
    bool executable,
) {
    if (state == null
        || record == null
        || record.device_state != state
        || !record.identity.is_valid()) {
        return gpu::INVALID_HANDLE~;
    }
    if (state.reject_discard) return gpu::DEVICE_BUSY~;
    bool state_matches = executable
        ? record.state == gpu::internal::CommandState.EXECUTABLE
        : record.state == gpu::internal::CommandState.RECORDING
            || record.state
                == gpu::internal::CommandState.RECORDING_RENDER_PASS;
    if (!state_matches) return gpu::COMMAND_RECORDING_ERROR~;
    uint index = record.identity.index;
    if (index >= state.command_records.len
        || !state.command_used[index]
        || record != &state.command_records[index]) {
        return gpu::INVALID_HANDLE~;
    }

    record.state = gpu::internal::CommandState.INACTIVE;
    state.discard_entered.store(true, AtomicOrdering.RELEASE);
    while (state.block_discard
        && !state.discard_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    state.command_records[index] = {};
    state.command_used[index] = false;
}

fn bool? vk_poll_completion(
    VkDeviceState* state,
    uint queue_id,
    ulong sequence,
) {
    if (state == null) return gpu::INVALID_HANDLE~;
    state.completion_poll_calls++;
    state.completion_last_queue_id = queue_id;
    state.completion_last_sequence = sequence;
    return state.completion_current >= sequence;
}

fn void? vk_wait_completion(
    VkDeviceState* state,
    uint queue_id,
    ulong sequence,
    ulong timeout_ns,
) {
    context_wait_calls++;
    if (state == null) return gpu::INVALID_HANDLE~;
    state.completion_wait_calls++;
    state.completion_last_queue_id = queue_id;
    state.completion_last_sequence = sequence;
    state.completion_last_timeout_ns = timeout_ns;
    if (state.completion_wait_fault) {
        return state.completion_wait_fault~;
    }
}

fn gpu::CompletionPoint? vk_submit(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::SubmitDesc* desc,
) {
    if (state == null || desc == null) return gpu::INVALID_HANDLE~;
    gpu::internal::DeviceData* data =
        gpu::internal::resolve_pinned_device(&queue.device);
    if (data == null || data.state != state) return gpu::INVALID_HANDLE~;
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)
            gpu::internal::device_slots[queue.device.index]
                .control.load(AtomicOrdering.ACQUIRE);
    state.pins_during_submit = control.pins;
    if (state.submit_busy) return gpu::DEVICE_BUSY~;
    if (state.completion_sequence == gpu::internal::MAX_COMPLETION_SEQUENCE) {
        return gpu::SLOT_TABLE_FULL~;
    }

    CommandRecord*[gpu::internal::MAX_DEVICE_COMMANDS] records;
    foreach (i, &commands : desc.command_lists) {
        CommandRecord* record =
            gpu::internal::executable_command(commands)!;
        if (record.device_state != state || record.queue != queue) {
            return gpu::INVALID_HANDLE~;
        }
        for (sz previous = 0; previous < i; previous++) {
            if (records[previous] == record) {
                return gpu::COMMAND_RECORDING_ERROR~;
            }
        }
        records[i] = record;
    }

    state.completion_sequence++;
    gpu::CompletionPoint point = gpu::internal::make_completion_point(
        queue,
        state.completion_sequence,
    )!;
    if (state.lose_during_submit) {
        gpu::internal::mark_device_lost(&queue.device);
    }
    foreach (record : records[:desc.command_lists.len]) {
        record.state = gpu::internal::CommandState.INACTIVE;
        state.command_used[record.identity.index] = false;
        gpu::internal::release_command_record_ownership(record.device);
    }
    return point;
}

fn gpu::CommandAllocatorHandle? vk_create_command_allocator(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::CommandAllocatorDesc* desc,
) {
    if (state == null || desc == null) return gpu::INVALID_HANDLE~;
    state.allocator_creates++;
    state.allocator_queue = queue;
    state.allocator_desc = *desc;
    if (configured_allocator_create_failure) return gpu::DEVICE_BUSY~;
    if (state.reject_allocator_create) return gpu::DEVICE_BUSY~;
    if (state.return_invalid_allocator) return {};
    gpu::CommandAllocatorHandle handle = {
        .owner = gpu::internal::resource_owner_tag(
            state.resource_owner,
            gpu::internal::ResourceOwnerKind.COMMAND_ALLOCATOR,
        ),
        .index = 0,
        .generation = state.allocator_creates,
    };
    if (state.return_foreign_allocator) handle.owner++;
    return handle;
}

fn void? vk_destroy_command_allocator(
    VkDeviceState* state,
    gpu::CommandAllocatorHandle handle,
) {
    if (state == null) return gpu::INVALID_HANDLE~;
    state.allocator_destroy_calls++;
    allocator_destroy_attempts++;
    if (!handle.is_valid()) return gpu::INVALID_HANDLE~;
    if (state.reject_allocator_destroy) return gpu::RESOURCE_IN_USE~;
}

fn void? vk_reserve_generated_work(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
    gpu::GeneratedWorkReservationDesc* desc,
) {
    (void)state;
    (void)allocator;
    (void)desc;
}

fn void? vk_release_generated_work(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::GeneratedWorkKind kind,
) {
    (void)state;
    (void)allocator;
    (void)pipeline;
    (void)kind;
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

fn gpu::TextureIndexRange? vk_reserve_texture_indices(VkDeviceState* state, uint count) {
    (void)state;
    (void)count;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_release_texture_indices(VkDeviceState* state, gpu::TextureIndexRange range) {
    (void)state;
    (void)range;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::TextureView? vk_create_texture_view_at(
    VkDeviceState* state,
    gpu::TextureIndex slot,
    gpu::TextureHandle texture,
    gpu::TextureViewDesc* desc,
) {
    (void)state;
    (void)slot;
    (void)texture;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_update_texture_view(
    VkDeviceState* state,
    gpu::TextureView* view,
    gpu::TextureHandle texture,
    gpu::TextureViewDesc* desc,
) {
    (void)state;
    (void)view;
    (void)texture;
    (void)desc;
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

fn gpu::PipelineHandle? vk_create_mesh_pipeline(
    VkDeviceState* state,
    gpu::MeshPipelineDesc* desc,
) {
    (void)state;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::PipelineHandle? vk_create_ray_tracing_pipeline(
    VkDeviceState* state,
    gpu::RayTracingPipelineDesc* desc,
) {
    (void)state;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::RayTracingPipelineInfo? vk_get_ray_tracing_pipeline_info(
    VkDeviceState* state,
    gpu::PipelineHandle pipeline,
) {
    (void)state;
    (void)pipeline;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_get_ray_tracing_shader_group_handles(
    VkDeviceState* state,
    gpu::PipelineHandle pipeline,
    uint first_group,
    uint group_count,
    char[] out,
) {
    (void)state;
    (void)pipeline;
    (void)first_group;
    (void)group_count;
    (void)out;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn ulong? vk_get_ray_tracing_shader_group_stack_size(
    VkDeviceState* state,
    gpu::PipelineHandle pipeline,
    uint group,
    gpu::RayTracingGroupShader group_shader,
) {
    (void)state;
    (void)pipeline;
    (void)group;
    (void)group_shader;
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
    gpu::internal::DeviceOperation operation =
        gpu::internal::begin_device_operation(device)!!;
    defer gpu::internal::end_device_operation(&operation);
    VkDeviceState* state = operation.state;
    lock_stub_samplers(state);
    defer unlock_stub_samplers(state);
    return state.sampler_intern_attempts;
}

fn uint stub_sampler_native_create_count(gpu::Device* device) {
    gpu::internal::DeviceOperation operation =
        gpu::internal::begin_device_operation(device)!!;
    defer gpu::internal::end_device_operation(&operation);
    VkDeviceState* state = operation.state;
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
        .device_class     = index < configured_adapter_classes.len
            && configured_adapter_class_overrides[index]
            ? configured_adapter_classes[index] : gpu::AdapterClass.SOFTWARE,
        .memory           = {
            .device_local_bytes = 1024,
            .host_visible_bytes = 1024,
            .device_local_host_visible_bytes = 1024,
        },
        .queues           = {
            .available = {
                .graphics,
                .compute,
                .transfer,
            },
        },
    };
}

fn bool? vk_adapter_supports_device_baseline(
    VkRuntimeState* state,
    uint adapter_index,
) {
    if (state == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_ARGUMENT~;
    }
    return configured_device_baseline_support
        && !(adapter_index < configured_adapter_unsupported.len
            && configured_adapter_unsupported[adapter_index]);
}

fn bool? vk_adapter_supports_ray_queries(
    VkRuntimeState* state,
    uint adapter_index,
) {
    if (state == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_ARGUMENT~;
    }
    return configured_ray_query_support;
}

fn bool? vk_adapter_supports_ray_tracing_pipelines(
    VkRuntimeState* state,
    uint adapter_index,
) {
    if (state == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_ARGUMENT~;
    }
    return configured_ray_tracing_pipeline_support;
}

fn bool? vk_adapter_supports_mesh_shaders(
    VkRuntimeState* state,
    uint adapter_index,
) {
    if (state == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_ARGUMENT~;
    }
    return true;
}

fn bool ray_tracing_queue_request_compatible(gpu::QueueRequest* request) {
    gpu::QueueRequest normalized = request == null
        ? gpu::internal::default_queue_request()
        : *request;
    return normalized.required.graphics || normalized.required.compute;
}

fn bool? vk_adapter_supports_ray_query_heap(
    VkRuntimeState* state,
    uint adapter_index,
    uint heap_capacity,
) {
    if (state == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_ARGUMENT~;
    }
    return heap_capacity > 0
        && heap_capacity <= configured_ray_query_heap_limit;
}

fn bool? vk_adapter_supports_sparse_textures(
    VkRuntimeState* state,
    uint adapter_index,
) {
    if (state == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_HANDLE~;
    }
    return configured_sparse_texture_support;
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

fn bool? vk_supports_queue_request(
    VkRuntimeState* state,
    uint adapter_index,
    gpu::QueueRequest* request,
    void* backend_surface,
    bool require_sparse_binding = false,
    bool require_ray_tracing = false,
) {
    (void)backend_surface;
    if (state == null || request == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_HANDLE~;
    }
    if (require_sparse_binding && !configured_sparse_queue_support) {
        return false;
    }
    if (require_ray_tracing && !configured_ray_tracing_queue_support) {
        return false;
    }
    if (request.single_queue && !configured_single_queue_support) {
        return false;
    }
    if (request.single_queue
        && backend_surface != null
        && !configured_single_queue_presentation_support) {
        return false;
    }
    gpu::internal::QueueSelection? selection = gpu::internal::queue_selection_for_request(
        request:                    *request,
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

fn gpu::Runtime? create_vk_runtime(gpu::RuntimeDesc* desc) {
    VkRuntimeState* state = mem::new(VkRuntimeState);
    state.adapter_count = configured_adapter_count;
    state.acceleration_structure_heap_capacity =
        desc.acceleration_structure_heap_capacity;
    runtime_creates++;
    defer catch destroy_vk_runtime_state(state);
    gpu::Runtime runtime = gpu::internal::register_runtime(
        state:         state,
        adapter_count: state.adapter_count,
    )!;
    state.runtime = runtime;
    if (desc.debug_callback != null) {
        gpu::DebugMessage message = {
            .severity     = gpu::DebugMessageSeverity.INFO,
            .category     = {
                .backend = true,
                .general = true,
            },
            .operation    = "create_runtime",
            .backend_text = "stub runtime initialized",
        };
        gpu::internal::dispatch_debug_message(
            desc.debug_callback,
            desc.debug_user_data,
            &message,
        );
    }
    return runtime;
}

fn void stub_set_adapter_count(uint count) {
    configured_adapter_count = count;
    configured_adapter_class_overrides = {};
    configured_adapter_unsupported = {};
    configured_device_baseline_support = true;
    configured_ray_query_support = true;
    configured_ray_tracing_pipeline_support = true;
    configured_ray_tracing_queue_support = true;
    configured_ray_query_heap_limit = uint::max;
    configured_sparse_texture_support = false;
    configured_sparse_queue_support = false;
    configured_single_queue_support = true;
    configured_single_queue_presentation_support = true;
}

fn void stub_set_adapter_class(uint index, gpu::AdapterClass device_class) {
    assert(index < configured_adapter_classes.len);
    configured_adapter_classes[index] = device_class;
    configured_adapter_class_overrides[index] = true;
}

fn void stub_set_adapter_supported(uint index, bool supported) {
    assert(index < configured_adapter_unsupported.len);
    configured_adapter_unsupported[index] = !supported;
}

fn void stub_set_context_creation_failures(bool allocator, bool swapchain, bool device_destroy = false) {
    configured_allocator_create_failure = allocator;
    configured_swapchain_create_failure = swapchain;
    configured_device_destroy_failure = device_destroy;
    configured_swapchain_info_override = false;
    configured_swapchain_info_fault = {};
}

fn void stub_set_context_swapchain_info(gpu::SwapchainInfo info, fault info_fault = {}) {
    configured_swapchain_info_override = true;
    configured_swapchain_info = info;
    configured_swapchain_info_fault = info_fault;
}

fn gpu::Device stub_last_created_device() => last_created_device;

fn void stub_reset_context_counts() {
    allocator_destroy_attempts = 0;
    swapchain_destroy_attempts = 0;
    context_wait_calls = 0;
}

fn uint stub_allocator_destroy_attempt_count() => allocator_destroy_attempts;

fn uint stub_swapchain_destroy_attempt_count() => swapchain_destroy_attempts;

fn uint stub_context_wait_count() => context_wait_calls;

fn void stub_set_device_baseline_support(bool supported) {
    configured_device_baseline_support = supported;
}

fn void stub_set_ray_query_support(bool supported) {
    configured_ray_query_support = supported;
}

fn void stub_set_ray_tracing_pipeline_support(bool supported) {
    configured_ray_tracing_pipeline_support = supported;
}

fn void stub_set_ray_tracing_queue_support(bool supported) {
    configured_ray_tracing_queue_support = supported;
}

fn void stub_set_ray_query_heap_limit(uint limit) {
    configured_ray_query_heap_limit = limit;
}

fn void stub_set_sparse_texture_support(bool supported) {
    configured_sparse_texture_support = supported;
}

fn void stub_set_sparse_queue_support(bool supported) {
    configured_sparse_queue_support = supported;
}

fn void stub_set_single_queue_support(bool supported) {
    configured_single_queue_support = supported;
}

fn void stub_set_single_queue_presentation_support(bool supported) {
    configured_single_queue_presentation_support = supported;
}

fn void stub_reset_device_counts() {
    device_support_queries = 0;
    device_temporary_creates = 0;
    device_temporary_destroys = 0;
    fail_device_after_temporary_allocation = false;
    last_device_adapter = {};
}

fn uint stub_device_support_query_count() => device_support_queries;

fn uint stub_device_temporary_create_count() => device_temporary_creates;

fn uint stub_device_temporary_destroy_count() => device_temporary_destroys;

fn void stub_fail_device_after_temporary_allocation(bool fail) {
    fail_device_after_temporary_allocation = fail;
}

fn gpu::Adapter stub_last_device_adapter() => last_device_adapter;

fn gpu::Adapter? stub_device_adapter(gpu::Device* device) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    VkDeviceState* state = operation.state;
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

fn void stub_reset_present_wait_counts(VkDeviceState* state) {
    if (state == null) return;
    state.present_wait_calls = 0;
    state.present_reset_calls = 0;
    state.present_wait_last_timeout_ns = 0;
}

fn uint stub_present_wait_call_count(VkDeviceState* state) =>
    state == null ? 0 : state.present_wait_calls;

fn uint stub_present_reset_call_count(VkDeviceState* state) =>
    state == null ? 0 : state.present_reset_calls;

fn ulong stub_present_wait_last_timeout(VkDeviceState* state) =>
    state == null ? 0 : state.present_wait_last_timeout_ns;

fn uint stub_present_wait_pending_count(VkDeviceState* state) =>
    state == null ? 0 : state.present_wait_pending_count;

fn void stub_set_present_wait_pending_count(
    VkDeviceState* state,
    uint pending_count,
) {
    if (state == null) return;
    state.present_wait_pending_count = pending_count;
}

fn void stub_set_present_wait_fault(VkDeviceState* state, fault failure) {
    if (state == null) return;
    state.present_wait_fault = failure;
}

fn void stub_set_present_reset_fault(VkDeviceState* state, fault failure) {
    if (state == null) return;
    state.present_reset_fault = failure;
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

fn gpu::TimestampPoolHandle? vk_create_timestamp_pool(
    VkDeviceState* state,
    gpu::TimestampPoolDesc* desc,
) {
    (void)state;
    (void)desc;
    return gpu::BACKEND_ERROR~;
}

fn void? vk_destroy_timestamp_pool(
    VkDeviceState* state,
    gpu::TimestampPoolHandle pool,
) {
    (void)state;
    (void)pool;
    return gpu::INVALID_HANDLE~;
}

fn void? vk_read_timestamps(
    VkDeviceState* state,
    gpu::TimestampPoolHandle pool,
    uint first,
    uint count,
    ulong[] out,
) {
    (void)state;
    (void)pool;
    (void)first;
    (void)count;
    (void)out;
    return gpu::DEVICE_BUSY~;
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

fn void vk_report_abandoned_allocation(
    VkDeviceState* state,
    gpu::GpuAllocation allocation,
    ZString debug_name,
) {
    state.abandoned_allocation_reports++;
    state.abandoned_allocation_index = allocation.index;
    state.abandoned_allocation_generation = allocation.generation;
    state.abandoned_allocation_debug_name = debug_name;
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
        .device_local = state.allocation_desc.memory_class
            == gpu::MemoryClass.GPU_PRIVATE
            || state.allocation_desc.memory_class
            == gpu::MemoryClass.CPU_WRITE_GPU_LOCAL
            || texture_memory,
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
    if (state.mapping_unavailable) return gpu::BACKEND_ERROR~;
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

fn gpu::TextureHandle? vk_create_sparse_texture(
    VkDeviceState* state,
    gpu::TextureDesc* desc,
) {
    (void)state;
    (void)desc;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::SparseTextureRequirements? vk_get_sparse_texture_requirements(
    VkDeviceState* state,
    gpu::TextureHandle texture,
) {
    (void)state;
    (void)texture;
    return gpu::INVALID_ARGUMENT~;
}

fn gpu::CompletionPoint? vk_bind_sparse_texture_memory(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::SparseTextureBindDesc* desc,
) {
    (void)state;
    (void)queue;
    (void)desc;
    return gpu::INVALID_ARGUMENT~;
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

fn gpu::SwapchainHandle? vk_create_swapchain(
    VkDeviceState* state,
    gpu::Surface* surface,
    gpu::SwapchainDesc* desc,
) {
    swapchain_create_attempts++;
    if (state == null || surface == null || desc == null) return gpu::INVALID_ARGUMENT~;
    if (state.swapchain_live) return gpu::INVALID_RESOURCE_STATE~;
    if (configured_swapchain_create_failure) return gpu::BACKEND_ERROR~;

    gpu::internal::surface_retain_swapchain(surface)!;
    state.surface = *surface;
    state.swapchain = {
        .owner      = gpu::internal::resource_owner_tag(state.resource_owner, gpu::internal::ResourceOwnerKind.SWAPCHAIN),
        .index      = 0,
        .generation = 1,
    };
    state.swapchain_info = {
        .format       = desc.preferred_format,
        .width        = desc.width,
        .height       = desc.height,
        .image_count  = desc.image_count,
        .present_mode = desc.present_mode,
    };
    if (configured_swapchain_info_override) state.swapchain_info = configured_swapchain_info;
    state.present_wait_pending_count = 0;
    state.present_wait_fault = {};
    state.present_reset_fault = {};
    state.swapchain_live = true;
    return state.swapchain;
}

fn void? vk_wait_swapchain_presentations(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
    ulong timeout_ns,
) {
    context_wait_calls++;
    if (state == null || !state.swapchain_live || state.swapchain != handle) {
        return gpu::INVALID_HANDLE~;
    }
    if (state.present_wait_pending_count == 0) return;
    state.present_wait_calls++;
    state.present_wait_last_timeout_ns = timeout_ns;
    if (state.present_wait_fault) return state.present_wait_fault~;
    if (timeout_ns == 0) return gpu::WAIT_TIMEOUT~;
    state.present_reset_calls++;
    if (state.present_reset_fault) return state.present_reset_fault~;
    state.present_wait_pending_count = 0;
}

fn void? vk_destroy_swapchain(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
) {
    swapchain_destroy_attempts++;
    if (state == null || !state.swapchain_live) return gpu::INVALID_HANDLE~;
    if (state.swapchain != handle) return gpu::INVALID_HANDLE~;
    if (state.reject_swapchain_destroy) return gpu::RESOURCE_IN_USE~;

    gpu::internal::surface_release_swapchain(&state.surface)!;
    state.surface = {};
    state.swapchain = {};
    state.swapchain_info = {};
    state.present_wait_pending_count = 0;
    state.present_wait_fault = {};
    state.present_reset_fault = {};
    state.swapchain_live = false;
}

fn void? vk_resize_swapchain(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
    uint width,
    uint height,
) {
    if (state == null || !state.swapchain_live || state.swapchain != handle) {
        return gpu::INVALID_HANDLE~;
    }
    state.swapchain_info.width = width;
    state.swapchain_info.height = height;
}

fn gpu::SwapchainInfo? vk_get_swapchain_info(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
) {
    if (state == null || !state.swapchain_live || state.swapchain != handle) {
        return gpu::INVALID_HANDLE~;
    }
    if (configured_swapchain_info_fault) return configured_swapchain_info_fault~;
    return state.swapchain_info;
}

fn gpu::AcquiredImage? vk_acquire_next_image(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
    ulong timeout_ns,
) {
    (void)timeout_ns;
    if (state == null || !state.swapchain_live || state.swapchain != handle) {
        return gpu::INVALID_HANDLE~;
    }
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_present(
    VkDeviceState* state,
    gpu::AcquiredImage* image,
    gpu::CompletionPoint render_completion,
) {
    (void)state;
    (void)image;
    (void)render_completion;
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::PresentModeSupport? vk_get_present_mode_support(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
) {
    if (state == null || !state.swapchain_live || state.swapchain != handle) {
        return gpu::INVALID_HANDLE~;
    }
    return { .fifo = true };
}

fn gpu::Device? stub_create_anisotropy_disabled_device() {
    VkDeviceState* state = mem::new(VkDeviceState);
    defer catch mem::free(state);
    state.free_on_destroy = true;
    state.resource_owner = gpu::internal::allocate_device_resource_owner()!;
    return gpu::internal::register_device(
        state: state,
        caps: {
            .max_sampler_lod_bias = STUB_SAMPLER_LOD_BIAS_LIMIT,
        },
        resource_owner: state.resource_owner,
    );
}

fn void? validate_device_baseline_support(
    VkRuntimeState* runtime_state,
    uint adapter_index,
) {
    if (runtime_state == null) return gpu::INVALID_HANDLE~;
    if (adapter_index >= runtime_state.adapter_count) return gpu::INVALID_HANDLE~;
    if (!configured_device_baseline_support) return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::Device? create_vk_device(
    gpu::Adapter* adapter,
    gpu::DeviceDesc* device_desc,
    gpu::Runtime* retained_runtime,
) {
    gpu::DeviceDesc normalized =
        gpu::internal::normalize_device_desc(device_desc)!;
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
    state.destroy_busy = configured_device_destroy_failure;
    last_device_adapter = *adapter;
    gpu::internal::QueueSelection queues = gpu::internal::queue_selection_for_request(
        request:                    normalized.queues,
        compute_aliases_graphics:    true,
        transfer_aliases_graphics:   true,
        transfer_aliases_compute:    true,
    )!;
    gpu::internal::RuntimeData* runtime_data =
        gpu::internal::resolve_runtime(retained_runtime);
    if (runtime_data == null) return gpu::INVALID_HANDLE~;
    gpu::DeviceCaps caps = {
        .presentation_enabled = normalized.surface.is_valid(),
        .max_sampler_lod_bias = STUB_SAMPLER_LOD_BIAS_LIMIT,
        .max_sampler_anisotropy = STUB_SAMPLER_ANISOTROPY_LIMIT,
        .depth_stencil_format = gpu::Format.D24_UNORM_S8_UINT,
        .acceleration_structures = (normalized.enable_ray_queries
                || normalized.enable_ray_tracing_pipelines)
            ? (gpu::AccelerationStructureCaps){
                .enabled = true,
                .indirect_build = true,
                .heap_capacity =
                    runtime_data.state.acceleration_structure_heap_capacity,
                .max_geometry_count = uint::max,
                .max_primitive_count = uint::max,
                .max_instance_count = uint::max,
                .scratch_alignment = 256,
            }
            : {},
        .ray_queries = { .enabled = normalized.enable_ray_queries },
        .mesh_shaders = normalized.enable_mesh_shaders
            ? (gpu::MeshShaderCaps){
                .enabled                    = true,
                .task_shaders               = true,
                .max_task_work_group_count  = { 65535, 65535, 65535 },
                .max_mesh_work_group_count  = { 65535, 65535, 65535 },
                .max_task_work_group_total_count = 4194304,
                .max_mesh_work_group_total_count = 4194304,
                .max_mesh_output_vertices   = 256,
                .max_mesh_output_primitives = 256,
            }
            : {},
        .unified_layouts = normalized.unified_layouts,
        .ray_tracing_pipelines = normalized.enable_ray_tracing_pipelines
            ? (gpu::RayTracingPipelineCaps){
                .enabled = true,
                .max_recursion_depth = 4,
                .max_ray_dispatch_invocation_count = uint::max,
                .max_ray_dispatch_dimensions = {
                    uint::max,
                    uint::max,
                    uint::max,
                },
                .shader_group_handle_size = 32,
                .shader_group_handle_alignment = 32,
                .shader_group_base_alignment = 64,
                .max_shader_group_stride = 4096,
                .max_hit_attribute_size = 32,
            }
            : {},
    };
    gpu::Device device = gpu::internal::register_device(
        state:                state,
        caps:                 caps,
        resource_owner:       state.resource_owner,
        retained_runtime:     retained_runtime,
        adapter:              adapter,
        presentation_surface: normalized.surface.is_valid()
            ? &normalized.surface
            : null,
        queues: queues,
        owner:  &state.owner,
    )!;
    last_created_device = device;
    return device;
}

fn void? vk_cmd_texture_barrier(
    CommandRecord* commands,
    gpu::TextureBarrier* barrier,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_build_acceleration_structure(
    CommandRecord* record,
    gpu::AccelerationStructureBuildDesc* desc,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_update_acceleration_structure(
    CommandRecord* record,
    gpu::AccelerationStructureBuildDesc* desc,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_build_acceleration_structure_indirect(
    CommandRecord* record,
    gpu::AccelerationStructureBuildDesc* desc,
    gpu::GpuSpan ranges,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_update_acceleration_structure_indirect(
    CommandRecord* record,
    gpu::AccelerationStructureBuildDesc* desc,
    gpu::GpuSpan ranges,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_clone_acceleration_structure(
    CommandRecord* record,
    gpu::AccelerationStructureHandle source,
    gpu::AccelerationStructureHandle destination,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_begin_label(CommandRecord* commands, ZString label, float[4] color)  {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_end_label(CommandRecord* commands)  {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_copy_buffer(
    CommandRecord* commands,
    gpu::BufferCopyDesc* desc,
) {
    $if $feat(COMMAND_WRAPPER_BENCH):
        gpu::observe_copy_buffer();
        return;
    $else
    return gpu::UNSUPPORTED_FEATURE~;
    $endif
}

fn void? vk_cmd_fill_buffer(
    CommandRecord* commands,
    gpu::GpuSpan dst,
    uint value,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_copy_buffer_to_texture(
    CommandRecord* commands,
    gpu::BufferTextureCopyDesc* desc,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_copy_texture_to_buffer(
    CommandRecord* commands,
    gpu::TextureBufferCopyDesc* desc,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_bind_pipeline(
    CommandRecord* commands,
    gpu::PipelineHandle pipeline,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_dispatch(
    CommandRecord* commands,
    gpu::GpuAddress root,
    gpu::Vec3u groups,
) {
    $if $feat(COMMAND_WRAPPER_BENCH):
        gpu::observe_dispatch();
        return;
    $else
    return gpu::UNSUPPORTED_FEATURE~;
    $endif
}

fn void? vk_cmd_dispatch_indirect(
    CommandRecord* commands,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_dispatch_generated(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_dispatch_count,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_begin_render_pass(
    CommandRecord* commands,
    gpu::RenderPassDesc* desc,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_set_viewport(
    CommandRecord* commands,
    gpu::Viewport* viewport,
) {
    $if $feat(COMMAND_WRAPPER_BENCH):
        gpu::observe_viewport();
        return;
    $else
    return gpu::UNSUPPORTED_FEATURE~;
    $endif
}

fn void? vk_cmd_set_graphics_state(
    CommandRecord* commands,
    gpu::GraphicsState* state,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_set_scissor(
    CommandRecord* commands,
    gpu::ScissorRect* scissor,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_end_render_pass(CommandRecord* commands)  {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_draw(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    $if $feat(COMMAND_WRAPPER_BENCH):
        gpu::observe_draw();
        return;
    $else
    return gpu::UNSUPPORTED_FEATURE~;
    $endif
}

fn void? vk_cmd_draw_indexed(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_draw_indirect(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_draw_indexed_indirect(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_draw_indexed_indirect_count(
    CommandRecord* commands,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_draw_generated(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_draw_indexed_generated(
    CommandRecord* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_draw_mesh_tasks(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::Vec3u groups,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_draw_mesh_tasks_indirect(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_draw_mesh_tasks_indirect_count(
    CommandRecord* commands,
    gpu::GpuAddress mesh_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_reset_timestamps(
    CommandRecord* record,
    gpu::TimestampPoolHandle pool,
    uint first,
    uint count,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_write_timestamp(
    CommandRecord* record,
    gpu::TimestampPoolHandle pool,
    uint index,
    gpu::StageMask stage,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_resolve_timestamps(
    CommandRecord* record,
    gpu::TimestampPoolHandle pool,
    uint first,
    uint count,
    gpu::GpuSpan destination,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_set_ray_tracing_pipeline_stack_size(
    CommandRecord* record,
    ulong pipeline_stack_size,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_trace_rays(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::Vec3u dimensions,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_trace_rays_indirect(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::RayTracingShaderBindingTable* table,
    gpu::GpuSpan args,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_trace_rays_indirect2(
    CommandRecord* record,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) {
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? vk_cmd_barrier(CommandRecord* commands, gpu::Barrier* barrier) {
    $if $feat(COMMAND_WRAPPER_BENCH):
        gpu::observe_barrier();
        return;
    $else
    VkDeviceState* state = commands.device_state;
    gpu::internal::DeviceSlotControl control =
        (gpu::internal::DeviceSlotControl)gpu::internal::device_slots[commands.device.index].control.load(
            AtomicOrdering.ACQUIRE,
        );
    state.pins_during_command = control.pins;
    state.command_entered.store(true, AtomicOrdering.RELEASE);
    while (state.block_command && !state.command_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    if (state.reject_command) return gpu::INVALID_HANDLE~;
    $endif
}
