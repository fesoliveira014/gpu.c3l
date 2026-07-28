module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import std::atomic;
import std::core::mem;
import std::thread;

alias CmdCopyBufferFn @private = fn void? (
    CommandRecord*,
    gpu::BufferCopyDesc*,
);
alias CmdFillBufferFn @private = fn void? (
    CommandRecord*,
    gpu::GpuSpan,
    uint,
);
alias CmdCopyBufferToTextureFn @private = fn void? (
    CommandRecord*,
    gpu::BufferTextureCopyDesc*,
);
alias CmdCopyTextureToBufferFn @private = fn void? (
    CommandRecord*,
    gpu::TextureBufferCopyDesc*,
);
alias CmdBarrierFn @private = fn void? (CommandRecord*, gpu::Barrier*);
alias CmdTextureBarrierFn @private = fn void? (
    CommandRecord*,
    gpu::TextureBarrier*,
);
alias CmdBindPipelineFn @private = fn void? (
    CommandRecord*,
    gpu::PipelineHandle,
);
alias CmdSetGraphicsStateFn @private = fn void? (
    CommandRecord*,
    gpu::GraphicsState*,
);
alias CmdDispatchFn @private = fn void? (
    CommandRecord*,
    gpu::GpuAddress,
    gpu::Vec3u,
);
alias CmdBeginRenderPassFn @private = fn void? (
    CommandRecord*,
    gpu::RenderPassDesc*,
);
alias CmdSetViewportFn @private = fn void? (
    CommandRecord*,
    gpu::Viewport*,
);
alias CmdSetScissorFn @private = fn void? (
    CommandRecord*,
    gpu::ScissorRect*,
);
alias CmdEndRenderPassFn @private = fn void? (CommandRecord*);
alias CmdDrawFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
);
alias CmdDrawIndexedFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan index_span,
    uint index_count,
    uint instance_count,
    gpu::IndexType index_type,
);
alias CmdDrawIndirectFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
);
alias CmdDrawIndexedIndirectFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    uint draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
);
alias CmdDrawIndexedIndirectCountFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuAddress vertex_root,
    gpu::GpuAddress fragment_root,
    gpu::GpuSpan args,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
);
alias CmdDispatchIndirectFn @private = fn void? (
    CommandRecord*,
    gpu::GpuAddress,
    gpu::GpuSpan,
);
alias CmdDrawGeneratedFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
);
alias CmdDrawIndexedGeneratedFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_draw_count,
    gpu::GpuSpan index_span,
    gpu::IndexType index_type,
);
alias CmdDispatchGeneratedFn @private = fn void? (
    CommandRecord* record,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_dispatch_count,
);
alias CmdBeginLabelFn @private = fn void? (
    CommandRecord*,
    ZString,
    float[4],
);
alias CmdEndLabelFn @private = fn void? (CommandRecord*);
alias CmdResetTimestampsFn @private = fn void? (
    CommandRecord*,
    gpu::TimestampPoolHandle,
    uint,
    uint,
);
alias CmdWriteTimestampFn @private = fn void? (
    CommandRecord*,
    gpu::TimestampPoolHandle,
    uint,
    gpu::StageMask,
);
alias CmdResolveTimestampsFn @private = fn void? (
    CommandRecord*,
    gpu::TimestampPoolHandle,
    uint,
    uint,
    gpu::GpuSpan,
);

const uint  STUB_SAMPLER_CAPACITY       @private = 16;
const float STUB_SAMPLER_LOD_BIAS_LIMIT @private = 16.0f;
const float STUB_SAMPLER_ANISOTROPY_LIMIT @private = 16.0f;
const uint STUB_COMMAND_CAPACITY @private = 64;

struct CommandRecord @public {
    CommandOps*                      ops;
    VkDeviceState*                   device_state;
    gpu::Device                      device;
    gpu::internal::CommandListHandle identity;
    gpu::internal::CommandState      state;
    gpu::Queue                       queue;
}

struct CommandOps @public {
    CmdCopyBufferFn               copy_buffer;
    CmdFillBufferFn               fill_buffer;
    CmdCopyBufferToTextureFn      copy_buffer_to_texture;
    CmdCopyTextureToBufferFn      copy_texture_to_buffer;
    CmdBarrierFn                  barrier;
    CmdTextureBarrierFn           texture_barrier;
    CmdBindPipelineFn             bind_pipeline;
    CmdSetGraphicsStateFn         set_graphics_state;
    CmdDispatchFn                 dispatch;
    CmdBeginRenderPassFn          begin_render_pass;
    CmdSetViewportFn              set_viewport;
    CmdSetScissorFn               set_scissor;
    CmdEndRenderPassFn            end_render_pass;
    CmdDrawFn                     draw;
    CmdDrawIndexedFn              draw_indexed;
    CmdDrawIndirectFn             draw_indirect;
    CmdDrawIndexedIndirectFn      draw_indexed_indirect;
    CmdDrawIndexedIndirectCountFn draw_indexed_indirect_count;
    CmdDispatchIndirectFn         dispatch_indirect;
    CmdDrawGeneratedFn            draw_generated;
    CmdDrawIndexedGeneratedFn     draw_indexed_generated;
    CmdDispatchGeneratedFn        dispatch_generated;
    CmdBeginLabelFn               begin_label;
    CmdEndLabelFn                 end_label;
    CmdResetTimestampsFn          reset_timestamps;
    CmdWriteTimestampFn           write_timestamp;
    CmdResolveTimestampsFn        resolve_timestamps;
}

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
    gpu::Queue           allocator_queue;
    gpu::CommandAllocatorDesc allocator_desc;
    CommandOps*          command_ops;
    CommandRecord[STUB_COMMAND_CAPACITY] command_records;
    gpu::GpuAllocation   allocation;
    gpu::AllocationDesc  allocation_desc;
    gpu::TextureHandle   texture;
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
    fault                completion_wait_fault;
    gpu::DebugResourceKind root_diagnostic_resource_kind;
    ZString              root_diagnostic_rejected_field;
    ZString              root_diagnostic_invariant;
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
    bool                 allocation_out_of_memory;
    bool                 texture_out_of_memory;
    bool                 block_memory_stats;
    bool                 free_on_destroy;
    bool                 reject_allocator_create;
    bool                 reject_allocator_destroy;
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

uint configured_adapter_count @private = 1;
bool configured_device_baseline_support @private = true;
bool configured_sparse_texture_support @private;
bool configured_sparse_queue_support @private;
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
    if (state.command_ops == null) return gpu::BACKEND_ERROR~;
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
        .ops          = state.command_ops,
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
    if (!handle.is_valid()) return gpu::INVALID_HANDLE~;
    if (state.reject_allocator_destroy) return gpu::RESOURCE_IN_USE~;
}

fn void? vk_reserve_generated_scratch(
    VkDeviceState* state,
    gpu::CommandAllocator* allocator,
    gpu::GeneratedScratchDesc* desc,
) {
    (void)state;
    (void)allocator;
    (void)desc;
}

fn void? vk_release_generated_scratch(
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
        .device_class     = gpu::AdapterClass.SOFTWARE,
        .memory           = {
            .device_local_bytes = 1024,
            .host_visible_bytes = 1024,
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
    return configured_device_baseline_support;
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
) {
    (void)backend_surface;
    if (state == null || request == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_HANDLE~;
    }
    if (require_sparse_binding && !configured_sparse_queue_support) {
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
    configured_device_baseline_support = true;
    configured_sparse_texture_support = false;
    configured_sparse_queue_support = false;
}

fn void stub_set_device_baseline_support(bool supported) {
    configured_device_baseline_support = supported;
}

fn void stub_set_sparse_texture_support(bool supported) {
    configured_sparse_texture_support = supported;
}

fn void stub_set_sparse_queue_support(bool supported) {
    configured_sparse_queue_support = supported;
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
    state.swapchain_live = true;
    return state.swapchain;
}

fn void? vk_destroy_swapchain(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
) {
    if (state == null || !state.swapchain_live) return gpu::INVALID_HANDLE~;
    if (state.swapchain != handle) return gpu::INVALID_HANDLE~;

    gpu::internal::surface_release_swapchain(&state.surface)!;
    state.surface = {};
    state.swapchain = {};
    state.swapchain_info = {};
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
    last_device_adapter = *adapter;
    gpu::internal::QueueSelection queues = gpu::internal::queue_selection_for_request(
        request:                    normalized.queues,
        compute_aliases_graphics:    true,
        transfer_aliases_graphics:   true,
        transfer_aliases_compute:    true,
    )!;
    gpu::DeviceCaps caps = {
        .presentation_enabled = normalized.surface.is_valid(),
        .max_sampler_lod_bias = STUB_SAMPLER_LOD_BIAS_LIMIT,
        .max_sampler_anisotropy = STUB_SAMPLER_ANISOTROPY_LIMIT,
    };
    return gpu::internal::register_device(
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
    );
}
