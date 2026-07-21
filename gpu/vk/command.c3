module gpu::vk @private;

import gpu @public;
import vk;
import vma;
import std::thread;
import std::core::mem;
import std::core::env;
import std::atomic;
import std::atomic::types;

alias CreateContextPoolFn @private = fn vk::CommandPool? (void*, vk::Device, uint);
alias DestroyContextPoolFn @private = fn void (void*, vk::Device, vk::CommandPool);

const uint MAX_RECORDING_CONTEXTS = 256;
const uint MAX_THREAD_DEVICE_CONTEXTS = 64;
const usz RETIRED_COMMAND_INITIAL_CAPACITY = 16;
const usz GENERATED_PREPROCESS_INITIAL_CAPACITY = 4;

struct RecordingContextHandle {
    uint index;
    uint generation;
}

struct ThreadRecordingContext {
    gpu::Device            device;
    RecordingContextHandle context;
}

struct GeneratedPreprocessBuffer {
    vk::Buffer        buffer;
    vma::Allocation   allocation;
    vk::DeviceAddress address;
    vk::DeviceSize    size;
    uint              memory_type;
    RecordingContextHandle reservation_context;
    uint              reservation_index;
    bool              reservation_owned;
    bool              reservation_in_use;
}

struct CommandRecordingStats {
    ulong preprocess_allocations;
    ulong preprocess_reuses;
    ulong recording_allocations;
}

struct RetiredCommandBuffer {
    vk::CommandPool             pool;
    vk::CommandBuffer           buffer;
    CommandResourceReference[]  references;
    GeneratedPreprocessBuffer[] generated_preprocess;
}

struct ContextPoolOps @private {
    void*                user_data;
    CreateContextPoolFn  create;
    DestroyContextPoolFn destroy;
}

struct RecordingContextState {
    vk::CommandPool[]             graphics_pools;
    // Aliases graphics_pools when the compute queue shares the graphics
    // family; owned storage only when distinct (compute_pools_distinct).
    vk::CommandPool[]             compute_pools;
    vk::CommandPool[]             transfer_pools;
    RetiredCommandBuffer[]        retired_commands;
    uint                          retired_command_count;
    gpu::GeneratedScratchDesc     generated_scratch;
    GeneratedPreprocessBuffer[] reserved_preprocess;
    bool                          compute_pools_distinct;
}

struct RecordingContextCell {
    RecordingContextState value;
    uint                  generation;
    bool                  used;
}

struct RecordingContextTable {
    RecordingContextCell[] slots;
    uint                   next_new;
}

tlocal ThreadRecordingContext[MAX_THREAD_DEVICE_CONTEXTS]
    thread_recording_contexts;

fn bool RecordingContextHandle.is_valid(self) => self.generation != 0;
fn bool RecordingContextHandle.equals(
    self,
    RecordingContextHandle other,
) @operator(==) {
    return self.index == other.index && self.generation == other.generation;
}

fn RecordingContextHandle? RecordingContextTable.alloc(&self, RecordingContextState value) {
    if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
    uint index = self.next_new++;
    RecordingContextCell* cell = &self.slots[index];
    cell.value = value;
    cell.generation = next_handle_generation(0)!;
    cell.used = true;
    return {
        .index      = index,
        .generation = cell.generation,
    };
}

fn RecordingContextState*? RecordingContextTable.get(
    &self,
    RecordingContextHandle handle,
) {
    if (handle.index >= self.next_new) return gpu::INVALID_HANDLE~;
    RecordingContextCell* cell = &self.slots[handle.index];
    if (!cell.used || cell.generation != handle.generation) {
        return gpu::INVALID_HANDLE~;
    }
    return &cell.value;
}

fn void note_recording_host_allocation(VkDeviceState* state) {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)state.recording_host_allocations.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_command_buffer_reset(VkDeviceState* state) {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)state.command_buffer_resets.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_command_buffer_allocation(VkDeviceState* state) {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)state.recording_command_buffer_allocations.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_command_buffer_free(VkDeviceState* state) {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)state.recording_command_buffer_frees.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_recording_vma_allocation(VkDeviceState* state) {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)state.recording_vma_allocations.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_generated_scratch_miss(VkDeviceState* state) {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)state.generated_scratch_misses.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn vk::CommandPoolCreateFlags context_pool_create_flags() {
    return (vk::CommandPoolCreateFlags)(
        (uint)vk::CommandPoolCreateFlagBits.TRANSIENT_BIT
        | (uint)vk::CommandPoolCreateFlagBits.RESET_COMMAND_BUFFER_BIT
    );
}

fn vk::CommandPool? create_context_pool_real(void* user_data, vk::Device device, uint family) @private {
    VkDeviceState* state = (VkDeviceState*)user_data;
    vk::CommandPoolCreateInfo info = vk::command_pool_create_info()
        .set_queue_family_index(family)
        .set_flags(context_pool_create_flags());
    vk::CommandPool pool;
    check_device_result(state, vk::create_command_pool(
        device:        device,
        p_create_info: &info,
        p_allocator:   null,
        p_command_pool: &pool,
    ))!;
    return pool;
}

fn void destroy_context_pool_real(void* user_data, vk::Device device, vk::CommandPool pool) @private {
    vk::destroy_command_pool(device, pool, null);
}

fn ContextPoolOps context_pool_ops_real(VkDeviceState* state) @private {
    return {
        .user_data = state,
        .create    = &create_context_pool_real,
        .destroy   = &destroy_context_pool_real,
    };
}

fn void destroy_context_pools_with_ops(
    VkDeviceState* state,
    RecordingContextState* context,
    ContextPoolOps* ops
) @private {
    if (context.retired_commands.ptr != null) {
        foreach (&unit : context.retired_commands[:context.retired_command_count]) {
            if (unit.references.ptr != null) {
                alloc::free(state.host_allocator, unit.references.ptr);
            }
            free_generated_preprocess_buffers(
                state,
                unit.generated_preprocess,
            );
        }
        alloc::free(state.host_allocator, context.retired_commands.ptr);
        context.retired_commands = {};
    }
    context.retired_command_count = 0;
    free_reserved_generated_preprocess_buffers(
        state,
        context.reserved_preprocess,
    );
    context.reserved_preprocess = {};
    context.generated_scratch = {};
    if (context.graphics_pools.ptr != null) {
        foreach (pool : context.graphics_pools) {
            if (pool != null) ops.destroy(ops.user_data, state.device, pool);
        }
        alloc::free(state.host_allocator, context.graphics_pools.ptr);
        context.graphics_pools = {};
    }
    if (context.compute_pools_distinct && context.compute_pools.ptr != null) {
        foreach (pool : context.compute_pools) {
            if (pool != null) ops.destroy(ops.user_data, state.device, pool);
        }
        alloc::free(state.host_allocator, context.compute_pools.ptr);
    }
    context.compute_pools = {};
    if (context.transfer_pools.ptr != null) {
        foreach (pool : context.transfer_pools) {
            if (pool != null) ops.destroy(ops.user_data, state.device, pool);
        }
        alloc::free(state.host_allocator, context.transfer_pools.ptr);
        context.transfer_pools = {};
    }
    context.compute_pools_distinct = false;
}

<* One command pool per admitted queue family, owned by a recording context. *>
fn void? create_context_pools_with_ops(
    VkDeviceState* state,
    RecordingContextState* context,
    ContextPoolOps* ops
) @private {
    *context = {};
    defer catch destroy_context_pools_with_ops(state, context, ops);

    if (state.selected_queue_roles.graphics) {
        context.graphics_pools = alloc::new_array(
            state.host_allocator,
            vk::CommandPool,
            1,
        );
        note_recording_host_allocation(state);
    }
    context.compute_pools_distinct = state.selected_queue_roles.compute
        && (!state.selected_queue_roles.graphics
            || state.compute_family != state.graphics_family);
    if (state.selected_queue_roles.compute) {
        context.compute_pools = context.compute_pools_distinct
            ? alloc::new_array(
                state.host_allocator,
                vk::CommandPool,
                1,
            )
            : context.graphics_pools;
        if (context.compute_pools_distinct) {
            note_recording_host_allocation(state);
        }
    }
    if (state.selected_queue_roles.transfer) {
        context.transfer_pools = alloc::new_array(
            state.host_allocator,
            vk::CommandPool,
            1,
        );
        note_recording_host_allocation(state);
    }
    if (state.selected_queue_roles.graphics) {
        context.graphics_pools[0] = ops.create(ops.user_data, state.device, state.graphics_family)!;
    }
    if (context.compute_pools_distinct) {
        context.compute_pools[0] = ops.create(ops.user_data, state.device, state.compute_family)!;
    }
    if (state.selected_queue_roles.transfer) {
        context.transfer_pools[0] = ops.create(ops.user_data, state.device, state.transfer_family)!;
    }
}

fn void? create_context_pools(VkDeviceState* state, RecordingContextState* context) @private {
    ContextPoolOps ops = context_pool_ops_real(state);
    return create_context_pools_with_ops(state, context, &ops);
}

fn void destroy_context_pools(VkDeviceState* state, RecordingContextState* context) @private {
    ContextPoolOps ops = context_pool_ops_real(state);
    destroy_context_pools_with_ops(state, context, &ops);
}

fn RecordingContextHandle? thread_recording_context(
    VkDeviceState* state,
    gpu::Device device,
) @private {
    if (device.index >= MAX_THREAD_DEVICE_CONTEXTS) {
        return gpu::INVALID_HANDLE~;
    }
    ThreadRecordingContext* cached = &thread_recording_contexts[device.index];
    if (cached.device == device && cached.context.is_valid()) {
        return cached.context;
    }

    RecordingContextState context;
    create_context_pools(state, &context)!;
    RecordingContextHandle? handle;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        handle = state.recording_contexts.alloc(context);
    }
    if (catch err = handle) {
        destroy_context_pools(state, &context);
        return err~;
    }
    *cached = {
        .device  = device,
        .context = handle,
    };
    return handle;
}

fn vk::CommandPool pool_for(
    RecordingContextState* context,
    gpu::Queue queue,
) {
    if (queue.roles.graphics) return context.graphics_pools[0];
    if (queue.roles.compute) return context.compute_pools[0];
    return context.transfer_pools[0];
}

fn bool recording_context_quiescent_locked(
    VkDeviceState* state,
    RecordingContextHandle context,
) {
    foreach (&cell : state.commands.slots[:state.commands.next_new]) {
        if (cell.used && cell.value.context == context) return false;
    }
    foreach (&batch : state.submitted_commands[:state.submitted_command_count]) {
        if (batch.context == context) return false;
    }
    return true;
}

fn uint preferred_generated_preprocess_memory_type_bits(
    VkDeviceState* state,
) {
    vk::PhysicalDeviceMemoryProperties properties;
    vk::get_physical_device_memory_properties(
        state.physical_device,
        &properties,
    );
    for (uint i = 0; i < properties.memory_type_count; i++) {
        if (((uint)properties.memory_types[i].property_flags
            & (uint)vk::MemoryPropertyFlagBits.DEVICE_LOCAL_BIT) != 0) {
            return 1 << i;
        }
    }
    return uint::max;
}

fn void? vk_reserve_generated_scratch(
    gpu::Queue queue,
    gpu::GeneratedScratchDesc desc,
) @private {
    gpu::Device device = queue.device;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(
        &device,
    );
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    RecordingContextHandle context_handle = thread_recording_context(
        state,
        device,
    )!;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        state.command_mutex.lock()!!;
        defer state.command_mutex.unlock();
        if (!recording_context_quiescent_locked(state, context_handle)) {
            return gpu::RESOURCE_IN_USE~;
        }
    }

    GeneratedPreprocessBuffer[] reserved;
    if (desc.preprocess_buffer_count > 0) {
        reserved = alloc::new_array(
            state.host_allocator,
            GeneratedPreprocessBuffer,
            desc.preprocess_buffer_count,
        );
        note_recording_host_allocation(state);
    }
    defer catch free_reserved_generated_preprocess_buffers(state, reserved);
    for (uint i = 0; i < desc.preprocess_buffer_count; i++) {
        reserved[i] = allocate_generated_preprocess_buffer(
            state:            state,
            size:             (vk::DeviceSize)desc.max_preprocess_bytes,
            alignment:        256,
            memory_type_bits: preferred_generated_preprocess_memory_type_bits(
                state,
            ),
        )!;
        reserved[i].reservation_context = context_handle;
        reserved[i].reservation_index = i;
        reserved[i].reservation_owned = true;
    }

    GeneratedPreprocessBuffer[] previous;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        state.command_mutex.lock()!!;
        defer state.command_mutex.unlock();
        if (!recording_context_quiescent_locked(state, context_handle)) {
            return gpu::RESOURCE_IN_USE~;
        }
        RecordingContextState* context = state.recording_contexts.get(
            context_handle,
        )!;
        foreach (&unit : context.retired_commands[
            :context.retired_command_count
        ]) {
            if (unit.generated_preprocess.len
                >= desc.preprocess_buffer_count) continue;
            GeneratedPreprocessBuffer[] grown = alloc::new_array(
                state.host_allocator,
                GeneratedPreprocessBuffer,
                desc.preprocess_buffer_count,
            );
            note_recording_host_allocation(state);
            if (unit.generated_preprocess.ptr != null) {
                alloc::free(
                    state.host_allocator,
                    unit.generated_preprocess.ptr,
                );
            }
            unit.generated_preprocess = grown;
        }
        previous = context.reserved_preprocess;
        context.generated_scratch = desc;
        context.reserved_preprocess = reserved;
        reserved = {};
    }
    free_reserved_generated_preprocess_buffers(state, previous);
}

fn void enqueue_retired_command_buffer_locked(
    VkDeviceState* state,
    RecordingContextHandle context_handle,
    vk::CommandPool pool,
    vk::CommandBuffer buffer,
    CommandResourceReference[] references = {},
    GeneratedPreprocessBuffer[] generated_preprocess = {},
) {
    if (buffer == null) return;
    RecordingContextState* context = state.recording_contexts.get(
        context_handle,
    )!!;
    if (context.retired_command_count >= context.retired_commands.len) {
        usz capacity = context.retired_commands.len == 0
            ? RETIRED_COMMAND_INITIAL_CAPACITY
            : context.retired_commands.len * 2;
        RetiredCommandBuffer[] grown = alloc::new_array(
            state.host_allocator,
            RetiredCommandBuffer,
            (sz)capacity,
        );
        note_recording_host_allocation(state);
        for (uint i = 0; i < context.retired_command_count; i++) {
            grown[i] = context.retired_commands[i];
        }
        if (context.retired_commands.ptr != null) {
            alloc::free(state.host_allocator, context.retired_commands.ptr);
        }
        context.retired_commands = grown;
    }
    context.retired_commands[context.retired_command_count++] = {
        .pool                 = pool,
        .buffer               = buffer,
        .references           = references,
        .generated_preprocess = generated_preprocess,
    };
}

fn bool take_available_command_buffer(
    VkDeviceState* state,
    RecordingContextHandle context_handle,
    vk::CommandPool pool,
    vk::CommandBuffer* out_buffer,
    CommandResourceReference[]* out_references,
    GeneratedPreprocessBuffer[]* out_generated_preprocess,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    RecordingContextState* context = state.recording_contexts.get(
        context_handle,
    )!!;
    for (uint i = 0; i < context.retired_command_count; i++) {
        RetiredCommandBuffer* available = &context.retired_commands[i];
        if (available.pool != pool) continue;
        *out_buffer = available.buffer;
        *out_references = available.references;
        *out_generated_preprocess = available.generated_preprocess;
        context.retired_command_count--;
        *available = context.retired_commands[
            context.retired_command_count
        ];
        context.retired_commands[context.retired_command_count] = {};
        return true;
    }
    return false;
}

fn void? vk_begin_commands_with_context(
    VkDeviceState* state,
    gpu::Queue queue,
    RecordingContextHandle context_handle,
    gpu::CommandList* out_commands,
    gpu::CommandBackendContext* out_backend,
) @private {
    gpu::Device device = queue.device;
    RecordingContextState* context = state.recording_contexts.get(context_handle)!;
    vk::CommandPool pool = pool_for(context, queue);
    vk::CommandBuffer cmd;
    CommandResourceReference[] references;
    GeneratedPreprocessBuffer[] generated_preprocess;
    bool newly_allocated;
    if (take_available_command_buffer(
        state,
        context_handle,
        pool,
        &cmd,
        &references,
        &generated_preprocess,
    )) {
        state.command_mutex.lock()!!;
        defer state.command_mutex.unlock();
        check_device_result(state, vk::reset_command_buffer(cmd, {}))!;
        note_command_buffer_reset(state);
    } else {
        vk::CommandBufferAllocateInfo alloc = vk::command_buffer_allocate_info()
            .set_command_pool(pool)
            .set_level(vk::CommandBufferLevel.PRIMARY)
            .set_command_buffer_count(1);
        state.command_mutex.lock()!!;
        defer state.command_mutex.unlock();
        check_device_result(
            state,
            vk::allocate_command_buffers(state.device, &alloc, &cmd),
        )!;
        state.command_buffers_allocated++;
        note_command_buffer_allocation(state);
        newly_allocated = true;
    }
    defer catch {
        if (newly_allocated) {
            state.command_mutex.lock()!!;
            defer state.command_mutex.unlock();
            vk::free_command_buffers(state.device, pool, 1, &cmd);
            state.command_buffers_freed++;
            note_command_buffer_free(state);
            if (references.ptr != null) {
                alloc::free(state.host_allocator, references.ptr);
            }
            free_generated_preprocess_buffers(
                state,
                generated_preprocess,
            );
        } else {
            state.resource_mutex.lock()!!;
            defer state.resource_mutex.unlock();
            enqueue_retired_command_buffer_locked(
                state:          state,
                context_handle: context_handle,
                pool:           pool,
                buffer:         cmd,
                references:     references,
                generated_preprocess: generated_preprocess,
            );
        }
    }

    vk::CommandBufferBeginInfo begin = vk::command_buffer_begin_info()
        .set_flags(vk::CommandBufferUsageFlagBits.ONE_TIME_SUBMIT_BIT);
    check_device_result(state, vk::begin_command_buffer(cmd, &begin))!;

    if (state.generated_work.enabled && generated_preprocess.ptr == null) {
        usz capacity = GENERATED_PREPROCESS_INITIAL_CAPACITY;
        usz reserved_capacity = (usz)context.generated_scratch
            .preprocess_buffer_count;
        if (reserved_capacity > capacity) capacity = reserved_capacity;
        generated_preprocess = alloc::new_array(
            state.host_allocator,
            GeneratedPreprocessBuffer,
            (sz)capacity,
        );
        note_recording_host_allocation(state);
    }

    CommandRecord record = {
        .command_buffer       = cmd,
        .command_pool         = pool,
        .context              = context_handle,
        .queue                = queue,
        .state                = CommandState.RECORDING,
        .references           = references,
        .generated_preprocess = generated_preprocess,
    };
    {
        state.command_mutex.lock()!!;
        defer state.command_mutex.unlock();
        if (try handle = state.commands.alloc(record)) {
            *out_commands = {
                .device = device,
                .handle = handle,
            };
            *out_backend = {
                .state   = state,
                .command = state.commands.get(handle)!!,
            };
            return;
        }
    }
    return gpu::SLOT_TABLE_FULL~;
}

fn void? vk_begin_commands(
    gpu::Queue queue,
    gpu::CommandList* out_commands,
    gpu::CommandBackendContext* out_backend,
) @private {
    gpu::Device device = queue.device;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&device);
    RecordingContextHandle context = thread_recording_context(state, device)!;
    return vk_begin_commands_with_context(
        state:          state,
        queue:          queue,
        context_handle: context,
        out_commands:   out_commands,
        out_backend:    out_backend,
    );
}

fn void? vk_end_commands(gpu::CommandEncoderState* commands) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    check_device_result(state, vk::end_command_buffer(record.command_buffer))!;
    record.state = CommandState.EXECUTABLE;
}

fn void? vk_discard_commands(
    gpu::Device* device,
    gpu::CommandListHandle handle,
    bool executable,
    gpu::CommandEncoderState* encoder,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    state.command_mutex.lock()!!;
    defer state.command_mutex.unlock();
    CommandRecord* record = state.commands.get(handle)!;
    bool state_matches = executable
        ? record.state == CommandState.EXECUTABLE
        : record.state == CommandState.RECORDING
            || record.state == CommandState.RECORDING_RENDER_PASS;
    if (!state_matches) return gpu::COMMAND_RECORDING_ERROR~;
    RecordingContextHandle context = record.context;
    vk::CommandBuffer command_buffer = record.command_buffer;
    vk::CommandPool command_pool = record.command_pool;
    if (encoder != null) gpu::invalidate_command_encoder(encoder);
    release_validation_references(
        state,
        record.references,
        record.reference_count,
    );
    recycle_generated_preprocess_buffers_locked(
        state,
        record.generated_preprocess,
        record.generated_preprocess_count,
    );
    CommandResourceReference[] references = record.references;
    GeneratedPreprocessBuffer[] generated_preprocess =
        record.generated_preprocess;
    record.references = {};
    record.generated_preprocess = {};
    record.reference_count = 0;
    record.generated_preprocess_count = 0;
    (void)state.commands.free(handle);
    enqueue_retired_command_buffer_locked(
        state:          state,
        context_handle: context,
        pool:           command_pool,
        buffer:         command_buffer,
        references:     references,
        generated_preprocess: generated_preprocess,
    );
}

fn void? vk_cmd_copy_buffer(gpu::CommandEncoderState* commands, gpu::BufferCopyDesc* desc) @private {
    CommandRecord* record = encoder_command(commands);
    VkDeviceState* state = encoder_device_state(commands);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    if (record.state != CommandState.RECORDING) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_copy_buffer",
            public_fault:        gpu::COMMAND_RECORDING_ERROR,
            resource_kind:       gpu::DebugResourceKind.COMMAND_LIST,
            has_public_identity: true,
            index:               commands.handle.index,
            generation:          commands.handle.generation,
            debug_name:          null,
            rejected_field:      "CommandList.state",
            invariant:           "command must be recording outside a render pass",
            backend_text:        null,
        )~;
    }
    ResolvedSpan src = resolve_span_slot(
        state:          state,
        span:           &desc.src,
        required:       0,
        operation:      "cmd_copy_buffer",
        rejected_field: "BufferCopyDesc.src",
    )!;
    ResolvedSpan dst = resolve_span_slot(
        state:          state,
        span:           &desc.dst,
        required:       0,
        operation:      "cmd_copy_buffer",
        rejected_field: "BufferCopyDesc.dst",
    )!;
    validate_span_transfer_usage(
        state:     state,
        resolved:  &src,
        span:      &desc.src,
        source:    true,
        operation: "cmd_copy_buffer",
    )!;
    validate_span_transfer_usage(
        state:     state,
        resolved:  &dst,
        span:      &desc.dst,
        source:    false,
        operation: "cmd_copy_buffer",
    )!;
    if (src.size != dst.size) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_copy_buffer",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       span_resource_kind(state, &desc.dst),
            has_public_identity: true,
            index:               desc.dst.index,
            generation:          desc.dst.generation,
            debug_name:          slot_name(&dst.slot.debug_name[0]),
            rejected_field:      "BufferCopyDesc.dst",
            invariant:           "source and destination spans must have equal sizes",
            backend_text:        null,
        )~;
    }
    if (src.slot.buffer == dst.slot.buffer
        && src.offset < dst.offset + dst.size
        && dst.offset < src.offset + src.size) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_copy_buffer",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       span_resource_kind(state, &desc.src),
            has_public_identity: true,
            index:               desc.src.index,
            generation:          desc.src.generation,
            debug_name:          slot_name(&src.slot.debug_name[0]),
            rejected_field:      "BufferCopyDesc",
            invariant:           "source and destination spans must not overlap",
            backend_text:        null,
        )~;
    }
    validate_span_recording_access(
        state:     state,
        record:    record,
        resolved:  &src,
        span:      &desc.src,
        operation: "cmd_copy_buffer",
    )!;
    validate_span_recording_access(
        state:     state,
        record:    record,
        resolved:  &dst,
        span:      &desc.dst,
        operation: "cmd_copy_buffer",
    )!;

    vk::BufferCopy region = {
        .src_offset = (vk::DeviceSize)src.offset,
        .dst_offset = (vk::DeviceSize)dst.offset,
        .size       = (vk::DeviceSize)src.size,
    };
    vk::cmd_copy_buffer(
        command_buffer: record.command_buffer,
        src_buffer:     src.slot.buffer,
        dst_buffer:     dst.slot.buffer,
        region_count:   1,
        p_regions:      &region,
    );
    gpu::note_command_native_call();
}

fn void? vk_cmd_fill_buffer(
    gpu::CommandEncoderState* commands,
    gpu::GpuSpan dst,
    uint value,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    ResolvedSpan resolved = resolve_span_slot(
        state:          state,
        span:           &dst,
        required:       0,
        operation:      "cmd_fill_buffer",
        rejected_field: "dst",
    )!;
    validate_span_transfer_usage(
        state:     state,
        resolved:  &resolved,
        span:      &dst,
        source:    false,
        operation: "cmd_fill_buffer",
    )!;
    if (resolved.offset % 4 != 0 || resolved.size % 4 != 0) {
        bool offset_misaligned = resolved.offset % 4 != 0;
        ZString rejected_field = offset_misaligned
            ? "GpuSpan.offset"
            : "GpuSpan.size";
        ZString invariant = offset_misaligned
            ? "span offset must be 4-byte aligned"
            : "span size must be 4-byte aligned";
        return report_contract_failure(
            state:               state,
            operation:           "cmd_fill_buffer",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       span_resource_kind(state, &dst),
            has_public_identity: true,
            index:               dst.index,
            generation:          dst.generation,
            debug_name:          slot_name(&resolved.slot.debug_name[0]),
            rejected_field:      rejected_field,
            invariant:           invariant,
            backend_text:        null,
        )~;
    }
    validate_span_recording_access(
        state:     state,
        record:    record,
        resolved:  &resolved,
        span:      &dst,
        operation: "cmd_fill_buffer",
    )!;
    vk::cmd_fill_buffer(
        command_buffer: record.command_buffer,
        dst_buffer:     resolved.slot.buffer,
        dst_offset:     (vk::DeviceSize)resolved.offset,
        size:           (vk::DeviceSize)resolved.size,
        data:           value,
    );
    gpu::note_command_native_call();
}

<* Validate and translate a buffer-image copy region. *>
fn vk::BufferImageCopy? resolve_texture_copy_region(
    ResolvedSpan* resolved,
    TextureSlot* slot,
    uint row_length_texels,
    uint mip,
    uint base_layer,
    uint layer_count,
    uint x,
    uint y,
    uint width,
    uint height,
) @private {
    if (mip >= slot.mip_levels) return gpu::INVALID_ARGUMENT~;
    uint layers = layer_count == 0 ? 1 : layer_count;
    if ((ulong)base_layer + (ulong)layers > (ulong)slot.array_layers) {
        return gpu::INVALID_ARGUMENT~;
    }

    MipExtent extent = texture_mip_extent(slot, mip);
    if (width == 0) width = extent.width;
    if (height == 0) height = extent.height;
    if ((ulong)x + (ulong)width > (ulong)extent.width
        || (ulong)y + (ulong)height > (ulong)extent.height) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (row_length_texels != 0 && row_length_texels < width) {
        return gpu::INVALID_ARGUMENT~;
    }

    usz texel_size = format_texel_size(slot.format);
    usz row_texels = row_length_texels == 0 ? width : row_length_texels;
    if (texel_size == 0 || row_texels == 0 || height == 0 || layers == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (row_texels > usz::max / texel_size) return gpu::INVALID_ARGUMENT~;
    usz copy_bytes = row_texels * texel_size;
    if ((usz)height > usz::max / copy_bytes) return gpu::INVALID_ARGUMENT~;
    copy_bytes *= height;
    if ((usz)layers > usz::max / copy_bytes) return gpu::INVALID_ARGUMENT~;
    copy_bytes *= layers;
    if (copy_bytes > resolved.size) return gpu::INVALID_ARGUMENT~;
    if (resolved.offset % texel_size != 0 || resolved.offset % 4 != 0) {
        return gpu::INVALID_ARGUMENT~;
    }

    return {
        .buffer_offset       = (vk::DeviceSize)resolved.offset,
        .buffer_row_length   = row_length_texels,
        .buffer_image_height = 0,
        .image_subresource   = {
            .aspect_mask      = format_aspect(slot.format),
            .mip_level        = mip,
            .base_array_layer = base_layer,
            .layer_count      = layers,
        },
        .image_offset = { .x = (int)x, .y = (int)y, .z = 0 },
        .image_extent = { .width = width, .height = height, .depth = 1 },
    };
}

fn void? vk_cmd_copy_buffer_to_texture(
    gpu::CommandEncoderState* commands,
    gpu::BufferTextureCopyDesc* desc,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    TextureSlot* texture = state.textures.get(desc.texture)!;
    ResolvedSpan buffer = resolve_span_slot(
        state:          state,
        span:           &desc.src,
        required:       0,
        operation:      "cmd_copy_buffer_to_texture",
        rejected_field: "GpuSpan",
    )!;
    validate_span_recording_access(
        state:     state,
        record:    record,
        resolved:  &buffer,
        span:      &desc.src,
        operation: "cmd_copy_buffer_to_texture",
    )!;

    validate_texture_recording_access(
        state:     state,
        record:    record,
        slot:      texture,
        handle:    desc.texture,
        operation: "cmd_copy_buffer_to_texture",
    )!;
    validate_span_transfer_usage(
        state:     state,
        resolved:  &buffer,
        span:      &desc.src,
        source:    true,
        operation: "cmd_copy_buffer_to_texture",
    )!;
    if (!texture.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;

    vk::BufferImageCopy region = resolve_texture_copy_region(
        resolved:          &buffer,
        slot:              texture,
        row_length_texels: desc.row_length_texels,
        mip:               desc.mip,
        base_layer:        desc.base_layer,
        layer_count:       desc.layer_count,
        x:                 desc.x,
        y:                 desc.y,
        width:             desc.width,
        height:            desc.height,
    )!;
    vk::cmd_copy_buffer_to_image(
        command_buffer:  record.command_buffer,
        src_buffer:      buffer.slot.buffer,
        dst_image:       texture.image,
        dst_image_layout: vk::ImageLayout.TRANSFER_DST_OPTIMAL,
        region_count:    1,
        p_regions:       &region,
    );
    gpu::note_command_native_call();
}

fn void? vk_cmd_copy_texture_to_buffer(
    gpu::CommandEncoderState* commands,
    gpu::TextureBufferCopyDesc* desc,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    TextureSlot* texture = state.textures.get(desc.texture)!;
    validate_texture_recording_access(
        state:     state,
        record:    record,
        slot:      texture,
        handle:    desc.texture,
        operation: "cmd_copy_texture_to_buffer",
    )!;
    ResolvedSpan buffer = resolve_span_slot(
        state:          state,
        span:           &desc.dst,
        required:       0,
        operation:      "cmd_copy_texture_to_buffer",
        rejected_field: "GpuSpan",
    )!;
    validate_span_recording_access(
        state:     state,
        record:    record,
        resolved:  &buffer,
        span:      &desc.dst,
        operation: "cmd_copy_texture_to_buffer",
    )!;
    validate_span_transfer_usage(
        state:     state,
        resolved:  &buffer,
        span:      &desc.dst,
        source:    false,
        operation: "cmd_copy_texture_to_buffer",
    )!;
    if (!texture.usage.transfer_src) return gpu::INVALID_ARGUMENT~;

    vk::BufferImageCopy region = resolve_texture_copy_region(
        resolved:          &buffer,
        slot:              texture,
        row_length_texels: desc.row_length_texels,
        mip:               desc.mip,
        base_layer:        desc.base_layer,
        layer_count:       desc.layer_count,
        x:                 desc.x,
        y:                 desc.y,
        width:             desc.width,
        height:            desc.height,
    )!;
    vk::cmd_copy_image_to_buffer(
        command_buffer:  record.command_buffer,
        src_image:       texture.image,
        src_image_layout: vk::ImageLayout.TRANSFER_SRC_OPTIMAL,
        dst_buffer:      buffer.slot.buffer,
        region_count:    1,
        p_regions:       &region,
    );
    gpu::note_command_native_call();
}

<* Resolve an indirect argument span: indirect usage, 4-byte offset alignment,
   checked argument bytes within the span, and the span within its backing buffer. *>
fn ResolvedSpan? resolve_indirect_span(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::GpuSpan* span,
    usz arg_size,
    uint count,
    ZString operation,
    ZString rejected_field,
    uint max_count = uint::max,
) {
    if (count > max_count && operation != null) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      rejected_field,
            invariant:           "count must not exceed DeviceCaps.max_draw_indirect_count",
            backend_text:        null,
        )~;
    }
    usz? required_result = gpu::indirect_argument_bytes(
        argument_size: arg_size,
        count:         count,
        max_count:     max_count,
    );
    if (catch size_fault = required_result) {
        if (operation == null) return size_fault~;
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        size_fault,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      rejected_field,
            invariant:           "indirect argument byte size must not overflow",
            backend_text:        null,
        )~;
    }
    usz required = required_result;
    ResolvedSpan resolved = resolve_span_slot(
        state:          state,
        span:           span,
        required:       required,
        operation:      operation,
        rejected_field: rejected_field,
    )!;
    validate_span_recording_access(
        state:     state,
        record:    record,
        resolved:  &resolved,
        span:      span,
        operation: operation,
    )!;
    if (!resolved.slot.usage.indirect) return gpu::INVALID_ARGUMENT~;
    if (resolved.offset % 4 != 0) return gpu::INVALID_ARGUMENT~;
    return resolved;
}

fn void? validate_bound_pipeline_identity(BoundPipeline* pipeline) @private {
    if (pipeline.validation_cell == null
        || !pipeline.validation_cell.used
        || pipeline.validation_cell.generation
            != pipeline.expected_generation) {
        return gpu::INVALID_HANDLE~;
    }
}

fn BoundPipeline*? active_bound_pipeline(
    VkDeviceState* state,
    CommandRecord* record,
    PipelineKind expected_kind,
    ZString operation,
) @private {
    BoundPipeline* pipeline = &record.bound_pipeline;
    if (!pipeline.handle.is_valid()) return gpu::COMMAND_RECORDING_ERROR~;
    validate_bound_pipeline_identity(pipeline)!;
    if (pipeline.kind != expected_kind) {
        ZString invariant = expected_kind == PipelineKind.COMPUTE
            ? "pipeline kind must match the COMPUTE bind point"
            : "pipeline kind must match the GRAPHICS bind point";
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.PIPELINE,
            has_public_identity: true,
            index:               pipeline.handle.index,
            generation:          pipeline.handle.generation,
            debug_name:          null,
            rejected_field:      "pipeline",
            invariant:           invariant,
            backend_text:        null,
        )~;
    }
    return pipeline;
}

fn void? validate_root_address(
    VkDeviceState* state,
    gpu::GpuAddress root,
    ZString operation,
    ZString rejected_field,
) @private {
    if (root != (gpu::GpuAddress)0) return;
    return report_contract_failure(
        state:               state,
        operation:           operation,
        public_fault:        gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
        debug_name:          null,
        rejected_field:      rejected_field,
        invariant:           "root GPU address must be nonzero",
        backend_text:        null,
    )~;
}

fn void? validate_pipeline_render_compatibility(
    BoundPipeline* pipeline,
    RenderCompatibility* render,
) @private {
    if (pipeline.kind != PipelineKind.GRAPHICS) return;
    RenderCompatibility* expected = &pipeline.render;
    if (expected.color_count != render.color_count
        || expected.depth_format != render.depth_format
        || expected.sample_count != render.sample_count) {
        return gpu::INVALID_ARGUMENT~;
    }
    for (uint i = 0; i < render.color_count; i++) {
        if (expected.color_formats[i] != render.color_formats[i]) {
            return gpu::INVALID_ARGUMENT~;
        }
    }
}

fn void? validate_active_pipeline_render_compatibility(
    VkDeviceState* state,
    CommandRecord* record,
    RenderCompatibility* render,
) @private {
    BoundPipeline* pipeline = &record.bound_pipeline;
    if (!pipeline.handle.is_valid()) return;
    validate_bound_pipeline_identity(pipeline)!;
    return validate_pipeline_render_compatibility(pipeline, render);
}

fn RenderCompatibility pipeline_render_compatibility(
    PipelineKey* key,
) @private {
    RenderCompatibility compatibility = {
        .color_count  = key.color_format_count,
        .depth_format = (gpu::Format)key.depth_format,
        .sample_count = (gpu::SampleCount)key.sample_count,
    };
    for (uint i = 0; i < key.color_format_count; i++) {
        compatibility.color_formats[i] =
            (gpu::Format)key.color_formats[i];
    }
    return compatibility;
}

fn void? bind_pipeline_state(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::PipelineHandle pipeline,
) @private {
    validate_descriptor_heap_binding(state)!;
    PipelineCell* cell = state.pipelines.get_cell(pipeline)!;
    PipelineSlot* slot = &cell.value;
    gpu::note_command_pipeline_cache_lookup();
    if (slot.entry >= state.pipeline_cache.entries.len
        || !state.pipeline_cache.entries[slot.entry].used) {
        return gpu::INVALID_HANDLE~;
    }
    PipelineCacheEntry* entry = &state.pipeline_cache.entries[slot.entry];
    BoundPipeline bound = {
        .handle                    = pipeline,
        .validation_cell           = cell,
        .pipeline                  = slot.pipeline,
        .layout                    = slot.layout,
        .generated_dispatch_layout = slot.generated_dispatch_layout,
        .expected_generation       = cell.generation,
        .cache_entry               = slot.entry,
        .kind                      = (PipelineKind)entry.key.kind,
        .render                    = pipeline_render_compatibility(&entry.key),
    };
    if (record.state == CommandState.RECORDING_RENDER_PASS) {
        validate_pipeline_render_compatibility(
            &bound,
            &record.active_render,
        )!;
    }
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    track_command_reference(
        state,
        record,
        pipeline.owner,
        pipeline.index,
        pipeline.generation,
    )!;
    if (pipeline == record.bound_pipeline.handle) return;

    if (bound.cache_entry != record.bound_pipeline.cache_entry
        || !record.bound_pipeline.handle.is_valid()) {
        vk::PipelineBindPoint bind_point = bound.kind == PipelineKind.COMPUTE
            ? vk::PipelineBindPoint.COMPUTE
            : vk::PipelineBindPoint.GRAPHICS;
        vk::cmd_bind_pipeline(record.command_buffer, bind_point, bound.pipeline);
        gpu::note_command_native_call();
        bind_descriptor_heap(
            state:      state,
            cb:         record.command_buffer,
            layout:     bound.layout,
            bind_point: bind_point,
        )!;
    }
    record.bound_pipeline = bound;
}

fn void? vk_cmd_bind_pipeline(
    gpu::CommandEncoderState* commands,
    gpu::PipelineHandle pipeline,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING
        && record.state != CommandState.RECORDING_RENDER_PASS) {
        return gpu::COMMAND_RECORDING_ERROR~;
    }
    VkDeviceState* state = encoder_device_state(commands);
    bind_pipeline_state(state, record, pipeline)!;
}

fn void push_compute_root(
    CommandRecord* record,
    BoundPipeline* pipeline,
    gpu::GpuAddress root,
) @private {
    gpu::RootPush push = { .root_gpu = root };
    vk::cmd_push_constants(
        command_buffer: record.command_buffer,
        layout:         pipeline.layout,
        stage_flags:    vk::ShaderStageFlagBits.COMPUTE_BIT,
        offset:         0,
        size:           gpu::RootPush::size,
        p_values:       &push,
    );
    gpu::note_command_native_call();
}

fn void? vk_cmd_dispatch(
    gpu::CommandEncoderState* commands,
    gpu::GpuAddress root,
    gpu::Vec3u groups,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    BoundPipeline* pipeline = active_bound_pipeline(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.COMPUTE,
        operation:     "cmd_dispatch",
    )!;
    validate_root_address(
        state:          state,
        root:           root,
        operation:      "cmd_dispatch",
        rejected_field: "root",
    )!;
    if (catch limit_fault = gpu::validate_dispatch_groups(groups, state.caps.max_compute_work_group_count)) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_dispatch",
            public_fault:        limit_fault,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "groups",
            invariant:           "group counts must not exceed DeviceCaps.max_compute_work_group_count",
            backend_text:        null,
        )~;
    }
    push_compute_root(record, pipeline, root);
    vk::cmd_dispatch(
        command_buffer: record.command_buffer,
        group_count_x:  groups.x,
        group_count_y:  groups.y,
        group_count_z:  groups.z,
    );
    gpu::note_command_native_call();
}

fn void? vk_cmd_dispatch_indirect(
    gpu::CommandEncoderState* commands,
    gpu::GpuAddress root,
    gpu::GpuSpan args,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    BoundPipeline* pipeline = active_bound_pipeline(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.COMPUTE,
        operation:     "cmd_dispatch_indirect",
    )!;
    validate_root_address(
        state:          state,
        root:           root,
        operation:      "cmd_dispatch_indirect",
        rejected_field: "root",
    )!;
    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    ResolvedSpan args_buffer = resolve_indirect_span(
        state:          state,
        record:         record,
        span:           &args,
        arg_size:       gpu::DispatchIndirectCommand::size,
        count:          1,
        operation:      "cmd_dispatch_indirect",
        rejected_field: "args",
    )!;
    push_compute_root(record, pipeline, root);
    vk::cmd_dispatch_indirect(
        record.command_buffer,
        args_buffer.slot.buffer,
        (vk::DeviceSize)args_buffer.offset,
    );
    gpu::note_command_native_call();
}

fn void? validate_generated_work_count(
    VkDeviceState* state,
    uint count,
    ZString operation,
    ZString rejected_field,
) {
    if (count <= state.caps.max_generated_work_count) return;
    return report_contract_failure(
        state:               state,
        operation:           operation,
        public_fault:        gpu::INVALID_ARGUMENT,
        resource_kind:       gpu::DebugResourceKind.NONE,
        has_public_identity: false,
        index:               0,
        generation:          0,
        debug_name:          null,
        rejected_field:      rejected_field,
        invariant:           "count must not exceed DeviceCaps.max_generated_work_count",
        backend_text:        null,
    )~;
}

fn ResolvedSpan? resolve_generated_work_records(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::GpuSpan* records,
    usz record_size,
    uint max_count,
    ZString operation,
) {
    validate_generated_work_count(state, max_count, operation, "max_count")!;
    usz required = gpu::generated_work_bytes(
        record_size,
        max_count,
        state.caps.max_generated_work_count,
    )!;
    ResolvedSpan resolved = resolve_span_slot(
        state:          state,
        span:           records,
        required:       required,
        operation:      operation,
        rejected_field: "records",
    )!;
    validate_span_recording_access(
        state:     state,
        record:    record,
        resolved:  &resolved,
        span:      records,
        operation: operation,
    )!;
    if (!resolved.slot.usage.indirect || !resolved.slot.usage.addressable) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (resolved.offset % 8 != 0) return gpu::INVALID_ARGUMENT~;
    return resolved;
}

fn void set_generated_dispatch_layout_read_hook(
    VkDeviceState* state,
    GeneratedDispatchLayoutReadHook hook,
    void* user_data,
) @if(env::TESTING) {
    state.generated_dispatch_layout_read_hook = hook;
    state.generated_dispatch_layout_read_user_data = user_data;
    state.generated_dispatch_layout_read_observed =
        (vk::IndirectCommandsLayoutEXT)null;
}

fn void? execute_generated_work(
    VkDeviceState* state,
    CommandRecord* record,
    BoundPipeline* pipeline,
    vk::IndirectCommandsLayoutEXT layout,
    vk::ShaderStageFlags shader_stages,
    ResolvedSpan* records,
    ResolvedSpan* count,
    uint max_count,
) {
    vk::GeneratedCommandsPipelineInfoEXT pipeline_info =
        vk::generated_commands_pipeline_info_ext()
            .set_pipeline(pipeline.pipeline);
    vk::GeneratedCommandsMemoryRequirementsInfoEXT requirements_info =
        vk::generated_commands_memory_requirements_info_ext()
            .set_next(&pipeline_info)
            .set_indirect_commands_layout(layout)
            .set_max_sequence_count(max_count);
    vk::MemoryRequirements2 requirements2 = vk::memory_requirements2();
    state.device_dispatch.generated_work.get_memory_requirements(
        state.device,
        &requirements_info,
        &requirements2,
    );

    GeneratedPreprocessBuffer* preprocess = null;
    if (requirements2.memory_requirements.size > 0) {
        preprocess = acquire_generated_preprocess_buffer(
            state,
            record,
            &requirements2.memory_requirements,
            max_count,
        )!;
    }
    vk::GeneratedCommandsInfoEXT info = vk::generated_commands_info_ext()
        .set_next(&pipeline_info)
        .set_shader_stages(shader_stages)
        .set_indirect_commands_layout(layout)
        .set_indirect_address((vk::DeviceAddress)(records.slot.gpu_base + (gpu::GpuAddress)records.offset))
        .set_indirect_address_size((vk::DeviceSize)records.size)
        .set_max_sequence_count(max_count)
        .set_sequence_count_address((vk::DeviceAddress)(count.slot.gpu_base + (gpu::GpuAddress)count.offset));
    if (preprocess != null) {
        info = info
            .set_preprocess_address(preprocess.address)
            .set_preprocess_size(preprocess.size);
    }
    state.device_dispatch.generated_work.cmd_execute_generated_commands(
        record.command_buffer,
        vk::FALSE,
        &info,
    );
    gpu::note_command_native_call();
}

fn void? vk_cmd_dispatch_generated(
    gpu::CommandEncoderState* commands,
    gpu::GpuSpan records,
    gpu::GpuSpan count_span,
    uint max_dispatch_count,
) @private {
    CommandRecord* record = encoder_command(commands);
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = encoder_device_state(commands);
    BoundPipeline* pipeline = active_bound_pipeline(
        state:         state,
        record:        record,
        expected_kind: PipelineKind.COMPUTE,
        operation:     "cmd_dispatch_generated",
    )!;
    if (!state.generated_work.enabled) return gpu::UNSUPPORTED_FEATURE~;
    validate_generated_work_count(
        state,
        max_dispatch_count,
        "cmd_dispatch_generated",
        "max_dispatch_count",
    )!;
    if (max_dispatch_count == 0) return;

    uint reference_checkpoint = record.reference_count;
    defer catch rollback_command_references(state, record, reference_checkpoint);
    ResolvedSpan resolved_records = resolve_generated_work_records(
        state:       state,
        record:      record,
        records:     &records,
        record_size: gpu::GeneratedDispatchRecord::size,
        max_count:   max_dispatch_count,
        operation:   "cmd_dispatch_generated",
    )!;
    ResolvedSpan resolved_count = resolve_indirect_span(
        state:          state,
        record:         record,
        span:           &count_span,
        arg_size:       4,
        count:          1,
        operation:      "cmd_dispatch_generated",
        rejected_field: "count_span",
    )!;
    $if env::TESTING:
        if (state.generated_dispatch_layout_read_hook != null) {
            state.generated_dispatch_layout_read_hook(
                state.generated_dispatch_layout_read_user_data,
            );
        }
    $endif
    vk::IndirectCommandsLayoutEXT layout = pipeline.generated_dispatch_layout;
    $if env::TESTING:
        state.generated_dispatch_layout_read_observed = layout;
    $endif
    if (layout == null) return gpu::BACKEND_ERROR~;
    execute_generated_work(
        state:         state,
        record:        record,
        pipeline:      pipeline,
        layout:        layout,
        shader_stages: vk::ShaderStageFlagBits.COMPUTE_BIT,
        records:       &resolved_records,
        count:         &resolved_count,
        max_count:     max_dispatch_count,
    )!;
}

fn bool generated_preprocess_compatible(
    GeneratedPreprocessBuffer* buffer,
    vk::MemoryRequirements* requirements,
) {
    if (buffer.size < requirements.size || buffer.memory_type >= 32) return false;
    if (requirements.alignment > 1
        && buffer.address % requirements.alignment != 0) return false;
    return (requirements.memory_type_bits & (1 << buffer.memory_type)) != 0;
}

fn void recycle_generated_preprocess_buffers_locked(
    VkDeviceState* state,
    GeneratedPreprocessBuffer[] buffers,
    uint count,
) {
    for (uint i = 0; i < count; i++) {
        GeneratedPreprocessBuffer* buffer = &buffers[i];
        if (buffer.reservation_owned) {
            RecordingContextState* context = state.recording_contexts.get(
                buffer.reservation_context,
            )!!;
            GeneratedPreprocessBuffer* reserved =
                &context.reserved_preprocess[buffer.reservation_index];
            reserved.reservation_in_use = false;
            *buffer = {};
            continue;
        }
        if (buffer.buffer != null) {
            state.allocator.destroy_buffer(buffer.buffer, buffer.allocation);
        }
        *buffer = {};
    }
}

fn CommandRecordingStats command_recording_stats(VkDeviceState* state) {
    return {
        .preprocess_allocations = state.generated_preprocess_allocations.load(
            AtomicOrdering.RELAXED,
        ),
        .preprocess_reuses = state.generated_preprocess_reuses.load(
            AtomicOrdering.RELAXED,
        ),
        .recording_allocations = state.generated_recording_allocations.load(
            AtomicOrdering.RELAXED,
        ),
    };
}

fn void free_generated_preprocess_buffers(
    VkDeviceState* state,
    GeneratedPreprocessBuffer[] buffers,
) {
    foreach (&buffer : buffers) {
        if (buffer.buffer != null && !buffer.reservation_owned) {
            state.allocator.destroy_buffer(buffer.buffer, buffer.allocation);
        }
    }
    if (buffers.ptr != null) alloc::free(state.host_allocator, buffers.ptr);
}

fn void free_reserved_generated_preprocess_buffers(
    VkDeviceState* state,
    GeneratedPreprocessBuffer[] buffers,
) {
    foreach (&buffer : buffers) {
        if (buffer.buffer != null) {
            state.allocator.destroy_buffer(buffer.buffer, buffer.allocation);
        }
    }
    if (buffers.ptr != null) alloc::free(state.host_allocator, buffers.ptr);
}

fn GeneratedPreprocessBuffer? allocate_generated_preprocess_buffer(
    VkDeviceState* state,
    vk::DeviceSize size,
    vk::DeviceSize alignment,
    uint memory_type_bits,
) {
    vk::BufferUsageFlags2CreateInfoKHR usage2 = {
        .s_type = vk::StructureType.BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR,
        .usage = (vk::BufferUsageFlags2KHR)(
            (ulong)vk::BufferUsageFlagBits2KHR.USAGE_2_PREPROCESS_BUFFER_BIT_EXT
            | (ulong)vk::BufferUsageFlagBits2KHR.USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR
        ),
    };
    vk::BufferCreateInfo buffer_info = vk::buffer_create_info()
        .set_next(&usage2)
        .set_size(size)
        .set_usage(vk::BufferUsageFlagBits.SHADER_DEVICE_ADDRESS_BIT);
    vma::AllocationCreateInfo allocation_info = {
        .usage            = vma::MemoryUsage.AUTO_PREFER_DEVICE,
        .memory_type_bits = memory_type_bits,
    };
    GeneratedPreprocessBuffer buffer;
    vma::AllocationInfo native_info;
    bool owns_pair;
    defer catch {
        if (owns_pair) state.allocator.destroy_buffer(buffer.buffer, buffer.allocation);
    }
    check_device_result(state, state.allocator.create_buffer_with_alignment(
        bi:            &buffer_info,
        ci:            &allocation_info,
        min_alignment: alignment,
        out_buffer:    &buffer.buffer,
        out_alloc:     &buffer.allocation,
        out_info:      &native_info,
    ))!;
    owns_pair = true;
    vk::BufferDeviceAddressInfo address_info =
        vk::buffer_device_address_info().set_buffer(buffer.buffer);
    buffer.address = vk::get_buffer_device_address(state.device, &address_info);
    buffer.size = size;
    buffer.memory_type = native_info.memory_type;
    if (buffer.address == 0) return gpu::BACKEND_ERROR~;
    owns_pair = false;
    (void)state.generated_preprocess_allocations.add(1, AtomicOrdering.RELAXED);
    note_recording_vma_allocation(state);
    return buffer;
}

fn GeneratedPreprocessBuffer*? acquire_generated_preprocess_buffer(
    VkDeviceState* state,
    CommandRecord* record,
    vk::MemoryRequirements* requirements,
    uint max_count,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    RecordingContextState* context = state.recording_contexts.get(
        record.context,
    )!;
    if (max_count > context.generated_scratch.max_commands_per_list
        || requirements.size
            > context.generated_scratch.max_preprocess_bytes
        || record.generated_preprocess_count
            >= record.generated_preprocess.len) {
        note_generated_scratch_miss(state);
        return gpu::CAPACITY_EXCEEDED~;
    }
    foreach (&reserved : context.reserved_preprocess) {
        if (reserved.reservation_in_use
            || !generated_preprocess_compatible(
                reserved,
                requirements,
            )) continue;
        reserved.reservation_in_use = true;
        uint index = record.generated_preprocess_count++;
        record.generated_preprocess[index] = *reserved;
        (void)state.generated_preprocess_reuses.add(
            1,
            AtomicOrdering.RELAXED,
        );
        return &record.generated_preprocess[index];
    }
    note_generated_scratch_miss(state);
    return gpu::CAPACITY_EXCEEDED~;
}

<* Snapshot test/benchmark recording work after externally synchronizing the
   measured interval. *>
fn gpu::RecordingWorkCounters command_recording_work_counters(
    VkDeviceState* state,
) @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS)) {
    return {
        .host_allocations = state.recording_host_allocations.load(
            AtomicOrdering.RELAXED,
        ),
        .command_buffer_allocations = state
            .recording_command_buffer_allocations.load(
                AtomicOrdering.RELAXED,
            ),
        .command_buffer_frees = state.recording_command_buffer_frees.load(
            AtomicOrdering.RELAXED,
        ),
        .command_buffer_resets = state.command_buffer_resets.load(
            AtomicOrdering.RELAXED,
        ),
        .image_view_creations = state.recording_image_view_creations.load(
            AtomicOrdering.RELAXED,
        ),
        .vma_allocations = state.recording_vma_allocations.load(
            AtomicOrdering.RELAXED,
        ),
        .generated_scratch_misses = state.generated_scratch_misses.load(
            AtomicOrdering.RELAXED,
        ),
    };
}

<* Reset test/benchmark recording-work counters while recording is quiescent. *>
fn void reset_command_recording_work_counters(
    VkDeviceState* state,
) @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS)) {
    state.recording_host_allocations.store(0, AtomicOrdering.RELAXED);
    state.recording_command_buffer_allocations.store(
        0,
        AtomicOrdering.RELAXED,
    );
    state.recording_command_buffer_frees.store(0, AtomicOrdering.RELAXED);
    state.command_buffer_resets.store(0, AtomicOrdering.RELAXED);
    state.recording_image_view_creations.store(0, AtomicOrdering.RELAXED);
    state.recording_vma_allocations.store(0, AtomicOrdering.RELAXED);
    state.generated_scratch_misses.store(0, AtomicOrdering.RELAXED);
}

fn vk::IndirectCommandsLayoutEXT? create_generated_work_layout(
    VkDeviceState* state,
    vk::PipelineLayout pipeline_layout,
    vk::ShaderStageFlags shader_stages,
    uint root_size,
    vk::IndirectCommandsTokenTypeEXT action_type,
    uint action_offset,
    uint stride,
) {
    vk::PushConstantRange range = {
        .stage_flags = shader_stages,
        .offset      = 0,
        .size        = root_size,
    };
    vk::IndirectCommandsPushConstantTokenEXT push = {
        .update_range = range,
    };
    vk::IndirectCommandsLayoutTokenEXT[2] tokens = {
        vk::indirect_commands_layout_token_ext()
            .set_type(vk::IndirectCommandsTokenTypeEXT.PUSH_CONSTANT_EXT)
            .set_data({ .p_push_constant = &push })
            .set_offset(0),
        vk::indirect_commands_layout_token_ext()
            .set_type(action_type)
            .set_offset(action_offset),
    };
    vk::IndirectCommandsLayoutCreateInfoEXT info =
        vk::indirect_commands_layout_create_info_ext()
            .set_shader_stages(shader_stages)
            .set_indirect_stride(stride)
            .set_pipeline_layout(pipeline_layout)
            .set_tokens(tokens[..]);
    vk::IndirectCommandsLayoutEXT layout;
    check_backend_result(
        state:     state,
        operation: "create_generated_work_layout",
        result:    state.device_dispatch.generated_work.create_layout(
            state.device,
            &info,
            null,
            &layout,
        ),
    )!;
    return layout;
}
