module gpu::vk;

import gpu;
import vk;
import std::thread;
import vma;
import std::core::mem;
import std::atomic;
import std::atomic::types;
import std::collections::map;

alias PersistentAllocMap = HashMap{usz, vma::VirtualAllocation};
alias QueueSubmit2Fn @private = fn vk::Result (vk::Queue, uint, vk::SubmitInfo2*, vk::Fence);
alias FrameCounterQueryFn @private = fn vk::Result (vk::Device, vk::Semaphore, ulong*);

const gpu::BufferUsage PERSISTENT_ARENA_USAGE = {
    .transfer_src, .transfer_dst, .uniform, .storage, .addressable, .indirect, .index, .shared_queues,
};

// One persistent arena per device: a large addressable buffer suballocated by a
// VMA virtual block. `live` maps each allocation's offset to its virtual
// allocation handle so a span can be freed by offset.
struct PersistentArenaState {
    gpu::BufferHandle  backing_buffer;
    vma::VirtualBlock  virtual_block;
    gpu::GpuAddress    gpu_base;
    void*              cpu_base;
    usz                size;
    PersistentAllocMap live;
}

struct PersistentArenaCreatePlan @private {
    gpu::BufferDesc          desc;
    uint[MAX_QUEUE_FAMILIES] families;
    uint                     family_count;
}

// Per-frame upload arena: a mapped, addressable buffer bump-allocated within a
// frame, reset once its timeline value retires.
struct FrameArenaState {
    gpu::BufferHandle backing_buffer;
    gpu::GpuAddress   gpu_base;
    void*             cpu_base;
    usz               size;
    Atomic{usz}       cursor;
    ulong             frame_timeline_value;
}

struct FrameArenaRange @private {
    usz offset;
    usz next_cursor;
}

<* Create the frame timeline semaphore and one arena buffer per frame-in-flight.
   Arena buffers are real device buffers, so the device's buffer sweep frees
   them; only the host array and the semaphore are released here. *>
fn void? create_frame_resources(VkDeviceState* state, gpu::DeviceDesc* desc) {
    state.frames_in_flight = desc.frames_in_flight == 0 ? 2 : desc.frames_in_flight;
    state.frame_index = 0;

    vk::SemaphoreTypeCreateInfo type_info = vk::semaphore_type_create_info()
        .set_semaphore_type(vk::SemaphoreType.TIMELINE)
        .set_initial_value(0);
    vk::SemaphoreCreateInfo semaphore_info = vk::semaphore_create_info().set_next(&type_info);
    check(vk::create_semaphore(state.device, &semaphore_info, null, &state.frame_timeline))!;
    check(vk::create_semaphore(state.device, &semaphore_info, null, &state.aux_compute_timeline))!;
    check(vk::create_semaphore(state.device, &semaphore_info, null, &state.aux_transfer_timeline))!;
    check(vk::create_semaphore(state.device, &semaphore_info, null, &state.helper_timeline))!;

    usz frame_arena_size = desc.frame_arena_size == 0 ? gpu::DEFAULT_FRAME_ARENA_SIZE : desc.frame_arena_size;
    state.frame_arenas = mem::new_array(FrameArenaState, state.frames_in_flight);
    gpu::Device handle = { .backend = gpu::BackendKind.VULKAN, .vtable = &VK_VTABLE, .backend_state = (void*)state };
    uint[2] frame_families;
    uint frame_family_count = graphics_compute_queue_families(state, frame_families[..]);
    for (uint i = 0; i < state.frames_in_flight; i++) {
        gpu::BufferDesc arena_desc = {
            .size        = frame_arena_size,
            .usage       = { .storage, .addressable, .indirect, .index },
            .memory_kind = gpu::MemoryKind.FRAME_UPLOAD,
            .debug_name  = "frame_arena",
        };
        gpu::BufferHandle buffer = vk_create_buffer_with_families(
            &handle,
            &arena_desc,
            frame_families[:frame_family_count],
        )!;
        mark_buffer_internal(state, buffer);
        gpu::GpuSpan span = vk_get_buffer_span(&handle, buffer)!;
        state.frame_arenas[i] = {
            .backing_buffer = buffer,
            .gpu_base       = span.gpu,
            .cpu_base       = span.cpu,
            .size           = span.size,
        };
    }
}

fn void destroy_frame_resources(VkDeviceState* state) @private {
    if (state.frame_arenas.ptr != null) {
        mem::free(state.frame_arenas.ptr);
        state.frame_arenas = {};
    }
    if (state.helper_timeline != null) {
        vk::destroy_semaphore(state.device, state.helper_timeline, null);
        state.helper_timeline = (vk::Semaphore)null;
    }
    if (state.aux_transfer_timeline != null) {
        vk::destroy_semaphore(state.device, state.aux_transfer_timeline, null);
        state.aux_transfer_timeline = (vk::Semaphore)null;
    }
    if (state.aux_compute_timeline != null) {
        vk::destroy_semaphore(state.device, state.aux_compute_timeline, null);
        state.aux_compute_timeline = (vk::Semaphore)null;
    }
    if (state.frame_timeline != null) {
        vk::destroy_semaphore(state.device, state.frame_timeline, null);
        state.frame_timeline = (vk::Semaphore)null;
    }
}

fn vk::Result real_frame_counter_query(vk::Device device, vk::Semaphore semaphore, ulong* value) @private {
    return vk::get_semaphore_counter_value(device, semaphore, value);
}

fn ulong? frame_timeline_value_with_query(
    VkDeviceState* state,
    ZString operation,
    FrameCounterQueryFn query_fn,
) @private {
    ulong value = 0;
    vk::Result result = query_fn(state.device, state.frame_timeline, &value);
    if (operation == null) {
        check(result)!;
        return value;
    }
    check_backend_result(
        state:     state,
        operation: operation,
        result:    result,
    )!;
    return value;
}

// True once the device timeline has reached the slot's recorded value.
fn bool? frame_arena_retired_with_query(
    VkDeviceState* state,
    FrameArenaState* slot,
    ZString operation,
    FrameCounterQueryFn query_fn,
) @private {
    ulong value = frame_timeline_value_with_query(
        state:     state,
        operation: operation,
        query_fn:  query_fn,
    )!;
    return value >= slot.frame_timeline_value;
}

fn void? report_unretired_frame(VkDeviceState* state) @private {
    if (state.debug_callback != null) {
        gpu::DebugMessage message = {
            .severity     = gpu::DebugMessageSeverity.ERROR,
            .category     = { .backend },
            .operation    = "begin_frame",
            .has_fault    = true,
            .public_fault = gpu::INVALID_RESOURCE_STATE,
            .resource     = {
                .kind                = gpu::DebugResourceKind.FRAME,
                .has_public_identity = false,
            },
            .rejected_field = "Device.frame_state",
            .invariant      = "the selected frame arena must be retired after its wait",
            .backend_text   = "frame timeline remained below the waited retirement value",
        };
        gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
    }
    return gpu::INVALID_RESOURCE_STATE~;
}

<* In-frame and off-frame-pending retires wait for the next end_frame value;
   outside both, reclaim is immediate. *>
fn ulong retire_timeline_value(VkDeviceState* state) {
    ulong counter = state.timeline_counter.load(AtomicOrdering.RELAXED);
    return (state.frame_active || state.off_frame_pending.load(AtomicOrdering.RELAXED)) ? counter + 1 : counter;
}

<* Reset an arena slot's cursor. Rejects if the slot has not yet retired. *>
fn void? reset_frame_arena(VkDeviceState* state, uint index) {
    FrameArenaState* slot = &state.frame_arenas[index];
    if (!frame_arena_retired_with_query(
        state:     state,
        slot:      slot,
        operation: null,
        query_fn:  &real_frame_counter_query,
    )!) {
        return gpu::INVALID_RESOURCE_STATE~;
    }
    slot.cursor.store(0, AtomicOrdering.RELAXED);
}

fn void? begin_frame_with_counter_query(gpu::Device* device, gpu::FrameToken* out_frame, FrameCounterQueryFn query_fn) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    if (state.frame_active) {
        return report_contract_failure(
            state:               state,
            operation:           "begin_frame",
            public_fault:        gpu::INVALID_RESOURCE_STATE,
            resource_kind:       gpu::DebugResourceKind.FRAME,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "Device.frame_state",
            invariant:           "no frame may already be active",
            backend_text:        null,
        )~;
    }
    if (boundary_blocked(state)) {
        return report_contract_failure(
            state:               state,
            operation:           "begin_frame",
            public_fault:        gpu::INVALID_RESOURCE_STATE,
            resource_kind:       gpu::DebugResourceKind.FRAME,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "Device.frame_state",
            invariant:           "no Tier S or Tier C call may cross a frame boundary",
            backend_text:        null,
        )~;
    }
    // WAIT_TIMEOUT must leave frame_index untouched so retry is clean.
    uint next_index = (state.frame_index + 1) % state.frames_in_flight;
    FrameArenaState* slot = &state.frame_arenas[next_index];

    if (slot.frame_timeline_value > 0) {
        ulong wait_value = slot.frame_timeline_value;
        vk::SemaphoreWaitInfo wait_info = vk::semaphore_wait_info()
            .set_semaphore_count(1)
            .set_semaphores((&state.frame_timeline)[:1])
            .set_values((&wait_value)[:1]);
        check_wait_backend_result(
            state:     state,
            operation: "begin_frame",
            result:    vk::wait_semaphores(state.device, &wait_info, 1_000_000_000),
        )!;
    }

    if (!frame_arena_retired_with_query(
        state:     state,
        slot:      slot,
        operation: "begin_frame",
        query_fn:  query_fn,
    )!) {
        return report_unretired_frame(state);
    }
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        // The host wait above proved the timeline reached this value.
        if (slot.frame_timeline_value > state.signaled_shadow) state.signaled_shadow = slot.frame_timeline_value;
        deferred_drain(state);
        state.frame_index = next_index;
    }

    slot.cursor.store(0, AtomicOrdering.RELAXED);
    release_commands_for_frame(state, state.frame_index);
    // Only pools dirtied since this slot last reset pay
    // the driver call; idle contexts cost nothing per frame.
    state.pools_reset_last_frame = 0;
    state.pools_skipped_last_frame = 0;
    foreach (&cell : state.recording_contexts.slots[:state.recording_contexts.next_new]) {
        if (!cell.used) continue;
        PoolDirtyMask* dirty = &cell.value.pool_dirty[state.frame_index];
        if (dirty.graphics) {
            check_backend_result(
                state:     state,
                operation: "begin_frame",
                result:    vk::reset_command_pool(state.device, cell.value.graphics_pools[state.frame_index], (vk::CommandPoolResetFlags)0),
            )!;
            dirty.graphics = false;
            state.pools_reset_last_frame++;
        } else {
            state.pools_skipped_last_frame++;
        }
        if (cell.value.compute_pools_distinct) {
            if (dirty.compute) {
                check_backend_result(
                    state:     state,
                    operation: "begin_frame",
                    result:    vk::reset_command_pool(state.device, cell.value.compute_pools[state.frame_index], (vk::CommandPoolResetFlags)0),
                )!;
                dirty.compute = false;
                state.pools_reset_last_frame++;
            } else {
                state.pools_skipped_last_frame++;
            }
        }
        if (dirty.transfer) {
            check_backend_result(
                state:     state,
                operation: "begin_frame",
                result:    vk::reset_command_pool(state.device, cell.value.transfer_pools[state.frame_index], (vk::CommandPoolResetFlags)0),
            )!;
            dirty.transfer = false;
            state.pools_reset_last_frame++;
        } else {
            state.pools_skipped_last_frame++;
        }
    }
    state.allocator.set_current_frame_index(state.frame_index);
    ulong generation = state.frame_generation + 1;
    if (generation == 0) generation = 1;
    state.frame_generation = generation;
    state.active_frame_generation = generation;
    state.frame_active = true;
    *out_frame = { .device = device, .generation = generation };
}

<* Resolve a bump range without forming an unchecked alignment or extent sum.
   Pure. *>
fn FrameArenaRange? frame_arena_range(usz cursor, usz capacity, usz size, usz align) @private {
    if (!gpu::is_pow2(align) || size == 0) return gpu::INVALID_ARGUMENT~;
    if (cursor > capacity) return gpu::ARENA_FULL~;

    usz mask = align - 1;
    usz remainder = cursor & mask;
    usz padding = remainder == 0 ? 0 : align - remainder;
    if (padding > capacity - cursor) return gpu::ARENA_FULL~;

    usz offset = cursor + padding;
    if (!gpu::range_fits(capacity, offset, size)) return gpu::ARENA_FULL~;
    return {
        .offset = offset,
        .next_cursor = offset + size,
    };
}

fn void? vk_begin_frame(gpu::Device* device, gpu::FrameToken* out_frame) @private {
    return begin_frame_with_counter_query(device, out_frame, &real_frame_counter_query);
}

fn VkDeviceState*? resolve_frame_token(gpu::FrameToken* frame, ZString operation) @private {
    if (frame == null || frame.device == null || frame.generation == 0) return gpu::INVALID_HANDLE~;
    if (frame.device.backend_state == null) return gpu::INVALID_HANDLE~;
    VkDeviceState* state = (VkDeviceState*)frame.device.backend_state;
    if (!state.frame_active || state.active_frame_generation != frame.generation) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_HANDLE,
            resource_kind:       gpu::DebugResourceKind.FRAME,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "FrameToken.generation",
            invariant:           "must identify the active frame generation",
            backend_text:        null,
        )~;
    }
    return state;
}

<* Lock-free CAS bump: concurrent callers get disjoint, aligned spans. *>
fn gpu::GpuSpan? vk_alloc_frame_span(gpu::FrameToken* frame, usz size, usz align) @private {
    VkDeviceState* state = resolve_frame_token(frame, "alloc_frame_span")!;
    enter_tracked(state);
    defer exit_tracked(state);
    if (!state.frame_active || state.active_frame_generation != frame.generation) {
        return report_contract_failure(
            state:               state,
            operation:           "alloc_frame_span",
            public_fault:        gpu::INVALID_HANDLE,
            resource_kind:       gpu::DebugResourceKind.FRAME,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "FrameToken.generation",
            invariant:           "must remain the active frame generation",
            backend_text:        null,
        )~;
    }
    if (!gpu::is_pow2(align)) {
        return report_contract_failure(
            state:               state,
            operation:           "alloc_frame_span",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.FRAME,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "align",
            invariant:           "must be a nonzero power of two",
            backend_text:        null,
        )~;
    }
    if (size == 0) {
        return report_contract_failure(
            state:               state,
            operation:           "alloc_frame_span",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.FRAME,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "size",
            invariant:           "must be greater than zero",
            backend_text:        null,
        )~;
    }
    FrameArenaState* slot = &state.frame_arenas[state.frame_index];

    usz old = slot.cursor.load(AtomicOrdering.RELAXED);
    while (true) {
        FrameArenaRange? range = frame_arena_range(
            cursor: old,
            capacity: slot.size,
            size: size,
            align: align,
        );
        if (catch range_fault = range) {
            return report_contract_failure(
                state:               state,
                operation:           "alloc_frame_span",
                public_fault:        range_fault,
                resource_kind:       gpu::DebugResourceKind.FRAME,
                has_public_identity: false,
                index:               0,
                generation:          0,
                debug_name:          null,
                rejected_field:      "size",
                invariant:           "the aligned allocation must fit the current frame arena",
                backend_text:        null,
            )~;
        }
        usz seen = mem::compare_exchange(
            &slot.cursor.data,
            old,
            range.next_cursor,
            AtomicOrdering.SEQ_CONSISTENT,
            AtomicOrdering.RELAXED,
        );
        if (seen == old) {
            return {
                .gpu    = (gpu::GpuAddress)((ulong)slot.gpu_base + range.offset),
                .cpu    = slot.cpu_base == null ? null : (char*)slot.cpu_base + range.offset,
                .size   = size,
                .buffer = slot.backing_buffer,
                .offset = range.offset,
                .kind   = gpu::MemoryKind.FRAME_UPLOAD,
            };
        }
        old = seen;
    }
}

fn vk::Result real_queue_submit2(vk::Queue queue, uint submit_count, vk::SubmitInfo2* submits, vk::Fence fence) @private {
    return vk::queue_submit2(queue, submit_count, submits, fence);
}

<* Queue-ordered empty submit signaling `value` on `semaphore` after all
   prior work on `queue` completes. *>
fn void? submit_timeline_signal(VkDeviceState* state, vk::Semaphore semaphore, vk::Queue queue, Mutex* mutex, ulong value, vk::SemaphoreSubmitInfo[] wait_infos, QueueSubmit2Fn submit_fn) @private {
    vk::SemaphoreSubmitInfo signal = vk::semaphore_submit_info()
        .set_semaphore(semaphore)
        .set_value(value)
        .set_stage_mask(vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT);
    vk::SubmitInfo2 submit = vk::submit_info2().set_signal_semaphore_infos((&signal)[:1]);
    if (wait_infos.len > 0) submit = submit.set_wait_semaphore_infos(wait_infos);
    mutex.lock()!!;
    defer mutex.unlock();
    check_backend_result(
        state:     state,
        operation: "end_frame",
        result:    submit_fn(
            queue,
            1,
            &submit,
            (vk::Fence)null,
        ),
    )!;
    state.submit_stats.empty_signal_submits++;
}

<* The frame fence is a single queue-ordered empty submit: compute/transfer
   queues used since the last boundary (in-frame or off) piggyback an aux
   signal onto their own user submits (vk_submit), and the graphics
   frame-signal submit waits each used queue's latest recorded aux value, so
   the frame value implies every queue finished. Off-frame graphics submits
   need no flag — this signal is queue-ordered behind them. *>
fn void? end_frame_with_submit(gpu::FrameToken* frame, QueueSubmit2Fn submit_fn) @private {
    VkDeviceState* state = resolve_frame_token(frame, "end_frame")!;
    if (boundary_blocked(state)) {
        return report_contract_failure(
            state:               state,
            operation:           "end_frame",
            public_fault:        gpu::INVALID_RESOURCE_STATE,
            resource_kind:       gpu::DebugResourceKind.FRAME,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "Device.frame_state",
            invariant:           "no Tier S or Tier C call may cross a frame boundary",
            backend_text:        null,
        )~;
    }
    FrameArenaState* slot = &state.frame_arenas[state.frame_index];

    // frame_timeline receives exactly one signal per frame (counter + 1),
    // preserving retire_timeline_value's mid-frame contract.
    ulong frame_value = state.timeline_counter.load(AtomicOrdering.RELAXED) + 1;

    vk::SemaphoreSubmitInfo[2] waits;
    uint wait_count = 0;
    if (state.compute_used_since_frame) {
        waits[wait_count++] = vk::semaphore_submit_info()
            .set_semaphore(state.aux_compute_timeline)
            .set_value(state.aux_compute_last_signaled)
            .set_stage_mask(vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT);
    }
    if (state.transfer_used_since_frame) {
        waits[wait_count++] = vk::semaphore_submit_info()
            .set_semaphore(state.aux_transfer_timeline)
            .set_value(state.aux_transfer_last_signaled)
            .set_stage_mask(vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT);
    }

    submit_timeline_signal(
        state:      state,
        semaphore:  state.frame_timeline,
        queue:      state.graphics_queue,
        mutex:      &state.graphics_queue_mutex,
        value:      frame_value,
        wait_infos: waits[:wait_count],
        submit_fn:  submit_fn,
    )!;

    state.timeline_counter.store(frame_value, AtomicOrdering.RELAXED);
    slot.frame_timeline_value = frame_value;
    state.compute_used_since_frame = false;
    state.transfer_used_since_frame = false;
    state.off_frame_pending.store(false, AtomicOrdering.RELAXED);
    state.frame_active = false;
    state.active_frame_generation = 0;
    *frame = {};
}

fn void? vk_end_frame(gpu::FrameToken* frame) @private {
    return end_frame_with_submit(frame, &real_queue_submit2);
}

fn gpu::BufferDesc persistent_arena_desc(usz arena_size) @private {
    return {
        .size        = arena_size,
        .usage       = PERSISTENT_ARENA_USAGE,
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .debug_name  = "persistent_arena",
    };
}

fn PersistentArenaCreatePlan persistent_arena_create_plan(VkDeviceState* state, usz arena_size) @private {
    PersistentArenaCreatePlan plan = {
        .desc = persistent_arena_desc(arena_size),
    };
    plan.family_count = used_queue_families(state, plan.families[..]);
    return plan;
}

<* Create the persistent arena: a large addressable buffer plus a VMA virtual
   block that suballocates it. *>
fn void? create_persistent_arena(VkDeviceState* state, gpu::DeviceDesc* device_desc) {
    usz arena_size = device_desc.persistent_arena_size == 0 ? gpu::DEFAULT_PERSISTENT_ARENA_SIZE : device_desc.persistent_arena_size;
    gpu::Device handle = { .backend = gpu::BackendKind.VULKAN, .vtable = &VK_VTABLE, .backend_state = (void*)state };
    PersistentArenaCreatePlan plan = persistent_arena_create_plan(state, arena_size);
    gpu::BufferHandle buffer = vk_create_buffer_with_families(&handle, &plan.desc, plan.families[:plan.family_count])!;
    mark_buffer_internal(state, buffer);
    gpu::GpuSpan span = vk_get_buffer_span(&handle, buffer)!;

    vma::VirtualBlockCreateInfo block_info = { .size = (vk::DeviceSize)arena_size };
    vma::VirtualBlock block;
    check(vma::create_virtual_block(&block_info, &block))!;

    state.persistent_arena.backing_buffer = buffer;
    state.persistent_arena.virtual_block  = block;
    state.persistent_arena.gpu_base       = span.gpu;
    state.persistent_arena.cpu_base       = span.cpu;
    state.persistent_arena.size           = span.size;
    state.persistent_arena.live.init(mem);
}

fn void destroy_persistent_arena(VkDeviceState* state) @private {
    if (state.persistent_arena.virtual_block == null) return;
    state.persistent_arena.virtual_block.destroy();
    state.persistent_arena.virtual_block = (vma::VirtualBlock)null;
    state.persistent_arena.live.free();
}

fn gpu::GpuSpan? vk_alloc_persistent_span(gpu::Device* device, gpu::PersistentAllocDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    if (desc.memory_kind != gpu::MemoryKind.PERSISTENT_UPLOAD) {
        return report_contract_failure(
            state:               state,
            operation:           "alloc_persistent_span",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.PERSISTENT_SPAN,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "PersistentAllocDesc.memory_kind",
            invariant:           "must be PERSISTENT_UPLOAD",
            backend_text:        null,
        )~;
    }
    if (desc.size == 0) {
        return report_contract_failure(
            state:               state,
            operation:           "alloc_persistent_span",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.PERSISTENT_SPAN,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "PersistentAllocDesc.size",
            invariant:           "must be greater than zero",
            backend_text:        null,
        )~;
    }
    // align 0 = no requirement (documented); anything else must be a power
    // of two (VMA virtual-alloc requirement).
    if (desc.align != 0 && !gpu::is_pow2(desc.align)) {
        return report_contract_failure(
            state:               state,
            operation:           "alloc_persistent_span",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.PERSISTENT_SPAN,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "PersistentAllocDesc.align",
            invariant:           "must be zero or a power of two",
            backend_text:        null,
        )~;
    }
    uint unsupported_usage = (uint)desc.usage & ~(uint)PERSISTENT_ARENA_USAGE;
    if (unsupported_usage != 0) {
        return report_contract_failure(
            state:               state,
            operation:           "alloc_persistent_span",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.PERSISTENT_SPAN,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "PersistentAllocDesc.usage",
            invariant:           "contains unsupported persistent-arena flags",
            backend_text:        null,
        )~;
    }
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    PersistentArenaState* arena = &state.persistent_arena;

    vma::VirtualAllocationCreateInfo create_info = {
        .size      = (vk::DeviceSize)desc.size,
        .alignment = (vk::DeviceSize)desc.align,
    };
    vma::VirtualAllocation allocation;
    vk::DeviceSize offset;
    check_virtual_allocate_backend_result(
        state:     state,
        operation: "alloc_persistent_span",
        result:    arena.virtual_block.allocate(&create_info, &allocation, &offset),
    )!;

    arena.live.set((usz)offset, allocation);
    return (gpu::GpuSpan){
        .gpu    = (gpu::GpuAddress)((ulong)arena.gpu_base + offset),
        .cpu    = arena.cpu_base == null ? null : (char*)arena.cpu_base + offset,
        .size   = desc.size,
        .buffer = arena.backing_buffer,
        .offset = (usz)offset,
        .kind   = gpu::MemoryKind.PERSISTENT_UPLOAD,
    };
}

fn void? vk_free_persistent_span(gpu::Device* device, gpu::GpuSpan span) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    PersistentArenaState* arena = &state.persistent_arena;
    if ((ulong)span.buffer != (ulong)arena.backing_buffer) {
        return report_contract_failure(
            state:               state,
            operation:           "free_persistent_span",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.PERSISTENT_SPAN,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "GpuSpan.buffer",
            invariant:           "must identify this device's persistent arena backing buffer",
            backend_text:        null,
        )~;
    }

    if (try allocation = arena.live.get(span.offset)) {
        arena.virtual_block.free(allocation);
        arena.live.remove(span.offset)!;
        return;
    }
    return report_contract_failure(
        state:               state,
        operation:           "free_persistent_span",
        public_fault:        gpu::INVALID_HANDLE,
        resource_kind:       gpu::DebugResourceKind.PERSISTENT_SPAN,
        has_public_identity: false,
        index:               0,
        generation:          0,
        debug_name:          null,
        rejected_field:      "GpuSpan.offset",
        invariant:           "must identify a live persistent allocation",
        backend_text:        null,
    )~;
}

fn gpu::PersistentArenaStats? vk_get_persistent_stats(gpu::Device* device) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    PersistentArenaState* arena = &state.persistent_arena;
    vma::Statistics stats;
    arena.virtual_block.get_statistics(&stats);
    return (gpu::PersistentArenaStats){
        .capacity_bytes   = (ulong)arena.size,
        .used_bytes       = (ulong)stats.allocation_bytes,
        .allocation_count = (ulong)stats.allocation_count,
    };
}
