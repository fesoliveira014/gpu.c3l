module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;
import vma;
import std::core::mem;
import std::atomic;
import std::atomic::types;

alias QueueSubmit2Fn @private = fn vk::Result (vk::Queue, uint, vk::SubmitInfo2*, vk::Fence);
alias FrameCounterQueryFn @private = fn vk::Result (vk::Device, vk::Semaphore, ulong*);

const uint PERSISTENT_ALLOCATION_FREE_NONE @private = 0;
const BufferUsage PERSISTENT_ARENA_USAGE = {
    .transfer_src, .transfer_dst, .uniform, .storage, .addressable, .indirect, .index,
};


struct PersistentAllocationHandle {
    ulong owner;
    uint  index;
    uint  generation;
}

struct PersistentAllocationState {
    vma::VirtualAllocation allocation;
    usz                    offset;
    usz                    size;
    gpu::QueueRoles        access;
    Atomic{uint}           validation_references;
}

struct PersistentAllocationCell {
    PersistentAllocationState value;
    uint                      generation;
    uint                      next_free;
    bool                      used;
}

struct PersistentAllocationTable {
    PersistentAllocationCell[] slots;
    ulong                      owner;
    uint                       free_head;
    uint                       next_new;
}

struct PersistentArenaState {
    BufferHandle               backing_buffer;
    vma::VirtualBlock          virtual_block;
    usz                        size;
    PersistentAllocationTable allocations;
}

struct PersistentArenaCreatePlan @private {
    BufferDesc               desc;
    uint[MAX_QUEUE_FAMILIES] families;
    uint                     family_count;
}

// Per-frame upload arena: a mapped, addressable buffer bump-allocated within a
// frame, reset once its timeline value retires.
struct FrameArenaState {
    BufferHandle      backing_buffer;
    usz               size;
    gpu::QueueRoles   access;
    Atomic{usz}       cursor;
    ulong             frame_timeline_value;
}

struct FrameArenaRange @private {
    usz offset;
    usz next_cursor;
}

fn PersistentAllocationHandle? PersistentAllocationTable.alloc(
    &self,
    PersistentAllocationState value,
) {
    uint index;
    bool found = false;
    while (self.free_head != PERSISTENT_ALLOCATION_FREE_NONE) {
        index = self.free_head - 1;
        self.free_head = self.slots[index].next_free;
        if (can_allocate_handle_generation(self.slots[index].generation)) {
            found = true;
            break;
        }
        self.slots[index].next_free = PERSISTENT_ALLOCATION_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        index = self.next_new++;
        self.slots[index].generation = 0;
    }

    self.slots[index].value = value;
    self.slots[index].generation = next_handle_generation(self.slots[index].generation)!;
    self.slots[index].used = true;
    return {
        .owner      = self.owner,
        .index      = index,
        .generation = self.slots[index].generation,
    };
}

fn PersistentAllocationState*? PersistentAllocationTable.get(
    &self,
    PersistentAllocationHandle handle,
) {
    if (handle.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint index = handle.index;
    if (index >= self.slots.len || !self.slots[index].used) return gpu::INVALID_HANDLE~;
    if (self.slots[index].generation != handle.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[index].value;
}

fn void? PersistentAllocationTable.free(&self, PersistentAllocationHandle handle) {
    self.get(handle)!;
    uint index = handle.index;
    self.slots[index].used = false;
    if (!should_recycle_handle_generation(self.slots[index].generation)) {
        self.slots[index].generation = HANDLE_GENERATION_MAX;
        self.slots[index].next_free = PERSISTENT_ALLOCATION_FREE_NONE;
        return;
    }
    self.slots[index].generation = next_handle_generation(self.slots[index].generation)!;
    self.slots[index].next_free = self.free_head;
    self.free_head = index + 1;
}

<* Create the frame timeline semaphore and one arena buffer per frame-in-flight.
   Teardown's internal-buffer pass frees the arena allocations; this function's
   counterpart releases the host array and semaphore. *>
fn void? create_frame_resources(VkDeviceState* state, gpu::DeviceDesc* desc) {
    state.frames_in_flight = desc.frames_in_flight == 0 ? 2 : desc.frames_in_flight;
    state.frame_index = 0;

    vk::SemaphoreTypeCreateInfo type_info = vk::semaphore_type_create_info()
        .set_semaphore_type(vk::SemaphoreType.TIMELINE)
        .set_initial_value(0);
    vk::SemaphoreCreateInfo semaphore_info = vk::semaphore_create_info().set_next(&type_info);
    check_device_result(state, vk::create_semaphore(
        device:        state.device,
        p_create_info: &semaphore_info,
        p_allocator:   null,
        p_semaphore:   &state.frame_timeline,
    ))!;
    check_device_result(state, vk::create_semaphore(
        device:        state.device,
        p_create_info: &semaphore_info,
        p_allocator:   null,
        p_semaphore:   &state.aux_compute_timeline,
    ))!;
    check_device_result(state, vk::create_semaphore(
        device:        state.device,
        p_create_info: &semaphore_info,
        p_allocator:   null,
        p_semaphore:   &state.aux_transfer_timeline,
    ))!;
    check_device_result(state, vk::create_semaphore(
        device:        state.device,
        p_create_info: &semaphore_info,
        p_allocator:   null,
        p_semaphore:   &state.helper_timeline,
    ))!;

    usz frame_arena_size = desc.frame_arena_size == 0 ? gpu::DEFAULT_FRAME_ARENA_SIZE : desc.frame_arena_size;
    state.frame_arenas = mem::new_array(FrameArenaState, state.frames_in_flight);
    uint[MAX_QUEUE_FAMILIES] frame_families;
    uint frame_family_count = graphics_compute_queue_families(state, frame_families[..]);
    gpu::QueueRoles frame_access = {
        .graphics = state.selected_queue_roles.graphics,
        .compute  = state.selected_queue_roles.compute,
    };
    if ((uint)frame_access == 0) frame_access.transfer = true;
    for (uint i = 0; i < state.frames_in_flight; i++) {
        BufferDesc arena_desc = {
            .size        = frame_arena_size,
            .usage       = { .storage, .addressable, .indirect, .index },
            .memory_kind = gpu::MemoryKind.FRAME_UPLOAD,
            .access      = frame_access,
            .debug_name  = "frame_arena",
        };
        BufferHandle buffer = create_buffer_with_families_for_state(
            state,
            &arena_desc,
            frame_families[:frame_family_count],
        )!;
        mark_buffer_internal(state, buffer);
        gpu::GpuSpan span = get_buffer_span_for_state(state, buffer)!;
        ResolvedSpan resolved = resolve_span(state, &span)!;
        state.frame_arenas[i] = {
            .backing_buffer = buffer,
            .size           = resolved.size,
            .access         = resolved.access,
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
        check_device_result(state, result)!;
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
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
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
    state.frame_index = next_index;

    slot.cursor.store(0, AtomicOrdering.RELAXED);
    state.allocator.set_current_frame_index(state.frame_index);
    ulong generation = state.frame_generation + 1;
    if (generation == 0) generation = 1;
    state.frame_generation = generation;
    state.active_frame_generation = generation;
    state.frame_active = true;
    *out_frame = { .device = *device, .generation = generation };
}

<* Resolve a bump range without forming an unchecked alignment or extent sum.
   Pure. *>
fn FrameArenaRange? frame_arena_range(usz cursor, usz capacity, usz size, usz align) @private {
    if (!is_pow2(align) || size == 0) return gpu::INVALID_ARGUMENT~;
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
    if (frame == null || frame.generation == 0) return gpu::INVALID_HANDLE~;
    if (gpu::device_backend_state_ptr(&frame.device) == null) return gpu::INVALID_HANDLE~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&frame.device);
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
    if (!is_pow2(align)) {
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
                .owner      = slot.backing_buffer.owner,
                .index      = slot.backing_buffer.index,
                .generation = slot.backing_buffer.generation,
                .offset     = range.offset,
                .size       = size,
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

<* The frame timeline is queue-ordered on the first admitted role. Other
   queues piggyback auxiliary signals that the frame submit waits. *>
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

    // frame_timeline receives exactly one signal per frame.
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
        queue:      frame_queue_for(state),
        mutex:      frame_queue_mutex_for(state),
        value:      frame_value,
        wait_infos: waits[:wait_count],
        submit_fn:  submit_fn,
    )!;

    state.timeline_counter.store(frame_value, AtomicOrdering.RELAXED);
    slot.frame_timeline_value = frame_value;
    state.compute_used_since_frame = false;
    state.transfer_used_since_frame = false;
    state.off_frame_pending_queues.store(0, AtomicOrdering.RELAXED);
    state.frame_active = false;
    state.active_frame_generation = 0;
    *frame = {};
}

fn void? vk_end_frame(gpu::FrameToken* frame) @private {
    return end_frame_with_submit(frame, &real_queue_submit2);
}

fn BufferDesc persistent_arena_desc(
    usz arena_size,
    gpu::QueueRoles access,
) @private {
    return {
        .size        = arena_size,
        .usage       = PERSISTENT_ARENA_USAGE,
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .access      = access,
        .debug_name  = "persistent_arena",
    };
}

fn PersistentArenaCreatePlan persistent_arena_create_plan(VkDeviceState* state, usz arena_size) @private {
    PersistentArenaCreatePlan plan = {
        .desc = persistent_arena_desc(arena_size, state.selected_queue_roles),
    };
    plan.family_count = resource_queue_families(state, plan.desc.access, plan.families[..]);
    return plan;
}

<* Create the persistent arena: a large addressable buffer plus a VMA virtual
   block that suballocates it. *>
fn void? create_persistent_arena(VkDeviceState* state, gpu::DeviceDesc* device_desc) {
    usz arena_size = device_desc.persistent_arena_size == 0 ? gpu::DEFAULT_PERSISTENT_ARENA_SIZE : device_desc.persistent_arena_size;
    PersistentArenaCreatePlan plan = persistent_arena_create_plan(state, arena_size);
    BufferHandle buffer = create_buffer_with_families_for_state(
        state:          state,
        desc:           &plan.desc,
        queue_families: plan.families[:plan.family_count],
    )!;
    mark_buffer_internal(state, buffer);
    gpu::GpuSpan span = get_buffer_span_for_state(state, buffer)!;
    ResolvedSpan resolved = resolve_span(state, &span)!;

    vma::VirtualBlockCreateInfo block_info = { .size = (vk::DeviceSize)arena_size };
    vma::VirtualBlock block;
    check(vma::create_virtual_block(&block_info, &block))!;

    state.persistent_arena.backing_buffer = buffer;
    state.persistent_arena.virtual_block  = block;
    state.persistent_arena.size           = resolved.size;
    state.persistent_arena.allocations.owner = gpu::resource_owner_tag(
        state.resource_owner,
        gpu::ResourceOwnerKind.PERSISTENT_ALLOCATION,
    );
    state.persistent_arena.allocations.slots = mem::new_array(
        PersistentAllocationCell,
        gpu::MAX_PERSISTENT_ALLOCATIONS,
    );
}

fn void destroy_persistent_arena(VkDeviceState* state) @private {
    if (state.persistent_arena.virtual_block == null) return;
    state.persistent_arena.virtual_block.destroy();
    state.persistent_arena.virtual_block = (vma::VirtualBlock)null;
    mem::free(state.persistent_arena.allocations.slots.ptr);
    state.persistent_arena.allocations = {};
}

fn gpu::GpuSpan? vk_alloc_persistent_span(gpu::Device* device, gpu::PersistentAllocDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
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
    if (desc.align != 0 && !is_pow2(desc.align)) {
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

    defer catch arena.virtual_block.free(allocation);
    PersistentAllocationState allocation_state = {
        .allocation = allocation,
        .offset     = (usz)offset,
        .size       = desc.size,
        .access     = desc.access,
    };
    PersistentAllocationHandle handle = arena.allocations.alloc(allocation_state)!;
    return {
        .owner      = handle.owner,
        .index      = handle.index,
        .generation = handle.generation,
        .size       = desc.size,
    };
}

fn void? vk_free_persistent_span(gpu::Device* device, gpu::GpuSpan span) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    PersistentArenaState* arena = &state.persistent_arena;
    if (span.owner != arena.allocations.owner) {
        return report_contract_failure(
            state:               state,
            operation:           "free_persistent_span",
            public_fault:        gpu::INVALID_HANDLE,
            resource_kind:       gpu::DebugResourceKind.PERSISTENT_SPAN,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "GpuSpan.owner",
            invariant:           "must identify this device's persistent allocation table",
            backend_text:        null,
        )~;
    }

    PersistentAllocationHandle handle = {
        .owner      = span.owner,
        .index      = span.index,
        .generation = span.generation,
    };
    PersistentAllocationState*? allocation_result = arena.allocations.get(handle);
    if (catch allocation_fault = allocation_result) {
        return report_contract_failure(
            state:               state,
            operation:           "free_persistent_span",
            public_fault:        allocation_fault,
            resource_kind:       gpu::DebugResourceKind.PERSISTENT_SPAN,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "GpuSpan.identity",
            invariant:           "must identify a live persistent allocation",
            backend_text:        null,
        )~;
    }
    PersistentAllocationState* allocation = allocation_result;
    if (span.offset != 0 || span.size != allocation.size) {
        return report_contract_failure(
            state:               state,
            operation:           "free_persistent_span",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.PERSISTENT_SPAN,
            has_public_identity: true,
            index:               span.index,
            generation:          span.generation,
            debug_name:          null,
            rejected_field:      "GpuSpan",
            invariant:           "must preserve the complete owning allocation range",
            backend_text:        null,
        )~;
    }

    if (allocation.validation_references.load(AtomicOrdering.RELAXED) > 0) {
        drain_completed_submitted_commands(state)!;
        if (allocation.validation_references.load(AtomicOrdering.RELAXED) > 0) {
            return gpu::RESOURCE_IN_USE~;
        }
    }
    arena.virtual_block.free(allocation.allocation);
    arena.allocations.free(handle)!;
}

fn gpu::PersistentArenaStats? vk_get_persistent_stats(gpu::Device* device) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    PersistentArenaState* arena = &state.persistent_arena;
    vma::Statistics stats;
    arena.virtual_block.get_statistics(&stats);
    return (gpu::PersistentArenaStats){
        .capacity_bytes   = (ulong)arena.size,
        .used_bytes       = (ulong)stats.allocation_bytes,
        .allocation_count = (ulong)stats.allocation_count,
    };
}
