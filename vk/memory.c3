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

// vertex stays out (unimplemented interop); shared_queues stays out (the
// arena is single-queue-consumed today).
const gpu::BufferUsage PERSISTENT_ARENA_USAGE = {
    .transfer_src, .transfer_dst, .uniform, .storage, .addressable, .indirect, .index,
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
    check(vk::create_semaphore(state.device, &semaphore_info, null, &state.aux_timeline))!;
    check(vk::create_semaphore(state.device, &semaphore_info, null, &state.helper_timeline))!;

    usz frame_arena_size = desc.frame_arena_size == 0 ? gpu::DEFAULT_FRAME_ARENA_SIZE : desc.frame_arena_size;
    state.frame_arenas = mem::new_array(FrameArenaState, state.frames_in_flight);
    gpu::Device handle = { .backend = gpu::BackendKind.VULKAN, .vtable = &VK_VTABLE, .backend_state = (void*)state };
    for (uint i = 0; i < state.frames_in_flight; i++) {
        gpu::BufferDesc arena_desc = {
            .size        = frame_arena_size,
            .usage       = { .storage, .addressable, .indirect, .index },
            .memory_kind = gpu::MemoryKind.FRAME_UPLOAD,
            .debug_name  = "frame_arena",
        };
        gpu::BufferHandle buffer = vk_create_buffer(&handle, &arena_desc)!;
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
    if (state.aux_timeline != null) {
        vk::destroy_semaphore(state.device, state.aux_timeline, null);
        state.aux_timeline = (vk::Semaphore)null;
    }
    if (state.frame_timeline != null) {
        vk::destroy_semaphore(state.device, state.frame_timeline, null);
        state.frame_timeline = (vk::Semaphore)null;
    }
}

// True once the device timeline has reached the slot's recorded value.
fn bool frame_arena_retired(VkDeviceState* state, FrameArenaState* slot) {
    ulong value = 0;
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &value);
    return value >= slot.frame_timeline_value;
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
    if (!frame_arena_retired(state, slot)) return gpu::INVALID_RESOURCE_STATE~;
    slot.cursor.store(0, AtomicOrdering.RELAXED);
}

fn void? vk_begin_frame(gpu::Device* device) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    if (boundary_blocked(state)) return gpu::INVALID_RESOURCE_STATE~;
    state.frame_index = (state.frame_index + 1) % state.frames_in_flight;
    FrameArenaState* slot = &state.frame_arenas[state.frame_index];

    if (slot.frame_timeline_value > 0) {
        ulong wait_value = slot.frame_timeline_value;
        vk::SemaphoreWaitInfo wait_info = vk::semaphore_wait_info()
            .set_semaphore_count(1)
            .set_semaphores((&state.frame_timeline)[:1])
            .set_values((&wait_value)[:1]);
        check(vk::wait_semaphores(state.device, &wait_info, 1_000_000_000))!;
    }

    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        deferred_drain(state);
    }

    reset_frame_arena(state, state.frame_index)!;
    foreach (&cell : state.recording_contexts.slots[:state.recording_contexts.next_new]) {
        if (!cell.used) continue;
        check(vk::reset_command_pool(state.device, cell.value.graphics_pools[state.frame_index], (vk::CommandPoolResetFlags)0))!;
        if (cell.value.compute_pools_distinct) {
            check(vk::reset_command_pool(state.device, cell.value.compute_pools[state.frame_index], (vk::CommandPoolResetFlags)0))!;
        }
        check(vk::reset_command_pool(state.device, cell.value.transfer_pools[state.frame_index], (vk::CommandPoolResetFlags)0))!;
    }
    state.allocator.set_current_frame_index(state.frame_index);
    state.frame_active = true;
}

<* Lock-free CAS bump: concurrent callers get disjoint, aligned spans. *>
fn gpu::GpuSpan? vk_alloc_frame_span(gpu::Device* device, usz size, usz align) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    FrameArenaState* slot = &state.frame_arenas[state.frame_index];
    if (!gpu::is_pow2(align)) return gpu::INVALID_ARGUMENT~;
    if (size == 0) return gpu::INVALID_ARGUMENT~;

    usz old = slot.cursor.load(AtomicOrdering.RELAXED);
    while (true) {
        usz aligned = gpu::align_up(old, align);
        if (aligned + size > slot.size) return gpu::ARENA_FULL~;
        usz seen = mem::compare_exchange(&slot.cursor.data, old, aligned + size, AtomicOrdering.SEQ_CONSISTENT, AtomicOrdering.RELAXED);
        if (seen == old) {
            return {
                .gpu    = (gpu::GpuAddress)((ulong)slot.gpu_base + aligned),
                .cpu    = slot.cpu_base == null ? null : (char*)slot.cpu_base + aligned,
                .size   = size,
                .buffer = slot.backing_buffer,
                .offset = aligned,
                .kind   = gpu::MemoryKind.FRAME_UPLOAD,
            };
        }
        old = seen;
    }
}

<* Queue-ordered empty submit signaling `value` on `semaphore` after all
   prior work on `queue` completes. *>
fn void? submit_timeline_signal(VkDeviceState* state, vk::Semaphore semaphore, vk::Queue queue, Mutex* mutex, ulong value, vk::SemaphoreSubmitInfo[] wait_infos = {}) @private {
    vk::SemaphoreSubmitInfo signal = vk::semaphore_submit_info()
        .set_semaphore(semaphore)
        .set_value(value)
        .set_stage_mask(vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT);
    vk::SubmitInfo2 submit = vk::submit_info2().set_signal_semaphore_infos((&signal)[:1]);
    if (wait_infos.len > 0) submit = submit.set_wait_semaphore_infos(wait_infos);
    mutex.lock()!!;
    defer mutex.unlock();
    return check(vk::queue_submit2(queue, 1, &submit, (vk::Fence)null));
}

<* The frame fence is a chain of queue-ordered empty submits: compute/
   transfer queues used since the last boundary (in-frame or off) signal aux
   values the graphics signal waits on, so the frame value implies every
   queue finished. Off-frame graphics submits need no flag — this signal is
   queue-ordered behind them. *>
fn void? vk_end_frame(gpu::Device* device) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    if (boundary_blocked(state)) return gpu::INVALID_RESOURCE_STATE~;
    FrameArenaState* slot = &state.frame_arenas[state.frame_index];

    // frame_timeline receives exactly one signal per frame (counter + 1),
    // preserving retire_timeline_value's mid-frame contract; the cross-queue
    // chain runs on aux_timeline.
    ulong frame_value = state.timeline_counter.add(1, AtomicOrdering.RELAXED) + 1;
    slot.frame_timeline_value = frame_value;

    vk::SemaphoreSubmitInfo[2] waits;
    uint wait_count = 0;
    ulong aux_value = frame_value * 2;
    if (state.compute_used_since_frame) {
        submit_timeline_signal(
            state:     state,
            semaphore: state.aux_timeline,
            queue:     state.compute_queue,
            mutex:     &state.compute_queue_mutex,
            value:     aux_value,
        )!;
        waits[wait_count++] = vk::semaphore_submit_info()
            .set_semaphore(state.aux_timeline)
            .set_value(aux_value)
            .set_stage_mask(vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT);
        aux_value++;
    }
    if (state.transfer_used_since_frame) {
        Mutex* transfer_mutex = state.transfer_queue == state.graphics_queue ? &state.graphics_queue_mutex : &state.transfer_queue_mutex;
        submit_timeline_signal(
            state:     state,
            semaphore: state.aux_timeline,
            queue:     state.transfer_queue,
            mutex:     transfer_mutex,
            value:     aux_value,
        )!;
        waits[wait_count++] = vk::semaphore_submit_info()
            .set_semaphore(state.aux_timeline)
            .set_value(aux_value)
            .set_stage_mask(vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT);
        aux_value++;
    }
    state.compute_used_since_frame = false;
    state.transfer_used_since_frame = false;
    state.off_frame_pending.store(false, AtomicOrdering.RELAXED);

    submit_timeline_signal(
        state:      state,
        semaphore:  state.frame_timeline,
        queue:      state.graphics_queue,
        mutex:      &state.graphics_queue_mutex,
        value:      frame_value,
        wait_infos: waits[:wait_count],
    )!;
    state.frame_active = false;
}

<* Create the persistent arena: a large addressable buffer plus a VMA virtual
   block that suballocates it. *>
fn void? create_persistent_arena(VkDeviceState* state, gpu::DeviceDesc* device_desc) {
    usz arena_size = device_desc.persistent_arena_size == 0 ? gpu::DEFAULT_PERSISTENT_ARENA_SIZE : device_desc.persistent_arena_size;
    gpu::Device handle = { .backend = gpu::BackendKind.VULKAN, .vtable = &VK_VTABLE, .backend_state = (void*)state };
    gpu::BufferDesc desc = {
        .size        = arena_size,
        .usage       = PERSISTENT_ARENA_USAGE,
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .debug_name  = "persistent_arena",
    };
    gpu::BufferHandle buffer = vk_create_buffer(&handle, &desc)!;
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
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if (desc.memory_kind != gpu::MemoryKind.PERSISTENT_UPLOAD) return gpu::INVALID_ARGUMENT~;
    if (desc.size == 0) return gpu::INVALID_ARGUMENT~;
    // align 0 = no requirement (documented); anything else must be a power
    // of two (VMA virtual-alloc requirement).
    if (desc.align != 0 && !gpu::is_pow2(desc.align)) return gpu::INVALID_ARGUMENT~;
    uint unsupported_usage = (uint)desc.usage & ~(uint)PERSISTENT_ARENA_USAGE;
    if (unsupported_usage != 0) return gpu::INVALID_ARGUMENT~;
    PersistentArenaState* arena = &state.persistent_arena;

    vma::VirtualAllocationCreateInfo create_info = {
        .size      = (vk::DeviceSize)desc.size,
        .alignment = (vk::DeviceSize)desc.align,
    };
    vma::VirtualAllocation allocation;
    vk::DeviceSize offset;
    check(arena.virtual_block.allocate(&create_info, &allocation, &offset))!;

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
    if ((ulong)span.buffer != (ulong)arena.backing_buffer) return gpu::INVALID_ARGUMENT~;

    if (try allocation = arena.live.get(span.offset)) {
        arena.virtual_block.free(allocation);
        arena.live.remove(span.offset)!;
        return;
    }
    return gpu::INVALID_HANDLE~;
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
