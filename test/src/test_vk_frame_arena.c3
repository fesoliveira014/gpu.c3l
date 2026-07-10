module gpu_test;

import gpu;
import gpu::vk;
import vk;
import std::atomic;

// Headless (lavapipe): frame upload arena bump allocation, reset across the
// frame loop, ARENA_FULL, and the timeline-gated reset.

struct FrameLifecycleSnapshot {
    uint  frame_index;
    usz   cursor;
    ulong slot_timeline_value;
    ulong timeline_counter;
    ulong signaled_shadow;
    ulong aux_compute_counter;
    ulong aux_compute_last_signaled;
    ulong aux_transfer_counter;
    ulong aux_transfer_last_signaled;
    ulong queue_submits;
    ulong empty_signal_submits;
    uint  pools_reset;
    uint  pools_skipped;
    bool  frame_active;
    bool  compute_used;
    bool  transfer_used;
    bool  off_frame_pending;
    bool  graphics_pool_dirty;
    bool  compute_pool_dirty;
    bool  transfer_pool_dirty;
}

fn gpu::Device make_frame_lifecycle_device(uint frames_in_flight = 1) {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = frames_in_flight,
        .application_name  = "gpu_frame_lifecycle_test",
    };
    return gpu::create_device(&desc)!!;
}

fn FrameLifecycleSnapshot frame_lifecycle_snapshot(vk::VkDeviceState* state) {
    vk::FrameArenaState* arena = &state.frame_arenas[state.frame_index];
    vk::RecordingContextState* context = state.recording_contexts.get(state.default_context)!!;
    vk::PoolDirtyMask dirty = context.pool_dirty[state.frame_index];
    vk::PoolResetStats pool_stats = vk::pool_reset_stats(state);
    vk::SubmitStats submit_stats = vk::submit_stats(state);
    return {
        .frame_index                = state.frame_index,
        .cursor                     = arena.cursor.load(AtomicOrdering.RELAXED),
        .slot_timeline_value        = arena.frame_timeline_value,
        .timeline_counter           = state.timeline_counter.load(AtomicOrdering.RELAXED),
        .signaled_shadow            = state.signaled_shadow,
        .aux_compute_counter        = state.aux_compute_counter,
        .aux_compute_last_signaled  = state.aux_compute_last_signaled,
        .aux_transfer_counter       = state.aux_transfer_counter,
        .aux_transfer_last_signaled = state.aux_transfer_last_signaled,
        .queue_submits              = submit_stats.queue_submits,
        .empty_signal_submits       = submit_stats.empty_signal_submits,
        .pools_reset                = pool_stats.reset,
        .pools_skipped              = pool_stats.skipped,
        .frame_active               = state.frame_active,
        .compute_used               = state.compute_used_since_frame,
        .transfer_used              = state.transfer_used_since_frame,
        .off_frame_pending          = state.off_frame_pending.load(AtomicOrdering.RELAXED),
        .graphics_pool_dirty        = dirty.graphics,
        .compute_pool_dirty         = dirty.compute,
        .transfer_pool_dirty        = dirty.transfer,
    };
}

fn void assert_frame_lifecycle_unchanged(FrameLifecycleSnapshot before, FrameLifecycleSnapshot after) {
    assert(after.frame_index == before.frame_index, "rejection preserves frame index");
    assert(after.cursor == before.cursor, "rejection preserves arena cursor");
    assert(after.slot_timeline_value == before.slot_timeline_value, "rejection preserves slot timeline value");
    assert(after.timeline_counter == before.timeline_counter, "rejection preserves frame timeline counter");
    assert(after.signaled_shadow == before.signaled_shadow, "rejection preserves retired timeline shadow");
    assert(after.aux_compute_counter == before.aux_compute_counter, "rejection preserves compute counter");
    assert(after.aux_compute_last_signaled == before.aux_compute_last_signaled, "rejection preserves compute signal");
    assert(after.aux_transfer_counter == before.aux_transfer_counter, "rejection preserves transfer counter");
    assert(after.aux_transfer_last_signaled == before.aux_transfer_last_signaled, "rejection preserves transfer signal");
    assert(after.queue_submits == before.queue_submits, "rejection performs no user queue submit");
    assert(after.empty_signal_submits == before.empty_signal_submits, "rejection performs no frame signal submit");
    assert(after.pools_reset == before.pools_reset, "rejection preserves pool reset accounting");
    assert(after.pools_skipped == before.pools_skipped, "rejection preserves pool skip accounting");
    assert(after.frame_active == before.frame_active, "rejection preserves lifecycle state");
    assert(after.compute_used == before.compute_used, "rejection preserves compute-used state");
    assert(after.transfer_used == before.transfer_used, "rejection preserves transfer-used state");
    assert(after.off_frame_pending == before.off_frame_pending, "rejection preserves off-frame state");
    assert(after.graphics_pool_dirty == before.graphics_pool_dirty, "rejection preserves graphics-pool dirtiness");
    assert(after.compute_pool_dirty == before.compute_pool_dirty, "rejection preserves compute-pool dirtiness");
    assert(after.transfer_pool_dirty == before.transfer_pool_dirty, "rejection preserves transfer-pool dirtiness");
}

fn void frame_span_allocation_requires_active_frame() @test {
    gpu::Device device = make_frame_lifecycle_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    FrameLifecycleSnapshot before_begin = frame_lifecycle_snapshot(state);
    fault before_fault = @catch(gpu::alloc_frame_span(&device, 64, 16));
    assert(before_fault == gpu::INVALID_RESOURCE_STATE, "frame allocation before begin faults INVALID_RESOURCE_STATE");
    assert_frame_lifecycle_unchanged(before_begin, frame_lifecycle_snapshot(state));

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan span = gpu::alloc_frame_span(&device, 64, 16)!!;
    assert(span.offset == 0, "active-frame allocation still succeeds");
    gpu::end_frame(&device)!!;

    FrameLifecycleSnapshot after_end = frame_lifecycle_snapshot(state);
    fault after_fault = @catch(gpu::alloc_frame_span(&device, 64, 16));
    assert(after_fault == gpu::INVALID_RESOURCE_STATE, "frame allocation after end faults INVALID_RESOURCE_STATE");
    assert_frame_lifecycle_unchanged(after_end, frame_lifecycle_snapshot(state));

    gpu::begin_frame(&device)!!;
    gpu::end_frame(&device)!!;
}

fn void end_before_begin_is_atomic() @test {
    gpu::Device device = make_frame_lifecycle_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    state.compute_used_since_frame = true;
    state.transfer_used_since_frame = true;
    state.off_frame_pending.store(true, AtomicOrdering.RELAXED);

    FrameLifecycleSnapshot before = frame_lifecycle_snapshot(state);
    fault rejected = @catch(gpu::end_frame(&device));
    assert(rejected == gpu::INVALID_RESOURCE_STATE, "end before begin faults INVALID_RESOURCE_STATE");
    assert_frame_lifecycle_unchanged(before, frame_lifecycle_snapshot(state));

    gpu::begin_frame(&device)!!;
    gpu::end_frame(&device)!!;
}

fn void double_end_is_atomic() @test {
    gpu::Device device = make_frame_lifecycle_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    gpu::begin_frame(&device)!!;
    gpu::end_frame(&device)!!;
    FrameLifecycleSnapshot before = frame_lifecycle_snapshot(state);
    fault rejected = @catch(gpu::end_frame(&device));
    assert(rejected == gpu::INVALID_RESOURCE_STATE, "double end faults INVALID_RESOURCE_STATE");
    assert_frame_lifecycle_unchanged(before, frame_lifecycle_snapshot(state));

    gpu::begin_frame(&device)!!;
    gpu::end_frame(&device)!!;
}

fn void double_begin_single_slot_is_atomic() @test {
    gpu::Device device = make_frame_lifecycle_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    gpu::begin_frame(&device)!!;
    (void)gpu::alloc_frame_span(&device, 64, 16)!!;
    vk::RecordingContextState* context = state.recording_contexts.get(state.default_context)!!;
    context.pool_dirty[state.frame_index].graphics = true;

    FrameLifecycleSnapshot before = frame_lifecycle_snapshot(state);
    fault rejected = @catch(gpu::begin_frame(&device));
    assert(rejected == gpu::INVALID_RESOURCE_STATE, "double begin faults INVALID_RESOURCE_STATE");
    assert_frame_lifecycle_unchanged(before, frame_lifecycle_snapshot(state));

    context.pool_dirty[state.frame_index].graphics = false;
    gpu::end_frame(&device)!!;
}

fn void double_begin_default_slots_is_atomic() @test {
    gpu::Device device = make_frame_lifecycle_device(2);
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    gpu::begin_frame(&device)!!;
    (void)gpu::alloc_frame_span(&device, 64, 16)!!;

    FrameLifecycleSnapshot before = frame_lifecycle_snapshot(state);
    fault rejected = @catch(gpu::begin_frame(&device));
    assert(rejected == gpu::INVALID_RESOURCE_STATE,
        "double begin faults independently of frame-slot retirement");
    assert_frame_lifecycle_unchanged(before, frame_lifecycle_snapshot(state));

    gpu::end_frame(&device)!!;
}

fn void frame_arena_alloc_and_reset() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_frame_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::begin_frame(&device)!!;

    gpu::GpuSpan a = gpu::alloc_frame_span(&device, 100, 16)!!;
    assert(a.cpu != null, "arena span is mapped");
    assert(a.offset == 0, "first allocation offset is 0");
    assert((ulong)a.gpu != 0, "arena GPU address is non-zero");

    ulong base_gpu = (ulong)a.gpu;
    char* base_cpu = (char*)a.cpu;

    gpu::GpuSpan b = gpu::alloc_frame_span(&device, 50, 64)!!;
    assert(b.offset == 128, "second allocation aligns up to 64 (align_up(100,64)=128)");
    assert((ulong)b.gpu == base_gpu + 128, "GPU address is base + aligned offset");
    assert((char*)b.cpu == base_cpu + 128, "CPU pointer is base + aligned offset");

    fault full = @catch(gpu::alloc_frame_span(&device, gpu::DEFAULT_FRAME_ARENA_SIZE, 16));
    assert(full == gpu::ARENA_FULL, "over-allocation returns ARENA_FULL");

    gpu::end_frame(&device)!!;

    // Next frame slot: cursor reset to 0.
    gpu::begin_frame(&device)!!;
    gpu::GpuSpan c = gpu::alloc_frame_span(&device, 10, 16)!!;
    assert(c.offset == 0, "new frame slot starts at offset 0");
    gpu::end_frame(&device)!!;

    // Wrap back to the first slot (waits on its host-signaled timeline value).
    gpu::begin_frame(&device)!!;
    gpu::GpuSpan d = gpu::alloc_frame_span(&device, 10, 16)!!;
    assert(d.offset == 0, "wrapped slot cursor is reset");
    gpu::end_frame(&device)!!;
}

fn void reset_before_retire_rejected() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)device.backend_state;

    // Slot demands a timeline value the counter has not reached.
    state.frame_arenas[0].frame_timeline_value = 999;
    fault rejected = @catch(gpu::vk::reset_frame_arena(state, 0));
    assert(rejected == gpu::INVALID_RESOURCE_STATE, "premature reset returns INVALID_RESOURCE_STATE");

    // A retired slot resets fine.
    state.frame_arenas[0].frame_timeline_value = 0;
    assert(!@catch(gpu::vk::reset_frame_arena(state, 0)), "retired slot may be reset");
}

fn void custom_frame_arena_size_holds_larger_spans() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .frame_arena_size  = 32 << 20,
        .application_name  = "gpu_frame_arena_size_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::begin_frame(&device)!!;
    // Well past the old 1 MiB default and the 16 MiB current default.
    gpu::GpuSpan span = gpu::alloc_frame_span(&device, 20 << 20, 16)!!;
    assert(span.size == 20 << 20, "oversized span allocates in a custom-sized arena");
    gpu::end_frame(&device)!!;
}

fn void frame_upload_memory_is_coherent() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_coherency_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc buf_desc = {
        .size        = 256,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.FRAME_UPLOAD,
        .debug_name  = "coherency_probe",
    };
    gpu::BufferHandle buf = gpu::create_buffer(&device, &buf_desc)!!;
    defer gpu::destroy_buffer(&device, buf)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    vk::BufferSlot* slot = state.buffers.get(buf)!!;
    vk::MemoryPropertyFlags props = state.allocator.allocation_memory_properties(slot.allocation);
    assert(((uint)props & (uint)vk::MemoryPropertyFlagBits.HOST_COHERENT_BIT) != 0,
        "frame-upload memory is host-coherent");
}
