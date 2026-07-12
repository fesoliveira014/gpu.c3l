module gpu_test;

import gpu;
import gpu::vk @public;
import vk;
import std::atomic;

// Pure range-boundary coverage plus headless (lavapipe) frame upload arena
// bump allocation, reset, ARENA_FULL, and timeline-gated reset.

struct FrameDiagnosticCapture {
    uint              calls;
    gpu::DebugMessage last;
    char[48]          operation;
    char[64]          rejected_field;
    char[128]         invariant;
}

fn void copy_frame_diagnostic_text(char* destination, usz capacity, ZString source) {
    usz index = 0;
    if (source != null) {
        while (index + 1 < capacity && source[index] != 0) {
            destination[index] = source[index];
            index++;
        }
    }
    destination[index] = 0;
}

fn void capture_frame_diagnostic(gpu::DebugMessage* message, void* user_data) {
    FrameDiagnosticCapture* capture = (FrameDiagnosticCapture*)user_data;
    capture.calls++;
    capture.last = *message;
    copy_frame_diagnostic_text(&capture.operation[0], 48, message.operation);
    copy_frame_diagnostic_text(&capture.rejected_field[0], 64, message.rejected_field);
    copy_frame_diagnostic_text(&capture.invariant[0], 128, message.invariant);
}

fn gpu::Device make_diagnostic_frame_device(FrameDiagnosticCapture* capture, uint frames_in_flight = 1) {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = frames_in_flight,
        .debug_callback    = &capture_frame_diagnostic,
        .debug_user_data   = capture,
        .application_name  = "gpu_frame_diagnostic_test",
    };
    return gpu::create_device(&desc)!!;
}

fn void assert_frame_diagnostic(
    FrameDiagnosticCapture* capture,
    uint before,
    fault returned,
    ZString operation,
    bool backend,
    ZString field,
) {
    assert(capture.calls == before + 1, "frame rejection emits exactly once");
    assert(capture.last.has_fault && capture.last.public_fault == returned,
        "frame diagnostic preserves the returned fault");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.FRAME,
        "frame diagnostic identifies the frame resource kind");
    assert(!capture.last.resource.has_public_identity,
        "frame diagnostic invents no slot identity for a frame token");
    assert(backend ? capture.last.category.backend : capture.last.category.public_contract,
        "frame diagnostic category is exact");
    assert(((ZString)&capture.operation[0]).str_view() == operation.str_view(),
        "frame diagnostic operation is exact");
    if (field != null) {
        assert(((ZString)&capture.rejected_field[0]).str_view() == field.str_view(),
            "frame diagnostic rejected field is exact");
    }
}

fn vk::Result reject_frame_counter_query(vk::Device device, vk::Semaphore semaphore, ulong* value) {
    return vk::Result.ERROR_DEVICE_LOST;
}

fn void frame_arena_range_preserves_fault_taxonomy() @test {
    fault zero_size = @catch(gpu::vk::frame_arena_range(
        cursor: 0,
        capacity: 256,
        size: 0,
        align: 16,
    ));
    fault zero_align = @catch(gpu::vk::frame_arena_range(
        cursor: 0,
        capacity: 256,
        size: 1,
        align: 0,
    ));
    fault bad_align = @catch(gpu::vk::frame_arena_range(
        cursor: 0,
        capacity: 256,
        size: 1,
        align: 3,
    ));
    fault huge_size = @catch(gpu::vk::frame_arena_range(
        cursor: 1,
        capacity: 256,
        size: usz::max,
        align: 1,
    ));

    fault cursor_past_capacity = @catch(gpu::vk::frame_arena_range(
        cursor: 257,
        capacity: 256,
        size: 1,
        align: 1,
    ));
    assert(zero_size == gpu::INVALID_ARGUMENT, "zero size remains invalid");
    assert(zero_align == gpu::INVALID_ARGUMENT, "zero alignment remains invalid");
    assert(bad_align == gpu::INVALID_ARGUMENT, "non-power-of-two alignment remains invalid");
    assert(huge_size == gpu::ARENA_FULL, "valid huge size exhausts the arena");
    assert(cursor_past_capacity == gpu::ARENA_FULL, "cursor past capacity faults ARENA_FULL");
}

fn void frame_arena_range_handles_exact_end() @test {
    gpu::vk::FrameArenaRange exact = gpu::vk::frame_arena_range(
        cursor: 241,
        capacity: 272,
        size: 16,
        align: 16,
    )!!;
    assert(exact.offset == 256, "padding resolves to the next aligned offset");
    assert(exact.next_cursor == 272, "exact fit reaches the arena end");

    fault one_over = @catch(gpu::vk::frame_arena_range(
        cursor: 241,
        capacity: 271,
        size: 16,
        align: 16,
    ));
    fault at_end = @catch(gpu::vk::frame_arena_range(
        cursor: 256,
        capacity: 256,
        size: 1,
        align: 1,
    ));
    assert(one_over == gpu::ARENA_FULL, "one-byte overflow faults ARENA_FULL");
    assert(at_end == gpu::ARENA_FULL, "nonzero allocation at arena end faults ARENA_FULL");
}

fn void frame_arena_range_handles_max_boundaries() @test {
    gpu::vk::FrameArenaRange high_exact = gpu::vk::frame_arena_range(
        cursor: usz::max - 1,
        capacity: usz::max,
        size: 1,
        align: 2,
    )!!;
    assert(high_exact.offset == usz::max - 1, "already-aligned high cursor is preserved");
    assert(high_exact.next_cursor == usz::max, "representable exact fit reaches usz::max");

    gpu::vk::FrameArenaRange full_exact = gpu::vk::frame_arena_range(
        cursor: 0,
        capacity: usz::max,
        size: usz::max,
        align: 1,
    )!!;
    assert(full_exact.offset == 0 && full_exact.next_cursor == usz::max, "maximum capacity exact fit remains valid");

    fault align_overflow = @catch(gpu::vk::frame_arena_range(
        cursor: usz::max,
        capacity: usz::max,
        size: 1,
        align: 2,
    ));
    fault extent_overflow = @catch(gpu::vk::frame_arena_range(
        cursor: usz::max - 1,
        capacity: usz::max,
        size: 2,
        align: 2,
    ));
    assert(align_overflow == gpu::ARENA_FULL, "unrepresentable alignment faults ARENA_FULL");
    assert(extent_overflow == gpu::ARENA_FULL, "unrepresentable extent faults ARENA_FULL");
}

struct FrameLifecycleSnapshot {
    uint  frame_index;
    ulong frame_generation;
    ulong active_frame_generation;
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
        .frame_generation           = state.frame_generation,
        .active_frame_generation    = state.active_frame_generation,
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
    assert(after.frame_generation == before.frame_generation, "rejection preserves issued generation");
    assert(after.active_frame_generation == before.active_frame_generation, "rejection preserves active generation");
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

    gpu::FrameToken frame;
    FrameLifecycleSnapshot before_begin = frame_lifecycle_snapshot(state);
    fault before_fault = @catch(gpu::alloc_frame_span(&frame, 64, 16));
    assert(before_fault == gpu::INVALID_HANDLE, "zero token faults INVALID_HANDLE");
    assert_frame_lifecycle_unchanged(before_begin, frame_lifecycle_snapshot(state));

    frame = gpu::begin_frame(&device)!!;
    gpu::FrameToken copy = frame;
    gpu::GpuSpan span = gpu::alloc_frame_span(&copy, 64, 16)!!;
    assert(span.offset == 0, "active-frame allocation still succeeds");
    gpu::end_frame(&frame)!!;
    gpu::FrameToken malformed = { .device = &device, .generation = frame.generation + 1 };
    FrameLifecycleSnapshot before_malformed = frame_lifecycle_snapshot(state);
    fault malformed_alloc = @catch(gpu::alloc_frame_span(&malformed, 64, 16));
    fault malformed_end = @catch(gpu::end_frame(&malformed));
    assert(malformed_alloc == gpu::INVALID_HANDLE, "wrong generation cannot allocate");
    assert(malformed_end == gpu::INVALID_HANDLE, "wrong generation cannot end");
    assert_frame_lifecycle_unchanged(before_malformed, frame_lifecycle_snapshot(state));


    FrameLifecycleSnapshot after_end = frame_lifecycle_snapshot(state);
    fault consumed_fault = @catch(gpu::alloc_frame_span(&frame, 64, 16));
    fault stale_fault = @catch(gpu::alloc_frame_span(&copy, 64, 16));
    assert(consumed_fault == gpu::INVALID_HANDLE, "consumed token faults INVALID_HANDLE");
    fault stale_end = @catch(gpu::end_frame(&copy));
    assert(stale_fault == gpu::INVALID_HANDLE, "stale alias faults INVALID_HANDLE");
    assert_frame_lifecycle_unchanged(after_end, frame_lifecycle_snapshot(state));
    assert(stale_end == gpu::INVALID_HANDLE, "stale alias cannot end the next lifecycle");

    frame = gpu::begin_frame(&device)!!;
    FrameLifecycleSnapshot next_active = frame_lifecycle_snapshot(state);
    fault next_alloc = @catch(gpu::alloc_frame_span(&copy, 64, 16));
    fault next_end = @catch(gpu::end_frame(&copy));
    assert(next_alloc == gpu::INVALID_HANDLE, "stale alias cannot allocate in a later active frame");
    assert(next_end == gpu::INVALID_HANDLE, "stale alias cannot end a later active frame");
    assert_frame_lifecycle_unchanged(next_active, frame_lifecycle_snapshot(state));
    gpu::end_frame(&frame)!!;
}

fn void frame_generation_wraps_without_revalidating_stale_token() @test {
    gpu::Device device = make_frame_lifecycle_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    state.frame_generation = ulong::max;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    assert(frame.generation == 1, "frame generation wraps to the first valid value");
    assert(state.active_frame_generation == 1, "wrapped generation becomes active");

    gpu::FrameToken stale = { .device = &device, .generation = ulong::max };
    FrameLifecycleSnapshot before = frame_lifecycle_snapshot(state);
    fault alloc_fault = @catch(gpu::alloc_frame_span(&stale, 64, 16));
    fault end_fault = @catch(gpu::end_frame(&stale));
    assert(alloc_fault == gpu::INVALID_HANDLE, "pre-wrap token cannot allocate in the wrapped frame");
    assert(end_fault == gpu::INVALID_HANDLE, "pre-wrap token cannot end the wrapped frame");
    assert_frame_lifecycle_unchanged(before, frame_lifecycle_snapshot(state));

    gpu::end_frame(&frame)!!;
}

fn void end_before_begin_is_atomic() @test {
    gpu::Device device = make_frame_lifecycle_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    state.compute_used_since_frame = true;
    state.transfer_used_since_frame = true;
    state.off_frame_pending.store(true, AtomicOrdering.RELAXED);

    gpu::FrameToken frame;
    FrameLifecycleSnapshot before = frame_lifecycle_snapshot(state);
    fault rejected = @catch(gpu::end_frame(&frame));
    assert(rejected == gpu::INVALID_HANDLE, "end without a token faults INVALID_HANDLE");
    assert_frame_lifecycle_unchanged(before, frame_lifecycle_snapshot(state));

    frame = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame)!!;
}

fn void double_end_is_atomic() @test {
    gpu::Device device = make_frame_lifecycle_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame)!!;
    FrameLifecycleSnapshot before = frame_lifecycle_snapshot(state);
    fault rejected = @catch(gpu::end_frame(&frame));
    assert(rejected == gpu::INVALID_HANDLE, "double end faults INVALID_HANDLE");
    assert_frame_lifecycle_unchanged(before, frame_lifecycle_snapshot(state));

    frame = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame)!!;
}

fn void double_begin_single_slot_is_atomic() @test {
    gpu::Device device = make_frame_lifecycle_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    (void)gpu::alloc_frame_span(&frame, 64, 16)!!;
    vk::RecordingContextState* context = state.recording_contexts.get(state.default_context)!!;
    context.pool_dirty[state.frame_index].graphics = true;

    FrameLifecycleSnapshot before = frame_lifecycle_snapshot(state);
    fault rejected = @catch(gpu::begin_frame(&device));
    assert(rejected == gpu::INVALID_RESOURCE_STATE, "double begin faults INVALID_RESOURCE_STATE");
    assert_frame_lifecycle_unchanged(before, frame_lifecycle_snapshot(state));

    context.pool_dirty[state.frame_index].graphics = false;
    gpu::end_frame(&frame)!!;
}

fn void double_begin_default_slots_is_atomic() @test {
    gpu::Device device = make_frame_lifecycle_device(2);
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    (void)gpu::alloc_frame_span(&frame, 64, 16)!!;

    FrameLifecycleSnapshot before = frame_lifecycle_snapshot(state);
    fault rejected = @catch(gpu::begin_frame(&device));
    assert(rejected == gpu::INVALID_RESOURCE_STATE,
        "double begin faults independently of frame-slot retirement");
    assert_frame_lifecycle_unchanged(before, frame_lifecycle_snapshot(state));

    gpu::end_frame(&frame)!!;
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

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;

    gpu::GpuSpan a = gpu::alloc_frame_span(&frame, 100, 16)!!;
    assert(a.cpu != null, "arena span is mapped");
    assert(a.offset == 0, "first allocation offset is 0");
    assert((ulong)a.gpu != 0, "arena GPU address is non-zero");

    ulong base_gpu = (ulong)a.gpu;
    char* base_cpu = (char*)a.cpu;

    gpu::GpuSpan b = gpu::alloc_frame_span(&frame, 50, 64)!!;
    assert(b.offset == 128, "second allocation aligns up to 64 (align_up(100,64)=128)");
    assert((ulong)b.gpu == base_gpu + 128, "GPU address is base + aligned offset");
    assert((char*)b.cpu == base_cpu + 128, "CPU pointer is base + aligned offset");

    fault full = @catch(gpu::alloc_frame_span(&frame, gpu::DEFAULT_FRAME_ARENA_SIZE, 16));
    assert(full == gpu::ARENA_FULL, "over-allocation returns ARENA_FULL");

    gpu::end_frame(&frame)!!;

    // Next frame slot: cursor reset to 0.
    frame = gpu::begin_frame(&device)!!;
    gpu::GpuSpan c = gpu::alloc_frame_span(&frame, 10, 16)!!;
    assert(c.offset == 0, "new frame slot starts at offset 0");
    gpu::end_frame(&frame)!!;

    // Wrap back to the first slot (waits on its host-signaled timeline value).
    frame = gpu::begin_frame(&device)!!;
    gpu::GpuSpan d = gpu::alloc_frame_span(&frame, 10, 16)!!;
    assert(d.offset == 0, "wrapped slot cursor is reset");
    gpu::end_frame(&frame)!!;
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

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    // Well past the old 1 MiB default and the 16 MiB current default.
    gpu::GpuSpan span = gpu::alloc_frame_span(&frame, 20 << 20, 16)!!;
    assert(span.size == 20 << 20, "oversized span allocates in a custom-sized arena");
    gpu::end_frame(&frame)!!;
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

fn void frame_contract_rejections_report_exact_context() @test {
    FrameDiagnosticCapture capture;
    gpu::Device device = make_diagnostic_frame_device(&capture);
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    gpu::FrameToken frame = gpu::begin_frame(&device)!!;

    uint before = capture.calls;
    fault rejected = @catch(gpu::begin_frame(&device));
    assert(rejected == gpu::INVALID_RESOURCE_STATE, "double begin keeps its fault");
    assert_frame_diagnostic(
        capture:   &capture,
        before:    before,
        returned:  rejected,
        operation: "begin_frame",
        backend:   false,
        field:     "Device.frame_state",
    );

    before = capture.calls;
    rejected = @catch(gpu::alloc_frame_span(&frame, 0, 0));
    assert(rejected == gpu::INVALID_ARGUMENT, "ambiguous allocation keeps INVALID_ARGUMENT");
    assert_frame_diagnostic(
        capture:   &capture,
        before:    before,
        returned:  rejected,
        operation: "alloc_frame_span",
        backend:   false,
        field:     "align",
    );

    before = capture.calls;
    rejected = @catch(gpu::alloc_frame_span(&frame, 0, 16));
    assert_frame_diagnostic(
        capture:   &capture,
        before:    before,
        returned:  rejected,
        operation: "alloc_frame_span",
        backend:   false,
        field:     "size",
    );

    before = capture.calls;
    rejected = @catch(gpu::alloc_frame_span(&frame, gpu::DEFAULT_FRAME_ARENA_SIZE + 1, 16));
    assert(rejected == gpu::ARENA_FULL, "frame exhaustion keeps ARENA_FULL");
    assert_frame_diagnostic(
        capture:   &capture,
        before:    before,
        returned:  rejected,
        operation: "alloc_frame_span",
        backend:   false,
        field:     "size",
    );

    gpu::FrameToken stale = frame;
    gpu::end_frame(&frame)!!;
    before = capture.calls;
    rejected = @catch(gpu::alloc_frame_span(&stale, 16, 16));
    assert(rejected == gpu::INVALID_HANDLE, "stale allocation keeps INVALID_HANDLE");
    assert_frame_diagnostic(
        capture:   &capture,
        before:    before,
        returned:  rejected,
        operation: "alloc_frame_span",
        backend:   false,
        field:     "FrameToken.generation",
    );

    frame = gpu::begin_frame(&device)!!;
    state.inflight_calls.add(1, AtomicOrdering.RELAXED);
    before = capture.calls;
    rejected = @catch(gpu::end_frame(&frame));
    assert(rejected == gpu::INVALID_RESOURCE_STATE, "blocked end keeps INVALID_RESOURCE_STATE");
    assert_frame_diagnostic(
        capture:   &capture,
        before:    before,
        returned:  rejected,
        operation: "end_frame",
        backend:   false,
        field:     "Device.frame_state",
    );
    assert(frame.device == &device && state.frame_active, "blocked end preserves the live frame");
    state.inflight_calls.sub(1, AtomicOrdering.RELAXED);
    gpu::end_frame(&frame)!!;

    state.inflight_calls.add(1, AtomicOrdering.RELAXED);
    before = capture.calls;
    rejected = @catch(gpu::begin_frame(&device));
    assert_frame_diagnostic(
        capture:   &capture,
        before:    before,
        returned:  rejected,
        operation: "begin_frame",
        backend:   false,
        field:     "Device.frame_state",
    );
    state.inflight_calls.sub(1, AtomicOrdering.RELAXED);
}

fn void frame_backend_result_helpers_report_without_fault_drift() @test {
    FrameDiagnosticCapture capture;
    gpu::Device device = make_diagnostic_frame_device(&capture);
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    uint before = capture.calls;
    fault query_fault = @catch(vk::frame_timeline_value_with_query(
        state:     state,
        operation: "begin_frame",
        query_fn:  &reject_frame_counter_query,
    ));
    assert(query_fault == gpu::DEVICE_LOST, "counter query keeps DEVICE_LOST");
    assert(capture.calls == before + 1, "counter query emits exactly once");
    assert(capture.last.category.backend, "counter query is a backend diagnostic");
    assert(((ZString)&capture.operation[0]).str_view() == "begin_frame", "counter query names begin_frame");

    before = capture.calls;
    fault timeout = @catch(vk::check_wait_backend_result(
        state:     state,
        operation: "begin_frame",
        result:    vk::Result.TIMEOUT,
    ));
    assert(timeout == gpu::WAIT_TIMEOUT, "operation-aware wait keeps WAIT_TIMEOUT");
    assert(capture.calls == before + 1, "wait timeout emits exactly once");
    assert(capture.last.validation_id_number == (int)vk::Result.TIMEOUT,
        "wait timeout carries the raw backend result");
    before = capture.calls;
    fault mismatch = @catch(vk::report_unretired_frame(state));
    assert(mismatch == gpu::INVALID_RESOURCE_STATE,
        "successful-wait retirement mismatch keeps INVALID_RESOURCE_STATE");
    assert(capture.calls == before + 1, "retirement mismatch emits exactly once");
    assert(capture.last.category.backend, "retirement mismatch is backend/internal");
    assert(capture.last.resource.kind == gpu::DebugResourceKind.FRAME,
        "retirement mismatch identifies frame context");
    assert(!capture.last.resource.has_public_identity,
        "retirement mismatch invents no public identity");
    assert(((ZString)&capture.operation[0]).str_view() == "begin_frame",
        "retirement mismatch names begin_frame");
    assert(((ZString)&capture.rejected_field[0]).str_view() == "Device.frame_state",
        "retirement mismatch identifies frame state");
}

fn void failed_frame_retirement_query_preserves_begin_state() @test {
    FrameDiagnosticCapture capture;
    gpu::Device device = make_diagnostic_frame_device(&capture, 2);
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    FrameLifecycleSnapshot before_state = frame_lifecycle_snapshot(state);
    assert(state.frames_in_flight == 2 && state.frame_index == 0,
        "retirement rollback test starts before a distinct next frame slot");
    gpu::FrameToken frame;
    uint before_calls = capture.calls;

    fault query_fault = @catch(vk::begin_frame_with_counter_query(
        device:    &device,
        out_frame: &frame,
        query_fn:  &reject_frame_counter_query,
    ));

    assert(query_fault == gpu::DEVICE_LOST, "begin retirement query keeps DEVICE_LOST");
    assert(capture.calls == before_calls + 1, "begin retirement query emits exactly once");
    assert(capture.last.category.backend, "begin retirement query is a backend diagnostic");
    assert(((ZString)&capture.operation[0]).str_view() == "begin_frame",
        "begin retirement query names begin_frame");
    assert(frame.device == null && frame.generation == 0,
        "failed retirement query issues no token");
    assert_frame_lifecycle_unchanged(before_state, frame_lifecycle_snapshot(state));
}
