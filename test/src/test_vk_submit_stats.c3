module gpu_test;

import gpu;
import gpu::vk @public;
import vk;

// Headless (lavapipe): aliased queues mean the piggyback branch never fires
// here; what stays testable is one-empty-submit-per-frame and rejected
// submits recording nothing.

struct SubmitDiagnosticCapture {
    uint              calls;
    gpu::DebugMessage last;
}

fn void capture_submit_diagnostic(gpu::DebugMessage* message, void* user_data) {
    SubmitDiagnosticCapture* capture = (SubmitDiagnosticCapture*)user_data;
    capture.calls++;
    capture.last = *message;
}

fn gpu::Device make_submit_stats_test_device(SubmitDiagnosticCapture* capture = null) {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_submit_stats_test",
    };
    if (capture != null) {
        desc.debug_callback = &capture_submit_diagnostic;
        desc.debug_user_data = capture;
    }
    return gpu::create_device(&desc)!!;
}

fn vk::SubmitStats stats_of(gpu::Device* device) {
    return vk::submit_stats((vk::VkDeviceState*)device.backend_state);
}

<* Any used-queue combination ends the frame with exactly one internal
   empty submit — the graphics frame signal. *>
fn void end_frame_issues_exactly_one_empty_submit() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::QueueKind[3] kinds = { gpu::QueueKind.GRAPHICS, gpu::QueueKind.COMPUTE, gpu::QueueKind.TRANSFER };
    foreach (kind : kinds) {
        gpu::CommandList cmd = gpu::begin_commands(&device, kind)!!;
        gpu::end_commands(&cmd)!!;
        gpu::CommandList[1] one = { cmd };
        gpu::SubmitDesc submit = { .command_lists = one[..] };
        gpu::submit(&device, &submit)!!;
    }

    vk::SubmitStats before = stats_of(&device);
    gpu::end_frame(&frame)!!;
    vk::SubmitStats after = stats_of(&device);
    assert(after.empty_signal_submits - before.empty_signal_submits == 1, "a frame using every queue kind still ends with exactly one internal empty submit");
    assert(after.queue_submits == before.queue_submits, "end_frame's own signal is not counted as a user queue submit");
}

<* Same invariant on a graphics-only frame — the spec's baseline scenario. *>
fn void graphics_only_frame_emits_one_empty_submit() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };

    vk::SubmitStats before_submit = stats_of(&device);
    gpu::submit(&device, &submit)!!;
    vk::SubmitStats after_submit = stats_of(&device);
    assert(after_submit.queue_submits - before_submit.queue_submits == 1, "one user submit records one queue submit");

    vk::SubmitStats before_end = stats_of(&device);
    gpu::end_frame(&frame)!!;
    vk::SubmitStats after_end = stats_of(&device);
    assert(after_end.empty_signal_submits - before_end.empty_signal_submits == 1, "a graphics-only frame issues exactly one internal empty submit");
}

<* A submit rejected before it ever reaches the driver must record nothing,
   and a subsequent end_frame must still complete without waiting on any
   value the rejected submit would have signaled. *>
fn void failed_submit_records_nothing_and_end_frame_stays_clean() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList unfinished = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandList[1] not_executable = { unfinished };
    gpu::SubmitDesc bad_submit = { .command_lists = not_executable[..] };

    vk::SubmitStats before = stats_of(&device);
    fault f = @catch(gpu::submit(&device, &bad_submit));
    assert(f == gpu::COMMAND_RECORDING_ERROR, "a not-yet-executable list rejects before touching the queue");
    vk::SubmitStats after = stats_of(&device);
    assert(after.queue_submits == before.queue_submits, "a rejected submit records no queue submit");

    gpu::end_commands(&unfinished)!!;
    gpu::end_frame(&frame)!!;

    // A clean cycle right after proves end_frame never waited a bogus value.
    frame = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame)!!;
}

ulong rejected_frame_signal_value;
uint rejected_frame_wait_count;
ulong[2] rejected_frame_wait_values;
vk::Semaphore[2] rejected_frame_wait_semaphores;
ulong retried_frame_signal_value;
uint retried_frame_wait_count;
ulong[2] retried_frame_wait_values;
vk::Semaphore[2] retried_frame_wait_semaphores;

fn vk::Result reject_frame_submit(vk::Queue queue, uint submit_count, vk::SubmitInfo2* submits, vk::Fence fence) {
    assert(submit_count == 1, "end_frame injects one submit");
    vk::SubmitInfo2* submit = &submits[0];
    rejected_frame_wait_count = submit.wait_semaphore_info_count;
    assert(rejected_frame_wait_count == 2, "injected submit preserves both cross-queue waits");
    for (uint i = 0; i < rejected_frame_wait_count; i++) {
        rejected_frame_wait_values[i] = submit.p_wait_semaphore_infos[i].value;
        rejected_frame_wait_semaphores[i] = submit.p_wait_semaphore_infos[i].semaphore;
    }
    assert(submit.signal_semaphore_info_count == 1, "end_frame injects one frame signal");
    rejected_frame_signal_value = submit.p_signal_semaphore_infos[0].value;
    return vk::Result.ERROR_OUT_OF_HOST_MEMORY;
}

fn vk::Result capture_retry_frame_submit(vk::Queue queue, uint submit_count, vk::SubmitInfo2* submits, vk::Fence fence) {
    assert(submit_count == 1, "retry injects one submit");
    vk::SubmitInfo2* submit = &submits[0];
    retried_frame_wait_count = submit.wait_semaphore_info_count;
    assert(retried_frame_wait_count == 2, "retry preserves both cross-queue waits");
    // Zero the waits after capture: the fabricated aux values are never signaled and would hang teardown.
    for (uint i = 0; i < retried_frame_wait_count; i++) {
        retried_frame_wait_values[i] = submit.p_wait_semaphore_infos[i].value;
        retried_frame_wait_semaphores[i] = submit.p_wait_semaphore_infos[i].semaphore;
        submit.p_wait_semaphore_infos[i].value = 0;
    }
    assert(submit.signal_semaphore_info_count == 1, "retry injects one frame signal");
    retried_frame_signal_value = submit.p_signal_semaphore_infos[0].value;
    return vk::queue_submit2(queue, submit_count, submits, fence);
}

fn void failed_end_frame_submit_preserves_retry_state() @test {
    SubmitDiagnosticCapture capture;
    gpu::Device device = make_submit_stats_test_device(&capture);
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    state.compute_used_since_frame = true;
    state.transfer_used_since_frame = true;
    state.off_frame_pending.store(true, AtomicOrdering.RELAXED);
    state.aux_compute_counter = 7;
    state.aux_compute_last_signaled = 7;
    state.aux_transfer_counter = 11;
    state.aux_transfer_last_signaled = 11;

    ulong counter_before = state.timeline_counter.load(AtomicOrdering.RELAXED);
    uint frame_index_before = state.frame_index;
    ulong slot_before = state.frame_arenas[frame_index_before].frame_timeline_value;
    ulong compute_counter_before = state.aux_compute_counter;
    ulong compute_signaled_before = state.aux_compute_last_signaled;
    ulong transfer_counter_before = state.aux_transfer_counter;
    ulong transfer_signaled_before = state.aux_transfer_last_signaled;
    ulong shadow_before = state.signaled_shadow;
    vk::SubmitStats stats_before = stats_of(&device);

    uint diagnostic_before = capture.calls;
    fault submit_fault = @catch(vk::end_frame_with_submit(&frame, &reject_frame_submit));
    assert(submit_fault == gpu::OUT_OF_HOST_MEMORY, "injected frame submit faults OUT_OF_HOST_MEMORY");
    assert(rejected_frame_signal_value == counter_before + 1, "failed submit uses the prospective frame value");
    assert(rejected_frame_wait_count == 2, "failed submit preserves both cross-queue waits");
    assert(capture.calls == diagnostic_before + 1, "failed frame signal emits exactly once");
    assert(capture.last.category.backend, "failed frame signal is a backend diagnostic");
    assert(capture.last.has_fault && capture.last.public_fault == submit_fault,
        "failed frame signal diagnostic preserves the returned fault");
    assert(capture.last.operation.str_view() == "end_frame", "failed frame signal names end_frame");
    assert(rejected_frame_wait_values[0] == compute_signaled_before, "compute wait uses last signaled value");
    assert(rejected_frame_wait_semaphores[0] == state.aux_compute_timeline, "compute wait uses the compute timeline");
    assert(rejected_frame_wait_semaphores[1] == state.aux_transfer_timeline, "transfer wait uses the transfer timeline");
    assert(rejected_frame_wait_values[1] == transfer_signaled_before, "transfer wait uses last signaled value");

    assert(state.timeline_counter.load(AtomicOrdering.RELAXED) == counter_before, "failed submit does not advance counter");
    assert(state.frame_index == frame_index_before, "failed submit does not change frame index");
    assert(state.frame_arenas[frame_index_before].frame_timeline_value == slot_before, "failed submit does not retire slot");
    assert(state.compute_used_since_frame && state.transfer_used_since_frame, "failed submit preserves queue-use flags");
    assert(state.off_frame_pending.load(AtomicOrdering.RELAXED), "failed submit preserves off-frame marker");
    assert(state.frame_active, "failed submit keeps frame active");
    assert(state.aux_compute_counter == compute_counter_before && state.aux_compute_last_signaled == compute_signaled_before, "failed submit preserves compute timeline state");
    assert(frame.device == &device && frame.generation == state.active_frame_generation, "failed submit retains the live frame token");
    assert(state.aux_transfer_counter == transfer_counter_before && state.aux_transfer_last_signaled == transfer_signaled_before, "failed submit preserves transfer timeline state");
    assert(state.signaled_shadow == shadow_before, "failed submit preserves signaled shadow");
    assert(stats_of(&device).empty_signal_submits == stats_before.empty_signal_submits, "failed submit records no empty signal");
    assert(stats_of(&device).queue_submits == stats_before.queue_submits, "failed frame signal records no user submit");

    vk::end_frame_with_submit(&frame, &capture_retry_frame_submit)!!;
    assert(retried_frame_signal_value == rejected_frame_signal_value, "retry reuses the prospective frame value");
    assert(retried_frame_wait_count == rejected_frame_wait_count, "retry reuses the wait count");
    assert(retried_frame_wait_values == rejected_frame_wait_values, "retry reuses the nonzero wait values");
    assert(retried_frame_wait_semaphores == rejected_frame_wait_semaphores, "retry reuses the wait semaphores");
    assert(state.timeline_counter.load(AtomicOrdering.RELAXED) == counter_before + 1, "retry commits one counter step");
    assert(state.frame_arenas[frame_index_before].frame_timeline_value == counter_before + 1, "retry commits slot retirement");
    assert(!state.compute_used_since_frame && !state.transfer_used_since_frame, "retry clears queue-use flags");
    assert(!state.off_frame_pending.load(AtomicOrdering.RELAXED), "retry clears off-frame marker");
    assert(!state.frame_active, "retry ends the frame");
    assert(frame.device == null && frame.generation == 0, "successful retry consumes the token");
    assert(stats_of(&device).empty_signal_submits == stats_before.empty_signal_submits + 1, "retry records one empty signal");
}

fn void post_claim_fault_restores_executable_token() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::CommandList cmd = gpu::begin_commands(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::end_commands(&cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SemaphoreSignal[1] bad_signals = {{
        .semaphore = gpu::SEMAPHORE_HANDLE_INVALID,
        .value = (gpu::SemaphoreValue)1,
        .stage = gpu::Stage.TRANSFER,
    }};
    gpu::SubmitDesc bad = {
        .command_lists = lists[..],
        .signals = bad_signals[..],
    };
    ulong before = stats_of(&device).queue_submits;
    assert(@catch(gpu::submit(&device, &bad)) == gpu::INVALID_HANDLE,
        "post-claim semaphore validation faults INVALID_HANDLE");
    assert(stats_of(&device).queue_submits == before,
        "post-claim rejection reaches no Vulkan queue submit");

    gpu::SubmitDesc good = { .command_lists = lists[..] };
    gpu::submit(&device, &good)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
}
