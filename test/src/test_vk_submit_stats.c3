module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;


struct SubmitDiagnosticCapture {
    uint              calls;
    gpu::DebugMessage last;
}

ulong rejected_frame_signal_value;
uint rejected_frame_wait_count;
ulong[2] rejected_frame_wait_values;
vk::Semaphore[2] rejected_frame_wait_semaphores;
ulong retried_frame_signal_value;
uint retried_frame_wait_count;
ulong[2] retried_frame_wait_values;
vk::Semaphore[2] retried_frame_wait_semaphores;
ulong rejected_user_completion_value;
uint rejected_user_signal_count;
uint counted_user_submit_calls;
uint completion_counter_query_calls;
ulong completion_counter_value;
vk::Result completion_counter_result;

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
    return gpu::create_device_from_desc(&desc)!!;
}

fn vk::SubmitStats stats_of(gpu::Device* device) {
    return vk::submit_stats((vk::VkDeviceState*)gpu::device_backend_state(*device));
}

<* Any used-queue combination ends the frame with exactly one internal
   empty submit — the graphics frame signal. *>
fn void end_frame_issues_exactly_one_empty_submit() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::QueueKind[3] kinds = { gpu::QueueKind.GRAPHICS, gpu::QueueKind.COMPUTE, gpu::QueueKind.TRANSFER };
    foreach (kind : kinds) {
        gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, kind)!!)!!;
        defer discard_test_commands(&cmd)!!;
        gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
        defer discard_test_executable(&cmd_executable)!!;
        gpu::ExecutableCommandList[1] one = { cmd_executable };
        gpu::SubmitDesc submit = { .command_lists = one[..] };
        submit_test_queue(&device, kind, &submit)!!;
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
    defer destroy_test_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };

    vk::SubmitStats before_submit = stats_of(&device);
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
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
    defer destroy_test_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList unfinished = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&unfinished)!!;
    gpu::ExecutableCommandList[1] not_executable = {{
        .device = unfinished.device,
        .handle = unfinished.handle,
    }};
    gpu::SubmitDesc bad_submit = { .command_lists = not_executable[..] };

    vk::SubmitStats before = stats_of(&device);
    fault f = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &bad_submit));
    assert(f == gpu::COMMAND_RECORDING_ERROR, "a not-yet-executable list rejects before touching the queue");
    vk::SubmitStats after = stats_of(&device);
    assert(after.queue_submits == before.queue_submits, "a rejected submit records no queue submit");

    gpu::ExecutableCommandList unfinished_executable = gpu::end_commands(&unfinished)!!;

    defer discard_test_executable(&unfinished_executable)!!;
    gpu::end_frame(&frame)!!;

    // A clean cycle right after proves end_frame never waited a bogus value.
    frame = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame)!!;
}


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
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    state.compute_used_since_frame = true;
    state.transfer_used_since_frame = true;
    state.off_frame_pending_queues.store(6, AtomicOrdering.RELAXED);
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
    assert(state.off_frame_pending_queues.load(AtomicOrdering.RELAXED) != 0, "failed submit preserves off-frame marker");
    assert(state.frame_active, "failed submit keeps frame active");
    assert(state.aux_compute_counter == compute_counter_before && state.aux_compute_last_signaled == compute_signaled_before, "failed submit preserves compute timeline state");
    assert(frame.device == device && frame.generation == state.active_frame_generation, "failed submit retains the live frame token");
    assert(state.aux_transfer_counter == transfer_counter_before && state.aux_transfer_last_signaled == transfer_signaled_before, "failed submit preserves transfer timeline state");
    assert(state.signaled_shadow == shadow_before, "failed submit preserves signaled shadow");
    assert(stats_of(&device).empty_signal_submits == stats_before.empty_signal_submits, "failed submit records no empty signal");
    assert(stats_of(&device).queue_submits == stats_before.queue_submits, "failed frame signal records no user submit");

    state.aux_compute_last_signaled = 0;
    state.aux_transfer_last_signaled = 0;
    gpu::end_frame(&frame)!!;
    assert(state.timeline_counter.load(AtomicOrdering.RELAXED) == counter_before + 1, "retry commits one counter step");
    assert(state.frame_arenas[frame_index_before].frame_timeline_value == counter_before + 1, "retry commits slot retirement");
    assert(!state.compute_used_since_frame && !state.transfer_used_since_frame, "retry clears queue-use flags");
    assert(state.off_frame_pending_queues.load(AtomicOrdering.RELAXED) == 0, "retry clears off-frame marker");
    assert(!state.frame_active, "retry ends the frame");
    assert((ulong)frame.device == 0 && frame.generation == 0, "successful retry consumes the token");
    assert(stats_of(&device).empty_signal_submits == stats_before.empty_signal_submits + 1, "retry records one empty signal");
}

fn vk::Result reject_user_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)fence;
    assert(submit_count == 1, "user submission emits one native submit");
    vk::SubmitInfo2* submit = &submits[0];
    rejected_user_signal_count = submit.signal_semaphore_info_count;
    assert(rejected_user_signal_count > 0,
        "user submission carries a private completion signal");
    rejected_user_completion_value =
        submit.p_signal_semaphore_infos[rejected_user_signal_count - 1].value;
    return vk::Result.ERROR_OUT_OF_HOST_MEMORY;
}

fn vk::Result lose_user_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)submits;
    (void)fence;
    assert(submit_count == 1, "device-loss injection sees one native submit");
    return vk::Result.ERROR_DEVICE_LOST;
}

fn vk::Result reject_counted_user_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)submit_count;
    (void)submits;
    (void)fence;
    counted_user_submit_calls++;
    return vk::Result.ERROR_OUT_OF_HOST_MEMORY;
}

fn vk::Result query_completion_counter(
    vk::Device device,
    vk::Semaphore semaphore,
    ulong* value,
) {
    (void)device;
    (void)semaphore;
    completion_counter_query_calls++;
    *value = completion_counter_value;
    return completion_counter_result;
}

fn void native_submit_failure_rolls_back_completion_and_tokens() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::QueueCompletionState* completion =
        &state.queue_completions[state.graphics_completion_id];

    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    ulong next_before = completion.next_sequence;
    ulong published_before = completion.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );

    assert(@catch(vk::vk_submit_with(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        &desc,
        &reject_user_submit,
    ))
        == gpu::OUT_OF_HOST_MEMORY,
        "native submission failure keeps its public fault");
    assert(completion.next_sequence == next_before,
        "native failure rolls back the reserved queue sequence");
    assert(completion.published_sequence.load(AtomicOrdering.ACQUIRE)
        == published_before,
        "native failure publishes no completion point");
    assert(rejected_user_completion_value == next_before + 1,
        "the rejected native submit used only the prospective sequence");
    assert(lists[0].handle.is_valid(),
        "native failure preserves the executable token for retry");

    gpu::CompletionPoint point = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)!!;
    assert(point.is_valid(), "retry returns a valid completion point");
    assert(completion.next_sequence == next_before + 1,
        "retry reuses the rolled-back sequence");
    assert(!lists[0].handle.is_valid(), "successful retry consumes the token");
    gpu::wait_completion(point, 1_000_000_000)!!;
}

fn vk::Result reject_swapchain_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)fence;
    assert(submit_count == 1, "readiness emits one native submit");
    vk::SubmitInfo2* submit = &submits[0];
    assert(submit.wait_semaphore_info_count == 1,
        "readiness adds one private acquire wait");
    assert(submit.p_wait_semaphore_infos[0].stage_mask
        == vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        "acquire readiness covers the complete submission");
    assert(submit.signal_semaphore_info_count == 2,
        "readiness adds one private present signal");
    assert(submit.p_signal_semaphore_infos[0].stage_mask
        == vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        "present readiness covers the complete submission");
    return vk::Result.ERROR_OUT_OF_HOST_MEMORY;
}
fn void native_submit_failure_preserves_swapchain_readiness() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::SwapchainCell[gpu::MAX_SWAPCHAINS] cells;
    state.presentation.swapchains.owner = gpu::resource_owner_tag(
        state.resource_owner,
        gpu::ResourceOwnerKind.SWAPCHAIN,
    );
    state.presentation.swapchains.slots = cells[..];
    defer state.presentation.swapchains.slots = {};

    vk::SwapchainSlot pending = {
        .acquire_pending           = true,
        .acquisition               = 1,
        .pending_acquire_semaphore = (vk::Semaphore)1,
    };
    pending.present_semaphores[0] = (vk::Semaphore)2;
    gpu::SwapchainHandle handle =
        state.presentation.swapchains.alloc(pending)!!;
    gpu::SwapchainReadiness readiness = vk::make_swapchain_readiness(
        device,
        handle,
        1,
    )!!;
    gpu::SubmitDesc desc = { .readiness = readiness };

    assert(@catch(vk::vk_submit_with(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        &desc,
        &reject_swapchain_submit,
    )) == gpu::OUT_OF_HOST_MEMORY,
        "native failure preserves the readiness fault contract");
    vk::SwapchainSlot* preserved =
        state.presentation.swapchains.get(handle)!!;
    assert(!preserved.readiness_consumed,
        "native failure leaves readiness retryable");
    assert(!preserved.render_completion.is_valid(),
        "native failure publishes no render completion");
}
fn void helper_submit_failure_preserves_cleanup_token() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;

    gpu::CommandList commands;
    vk::vk_begin_commands_with_context(
        state:          state,
        queue:          queue,
        context_handle: state.helper_context,
        out_commands:   &commands,
    )!!;
    vk::vk_end_commands(&commands)!!;

    fault submit_fault = @catch(vk::submit_and_wait_with(
        &device,
        &commands,
        &reject_user_submit,
    ));
    assert(submit_fault == gpu::OUT_OF_HOST_MEMORY,
        "helper submission preserves the native fault");
    assert(commands.handle.is_valid(),
        "helper submission failure preserves its cleanup token");

    vk::release_helper_command_on_fault(state, &commands);
    assert(!commands.handle.is_valid(),
        "helper fault cleanup consumes the preserved token");
}
fn void completion_distance_backpressure_preserves_submit_state() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::QueueCompletionState* completion =
        &state.queue_completions[state.graphics_completion_id];

    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    ulong next_before = completion.next_sequence;
    ulong observed_before = completion.observed_completed_sequence;
    ulong limit_before = state.max_timeline_semaphore_value_difference;

    completion.next_sequence = 2;
    completion.observed_completed_sequence = 0;
    state.max_timeline_semaphore_value_difference = 2;
    counted_user_submit_calls = 0;
    completion_counter_query_calls = 0;
    completion_counter_value = 0;
    completion_counter_result = vk::Result.SUCCESS;

    assert(@catch(vk::vk_submit_with_queries(
        queue:            gpu::get_queue(
            &device,
            gpu::QueueKind.GRAPHICS,
        )!!,
        desc:             &desc,
        submit_fn:        &reject_counted_user_submit,
        counter_query_fn: &query_completion_counter,
    )) == gpu::DEVICE_BUSY,
        "completion-distance backpressure is retryable");
    assert(completion_counter_query_calls == 1,
        "headroom is queried only when the cached value is insufficient");
    assert(counted_user_submit_calls == 0,
        "backpressure reaches no native queue submit");
    assert(completion.next_sequence == 2,
        "backpressure reserves no completion sequence");
    assert(lists[0].handle.is_valid(),
        "backpressure preserves the executable token");

    completion.next_sequence = next_before;
    completion.observed_completed_sequence = observed_before;
    state.max_timeline_semaphore_value_difference = limit_before;
    gpu::CompletionPoint point = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)!!;
    gpu::wait_completion(point, 1_000_000_000)!!;
}

fn void completion_sequence_exhaustion_precedes_native_submit() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::QueueCompletionState* completion =
        &state.queue_completions[state.graphics_completion_id];

    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    ulong next_before = completion.next_sequence;

    completion.next_sequence = gpu::MAX_COMPLETION_SEQUENCE;
    counted_user_submit_calls = 0;
    completion_counter_query_calls = 0;

    assert(@catch(vk::vk_submit_with_queries(
        queue:            gpu::get_queue(
            &device,
            gpu::QueueKind.GRAPHICS,
        )!!,
        desc:             &desc,
        submit_fn:        &reject_counted_user_submit,
        counter_query_fn: &query_completion_counter,
    )) == gpu::SLOT_TABLE_FULL,
        "sequence exhaustion keeps its public fault");
    assert(completion_counter_query_calls == 0,
        "sequence exhaustion requires no counter query");
    assert(counted_user_submit_calls == 0,
        "sequence exhaustion reaches no native queue submit");
    assert(completion.next_sequence == gpu::MAX_COMPLETION_SEQUENCE,
        "sequence exhaustion does not mutate queue state");
    assert(lists[0].handle.is_valid(),
        "sequence exhaustion preserves the executable token");

    completion.next_sequence = next_before;
    gpu::CompletionPoint point = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)!!;
    gpu::wait_completion(point, 1_000_000_000)!!;
}

fn void native_device_loss_rolls_back_and_allows_discard() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::QueueCompletionState* completion =
        &state.queue_completions[state.graphics_completion_id];

    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    ulong next_before = completion.next_sequence;
    ulong published_before = completion.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );

    assert(@catch(vk::vk_submit_with(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        &desc,
        &lose_user_submit,
    ))
        == gpu::DEVICE_LOST,
        "native device loss keeps its public fault");
    assert(completion.next_sequence == next_before,
        "device loss rolls back the reserved queue sequence");
    assert(completion.published_sequence.load(AtomicOrdering.ACQUIRE)
        == published_before,
        "device loss publishes no completion point");
    assert(lists[0].handle.is_valid(),
        "device loss leaves the executable token discardable");

    gpu::discard_executable_commands(&lists[0])!!;
    assert(!lists[0].handle.is_valid(),
        "discard consumes the token after device loss");
}

fn void completion_wait_infos_map_cross_queue_and_elide_same_queue() @test {
    gpu::Device device = { .index = 1, .generation = 1 };
    gpu::Queue source_queue = {
        .device = device,
        .id     = 1,
        .roles  = { .compute },
    };
    gpu::Queue target_queue = {
        .device = device,
        .id     = 0,
        .roles  = { .graphics },
    };
    gpu::CompletionPoint[2] points = {
        gpu::make_completion_point(source_queue, 7)!!,
        gpu::make_completion_point(target_queue, 3)!!,
    };
    vk::QueueCompletionState source = {
        .timeline = (vk::Semaphore)(uptr)0x10,
    };
    vk::QueueCompletionState target = {
        .timeline = (vk::Semaphore)(uptr)0x20,
    };
    vk::QueueCompletionState*[2] completions = { &source, &target };
    vk::SemaphoreSubmitInfo[2] waits;

    usz count = vk::append_completion_wait_infos(
        waits:       waits[..],
        next_wait:   0,
        points:      points[..],
        completions: completions[..],
        target:      &target,
    );
    vk::SubmitInfo2 submit = vk::submit_info2()
        .set_wait_semaphore_infos(waits[:count]);

    assert(submit.wait_semaphore_info_count == 1,
        "same-queue completion waits emit no native dependency");
    assert(submit.p_wait_semaphore_infos[0].semaphore == source.timeline,
        "cross-queue waits map to the source timeline");
    assert(submit.p_wait_semaphore_infos[0].value == 7,
        "cross-queue waits keep the source sequence");
    assert(submit.p_wait_semaphore_infos[0].stage_mask
        == (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        "cross-queue waits cover all commands");
}

fn void empty_submit_returns_completion_point() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    gpu::SubmitDesc desc;
    ulong before = stats_of(&device).queue_submits;

    gpu::CompletionPoint point = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)!!;

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(point.is_valid(), "an empty submit still returns queue progress");
    assert(gpu::completion_point_queue_id(point)
        == state.graphics_completion_id,
        "an empty submit targets the selected graphics queue");
    assert(stats_of(&device).queue_submits == before + 1,
        "an empty submit emits one completion signal");
    gpu::wait_completion(point, 1_000_000_000)!!;
}

fn void foreign_completion_wait_rejects_before_native_submit() @test {
    gpu::Device producer = make_submit_stats_test_device();
    defer destroy_test_device(&producer)!!;
    gpu::Device consumer = make_submit_stats_test_device();
    defer destroy_test_device(&consumer)!!;

    gpu::SubmitDesc producer_submit;
    gpu::CompletionPoint foreign = submit_test_queue(&producer, gpu::QueueKind.GRAPHICS,
        &producer_submit,
    )!!;
    gpu::wait_completion(foreign, 1_000_000_000)!!;

    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&consumer, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::CompletionPoint[1] waits = { foreign };
    gpu::SubmitDesc desc = {
        .command_lists    = lists[..],
        .completion_waits = waits[..],
    };
    ulong submits_before = stats_of(&consumer).queue_submits;

    assert(@catch(submit_test_queue(&consumer, gpu::QueueKind.GRAPHICS, &desc)) == gpu::INVALID_HANDLE,
        "a completion point from another device is rejected");
    assert(stats_of(&consumer).queue_submits == submits_before,
        "foreign completion rejection reaches no native submit");
    assert(lists[0].handle.is_valid(),
        "foreign completion rejection preserves the executable token");

    desc.completion_waits = {};
    gpu::CompletionPoint point = submit_test_queue(&consumer, gpu::QueueKind.GRAPHICS, &desc)!!;
    gpu::wait_completion(point, 1_000_000_000)!!;
}

fn void unpublished_same_queue_wait_preserves_token() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CompletionPoint future = vk::reserve_queue_completion(
        state,
        queue,
    )!!;

    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::CompletionPoint[1] waits = { future };
    gpu::SubmitDesc desc = {
        .command_lists    = lists[..],
        .completion_waits = waits[..],
    };

    assert(@catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)) == gpu::INVALID_HANDLE,
        "an unpublished same-queue wait is rejected");
    assert(lists[0].handle.is_valid(),
        "invalid completion wait preserves the executable token");

    vk::QueueCompletionState* completion =
        &state.queue_completions[state.graphics_completion_id];
    completion.submit_mutex.lock()!!;
    vk::cancel_queue_completion_locked(state, future)!!;
    completion.submit_mutex.unlock();
    desc.completion_waits = {};

    gpu::CompletionPoint point = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)!!;
    gpu::wait_completion(point, 1_000_000_000)!!;
}
