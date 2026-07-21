module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;


ulong rejected_user_completion_value;
uint rejected_user_signal_count;
uint counted_user_submit_calls;
uint accepted_swapchain_submit_calls;
uint completion_counter_query_calls;
ulong completion_counter_value;
vk::Result completion_counter_result;

fn gpu::Device make_submit_stats_test_device() {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_submit_stats_test",
    };
    return create_test_device(&desc)!!;
}

fn vk::SubmitStats stats_of(gpu::Device* device) {
    return vk::submit_stats((vk::VkDeviceState*)gpu::device_backend_state(*device));
}

<* A submit rejected before reaching the driver records no queue submit. *>
fn void failed_submit_records_nothing() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;

    gpu::CommandList unfinished = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&unfinished)!!;
    gpu::ExecutableCommandList[1] not_executable = {{
        .device = unfinished.device,
        .handle = unfinished.handle,
        .encoder = unfinished.encoder,
    }};
    gpu::SubmitDesc bad_submit = { .command_lists = not_executable[..] };

    vk::SubmitStats before = stats_of(&device);
    fault submit_fault = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &bad_submit));
    assert(submit_fault == gpu::COMMAND_RECORDING_ERROR, "a not-yet-executable list rejects before touching the queue");
    vk::SubmitStats after = stats_of(&device);
    assert(after.queue_submits == before.queue_submits, "a rejected submit records no queue submit");

    gpu::ExecutableCommandList unfinished_executable = gpu::end_commands(&unfinished)!!;
    defer discard_test_executable(&unfinished_executable)!!;
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

fn vk::Result accept_swapchain_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)submit_count;
    (void)submits;
    (void)fence;
    accepted_swapchain_submit_calls++;
    return vk::Result.SUCCESS;
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

fn void swapchain_submit_path_guards_precede_native_mutation() @test {
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

    gpu::CompletionPoint[1] retirements;
    vk::SwapchainSlot pending = {
        .acquire_retirements       = retirements[..],
        .acquire_pending           = true,
        .acquisition               = 7,
        .pending_acquire_semaphore = (vk::Semaphore)1,
    };
    pending.present_semaphores[0] = (vk::Semaphore)2;
    gpu::SwapchainHandle handle =
        state.presentation.swapchains.alloc(pending)!!;
    gpu::SwapchainReadiness readiness = vk::make_swapchain_readiness(
        device,
        handle,
        7,
    )!!;
    gpu::SubmitDesc desc = { .readiness = readiness };
    gpu::Queue graphics = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, graphics.id)!!;
    ulong next_before = completion.next_sequence;
    ulong observed_before = completion.observed_completed_sequence;
    ulong published_before = completion.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    vk::SubmitStats stats_before = state.submit_stats;
    accepted_swapchain_submit_calls = 0;

    vk::vk_submit_with(graphics, &desc, &accept_swapchain_submit)!!;
    assert(accepted_swapchain_submit_calls == 1,
        "valid readiness reaches one native submit");

    fault replay = @catch(vk::vk_submit_with(
        graphics,
        &desc,
        &accept_swapchain_submit,
    ));
    assert(replay == gpu::INVALID_RESOURCE_STATE,
        "readiness replay faults INVALID_RESOURCE_STATE");
    assert(accepted_swapchain_submit_calls == 1,
        "readiness replay reaches no second native submit");

    vk::SwapchainSlot* slot = state.presentation.swapchains.get(handle)!!;
    slot.readiness_consumed = false;
    slot.render_completion = {};
    gpu::SwapchainReadiness stale = vk::make_swapchain_readiness(
        device,
        handle,
        6,
    )!!;
    gpu::SubmitDesc stale_desc = { .readiness = stale };
    fault stale_fault = @catch(vk::vk_submit_with(
        graphics,
        &stale_desc,
        &accept_swapchain_submit,
    ));
    assert(stale_fault == gpu::INVALID_RESOURCE_STATE,
        "stale readiness faults INVALID_RESOURCE_STATE through submit");
    assert(accepted_swapchain_submit_calls == 1,
        "stale readiness reaches no native submit");

    gpu::Queue compute_only = graphics;
    compute_only.roles = { .compute };
    fault role_fault = @catch(vk::vk_submit_with(
        compute_only,
        &desc,
        &accept_swapchain_submit,
    ));
    assert(role_fault == gpu::INVALID_ARGUMENT,
        "non-graphics readiness submit faults INVALID_ARGUMENT");
    assert(accepted_swapchain_submit_calls == 1,
        "non-graphics readiness reaches no native submit");

    completion.next_sequence = next_before;
    completion.observed_completed_sequence = observed_before;
    completion.published_sequence.store(
        published_before,
        AtomicOrdering.RELEASE,
    );
    state.submit_stats = stats_before;
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
    vk::SubmitStats before = stats_of(&device);

    gpu::CompletionPoint point = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)!!;

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(point.is_valid(), "an empty submit still returns queue progress");
    assert(gpu::completion_point_queue_id(point)
        == state.graphics_completion_id,
        "an empty submit targets the selected graphics queue");
    assert(stats_of(&device).queue_submits == before.queue_submits + 1,
        "an empty submit emits one completion signal");
    vk::SubmitStats after = stats_of(&device);
    assert(after.completion_point_publications
        == before.completion_point_publications + 1,
        "a successful submit publishes one scalar completion point");
    assert(after.completion_point_allocations
        == before.completion_point_allocations,
        "publishing a scalar completion point allocates no table object");
    gpu::wait_completion(point, 1_000_000_000)!!;
    vk::SubmitStats waited = stats_of(&device);
    assert(waited.completion_wait_calls == after.completion_wait_calls + 1,
        "a completion wait records one blocking wait call");
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
