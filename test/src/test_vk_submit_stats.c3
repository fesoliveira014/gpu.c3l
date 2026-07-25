module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::thread;
import std::atomic::types;
import std::io;


ulong rejected_user_completion_value;
uint rejected_user_signal_count;
uint counted_user_submit_calls;
uint accepted_swapchain_submit_calls;
uint completion_counter_query_calls;
ulong completion_counter_value;
vk::Result completion_counter_result;
uint same_queue_elision_submit_calls;
uint draw_argument_wait_submit_calls;
vk::PipelineStageFlags2 expected_draw_argument_wait_stages;
Atomic{uint} distinct_native_entered;
Atomic{uint} distinct_native_active;
Atomic{bool} distinct_native_release;
Atomic{bool} distinct_native_overlap;

struct SubmissionCheckpointGate {
    vk::SubmissionCheckpoint target;
    gpu::CompletionPoint     point;
    Atomic{bool}             entered;
    Atomic{bool}             release;
}

struct SubmissionCheckpointWorkerArgs {
    gpu::Queue                 queue;
    gpu::ExecutableCommandList executable;
    gpu::CompletionPoint       submitted;
}

struct ConcurrentSubmitWorkerArgs {
    gpu::Queue                  queue;
    gpu::ExecutableCommandList  executable;
    Atomic{uint}*               ready;
    Atomic{bool}*               release;
    gpu::CompletionPoint        submitted;
}

struct DistinctSubmitWorkerArgs {
    gpu::Queue                  queue;
    gpu::ExecutableCommandList executable;
    Atomic{uint}*               ready;
    Atomic{bool}*               start;
    gpu::CompletionPoint        submitted;
}

struct AllocatorSubmissionState {
    uint available;
    uint recordings;
    uint executables;
    uint in_flight;
    bool scratch_in_use;
}

fn bool submit_stats_work_matches_exact(
    vk::SubmitWorkCounters work,
    ulong resolutions,
    ulong duplicate_visits,
    ulong command_mutex_acquisitions,
    ulong rollback_mutex_acquisitions,
    ulong native_submissions,
) {
    return work.authoritative_token_resolutions == resolutions
        && work.duplicate_token_visits == duplicate_visits
        && work.epoch_reset_cells == 0
        && work.command_mutex_acquisitions == command_mutex_acquisitions
        && work.queue_submission_mutex_acquisitions == 1
        && work.rollback_mutex_acquisitions == rollback_mutex_acquisitions
        && work.native_submissions == native_submissions;
}

fn bool submit_stats_work_matches(
    vk::SubmitWorkCounters work,
    ulong inspected,
    ulong command_mutex_acquisitions,
    ulong rollback_mutex_acquisitions,
    ulong native_submissions,
) {
    return submit_stats_work_matches_exact(
        work:                        work,
        resolutions:                 inspected,
        duplicate_visits:            inspected,
        command_mutex_acquisitions:  command_mutex_acquisitions,
        rollback_mutex_acquisitions: rollback_mutex_acquisitions,
        native_submissions:          native_submissions,
    );
}

fn vk::Result accept_injected_publication_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)submits;
    (void)fence;
    assert(submit_count == 1,
        "the publication seam observes one native submission");
    return vk::Result.SUCCESS;
}

fn void gate_submission_checkpoint(
    void* user_data,
    vk::SubmissionCheckpoint checkpoint,
    gpu::CompletionPoint point,
) {
    SubmissionCheckpointGate* gate = (SubmissionCheckpointGate*)user_data;
    if (checkpoint != gate.target) return;
    gate.point = point;
    gate.entered.store(true, AtomicOrdering.RELEASE);
    while (!gate.release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
}

fn int submission_checkpoint_worker(void* raw_args) {
    SubmissionCheckpointWorkerArgs* args =
        (SubmissionCheckpointWorkerArgs*)raw_args;
    gpu::ExecutableCommandList[1] lists = { args.executable };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    args.submitted = vk::vk_submit_with(
        args.queue,
        &desc,
        &accept_injected_publication_submit,
    )!!;
    gpu::Device device = args.queue.device;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(
        &device,
    )!!;
    release_submitted_executables(&operation, lists[..]);
    args.executable = lists[0];
    gpu::internal::end_device_operation(&operation);
    return 0;
}

fn int concurrent_submit_worker(void* raw_args) {
    ConcurrentSubmitWorkerArgs* args =
        (ConcurrentSubmitWorkerArgs*)raw_args;
    (void)args.ready.add(1, AtomicOrdering.RELEASE);
    while (!args.release.load(AtomicOrdering.ACQUIRE)) thread::yield();
    gpu::ExecutableCommandList[1] lists = { args.executable };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    gpu::CompletionPoint? submitted = gpu::submit(args.queue, &desc);
    if (catch submitted) return 1;
    args.submitted = submitted;
    return 0;
}

fn vk::Result overlap_distinct_native_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)submits;
    (void)fence;
    assert(submit_count == 1,
        "each distinct queue reaches one native submit");
    (void)distinct_native_entered.add(1, AtomicOrdering.ACQUIRE_RELEASE);
    uint active = distinct_native_active.add(
        1,
        AtomicOrdering.ACQUIRE_RELEASE,
    ) + 1;
    if (active == 2) {
        distinct_native_overlap.store(true, AtomicOrdering.RELEASE);
        distinct_native_release.store(true, AtomicOrdering.RELEASE);
    }
    for (uint spin = 0;
        spin < 1_000_000
            && !distinct_native_release.load(AtomicOrdering.ACQUIRE);
        spin++) {
        thread::yield();
    }
    distinct_native_release.store(true, AtomicOrdering.RELEASE);
    (void)distinct_native_active.sub(1, AtomicOrdering.ACQUIRE_RELEASE);
    return vk::Result.SUCCESS;
}

fn int distinct_submit_worker(void* raw_args) {
    DistinctSubmitWorkerArgs* args = (DistinctSubmitWorkerArgs*)raw_args;
    (void)args.ready.add(1, AtomicOrdering.RELEASE);
    while (!args.start.load(AtomicOrdering.ACQUIRE)) thread::yield();
    gpu::ExecutableCommandList[1] lists = { args.executable };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    gpu::CompletionPoint? submitted = vk::vk_submit_with(
        args.queue,
        &desc,
        &overlap_distinct_native_submit,
    );
    if (catch submitted) return 1;
    args.submitted = submitted;
    gpu::Device device = args.queue.device;
    gpu::internal::DeviceOperation operation =
        gpu::internal::begin_device_operation(&device)!!;
    release_submitted_executables(&operation, lists[..]);
    gpu::internal::end_device_operation(&operation);
    args.executable = lists[0];
    return 0;
}

fn AllocatorSubmissionState allocator_submission_state(
    vk::VkDeviceState* state,
    gpu::CommandAllocatorHandle handle,
    uint buffer_index,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    vk::CommandAllocatorSlot* slot = state.command_allocators.get(handle)!!;
    slot.mutex.lock()!!;
    defer slot.mutex.unlock();
    return {
        .available      = slot.available_count,
        .recordings     = slot.live_recordings,
        .executables    = slot.live_executables,
        .in_flight      = slot.in_flight_buffers,
        .scratch_in_use = slot.scratch[buffer_index].in_use,
    };
}

fn void assert_allocator_submission_state(
    AllocatorSubmissionState actual,
    uint available,
    uint recordings,
    uint executables,
    uint in_flight,
    bool scratch_in_use,
    String message,
) {
    assert(actual.available == available
            && actual.recordings == recordings
            && actual.executables == executables
            && actual.in_flight == in_flight
            && actual.scratch_in_use == scratch_in_use,
        "%s", message);
}

fn gpu::Device make_submit_stats_test_device() {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_submit_stats_test",
    };
    return create_test_device(&desc)!!;
}

fn void concurrent_same_device_submits_allocate_distinct_visit_epochs() @test {
    const uint WORKER_COUNT = 2;
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    Atomic{uint} ready;
    Atomic{bool} release;
    ConcurrentSubmitWorkerArgs[WORKER_COUNT] args;
    gpu::CommandAllocator[WORKER_COUNT] allocators;
    Thread[WORKER_COUNT] workers;

    foreach (i, &worker : workers) {
        allocators[i] = gpu::create_command_allocator(&device, queue)!!;
        gpu::CommandList commands = gpu::begin_commands(&allocators[i])!!;
        args[i] = {
            .queue      = queue,
            .executable = gpu::end_commands(&commands)!!,
            .ready      = &ready,
            .release    = &release,
        };
        worker.create(&concurrent_submit_worker, &args[i])!!;
    }
    while (ready.load(AtomicOrdering.ACQUIRE) < WORKER_COUNT) thread::yield();
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::reset_submit_work_counters(state);
    ulong submits_before = stats_of(&device).queue_submits;
    release.store(true, AtomicOrdering.RELEASE);

    foreach (i, &worker : workers) {
        assert(worker.join()!! == 0,
            "each concurrent submit succeeds");
    }
    vk::SubmitWorkCounters work = vk::submit_work_counters(state);
    assert(work.authoritative_token_resolutions == WORKER_COUNT
            && work.duplicate_token_visits == WORKER_COUNT
            && work.epoch_reset_cells == 0
            && work.command_mutex_acquisitions == WORKER_COUNT
            && work.queue_submission_mutex_acquisitions == WORKER_COUNT
            && work.native_submissions == WORKER_COUNT,
        "concurrent submits charge one visit under distinct serialized epochs");
    assert(stats_of(&device).queue_submits == submits_before + WORKER_COUNT,
        "both concurrent submits reach the native queue");
    foreach (&worker_args : args) {
        gpu::wait_completion(worker_args.submitted)!!;
    }
    foreach (&allocator : allocators) {
        gpu::destroy_command_allocator(allocator)!!;
    }
}

fn void distinct_selected_queues_reach_native_submit_concurrently() @test {
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_distinct_submit_concurrency_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 1 },
        .distinct = { .compute },
    };
    gpu::DeviceRequest request = gpu::request_queues(
        gpu::strict_device_request(),
        requirements,
    )!!;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) {
            continue;
        }
        gpu::Device device = gpu::create_device(&adapter, &request)!!;
        defer gpu::destroy_device(&device)!!;
        gpu::Queue[2] queues = {
            gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
            gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!,
        };
        assert(queues[0].id != queues[1].id,
            "the concurrency regression requires distinct queue identities");
        gpu::CommandAllocator[2] allocators;
        DistinctSubmitWorkerArgs[2] args;
        Thread[2] workers;
        Atomic{uint} ready;
        Atomic{bool} start;

        foreach (worker_index, &worker : workers) {
            allocators[worker_index] = gpu::create_command_allocator(
                &device,
                queues[worker_index],
            )!!;
            gpu::CommandList commands =
                gpu::begin_commands(&allocators[worker_index])!!;
            args[worker_index] = {
                .queue      = queues[worker_index],
                .executable = gpu::end_commands(&commands)!!,
                .ready      = &ready,
                .start      = &start,
            };
            worker.create(&distinct_submit_worker, &args[worker_index])!!;
        }
        while (ready.load(AtomicOrdering.ACQUIRE) < workers.len) {
            thread::yield();
        }
        vk::VkDeviceState* state =
            (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
        vk::SubmitStats stats_before = vk::submit_stats(state);
        vk::reset_submit_work_counters(state);
        distinct_native_entered.store(0, AtomicOrdering.RELAXED);
        distinct_native_active.store(0, AtomicOrdering.RELAXED);
        distinct_native_release.store(false, AtomicOrdering.RELAXED);
        distinct_native_overlap.store(false, AtomicOrdering.RELAXED);
        start.store(true, AtomicOrdering.RELEASE);

        foreach (&worker : workers) {
            assert(worker.join()!! == 0,
                "each distinct-queue submit succeeds");
        }
        vk::SubmitWorkCounters work = vk::submit_work_counters(state);
        assert(distinct_native_overlap.load(AtomicOrdering.ACQUIRE)
                && distinct_native_entered.load(AtomicOrdering.ACQUIRE) == 2,
            "distinct selected queues overlap at the native submit seam");
        assert(work.authoritative_token_resolutions == 2
                && work.duplicate_token_visits == 2
                && work.command_mutex_acquisitions == 2
                && work.queue_submission_mutex_acquisitions == 2
                && work.native_submissions == 2
                && work.epoch_reset_cells == 0
                && work.rollback_mutex_acquisitions == 0,
            "distinct submits preserve exact concurrent work accounting");
        vk::SubmitStats stats_after = vk::submit_stats(state);
        assert(stats_after.queue_submits == stats_before.queue_submits + 2
                && stats_after.completion_point_publications
                    == stats_before.completion_point_publications + 2,
            "distinct submits publish exact atomic aggregate statistics");

        foreach (worker_index, &worker_args : args) {
            vk::QueueCompletionState* completion =
                vk::queue_completion_state(
                    state,
                    queues[worker_index].id,
                )!!;
            completion.retirement_mutex.lock()!!;
            vk::retire_queue_through_locked(
                state,
                completion,
                gpu::internal::completion_point_sequence(
                    worker_args.submitted,
                ),
            );
            completion.retirement_mutex.unlock();
        }
        foreach (&allocator : allocators) {
            gpu::destroy_command_allocator(allocator)!!;
        }
        return;
    }
    io::eprintfn(
        "distinct_selected_queues_reach_native_submit_concurrently: "
        "distinct graphics/compute queues unavailable, skipping",
    );
}

fn vk::SubmitStats stats_of(gpu::Device* device) {
    return vk::submit_stats((vk::VkDeviceState*)gpu::internal::device_backend_state(*device));
}

<* A submit rejected before reaching the driver records no queue submit. *>
fn void failed_submit_records_nothing() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList unfinished = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&unfinished)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::CommandRecord* record = vk::resolve_command(&unfinished)!!;
    vk::CommandAllocatorSlot* allocator_slot =
        state.command_allocators.get(command_allocator.handle)!!;
    uint available_before = allocator_slot.available_count;
    gpu::ExecutableCommandList[1] not_executable;
    gpu::internal::publish_executable_command_list(
        record,
        &not_executable[0],
    );
    gpu::SubmitDesc bad_submit = { .command_lists = not_executable[..] };

    vk::SubmitStats before = stats_of(&device);
    fault submit_fault = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &bad_submit));
    assert(submit_fault == gpu::COMMAND_RECORDING_ERROR, "a not-yet-executable list rejects before touching the queue");
    vk::SubmitStats after = stats_of(&device);
    assert(after.queue_submits == before.queue_submits, "a rejected submit records no queue submit");
    assert_allocator_submission_state(
        actual:         allocator_submission_state(
            state,
            command_allocator.handle,
            record.allocator_buffer,
        ),
        available:      available_before,
        recordings:     1,
        executables:    0,
        in_flight:      0,
        scratch_in_use: true,
        message:        "validation failure leaves the allocator-owned recording exact",
    );
    assert(@catch(gpu::destroy_command_allocator(&command_allocator))
            == gpu::RESOURCE_IN_USE
        && command_allocator.handle.is_valid(),
        "validation failure leaves allocator destruction retryably blocked");

    gpu::ExecutableCommandList unfinished_executable = gpu::end_commands(&unfinished)!!;
    defer discard_test_executable(&unfinished_executable)!!;
}

fn void different_queue_submit_preserves_executable_allocator_unit_when_available() @test {
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_different_queue_submit_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 1 },
        .distinct = { .compute },
    };
    gpu::DeviceRequest request = gpu::request_queues(
        gpu::strict_device_request(),
        requirements,
    )!!;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) {
            continue;
        }
        gpu::Device device = gpu::create_device(&adapter, &request)!!;
        defer gpu::destroy_device(&device)!!;
        gpu::Queue graphics = gpu::get_queue(
            &device,
            gpu::QueueKind.GRAPHICS,
        )!!;
        gpu::Queue compute = gpu::get_queue(
            &device,
            gpu::QueueKind.COMPUTE,
        )!!;
        assert(graphics.id != compute.id,
            "the supported distinct request publishes different queue identities");
        gpu::CommandAllocatorDesc allocator_desc = {
            .command_buffer_capacity = 1,
        };
        gpu::CommandAllocator allocator = gpu::create_command_allocator(
            &device,
            compute,
            &allocator_desc,
        )!!;
        defer gpu::destroy_command_allocator(&allocator)!!;
        gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
        vk::CommandRecord* record = vk::resolve_command(&commands)!!;
        uint allocator_buffer = record.allocator_buffer;
        gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
        defer discard_test_executable(&executable)!!;
        gpu::ExecutableCommandList[1] lists = { executable };
        gpu::SubmitDesc desc = { .command_lists = lists[..] };
        vk::VkDeviceState* state =
            (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
        ulong submits_before = stats_of(&device).queue_submits;
        vk::reset_submit_work_counters(state);

        assert(@catch(gpu::submit(graphics, &desc)) == gpu::INVALID_ARGUMENT,
            "an executable faults before native submit on a different queue");
        assert(stats_of(&device).queue_submits == submits_before
                && lists[0].is_valid()
                && record.state == gpu::internal::CommandState.EXECUTABLE,
            "different-queue rejection preserves the token before native work");
        assert(submit_stats_work_matches_exact(
            work:                        vk::submit_work_counters(state),
            resolutions:                 1,
            duplicate_visits:            0,
            command_mutex_acquisitions:  1,
            rollback_mutex_acquisitions: 0,
            native_submissions:          0,
        ), "exact wrong-queue rejection stops before duplicate visitation");
        assert_allocator_submission_state(
            actual:         allocator_submission_state(
                state,
                allocator.handle,
                allocator_buffer,
            ),
            available:      0,
            recordings:     0,
            executables:    1,
            in_flight:      0,
            scratch_in_use: true,
            message:        "different-queue rejection preserves exact allocator ownership",
        );
        assert(@catch(gpu::destroy_command_allocator(&allocator))
                == gpu::RESOURCE_IN_USE,
            "different-queue rejection keeps allocator destruction retryable");

        gpu::CompletionPoint completion = gpu::submit(compute, &desc)!!;
        gpu::wait_completion(completion, ulong::max)!!;
        return;
    }
    io::eprintfn(
        "different_queue_submit_preserves_executable_allocator_unit_when_available: "
        "distinct graphics/compute queues unavailable, skipping",
    );
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

fn vk::Result observe_same_queue_elision(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)fence;
    assert(submit_count == 1, "same-queue test emits one native submit");
    assert(submits[0].wait_semaphore_info_count == 0,
        "the full submit path elides a same-queue timeline wait");
    same_queue_elision_submit_calls++;
    return vk::Result.ERROR_OUT_OF_HOST_MEMORY;
}

fn vk::Result observe_draw_argument_wait(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)fence;
    assert(submit_count == 1,
        "the draw-argument wait emits one native submit");
    assert(submits[0].wait_semaphore_info_count == 1,
        "the cross-queue dependency emits one native wait");
    assert(submits[0].p_wait_semaphore_infos[0].stage_mask
            == expected_draw_argument_wait_stages,
        "the native wait uses the exact generated-work-aware consumer stages");
    draw_argument_wait_submit_calls++;
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

fn void prove_submission_publication_gap(
    vk::SubmissionCheckpoint checkpoint,
) {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, queue.id)!!;
    gpu::CompletionPoint prior = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, prior)!!;
    gpu::CompletionPoint predicted = gpu::internal::make_completion_point(
        queue,
        gpu::internal::completion_point_sequence(prior) + 1,
    )!!;

    OwnedSpan retained = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics },
        debug_name:   "publication_retained_resource",
    )!!;

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    uint allocator_buffer = record.allocator_buffer;
    vk::CommandAllocatorSlot* allocator_slot =
        state.command_allocators.get(allocator.handle)!!;
    uint available_while_owned = allocator_slot.available_count;
    vk::track_command_reference(
        state:      state,
        record:     record,
        owner:      retained.allocation.owner,
        index:      retained.allocation.index,
        generation: retained.allocation.generation,
    )!!;
    SubmissionCheckpointWorkerArgs args = {
        .queue      = queue,
        .executable = gpu::end_commands(&commands)!!,
    };
    SubmissionCheckpointGate gate = { .target = checkpoint };
    vk::set_submission_checkpoint_hook(
        state,
        &gate_submission_checkpoint,
        &gate,
    );
    defer vk::set_submission_checkpoint_hook(state, null, null);
    Thread worker;
    worker.create(&submission_checkpoint_worker, &args)!!;
    while (!gate.entered.load(AtomicOrdering.ACQUIRE)) thread::yield();

    bool native_accepted =
        checkpoint == vk::SubmissionCheckpoint.NATIVE_ACCEPTED;
    assert_allocator_submission_state(
        actual:         allocator_submission_state(
            state,
            allocator.handle,
            allocator_buffer,
        ),
        available:      available_while_owned,
        recordings:     0,
        executables:    native_accepted ? 1 : 0,
        in_flight:      native_accepted ? 0 : 1,
        scratch_in_use: true,
        message:        "each submission checkpoint has one exact allocator owner",
    );
    assert(args.executable.is_valid(),
        "the backend checkpoint precedes caller-token publication");
    assert(@catch(gpu::destroy_command_allocator(&allocator))
            == gpu::RESOURCE_IN_USE
        && allocator.handle.is_valid(),
        "allocator destruction stays retryably blocked across submission publication");

    completion_counter_query_calls = 0;
    completion_counter_value = gpu::internal::completion_point_sequence(predicted);
    completion_counter_result = vk::Result.SUCCESS;
    if (checkpoint == vk::SubmissionCheckpoint.TOKEN_PUBLISHED) {
        assert(vk::published_queue_completion_state(
            state,
            queue.id,
            gpu::internal::completion_point_sequence(predicted),
        )!! == completion,
            "the token-publication checkpoint exposes the predicted point to observers");
    } else {
        assert(@catch(vk::vk_poll_completion_with_query(
            device:   &device,
            queue_id: queue.id,
            sequence: gpu::internal::completion_point_sequence(predicted),
            query_fn: &query_completion_counter,
        )) == gpu::INVALID_HANDLE,
            "a predicted point remains private across the submission checkpoint");
        assert(completion_counter_query_calls == 0,
            "an unpublished predicted point reaches no native query");
        assert(vk::vk_poll_completion_with_query(
            device:   &device,
            queue_id: queue.id,
            sequence: gpu::internal::completion_point_sequence(prior),
            query_fn: &query_completion_counter,
        )!!, "the prior point can observe newer native progress");
        assert(completion.retired_sequence.load(AtomicOrdering.ACQUIRE)
            == gpu::internal::completion_point_sequence(prior),
            "native progress is capped to the published prefix");
    }
    uint expected_batches = checkpoint == vk::SubmissionCheckpoint.NATIVE_ACCEPTED
        ? 0
        : 1;
    assert(completion.submitted_count == expected_batches,
        "the checkpoint exposes only its intended batch-publication phase");
    if (checkpoint != vk::SubmissionCheckpoint.NATIVE_ACCEPTED) {
        vk::CommandRecord* submitted =
            completion.submitted_head;
        vk::CommandAllocatorSlot* submitted_allocator_slot =
            state.command_allocators.get(submitted.allocator)!!;
        vk::CommandResourceReference* reference =
            &submitted_allocator_slot.scratch[
                submitted.allocator_buffer
            ].references[0];
        assert(submitted_allocator_slot.scratch[
                submitted.allocator_buffer
            ].reference_count == 1
            && reference.owner == retained.allocation.owner
            && reference.index == retained.allocation.index
            && reference.generation
                == retained.allocation.generation,
            "the submitted batch owns every transferred resource before observation");
    }
    if (checkpoint == vk::SubmissionCheckpoint.BATCHES_PUBLISHED) {
        assert(@catch(gpu::free_allocation(
            &device,
            &retained.allocation,
        )) == gpu::RESOURCE_IN_USE,
            "caller release during submitted ownership faults RESOURCE_IN_USE");
        assert(retained.allocation.is_valid()
            && gpu::get_allocation_info(
                &device,
                retained.allocation,
            )!!.size == retained.span.size,
            "rejected submitted release preserves the caller token and native resource");
    }

    gate.release.store(true, AtomicOrdering.RELEASE);
    assert(worker.join()!! == 0 && args.submitted == predicted,
        "submission resumes and publishes the predicted point");
    if (checkpoint != vk::SubmissionCheckpoint.TOKEN_PUBLISHED) {
        assert(completion.retired_sequence.load(AtomicOrdering.ACQUIRE)
            == gpu::internal::completion_point_sequence(prior),
            "resuming private publication does not invent native retirement");
    }
    assert(vk::vk_poll_completion_with_query(
        device:   &device,
        queue_id: queue.id,
        sequence: gpu::internal::completion_point_sequence(args.submitted),
        query_fn: &query_completion_counter,
    )!!, "a later observation retires the published submission");
    assert(completion.submitted_count == 0,
        "full retirement releases the published command batch");
    assert_allocator_submission_state(
        actual:         allocator_submission_state(
            state,
            allocator.handle,
            allocator_buffer,
        ),
        available:      available_while_owned + 1,
        recordings:     0,
        executables:    0,
        in_flight:      0,
        scratch_in_use: false,
        message:        "retirement returns the exact allocator buffer and scratch",
    );
    gpu::free_allocation(&device, &retained.allocation)!!;
    retained.span = {};
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(
        &device,
        true,
    )!!;
    assert(operation.slot.live_children.load(AtomicOrdering.ACQUIRE) == 1,
        "publication regression leaves only the explicit command allocator live");
    gpu::internal::end_device_operation(&operation);
}

fn void native_acceptance_remains_capped_before_batch_publication() @test {
    prove_submission_publication_gap(
        vk::SubmissionCheckpoint.NATIVE_ACCEPTED,
    );
}

fn void batch_publication_remains_private_before_point_publication() @test {
    prove_submission_publication_gap(
        vk::SubmissionCheckpoint.BATCHES_PUBLISHED,
    );
}

fn void token_publication_follows_complete_ownership_transfer() @test {
    prove_submission_publication_gap(
        vk::SubmissionCheckpoint.TOKEN_PUBLISHED,
    );
}

fn void native_submit_failure_rolls_back_completion_and_tokens() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::QueueCompletionState* completion =
        &state.queue_completions[state.graphics_completion_id];

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    uint allocator_buffer = record.allocator_buffer;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    ulong next_before = completion.next_sequence;
    ulong published_before = completion.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    AllocatorSubmissionState allocator_before = allocator_submission_state(
        state,
        command_allocator.handle,
        allocator_buffer,
    );
    vk::reset_submit_work_counters(state);

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
    assert(submit_stats_work_matches(
        work:                        vk::submit_work_counters(state),
        inspected:                   1,
        command_mutex_acquisitions:  1,
        rollback_mutex_acquisitions: 1,
        native_submissions:          1,
    ), "native rejection performs one claim, one submit, and one rollback");
    assert(completion.submitted_count == 0
            && completion.submitted_head == null
            && completion.submitted_tail == null,
        "native rejection publishes no intrusive pending record");
    assert(lists[0].is_valid(),
        "native failure preserves the executable token for retry");
    AllocatorSubmissionState allocator_after = allocator_submission_state(
        state,
        command_allocator.handle,
        allocator_buffer,
    );
    assert(allocator_after.available == allocator_before.available
            && allocator_after.recordings == allocator_before.recordings
            && allocator_after.executables == allocator_before.executables
            && allocator_after.in_flight == allocator_before.in_flight
            && allocator_after.scratch_in_use
                == allocator_before.scratch_in_use,
        "native failure preserves exact allocator counters and scratch ownership");
    assert(@catch(gpu::destroy_command_allocator(&command_allocator))
            == gpu::RESOURCE_IN_USE
        && command_allocator.handle.is_valid(),
        "native failure leaves the executable allocator retryably live");

    gpu::CompletionPoint point = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)!!;
    assert(point.is_valid(), "retry returns a valid completion point");
    assert(completion.next_sequence == next_before + 1,
        "retry reuses the rolled-back sequence");
    assert(!lists[0].is_valid(), "successful retry consumes the token");
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
        == vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        "acquire readiness keeps the requested first destination stage");
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
    (void)fence;
    assert(submit_count == 1, "readiness emits one native submit");
    vk::SubmitInfo2* submit = &submits[0];
    assert(submit.wait_semaphore_info_count == 1,
        "readiness emits one acquire wait");
    assert(submit.p_wait_semaphore_infos[0].stage_mask
        == vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT,
        "readiness keeps the requested transfer destination stage");
    accepted_swapchain_submit_calls++;
    return vk::Result.SUCCESS;
}

fn void native_submit_failure_preserves_swapchain_readiness() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::SwapchainCell[gpu::MAX_SWAPCHAINS] cells;
    state.presentation.swapchains.owner = gpu::internal::resource_owner_tag(
        state.resource_owner,
        gpu::internal::ResourceOwnerKind.SWAPCHAIN,
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
    gpu::SubmitDesc desc = {
        .readiness        = readiness,
        .readiness_before = { .color_output },
    };

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
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::SwapchainCell[gpu::MAX_SWAPCHAINS] cells;
    state.presentation.swapchains.owner = gpu::internal::resource_owner_tag(
        state.resource_owner,
        gpu::internal::ResourceOwnerKind.SWAPCHAIN,
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
    gpu::SubmitDesc desc = {
        .readiness        = readiness,
        .readiness_before = { .transfer },
    };
    gpu::Queue graphics = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::SubmitDesc empty_submit;
    gpu::CompletionPoint prior = gpu::submit(graphics, &empty_submit)!!;
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, graphics.id)!!;
    ulong next_before = completion.next_sequence;
    ulong retired_before = completion.retired_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    ulong published_before = completion.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    vk::SubmitStats stats_before = vk::submit_stats(state);
    accepted_swapchain_submit_calls = 0;

    gpu::SubmitDesc missing_stage = { .readiness = readiness };
    assert(@catch(vk::vk_submit_with(
        graphics,
        &missing_stage,
        &accept_swapchain_submit,
    )) == gpu::INVALID_ARGUMENT,
        "readiness requires a destination stage");
    assert(accepted_swapchain_submit_calls == 0,
        "missing readiness stage reaches no native submit");
    assert(!state.presentation.swapchains.get(handle)!!.readiness_consumed,
        "missing readiness stage preserves the one-shot token");

    gpu::SubmitDesc orphan_stage = {
        .readiness_before = { .color_output },
    };
    assert(@catch(vk::vk_submit_with(
        graphics,
        &orphan_stage,
        &accept_swapchain_submit,
    )) == gpu::INVALID_ARGUMENT,
        "a readiness stage without readiness is rejected");
    assert(accepted_swapchain_submit_calls == 0,
        "orphan readiness stage reaches no native submit");

    gpu::SubmitDesc host_stage = {
        .readiness        = readiness,
        .readiness_before = { .host },
    };
    assert(@catch(vk::vk_submit_with(
        graphics,
        &host_stage,
        &accept_swapchain_submit,
    )) == gpu::INVALID_ARGUMENT,
        "readiness rejects a host destination stage");
    assert(accepted_swapchain_submit_calls == 0,
        "invalid readiness stage reaches no native submit");
    assert(!state.presentation.swapchains.get(handle)!!.readiness_consumed,
        "invalid readiness stage preserves the one-shot token");

    gpu::CompletionWait[1] invalid_waits = {{
        .point     = prior,
        .consumers = (gpu::CompletionConsumerFlags)(1u << 1),
    }};
    gpu::SubmitDesc invalid_scope = {
        .completion_waits = invalid_waits[..],
        .readiness        = readiness,
        .readiness_before = { .transfer },
    };
    assert(@catch(vk::vk_submit_with(
        graphics,
        &invalid_scope,
        &accept_swapchain_submit,
    )) == gpu::INVALID_ARGUMENT,
        "an invalid completion scope faults before native submit");
    assert(accepted_swapchain_submit_calls == 0,
        "an invalid completion scope reaches no native submit");
    assert(!state.presentation.swapchains.get(handle)!!.readiness_consumed,
        "an invalid completion scope preserves readiness");

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
    gpu::SubmitDesc stale_desc = {
        .readiness        = stale,
        .readiness_before = { .color_output },
    };
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
    completion.retired_sequence.store(retired_before, AtomicOrdering.RELEASE);
    completion.published_sequence.store(
        published_before,
        AtomicOrdering.RELEASE,
    );
    state.submit_queue_submits.store(
        stats_before.queue_submits,
        AtomicOrdering.RELEASE,
    );
    state.submit_completion_point_publications.store(
        stats_before.completion_point_publications,
        AtomicOrdering.RELEASE,
    );
    state.submit_completion_wait_calls.store(
        stats_before.completion_wait_calls,
        AtomicOrdering.RELEASE,
    );
}

fn void completion_distance_backpressure_preserves_submit_state() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::QueueCompletionState* completion =
        &state.queue_completions[state.graphics_completion_id];

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    uint allocator_buffer = record.allocator_buffer;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    ulong next_before = completion.next_sequence;
    ulong retired_before = completion.retired_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    ulong published_before = completion.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    AllocatorSubmissionState allocator_before = allocator_submission_state(
        state,
        command_allocator.handle,
        allocator_buffer,
    );
    ulong limit_before = state.max_timeline_semaphore_value_difference;

    completion.next_sequence = 2;
    completion.published_sequence.store(2, AtomicOrdering.RELEASE);
    completion.retired_sequence.store(0, AtomicOrdering.RELEASE);
    state.max_timeline_semaphore_value_difference = 2;
    counted_user_submit_calls = 0;
    completion_counter_query_calls = 0;
    completion_counter_value = 0;
    completion_counter_result = vk::Result.SUCCESS;
    vk::reset_submit_work_counters(state);

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
    assert(submit_stats_work_matches(
        work:                        vk::submit_work_counters(state),
        inspected:                   1,
        command_mutex_acquisitions:  1,
        rollback_mutex_acquisitions: 1,
        native_submissions:          0,
    ), "pre-native headroom rejection rolls back one complete claim");
    assert(completion.next_sequence == 2,
        "backpressure reserves no completion sequence");
    assert(lists[0].is_valid(),
        "backpressure preserves the executable token");
    AllocatorSubmissionState allocator_after = allocator_submission_state(
        state,
        command_allocator.handle,
        allocator_buffer,
    );
    assert(allocator_after.available == allocator_before.available
            && allocator_after.executables == allocator_before.executables
            && allocator_after.in_flight == allocator_before.in_flight
            && allocator_after.scratch_in_use
                == allocator_before.scratch_in_use,
        "completion validation preserves allocator ownership exactly");
    assert(@catch(gpu::destroy_command_allocator(&command_allocator))
            == gpu::RESOURCE_IN_USE,
        "pre-native validation leaves allocator destruction retryably blocked");

    completion_counter_query_calls = 0;
    completion_counter_value = 2;
    vk::reset_submit_work_counters(state);
    assert(@catch(vk::vk_submit_with_queries(
        queue:            gpu::get_queue(
            &device,
            gpu::QueueKind.GRAPHICS,
        )!!,
        desc:             &desc,
        submit_fn:        &reject_counted_user_submit,
        counter_query_fn: &query_completion_counter,
    )) == gpu::OUT_OF_HOST_MEMORY,
        "sufficient headroom reaches the injected native submit");
    assert(completion_counter_query_calls == 1
        && completion.retired_sequence.load(AtomicOrdering.ACQUIRE) == 2,
        "a required headroom query advances the retired prefix");
    assert(lists[0].is_valid(),
        "native rejection after headroom preserves the executable token");
    assert(submit_stats_work_matches(
        work:                        vk::submit_work_counters(state),
        inspected:                   1,
        command_mutex_acquisitions:  1,
        rollback_mutex_acquisitions: 1,
        native_submissions:          1,
    ), "native rejection after headroom remains one claim and rollback");

    completion.next_sequence = next_before;
    completion.retired_sequence.store(retired_before, AtomicOrdering.RELEASE);
    completion.published_sequence.store(
        published_before,
        AtomicOrdering.RELEASE,
    );
    state.max_timeline_semaphore_value_difference = limit_before;
    gpu::CompletionPoint point = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)!!;
    gpu::wait_completion(point, 1_000_000_000)!!;
}

fn void completion_sequence_exhaustion_precedes_native_submit() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::QueueCompletionState* completion =
        &state.queue_completions[state.graphics_completion_id];

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    ulong next_before = completion.next_sequence;

    completion.next_sequence = gpu::internal::MAX_COMPLETION_SEQUENCE;
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
    assert(completion.next_sequence == gpu::internal::MAX_COMPLETION_SEQUENCE,
        "sequence exhaustion does not mutate queue state");
    assert(lists[0].is_valid(),
        "sequence exhaustion preserves the executable token");

    completion.next_sequence = next_before;
    gpu::CompletionPoint point = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)!!;
    gpu::wait_completion(point, 1_000_000_000)!!;
}

fn void native_device_loss_rolls_back_and_allows_discard() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::QueueCompletionState* completion =
        &state.queue_completions[state.graphics_completion_id];

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    uint allocator_buffer = record.allocator_buffer;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    ulong next_before = completion.next_sequence;
    ulong published_before = completion.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    AllocatorSubmissionState allocator_before = allocator_submission_state(
        state,
        command_allocator.handle,
        allocator_buffer,
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
    assert(lists[0].is_valid(),
        "device loss leaves the executable token discardable");
    AllocatorSubmissionState allocator_after = allocator_submission_state(
        state,
        command_allocator.handle,
        allocator_buffer,
    );
    assert(allocator_after.available == allocator_before.available
            && allocator_after.recordings == allocator_before.recordings
            && allocator_after.executables == allocator_before.executables
            && allocator_after.in_flight == allocator_before.in_flight
            && allocator_after.scratch_in_use
                == allocator_before.scratch_in_use,
        "device-loss submission failure preserves allocator ownership exactly");
    assert(@catch(gpu::destroy_command_allocator(&command_allocator))
            == gpu::RESOURCE_IN_USE,
        "device-loss submission failure leaves allocator destruction retryable");

    gpu::discard_executable_commands(&lists[0])!!;
    assert(!lists[0].is_valid(),
        "discard consumes the token after device loss");
}

fn void completion_wait_infos_map_cross_queue_and_elide_same_queue() @test {
    vk::QueueCompletionState source = {
        .timeline = (vk::Semaphore)(uptr)0x10,
    };
    vk::QueueCompletionState target = {
        .timeline = (vk::Semaphore)(uptr)0x20,
    };
    vk::ResolvedCompletionWait[2] completion_waits = {
        {
            .source   = &source,
            .stages   = (vk::PipelineStageFlags2)(
                (ulong)vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT
                    | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
            ),
            .sequence = 7,
        },
        {
            .source   = &target,
            .stages   = (vk::PipelineStageFlags2)
                vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
            .sequence = 3,
        },
    };
    vk::SemaphoreSubmitInfo[2] waits;

    usz count = vk::append_completion_wait_infos(
        waits:            waits[..],
        next_wait:        0,
        completion_waits: completion_waits[..],
        target:           &target,
    );
    vk::SubmitInfo2 submit = vk::submit_info2()
        .set_wait_semaphore_infos(waits[:count]);

    assert(submit.wait_semaphore_info_count == 1,
        "same-queue completion waits emit no native dependency");
    assert(submit.p_wait_semaphore_infos[0].semaphore == source.timeline,
        "cross-queue waits map to the source timeline");
    assert(submit.p_wait_semaphore_infos[0].value == 7,
        "cross-queue waits keep the source sequence");
    ulong expected =
        (ulong)vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    assert((ulong)submit.p_wait_semaphore_infos[0].stage_mask == expected,
        "cross-queue waits keep exact combined destination stages");
}

fn void completion_wait_infos_emit_every_exact_native_stage() @test {
    vk::QueueCompletionState source = {
        .timeline = (vk::Semaphore)(uptr)0x10,
    };
    vk::QueueCompletionState target = {
        .timeline = (vk::Semaphore)(uptr)0x20,
    };
    vk::PipelineStageFlags2 depth = (vk::PipelineStageFlags2)(
        (ulong)vk::PipelineStageFlagBits2.STAGE_2_EARLY_FRAGMENT_TESTS_BIT
            | (ulong)vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT
    );
    vk::PipelineStageFlags2 combined = (vk::PipelineStageFlags2)(
        (ulong)vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT
            | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
    );
    vk::PipelineStageFlags2[8] expected = {
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_VERTEX_SHADER_BIT,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        depth,
        (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        combined,
    };
    vk::ResolvedCompletionWait[8] completion_waits;
    foreach (i, stage : expected) {
        completion_waits[i] = {
            .source   = &source,
            .stages   = stage,
            .sequence = (ulong)i + 1,
        };
    }
    vk::SemaphoreSubmitInfo[8] native_waits;

    usz count = vk::append_completion_wait_infos(
        waits:            native_waits[..],
        next_wait:        0,
        completion_waits: completion_waits[..],
        target:           &target,
    );

    assert(count == expected.len,
        "every cross-queue completion wait emits one native wait");
    foreach (i, stage : expected) {
        assert(native_waits[i].stage_mask == stage,
            "native completion waits preserve every exact destination mask");
    }
}

fn void cross_queue_draw_argument_wait_reaches_exact_submit_scope() @test {
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation       = gpu::ContractValidation.FULL,
        .track_resource_lifetimes  = true,
        .enable_vulkan_validation  = true,
        .application_name          = "gpu_draw_argument_wait_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 1 },
        .distinct = { .compute },
    };
    gpu::DeviceRequest request = gpu::request_queues(
        gpu::strict_device_request(),
        requirements,
    )!!;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) {
            continue;
        }
        gpu::Device device = gpu::create_device(&adapter, &request)!!;
        defer gpu::destroy_device(&device)!!;
        gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
        gpu::Queue compute = gpu::get_queue(
            &device,
            gpu::QueueKind.COMPUTE,
        )!!;
        gpu::Queue graphics = gpu::get_queue(
            &device,
            gpu::QueueKind.GRAPHICS,
        )!!;
        assert(compute.id != graphics.id,
            "the requested compute and graphics identities are distinct");
        gpu::SubmitDesc producer;
        gpu::CompletionPoint produced = gpu::submit(compute, &producer)!!;
        gpu::wait_completion(produced)!!;
        gpu::CompletionWait[1] waits = {{
            .point     = produced,
            .consumers = { .draw_arguments },
        }};
        gpu::SubmitDesc consumer = { .completion_waits = waits[..] };
        draw_argument_wait_submit_calls = 0;
        expected_draw_argument_wait_stages = (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT;
        if (caps.generated_work) {
            ulong command_preprocess = (ulong)
                vk::PipelineStageFlagBits2.STAGE_2_COMMAND_PREPROCESS_BIT_NV;
            expected_draw_argument_wait_stages = (vk::PipelineStageFlags2)(
                (ulong)expected_draw_argument_wait_stages | command_preprocess
            );
        }

        assert(@catch(vk::vk_submit_with(
            graphics,
            &consumer,
            &observe_draw_argument_wait,
        )) == gpu::OUT_OF_HOST_MEMORY,
            "the observation seam preserves native failure rollback");
        assert(draw_argument_wait_submit_calls == 1,
            "the cross-queue draw-argument wait reaches the native seam");
        return;
    }
}

fn void invalid_completion_wait_stage_preserves_token() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::SubmitDesc empty;
    gpu::CompletionPoint prior = gpu::submit(queue, &empty)!!;

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::CompletionWait[1] waits = {{ .point = prior }};
    gpu::SubmitDesc desc = {
        .command_lists    = lists[..],
        .completion_waits = waits[..],
    };
    ulong submits_before = stats_of(&device).queue_submits;

    assert(@catch(gpu::submit(queue, &desc)) == gpu::INVALID_ARGUMENT,
        "an empty completion wait destination stage is rejected");
    assert(stats_of(&device).queue_submits == submits_before,
        "invalid wait stages reach no native submit");
    assert(lists[0].is_valid(),
        "invalid wait stages preserve the executable token");

    waits[0].consumers = (gpu::CompletionConsumerFlags)(1u << 1);
    assert(@catch(gpu::submit(queue, &desc)) == gpu::INVALID_ARGUMENT,
        "unknown completion consumer bits are rejected");
    assert(lists[0].is_valid(),
        "unknown completion consumers preserve the executable token");

    waits[0] = {
        .point     = prior,
        .before    = { .all },
        .consumers = { .draw_arguments },
    };
    assert(@catch(gpu::submit(queue, &desc)) == gpu::INVALID_ARGUMENT,
        "all cannot be combined with a completion consumer");
    assert(lists[0].is_valid(),
        "contradictory completion scopes preserve the executable token");

    waits[0] = {
        .point     = prior,
        .consumers = { .draw_arguments },
    };
    gpu::CompletionPoint completion = gpu::submit(queue, &desc)!!;
    gpu::wait_completion(completion, 1_000_000_000)!!;
}

fn void queue_incompatible_completion_consumer_rejects_through_submit() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::SubmitDesc empty;
    gpu::CompletionPoint prior = gpu::submit(queue, &empty)!!;

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::CompletionWait[1] waits = {{
        .point     = prior,
        .consumers = { .draw_arguments },
    }};
    gpu::SubmitDesc desc = {
        .command_lists    = lists[..],
        .completion_waits = waits[..],
    };
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, queue.id)!!;
    vk::QueueFamilyFlags selected_flags = completion.queue_flags;
    defer completion.queue_flags = selected_flags;
    completion.queue_flags = (vk::QueueFamilyFlags)(uint)
        vk::QueueFlagBits.TRANSFER_BIT;
    ulong submits_before = stats_of(&device).queue_submits;

    assert(@catch(gpu::submit(queue, &desc)) == gpu::INVALID_ARGUMENT,
        "a real submit rejects consumers unsupported by the target queue");
    assert(stats_of(&device).queue_submits == submits_before,
        "queue-consumer rejection reaches no native submit");
    assert(lists[0].is_valid(),
        "queue-consumer rejection preserves the executable token");

    completion.queue_flags = selected_flags;
    gpu::CompletionPoint completed = gpu::submit(queue, &desc)!!;
    gpu::wait_completion(completed, 1_000_000_000)!!;
}

fn void same_queue_wait_elides_through_full_submit_path() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::SubmitDesc empty;
    gpu::CompletionPoint prior = gpu::submit(queue, &empty)!!;

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::CompletionWait[1] waits = {{
        .point     = prior,
        .consumers = { .draw_arguments },
    }};
    gpu::SubmitDesc desc = {
        .command_lists    = lists[..],
        .completion_waits = waits[..],
    };
    same_queue_elision_submit_calls = 0;

    assert(@catch(vk::vk_submit_with(
        queue,
        &desc,
        &observe_same_queue_elision,
    )) == gpu::OUT_OF_HOST_MEMORY,
        "the injected native failure remains retryable after elision");
    assert(same_queue_elision_submit_calls == 1,
        "the full backend submit seam observed same-queue elision");
    assert(lists[0].is_valid(),
        "the injected failure preserves the executable token");

    desc.completion_waits = {};
    gpu::CompletionPoint completed = gpu::submit(queue, &desc)!!;
    gpu::wait_completion(completed, 1_000_000_000)!!;
}

fn void invalid_readiness_handle_precedes_invalid_stage_mask() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::SubmitDesc desc = {
        .readiness        = { .device = device },
        .readiness_before = { .host },
    };
    ulong submits_before = stats_of(&device).queue_submits;

    assert(@catch(gpu::submit(queue, &desc)) == gpu::INVALID_HANDLE,
        "invalid readiness identity precedes invalid stage shape");
    assert(stats_of(&device).queue_submits == submits_before,
        "readiness precedence is settled before native submission");
}

fn void empty_submit_returns_completion_point() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    gpu::SubmitDesc desc;
    vk::SubmitStats before = stats_of(&device);

    gpu::CompletionPoint point = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)!!;

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    assert(point.is_valid(), "an empty submit still returns queue progress");
    assert(gpu::internal::completion_point_queue_id(point)
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

    gpu::Queue command_queue = gpu::get_queue(&consumer, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&consumer, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::CompletionWait[1] waits = {{
        .point  = foreign,
        .before = { .all },
    }};
    gpu::SubmitDesc desc = {
        .command_lists    = lists[..],
        .completion_waits = waits[..],
    };
    ulong submits_before = stats_of(&consumer).queue_submits;

    assert(@catch(submit_test_queue(&consumer, gpu::QueueKind.GRAPHICS, &desc)) == gpu::INVALID_HANDLE,
        "a completion point from another device is rejected");
    assert(stats_of(&consumer).queue_submits == submits_before,
        "foreign completion rejection reaches no native submit");
    assert(lists[0].is_valid(),
        "foreign completion rejection preserves the executable token");

    desc.completion_waits = {};
    gpu::CompletionPoint point = submit_test_queue(&consumer, gpu::QueueKind.GRAPHICS, &desc)!!;
    gpu::wait_completion(point, 1_000_000_000)!!;
}

fn void unpublished_same_queue_wait_preserves_token() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CompletionPoint future = vk::reserve_queue_completion(
        state,
        queue,
    )!!;

    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::CompletionWait[1] waits = {{
        .point  = future,
        .before = { .all },
    }};
    gpu::SubmitDesc desc = {
        .command_lists    = lists[..],
        .completion_waits = waits[..],
    };

    assert(@catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)) == gpu::INVALID_HANDLE,
        "an unpublished same-queue wait is rejected");
    assert(lists[0].is_valid(),
        "invalid completion wait preserves the executable token");

    vk::QueueCompletionState* completion =
        &state.queue_completions[state.graphics_completion_id];
    completion.submission_mutex.lock()!!;
    vk::cancel_queue_completion_locked(state, future)!!;
    completion.submission_mutex.unlock();
    desc.completion_waits = {};

    gpu::CompletionPoint point = submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &desc)!!;
    gpu::wait_completion(point, 1_000_000_000)!!;
}
