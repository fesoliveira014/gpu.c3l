module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::thread;
import std::core::mem::allocators;
import std::atomic::types;

ulong completion_injected_counter;
uint completion_injected_query_calls;
uint completion_injected_wait_calls;
vk::Result completion_injected_wait_result;
Atomic{uint} completion_race_arrived;
Atomic{bool} completion_race_release;
Atomic{uint} completion_retirement_arrived;
Atomic{bool} completion_retirement_release;

struct CompletionPollWorkerArgs {
    vk::VkDeviceState* state;
    uint               queue_id;
    ulong              sequence;
    bool               result;
}

struct CompletionWaitWorkerArgs {
    vk::VkDeviceState* state;
    uint               queue_id;
    ulong              sequence;
    bool               completed;
}

struct CompletionMutexHolderArgs {
    Mutex*       mutex;
    Atomic{bool} entered;
    Atomic{bool} release;
}

fn vk::Result query_injected_completion(
    vk::Device device,
    vk::Semaphore semaphore,
    ulong* value,
) {
    (void)device;
    (void)semaphore;
    completion_injected_query_calls++;
    *value = completion_injected_counter;
    return vk::Result.SUCCESS;
}

fn vk::Result wait_injected_completion(
    vk::Device device,
    vk::SemaphoreWaitInfo* wait_info,
    ulong timeout_ns,
) {
    (void)device;
    (void)wait_info;
    (void)timeout_ns;
    completion_injected_wait_calls++;
    return completion_injected_wait_result;
}

fn vk::Result query_completion_rendezvous(
    vk::Device device,
    vk::Semaphore semaphore,
    ulong* value,
) {
    (void)device;
    (void)semaphore;
    (void)completion_race_arrived.add(1, AtomicOrdering.RELEASE);
    while (completion_race_arrived.load(AtomicOrdering.ACQUIRE) < 2) {
        thread::yield();
    }
    while (!completion_race_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    *value = completion_injected_counter;
    return vk::Result.SUCCESS;
}

fn void rendezvous_before_retirement_lock(void* user_data) {
    (void)user_data;
    (void)completion_retirement_arrived.add(1, AtomicOrdering.RELEASE);
    while (!completion_retirement_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
}

fn int completion_poll_worker(void* raw_args) {
    CompletionPollWorkerArgs* args = (CompletionPollWorkerArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        args.result = vk::vk_poll_completion_with_query(
            state:    args.state,
            queue_id: args.queue_id,
            sequence: args.sequence,
            query_fn: &query_completion_rendezvous,
        )!!;
    };
    return 0;
}

fn int completion_wait_worker(void* raw_args) {
    CompletionWaitWorkerArgs* args = (CompletionWaitWorkerArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        vk::vk_wait_completion_with_wait(
            state:      args.state,
            queue_id:   args.queue_id,
            sequence:   args.sequence,
            timeout_ns: 0,
            wait_fn:    &wait_injected_completion,
        )!!;
        args.completed = true;
    };
    return 0;
}

fn int completion_mutex_holder(void* raw_args) {
    CompletionMutexHolderArgs* args =
        (CompletionMutexHolderArgs*)raw_args;
    args.mutex.lock()!!;
    args.entered.store(true, AtomicOrdering.RELEASE);
    while (!args.release.load(AtomicOrdering.ACQUIRE)) thread::yield();
    args.mutex.unlock();
    return 0;
}

fn gpu::Device create_completion_test_device() {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_completion_test",
    };
    return create_test_device(&desc)!!;
}

fn void queue_completion_publication_rejects_gaps() @test {
    gpu::Device device = create_completion_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    gpu::CompletionPoint first = vk::reserve_queue_completion(
        state,
        queue,
    )!!;
    gpu::CompletionPoint second = vk::reserve_queue_completion(
        state,
        queue,
    )!!;

    assert(@catch(vk::publish_queue_completion(state, second))
        == gpu::INVALID_HANDLE,
        "publication cannot skip an earlier reserved sequence");
    assert(@catch(gpu::poll_completion(first)) == gpu::INVALID_HANDLE,
        "the skipped sequence remains unpublished");
    assert(@catch(gpu::poll_completion(second)) == gpu::INVALID_HANDLE,
        "the later sequence remains unpublished after rejection");

    vk::publish_queue_completion(state, first)!!;
    assert(@catch(gpu::poll_completion(second)) == gpu::INVALID_HANDLE,
        "publishing an earlier point does not publish a later point");
    vk::publish_queue_completion(state, second)!!;

    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, queue.id)!!;
    vk::SemaphoreSignalInfo signal = vk::semaphore_signal_info()
        .set_semaphore(completion.timeline)
        .set_value(completion.next_sequence);
    gpu::internal::vk::check_backend_result(
        state:     state,
        operation: "signal test completion",
        result:    vk::signal_semaphore(state.device, &signal),
    )!!;
}

fn void completion_observation_advances_for_poll_and_wait() @test {
    gpu::Device device = create_completion_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, queue.id)!!;

    gpu::CompletionPoint first = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, first)!!;
    gpu::CompletionPoint second = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, second)!!;

    vk::SemaphoreSignalInfo signal = vk::semaphore_signal_info()
        .set_semaphore(completion.timeline)
        .set_value(gpu::internal::completion_point_sequence(first));
    gpu::internal::vk::check_backend_result(
        state:     state,
        operation: "signal first observed completion",
        result:    vk::signal_semaphore(state.device, &signal),
    )!!;
    assert(gpu::poll_completion(first)!!, "poll observes the first completion");
    assert(
        completion.retired_sequence.load(AtomicOrdering.ACQUIRE)
            == gpu::internal::completion_point_sequence(first),
        "poll advances the retired completion sequence"
    );

    signal = signal.set_value(gpu::internal::completion_point_sequence(second));
    gpu::internal::vk::check_backend_result(
        state:     state,
        operation: "signal second observed completion",
        result:    vk::signal_semaphore(state.device, &signal),
    )!!;
    gpu::wait_completion(second, 0)!!;
    assert(
        completion.retired_sequence.load(AtomicOrdering.ACQUIRE)
            == gpu::internal::completion_point_sequence(second),
        "wait advances the retired completion sequence"
    );
}

fn void queue_completion_poll_wait_is_reusable() @test {
    gpu::Device device = create_completion_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    gpu::CompletionPoint point = vk::reserve_queue_completion(
        state,
        queue,
    )!!;
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, queue.id)!!;

    assert(@catch(gpu::poll_completion(point)) == gpu::INVALID_HANDLE,
        "an unpublished queue sequence is not a public completion point");
    vk::publish_queue_completion(state, point)!!;
    assert(!gpu::poll_completion(point)!!,
        "a published queue sequence remains incomplete until signaled");
    assert(@catch(gpu::wait_completion(point, 0)) == gpu::WAIT_TIMEOUT,
        "a zero-timeout wait reports retryable incompletion");

    gpu::CompletionPoint future = vk::reserve_queue_completion(
        state,
        queue,
    )!!;
    assert(@catch(gpu::poll_completion(future)) == gpu::INVALID_HANDLE,
        "a later unpublished sequence is rejected");
    completion.submission_mutex.lock()!!;
    vk::cancel_queue_completion_locked(state, future)!!;
    completion.submission_mutex.unlock();

    vk::SemaphoreSignalInfo signal = vk::semaphore_signal_info()
        .set_semaphore(completion.timeline)
        .set_value(gpu::internal::completion_point_sequence(point));
    gpu::internal::vk::check_backend_result(
        state:     state,
        operation: "signal test completion",
        result:    vk::signal_semaphore(state.device, &signal),
    )!!;

    assert(gpu::poll_completion(point)!!,
        "poll observes the completed queue sequence");
    gpu::wait_completion(point, 0)!!;
    gpu::wait_completion(point, 0)!!;
}

fn void completion_cache_skips_native_and_locked_work() @test {
    gpu::Device device = create_completion_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    gpu::CompletionPoint point = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, point)!!;
    ulong sequence = gpu::internal::completion_point_sequence(point);

    completion_injected_counter = sequence;
    completion_injected_query_calls = 0;
    assert(vk::vk_poll_completion_with_query(
        state:    state,
        queue_id: queue.id,
        sequence: sequence,
        query_fn: &query_injected_completion,
    )!!, "the first observation retires the point");
    assert(completion_injected_query_calls == 1,
        "the first observation performs one native query");

    vk::reset_completion_work_counters(state);
    completion_injected_query_calls = 0;
    completion_injected_wait_calls = 0;
    completion_injected_wait_result = vk::Result.ERROR_DEVICE_LOST;
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, queue.id)!!;
    CompletionMutexHolderArgs holder_args = {
        .mutex = &completion.retirement_mutex,
    };
    Thread holder;
    holder.create(&completion_mutex_holder, &holder_args)!!;
    while (!holder_args.entered.load(AtomicOrdering.ACQUIRE)) thread::yield();
    for (uint i = 0; i < 100_000; i++) {
        assert(vk::vk_poll_completion_with_query(
            state:    state,
            queue_id: queue.id,
            sequence: sequence,
            query_fn: &query_injected_completion,
        )!!, "the cached point stays complete");
    }
    vk::vk_wait_completion_with_wait(
        state:      state,
        queue_id:   queue.id,
        sequence:   sequence,
        timeout_ns: 0,
        wait_fn:    &wait_injected_completion,
    )!!;
    holder_args.release.store(true, AtomicOrdering.RELEASE);
    assert(holder.join()!! == 0,
        "the submit-mutex holder exits after cached observations");

    vk::CompletionWorkCounters counters =
        vk::completion_work_counters(state);
    assert(completion_injected_query_calls == 0
        && completion_injected_wait_calls == 0,
        "cached observations perform no injected native work");
    assert(counters.native_counter_queries == 0
        && counters.native_wait_calls == 0
        && counters.retirement_lock_acquisitions == 0,
        "cached observations perform no native or retirement-lock work");
    assert(counters.cached_poll_hits == 100_000
        && counters.cached_wait_hits == 1,
        "every cached observation is counted exactly once");
}

fn void concurrent_cached_waits_update_atomic_stats() @test {
    gpu::Device device = create_completion_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    gpu::CompletionPoint point = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, point)!!;
    ulong sequence = gpu::internal::completion_point_sequence(point);
    completion_injected_counter = sequence;
    assert(vk::vk_poll_completion_with_query(
        state:    state,
        queue_id: queue.id,
        sequence: sequence,
        query_fn: &query_injected_completion,
    )!!, "the setup poll retires the shared wait point");

    vk::SubmitStats before = vk::submit_stats(state);
    vk::reset_completion_work_counters(state);
    completion_injected_wait_calls = 0;
    completion_injected_wait_result = vk::Result.ERROR_DEVICE_LOST;
    CompletionWaitWorkerArgs[2] args;
    Thread[2] workers;
    for (uint i = 0; i < 2; i++) {
        args[i] = {
            .state    = state,
            .queue_id = queue.id,
            .sequence = sequence,
        };
        workers[i].create(&completion_wait_worker, &args[i])!!;
    }
    foreach (i, &worker : workers) {
        assert(worker.join()!! == 0 && args[i].completed,
            "each concurrent cached wait completes");
    }

    vk::SubmitStats after = vk::submit_stats(state);
    vk::CompletionWorkCounters work = vk::completion_work_counters(state);
    assert(after.completion_wait_calls == before.completion_wait_calls + 2,
        "concurrent waits update the atomic live wait count exactly");
    assert(completion_injected_wait_calls == 0
        && work.native_wait_calls == 0
        && work.cached_wait_hits == 2,
        "concurrent cached waits perform no native wait work");
}

fn void wait_retires_exact_sequence_and_timeout_preserves_state() @test {
    gpu::Device device = create_completion_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    gpu::CompletionPoint first = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, first)!!;
    gpu::CompletionPoint second = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, second)!!;
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, queue.id)!!;

    completion_injected_wait_calls = 0;
    completion_injected_wait_result = vk::Result.SUCCESS;
    vk::vk_wait_completion_with_wait(
        state:      state,
        queue_id:   queue.id,
        sequence:   gpu::internal::completion_point_sequence(first),
        timeout_ns: 0,
        wait_fn:    &wait_injected_completion,
    )!!;
    assert(completion_injected_wait_calls == 1
        && completion.retired_sequence.load(AtomicOrdering.ACQUIRE)
            == gpu::internal::completion_point_sequence(first),
        "a successful wait retires exactly its requested sequence");

    completion_injected_wait_result = vk::Result.TIMEOUT;
    assert(@catch(vk::vk_wait_completion_with_wait(
        state:      state,
        queue_id:   queue.id,
        sequence:   gpu::internal::completion_point_sequence(second),
        timeout_ns: 0,
        wait_fn:    &wait_injected_completion,
    )) == gpu::WAIT_TIMEOUT,
        "the injected timeout keeps the public wait fault");
    assert(completion.retired_sequence.load(AtomicOrdering.ACQUIRE)
        == gpu::internal::completion_point_sequence(first),
        "a timeout cannot advance retirement");

    completion_injected_counter = gpu::internal::completion_point_sequence(second);
    assert(vk::vk_poll_completion_with_query(
        state:    state,
        queue_id: queue.id,
        sequence: gpu::internal::completion_point_sequence(second),
        query_fn: &query_injected_completion,
    )!!, "the timed-out point remains reusable");
}

fn void concurrent_first_observers_publish_one_retirement() @test {
    gpu::Device device = create_completion_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    gpu::CompletionPoint point = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, point)!!;
    ulong sequence = gpu::internal::completion_point_sequence(point);

    completion_injected_counter = sequence;
    completion_race_arrived.store(0, AtomicOrdering.RELAXED);
    completion_race_release.store(false, AtomicOrdering.RELAXED);
    completion_retirement_arrived.store(0, AtomicOrdering.RELAXED);
    completion_retirement_release.store(false, AtomicOrdering.RELAXED);
    vk::set_completion_retirement_lock_hook(
        state,
        &rendezvous_before_retirement_lock,
        null,
    );
    defer vk::set_completion_retirement_lock_hook(state, null, null);
    vk::reset_completion_work_counters(state);
    CompletionPollWorkerArgs[2] args;
    Thread[2] workers;
    for (uint i = 0; i < 2; i++) {
        args[i] = {
            .state    = state,
            .queue_id = queue.id,
            .sequence = sequence,
        };
        workers[i].create(&completion_poll_worker, &args[i])!!;
    }
    while (completion_race_arrived.load(AtomicOrdering.ACQUIRE) < 2) {
        thread::yield();
    }
    completion_race_release.store(true, AtomicOrdering.RELEASE);
    while (completion_retirement_arrived.load(AtomicOrdering.ACQUIRE) < 2) {
        thread::yield();
    }
    completion_retirement_release.store(true, AtomicOrdering.RELEASE);
    foreach (i, &worker : workers) {
        assert(worker.join()!! == 0 && args[i].result,
            "both concurrent observers report full retirement");
    }

    vk::CompletionWorkCounters counters =
        vk::completion_work_counters(state);
    assert(counters.native_counter_queries == 2,
        "both first observers reach their controlled native query");
    assert(counters.retirement_passes == 1,
        "competing observers publish retirement exactly once");
    assert(vk::queue_completion_state(state, queue.id)!!
            .retired_sequence.load(AtomicOrdering.ACQUIRE) == sequence,
        "the concurrent retired prefix is exact and monotonic");
}
