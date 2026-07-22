module gpu_test;

import gpu @public;
import gpu::vk @public;
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

struct CompletionPollWorkerArgs {
    gpu::Device* device;
    uint         queue_id;
    ulong        sequence;
    bool         result;
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

fn int completion_poll_worker(void* raw_args) {
    CompletionPollWorkerArgs* args = (CompletionPollWorkerArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        args.result = vk::vk_poll_completion_with_query(
            device:   args.device,
            queue_id: args.queue_id,
            sequence: args.sequence,
            query_fn: &query_completion_rendezvous,
        )!!;
    };
    return 0;
}

fn gpu::Device create_completion_test_device() {
    gpu::RuntimeDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
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
        (vk::VkDeviceState*)gpu::device_backend_state(device);
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
    gpu::vk::check_backend_result(
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
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, queue.id)!!;

    gpu::CompletionPoint first = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, first)!!;
    gpu::CompletionPoint second = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, second)!!;

    vk::SemaphoreSignalInfo signal = vk::semaphore_signal_info()
        .set_semaphore(completion.timeline)
        .set_value(gpu::completion_point_sequence(first));
    gpu::vk::check_backend_result(
        state:     state,
        operation: "signal first observed completion",
        result:    vk::signal_semaphore(state.device, &signal),
    )!!;
    assert(gpu::poll_completion(first)!!, "poll observes the first completion");
    assert(
        completion.retired_sequence.load(AtomicOrdering.ACQUIRE)
            == gpu::completion_point_sequence(first),
        "poll advances the retired completion sequence"
    );

    signal = signal.set_value(gpu::completion_point_sequence(second));
    gpu::vk::check_backend_result(
        state:     state,
        operation: "signal second observed completion",
        result:    vk::signal_semaphore(state.device, &signal),
    )!!;
    gpu::wait_completion(second, 0)!!;
    assert(
        completion.retired_sequence.load(AtomicOrdering.ACQUIRE)
            == gpu::completion_point_sequence(second),
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
        (vk::VkDeviceState*)gpu::device_backend_state(device);
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
    completion.submit_mutex.lock()!!;
    vk::cancel_queue_completion_locked(state, future)!!;
    completion.submit_mutex.unlock();

    vk::SemaphoreSignalInfo signal = vk::semaphore_signal_info()
        .set_semaphore(completion.timeline)
        .set_value(gpu::completion_point_sequence(point));
    gpu::vk::check_backend_result(
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
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    gpu::CompletionPoint point = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, point)!!;
    ulong sequence = gpu::completion_point_sequence(point);

    completion_injected_counter = sequence;
    completion_injected_query_calls = 0;
    assert(vk::vk_poll_completion_with_query(
        device:   &device,
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
    completion.submit_mutex.lock()!!;
    for (uint i = 0; i < 100_000; i++) {
        assert(vk::vk_poll_completion_with_query(
            device:   &device,
            queue_id: queue.id,
            sequence: sequence,
            query_fn: &query_injected_completion,
        )!!, "the cached point stays complete");
    }
    vk::vk_wait_completion_with_wait(
        device:     &device,
        queue_id:   queue.id,
        sequence:   sequence,
        timeout_ns: 0,
        wait_fn:    &wait_injected_completion,
    )!!;
    completion.submit_mutex.unlock();

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

fn void wait_retires_exact_sequence_and_timeout_preserves_state() @test {
    gpu::Device device = create_completion_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    gpu::CompletionPoint first = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, first)!!;
    gpu::CompletionPoint second = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, second)!!;
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, queue.id)!!;

    completion_injected_wait_calls = 0;
    completion_injected_wait_result = vk::Result.SUCCESS;
    vk::vk_wait_completion_with_wait(
        device:     &device,
        queue_id:   queue.id,
        sequence:   gpu::completion_point_sequence(first),
        timeout_ns: 0,
        wait_fn:    &wait_injected_completion,
    )!!;
    assert(completion_injected_wait_calls == 1
        && completion.retired_sequence.load(AtomicOrdering.ACQUIRE)
            == gpu::completion_point_sequence(first),
        "a successful wait retires exactly its requested sequence");

    completion_injected_wait_result = vk::Result.TIMEOUT;
    assert(@catch(vk::vk_wait_completion_with_wait(
        device:     &device,
        queue_id:   queue.id,
        sequence:   gpu::completion_point_sequence(second),
        timeout_ns: 0,
        wait_fn:    &wait_injected_completion,
    )) == gpu::WAIT_TIMEOUT,
        "the injected timeout keeps the public wait fault");
    assert(completion.retired_sequence.load(AtomicOrdering.ACQUIRE)
        == gpu::completion_point_sequence(first),
        "a timeout cannot advance retirement");

    completion_injected_counter = gpu::completion_point_sequence(second);
    assert(vk::vk_poll_completion_with_query(
        device:   &device,
        queue_id: queue.id,
        sequence: gpu::completion_point_sequence(second),
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
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    gpu::CompletionPoint point = vk::reserve_queue_completion(state, queue)!!;
    vk::publish_queue_completion(state, point)!!;
    ulong sequence = gpu::completion_point_sequence(point);

    completion_injected_counter = sequence;
    completion_race_arrived.store(0, AtomicOrdering.RELAXED);
    completion_race_release.store(false, AtomicOrdering.RELAXED);
    vk::reset_completion_work_counters(state);
    CompletionPollWorkerArgs[2] args;
    Thread[2] workers;
    for (uint i = 0; i < 2; i++) {
        args[i] = {
            .device   = &device,
            .queue_id = queue.id,
            .sequence = sequence,
        };
        workers[i].create(&completion_poll_worker, &args[i])!!;
    }
    while (completion_race_arrived.load(AtomicOrdering.ACQUIRE) < 2) {
        thread::yield();
    }
    completion_race_release.store(true, AtomicOrdering.RELEASE);
    foreach (i, &worker : workers) {
        assert(worker.join()!! == 0 && args[i].result,
            "both concurrent observers report full retirement");
    }

    vk::CompletionWorkCounters counters =
        vk::completion_work_counters(state);
    assert(counters.native_counter_queries == 2,
        "both first observers reach their controlled native query");
    assert(counters.retirement_passes == 1
        && counters.retirement_lock_acquisitions == 1,
        "only one observer publishes the retirement boundary");
    assert(vk::queue_completion_state(state, queue.id)!!
            .retired_sequence.load(AtomicOrdering.ACQUIRE) == sequence,
        "the concurrent retired prefix is exact and monotonic");
}
