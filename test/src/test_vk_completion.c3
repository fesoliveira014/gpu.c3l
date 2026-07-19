module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;

fn gpu::Device create_completion_test_device() {
    gpu::DeviceDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_completion_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
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
        completion.observed_completed_sequence
            == gpu::completion_point_sequence(first),
        "poll advances the observed completion sequence"
    );

    signal = signal.set_value(gpu::completion_point_sequence(second));
    gpu::vk::check_backend_result(
        state:     state,
        operation: "signal second observed completion",
        result:    vk::signal_semaphore(state.device, &signal),
    )!!;
    gpu::wait_completion(second, 0)!!;
    assert(
        completion.observed_completed_sequence
            == gpu::completion_point_sequence(second),
        "wait advances the observed completion sequence"
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
