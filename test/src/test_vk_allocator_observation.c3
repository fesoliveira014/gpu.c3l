module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import std::core::mem;
import std::core::mem::alloc;

faultdef CONFINED_THREAD_CALLBACK_FAILED;

struct WarmCommandContext {
    gpu::CommandAllocator* allocator;
}

fn gpu::Device make_allocator_observation_device() {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_allocator_observation",
    };
    return create_test_device(&desc)!!;
}

fn vk::RecordingHostAllocatorSnapshot allocator_delta(
    vk::RecordingHostAllocatorSnapshot before,
    vk::RecordingHostAllocatorSnapshot after,
) {
    return {
        .allocations = after.allocations - before.allocations,
        .resizes     = after.resizes - before.resizes,
        .releases    = after.releases - before.releases,
    };
}

fn void assert_allocator_delta(
    vk::RecordingHostAllocatorSnapshot before,
    vk::RecordingHostAllocatorSnapshot after,
    ulong allocations,
    ulong resizes,
    ulong releases,
) {
    vk::RecordingHostAllocatorSnapshot delta = allocator_delta(before, after);
    assert(delta.allocations == allocations
        && delta.resizes == resizes
        && delta.releases == releases,
        "allocator operations match the exact expected delta");
}

fn void hidden_thread_allocator_work() {
    void* pointer = mem::calloc(32);
    pointer = mem::realloc(pointer, 64);
    mem::free(pointer);
}

fn void? run_hidden_thread_allocator_work(void* user_data) {
    (void)user_data;
    hidden_thread_allocator_work();
}

fn void? fail_confined_thread_callback(void* user_data) {
    (void)user_data;
    return CONFINED_THREAD_CALLBACK_FAILED~;
}

fn void? record_warm_command_cycle(void* user_data) {
    WarmCommandContext* context = (WarmCommandContext*)user_data;
    gpu::CommandList commands = gpu::begin_commands(context.allocator)!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!;
    gpu::discard_executable_commands(&executable)!;
}

fn void state_allocator_reports_acquire_resize_and_release_exactly() @test {
    gpu::Device device = make_allocator_observation_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::internal::device_backend_state(device);
    vk::RecordingHostAllocatorSnapshot before =
        vk::recording_host_allocator_snapshot(state);
    gpu::internal::RecordingWorkCounters work_before =
        vk::command_recording_work_counters(state);

    void* pointer = state.host_allocator.acquire(
        32,
        AllocInitType.NO_ZERO,
        0,
    )!!;
    pointer = state.host_allocator.resize(pointer, 64, 0)!!;
    state.host_allocator.release(pointer, false);

    vk::RecordingHostAllocatorSnapshot after =
        vk::recording_host_allocator_snapshot(state);
    gpu::internal::RecordingWorkCounters work_after =
        vk::command_recording_work_counters(state);
    assert_allocator_delta(
        before:      before,
        after:       after,
        allocations: 1,
        resizes:     1,
        releases:    1,
    );
    assert(work_after.host_allocations == work_before.host_allocations + 2,
        "recording work counts acquire plus resize operations");
}

fn void confined_thread_allocator_reports_hidden_work_and_restores() @test {
    gpu::Device device = make_allocator_observation_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::internal::device_backend_state(device);
    Allocator previous = mem;
    vk::RecordingHostAllocatorSnapshot before =
        vk::recording_host_allocator_snapshot(state);

    observe_confined_thread_allocator(
        &state.recording_host_allocator,
        &run_hidden_thread_allocator_work,
        null,
    )!!;

    vk::RecordingHostAllocatorSnapshot after =
        vk::recording_host_allocator_snapshot(state);
    assert_allocator_delta(
        before:      before,
        after:       after,
        allocations: 1,
        resizes:     1,
        releases:    1,
    );
    assert(mem.ptr == previous.ptr && mem.type == previous.type,
        "successful observation restores the confined thread allocator");
}

fn void confined_thread_allocator_restores_after_named_fault() @test {
    gpu::Device device = make_allocator_observation_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::internal::device_backend_state(device);
    Allocator previous = mem;
    vk::RecordingHostAllocatorSnapshot before =
        vk::recording_host_allocator_snapshot(state);

    fault callback_fault = @catch(observe_confined_thread_allocator(
        &state.recording_host_allocator,
        &fail_confined_thread_callback,
        null,
    ));

    vk::RecordingHostAllocatorSnapshot after =
        vk::recording_host_allocator_snapshot(state);
    assert(callback_fault == CONFINED_THREAD_CALLBACK_FAILED,
        "the scoped observer preserves the callback fault");
    assert_allocator_delta(
        before:      before,
        after:       after,
        allocations: 0,
        resizes:     0,
        releases:    0,
    );
    assert(mem.ptr == previous.ptr && mem.type == previous.type,
        "faulting observation restores the confined thread allocator");
}

fn void warm_command_cycle_has_zero_allocator_delta() @test {
    gpu::Device device = make_allocator_observation_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    WarmCommandContext context = { .allocator = &allocator };
    record_warm_command_cycle(&context)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::internal::device_backend_state(device);
    vk::RecordingHostAllocatorSnapshot before =
        vk::recording_host_allocator_snapshot(state);

    observe_confined_thread_allocator(
        &state.recording_host_allocator,
        &record_warm_command_cycle,
        &context,
    )!!;

    vk::RecordingHostAllocatorSnapshot after =
        vk::recording_host_allocator_snapshot(state);
    assert_allocator_delta(
        before:      before,
        after:       after,
        allocations: 0,
        resizes:     0,
        releases:    0,
    );
}

fn void recording_wrapper_remains_owned_until_device_teardown() @test {
    Allocator previous = mem;
    gpu::Device device = make_allocator_observation_device();
    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::internal::device_backend_state(device);
    assert(state.host_allocator.ptr == &state.recording_host_allocator
        && state.host_allocator.type
            == vk::RecordingHostAllocator::typeid,
        "device state owns its recording allocator wrapper");

    observe_confined_thread_allocator(
        &state.recording_host_allocator,
        &run_hidden_thread_allocator_work,
        null,
    )!!;
    assert(mem.ptr == previous.ptr && mem.type == previous.type,
        "teardown starts after confined allocator restoration");
    destroy_test_device(&device)!!;
    assert(mem.ptr == previous.ptr && mem.type == previous.type,
        "device teardown leaves the caller thread allocator unchanged");
}
