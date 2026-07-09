module gpu_test;

import gpu;
import gpu::vk;
import vk;

// Headless (lavapipe): aliased queues mean the piggyback branch never fires
// here; what stays testable is one-empty-submit-per-frame and rejected
// submits recording nothing.

fn gpu::Device make_submit_stats_test_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_submit_stats_test",
    };
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

    gpu::begin_frame(&device)!!;
    gpu::QueueKind[3] kinds = { gpu::QueueKind.GRAPHICS, gpu::QueueKind.COMPUTE, gpu::QueueKind.TRANSFER };
    foreach (kind : kinds) {
        gpu::CommandList cmd = gpu::begin_commands(&device, kind)!!;
        gpu::end_commands(&device, &cmd)!!;
        gpu::CommandList[1] one = { cmd };
        gpu::SubmitDesc submit = { .command_lists = one[..] };
        gpu::submit(&device, &submit)!!;
    }

    vk::SubmitStats before = stats_of(&device);
    gpu::end_frame(&device)!!;
    vk::SubmitStats after = stats_of(&device);
    assert(after.empty_signal_submits - before.empty_signal_submits == 1, "a frame using every queue kind still ends with exactly one internal empty submit");
    assert(after.queue_submits == before.queue_submits, "end_frame's own signal is not counted as a user queue submit");
}

<* Same invariant on a graphics-only frame — the spec's baseline scenario. *>
fn void graphics_only_frame_emits_one_empty_submit() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };

    vk::SubmitStats before_submit = stats_of(&device);
    gpu::submit(&device, &submit)!!;
    vk::SubmitStats after_submit = stats_of(&device);
    assert(after_submit.queue_submits - before_submit.queue_submits == 1, "one user submit records one queue submit");

    vk::SubmitStats before_end = stats_of(&device);
    gpu::end_frame(&device)!!;
    vk::SubmitStats after_end = stats_of(&device);
    assert(after_end.empty_signal_submits - before_end.empty_signal_submits == 1, "a graphics-only frame issues exactly one internal empty submit");
}

<* A submit rejected before it ever reaches the driver must record nothing,
   and a subsequent end_frame must still complete without waiting on any
   value the rejected submit would have signaled. *>
fn void failed_submit_records_nothing_and_end_frame_stays_clean() @test {
    gpu::Device device = make_submit_stats_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList unfinished = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandList[1] not_executable = { unfinished };
    gpu::SubmitDesc bad_submit = { .command_lists = not_executable[..] };

    vk::SubmitStats before = stats_of(&device);
    fault f = @catch(gpu::submit(&device, &bad_submit));
    assert(f == gpu::COMMAND_RECORDING_ERROR, "a not-yet-executable list rejects before touching the queue");
    vk::SubmitStats after = stats_of(&device);
    assert(after.queue_submits == before.queue_submits, "a rejected submit records no queue submit");

    gpu::end_commands(&device, &unfinished)!!;
    gpu::end_frame(&device)!!;

    // A clean cycle right after proves end_frame never waited a bogus value.
    gpu::begin_frame(&device)!!;
    gpu::end_frame(&device)!!;
}
