module gpu_test;

import gpu::internal @public;
import gpu::internal::vk @public;
import gpu @public;
import std::atomic;
import std::thread;
import vk;

const ulong LOSS_TEST_WAIT_NS = 1_000_000_000;

struct LossDiagnosticCapture {
    Atomic{uint} loss_messages;
}

struct LossReportArgs {
    gpu::Device        device;
    vk::VkDeviceState* state;
    Atomic{bool}*      start;
}

fn void capture_loss_diagnostic(gpu::DebugMessage* message, void* user_data) {
    LossDiagnosticCapture* capture = (LossDiagnosticCapture*)user_data;
    if (message.operation != null
        && message.operation.str_view() == "loss_test"
        && message.backend_text != null
        && message.backend_text.str_view() == "Vulkan device lost"
        && !message.has_fault) {
        capture.loss_messages.add(1);
    }
}

fn int report_loss_worker(void* raw_args) {
    LossReportArgs* args = (LossReportArgs*)raw_args;
    while (!args.start.load(AtomicOrdering.ACQUIRE)) thread::yield();
    (void)vk::absorb_device_loss(args.state, "loss_test", vk::Result.ERROR_DEVICE_LOST);
    return gpu::is_device_lost(&args.device) ? 0 : 1;
}

fn vk::Result lose_loss_test_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)submit_count;
    (void)submits;
    (void)fence;
    return vk::Result.ERROR_DEVICE_LOST;
}

<* Accept a submission without handing it to the driver, so its completion
   never signals on a healthy device. *>
fn vk::Result drop_loss_test_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)submit_count;
    (void)submits;
    (void)fence;
    return vk::Result.SUCCESS;
}

fn gpu::Device loss_test_device() {
    gpu::RuntimeDesc desc = { .application_name = "gpu_device_loss_test" };
    return create_test_device(&desc)!!;
}

fn vk::VkDeviceState* loss_test_state(gpu::Device* device) {
    return gpu::internal::device_slots[device.index].data.state;
}

fn gpu::ExecutableCommandList record_loss_test_commands(gpu::CommandAllocator* allocator) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::Barrier barrier = { .before = { .compute }, .after = { .compute } };
    gpu::cmd_barrier(&commands, &barrier)!!;
    return gpu::end_commands(&commands)!!;
}

<* Submit through the private path with a native submit that never runs. *>
fn gpu::CompletionPoint submit_never_completing(
    gpu::Device* device,
    gpu::Queue queue,
    gpu::CommandAllocator* allocator,
) {
    gpu::ExecutableCommandList[1] lists = { record_loss_test_commands(allocator) };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    return vk::vk_submit_with(loss_test_state(device), queue, &desc, &drop_loss_test_submit)!!;
}

fn void healthy_device_is_not_lost() @test {
    gpu::Device device = loss_test_device();
    defer destroy_test_device(&device)!!;
    assert(!gpu::is_device_lost(&device), "a healthy device reports no loss");
}

fn void observed_loss_is_sticky() @test {
    gpu::Device device = loss_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::ExecutableCommandList[1] lists = { record_loss_test_commands(&allocator) };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };

    gpu::CompletionPoint point = vk::vk_submit_with(
        loss_test_state(&device),
        queue,
        &desc,
        &lose_loss_test_submit,
    )!!;

    assert(gpu::poll_completion(point)!!, "a submit that observed loss is already complete");
    assert(gpu::is_device_lost(&device), "the submit marked the device lost");
    assert(gpu::is_device_lost(&device), "loss stays observed");
}

fn void peer_device_unaffected() @test {
    gpu::Device lost = loss_test_device();
    defer destroy_test_device(&lost)!!;
    gpu::Device peer = loss_test_device();
    defer destroy_test_device(&peer)!!;
    gpu::internal::mark_device_lost(&lost);

    assert(gpu::is_device_lost(&lost), "the marked device is lost");
    assert(!gpu::is_device_lost(&peer), "a peer device stays healthy");
    gpu::Queue queue = gpu::get_queue(&peer, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&peer, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::ExecutableCommandList[1] lists = { record_loss_test_commands(&allocator) };
    gpu::SubmitDesc desc = { .command_lists = lists[..] };
    gpu::wait_completion(gpu::submit(queue, &desc)!!, LOSS_TEST_WAIT_NS)!!;
}

fn void wait_returns_after_loss() @test {
    gpu::Device device = loss_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CompletionPoint point = submit_never_completing(&device, queue, &allocator);
    assert(!gpu::poll_completion(point)!!, "a dropped submission never completes on a healthy device");

    gpu::internal::mark_device_lost(&device);
    gpu::wait_completion(point, LOSS_TEST_WAIT_NS)!!;
    assert(gpu::poll_completion(point)!!, "loss completes every submitted point");
}

fn void pending_units_recycle_after_loss() @test {
    gpu::Device device = loss_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocatorDesc allocator_desc = { .command_buffer_capacity = 2 };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue, &allocator_desc)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    (void)submit_never_completing(&device, queue, &allocator);
    (void)submit_never_completing(&device, queue, &allocator);
    assert(@catch(gpu::begin_commands(&allocator)) == gpu::DEVICE_BUSY,
        "every unit is in flight on a healthy device");

    gpu::internal::mark_device_lost(&device);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::discard_commands(&commands)!!;
}

fn void frame_loop_without_presentation_after_loss() @test {
    gpu::Device device = loss_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::internal::mark_device_lost(&device);

    for (uint frame = 0; frame < 3; frame++) {
        gpu::ExecutableCommandList[1] lists = { record_loss_test_commands(&allocator) };
        gpu::SubmitDesc desc = { .command_lists = lists[..] };
        gpu::CompletionPoint point = gpu::submit(queue, &desc)!!;
        assert(gpu::poll_completion(point)!!, "a submit after loss is already complete");
        gpu::wait_completion(point)!!;
    }
}

fn void creation_after_loss_proceeds() @test {
    gpu::Device device = loss_test_device();
    defer destroy_test_device(&device)!!;
    gpu::internal::mark_device_lost(&device);
    gpu::TextureDesc desc = {
        .access = { .graphics, .compute },
        .width  = 16,
        .height = 16,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void normal_teardown_after_loss() @test {
    gpu::Device device = loss_test_device();
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS);
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    gpu::AllocationDesc allocation_desc = {
        .size         = 256,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .graphics, .compute, .transfer },
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(&device, &allocation_desc)!!;
    gpu::TextureDesc texture_desc = {
        .access = { .graphics, .compute },
        .width  = 16,
        .height = 16,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &texture_desc)!!;
    gpu::CompletionPoint last = submit_never_completing(&device, queue, &allocator);

    gpu::internal::mark_device_lost(&device);
    gpu::wait_completion(last, LOSS_TEST_WAIT_NS)!!;
    gpu::destroy_texture(&device, texture)!!;
    gpu::free_allocation(&device, &allocation)!!;
    gpu::destroy_command_allocator(&allocator)!!;
    destroy_test_device(&device)!!;
    assert(!device.is_valid(), "the lost device is destroyed");
}

fn void live_child_blocks_lost_device_destroy() @test {
    gpu::Device device = loss_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .access = { .graphics, .compute },
        .width  = 16,
        .height = 16,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    gpu::internal::mark_device_lost(&device);

    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE,
        "a live child blocks destroying a lost device");
    gpu::destroy_texture(&device, texture)!!;
}

fn void concurrent_loss_reports_once() @test {
    LossDiagnosticCapture capture;
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_device_loss_report_test",
        .debug_callback   = &capture_loss_diagnostic,
        .debug_user_data  = &capture,
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    Atomic{bool} start;
    LossReportArgs args = {
        .device = device,
        .state  = loss_test_state(&device),
        .start  = &start,
    };
    Thread[2] workers;
    foreach (&worker : workers) worker.create(&report_loss_worker, &args)!!;
    start.store(true, AtomicOrdering.RELEASE);
    foreach (&worker : workers) {
        assert(worker.join()!! == 0, "each observing thread sees the device lost");
    }

    assert(capture.loss_messages.load(AtomicOrdering.ACQUIRE) == 1,
        "concurrent loss observations deliver one diagnostic");
    assert(gpu::is_device_lost(&device), "both observations leave the device lost");
}
