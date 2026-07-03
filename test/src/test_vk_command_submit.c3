module gpu_test;

import gpu;
import gpu::vk;

// Headless (lavapipe): record a buffer copy from a host-written upload buffer to
// a readback buffer, submit signaling a timeline semaphore, wait, and verify the
// bytes. Assumes host-coherent memory (non-coherent flush/invalidate is later).

fn void copy_upload_to_readback() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_command_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc up_desc = {
        .size = 256, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.STAGING, .debug_name = "upload",
    };
    gpu::BufferHandle upload = gpu::create_buffer(&device, &up_desc)!!;
    defer gpu::destroy_buffer(&device, upload)!!;

    gpu::BufferDesc rb_desc = {
        .size = 256, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK, .debug_name = "readback",
    };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, readback)!!;

    gpu::GpuSpan up_span = gpu::get_buffer_span(&device, upload)!!;
    char* src = (char*)up_span.cpu;
    for (uint i = 0; i < 256; i++) src[i] = (char)i;

    gpu::SemaphoreDesc sem_desc = { .initial_value = (gpu::SemaphoreValue)0 };
    gpu::SemaphoreHandle sem = gpu::create_semaphore(&device, &sem_desc)!!;
    defer gpu::destroy_semaphore(&device, sem)!!;

    gpu::begin_frame(&device)!!;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::BufferCopyDesc copy = { .src = upload, .dst = readback, .src_offset = 0, .dst_offset = 0, .size = 256 };
    gpu::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::BufferBarrier bar = {
        .buffer = readback, .offset = 0, .size = 256,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &bar)!!;
    gpu::end_commands(&device, &cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SemaphoreSignal[1] sigs = { { .semaphore = sem, .value = (gpu::SemaphoreValue)1, .stage = gpu::Stage.TRANSFER } };
    gpu::SubmitDesc submit = { .command_lists = lists[..], .signals = sigs[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_semaphore(
        device:     &device,
        semaphore:  sem,
        value:      (gpu::SemaphoreValue)1,
        timeout_ns: 1_000_000_000,
    )!!;

    gpu::GpuSpan rb_span = gpu::get_buffer_span(&device, readback)!!;
    char* dst = (char*)rb_span.cpu;
    for (uint i = 0; i < 256; i++) assert(dst[i] == (char)i, "readback byte matches uploaded byte");

    gpu::end_frame(&device)!!;
}

fn void empty_command_list_submits() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&device, &cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&device)!!;
}

fn void bad_command_state_rejected() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&device, &cmd)!!;

    // Recording after end must be rejected.
    gpu::GlobalBarrier gb = {
        .before_stage = gpu::Stage.HOST, .after_stage = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.HOST_WRITE, .after_hazard = gpu::Hazard.TRANSFER_READ,
    };
    fault rejected = @catch(gpu::cmd_global_barrier(&cmd, &gb));
    assert(rejected == gpu::COMMAND_RECORDING_ERROR, "recording after end returns COMMAND_RECORDING_ERROR");
    gpu::end_frame(&device)!!;
}

fn void texture_barrier_faults() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::TextureBarrier bar = {
        .texture = gpu::TEXTURE_HANDLE_INVALID,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.FRAGMENT_SHADER,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
        .old_layout = gpu::TextureLayout.TRANSFER_DST, .new_layout = gpu::TextureLayout.SHADER_READ,
    };
    fault barrier_fault = @catch(gpu::cmd_texture_barrier(&cmd, &bar));
    assert(barrier_fault == gpu::INVALID_HANDLE, "barrier on an invalid texture handle faults INVALID_HANDLE");
    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}
