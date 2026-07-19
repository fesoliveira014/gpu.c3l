module gpu_test;

import gpu @public;
import gpu::vk @public;
import std::atomic;


fn ulong command_submit_count(gpu::Device* device) {
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(*device);
    return vk::submit_stats(state).queue_submits;
}


fn void command_tokens_are_small_owner_values() @test {
    assert(gpu::CommandList::size <= 24,
        "recording token stays within three machine words");
    assert(gpu::ExecutableCommandList::size <= 24,
        "executable token stays within three machine words");
}

fn void discard_consumes_recording_and_executable_lists() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;

    gpu::CommandList recording = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&recording)!!;
    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE,
        "a recording command list blocks device destruction");
    gpu::discard_commands(&recording)!!;
    assert(!recording.handle.is_valid(), "discard consumes a recording list");

    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE,
        "an executable command list blocks device destruction");
    gpu::ExecutableCommandList stale = executable;
    gpu::discard_executable_commands(&executable)!!;
    assert(!executable.handle.is_valid(), "discard consumes an executable list");
    assert(@catch(gpu::discard_executable_commands(&stale)) == gpu::INVALID_HANDLE,
        "a consumed executable-list alias is stale");

    gpu::destroy_device(&device)!!;
}

fn void copy_upload_to_readback() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_command_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    OwnedSpan upload = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "upload",
    )!!;
    defer free_owned_span(&device, &upload)!!;
    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         256,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;

    gpu::GpuSpan up_span = upload.span;
    gpu::GpuSpan rb_span = readback.span;
    char[] up_mapping = gpu::get_span_mapping(&device, up_span)!!;
    char* src = up_mapping.ptr;
    for (uint i = 0; i < 256; i++) src[i] = (char)i;
    gpu::flush_mapped_span(&device, upload.span)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::BufferCopyDesc copy = { .src = up_span, .dst = rb_span };
    gpu::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::BufferBarrier bar = {
        .span = rb_span,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &bar)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;

    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.TRANSFER,
        &submit,
    )!!;
    gpu::wait_completion(completion, 1_000_000_000)!!;

    gpu::invalidate_mapped_span(&device, readback.span)!!;
    char[] rb_mapping = gpu::get_span_mapping(&device, rb_span)!!;
    char* dst = rb_mapping.ptr;
    for (uint i = 0; i < 256; i++) assert(dst[i] == (char)i, "readback byte matches uploaded byte");

    gpu::end_frame(&frame)!!;
}

fn void empty_command_list_submits() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;

    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame)!!;
}

fn void command_list_owns_its_device_token_value() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::Device temporary = device;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&temporary, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    temporary = {};
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;

    gpu::ExecutableCommandList[1] lists = { commands_executable };
    gpu::SubmitDesc submit_desc = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit_desc)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void stale_command_owner_faults_invalid_handle() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&commands_executable)!!;
    gpu::ExecutableCommandList live_commands = commands_executable;
    commands_executable.device = {};

    gpu::ExecutableCommandList[1] stale_lists = { commands_executable };
    gpu::SubmitDesc stale_submit = { .command_lists = stale_lists[..] };
    fault rejected = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &stale_submit));
    assert(rejected == gpu::INVALID_HANDLE, "stale command owner faults INVALID_HANDLE");

    gpu::ExecutableCommandList[1] live_lists = { live_commands };
    gpu::SubmitDesc live_submit = { .command_lists = live_lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &live_submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void copied_alias_cannot_resubmit_after_success() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::CommandList original = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&original)!!;
    gpu::ExecutableCommandList original_executable = gpu::end_commands(&original)!!;
    defer discard_test_executable(&original_executable)!!;
    gpu::ExecutableCommandList[1] copied = { original_executable };
    gpu::SubmitDesc first = { .command_lists = copied[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &first)!!;
    gpu::CommandList replacement = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&replacement)!!;
    gpu::ExecutableCommandList replacement_executable = gpu::end_commands(&replacement)!!;
    defer discard_test_executable(&replacement_executable)!!;
    assert(replacement_executable.handle.index == original_executable.handle.index,
        "the freed command slot is reused");
    assert(replacement_executable.handle.generation != original_executable.handle.generation,
        "slot reuse advances the command generation");


    ulong before = command_submit_count(&device);
    gpu::ExecutableCommandList[1] original_alias = { original_executable };
    gpu::SubmitDesc second = { .command_lists = original_alias[..] };
    fault rejected = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &second));
    assert(rejected == gpu::INVALID_HANDLE, "a consumed alias faults INVALID_HANDLE");
    assert(command_submit_count(&device) == before, "a consumed alias reaches no Vulkan queue submit");
    gpu::ExecutableCommandList[1] replacement_lists = { replacement_executable };
    gpu::SubmitDesc replacement_submit = { .command_lists = replacement_lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &replacement_submit)!!;

    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void command_generation_exhaustion_retires_slot() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::CommandList final_generation = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&final_generation)!!;
    gpu::ExecutableCommandList final_generation_executable = gpu::end_commands(&final_generation)!!;
    defer discard_test_executable(&final_generation_executable)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    state.commands.slots[final_generation_executable.handle.index].generation =
        vk::HANDLE_GENERATION_MAX - 1;
    final_generation_executable.handle.generation = vk::HANDLE_GENERATION_MAX - 1;
    gpu::device_slots[device.index].command_leases[
        final_generation_executable.handle.index
    ].store(
        final_generation_executable.handle.generation,
        AtomicOrdering.RELEASE,
    );

    gpu::ExecutableCommandList[1] first_lists = { final_generation_executable };
    gpu::SubmitDesc first_submit = { .command_lists = first_lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &first_submit)!!;

    gpu::CommandList next = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&next)!!;
    assert(next.handle.index != final_generation_executable.handle.index,
        "exhausted command slot is retired");
    assert(next.handle.generation == 1,
        "replacement uses a different slot's first generation");
    gpu::ExecutableCommandList next_executable = gpu::end_commands(&next)!!;
    defer discard_test_executable(&next_executable)!!;
    gpu::ExecutableCommandList[1] next_lists = { next_executable };
    gpu::SubmitDesc next_submit = { .command_lists = next_lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &next_submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void duplicate_token_batch_rejects_without_consuming() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[2] duplicates = { cmd_executable, cmd_executable };
    gpu::SubmitDesc bad = { .command_lists = duplicates[..] };
    ulong before = command_submit_count(&device);
    fault rejected = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &bad));
    assert(rejected == gpu::COMMAND_RECORDING_ERROR, "duplicate tokens fault before submission");
    assert(command_submit_count(&device) == before, "duplicate-token rejection records no queue submit");

    gpu::ExecutableCommandList[1] one = { cmd_executable };
    gpu::SubmitDesc good = { .command_lists = one[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &good)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void readiness_only_submit_validates_stale_swapchain() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    gpu::SwapchainHandle stale = {
        .owner      = gpu::resource_owner_tag(state.resource_owner, gpu::ResourceOwnerKind.SWAPCHAIN),
        .generation = 1,
    };
    gpu::SwapchainReadiness readiness = gpu::vk::make_swapchain_readiness(
        device,
        stale,
        1,
    )!!;
    gpu::SubmitDesc submit = { .readiness = readiness };

    ulong before = command_submit_count(&device);
    fault rejected = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit));
    assert(rejected == gpu::INVALID_HANDLE, "stale readiness faults INVALID_HANDLE");
    assert(command_submit_count(&device) == before, "stale readiness records no queue submit");
}

fn void readiness_only_submit_rejects_malformed_identity() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::SwapchainReadiness malformed = {
        .device  = device,
        .payload = 1,
    };
    gpu::SubmitDesc submit = { .readiness = malformed };

    ulong before = command_submit_count(&device);
    fault rejected = @catch(submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit));
    assert(rejected == gpu::INVALID_HANDLE, "malformed readiness faults INVALID_HANDLE");
    assert(command_submit_count(&device) == before, "malformed readiness records no queue submit");
}

fn void simultaneous_devices_submit_independently() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device first = gpu::create_device_from_desc(&desc)!!;
    gpu::Device second = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&second)!!;

    gpu::CommandList first_cmd = gpu::begin_commands(gpu::get_queue(&first, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&first_cmd)!!;
    gpu::ExecutableCommandList first_cmd_executable = gpu::end_commands(&first_cmd)!!;
    defer discard_test_executable(&first_cmd_executable)!!;
    gpu::ExecutableCommandList[1] first_lists = { first_cmd_executable };
    gpu::SubmitDesc first_submit = { .command_lists = first_lists[..] };
    submit_test_queue(&first, gpu::QueueKind.GRAPHICS, &first_submit)!!;
    wait_test_queue(&first, gpu::QueueKind.GRAPHICS)!!;
    gpu::destroy_device(&first)!!;

    assert(gpu::get_device_caps(&second)!!.strict_enabled, "destroying one device leaves the other live");
    gpu::CommandList second_cmd = gpu::begin_commands(gpu::get_queue(&second, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&second_cmd)!!;
    gpu::ExecutableCommandList second_cmd_executable = gpu::end_commands(&second_cmd)!!;
    defer discard_test_executable(&second_cmd_executable)!!;
    gpu::ExecutableCommandList[1] second_lists = { second_cmd_executable };
    gpu::SubmitDesc second_submit = { .command_lists = second_lists[..] };
    submit_test_queue(&second, gpu::QueueKind.GRAPHICS, &second_submit)!!;
    wait_test_queue(&second, gpu::QueueKind.GRAPHICS)!!;
}

fn void discarded_command_buffer_is_reclaimed_by_recording_owner() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::CommandBufferStats before = vk::command_buffer_stats(state);
    gpu::CommandList abandoned = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&abandoned)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&abandoned)!!;
    defer discard_test_executable(&executable)!!;
    gpu::discard_executable_commands(&executable)!!;

    vk::CommandBufferStats discarded = vk::command_buffer_stats(state);
    assert(discarded.allocated == before.allocated + 1, "one native command buffer was allocated");
    assert(discarded.freed == before.freed, "discard leaves reclamation to the recording owner");
    assert(discarded.pending == before.pending, "discard does not create completion-tracked work");
    fault stale = @catch(gpu::discard_executable_commands(&executable));
    assert(stale == gpu::INVALID_HANDLE, "discard invalidates the executable token");

    gpu::CommandList next = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&next)!!;
    vk::CommandBufferStats reclaimed = vk::command_buffer_stats(state);
    assert(reclaimed.freed == before.freed + 1,
        "the recording owner reclaims the discarded command buffer before allocating again");
}

fn void bad_command_state_rejected() @test {
    gpu::DeviceDesc desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .frames_in_flight = 2,
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;
    gpu::CommandList stale_recording = commands;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;

    gpu::GlobalBarrier barrier = {
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
    };
    fault rejected = @catch(gpu::cmd_global_barrier(
        &stale_recording,
        &barrier,
    ));
    assert(rejected == gpu::COMMAND_RECORDING_ERROR,
        "recording through an ended alias faults COMMAND_RECORDING_ERROR");
    gpu::end_frame(&frame)!!;
}
fn void texture_barrier_faults() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN, .frames_in_flight = 2 };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureBarrier bar = {
        .texture = gpu::TEXTURE_HANDLE_INVALID,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.FRAGMENT_SHADER,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.SHADER_READ,
        .old_layout = gpu::TextureLayout.TRANSFER_DST, .new_layout = gpu::TextureLayout.SHADER_READ,
    };
    fault barrier_fault = @catch(gpu::cmd_texture_barrier(&cmd, &bar));
    assert(barrier_fault == gpu::INVALID_HANDLE, "barrier on an invalid texture handle faults INVALID_HANDLE");
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
    gpu::end_frame(&frame)!!;
}

fn void submitted_command_buffer_retires_at_completion() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::CommandBufferStats before = vk::command_buffer_stats(state);
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;

    vk::CommandBufferStats submitted = vk::command_buffer_stats(state);
    assert(submitted.allocated == before.allocated + 1, "submit allocated one native command buffer");
    assert(submitted.freed == before.freed, "submit retains the native command buffer");
    assert(submitted.pending == before.pending + 1, "submit tracks one pending native command buffer");

    gpu::wait_completion(completion, ulong::max)!!;
    vk::CommandBufferStats completed = vk::command_buffer_stats(state);
    assert(completed.freed == before.freed, "wait leaves native reclamation to the recording owner");
    assert(completed.pending == before.pending, "wait retires the completion-tracked command buffer");

    gpu::CommandList next = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&next)!!;
    vk::CommandBufferStats reclaimed = vk::command_buffer_stats(state);
    assert(reclaimed.freed == before.freed + 1,
        "the recording owner reclaims the completed command buffer before allocating again");
}

fn void command_recording_does_not_require_frame_lifecycle() @test {
    gpu::DeviceDesc desc = { .backend = gpu::BackendKind.VULKAN };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    for (uint i = 0; i < 4; i++) {
        gpu::CommandList commands = gpu::begin_commands(
            gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        )!!;
        gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
        gpu::ExecutableCommandList[1] lists = { executable };
        gpu::SubmitDesc submit = { .command_lists = lists[..] };
        gpu::CompletionPoint completion = submit_test_queue(
            &device,
            gpu::QueueKind.GRAPHICS,
            &submit,
        )!!;
        gpu::wait_completion(completion, ulong::max)!!;
    }
}
