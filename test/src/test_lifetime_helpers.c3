module gpu_test;

import gpu @public;

struct OwnedSpan {
    gpu::GpuAllocation allocation;
    gpu::GpuSpan       span;
}

fn OwnedSpan? allocate_owned_span(
    gpu::Device* device,
    usz size,
    gpu::MemoryClass memory_class,
    gpu::QueueRoles access,
    ZString debug_name,
) {
    gpu::AllocationDesc desc = {
        .size         = size,
        .alignment    = 16,
        .memory_class = memory_class,
        .access       = access,
        .debug_name   = debug_name,
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(device, &desc)!;
    defer catch (void)gpu::free_allocation(device, &allocation);
    gpu::GpuSpan span = gpu::get_allocation_span(device, allocation)!;
    return { .allocation = allocation, .span = span };
}

fn void? free_owned_span(gpu::Device* device, OwnedSpan* owned) {
    if (owned == null || !owned.allocation.is_valid()) return;
    gpu::free_allocation(device, &owned.allocation)!;
    owned.span = {};
}

fn char[] test_span_mapping(gpu::Device* device, gpu::GpuSpan span) {
    return gpu::get_span_mapping(device, span)!!;
}

fn gpu::GpuAddress test_span_address(gpu::Device* device, gpu::GpuSpan span) {
    return gpu::get_span_address(device, span)!!;
}

fn void? discard_test_commands(gpu::CommandList* commands) {
    if (commands == null || !commands.handle.is_valid()) return;
    if (catch err = gpu::discard_commands(commands)) {
        if (err != gpu::INVALID_HANDLE) return err~;
    }
}

fn void? discard_test_executable(gpu::ExecutableCommandList* commands) {
    if (commands == null || !commands.handle.is_valid()) return;
    if (catch err = gpu::discard_executable_commands(commands)) {
        if (err != gpu::INVALID_HANDLE) return err~;
    }
}

fn gpu::CompletionPoint? submit_test_queue(
    gpu::Device* device,
    gpu::QueueKind kind,
    gpu::SubmitDesc* desc,
) {
    return gpu::submit(gpu::get_queue(device, kind)!, desc);
}

fn void? wait_test_queue(gpu::Device* device, gpu::QueueKind queue) {
    gpu::Queue target = gpu::get_queue(device, queue)!;
    gpu::CommandList commands = gpu::begin_commands(target)!;
    defer (void)gpu::discard_commands(&commands);
    gpu::ExecutableCommandList[1] lists = { gpu::end_commands(&commands)! };
    defer (void)gpu::discard_executable_commands(&lists[0]);
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(target, &submit)!;
    gpu::wait_completion(completion)!;
}

fn void? destroy_test_device(gpu::Device* device) {
    if (device == null || !device.is_valid()) return;
    gpu::QueueCounts? counts_result = gpu::get_queue_counts(device);
    if (catch err = counts_result) {
        if (err == gpu::DEVICE_LOST) {
            gpu::destroy_device(device)!;
            return;
        }
        return err~;
    }
    gpu::QueueCounts counts = counts_result;
    if (counts.graphics > 0) {
        wait_test_queue(device, gpu::QueueKind.GRAPHICS)!;
    }
    if (counts.compute > 0) {
        wait_test_queue(device, gpu::QueueKind.COMPUTE)!;
    }
    if (counts.transfer > 0) {
        wait_test_queue(device, gpu::QueueKind.TRANSFER)!;
    }
    gpu::destroy_device(device)!;
}

fn void? upload_test_texture_data(
    gpu::Device* device,
    gpu::TextureHandle dst,
    uint mip,
    char[] data,
    gpu::Stage next_stage,
    gpu::Hazard next_hazard,
    gpu::TextureLayout next_layout,
) {
    OwnedSpan upload = allocate_owned_span(
        device:       device,
        size:         data.len,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "test_texture_upload",
    )!;
    defer (void)free_owned_span(device, &upload);
    char[] mapping = gpu::get_span_mapping(device, upload.span)!;
    mapping[..] = data[..];
    gpu::flush_mapped_span(device, upload.span)!;

    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!;
    gpu::CommandList commands = gpu::begin_commands(queue)!;
    defer (void)discard_test_commands(&commands);
    gpu::TextureBarrier to_transfer = gpu::texture_transition(
        texture: dst,
        before:  gpu::TextureUse.UNDEFINED,
        after:   gpu::TextureUse.TRANSFER_DESTINATION,
    )!;
    gpu::cmd_texture_barrier(&commands, &to_transfer)!;
    gpu::BufferTextureCopyDesc copy = {
        .src     = upload.span,
        .texture = dst,
        .mip     = mip,
    };
    gpu::cmd_copy_buffer_to_texture(&commands, &copy)!;
    gpu::TextureBarrier next_use = {
        .texture       = dst,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = next_stage,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = next_hazard,
        .old_layout    = gpu::TextureLayout.TRANSFER_DST,
        .new_layout    = next_layout,
    };
    gpu::cmd_texture_barrier(&commands, &next_use)!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!;
    defer (void)discard_test_executable(&executable);
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!;
    gpu::wait_completion(completion)!;
}

fn void? readback_test_texture_data(
    gpu::Device* device,
    gpu::TextureHandle src,
    uint mip,
    char[] out_data,
    gpu::Stage from_stage,
    gpu::Hazard from_hazard,
    gpu::TextureLayout from_layout,
) {
    OwnedSpan readback = allocate_owned_span(
        device:       device,
        size:         out_data.len,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "test_texture_readback",
    )!;
    defer (void)free_owned_span(device, &readback);

    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!;
    gpu::CommandList commands = gpu::begin_commands(queue)!;
    defer (void)discard_test_commands(&commands);
    gpu::TextureBarrier to_transfer = {
        .texture       = src,
        .before_stage  = from_stage,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = from_hazard,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = from_layout,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(&commands, &to_transfer)!;
    gpu::TextureBufferCopyDesc copy = {
        .texture = src,
        .dst     = readback.span,
        .mip     = mip,
    };
    gpu::cmd_copy_texture_to_buffer(&commands, &copy)!;
    gpu::BufferBarrier host_read = {
        .span          = readback.span,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&commands, &host_read)!;
    gpu::TextureBarrier restore = {
        .texture       = src,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = from_stage,
        .before_hazard = gpu::Hazard.TRANSFER_READ,
        .after_hazard  = from_hazard,
        .old_layout    = gpu::TextureLayout.TRANSFER_SRC,
        .new_layout    = from_layout,
    };
    gpu::cmd_texture_barrier(&commands, &restore)!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!;
    defer (void)discard_test_executable(&executable);
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!;
    gpu::wait_completion(completion)!;

    gpu::invalidate_mapped_span(device, readback.span)!;
    char[] mapping = gpu::get_span_mapping(device, readback.span)!;
    out_data[..] = mapping[..];
}
