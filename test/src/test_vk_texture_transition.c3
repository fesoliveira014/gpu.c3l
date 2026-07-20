module gpu_test;

import gpu;
import gpu::vk @public;

fn gpu::Device make_transition_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_texture_transition_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
}

fn gpu::TextureHandle make_transition_texture(
    gpu::Device* device,
    uint mip_levels = 1,
) {
    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access     = { .graphics, .compute, .transfer },
        .width      = 4,
        .height     = 4,
        .mip_levels = mip_levels,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .transfer_src, .transfer_dst },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void record_transition(
    gpu::CommandList* commands,
    gpu::TextureHandle texture,
    gpu::TextureViewDesc view,
    gpu::TextureUse before,
    gpu::TextureUse after,
) {
    gpu::TextureBarrier barrier = gpu::texture_view_transition(
        texture: texture,
        view:    view,
        before:  before,
        after:   after,
    )!!;
    gpu::cmd_texture_barrier(commands, &barrier)!!;
}

fn void mismatched_previous_use_remains_explicit_at_recording() @test {
    gpu::Device device = make_transition_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_transition_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    defer discard_test_commands(&commands)!!;
    record_transition(
        commands: &commands,
        texture:  texture,
        view:     {},
        before:   gpu::TextureUse.SAMPLED_FRAGMENT,
        after:    gpu::TextureUse.TRANSFER_SOURCE,
    );
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    discard_test_executable(&executable)!!;
}

fn void explicit_transition_copy_and_sample_submit() @test {
    gpu::Device device = make_transition_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_transition_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    OwnedSpan upload = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "texture_transition_upload",
    )!!;
    defer free_owned_span(&device, &upload)!!;
    char[] mapping = gpu::get_span_mapping(&device, upload.span)!!;
    for (uint i = 0; i < mapping.len; i++) mapping[i] = (char)i;
    gpu::flush_mapped_span(&device, upload.span)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    defer discard_test_commands(&commands)!!;
    record_transition(
        commands: &commands,
        texture:  texture,
        view:     {},
        before:   gpu::TextureUse.UNDEFINED,
        after:    gpu::TextureUse.TRANSFER_DESTINATION,
    );
    gpu::BufferTextureCopyDesc copy = {
        .src     = upload.span,
        .texture = texture,
        .width   = 4,
        .height  = 4,
    };
    gpu::cmd_copy_buffer_to_texture(&commands, &copy)!!;
    record_transition(
        commands: &commands,
        texture:  texture,
        view:     {},
        before:   gpu::TextureUse.TRANSFER_DESTINATION,
        after:    gpu::TextureUse.SAMPLED_FRAGMENT,
    );

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;
}

fn void partial_mip_transition_copy_submits() @test {
    gpu::Device device = make_transition_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_transition_texture(&device, 2);
    defer gpu::destroy_texture(&device, texture)!!;
    OwnedSpan upload = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "texture_mip_transition_upload",
    )!!;
    defer free_owned_span(&device, &upload)!!;
    gpu::flush_mapped_span(&device, upload.span)!!;

    gpu::TextureViewDesc view = {
        .base_mip  = 1,
        .mip_count = 1,
    };
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    defer discard_test_commands(&commands)!!;
    record_transition(
        commands: &commands,
        texture:  texture,
        view:     view,
        before:   gpu::TextureUse.UNDEFINED,
        after:    gpu::TextureUse.TRANSFER_DESTINATION,
    );
    gpu::BufferTextureCopyDesc copy = {
        .src     = upload.span,
        .texture = texture,
        .mip     = 1,
        .width   = 2,
        .height  = 2,
    };
    gpu::cmd_copy_buffer_to_texture(&commands, &copy)!!;
    record_transition(
        commands: &commands,
        texture:  texture,
        view:     view,
        before:   gpu::TextureUse.TRANSFER_DESTINATION,
        after:    gpu::TextureUse.SAMPLED_COMPUTE,
    );

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;
}

fn void many_explicit_texture_transitions_submit() @test {
    const uint TEXTURE_COUNT = 17;

    gpu::Device device = make_transition_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle[TEXTURE_COUNT] textures;
    for (uint i = 0; i < TEXTURE_COUNT; i++) {
        textures[i] = make_transition_texture(&device);
    }
    defer {
        foreach (texture : textures) {
            (void)gpu::destroy_texture(&device, texture);
        }
    }

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    defer discard_test_commands(&commands)!!;
    foreach (texture : textures) {
        record_transition(
            commands: &commands,
            texture:  texture,
            view:     {},
            before:   gpu::TextureUse.UNDEFINED,
            after:    gpu::TextureUse.TRANSFER_DESTINATION,
        );
    }
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;
}
