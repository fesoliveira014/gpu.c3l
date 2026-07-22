module gpu_test;

import gpu;
import gpu::vk @public;

fn gpu::Device make_transition_device() {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_texture_transition_test",
    };
    return create_test_device(&desc)!!;
}

fn gpu::TextureHandle make_transition_texture(
    gpu::Device* device,
    uint mip_levels = 1,
) {
    gpu::TextureDesc desc = {
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
    gpu::TextureState before,
    gpu::TextureState after,
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
        before:   gpu::sampled_at({ .fragment_shader }),
        after:    {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
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
        before:   { .layout = gpu::TextureLayout.UNDEFINED },
        after:    {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
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
        before: {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
        after: gpu::sampled_at({ .fragment_shader }),
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
        before:   { .layout = gpu::TextureLayout.UNDEFINED },
        after:    {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
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
        before: {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
        after: gpu::sampled_at({ .compute }),
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
            before:   { .layout = gpu::TextureLayout.UNDEFINED },
            after:    {
                .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
                .stages = { .transfer },
                .access = { .write },
            },
        );
    }
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;
}
