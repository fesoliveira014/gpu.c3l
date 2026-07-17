module gpu_test;

import gpu;
import gpu::vk @public;

// Headless (lavapipe): texture layouts commit onto tracked state at submit,
// not at record time — a dropped list must leave no trace, submitted lists
// must commit in submission order, and per-list pending state grows as needed.

fn gpu::Device make_layout_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_layout_commit_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
}

fn gpu::TextureHandle make_layout_texture(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_src, .transfer_dst },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void record_layout_barrier(gpu::CommandList* cmd, gpu::TextureHandle tex, gpu::TextureLayout old_layout, gpu::TextureLayout new_layout) {
    gpu::TextureBarrier barrier = {
        .texture       = tex,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
        .old_layout    = old_layout,
        .new_layout    = new_layout,
    };
    gpu::cmd_texture_barrier(cmd, &barrier)!!;
}

fn void dropped_list_leaves_layout_uncommitted() @test {
    gpu::Device device = make_layout_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle tex = make_layout_texture(&device);
    defer gpu::destroy_texture(&device, tex)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;

    // Record a transition and end the list, but never submit it.
    gpu::CommandList dropped = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&dropped)!!;
    record_layout_barrier(
        cmd:        &dropped,
        tex:        tex,
        old_layout: gpu::TextureLayout.UNDEFINED,
        new_layout: gpu::TextureLayout.TRANSFER_DST,
    );
    gpu::ExecutableCommandList dropped_executable = gpu::end_commands(&dropped)!!;
    defer discard_test_executable(&dropped_executable)!!;

    // A fresh list must still see the pre-recording layout: the dropped
    // list's transition never committed.
    gpu::CommandList next = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&next)!!;
    record_layout_barrier(
        cmd:        &next,
        tex:        tex,
        old_layout: gpu::TextureLayout.UNDEFINED,
        new_layout: gpu::TextureLayout.TRANSFER_DST,
    );
    gpu::ExecutableCommandList next_executable = gpu::end_commands(&next)!!;
    defer discard_test_executable(&next_executable)!!;

    gpu::ExecutableCommandList[1] lists = { next_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::end_frame(&frame_token)!!;
}

fn void two_lists_submitted_in_order_validate() @test {
    gpu::Device device = make_layout_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle tex = make_layout_texture(&device);
    defer gpu::destroy_texture(&device, tex)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;

    gpu::CommandList first = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&first)!!;
    record_layout_barrier(
        cmd:        &first,
        tex:        tex,
        old_layout: gpu::TextureLayout.UNDEFINED,
        new_layout: gpu::TextureLayout.TRANSFER_DST,
    );
    gpu::ExecutableCommandList first_executable = gpu::end_commands(&first)!!;
    defer discard_test_executable(&first_executable)!!;
    gpu::ExecutableCommandList[1] first_lists = { first_executable };
    gpu::SubmitDesc first_submit = { .command_lists = first_lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &first_submit)!!;

    // Depends on the first list's commit having already landed: its
    // old_layout only matches if commit-at-submit ran before this validates.
    gpu::CommandList second = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&second)!!;
    record_layout_barrier(
        cmd:        &second,
        tex:        tex,
        old_layout: gpu::TextureLayout.TRANSFER_DST,
        new_layout: gpu::TextureLayout.SHADER_READ,
    );
    gpu::ExecutableCommandList second_executable = gpu::end_commands(&second)!!;
    defer discard_test_executable(&second_executable)!!;
    gpu::ExecutableCommandList[1] second_lists = { second_executable };
    gpu::SubmitDesc second_submit = { .command_lists = second_lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &second_submit)!!;

    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void same_list_barrier_then_copy_validates() @test {
    gpu::Device device = make_layout_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle tex = make_layout_texture(&device);
    defer gpu::destroy_texture(&device, tex)!!;

    OwnedSpan staging = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "layout_copy_source",
    )!!;
    defer free_owned_span(&device, &staging)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    record_layout_barrier(
        cmd:        &cmd,
        tex:        tex,
        old_layout: gpu::TextureLayout.UNDEFINED,
        new_layout: gpu::TextureLayout.TRANSFER_DST,
    );

    gpu::GpuSpan staging_span = staging.span;
    gpu::BufferTextureCopyDesc copy = { .src = staging_span, .texture = tex, .width = 4, .height = 4 };
    gpu::cmd_copy_buffer_to_texture(&cmd, &copy)!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void more_than_sixteen_pending_layouts_submit() @test {
    gpu::Device device = make_layout_device();
    defer destroy_test_device(&device)!!;

    const uint TEXTURE_COUNT = 17;
    gpu::TextureHandle[TEXTURE_COUNT] textures;
    for (uint i = 0; i < TEXTURE_COUNT; i++) textures[i] = make_layout_texture(&device);
    defer {
        foreach (t : textures) (void)gpu::destroy_texture(&device, t);
    }

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    for (uint i = 0; i < TEXTURE_COUNT; i++) {
        record_layout_barrier(
            cmd:        &cmd,
            tex:        textures[i],
            old_layout: gpu::TextureLayout.UNDEFINED,
            new_layout: gpu::TextureLayout.TRANSFER_DST,
        );
    }
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;

    gpu::CommandList verify = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&verify)!!;
    for (uint i = 0; i < TEXTURE_COUNT; i++) {
        record_layout_barrier(
            cmd:        &verify,
            tex:        textures[i],
            old_layout: gpu::TextureLayout.TRANSFER_DST,
            new_layout: gpu::TextureLayout.SHADER_READ,
        );
    }
    gpu::ExecutableCommandList verify_executable = gpu::end_commands(&verify)!!;
    defer discard_test_executable(&verify_executable)!!;
    gpu::ExecutableCommandList[1] verify_lists = { verify_executable };
    gpu::SubmitDesc verify_submit = { .command_lists = verify_lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &verify_submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;
}
