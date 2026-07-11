module gpu_test;

import gpu;
import gpu::vk;

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
    return gpu::create_device(&desc)!!;
}

fn gpu::TextureHandle make_layout_texture(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
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
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle tex = make_layout_texture(&device);
    defer gpu::destroy_texture(&device, tex)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;

    // Record a transition and end the list, but never submit it.
    gpu::CommandList dropped = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_layout_barrier(
        cmd:        &dropped,
        tex:        tex,
        old_layout: gpu::TextureLayout.UNDEFINED,
        new_layout: gpu::TextureLayout.TRANSFER_DST,
    );
    gpu::end_commands(&dropped)!!;

    // A fresh list must still see the pre-recording layout: the dropped
    // list's transition never committed.
    gpu::CommandList next = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_layout_barrier(
        cmd:        &next,
        tex:        tex,
        old_layout: gpu::TextureLayout.UNDEFINED,
        new_layout: gpu::TextureLayout.TRANSFER_DST,
    );
    gpu::end_commands(&next)!!;

    gpu::CommandList[1] lists = { next };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::end_frame(&frame_token)!!;
}

fn void two_lists_submitted_in_order_validate() @test {
    gpu::Device device = make_layout_device();
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle tex = make_layout_texture(&device);
    defer gpu::destroy_texture(&device, tex)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;

    gpu::CommandList first = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_layout_barrier(
        cmd:        &first,
        tex:        tex,
        old_layout: gpu::TextureLayout.UNDEFINED,
        new_layout: gpu::TextureLayout.TRANSFER_DST,
    );
    gpu::end_commands(&first)!!;
    gpu::CommandList[1] first_lists = { first };
    gpu::SubmitDesc first_submit = { .command_lists = first_lists[..] };
    gpu::submit(&device, &first_submit)!!;

    // Depends on the first list's commit having already landed: its
    // old_layout only matches if commit-at-submit ran before this validates.
    gpu::CommandList second = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_layout_barrier(
        cmd:        &second,
        tex:        tex,
        old_layout: gpu::TextureLayout.TRANSFER_DST,
        new_layout: gpu::TextureLayout.SHADER_READ,
    );
    gpu::end_commands(&second)!!;
    gpu::CommandList[1] second_lists = { second };
    gpu::SubmitDesc second_submit = { .command_lists = second_lists[..] };
    gpu::submit(&device, &second_submit)!!;

    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void same_list_barrier_then_copy_validates() @test {
    gpu::Device device = make_layout_device();
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle tex = make_layout_texture(&device);
    defer gpu::destroy_texture(&device, tex)!!;

    gpu::BufferDesc staging_desc = { .size = 64, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.STAGING };
    gpu::BufferHandle staging = gpu::create_buffer(&device, &staging_desc)!!;
    defer gpu::destroy_buffer(&device, staging)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    record_layout_barrier(
        cmd:        &cmd,
        tex:        tex,
        old_layout: gpu::TextureLayout.UNDEFINED,
        new_layout: gpu::TextureLayout.TRANSFER_DST,
    );

    gpu::GpuSpan staging_span = gpu::get_buffer_span(&device, staging)!!;
    gpu::BufferTextureCopyDesc copy = { .src = staging_span, .texture = tex, .width = 4, .height = 4 };
    gpu::cmd_copy_buffer_to_texture(&cmd, &copy)!!;

    gpu::end_commands(&cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void more_than_sixteen_pending_layouts_submit() @test {
    gpu::Device device = make_layout_device();
    defer gpu::destroy_device(&device)!!;

    const uint TEXTURE_COUNT = 17;
    gpu::TextureHandle[TEXTURE_COUNT] textures;
    for (uint i = 0; i < TEXTURE_COUNT; i++) textures[i] = make_layout_texture(&device);
    defer {
        foreach (t : textures) (void)gpu::destroy_texture(&device, t);
    }

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    for (uint i = 0; i < TEXTURE_COUNT; i++) {
        record_layout_barrier(
            cmd:        &cmd,
            tex:        textures[i],
            old_layout: gpu::TextureLayout.UNDEFINED,
            new_layout: gpu::TextureLayout.TRANSFER_DST,
        );
    }
    gpu::end_commands(&cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;

    gpu::CommandList verify = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    for (uint i = 0; i < TEXTURE_COUNT; i++) {
        record_layout_barrier(
            cmd:        &verify,
            tex:        textures[i],
            old_layout: gpu::TextureLayout.TRANSFER_DST,
            new_layout: gpu::TextureLayout.SHADER_READ,
        );
    }
    gpu::end_commands(&verify)!!;
    gpu::CommandList[1] verify_lists = { verify };
    gpu::SubmitDesc verify_submit = { .command_lists = verify_lists[..] };
    gpu::submit(&device, &verify_submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;
}
