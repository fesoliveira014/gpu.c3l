module gpu_test;

import gpu::compat;
import gpu::compat::vk @public;

// Headless (lavapipe): texture layouts commit onto tracked state at submit,
// not at record time — a dropped list must leave no trace, submitted lists
// must commit in submission order, and per-list pending state grows as needed.

fn gpu::compat::Device make_layout_device() {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_layout_commit_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

fn gpu::compat::TextureHandle make_layout_texture(gpu::compat::Device* device) {
    gpu::compat::TextureDesc desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_src, .transfer_dst },
    };
    return gpu::compat::create_texture(device, &desc)!!;
}

fn void record_layout_barrier(gpu::compat::CommandList* cmd, gpu::compat::TextureHandle tex, gpu::compat::TextureLayout old_layout, gpu::compat::TextureLayout new_layout) {
    gpu::compat::TextureBarrier barrier = {
        .texture       = tex,
        .before_stage  = gpu::compat::Stage.TRANSFER,
        .after_stage   = gpu::compat::Stage.TRANSFER,
        .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::compat::Hazard.TRANSFER_WRITE,
        .old_layout    = old_layout,
        .new_layout    = new_layout,
    };
    gpu::compat::cmd_texture_barrier(cmd, &barrier)!!;
}

fn void dropped_list_leaves_layout_uncommitted() @test {
    gpu::compat::Device device = make_layout_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle tex = make_layout_texture(&device);
    defer gpu::compat::destroy_texture(&device, tex)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;

    // Record a transition and end the list, but never submit it.
    gpu::compat::CommandList dropped = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_layout_barrier(
        cmd:        &dropped,
        tex:        tex,
        old_layout: gpu::compat::TextureLayout.UNDEFINED,
        new_layout: gpu::compat::TextureLayout.TRANSFER_DST,
    );
    gpu::compat::end_commands(&dropped)!!;

    // A fresh list must still see the pre-recording layout: the dropped
    // list's transition never committed.
    gpu::compat::CommandList next = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_layout_barrier(
        cmd:        &next,
        tex:        tex,
        old_layout: gpu::compat::TextureLayout.UNDEFINED,
        new_layout: gpu::compat::TextureLayout.TRANSFER_DST,
    );
    gpu::compat::end_commands(&next)!!;

    gpu::compat::CommandList[1] lists = { next };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::end_frame(&frame_token)!!;
}

fn void two_lists_submitted_in_order_validate() @test {
    gpu::compat::Device device = make_layout_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle tex = make_layout_texture(&device);
    defer gpu::compat::destroy_texture(&device, tex)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;

    gpu::compat::CommandList first = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_layout_barrier(
        cmd:        &first,
        tex:        tex,
        old_layout: gpu::compat::TextureLayout.UNDEFINED,
        new_layout: gpu::compat::TextureLayout.TRANSFER_DST,
    );
    gpu::compat::end_commands(&first)!!;
    gpu::compat::CommandList[1] first_lists = { first };
    gpu::compat::SubmitDesc first_submit = { .command_lists = first_lists[..] };
    gpu::compat::submit(&device, &first_submit)!!;

    // Depends on the first list's commit having already landed: its
    // old_layout only matches if commit-at-submit ran before this validates.
    gpu::compat::CommandList second = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_layout_barrier(
        cmd:        &second,
        tex:        tex,
        old_layout: gpu::compat::TextureLayout.TRANSFER_DST,
        new_layout: gpu::compat::TextureLayout.SHADER_READ,
    );
    gpu::compat::end_commands(&second)!!;
    gpu::compat::CommandList[1] second_lists = { second };
    gpu::compat::SubmitDesc second_submit = { .command_lists = second_lists[..] };
    gpu::compat::submit(&device, &second_submit)!!;

    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void same_list_barrier_then_copy_validates() @test {
    gpu::compat::Device device = make_layout_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle tex = make_layout_texture(&device);
    defer gpu::compat::destroy_texture(&device, tex)!!;

    gpu::compat::BufferDesc staging_desc = { .size = 64, .usage = { .transfer_src }, .memory_kind = gpu::compat::MemoryKind.STAGING };
    gpu::compat::BufferHandle staging = gpu::compat::create_buffer(&device, &staging_desc)!!;
    defer gpu::compat::destroy_buffer(&device, staging)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    record_layout_barrier(
        cmd:        &cmd,
        tex:        tex,
        old_layout: gpu::compat::TextureLayout.UNDEFINED,
        new_layout: gpu::compat::TextureLayout.TRANSFER_DST,
    );

    gpu::compat::GpuSpan staging_span = gpu::compat::get_buffer_span(&device, staging)!!;
    gpu::compat::BufferTextureCopyDesc copy = { .src = staging_span, .texture = tex, .width = 4, .height = 4 };
    gpu::compat::cmd_copy_buffer_to_texture(&cmd, &copy)!!;

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_frame(&frame_token)!!;
}

fn void more_than_sixteen_pending_layouts_submit() @test {
    gpu::compat::Device device = make_layout_device();
    defer gpu::compat::destroy_device(&device)!!;

    const uint TEXTURE_COUNT = 17;
    gpu::compat::TextureHandle[TEXTURE_COUNT] textures;
    for (uint i = 0; i < TEXTURE_COUNT; i++) textures[i] = make_layout_texture(&device);
    defer {
        foreach (t : textures) (void)gpu::compat::destroy_texture(&device, t);
    }

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    for (uint i = 0; i < TEXTURE_COUNT; i++) {
        record_layout_barrier(
            cmd:        &cmd,
            tex:        textures[i],
            old_layout: gpu::compat::TextureLayout.UNDEFINED,
            new_layout: gpu::compat::TextureLayout.TRANSFER_DST,
        );
    }
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;

    gpu::compat::CommandList verify = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    for (uint i = 0; i < TEXTURE_COUNT; i++) {
        record_layout_barrier(
            cmd:        &verify,
            tex:        textures[i],
            old_layout: gpu::compat::TextureLayout.TRANSFER_DST,
            new_layout: gpu::compat::TextureLayout.SHADER_READ,
        );
    }
    gpu::compat::end_commands(&verify)!!;
    gpu::compat::CommandList[1] verify_lists = { verify };
    gpu::compat::SubmitDesc verify_submit = { .command_lists = verify_lists[..] };
    gpu::compat::submit(&device, &verify_submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_frame(&frame_token)!!;
}
