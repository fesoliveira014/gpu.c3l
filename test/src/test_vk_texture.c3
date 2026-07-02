module gpu_test;

import gpu;
import gpu::vk;

// Headless (lavapipe): minimal texture lifecycle and whole-image barriers.

fn gpu::Device make_texture_test_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_texture_test",
    };
    return gpu::create_device(&desc)!!;
}

fn void texture_create_destroy() @test {
    gpu::Device device = make_texture_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = 16,
        .height     = 16,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .debug_name = "test_texture",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    assert(texture.is_valid(), "created texture handle is valid");
    gpu::destroy_texture(&device, texture)!!;
}

fn void texture_unsupported_desc_rejected() @test {
    gpu::Device device = make_texture_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc depth_format = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.D32_FLOAT,
        .usage     = { .sampled },
    };
    bool depth_faulted = false;
    if (catch gpu::create_texture(&device, &depth_format)) depth_faulted = true;
    assert(depth_faulted, "depth formats are outside the slice and must fault");

    gpu::TextureDesc no_heap_usage = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    bool usage_faulted = false;
    if (catch gpu::create_texture(&device, &no_heap_usage)) usage_faulted = true;
    assert(usage_faulted, "non-heap usage is outside the slice and must fault");

    gpu::TextureDesc not_2d = {
        .dimension = gpu::TextureDimension.TEX_3D,
        .width     = 16,
        .height    = 16,
        .depth     = 16,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled },
    };
    bool dim_faulted = false;
    if (catch gpu::create_texture(&device, &not_2d)) dim_faulted = true;
    assert(dim_faulted, "non-2D dimension is outside the slice and must fault");
}

fn void texture_stale_handle_rejected() @test {
    gpu::Device device = make_texture_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    gpu::destroy_texture(&device, texture)!!;

    bool faulted = false;
    if (catch gpu::destroy_texture(&device, texture)) faulted = true;
    assert(faulted, "double destroy must fault INVALID_HANDLE");
}

fn void texture_barrier_transitions_layout() @test {
    gpu::Device device = make_texture_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::TextureBarrier to_general = {
        .texture       = texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&cmd, &to_general)!!;

    bool mismatch_faulted = false;
    if (catch gpu::cmd_texture_barrier(&cmd, &to_general)) mismatch_faulted = true;
    assert(mismatch_faulted, "old_layout disagreeing with the tracked layout must fault");

    gpu::TextureBarrier general_to_general = {
        .texture       = texture,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
        .old_layout    = gpu::TextureLayout.GENERAL,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&cmd, &general_to_general)!!;

    gpu::end_commands(&device, &cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&device)!!;
}

fn void texture_multi_mip_create_destroy() @test {
    gpu::Device device = make_texture_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = 16,
        .height     = 16,
        .mip_levels = 4,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    assert(texture.is_valid(), "multi-mip texture created");
    gpu::destroy_texture(&device, texture)!!;

    gpu::TextureDesc over_chain = desc;
    over_chain.mip_levels = 6;
    bool faulted = false;
    if (catch gpu::create_texture(&device, &over_chain)) faulted = true;
    assert(faulted, "mip count past the full chain (16x16 -> 5) must fault");
}

fn void texture_multi_mip_barrier_covers_chain() @test {
    gpu::Device device = make_texture_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = 16,
        .height     = 16,
        .mip_levels = 5,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::TextureBarrier to_general = {
        .texture       = texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&cmd, &to_general)!!;
    gpu::end_commands(&device, &cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&device)!!;
}

fn void texture_barrier_stale_handle_rejected() @test {
    gpu::Device device = make_texture_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    gpu::destroy_texture(&device, texture)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::TextureBarrier barrier = {
        .texture    = texture,
        .old_layout = gpu::TextureLayout.UNDEFINED,
        .new_layout = gpu::TextureLayout.GENERAL,
    };
    bool faulted = false;
    if (catch gpu::cmd_texture_barrier(&cmd, &barrier)) faulted = true;
    assert(faulted, "barrier on a destroyed texture must fault INVALID_HANDLE");

    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}
