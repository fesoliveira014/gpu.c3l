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

    gpu::TextureDesc packed_depth = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.D24_UNORM_S8_UINT,
        .usage     = { .depth_attach },
    };
    fault packed_fault = @catch(gpu::create_texture(&device, &packed_depth));
    assert(packed_fault == gpu::INVALID_ARGUMENT, "D24_UNORM_S8_UINT faults INVALID_ARGUMENT");

    gpu::TextureDesc depth_storage = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.D32_FLOAT,
        .usage     = { .depth_attach, .storage },
    };
    fault storage_fault = @catch(gpu::create_texture(&device, &depth_storage));
    assert(storage_fault == gpu::INVALID_ARGUMENT, "storage usage on a depth format faults INVALID_ARGUMENT");

    gpu::TextureDesc color_with_depth_usage = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .depth_attach },
    };
    fault cross_fault = @catch(gpu::create_texture(&device, &color_with_depth_usage));
    assert(cross_fault == gpu::INVALID_ARGUMENT, "depth_attach on a color format faults INVALID_ARGUMENT");

    gpu::TextureDesc transfer_only = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::TextureHandle staging_target = gpu::create_texture(&device, &transfer_only)!!;
    fault descriptor_fault = @catch(gpu::create_texture_descriptor(&device, staging_target, null));
    assert(descriptor_fault == gpu::INVALID_ARGUMENT, "descriptor on a non-shader-visible texture faults INVALID_ARGUMENT");
    gpu::destroy_texture(&device, staging_target)!!;

    gpu::TextureDesc not_2d = {
        .dimension = gpu::TextureDimension.TEX_3D,
        .width     = 16,
        .height    = 16,
        .depth     = 16,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled },
    };
    fault dim_fault = @catch(gpu::create_texture(&device, &not_2d));
    assert(dim_fault == gpu::INVALID_ARGUMENT, "non-2D dimensions fault INVALID_ARGUMENT");
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

    fault stale_fault = @catch(gpu::destroy_texture(&device, texture));
    assert(stale_fault == gpu::INVALID_HANDLE, "double destroy faults INVALID_HANDLE");
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

    fault mismatch_fault = @catch(gpu::cmd_texture_barrier(&cmd, &to_general));
    assert(mismatch_fault == gpu::INVALID_RESOURCE_STATE, "old_layout disagreeing with the tracked layout faults INVALID_RESOURCE_STATE");

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
    fault chain_fault = @catch(gpu::create_texture(&device, &over_chain));
    assert(chain_fault == gpu::INVALID_ARGUMENT, "mip count past the full chain (16x16 -> 5) faults INVALID_ARGUMENT");
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
    fault barrier_fault = @catch(gpu::cmd_texture_barrier(&cmd, &barrier));
    assert(barrier_fault == gpu::INVALID_HANDLE, "barrier on a destroyed texture faults INVALID_HANDLE");

    gpu::end_commands(&device, &cmd)!!;
    gpu::end_frame(&device)!!;
}

fn void raised_texture_capacity_holds_more_than_default() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .texture_capacity  = 2048,
        .application_name  = "gpu_texture_capacity_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    const uint COUNT = 1500;
    gpu::TextureHandle[] handles = mem::new_array(gpu::TextureHandle, COUNT);
    defer mem::free(handles.ptr);
    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled },
    };
    foreach (&h : handles) *h = gpu::create_texture(&device, &tex_desc)!!;
    foreach (h : handles) gpu::destroy_texture(&device, h)!!;
}

fn void unsupported_dimensions_and_formats_fault() @test {
    gpu::Device device = make_texture_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDimension[3] dims = { gpu::TextureDimension.TEX_1D, gpu::TextureDimension.TEX_3D, gpu::TextureDimension.CUBE };
    foreach (dim : dims) {
        gpu::TextureDesc desc = {
            .dimension = dim,
            .width     = 4,
            .height    = 4,
            .format    = gpu::Format.RGBA8_UNORM,
            .usage     = { .sampled },
        };
        fault f = @catch(gpu::create_texture(&device, &desc));
        assert(f == gpu::INVALID_ARGUMENT, "unsupported dimension faults INVALID_ARGUMENT");
    }

    gpu::TextureDesc d24 = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.D24_UNORM_S8_UINT,
        .usage     = { .depth_attach },
    };
    fault f = @catch(gpu::create_texture(&device, &d24));
    assert(f == gpu::INVALID_ARGUMENT, "D24S8 faults INVALID_ARGUMENT");
}

fn void transfer_only_texture_round_trips() @test {
    gpu::Device device = make_texture_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_src, .transfer_dst },
    };
    gpu::TextureHandle tex = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, tex)!!;

    char[64] pixels;
    for (uint i = 0; i < 64; i++) pixels[i] = (char)i;
    gpu::upload_texture_data(
        device:      &device,
        dst:         tex,
        mip:         0,
        data:        pixels[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_WRITE,
        next_layout: gpu::TextureLayout.TRANSFER_DST,
    )!!;

    char[64] readback;
    gpu::readback_texture_data(
        device:      &device,
        src:         tex,
        mip:         0,
        out_data:    readback[..],
        from_stage:  gpu::Stage.TRANSFER,
        from_hazard: gpu::Hazard.TRANSFER_WRITE,
        from_layout: gpu::TextureLayout.TRANSFER_DST,
    )!!;
    for (uint i = 0; i < 64; i++) {
        assert(readback[i] == (char)i, "transfer-only texture round-trips bytes");
    }
}
