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

fn void unsupported_dimensions_fault() @test {
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
fn bool support_feature_for_usage(
    gpu::TextureFormatSupport support,
    gpu::TextureUsage usage,
) {
    if (usage.sampled) return support.features.sampled;
    if (usage.storage) return support.features.storage;
    if (usage.color_attach) return support.features.color_attach;
    if (usage.depth_attach) return support.features.depth_attach;
    if (usage.transfer_src) return support.features.transfer_src;
    return support.features.transfer_dst;
}

fn void texture_capability_queries_match_creation() @test {
    gpu::Device device = make_texture_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::Format[16] formats = {
        gpu::Format.R8_UNORM,
        gpu::Format.R8_UINT,
        gpu::Format.RG8_UNORM,
        gpu::Format.RGBA8_UNORM,
        gpu::Format.RGBA8_SRGB,
        gpu::Format.BGRA8_UNORM,
        gpu::Format.BGRA8_SRGB,
        gpu::Format.R16_UINT,
        gpu::Format.R16_FLOAT,
        gpu::Format.RG16_FLOAT,
        gpu::Format.RGBA16_FLOAT,
        gpu::Format.R32_UINT,
        gpu::Format.R32_FLOAT,
        gpu::Format.RG32_FLOAT,
        gpu::Format.RGBA32_FLOAT,
        gpu::Format.D32_FLOAT,
    };
    gpu::TextureUsage[6] usages = {
        { .sampled },
        { .storage },
        { .color_attach },
        { .depth_attach },
        { .transfer_src },
        { .transfer_dst },
    };

    foreach (format : formats) {
        gpu::TextureFormatSupport support = gpu::get_texture_format_support(&device, format)!!;
        assert(support.dimensions.tex_2d, "creatable formats report 2D support");
        assert(!support.dimensions.tex_1d && !support.dimensions.tex_3d
            && !support.dimensions.cube, "unimplemented dimensions stay masked");
        assert(support.sample_counts.one, "single-sample support is reported");
        assert(!support.sample_counts.two && !support.sample_counts.four
            && !support.sample_counts.eight && !support.sample_counts.sixteen
            && !support.sample_counts.thirty_two && !support.sample_counts.sixty_four,
            "unimplemented sample counts stay masked");

        bool is_depth = format == gpu::Format.D32_FLOAT;
        foreach (usage : usages) {
            if (is_depth && (usage.storage || usage.color_attach)) continue;
            if (!is_depth && usage.depth_attach) continue;

            gpu::TextureDesc desc = {
                .dimension = gpu::TextureDimension.TEX_2D,
                .width     = 4,
                .height    = 4,
                .format    = format,
                .usage     = usage,
            };
            bool supported = gpu::supports_texture_desc(&device, &desc)!!;
            assert(support_feature_for_usage(support, usage) == supported,
                "single-usage format summary matches exact descriptor support");
            if (supported) {
                gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
                gpu::destroy_texture(&device, texture)!!;
            } else {
                fault create_fault = @catch(gpu::create_texture(&device, &desc));
                assert(create_fault == gpu::UNSUPPORTED_FEATURE,
                    "adapter-rejected descriptor faults UNSUPPORTED_FEATURE");
            }
        }
    }
}

fn void texture_capability_queries_reject_uncreatable_descriptors() @test {
    gpu::Device device = make_texture_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureFormatSupport undefined_support =
        gpu::get_texture_format_support(&device, gpu::Format.UNDEFINED)!!;
    assert((uint)undefined_support.features == 0
        && (uint)undefined_support.dimensions == 0
        && (uint)undefined_support.sample_counts == 0,
        "undefined format reports empty support");

    gpu::TextureFormatSupport d24_support =
        gpu::get_texture_format_support(&device, gpu::Format.D24_UNORM_S8_UINT)!!;
    assert((uint)d24_support.features == 0
        && (uint)d24_support.dimensions == 0
        && (uint)d24_support.sample_counts == 0,
        "D24S8 reports empty support until the rendering path supports it end to end");

    gpu::TextureDesc d24_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.D24_UNORM_S8_UINT,
        .usage     = { .depth_attach },
    };
    assert(!gpu::supports_texture_desc(&device, &d24_desc)!!,
        "D24S8 exact descriptor preflights false");
    fault d24_fault = @catch(gpu::create_texture(&device, &d24_desc));
    assert(d24_fault == gpu::INVALID_ARGUMENT,
        "D24S8 texture creation remains INVALID_ARGUMENT");

    gpu::TextureDesc invalid = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = 16,
        .height     = 16,
        .mip_levels = 6,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled },
    };
    assert(!gpu::supports_texture_desc(&device, &invalid)!!,
        "deterministically invalid mip count preflights false");
    fault invalid_fault = @catch(gpu::create_texture(&device, &invalid));
    assert(invalid_fault == gpu::INVALID_ARGUMENT,
        "deterministically invalid descriptor keeps INVALID_ARGUMENT");

    gpu::TextureDesc beyond_adapter = {
        .dimension    = gpu::TextureDimension.TEX_2D,
        .width        = 1,
        .height       = 1,
        .array_layers = 4294967295,
        .format       = gpu::Format.RGBA8_UNORM,
        .usage        = { .sampled },
    };
    assert(!gpu::supports_texture_desc(&device, &beyond_adapter)!!,
        "descriptor beyond adapter layer limits preflights false");
    fault unsupported_fault = @catch(gpu::create_texture(&device, &beyond_adapter));
    assert(unsupported_fault == gpu::UNSUPPORTED_FEATURE,
        "adapter-limit rejection faults UNSUPPORTED_FEATURE before allocation");
}
