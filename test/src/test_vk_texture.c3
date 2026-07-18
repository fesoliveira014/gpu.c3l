module gpu_test;

import gpu @public;
import gpu::vk @public;
import vma;
import vk;

struct PlacementNativeProbe {
    vk::Result create_result;
    vk::Result view_result;
    uint       create_calls;
    uint       view_calls;
    uint       destroy_image_calls;
    uint       destroy_view_calls;
}

struct PlacementFixture {
    vk::VkDeviceState     state;
    vk::AllocationCell[1] allocations;
    vk::TextureCell[1]    textures;
    vk::DeferredEntry[1]  deferred;
    PlacementNativeProbe  probe;
}

fn vk::Result placement_probe_create_image(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
    vk::DeviceSize offset,
    vk::ImageCreateInfo* image_info,
    vk::Image* out_image,
) {
    (void)allocator;
    (void)allocation;
    (void)offset;
    (void)image_info;
    PlacementNativeProbe* probe = (PlacementNativeProbe*)user_data;
    probe.create_calls++;
    if (probe.create_result != vk::Result.SUCCESS) return probe.create_result;
    *out_image = (vk::Image)(uptr)0x100;
    return vk::Result.SUCCESS;
}

fn vk::Result placement_probe_create_view(
    void* user_data,
    vk::Device device,
    vk::ImageViewCreateInfo* view_info,
    vk::ImageView* out_view,
) {
    (void)device;
    (void)view_info;
    PlacementNativeProbe* probe = (PlacementNativeProbe*)user_data;
    probe.view_calls++;
    if (probe.view_result != vk::Result.SUCCESS) return probe.view_result;
    *out_view = (vk::ImageView)(uptr)0x200;
    return vk::Result.SUCCESS;
}

fn void placement_probe_destroy_image(
    void* user_data,
    vk::Device device,
    vk::Image image,
) {
    (void)device;
    (void)image;
    PlacementNativeProbe* probe = (PlacementNativeProbe*)user_data;
    probe.destroy_image_calls++;
}

fn void placement_probe_destroy_view(
    void* user_data,
    vk::Device device,
    vk::ImageView view,
) {
    (void)device;
    (void)view;
    PlacementNativeProbe* probe = (PlacementNativeProbe*)user_data;
    probe.destroy_view_calls++;
}

fn vk::TexturePlacementNativeOps placement_probe_ops(
    PlacementNativeProbe* probe,
) {
    return {
        .user_data             = probe,
        .create_aliasing_image = &placement_probe_create_image,
        .create_view           = &placement_probe_create_view,
        .destroy_image         = &placement_probe_destroy_image,
        .destroy_view          = &placement_probe_destroy_view,
    };
}

fn gpu::GpuAllocation init_placement_fixture(PlacementFixture* fixture) {
    const ulong OWNER = 119;
    fixture.state.resource_owner = OWNER;
    fixture.state.compute_family = 0;
    fixture.state.allocations.owner = gpu::resource_owner_tag(
        OWNER,
        gpu::ResourceOwnerKind.ALLOCATION,
    );
    fixture.state.allocations.slots = fixture.allocations[..];
    fixture.state.allocations.next_new = 1;
    fixture.allocations[0] = {
        .used       = true,
        .generation = 1,
        .value = {
            .raw_allocation = (vma::Allocation)(void*)0x300,
            .raw_info       = {
                .memory_type = 1,
                .size        = 256,
            },
            .size           = 256,
            .alignment      = 64,
            .access         = { .compute },
            .texture_memory = true,
            .memory_class   = gpu::MemoryClass.TEXTURE,
        },
    };
    fixture.state.textures.owner = gpu::resource_owner_tag(
        OWNER,
        gpu::ResourceOwnerKind.TEXTURE,
    );
    fixture.state.textures.slots = fixture.textures[..];
    fixture.state.resource_mutex.init()!!;
    return {
        .owner      = fixture.state.allocations.owner,
        .index      = 0,
        .generation = 1,
    };
}

fn gpu::TextureDesc placement_test_desc() {
    return {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .compute },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled },
    };
}

fn gpu::TextureRequirements placement_test_requirements(
    PlacementFixture* fixture,
) {
    return {
        .size      = 64,
        .alignment = 64,
        .compatibility = vk::texture_compatibility(
            fixture.state.allocations.owner,
            0b0010,
        ),
    };
}


// Headless (lavapipe): minimal texture lifecycle and whole-image barriers.

struct TextureDiagnosticCapture {
    uint                   calls;
    bool                   has_fault;
    fault                  public_fault;
    bool                   public_contract;
    gpu::DebugResourceKind resource_kind;
    bool                   has_public_identity;
    uint                   index;
    uint                   generation;
    ZString                expected_operation;
    ZString                expected_debug_name;
    ZString                expected_rejected_field;
    ZString                expected_invariant;
    bool                   operation_matches;
    bool                   debug_name_matches;
    bool                   rejected_field_matches;
    bool                   invariant_matches;
}

fn void capture_texture_diagnostic(gpu::DebugMessage* message, void* user_data) {
    if (message.operation == null) return;
    TextureDiagnosticCapture* capture = (TextureDiagnosticCapture*)user_data;
    capture.calls++;
    capture.has_fault = message.has_fault;
    capture.public_fault = message.public_fault;
    capture.public_contract = message.category.public_contract;
    capture.resource_kind = message.resource.kind;
    capture.has_public_identity = message.resource.has_public_identity;
    capture.index = message.resource.index;
    capture.generation = message.resource.generation;
    capture.operation_matches = message.operation.str_view() == capture.expected_operation.str_view();
    capture.debug_name_matches = message.resource.debug_name != null
        && message.resource.debug_name.str_view() == capture.expected_debug_name.str_view();
    capture.rejected_field_matches = message.rejected_field != null
        && message.rejected_field.str_view() == capture.expected_rejected_field.str_view();
    capture.invariant_matches = message.invariant != null
        && message.invariant.str_view() == capture.expected_invariant.str_view();
}

fn gpu::Device make_texture_diagnostic_device(TextureDiagnosticCapture* capture) {
    gpu::DeviceDesc desc = {
        .backend              = gpu::BackendKind.VULKAN,
        .enable_validation    = true,
        .frames_in_flight     = 2,
        .application_name     = "gpu_texture_diagnostic_test",
        .debug_callback       = &capture_texture_diagnostic,
        .debug_user_data      = capture,
    };
    return gpu::create_device_from_desc(&desc)!!;
}

fn gpu::Device make_texture_test_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_texture_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
}

fn void texture_create_destroy() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access     = { .graphics, .compute },
        .width      = 16,
        .height     = 16,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .debug_name = "test_texture",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    assert(texture.is_valid(), "created texture handle is valid");
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state_ptr(&device);
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    assert((uint)slot.access == (uint)desc.access,
        "texture slot retains the descriptor access roles");
    gpu::destroy_texture(&device, texture)!!;
}

fn void texture_unsupported_desc_rejected() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc depth_storage = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.D32_FLOAT,
        .usage     = { .depth_attach, .storage },
    };
    fault storage_fault = @catch(gpu::create_texture(&device, &depth_storage));
    assert(storage_fault == gpu::INVALID_ARGUMENT, "storage usage on a depth format faults INVALID_ARGUMENT");

    gpu::TextureDesc color_with_depth_usage = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .depth_attach },
    };
    fault cross_fault = @catch(gpu::create_texture(&device, &color_with_depth_usage));
    assert(cross_fault == gpu::INVALID_ARGUMENT, "depth_attach on a color format faults INVALID_ARGUMENT");

    gpu::TextureDesc transfer_only = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
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
        .access    = { .graphics, .compute, .transfer },
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
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
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
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

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

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;

    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void texture_multi_mip_create_destroy() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access     = { .graphics, .compute },
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
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access     = { .graphics, .compute },
        .width      = 16,
        .height     = 16,
        .mip_levels = 5,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
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
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;

    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void texture_barrier_stale_handle_rejected() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    gpu::destroy_texture(&device, texture)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureBarrier barrier = {
        .texture    = texture,
        .old_layout = gpu::TextureLayout.UNDEFINED,
        .new_layout = gpu::TextureLayout.GENERAL,
    };
    fault barrier_fault = @catch(gpu::cmd_texture_barrier(&cmd, &barrier));
    assert(barrier_fault == gpu::INVALID_HANDLE, "barrier on a destroyed texture faults INVALID_HANDLE");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void raised_texture_capacity_holds_more_than_default() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .texture_capacity  = 2048,
        .application_name  = "gpu_texture_capacity_test",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_test_device(&device)!!;

    const uint COUNT = 1500;
    gpu::TextureHandle[] handles = mem::new_array(gpu::TextureHandle, COUNT);
    defer mem::free(handles.ptr);
    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
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
    defer destroy_test_device(&device)!!;

    gpu::TextureDimension[3] dims = { gpu::TextureDimension.TEX_1D, gpu::TextureDimension.TEX_3D, gpu::TextureDimension.CUBE };
    foreach (dim : dims) {
        gpu::TextureDesc desc = {
            .dimension = dim,
            .access    = { .graphics, .compute, .transfer },
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
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
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
    defer destroy_test_device(&device)!!;

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
                .access    = { .graphics, .compute, .transfer },
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
    defer destroy_test_device(&device)!!;

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
        .access    = { .graphics, .compute, .transfer },
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
        .access     = { .graphics, .compute },
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
        .access    = { .graphics, .compute, .transfer },
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

fn void invalid_texture_descriptor_diagnostic_preserves_fault() @test {
    TextureDiagnosticCapture capture = {
        .expected_operation      = "create_texture",
        .expected_debug_name     = "invalid_texture_desc",
        .expected_rejected_field = "TextureDesc",
        .expected_invariant      = "descriptor must satisfy the texture creation contract",
    };
    gpu::Device device = make_texture_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc invalid = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access     = { .graphics, .compute },
        .width      = 0,
        .height     = 16,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled },
        .debug_name = "invalid_texture_desc",
    };
    fault with_callback = @catch(gpu::create_texture(&device, &invalid));

    assert(with_callback == gpu::INVALID_ARGUMENT, "invalid descriptor keeps INVALID_ARGUMENT");
    assert(capture.calls == 1, "invalid descriptor emits exactly one diagnostic");
    assert(capture.operation_matches, "operation is create_texture");
    assert(capture.has_fault && capture.public_fault == with_callback,
        "diagnostic carries the unchanged public fault");
    assert(capture.public_contract, "invalid descriptor is a public-contract diagnostic");
    assert(capture.resource_kind == gpu::DebugResourceKind.TEXTURE, "resource kind is texture");
    assert(!capture.has_public_identity, "rejected creation has no public identity");
    assert(capture.debug_name_matches, "descriptor debug name is preserved");
    assert(capture.rejected_field_matches, "invalid descriptor is reported without re-deriving its field");
    assert(capture.invariant_matches, "generic texture creation invariant is identified");

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    state.debug_callback = null;
    fault without_callback = @catch(gpu::create_texture(&device, &invalid));
    assert(without_callback == with_callback, "callback presence does not alter the descriptor fault");
    assert(capture.calls == 1, "null callback emits no additional diagnostic");
}

fn void unsupported_texture_capability_diagnostic_preserves_fault() @test {
    TextureDiagnosticCapture capture = {
        .expected_operation      = "create_texture",
        .expected_debug_name     = "unsupported_texture_desc",
        .expected_rejected_field = "TextureDesc",
        .expected_invariant      = "descriptor must be supported by selected device capabilities",
    };
    gpu::Device device = make_texture_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc unsupported = {
        .dimension    = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width        = 1,
        .height       = 1,
        .array_layers = 4294967295,
        .format       = gpu::Format.RGBA8_UNORM,
        .usage        = { .sampled },
        .debug_name   = "unsupported_texture_desc",
    };
    fault with_callback = @catch(gpu::create_texture(&device, &unsupported));

    assert(with_callback == gpu::UNSUPPORTED_FEATURE, "adapter rejection keeps UNSUPPORTED_FEATURE");
    assert(capture.calls == 1, "unsupported descriptor emits exactly one diagnostic");
    assert(capture.operation_matches, "operation is create_texture");
    assert(capture.has_fault && capture.public_fault == with_callback,
        "diagnostic carries the unchanged capability fault");
    assert(capture.public_contract, "adapter rejection is a public-contract diagnostic");
    assert(capture.resource_kind == gpu::DebugResourceKind.TEXTURE, "resource kind is texture");
    assert(!capture.has_public_identity, "rejected creation has no public identity");
    assert(capture.debug_name_matches, "descriptor debug name is preserved");
    assert(capture.rejected_field_matches, "capability rejection avoids guessing a limiting field");
    assert(capture.invariant_matches, "generic capability invariant is accurate");

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    state.debug_callback = null;
    fault without_callback = @catch(gpu::create_texture(&device, &unsupported));
    assert(without_callback == with_callback, "callback presence does not alter the capability fault");
    assert(capture.calls == 1, "null callback emits no additional diagnostic");
}

fn void texture_layout_mismatch_diagnostic_identifies_resource() @test {
    TextureDiagnosticCapture capture = {
        .expected_operation      = "cmd_texture_barrier",
        .expected_debug_name     = "layout_diagnostic_texture",
        .expected_rejected_field = "TextureBarrier.old_layout",
        .expected_invariant      = "old_layout must match the command list effective layout",
    };
    gpu::Device device = make_texture_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access     = { .graphics, .compute },
        .width      = 8,
        .height     = 8,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .debug_name = "layout_diagnostic_texture",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    gpu::CommandList commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&commands)!!;
    gpu::TextureBarrier barrier = {
        .texture       = texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&commands, &barrier)!!;
    fault with_callback = @catch(gpu::cmd_texture_barrier(&commands, &barrier));

    assert(with_callback == gpu::INVALID_RESOURCE_STATE, "layout mismatch keeps INVALID_RESOURCE_STATE");
    assert(capture.calls == 1, "layout mismatch emits exactly one diagnostic");
    assert(capture.operation_matches, "operation is cmd_texture_barrier");
    assert(capture.has_fault && capture.public_fault == with_callback,
        "diagnostic carries the unchanged layout fault");
    assert(capture.public_contract, "layout mismatch is a public-contract diagnostic");
    assert(capture.resource_kind == gpu::DebugResourceKind.TEXTURE, "resource kind is texture");
    assert(capture.has_public_identity, "live texture identity is present");
    assert(capture.index == texture.index, "texture index is preserved");
    assert(capture.generation == texture.generation, "texture generation is preserved");
    assert(capture.debug_name_matches, "texture debug name is preserved");
    assert(capture.rejected_field_matches, "old_layout field is identified");
    assert(capture.invariant_matches, "tracked-layout invariant is identified");

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    state.debug_callback = null;
    fault without_callback = @catch(gpu::cmd_texture_barrier(&commands, &barrier));
    assert(without_callback == with_callback, "callback presence does not alter the layout fault");
    assert(capture.calls == 1, "null callback emits no additional diagnostic");

    gpu::ExecutableCommandList commands_executable = gpu::end_commands(&commands)!!;

    defer discard_test_executable(&commands_executable)!!;
    gpu::end_frame(&frame)!!;
}

fn void placed_texture_requirements_validate_and_release_storage() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc texture_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute },
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureRequirements requirements =
        gpu::get_texture_requirements(&device, &texture_desc)!!;
    assert(requirements.size > 0
        && requirements.alignment > 0
        && requirements.compatibility.is_valid()
        && !requirements.dedicated_only,
        "a basic texture exposes reusable device-bound requirements");

    gpu::TextureRequirements[1] queried = { requirements };
    gpu::AllocationDesc allocation_desc = {
        .size                 = requirements.size * 2,
        .memory_class         = gpu::MemoryClass.TEXTURE,
        .access               = texture_desc.access,
        .texture_requirements = queried[..],
    };
    gpu::GpuAllocation allocation =
        gpu::allocate_memory(&device, &allocation_desc)!!;
    gpu::AllocationInfo info =
        gpu::get_allocation_info(&device, allocation)!!;
    assert(info.memory_class == gpu::MemoryClass.TEXTURE
        && !info.mapped && !info.coherent && !info.addressable,
        "texture memory exposes no generic mapping or address");
    assert(@catch(gpu::get_allocation_span(&device, allocation))
        == gpu::INVALID_ARGUMENT,
        "texture memory cannot be borrowed as a generic data span");

    gpu::TextureHandle first = gpu::create_placed_texture(
        device:     &device,
        desc:       &texture_desc,
        allocation: allocation,
        offset:     0,
    )!!;
    assert(@catch(gpu::create_placed_texture(
        device:     &device,
        desc:       &texture_desc,
        allocation: allocation,
        offset:     0,
    )) == gpu::INVALID_ARGUMENT,
        "live placement ranges cannot overlap");
    assert(@catch(gpu::create_placed_texture(
        device:     &device,
        desc:       &texture_desc,
        allocation: allocation,
        offset:     allocation_desc.size,
    )) == gpu::INVALID_ARGUMENT,
        "placement ranges must fit in the allocation");
    if (requirements.alignment > 1) {
        assert(@catch(gpu::create_placed_texture(
            device:     &device,
            desc:       &texture_desc,
            allocation: allocation,
            offset:     1,
        )) == gpu::INVALID_ARGUMENT,
            "placement offsets must satisfy texture alignment");
    }

    gpu::TextureHandle second = gpu::create_placed_texture(
        device:     &device,
        desc:       &texture_desc,
        allocation: allocation,
        offset:     requirements.size,
    )!!;
    gpu::destroy_texture(&device, first)!!;
    gpu::destroy_texture(&device, second)!!;
    gpu::GpuAllocation stale = allocation;
    gpu::free_allocation(&device, &allocation)!!;
    assert(@catch(gpu::create_placed_texture(
        device:     &device,
        desc:       &texture_desc,
        allocation: stale,
        offset:     0,
    )) == gpu::INVALID_HANDLE,
        "released placement storage is stale");

    gpu::AllocationDesc generic_desc = {
        .size         = requirements.size,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = texture_desc.access,
    };
    gpu::GpuAllocation generic =
        gpu::allocate_memory(&device, &generic_desc)!!;
    assert(@catch(gpu::create_placed_texture(
        device:     &device,
        desc:       &texture_desc,
        allocation: generic,
        offset:     0,
    )) == gpu::INVALID_ARGUMENT,
        "generic data allocations cannot back textures");
    gpu::free_allocation(&device, &generic)!!;
}

fn void placed_texture_validation_precedes_native_mutation() @test {
    PlacementFixture fixture;
    gpu::GpuAllocation allocation = init_placement_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    vk::TexturePlacementNativeOps ops = placement_probe_ops(&fixture.probe);
    gpu::TextureDesc desc = placement_test_desc();
    gpu::TextureRequirements requirements =
        placement_test_requirements(&fixture);

    requirements.alignment = 0;
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_ARGUMENT);
    requirements.alignment = 64;

    requirements.dedicated_only = true;
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::UNSUPPORTED_FEATURE);
    requirements.dedicated_only = false;
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       1,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       256,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       usz::max - 63,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_ARGUMENT);

    requirements.compatibility = vk::texture_compatibility(
        fixture.state.allocations.owner,
        0b0100,
    );
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_ARGUMENT);
    requirements.compatibility = vk::texture_compatibility(
        fixture.state.allocations.owner,
        0b0010,
    );
    requirements.alignment = 128;
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_ARGUMENT);
    requirements.alignment = 64;
    fixture.allocations[0].value.texture_memory = false;
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_ARGUMENT);
    fixture.allocations[0].value.texture_memory = true;
    fixture.allocations[0].value.dedicated_texture = true;
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_ARGUMENT);
    fixture.allocations[0].value.dedicated_texture = false;


    gpu::TextureDesc wrong_access = desc;
    wrong_access.access = { .graphics };
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &wrong_access,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_ARGUMENT);
    gpu::GpuAllocation stale = allocation;
    stale.generation++;
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    stale,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_HANDLE);

    fixture.state.textures.next_new = 1;
    fixture.allocations[0].value.live_placements = uint::max;
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_RESOURCE_STATE);
    fixture.allocations[0].value.live_placements = 0;

    fixture.textures[0] = {
        .used       = true,
        .generation = 1,
        .value = {
            .placed           = true,
            .placement        = allocation,
            .placement_offset = 0,
            .allocation_size  = 64,
        },
    };
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_ARGUMENT);
    fixture.textures[0].used = false;
    fixture.state.deferred.entries = fixture.deferred[..];
    fixture.state.deferred.count = 1;
    fixture.deferred[0] = {
        .kind             = vk::DeferredKind.PLACED_IMAGE,
        .placement        = allocation,
        .placement_offset = 0,
        .placement_size   = 64,
    };
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(fixture.probe.create_calls == 0
        && fixture.probe.view_calls == 0
        && fixture.allocations[0].value.live_placements == 0,
        "every invalid placement stops before native mutation");
}

fn void placed_texture_publication_rolls_back_native_failures() @test {
    PlacementFixture fixture;
    gpu::GpuAllocation allocation = init_placement_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    vk::TexturePlacementNativeOps ops = placement_probe_ops(&fixture.probe);
    gpu::TextureDesc desc = placement_test_desc();
    gpu::TextureRequirements requirements =
        placement_test_requirements(&fixture);

    fixture.probe.create_result = vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::OUT_OF_DEVICE_MEMORY);
    assert(fixture.probe.create_calls == 1
        && fixture.probe.view_calls == 0
        && fixture.probe.destroy_image_calls == 0
        && fixture.allocations[0].value.live_placements == 0,
        "image creation failure publishes and retains nothing");

    fixture.probe = {
        .view_result = vk::Result.ERROR_OUT_OF_HOST_MEMORY,
    };
    ops = placement_probe_ops(&fixture.probe);
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::OUT_OF_HOST_MEMORY);
    assert(fixture.probe.create_calls == 1
        && fixture.probe.view_calls == 1
        && fixture.probe.destroy_image_calls == 1
        && fixture.probe.destroy_view_calls == 0
        && fixture.allocations[0].value.live_placements == 0,
        "view failure destroys the image and releases placement retention");

    fixture.probe = {};
    ops = placement_probe_ops(&fixture.probe);
    fixture.state.textures.next_new = 1;
    assert(@catch(vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )) == gpu::SLOT_TABLE_FULL);
    assert(fixture.probe.destroy_view_calls == 1
        && fixture.probe.destroy_image_calls == 1
        && fixture.allocations[0].value.live_placements == 0,
        "publication failure rolls back view, image, and placement retention");

    fixture.probe = {};
    ops = placement_probe_ops(&fixture.probe);
    fixture.state.textures.next_new = 0;
    gpu::TextureHandle texture = vk::create_placed_texture_for_state_with_ops(
        state:        &fixture.state,
        desc:         &desc,
        placement:    allocation,
        offset:       0,
        requirements: requirements,
        ops:          &ops,
    )!!;
    assert(texture.is_valid()
        && fixture.probe.create_calls == 1
        && fixture.probe.view_calls == 1
        && fixture.allocations[0].value.live_placements == 1,
        "successful placement publishes once and retains its allocation");
    vk::TextureSlot* slot = fixture.state.textures.get(texture)!!;
    ops.destroy_view(ops.user_data, fixture.state.device, slot.default_view);
    ops.destroy_image(ops.user_data, fixture.state.device, slot.image);
    fixture.state.textures.free(texture)!!;
    fixture.allocations[0].value.live_placements--;
}
