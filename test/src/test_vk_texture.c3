module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vma;
import vk;
import std::atomic;

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
    PlacementNativeProbe  probe;
}

struct TextureBarrierEmitCapture {
    uint                    calls;
    vk::ImageMemoryBarrier2 native;
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

fn void capture_texture_barrier_emit(
    void* user_data,
    vk::ImageMemoryBarrier2* native,
) {
    TextureBarrierEmitCapture* capture =
        (TextureBarrierEmitCapture*)user_data;
    capture.calls++;
    capture.native = *native;
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
    fixture.state.allocations.owner = gpu::internal::resource_owner_tag(
        OWNER,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
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
    fixture.state.textures.owner = gpu::internal::resource_owner_tag(
        OWNER,
        gpu::internal::ResourceOwnerKind.TEXTURE,
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
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_texture_diagnostic_test",
        .debug_callback    = &capture_texture_diagnostic,
        .debug_user_data   = capture,
    };
    return create_test_device(&desc)!!;
}

fn gpu::Device make_texture_test_device() {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_texture_test",
    };
    return create_test_device(&desc)!!;
}

fn void assert_texture_state_accepted(
    gpu::TextureState texture_state,
    vk::TextureSlot* slot,
    vk::QueueFlags flags,
    bool before,
) {
    vk::TextureBarrierRejection rejection = vk::texture_state_rejection(
        texture_state: texture_state,
        slot:          slot,
        flags:         flags,
        before:        before,
    );
    assert(rejection.field == null, "texture state is accepted by the semantic matrix");
}

fn void assert_texture_state_rejected(
    gpu::TextureState texture_state,
    vk::TextureSlot* slot,
    vk::QueueFlags flags,
    bool before,
    ZString expected_field,
) {
    vk::TextureBarrierRejection rejection = vk::texture_state_rejection(
        texture_state: texture_state,
        slot:          slot,
        flags:         flags,
        before:        before,
    );
    assert(rejection.field != null,
        "invalid texture state is rejected by the semantic matrix");
    assert(rejection.field.str_view() == expected_field.str_view(),
        "texture state rejection identifies the exact nested field");
}

fn void texture_shader_state_matrix_accepts_combinations_and_access_modes() @test {
    vk::TextureSlot slot = {
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled, .storage },
    };
    vk::QueueFlags graphics_compute = (vk::QueueFlags)(
        (uint)vk::QueueFlagBits.GRAPHICS_BIT
            | (uint)vk::QueueFlagBits.COMPUTE_BIT
    );
    gpu::StageMask[5] sampled_stages = {
        { .vertex_shader },
        { .fragment_shader },
        { .compute },
        { .vertex_shader, .fragment_shader },
        { .vertex_shader, .fragment_shader, .compute },
    };
    foreach (stages : sampled_stages) {
        assert_texture_state_accepted(
            texture_state: gpu::sampled_at(stages),
            slot:          &slot,
            flags:         graphics_compute,
            before:        false,
        );
    }

    gpu::TextureAccess[3] storage_access = {
        { .read },
        { .write },
        { .read, .write },
    };
    foreach (access : storage_access) {
        assert_texture_state_accepted(
            texture_state: gpu::storage_at({ .compute }, access),
            slot:          &slot,
            flags:         graphics_compute,
            before:        false,
        );
    }

    assert_texture_state_accepted(
        texture_state: gpu::sampled_at({ .all }),
        slot:          &slot,
        flags:         graphics_compute,
        before:        false,
    );
    assert_texture_state_accepted(
        texture_state: gpu::storage_at({ .all }, { .read, .write }),
        slot:          &slot,
        flags:         graphics_compute,
        before:        false,
    );
}

fn void texture_transfer_state_matrix_requires_exact_access() @test {
    vk::TextureSlot slot = {
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .transfer_src, .transfer_dst },
    };
    vk::QueueFlags transfer_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.TRANSFER_BIT;
    gpu::TextureState source = {
        .layout = gpu::TextureLayout.TRANSFER_SOURCE,
        .stages = { .transfer },
        .access = { .read },
    };
    gpu::TextureState destination = {
        .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
        .stages = { .transfer },
        .access = { .write },
    };
    assert_texture_state_accepted(
        texture_state: source,
        slot:          &slot,
        flags:         transfer_only,
        before:        true,
    );
    assert_texture_state_accepted(
        texture_state: destination,
        slot:          &slot,
        flags:         transfer_only,
        before:        false,
    );

    gpu::TextureAccess[3] invalid_source_access = {
        {},
        { .write },
        { .read, .write },
    };
    foreach (access : invalid_source_access) {
        source.access = access;
        assert_texture_state_rejected(
            texture_state: source,
            slot:          &slot,
            flags:         transfer_only,
            before:        true,
            expected_field: "TextureBarrier.before.access",
        );
    }
    gpu::TextureAccess[3] invalid_destination_access = {
        {},
        { .read },
        { .read, .write },
    };
    foreach (access : invalid_destination_access) {
        destination.access = access;
        assert_texture_state_rejected(
            texture_state: destination,
            slot:          &slot,
            flags:         transfer_only,
            before:        false,
            expected_field: "TextureBarrier.after.access",
        );
    }

    source = {
        .layout = gpu::TextureLayout.TRANSFER_SOURCE,
        .stages = { .all },
        .access = { .read },
    };
    assert_texture_state_accepted(
        texture_state: source,
        slot:          &slot,
        flags:         transfer_only,
        before:        true,
    );
    source.stages = { .all = true, .transfer = true };
    assert_texture_state_rejected(
        texture_state: source,
        slot:          &slot,
        flags:         transfer_only,
        before:        true,
        expected_field: "TextureBarrier.before.stages",
    );
}

fn void texture_state_matrix_rejects_reserved_stages_and_access() @test {
    vk::TextureSlot slot = {
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled },
    };
    vk::QueueFlags graphics_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.GRAPHICS_BIT;
    gpu::TextureState sampled = gpu::sampled_at({ .fragment_shader });

    sampled.stages = { .host };
    assert_texture_state_rejected(
        texture_state: sampled,
        slot:          &slot,
        flags:         graphics_only,
        before:        true,
        expected_field: "TextureBarrier.before.stages",
    );
    sampled.stages = { .present };
    assert_texture_state_rejected(
        texture_state: sampled,
        slot:          &slot,
        flags:         graphics_only,
        before:        false,
        expected_field: "TextureBarrier.after.stages",
    );
    sampled.stages = (gpu::StageMask)0x200u;
    assert_texture_state_rejected(
        texture_state: sampled,
        slot:          &slot,
        flags:         graphics_only,
        before:        false,
        expected_field: "TextureBarrier.after.stages",
    );
    sampled.stages = { .fragment_shader };
    sampled.access = (gpu::TextureAccess)0x4u;
    assert_texture_state_rejected(
        texture_state: sampled,
        slot:          &slot,
        flags:         graphics_only,
        before:        true,
        expected_field: "TextureBarrier.before.access",
    );

    gpu::TextureLayout unknown_layout = bitcast(999, gpu::TextureLayout);
    sampled = { .layout = unknown_layout };
    assert_texture_state_rejected(
        texture_state: sampled,
        slot:          &slot,
        flags:         graphics_only,
        before:        true,
        expected_field: "TextureBarrier.before.layout",
    );
    sampled = { .layout = gpu::TextureLayout.UNDEFINED };
    assert_texture_state_rejected(
        texture_state: sampled,
        slot:          &slot,
        flags:         graphics_only,
        before:        false,
        expected_field: "TextureBarrier.after.layout",
    );
}

fn void texture_attachment_and_present_state_matrix_enforces_texture_kind() @test {
    vk::QueueFlags graphics_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.GRAPHICS_BIT;
    gpu::TextureState color = {
        .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
        .stages = { .color_output },
        .access = { .read, .write },
    };
    gpu::TextureState depth = {
        .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
        .stages = { .depth_output },
        .access = { .write },
    };
    vk::TextureSlot color_slot = {
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .color_attach },
    };
    vk::TextureSlot depth_slot = {
        .format = gpu::Format.D32_FLOAT,
        .usage  = { .depth_attach },
    };
    assert_texture_state_accepted(
        texture_state: color,
        slot:          &color_slot,
        flags:         graphics_only,
        before:        false,
    );
    assert_texture_state_accepted(
        texture_state: depth,
        slot:          &depth_slot,
        flags:         graphics_only,
        before:        false,
    );

    color_slot.format = gpu::Format.D32_FLOAT;
    assert_texture_state_rejected(
        texture_state: color,
        slot:          &color_slot,
        flags:         graphics_only,
        before:        false,
        expected_field: "TextureBarrier.after.layout",
    );
    color_slot = { .format = gpu::Format.RGBA8_UNORM };
    assert_texture_state_rejected(
        texture_state: color,
        slot:          &color_slot,
        flags:         graphics_only,
        before:        false,
        expected_field: "TextureBarrier.after.layout",
    );
    depth_slot.format = gpu::Format.RGBA8_UNORM;
    assert_texture_state_rejected(
        texture_state: depth,
        slot:          &depth_slot,
        flags:         graphics_only,
        before:        true,
        expected_field: "TextureBarrier.before.layout",
    );
    depth_slot = { .format = gpu::Format.D32_FLOAT };
    assert_texture_state_rejected(
        texture_state: depth,
        slot:          &depth_slot,
        flags:         graphics_only,
        before:        true,
        expected_field: "TextureBarrier.before.layout",
    );

    assert_texture_state_rejected(
        texture_state: { .layout = gpu::TextureLayout.PRESENT },
        slot:          &color_slot,
        flags:         graphics_only,
        before:        false,
        expected_field: "TextureBarrier.after.layout",
    );
}

fn void texture_create_destroy() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
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
        (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(&device);
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    assert((uint)slot.access == (uint)desc.access,
        "texture slot retains the descriptor access roles");
    gpu::destroy_texture(&device, texture)!!;
}

fn void texture_unsupported_desc_rejected() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc depth_storage = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.D32_FLOAT,
        .usage     = { .depth_attach, .storage },
    };
    fault storage_fault = @catch(gpu::create_texture(&device, &depth_storage));
    assert(storage_fault == gpu::INVALID_ARGUMENT, "storage usage on a depth format faults INVALID_ARGUMENT");

    gpu::TextureDesc color_with_depth_usage = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .depth_attach },
    };
    fault cross_fault = @catch(gpu::create_texture(&device, &color_with_depth_usage));
    assert(cross_fault == gpu::INVALID_ARGUMENT, "depth_attach on a color format faults INVALID_ARGUMENT");

    gpu::TextureDesc transfer_only = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::TextureHandle staging_target = gpu::create_texture(&device, &transfer_only)!!;
    fault view_fault = @catch(gpu::create_texture_view(&device, staging_target, null));
    assert(view_fault == gpu::INVALID_ARGUMENT,
        "a view on a non-shader-visible texture faults INVALID_ARGUMENT");
    gpu::destroy_texture(&device, staging_target)!!;
}

fn void texture_stale_handle_rejected() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
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

fn void texture_barrier_records_texture_states() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureBarrier to_general = {
        .texture = texture,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = gpu::storage_at(
            { .compute },
            { .read, .write },
        ),
    };
    gpu::cmd_texture_barrier(&cmd, &to_general)!!;

    gpu::TextureBarrier general_to_general = {
        .texture = texture,
        .before  = gpu::storage_at(
            { .compute },
            { .read, .write },
        ),
        .after   = gpu::storage_at(
            { .compute },
            { .read, .write },
        ),
    };
    gpu::cmd_texture_barrier(&cmd, &general_to_general)!!;

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;

    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void texture_multi_mip_create_destroy() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
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
        .access     = { .graphics, .compute },
        .width      = 16,
        .height     = 16,
        .mip_levels = 5,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureBarrier to_general = {
        .texture = texture,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = gpu::storage_at(
            { .compute },
            { .read, .write },
        ),
    };
    gpu::cmd_texture_barrier(&cmd, &to_general)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;

    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
}

fn void texture_barrier_valid_path_lowers_and_emits_once() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .access       = { .graphics },
        .width        = 16,
        .height       = 16,
        .mip_levels   = 4,
        .array_layers = 3,
        .format       = gpu::Format.RGBA8_UNORM,
        .usage        = { .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(&device);
    vk::TextureSlot* slot = state.textures.get(texture)!!;
    TextureBarrierEmitCapture capture;
    vk::set_texture_barrier_emit_hook(
        state,
        &capture_texture_barrier_emit,
        &capture,
    );
    defer vk::set_texture_barrier_emit_hook(state, null, null);
    vk::reset_texture_barrier_work_counters(state);

    gpu::TextureBarrier barrier = {
        .texture = texture,
        .view = {
            .base_mip    = 1,
            .mip_count   = 2,
            .base_layer  = 1,
            .layer_count = 1,
        },
        .before = { .layout = gpu::TextureLayout.UNDEFINED },
        .after  = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    };
    gpu::cmd_texture_barrier(&commands, &barrier)!!;

    vk::TextureBarrierWorkCounters counters =
        vk::texture_barrier_work_counters(state);
    assert(counters.helper_calls == 1, "barrier uses one validation and lowering helper");
    assert(counters.handle_resolutions == 1, "barrier resolves its texture handle once");
    assert(counters.access_validations == 1, "barrier validates recording access once");
    assert(counters.range_resolutions == 1, "barrier resolves its view range once");
    assert(counters.state_validations == 2,
        "barrier validates each texture state once");
    assert(counters.native_assemblies == 1, "barrier assembles one native barrier");
    assert(counters.native_emissions == 1, "barrier emits one native barrier");
    assert(capture.calls == 1, "native emission hook receives exactly one barrier");
    assert(capture.native.src_stage_mask
            == vk::PipelineStageFlagBits2.STAGE_2_NONE,
        "undefined source lowers to no source stages");
    assert(capture.native.src_access_mask
            == vk::AccessFlagBits2.ACCESS_2_NONE,
        "undefined source lowers to no source access");
    assert(capture.native.dst_stage_mask
            == vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT,
        "transfer destination lowers to the transfer stage");
    assert(capture.native.dst_access_mask
            == vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT,
        "transfer destination lowers to transfer-write access");
    assert(capture.native.old_layout == vk::ImageLayout.UNDEFINED,
        "undefined source lowers to the undefined layout");
    assert(capture.native.new_layout == vk::ImageLayout.TRANSFER_DST_OPTIMAL,
        "transfer destination lowers to its optimal layout");
    assert(capture.native.src_queue_family_index == vk::QUEUE_FAMILY_IGNORED
        && capture.native.dst_queue_family_index == vk::QUEUE_FAMILY_IGNORED,
        "same-queue barriers ignore ownership transfer indices");
    assert(capture.native.image == slot.image,
        "native barrier uses the resolved texture image");
    assert(capture.native.subresource_range.aspect_mask
            == vk::ImageAspectFlagBits.COLOR_BIT
        && capture.native.subresource_range.base_mip_level == 1
        && capture.native.subresource_range.level_count == 2
        && capture.native.subresource_range.base_array_layer == 1
        && capture.native.subresource_range.layer_count == 1,
        "native barrier preserves the exact resolved color subresource range");
}

fn void texture_barrier_emit_seam_captures_composed_state_masks() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .access    = { .graphics, .compute },
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = {
            .transfer_dst,
            .sampled,
            .storage,
            .color_attach,
        },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(&device);
    vk::TextureSlot* slot = state.textures.get(texture)!!;
    TextureBarrierEmitCapture capture;
    vk::set_texture_barrier_emit_hook(
        state,
        &capture_texture_barrier_emit,
        &capture,
    );
    defer vk::set_texture_barrier_emit_hook(state, null, null);

    gpu::TextureBarrier sampled = {
        .texture = texture,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = gpu::sampled_at({ .vertex_shader, .fragment_shader }),
    };
    gpu::cmd_texture_barrier(&commands, &sampled)!!;
    vk::PipelineStageFlags2 sampled_stages = (vk::PipelineStageFlags2)(
        (ulong)vk::PipelineStageFlagBits2.STAGE_2_VERTEX_SHADER_BIT
            | (ulong)vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT
    );
    assert(capture.native.dst_stage_mask == sampled_stages,
        "multi-stage sampled state preserves both shader stages");
    assert(capture.native.dst_access_mask
            == vk::AccessFlagBits2.ACCESS_2_SHADER_SAMPLED_READ_BIT,
        "sampled state lowers to sampled-read access");
    assert(capture.native.new_layout
            == vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
        "sampled state lowers to the shader-read layout");

    gpu::TextureBarrier storage = {
        .texture = texture,
        .before  = gpu::storage_at({ .compute }, { .write }),
        .after   = gpu::storage_at({ .compute }, { .read }),
    };
    gpu::cmd_texture_barrier(&commands, &storage)!!;
    assert(capture.native.src_stage_mask
            == vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT
        && capture.native.dst_stage_mask
            == vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
        "storage states preserve their compute stages");
    assert(capture.native.src_access_mask
            == vk::AccessFlagBits2.ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        "write-only storage source lowers to storage-write access");
    assert(capture.native.dst_access_mask
            == vk::AccessFlagBits2.ACCESS_2_SHADER_STORAGE_READ_BIT,
        "read-only storage destination lowers to storage-read access");
    assert(capture.native.old_layout == vk::ImageLayout.GENERAL
        && capture.native.new_layout == vk::ImageLayout.GENERAL,
        "storage states lower to the general layout");

    slot.swapchain_owned = true;
    gpu::TextureBarrier present = {
        .texture = texture,
        .before  = { .layout = gpu::TextureLayout.PRESENT },
        .after   = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    };
    gpu::cmd_texture_barrier(&commands, &present)!!;
    slot.swapchain_owned = false;
    assert(capture.calls == 3,
        "injected seam captures every composed texture barrier");
    assert(capture.native.src_stage_mask
            == vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        "present source lowers to the fixed color-output source scope");
    assert(capture.native.src_access_mask
            == vk::AccessFlagBits2.ACCESS_2_NONE,
        "present source lowers to no source access");
    assert(capture.native.old_layout == vk::ImageLayout.PRESENT_SRC_KHR,
        "present source lowers to the native presentation layout");
    assert(capture.native.dst_stage_mask
            == vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT
        && capture.native.dst_access_mask
            == vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT
        && capture.native.new_layout
            == vk::ImageLayout.TRANSFER_DST_OPTIMAL,
        "present-to-transfer lowers the exact destination scope and layout");
}

fn void texture_barrier_invalid_state_rolls_back_post_retain_work() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .access     = { .graphics },
        .width      = 8,
        .height     = 8,
        .mip_levels = 2,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(&device);
    vk::TextureSlot* slot = state.textures.get(texture)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    uint references_before = record.scratch.reference_count;
    uint tracked_references_before = slot.tracked_command_references.load(
        AtomicOrdering.RELAXED,
    );
    TextureBarrierEmitCapture capture;
    vk::set_texture_barrier_emit_hook(
        state,
        &capture_texture_barrier_emit,
        &capture,
    );
    defer vk::set_texture_barrier_emit_hook(state, null, null);
    vk::reset_texture_barrier_work_counters(state);

    gpu::TextureBarrier barrier = {
        .texture = texture,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.SAMPLED,
            .stages = { .fragment_shader },
        },
    };
    assert(@catch(gpu::cmd_texture_barrier(&commands, &barrier))
            == gpu::INVALID_ARGUMENT,
        "sampled state without read access faults after retaining the texture");

    vk::TextureBarrierWorkCounters counters =
        vk::texture_barrier_work_counters(state);
    assert(counters.helper_calls == 1,
        "invalid state uses one validation and lowering helper");
    assert(counters.handle_resolutions == 1,
        "invalid state resolves its texture handle once");
    assert(counters.access_validations == 1,
        "invalid state validates recording access once");
    assert(counters.range_resolutions == 1,
        "invalid state resolves its view range once");
    assert(counters.state_validations == 2,
        "invalid state validates both texture states");
    assert(counters.native_assemblies == 0,
        "invalid destination state assembles no native barrier");
    assert(counters.native_emissions == 0,
        "invalid destination state emits no native barrier");
    assert(capture.calls == 0, "invalid destination state never reaches emission");
    assert(record.scratch.reference_count == references_before,
        "failed state validation rolls back the command reference");
    assert(slot.tracked_command_references.load(AtomicOrdering.RELAXED)
            == tracked_references_before,
        "failed state validation rolls back the texture validation reference");
}

fn void texture_barrier_rejects_unsupported_texture_states() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .access    = { .graphics },
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;

    gpu::TextureBarrier sampled = gpu::texture_transition(
        texture: texture,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   gpu::sampled_at({ .fragment_shader }),
    )!!;
    assert(@catch(gpu::cmd_texture_barrier(&commands, &sampled))
        == gpu::INVALID_ARGUMENT,
        "sampled use requires sampled texture usage");

    gpu::TextureBarrier present = gpu::texture_transition(
        texture: texture,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   { .layout = gpu::TextureLayout.PRESENT },
    )!!;
    assert(@catch(gpu::cmd_texture_barrier(&commands, &present))
        == gpu::INVALID_ARGUMENT,
        "present use requires a swapchain texture");
}
fn void texture_barrier_stale_handle_rejected() @test {
    TextureDiagnosticCapture diagnostic = {
        .expected_operation      = "cmd_texture_barrier",
        .expected_rejected_field = "TextureBarrier.texture",
        .expected_invariant      = "must identify one live texture",
    };
    gpu::Device device = make_texture_diagnostic_device(&diagnostic);
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    gpu::destroy_texture(&device, texture)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(&device);
    vk::CommandRecord* record = vk::resolve_command(&cmd)!!;
    uint references_before = record.scratch.reference_count;
    TextureBarrierEmitCapture emission;
    vk::set_texture_barrier_emit_hook(
        state,
        &capture_texture_barrier_emit,
        &emission,
    );
    defer vk::set_texture_barrier_emit_hook(state, null, null);
    vk::reset_texture_barrier_work_counters(state);

    gpu::TextureBarrier barrier = {
        .texture = texture,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = gpu::storage_at(
            { .compute },
            { .read, .write },
        ),
    };
    fault barrier_fault = @catch(gpu::cmd_texture_barrier(&cmd, &barrier));
    assert(barrier_fault == gpu::INVALID_HANDLE, "barrier on a destroyed texture faults INVALID_HANDLE");
    assert(diagnostic.calls == 1 && diagnostic.operation_matches
            && diagnostic.rejected_field_matches
            && diagnostic.invariant_matches,
        "stale texture emits one exact texture-barrier diagnostic");
    assert(diagnostic.has_fault && diagnostic.public_fault == barrier_fault
            && diagnostic.public_contract,
        "stale texture diagnostic preserves INVALID_HANDLE");
    assert(diagnostic.resource_kind == gpu::DebugResourceKind.TEXTURE
            && !diagnostic.has_public_identity
            && diagnostic.index == 0
            && diagnostic.generation == 0,
        "stale texture diagnostic claims no resolved resource identity");
    vk::TextureBarrierWorkCounters counters =
        vk::texture_barrier_work_counters(state);
    assert(counters.helper_calls == 1
            && counters.handle_resolutions == 1,
        "stale texture performs one authoritative lookup attempt");
    assert(counters.access_validations == 0
            && counters.range_resolutions == 0
            && counters.state_validations == 0
            && counters.native_assemblies == 0
            && counters.native_emissions == 0
            && emission.calls == 0,
        "stale texture stops before range, state, assembly, or emission work");
    assert(record.scratch.reference_count == references_before,
        "stale texture retains no command reference");

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
}

fn void attachment_views_own_exact_subresources_and_retain_textures() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .access     = { .graphics },
        .width      = 16,
        .height     = 8,
        .mip_levels = 2,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .color_attach },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    gpu::AttachmentViewDesc view_desc = {
        .texture   = texture,
        .mip_level = 1,
    };
    gpu::AttachmentViewHandle view = gpu::create_attachment_view(
        &device,
        &view_desc,
    )!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(
        &device,
    );
    vk::AttachmentViewSlot* slot = state.attachment_views.get(view)!!;
    assert(slot.texture == texture && slot.width == 8 && slot.height == 4,
        "attachment slots pin exact immutable subresource metadata");
    assert(slot.owns_native,
        "non-default attachment subresources own a cold-created native view");
    assert(@catch(gpu::destroy_texture(&device, texture))
        == gpu::RESOURCE_IN_USE,
        "a live attachment view retains its texture");

    gpu::destroy_attachment_view(&device, view)!!;
    assert(@catch(gpu::destroy_attachment_view(&device, view))
        == gpu::INVALID_HANDLE,
        "destroyed attachment handles are stale");
    gpu::destroy_texture(&device, texture)!!;
}

fn void default_attachment_view_aliases_texture_native_view() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .access    = { .graphics },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    gpu::AttachmentViewDesc view_desc = { .texture = texture };
    gpu::AttachmentViewHandle view = gpu::create_attachment_view(
        &device,
        &view_desc,
    )!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(
        &device,
    );
    vk::AttachmentViewSlot* slot = state.attachment_views.get(view)!!;
    assert(!slot.owns_native
        && slot.native == state.textures.get(texture)!!.default_view,
        "single-subresource attachments alias the texture-owned default view");
    gpu::destroy_attachment_view(&device, view)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void swapchain_owned_attachment_view_rejects_caller_destruction() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .access    = { .graphics },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    gpu::AttachmentViewDesc view_desc = { .texture = texture };
    gpu::AttachmentViewHandle view = gpu::create_attachment_view(
        &device,
        &view_desc,
    )!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(
        &device,
    );
    vk::AttachmentViewSlot* slot = state.attachment_views.get(view)!!;
    slot.swapchain_owned = true;

    assert(@catch(gpu::destroy_attachment_view(&device, view))
        == gpu::INVALID_RESOURCE_STATE,
        "borrowed swapchain attachment views reject caller destruction");
    assert(state.attachment_views.get(view)!! == slot,
        "rejected borrowed-view destruction preserves the handle");

    slot.swapchain_owned = false;
    gpu::destroy_attachment_view(&device, view)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void attachment_view_publication_failure_rolls_back_cold_view() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .access     = { .graphics },
        .width      = 8,
        .height     = 8,
        .mip_levels = 2,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .color_attach },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::AttachmentViewDesc view_desc = {
        .texture   = texture,
        .mip_level = 1,
    };
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(
        &device,
    );
    vk::TextureSlot* texture_slot = state.textures.get(texture)!!;
    uint references_before = texture_slot.tracked_command_references.load(
        AtomicOrdering.RELAXED,
    );
    uint next_new = state.attachment_views.next_new;
    uint free_head = state.attachment_views.free_head;
    state.attachment_views.next_new = (uint)state.attachment_views.slots.len;
    state.attachment_views.free_head = 0;

    assert(@catch(gpu::create_attachment_view(&device, &view_desc))
        == gpu::SLOT_TABLE_FULL,
        "attachment publication capacity faults after cold native creation");
    assert(texture_slot.tracked_command_references.load(AtomicOrdering.RELAXED)
        == references_before,
        "failed attachment publication does not retain the texture");
    state.attachment_views.next_new = next_new;
    state.attachment_views.free_head = free_head;

    gpu::AttachmentViewHandle retry = gpu::create_attachment_view(
        &device,
        &view_desc,
    )!!;
    gpu::destroy_attachment_view(&device, retry)!!;
}

fn void foreign_attachment_view_destroy_rejects_without_consuming() @test {
    gpu::Device owner = make_texture_test_device();
    defer destroy_test_device(&owner)!!;
    gpu::Device foreign = make_texture_test_device();
    defer destroy_test_device(&foreign)!!;
    gpu::TextureDesc desc = {
        .access    = { .graphics },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::TextureHandle texture = gpu::create_texture(&owner, &desc)!!;
    gpu::AttachmentViewDesc view_desc = { .texture = texture };
    gpu::AttachmentViewHandle view = gpu::create_attachment_view(
        &owner,
        &view_desc,
    )!!;

    assert(@catch(gpu::destroy_attachment_view(&foreign, view))
        == gpu::INVALID_HANDLE,
        "foreign devices cannot consume an attachment-view handle");
    gpu::destroy_attachment_view(&owner, view)!!;
    gpu::destroy_texture(&owner, texture)!!;
}

fn void recorded_attachment_view_destruction_is_retryable() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .access    = { .graphics },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    gpu::AttachmentViewDesc view_desc = { .texture = texture };
    gpu::AttachmentViewHandle view = gpu::create_attachment_view(
        &device,
        &view_desc,
    )!!;
    gpu::ColorTargetDesc[1] colors = { { .view = view } };
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = 4,
        .height = 4,
    };
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    gpu::GraphicsState graphics =
        gpu::full_render_graphics_state(pass.width, pass.height)!!;
    gpu::cmd_begin_render_pass(&commands, &pass, &graphics)!!;

    assert(@catch(gpu::destroy_attachment_view(&device, view))
        == gpu::RESOURCE_IN_USE,
        "recorded attachment references make destruction retryable");
    gpu::discard_commands(&commands)!!;
    gpu::destroy_attachment_view(&device, view)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void raised_texture_capacity_holds_more_than_default() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .texture_capacity  = 2048,
        .application_name  = "gpu_texture_capacity_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    const uint COUNT = 1500;
    gpu::TextureHandle[] handles = mem::new_array(gpu::TextureHandle, COUNT);
    defer mem::free(handles.ptr);
    gpu::TextureDesc tex_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled },
    };
    foreach (&h : handles) *h = gpu::create_texture(&device, &tex_desc)!!;
    foreach (h : handles) gpu::destroy_texture(&device, h)!!;
}

fn void transfer_only_texture_round_trips() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
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
    upload_test_texture_data(
        device:   &device,
        dst:      tex,
        mip:      0,
        data:     pixels[..],
        next_use: {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    )!!;

    char[64] readback;
    readback_test_texture_data(
        device:   &device,
        src:      tex,
        mip:      0,
        out_data: readback[..],
        from_use: {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
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
        assert(support.sample_counts.one, "single-sample support is reported");
        bool is_depth = format == gpu::Format.D32_FLOAT;
        gpu::TextureUsage attachment_usage = is_depth
            ? (gpu::TextureUsage){ .depth_attach }
            : (gpu::TextureUsage){ .color_attach };
        bool attachment_supported = is_depth
            ? support.features.depth_attach
            : support.features.color_attach;
        gpu::SampleCount[6] sample_counts = {
            gpu::SampleCount.TWO,
            gpu::SampleCount.FOUR,
            gpu::SampleCount.EIGHT,
            gpu::SampleCount.SIXTEEN,
            gpu::SampleCount.THIRTY_TWO,
            gpu::SampleCount.SIXTY_FOUR,
        };
        bool[6] reported_counts = {
            support.sample_counts.two,
            support.sample_counts.four,
            support.sample_counts.eight,
            support.sample_counts.sixteen,
            support.sample_counts.thirty_two,
            support.sample_counts.sixty_four,
        };
        foreach (i, sample_count : sample_counts) {
            gpu::TextureDesc sample_desc = {
                .access       = { .graphics, .compute, .transfer },
                .width        = 4,
                .height       = 4,
                .format       = format,
                .usage        = attachment_usage,
                .sample_count = sample_count,
            };
            bool exact_support = attachment_supported
                && gpu::supports_texture_desc(&device, &sample_desc)!!;
            assert(reported_counts[i] == exact_support,
                "reported sample counts match exact attachment descriptors");
        }

        foreach (usage : usages) {
            if (is_depth && (usage.storage || usage.color_attach)) continue;
            if (!is_depth && usage.depth_attach) continue;

            gpu::TextureDesc desc = {
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
        && (uint)undefined_support.sample_counts == 0,
        "undefined format reports empty support");

    gpu::TextureDesc invalid = {
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

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
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

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    state.debug_callback = null;
    fault without_callback = @catch(gpu::create_texture(&device, &unsupported));
    assert(without_callback == with_callback, "callback presence does not alter the capability fault");
    assert(capture.calls == 1, "null callback emits no additional diagnostic");
}

fn void texture_barrier_invalid_use_diagnostic_identifies_resource() @test {
    TextureDiagnosticCapture capture = {
        .expected_operation      = "cmd_texture_barrier",
        .expected_debug_name     = "texture_barrier_diagnostic",
        .expected_rejected_field = "TextureBarrier.after.layout",
        .expected_invariant      = "UNDEFINED is source-only",
    };
    gpu::Device device = make_texture_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .access     = { .graphics, .compute },
        .width      = 8,
        .height     = 8,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .debug_name = "texture_barrier_diagnostic",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::TextureBarrier barrier = {
        .texture = texture,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = { .layout = gpu::TextureLayout.UNDEFINED },
    };
    fault with_callback = @catch(gpu::cmd_texture_barrier(
        &commands,
        &barrier,
    ));

    assert(with_callback == gpu::INVALID_ARGUMENT,
        "invalid semantic use keeps INVALID_ARGUMENT");
    assert(capture.calls == 1, "invalid semantic use emits exactly one diagnostic");
    assert(capture.operation_matches, "operation is cmd_texture_barrier");
    assert(capture.has_fault && capture.public_fault == with_callback,
        "diagnostic carries the unchanged semantic-use fault");
    assert(capture.public_contract, "invalid semantic use is a public-contract diagnostic");
    assert(capture.resource_kind == gpu::DebugResourceKind.TEXTURE,
        "resource kind is texture");
    assert(capture.has_public_identity, "live texture identity is present");
    assert(capture.index == texture.index, "texture index is preserved");
    assert(capture.generation == texture.generation,
        "texture generation is preserved");
    assert(capture.debug_name_matches, "texture debug name is preserved");
    assert(capture.rejected_field_matches, "after-use field is identified");
    assert(capture.invariant_matches, "semantic-use invariant is identified");

    gpu::internal::vk::VkDeviceState* state =
        (gpu::internal::vk::VkDeviceState*)gpu::internal::device_backend_state(device);
    state.debug_callback = null;
    fault without_callback = @catch(gpu::cmd_texture_barrier(
        &commands,
        &barrier,
    ));
    assert(without_callback == with_callback,
        "callback presence does not alter the semantic-use fault");
    assert(capture.calls == 1, "null callback emits no additional diagnostic");
}

fn void texture_barrier_state_diagnostics_name_nested_side_fields() @test {
    TextureDiagnosticCapture capture = {
        .expected_operation      = "cmd_texture_barrier",
        .expected_debug_name     = "texture_state_diagnostic",
        .expected_rejected_field = "TextureBarrier.before.access",
        .expected_invariant      = "sampled textures require read-only access",
    };
    gpu::Device device = make_texture_diagnostic_device(&capture);
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .access     = { .graphics },
        .width      = 8,
        .height     = 8,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled },
        .debug_name = "texture_state_diagnostic",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::TextureBarrier barrier = {
        .texture = texture,
        .before = {
            .layout = gpu::TextureLayout.SAMPLED,
            .stages = { .fragment_shader },
        },
        .after  = gpu::sampled_at({ .fragment_shader }),
    };
    assert(@catch(gpu::cmd_texture_barrier(&commands, &barrier))
            == gpu::INVALID_ARGUMENT,
        "invalid source access faults INVALID_ARGUMENT");
    assert(capture.calls == 1 && capture.operation_matches
        && capture.rejected_field_matches && capture.invariant_matches,
        "source-state diagnostic names TextureBarrier.before.access exactly");

    capture = {
        .expected_operation      = "cmd_texture_barrier",
        .expected_debug_name     = "texture_state_diagnostic",
        .expected_rejected_field = "TextureBarrier.after.stages",
        .expected_invariant      = "stages must contain known non-host pipeline stages and all must be exclusive",
    };
    barrier.before = { .layout = gpu::TextureLayout.UNDEFINED };
    barrier.after  = {
        .layout = gpu::TextureLayout.SAMPLED,
        .stages = { .host },
        .access = { .read },
    };
    assert(@catch(gpu::cmd_texture_barrier(&commands, &barrier))
            == gpu::INVALID_ARGUMENT,
        "invalid destination stages fault INVALID_ARGUMENT");
    assert(capture.calls == 1 && capture.operation_matches
        && capture.rejected_field_matches && capture.invariant_matches,
        "destination-state diagnostic names TextureBarrier.after.stages exactly");
}

fn void placed_texture_requirements_validate_and_release_storage() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc texture_desc = {
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
