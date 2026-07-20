module gpu_test;

import gpu;
import gpu::vk @public;
import vk;

// Pure-CPU: feeds the feature gate synthetic structs, no ICD required.

fn vk::PhysicalDeviceFeatures base_with_required() {
    vk::PhysicalDeviceFeatures base;
    base.shader_int64 = vk::TRUE;
    base.multi_draw_indirect = vk::TRUE;
    return base;
}

fn vk::PhysicalDeviceVulkan11Features v11_with_required() {
    return vk::physical_device_vulkan11features()
        .set_shader_draw_parameters(vk::TRUE);
}

fn void required_features_present_ok() @test {
    vk::PhysicalDeviceFeatures base = base_with_required();
    vk::PhysicalDeviceVulkan11Features v11 = v11_with_required();
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features()
        .set_synchronization2(vk::TRUE)
        .set_dynamic_rendering(vk::TRUE);

    assert(!@catch(gpu::vk::check_required_features(base: &base, v11: &v11, v12: &v12, v13: &v13)), "all required features present must pass the gate");
}

fn void missing_feature_rejected() @test {
    // All-zero structs: no required feature set.
    vk::PhysicalDeviceFeatures base;
    vk::PhysicalDeviceVulkan11Features v11 = vk::physical_device_vulkan11features();
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features();
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features();

    fault gate_fault = @catch(gpu::vk::check_required_features(base: &base, v11: &v11, v12: &v12, v13: &v13));
    assert(gate_fault == gpu::UNSUPPORTED_FEATURE, "a missing required feature raises UNSUPPORTED_FEATURE");
}

fn void one_missing_feature_rejected() @test {
    // Everything but dynamic_rendering.
    vk::PhysicalDeviceFeatures base = base_with_required();
    vk::PhysicalDeviceVulkan11Features v11 = v11_with_required();
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features()
        .set_synchronization2(vk::TRUE);

    fault gate_fault = @catch(gpu::vk::check_required_features(base: &base, v11: &v11, v12: &v12, v13: &v13));
    assert(gate_fault == gpu::UNSUPPORTED_FEATURE, "a single missing feature still raises UNSUPPORTED_FEATURE");
}

fn void missing_shader_int64_rejected() @test {
    // Everything except shaderInt64.
    vk::PhysicalDeviceFeatures base;
    base.multi_draw_indirect = vk::TRUE;
    vk::PhysicalDeviceVulkan11Features v11 = v11_with_required();
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features()
        .set_synchronization2(vk::TRUE)
        .set_dynamic_rendering(vk::TRUE);

    fault gate_fault = @catch(gpu::vk::check_required_features(base: &base, v11: &v11, v12: &v12, v13: &v13));
    assert(gate_fault == gpu::UNSUPPORTED_FEATURE, "missing shaderInt64 raises UNSUPPORTED_FEATURE");
}

fn void missing_indirect_features_rejected() @test {
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features()
        .set_synchronization2(vk::TRUE)
        .set_dynamic_rendering(vk::TRUE);

    // multiDrawIndirect unset.
    vk::PhysicalDeviceFeatures base_no_mdi;
    base_no_mdi.shader_int64 = vk::TRUE;
    vk::PhysicalDeviceVulkan11Features v11 = v11_with_required();
    fault mdi_fault = @catch(gpu::vk::check_required_features(base: &base_no_mdi, v11: &v11, v12: &v12, v13: &v13));
    assert(mdi_fault == gpu::UNSUPPORTED_FEATURE, "missing multiDrawIndirect raises UNSUPPORTED_FEATURE");

    // shaderDrawParameters unset.
    vk::PhysicalDeviceFeatures base = base_with_required();
    vk::PhysicalDeviceVulkan11Features v11_empty = vk::physical_device_vulkan11features();
    fault sdp_fault = @catch(gpu::vk::check_required_features(base: &base, v11: &v11_empty, v12: &v12, v13: &v13));
    assert(sdp_fault == gpu::UNSUPPORTED_FEATURE, "missing shaderDrawParameters raises UNSUPPORTED_FEATURE");
}

fn vk::PhysicalDeviceVulkan12Features v12_with_heap_bits() {
    return vk::physical_device_vulkan12features()
        .set_runtime_descriptor_array(vk::TRUE)
        .set_shader_sampled_image_array_non_uniform_indexing(vk::TRUE)
        .set_shader_storage_image_array_non_uniform_indexing(vk::TRUE)
        .set_descriptor_binding_partially_bound(vk::TRUE)
        .set_descriptor_binding_sampled_image_update_after_bind(vk::TRUE)
        .set_descriptor_binding_storage_image_update_after_bind(vk::TRUE)
        .set_descriptor_binding_update_unused_while_pending(vk::TRUE);
}

fn vk::PhysicalDeviceFeatures base_with_storage_access() {
    vk::PhysicalDeviceFeatures base;
    base.shader_storage_image_read_without_format  = vk::TRUE;
    base.shader_storage_image_write_without_format = vk::TRUE;
    return base;
}

fn void heap_indexing_gate() @test {
    vk::PhysicalDeviceFeatures base = base_with_storage_access();
    vk::PhysicalDeviceVulkan12Features full = v12_with_heap_bits();
    assert(gpu::vk::heap_shared_features_supported(&base, &full), "full bits pass the shared gate");
    assert(gpu::vk::heap_indexing_supported(&base, &full), "full bits pass the indexing gate");

    vk::PhysicalDeviceVulkan12Features no_partial = v12_with_heap_bits()
        .set_descriptor_binding_partially_bound(vk::FALSE);
    assert(gpu::vk::heap_shared_features_supported(&base, &no_partial), "shared gate ignores set-path bits");
    assert(!gpu::vk::heap_indexing_supported(&base, &no_partial), "missing partially-bound fails the indexing gate");

    vk::PhysicalDeviceFeatures no_storage_access;
    assert(!gpu::vk::heap_shared_features_supported(&no_storage_access, &full), "missing format-less storage access fails the shared gate");

    vk::PhysicalDeviceVulkan12Features empty = vk::physical_device_vulkan12features();
    assert(!gpu::vk::heap_shared_features_supported(&base, &empty), "empty 1.2 features fail the shared gate");
}

fn vk::PhysicalDeviceDeviceGeneratedCommandsPropertiesEXT
generated_work_properties() {
    vk::PhysicalDeviceDeviceGeneratedCommandsPropertiesEXT properties =
        vk::physical_device_device_generated_commands_properties_ext();
    properties.max_indirect_sequence_count = 4096;
    properties.max_indirect_commands_token_count = 2;
    properties.max_indirect_commands_token_offset = 16;
    properties.max_indirect_commands_indirect_stride = 40;
    properties.supported_indirect_commands_shader_stages =
        vk::ShaderStageFlagBits.VERTEX_BIT
        | vk::ShaderStageFlagBits.FRAGMENT_BIT
        | vk::ShaderStageFlagBits.COMPUTE_BIT;
    return properties;
}

fn void generated_work_support_requires_the_complete_semantic_profile() @test {
    vk::PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT features =
        vk::physical_device_device_generated_commands_features_ext()
            .set_device_generated_commands(vk::TRUE);
    vk::PhysicalDeviceDeviceGeneratedCommandsPropertiesEXT properties =
        generated_work_properties();

    gpu::vk::GeneratedWorkSupport supported =
        gpu::vk::resolve_generated_work_support(
            extension_supported: true,
            features:            &features,
            properties:          &properties,
        );
    assert(supported.enabled);
    assert(supported.max_count == 4096);

    gpu::vk::GeneratedWorkSupport missing_extension =
        gpu::vk::resolve_generated_work_support(
            extension_supported: false,
            features:            &features,
            properties:          &properties,
        );
    assert(!missing_extension.enabled && missing_extension.max_count == 0);

    features.device_generated_commands = vk::FALSE;
    gpu::vk::GeneratedWorkSupport missing_feature =
        gpu::vk::resolve_generated_work_support(
            extension_supported: true,
            features:            &features,
            properties:          &properties,
        );
    assert(!missing_feature.enabled && missing_feature.max_count == 0);
}

fn void generated_work_support_rejects_incomplete_layout_limits() @test {
    vk::PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT features =
        vk::physical_device_device_generated_commands_features_ext()
            .set_device_generated_commands(vk::TRUE);
    vk::PhysicalDeviceDeviceGeneratedCommandsPropertiesEXT properties =
        generated_work_properties();

    properties.supported_indirect_commands_shader_stages =
        vk::ShaderStageFlagBits.VERTEX_BIT
        | vk::ShaderStageFlagBits.FRAGMENT_BIT;
    assert(!gpu::vk::resolve_generated_work_support(
        extension_supported: true,
        features:            &features,
        properties:          &properties,
    ).enabled);

    properties = generated_work_properties();
    properties.max_indirect_commands_token_count = 1;
    assert(!gpu::vk::resolve_generated_work_support(
        extension_supported: true,
        features:            &features,
        properties:          &properties,
    ).enabled);

    properties = generated_work_properties();
    properties.max_indirect_commands_token_offset = 15;
    assert(!gpu::vk::resolve_generated_work_support(
        extension_supported: true,
        features:            &features,
        properties:          &properties,
    ).enabled);

    properties = generated_work_properties();
    properties.max_indirect_commands_indirect_stride = 39;
    assert(!gpu::vk::resolve_generated_work_support(
        extension_supported: true,
        features:            &features,
        properties:          &properties,
    ).enabled);

    properties = generated_work_properties();
    properties.max_indirect_sequence_count = 0;
    gpu::vk::GeneratedWorkSupport zero_count =
        gpu::vk::resolve_generated_work_support(
            extension_supported: true,
            features:            &features,
            properties:          &properties,
        );
    assert(!zero_count.enabled && zero_count.max_count == 0);
}

fn void heap_implementation_resolution_is_private() @test {
    gpu::vk::VkHeapImplementation both =
        gpu::vk::resolve_heap_implementation(true, true)!!;
    assert(
        both == gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        "indexing is preferred when both private implementations are usable"
    );

    gpu::vk::VkHeapImplementation buffer_only =
        gpu::vk::resolve_heap_implementation(false, true)!!;
    assert(
        buffer_only == gpu::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        "descriptor buffers provide the private fallback"
    );

    fault unavailable = @catch(gpu::vk::resolve_heap_implementation(false, false));
    assert(
        unavailable == gpu::UNSUPPORTED_FEATURE,
        "strict heaps require at least one exact implementation"
    );
}

fn void line_polygon_mode_feature_support() @test {
    vk::PhysicalDeviceFeatures unsupported;
    assert(!gpu::vk::line_polygon_mode_supported(&unsupported), "unset fillModeNonSolid reports no line polygon support");

    vk::PhysicalDeviceFeatures supported;
    supported.fill_mode_non_solid = vk::TRUE;
    assert(gpu::vk::line_polygon_mode_supported(&supported), "fillModeNonSolid reports line polygon support");
}

fn void polygon_mode_optional_gate() @test {
    assert(!@catch(gpu::vk::check_polygon_mode_support(gpu::PolygonMode.FILL, false)),
        "FILL remains available without optional support");
    assert(!@catch(gpu::vk::check_polygon_mode_support(gpu::PolygonMode.LINE, true)),
        "LINE is available when optional support is enabled");
    assert(@catch(gpu::vk::check_polygon_mode_support(gpu::PolygonMode.LINE, false)) == gpu::UNSUPPORTED_FEATURE,
        "LINE without optional support faults UNSUPPORTED_FEATURE");
}

fn void semantic_barrier_stage_masks_map_exactly() @test {
    gpu::StageMask stages = {
        .compute = true,
        .color_output = true,
    };
    ulong expected = (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    assert((ulong)gpu::vk::barrier_before_stages_to_vk(stages) == expected,
        "combined semantic producer stages map exactly");

    gpu::HazardFlags hazards = {
        .draw_arguments = true,
        .depth_stencil = true,
    };
    expected |= (ulong)vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_EARLY_FRAGMENT_TESTS_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    assert((ulong)gpu::vk::barrier_after_stages_to_vk(stages, hazards) == expected,
        "special consumer hazards extend the destination stage scope");

    gpu::StageMask all = { .all = true };
    assert(gpu::vk::barrier_before_stages_to_vk(all)
            == vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        "all maps to the broad execution scope");
    gpu::StageMask present = { .present = true };
    assert(gpu::vk::barrier_after_stages_to_vk(present, {})
            == vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        "presentation maps to the broad execution scope");
}

fn void semantic_barrier_access_masks_map_exactly() @test {
    gpu::StageMask stages = {
        .host = true,
        .transfer = true,
        .compute = true,
        .color_output = true,
    };
    ulong before_expected = (ulong)vk::AccessFlagBits2.ACCESS_2_HOST_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    assert((ulong)gpu::vk::barrier_before_access_to_vk(stages) == before_expected,
        "producer stages make their writes available");

    gpu::HazardFlags hazards = {
        .draw_arguments = true,
        .descriptors = true,
        .depth_stencil = true,
    };
    ulong after_expected = (ulong)vk::AccessFlagBits2.ACCESS_2_HOST_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_TRANSFER_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_INDIRECT_COMMAND_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    assert((ulong)gpu::vk::barrier_after_access_to_vk(stages, hazards, true)
            == after_expected,
        "consumer stages and special hazards map to required access scopes");
    ulong indexing_expected = after_expected
        & ~(ulong)vk::AccessFlagBits2.ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT;
    assert((ulong)gpu::vk::barrier_after_access_to_vk(stages, hazards, false)
            == indexing_expected,
        "descriptor-buffer access is omitted for the indexing implementation");

    gpu::StageMask present = { .present = true };
    assert(gpu::vk::barrier_after_access_to_vk(present, {}, false)
            == vk::AccessFlagBits2.ACCESS_2_MEMORY_READ_BIT,
        "presentation consumes memory");
}

fn void semantic_barrier_native_wiring_preserves_direction() @test {
    gpu::Barrier barrier = {
        .before = { .host = true },
        .after = { .compute = true },
        .hazards = {
            .draw_arguments = true,
            .descriptors = true,
        },
    };
    vk::MemoryBarrier2 native = gpu::vk::barrier_to_vk(&barrier, true);

    assert(native.src_stage_mask
            == vk::PipelineStageFlagBits2.STAGE_2_HOST_BIT,
        "the source stage comes from Barrier.before");
    assert(native.src_access_mask
            == vk::AccessFlagBits2.ACCESS_2_HOST_WRITE_BIT,
        "the source access comes from Barrier.before");
    ulong expected_dst_stages =
        (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT;
    assert((ulong)native.dst_stage_mask == expected_dst_stages,
        "the destination stages come from Barrier.after and hazards");
    ulong expected_dst_access =
        (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_INDIRECT_COMMAND_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT;
    assert((ulong)native.dst_access_mask == expected_dst_access,
        "the destination access comes from Barrier.after and hazards");
}

fn void semantic_barrier_masks_reject_empty_and_contradictory_values() @test {
    gpu::Barrier valid = {
        .before = { .compute = true },
        .after = { .fragment_shader = true },
        .hazards = { .descriptors = true },
    };
    assert(gpu::vk::barrier_masks_valid(&valid),
        "nonempty semantic masks are valid");

    gpu::Barrier empty_before = valid;
    empty_before.before = {};
    assert(!gpu::vk::barrier_masks_valid(&empty_before),
        "empty producer masks are invalid");

    gpu::Barrier empty_after = valid;
    empty_after.after = {};
    assert(!gpu::vk::barrier_masks_valid(&empty_after),
        "empty consumer masks are invalid");

    gpu::Barrier all_and_specific = valid;
    all_and_specific.before.all = true;
    assert(!gpu::vk::barrier_masks_valid(&all_and_specific),
        "all cannot be combined with a specific stage");

    gpu::Barrier present_and_specific = valid;
    present_and_specific.after.present = true;
    assert(!gpu::vk::barrier_masks_valid(&present_and_specific),
        "present cannot be combined with a device stage");

    gpu::Barrier present_with_hazard = {
        .before = { .color_output = true },
        .after = { .present = true },
        .hazards = { .draw_arguments = true },
    };
    assert(!gpu::vk::barrier_masks_valid(&present_with_hazard),
        "presentation cannot carry a special hazard");

    gpu::Barrier unknown_stage = valid;
    unknown_stage.before = (gpu::StageMask)((uint)valid.before | 0x200u);
    assert(!gpu::vk::barrier_masks_valid(&unknown_stage),
        "mixed known and unknown stage-mask bits are invalid");

    gpu::Barrier unknown_hazard = valid;
    unknown_hazard.hazards = (gpu::HazardFlags)0x8u;
    assert(!gpu::vk::barrier_masks_valid(&unknown_hazard),
        "unknown hazard bits are invalid");

    gpu::Barrier descriptor_without_shader = {
        .before = { .transfer = true },
        .after = { .transfer = true },
        .hazards = { .descriptors = true },
    };
    assert(!gpu::vk::barrier_masks_valid(&descriptor_without_shader),
        "descriptor hazards require a shader consumer stage");
}

fn void semantic_barrier_queue_flags_reject_unsupported_stages() @test {
    vk::QueueFlags transfer_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.TRANSFER_BIT;
    vk::QueueFlags compute_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.COMPUTE_BIT;
    vk::QueueFlags graphics_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.GRAPHICS_BIT;
    vk::QueueFlags graphics_compute = (vk::QueueFlags)(
        (uint)vk::QueueFlagBits.GRAPHICS_BIT
            | (uint)vk::QueueFlagBits.COMPUTE_BIT
    );

    gpu::Barrier transfer = {
        .before = { .host = true },
        .after = { .transfer = true },
    };
    assert(gpu::vk::barrier_queue_flags_valid(&transfer, transfer_only),
        "transfer-only queues accept host and transfer scopes");

    gpu::Barrier compute = {
        .before = { .transfer = true },
        .after = { .compute = true },
    };
    assert(!gpu::vk::barrier_queue_flags_valid(&compute, transfer_only),
        "transfer-only queues reject compute scopes");
    assert(gpu::vk::barrier_queue_flags_valid(&compute, compute_only),
        "compute-capable families accept transfer and compute scopes");

    gpu::Barrier color = {
        .before = { .compute = true },
        .after = { .color_output = true },
    };
    assert(!gpu::vk::barrier_queue_flags_valid(&color, compute_only),
        "compute-only queues reject graphics scopes");
    assert(gpu::vk::barrier_queue_flags_valid(&color, graphics_compute),
        "graphics-compute families accept both scopes");

    gpu::Barrier draw_arguments = {
        .before = { .transfer = true },
        .after = { .transfer = true },
        .hazards = { .draw_arguments = true },
    };
    assert(!gpu::vk::barrier_queue_flags_valid(
            &draw_arguments,
            transfer_only,
        ),
        "transfer-only queues reject indirect-command hazards");
    assert(gpu::vk::barrier_queue_flags_valid(
            &draw_arguments,
            compute_only,
        ),
        "compute-capable families accept indirect-command hazards");

    gpu::Barrier depth_stencil = {
        .before = { .transfer = true },
        .after = { .transfer = true },
        .hazards = { .depth_stencil = true },
    };
    assert(!gpu::vk::barrier_queue_flags_valid(
            &depth_stencil,
            compute_only,
        ),
        "compute-only queues reject depth/stencil hazards");
    assert(gpu::vk::barrier_queue_flags_valid(
            &depth_stencil,
            graphics_only,
        ),
        "graphics-capable families accept depth/stencil hazards");
}

fn void combined_texture_hazards_map_to_both_access_bits() @test {
    assert((ulong)gpu::vk::hazard_to_vk_access(gpu::Hazard.NONE) == 0,
        "NONE maps to no Vulkan access");

    ulong shader_expected = (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT;
    assert((ulong)gpu::vk::hazard_to_vk_access(gpu::Hazard.SHADER_READ_WRITE) == shader_expected,
        "shader read/write maps both access bits");

    ulong color_expected = (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    assert((ulong)gpu::vk::hazard_to_vk_access(gpu::Hazard.COLOR_READ_WRITE) == color_expected,
        "color read/write maps both access bits");

    ulong depth_expected = (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    assert((ulong)gpu::vk::hazard_to_vk_access(gpu::Hazard.DEPTH_READ_WRITE) == depth_expected,
        "depth read/write maps both access bits");
}

fn void presentation_scope_translations_are_exact() @test {
    assert(gpu::vk::stage_to_vk(gpu::Stage.NONE) == vk::PipelineStageFlagBits2.STAGE_2_NONE,
        "NONE maps to no Vulkan execution stage");
    assert(gpu::vk::hazard_to_vk_access(gpu::Hazard.NONE) == vk::AccessFlagBits2.ACCESS_2_NONE,
        "NONE maps to no Vulkan access");
    assert(gpu::vk::stage_to_vk(gpu::Stage.PRESENT) == vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        "legacy PRESENT keeps its broad execution stage");
    assert(gpu::vk::hazard_to_vk_access(gpu::Hazard.PRESENT_READ) == vk::AccessFlagBits2.ACCESS_2_MEMORY_READ_BIT,
        "legacy PRESENT_READ keeps its broad memory access");
}
