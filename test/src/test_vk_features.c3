module gpu_test;

import gpu;
import gpu::internal::vk @public;
import vk;

// Pure-CPU: feeds the feature gate synthetic structs, no ICD required.

struct TextureStateMapping {
    gpu::TextureState       state;
    vk::PipelineStageFlags2 stages;
    vk::AccessFlags2        access;
    vk::ImageLayout         layout;
}

fn vk::PhysicalDeviceFeatures base_with_required() {
    vk::PhysicalDeviceFeatures base;
    base.shader_int64 = vk::TRUE;
    base.multi_draw_indirect = vk::TRUE;
    base.independent_blend = vk::TRUE;
    base.depth_bias_clamp = vk::TRUE;
    return base;
}

fn vk::PhysicalDeviceVulkan11Features v11_with_required() {
    return vk::physical_device_vulkan11features()
        .set_shader_draw_parameters(vk::TRUE);
}

fn vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT
dynamic_properties_with_required() {
    return vk::physical_device_extended_dynamic_state3properties_ext()
        .set_dynamic_primitive_topology_unrestricted(vk::TRUE);
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
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        dynamic_properties_with_required();

    assert(!@catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_properties: &dynamic_properties,
    )), "all required features present must pass the gate");
}

fn bool? dynamic_state3_extension_absent(
    vk::PhysicalDevice physical_device,
    ZString extension,
) {
    return extension.str_view() != "VK_EXT_extended_dynamic_state3";
}

fn void required_dynamic_state3_extension_absence_is_mocked() @test {
    vk::PhysicalDevice physical_device = (vk::PhysicalDevice)(uptr)0xA0;
    assert(@catch(gpu::internal::vk::check_required_dynamic_state3_extension_with_query(
        physical_device,
        &dynamic_state3_extension_absent,
    ))
        == gpu::UNSUPPORTED_FEATURE,
        "mocked EDS3 absence deterministically rejects adapter admission");
}

fn void missing_feature_rejected() @test {
    // All-zero structs: no required feature set.
    vk::PhysicalDeviceFeatures base;
    vk::PhysicalDeviceVulkan11Features v11 = vk::physical_device_vulkan11features();
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features();
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features();
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        vk::physical_device_extended_dynamic_state3properties_ext();

    fault gate_fault = @catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_properties: &dynamic_properties,
    ));
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
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        dynamic_properties_with_required();

    fault gate_fault = @catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_properties: &dynamic_properties,
    ));
    assert(gate_fault == gpu::UNSUPPORTED_FEATURE, "a single missing feature still raises UNSUPPORTED_FEATURE");
}

fn void missing_shader_int64_rejected() @test {
    // Everything except shaderInt64.
    vk::PhysicalDeviceFeatures base;
    base.multi_draw_indirect = vk::TRUE;
    base.independent_blend = vk::TRUE;
    base.depth_bias_clamp = vk::TRUE;
    vk::PhysicalDeviceVulkan11Features v11 = v11_with_required();
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features()
        .set_synchronization2(vk::TRUE)
        .set_dynamic_rendering(vk::TRUE);
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        dynamic_properties_with_required();

    fault gate_fault = @catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_properties: &dynamic_properties,
    ));
    assert(gate_fault == gpu::UNSUPPORTED_FEATURE, "missing shaderInt64 raises UNSUPPORTED_FEATURE");
}

fn void missing_indirect_features_rejected() @test {
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features()
        .set_synchronization2(vk::TRUE)
        .set_dynamic_rendering(vk::TRUE);
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        dynamic_properties_with_required();

    // multiDrawIndirect unset.
    vk::PhysicalDeviceFeatures base_no_mdi;
    base_no_mdi.shader_int64 = vk::TRUE;
    base_no_mdi.independent_blend = vk::TRUE;
    base_no_mdi.depth_bias_clamp = vk::TRUE;
    vk::PhysicalDeviceVulkan11Features v11 = v11_with_required();
    fault mdi_fault = @catch(gpu::internal::vk::check_required_features(
        base:               &base_no_mdi,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_properties: &dynamic_properties,
    ));
    assert(mdi_fault == gpu::UNSUPPORTED_FEATURE, "missing multiDrawIndirect raises UNSUPPORTED_FEATURE");

    // shaderDrawParameters unset.
    vk::PhysicalDeviceFeatures base = base_with_required();
    vk::PhysicalDeviceVulkan11Features v11_empty = vk::physical_device_vulkan11features();
    fault sdp_fault = @catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11_empty,
        v12:                &v12,
        v13:                &v13,
        dynamic_properties: &dynamic_properties,
    ));
    assert(sdp_fault == gpu::UNSUPPORTED_FEATURE, "missing shaderDrawParameters raises UNSUPPORTED_FEATURE");
}

fn void raster_feature_requirements_rejected_individually() @test {
    vk::PhysicalDeviceFeatures base = base_with_required();
    vk::PhysicalDeviceVulkan11Features v11 = v11_with_required();
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features()
        .set_synchronization2(vk::TRUE)
        .set_dynamic_rendering(vk::TRUE);
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        dynamic_properties_with_required();

    base.independent_blend = vk::FALSE;
    assert(@catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_properties: &dynamic_properties,
    )) == gpu::UNSUPPORTED_FEATURE, "missing independentBlend rejects the candidate");

    base = base_with_required();
    base.depth_bias_clamp = vk::FALSE;
    assert(@catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_properties: &dynamic_properties,
    )) == gpu::UNSUPPORTED_FEATURE, "missing depthBiasClamp rejects the candidate");

    base = base_with_required();
    dynamic_properties.dynamic_primitive_topology_unrestricted = vk::FALSE;
    assert(@catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_properties: &dynamic_properties,
    )) == gpu::UNSUPPORTED_FEATURE,
        "restricted dynamic topology rejects the candidate before scoring");
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
    assert(gpu::internal::vk::heap_shared_features_supported(&base, &full), "full bits pass the shared gate");
    assert(gpu::internal::vk::heap_indexing_supported(&base, &full), "full bits pass the indexing gate");

    vk::PhysicalDeviceVulkan12Features no_partial = v12_with_heap_bits()
        .set_descriptor_binding_partially_bound(vk::FALSE);
    assert(gpu::internal::vk::heap_shared_features_supported(&base, &no_partial), "shared gate ignores set-path bits");
    assert(!gpu::internal::vk::heap_indexing_supported(&base, &no_partial), "missing partially-bound fails the indexing gate");

    vk::PhysicalDeviceFeatures no_storage_access;
    assert(!gpu::internal::vk::heap_shared_features_supported(&no_storage_access, &full), "missing format-less storage access fails the shared gate");

    vk::PhysicalDeviceVulkan12Features empty = vk::physical_device_vulkan12features();
    assert(!gpu::internal::vk::heap_shared_features_supported(&base, &empty), "empty 1.2 features fail the shared gate");
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

    gpu::internal::vk::GeneratedWorkSupport supported =
        gpu::internal::vk::resolve_generated_work_support(
            extension_supported:    true,
            maintenance5_supported: true,
            features:               &features,
            properties:             &properties,
        );
    assert(supported.enabled);
    assert(supported.max_count == 4096);

    gpu::internal::vk::GeneratedWorkSupport missing_extension =
        gpu::internal::vk::resolve_generated_work_support(
            extension_supported:    false,
            maintenance5_supported: true,
            features:               &features,
            properties:             &properties,
        );
    assert(!missing_extension.enabled && missing_extension.max_count == 0);

    gpu::internal::vk::GeneratedWorkSupport missing_maintenance5 =
        gpu::internal::vk::resolve_generated_work_support(
            extension_supported:    true,
            maintenance5_supported: false,
            features:               &features,
            properties:             &properties,
        );
    assert(!missing_maintenance5.enabled && missing_maintenance5.max_count == 0);

    features.device_generated_commands = vk::FALSE;
    gpu::internal::vk::GeneratedWorkSupport missing_feature =
        gpu::internal::vk::resolve_generated_work_support(
            extension_supported:    true,
            maintenance5_supported: true,
            features:               &features,
            properties:             &properties,
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
    assert(!gpu::internal::vk::resolve_generated_work_support(
        extension_supported:    true,
        maintenance5_supported: true,
        features:               &features,
        properties:             &properties,
    ).enabled);

    properties = generated_work_properties();
    properties.max_indirect_commands_token_count = 1;
    assert(!gpu::internal::vk::resolve_generated_work_support(
        extension_supported:    true,
        maintenance5_supported: true,
        features:               &features,
        properties:             &properties,
    ).enabled);

    properties = generated_work_properties();
    properties.max_indirect_commands_token_offset = 15;
    assert(!gpu::internal::vk::resolve_generated_work_support(
        extension_supported:    true,
        maintenance5_supported: true,
        features:               &features,
        properties:             &properties,
    ).enabled);

    properties = generated_work_properties();
    properties.max_indirect_commands_indirect_stride = 39;
    assert(!gpu::internal::vk::resolve_generated_work_support(
        extension_supported:    true,
        maintenance5_supported: true,
        features:               &features,
        properties:             &properties,
    ).enabled);

    properties = generated_work_properties();
    properties.max_indirect_sequence_count = 0;
    gpu::internal::vk::GeneratedWorkSupport zero_count =
        gpu::internal::vk::resolve_generated_work_support(
            extension_supported:    true,
            maintenance5_supported: true,
            features:               &features,
            properties:             &properties,
        );
    assert(!zero_count.enabled && zero_count.max_count == 0);
}

fn void heap_implementation_resolution_is_private() @test {
    gpu::internal::vk::VkHeapImplementation both =
        gpu::internal::vk::resolve_heap_implementation(true, true)!!;
    assert(
        both == gpu::internal::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        "indexing is preferred when both private implementations are usable"
    );

    gpu::internal::vk::VkHeapImplementation buffer_only =
        gpu::internal::vk::resolve_heap_implementation(false, true)!!;
    assert(
        buffer_only == gpu::internal::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        "descriptor buffers provide the private fallback"
    );

    fault unavailable = @catch(gpu::internal::vk::resolve_heap_implementation(false, false));
    assert(
        unavailable == gpu::UNSUPPORTED_FEATURE,
        "strict heaps require at least one exact implementation"
    );
}

fn void line_polygon_mode_feature_support() @test {
    vk::PhysicalDeviceFeatures unsupported;
    assert(!gpu::internal::vk::line_polygon_mode_supported(&unsupported), "unset fillModeNonSolid reports no line polygon support");

    vk::PhysicalDeviceFeatures supported;
    supported.fill_mode_non_solid = vk::TRUE;
    assert(gpu::internal::vk::line_polygon_mode_supported(&supported), "fillModeNonSolid reports line polygon support");
}

fn void polygon_mode_optional_gate() @test {
    assert(!@catch(gpu::internal::vk::check_polygon_mode_support(gpu::PolygonMode.FILL, false)),
        "FILL remains available without optional support");
    assert(!@catch(gpu::internal::vk::check_polygon_mode_support(gpu::PolygonMode.LINE, true)),
        "LINE is available when optional support is enabled");
    assert(@catch(gpu::internal::vk::check_polygon_mode_support(gpu::PolygonMode.LINE, false)) == gpu::UNSUPPORTED_FEATURE,
        "LINE without optional support faults UNSUPPORTED_FEATURE");
}

fn void compositional_texture_states_map_exactly() @test {
    ulong shader_storage_access =
        (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_STORAGE_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    ulong color_access =
        (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    ulong depth_stages =
        (ulong)vk::PipelineStageFlagBits2.STAGE_2_EARLY_FRAGMENT_TESTS_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    ulong depth_access =
        (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    TextureStateMapping[8] cases = {
        {
            { .layout = gpu::TextureLayout.UNDEFINED },
            vk::PipelineStageFlagBits2.STAGE_2_NONE,
            vk::AccessFlagBits2.ACCESS_2_NONE,
            vk::ImageLayout.UNDEFINED,
        },
        {
            {
                .layout = gpu::TextureLayout.TRANSFER_SOURCE,
                .stages = { .transfer },
                .access = { .read },
            },
            vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT,
            vk::AccessFlagBits2.ACCESS_2_TRANSFER_READ_BIT,
            vk::ImageLayout.TRANSFER_SRC_OPTIMAL,
        },
        {
            {
                .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
                .stages = { .transfer },
                .access = { .write },
            },
            vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT,
            vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT,
            vk::ImageLayout.TRANSFER_DST_OPTIMAL,
        },
        {
            gpu::sampled_at({ .compute }),
            vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
            vk::AccessFlagBits2.ACCESS_2_SHADER_SAMPLED_READ_BIT,
            vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
        },
        {
            gpu::sampled_at({ .fragment_shader }),
            vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT,
            vk::AccessFlagBits2.ACCESS_2_SHADER_SAMPLED_READ_BIT,
            vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
        },
        {
            gpu::storage_at(
                { .compute },
                { .read, .write },
            ),
            vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
            (vk::AccessFlags2)shader_storage_access,
            vk::ImageLayout.GENERAL,
        },
        {
            {
                .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
                .stages = { .color_output },
                .access = { .read, .write },
            },
            vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            (vk::AccessFlags2)color_access,
            vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
        },
        {
            {
                .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
                .stages = { .depth_output },
                .access = { .read, .write },
            },
            (vk::PipelineStageFlags2)depth_stages,
            (vk::AccessFlags2)depth_access,
            vk::ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
    };

    foreach (&expected : cases) {
        gpu::internal::vk::TextureStateScope actual = gpu::internal::vk::texture_state_to_vk(
            state:         expected.state,
            format:        gpu::Format.RGBA8_UNORM,
            side:          gpu::internal::vk::TextureStateSide.BEFORE,
            paired_stages: {},
        );
        assert(actual.stages == expected.stages, "texture state stage scope maps exactly");
        assert(actual.access == expected.access, "texture state access scope maps exactly");
        assert(actual.layout == expected.layout, "texture state layout maps exactly");
    }

    gpu::internal::vk::TextureStateScope ordinary_after =
        gpu::internal::vk::texture_state_to_vk(
            state:         gpu::storage_at({ .compute }, { .read, .write }),
            format:        gpu::Format.RGBA8_UNORM,
            side:          gpu::internal::vk::TextureStateSide.AFTER,
            paired_stages: { .color_output },
        );
    assert(ordinary_after.stages
            == vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
        "ordinary destination stages ignore paired presentation context");
    assert(ordinary_after.access == (vk::AccessFlags2)shader_storage_access,
        "ordinary destination access remains unchanged");
    assert(ordinary_after.layout == vk::ImageLayout.GENERAL,
        "ordinary destination layout remains unchanged");

    gpu::internal::vk::TextureStateScope present_before =
        gpu::internal::vk::texture_state_to_vk(
            state:         { .layout = gpu::TextureLayout.PRESENT },
            format:        gpu::Format.RGBA8_UNORM,
            side:          gpu::internal::vk::TextureStateSide.BEFORE,
            paired_stages: { .transfer },
        );
    assert(present_before.stages
            == vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT,
        "present source anchors to the paired first-consumer stage");
    assert(present_before.access == vk::AccessFlagBits2.ACCESS_2_NONE,
        "present source has no native access");
    assert(present_before.layout == vk::ImageLayout.PRESENT_SRC_KHR,
        "present source retains the presentation layout");

    gpu::internal::vk::TextureStateScope present_after =
        gpu::internal::vk::texture_state_to_vk(
            state:         { .layout = gpu::TextureLayout.PRESENT },
            format:        gpu::Format.RGBA8_UNORM,
            side:          gpu::internal::vk::TextureStateSide.AFTER,
            paired_stages: { .color_output },
        );
    assert(present_after.stages == vk::PipelineStageFlagBits2.STAGE_2_NONE,
        "present destination has no native stage");
    assert(present_after.access == vk::AccessFlagBits2.ACCESS_2_NONE,
        "present destination has no native access");
    assert(present_after.layout == vk::ImageLayout.PRESENT_SRC_KHR,
        "present destination retains the presentation layout");
}

fn void sampled_depth_state_chooses_depth_read_only_layout() @test {
    gpu::internal::vk::TextureStateScope sampled = gpu::internal::vk::texture_state_to_vk(
        state:         gpu::sampled_at({ .fragment_shader }),
        format:        gpu::Format.D32_FLOAT,
        side:          gpu::internal::vk::TextureStateSide.AFTER,
        paired_stages: {},
    );
    assert(sampled.layout == vk::ImageLayout.DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        "sampled depth textures use the read-only layout");
}

fn void sampled_descriptor_layout_matches_semantic_layout() @test {
    gpu::internal::vk::TextureSlot color = {
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled },
    };
    assert(gpu::internal::vk::texture_descriptor_layout(
            &color,
            gpu::internal::vk::TextureDescriptorKind.SAMPLED,
        ) == vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
        "sampled color descriptors use the shader read-only layout");

    gpu::internal::vk::TextureSlot depth = {
        .format = gpu::Format.D32_FLOAT,
        .usage  = { .sampled },
    };
    assert(gpu::internal::vk::texture_descriptor_layout(
            &depth,
            gpu::internal::vk::TextureDescriptorKind.SAMPLED,
        ) == vk::ImageLayout.DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        "sampled depth descriptors use the depth-stencil read-only layout");

    gpu::internal::vk::TextureSlot combined = {
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled, .storage },
    };
    assert(gpu::internal::vk::texture_descriptor_layout(
            &combined,
            gpu::internal::vk::TextureDescriptorKind.SAMPLED,
        ) == vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
        "sampled bindings retain read-only layout on storage-capable textures");
    assert(gpu::internal::vk::texture_descriptor_layout(
            &combined,
            gpu::internal::vk::TextureDescriptorKind.STORAGE,
        ) == vk::ImageLayout.GENERAL,
        "storage bindings use the general layout");
}

fn void combined_texture_batch_keeps_binding_layouts_distinct() @test {
    gpu::internal::vk::DescriptorHeapState heap;
    gpu::internal::vk::TextureSlot texture = {
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled, .storage },
    };
    vk::DescriptorImageInfo[2] image_infos;
    vk::WriteDescriptorSet[2] writes;
    uint write_count = gpu::internal::vk::accumulate_texture_descriptor_write(
        heap:         &heap,
        slot:         3,
        texture_slot: &texture,
        view:         {},
        image_infos:  image_infos[..],
        writes:       writes[..],
        write_count:  0,
    );
    assert(write_count == 2, "combined textures publish both descriptor bindings");
    assert(image_infos[0].image_layout == vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
        "batched sampled descriptors retain the read-only layout");
    assert(image_infos[1].image_layout == vk::ImageLayout.GENERAL,
        "batched storage descriptors use the general layout");
}

fn void semantic_texture_barrier_maps_partial_ranges_and_aspects() @test {
    gpu::internal::vk::TextureSlot color_slot = {
        .format       = gpu::Format.RGBA8_UNORM,
        .mip_levels   = 6,
        .array_layers = 8,
    };
    gpu::TextureBarrier color_barrier = {
        .view = {
            .base_mip    = 2,
            .mip_count   = 0,
            .base_layer  = 3,
            .layer_count = 2,
        },
        .before = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
        .after = gpu::sampled_at({ .fragment_shader }),
    };
    vk::ImageSubresourceRange color = gpu::internal::vk::resolve_texture_barrier_range(
        color_barrier.view,
        &color_slot,
    )!!;
    assert(color.aspect_mask == vk::ImageAspectFlagBits.COLOR_BIT);
    assert(color.base_mip_level == 2);
    assert(color.level_count == 4);
    assert(color.base_array_layer == 3);
    assert(color.layer_count == 2);

    gpu::internal::vk::TextureSlot depth_slot = {
        .format       = gpu::Format.D32_FLOAT,
        .mip_levels   = 1,
        .array_layers = 1,
    };
    gpu::TextureBarrier depth_barrier = {
        .before = {
            .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
            .stages = { .depth_output },
            .access = { .read, .write },
        },
        .after = gpu::sampled_at({ .compute }),
    };
    vk::ImageSubresourceRange depth = gpu::internal::vk::resolve_texture_barrier_range(
        depth_barrier.view,
        &depth_slot,
    )!!;
    ulong expected_aspects = (ulong)vk::ImageAspectFlagBits.DEPTH_BIT;
    assert((ulong)depth.aspect_mask == expected_aspects);
}

fn void texture_barrier_range_rejects_invalid_ranges() @test {
    gpu::internal::vk::TextureSlot slot = {
        .format       = gpu::Format.RGBA8_UNORM,
        .mip_levels   = 4,
        .array_layers = 2,
    };
    gpu::TextureBarrier barrier = {
        .before = { .layout = gpu::TextureLayout.UNDEFINED },
        .after = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    };

    barrier.view.base_mip = 4;
    assert(@catch(gpu::internal::vk::resolve_texture_barrier_range(barrier.view, &slot))
            == gpu::INVALID_ARGUMENT,
        "a mip base at the texture limit is invalid");

    barrier.view = { .base_layer = 1, .layer_count = 2 };
    assert(@catch(gpu::internal::vk::resolve_texture_barrier_range(barrier.view, &slot))
            == gpu::INVALID_ARGUMENT,
        "a layer range extending past the texture is invalid");

}

fn void semantic_barrier_stage_masks_map_exactly() @test {
    gpu::StageMask stages = {
        .compute = true,
        .color_output = true,
    };
    ulong expected = (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    assert((ulong)gpu::internal::vk::barrier_before_stages_to_vk(stages) == expected,
        "combined semantic producer stages map exactly");

    gpu::HazardFlags hazards = {
        .draw_arguments = true,
        .depth_stencil = true,
    };
    expected |= (ulong)vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_EARLY_FRAGMENT_TESTS_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    assert((ulong)gpu::internal::vk::barrier_after_stages_to_vk(stages, hazards) == expected,
        "special consumer hazards extend the destination stage scope");

    ulong generated_expected = expected
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMMAND_PREPROCESS_BIT_NV;
    assert((ulong)gpu::internal::vk::barrier_after_stages_to_vk(stages, hazards, true)
            == generated_expected,
        "generated work adds the command-preprocess destination stage");

    gpu::StageMask all = { .all = true };
    assert(gpu::internal::vk::barrier_before_stages_to_vk(all)
            == vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        "all maps to the broad execution scope");
    gpu::StageMask present = { .present = true };
    assert(gpu::internal::vk::barrier_after_stages_to_vk(present, {})
            == vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        "presentation maps to the broad execution scope");
}

fn void submit_wait_stage_masks_validate_and_map_exactly() @test {
    vk::QueueFlags transfer_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.TRANSFER_BIT;
    vk::QueueFlags compute_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.COMPUTE_BIT;
    vk::QueueFlags graphics_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.GRAPHICS_BIT;
    gpu::StageMask combined = {
        .fragment_shader = true,
        .color_output    = true,
    };
    ulong expected =
        (ulong)vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    assert((ulong)gpu::internal::vk::submit_wait_stages_to_vk(
            combined,
            graphics_only,
        )!! == expected,
        "submission waits preserve combined destination stage bits");
    assert(gpu::internal::vk::submit_wait_stages_to_vk(
            { .all },
            transfer_only,
        )!! == vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        "the explicit all stage remains available on transfer queues");
    assert(gpu::internal::vk::submit_wait_stages_to_vk(
            { .transfer },
            compute_only,
        )!! == vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT,
        "compute queues support transfer destinations");
    assert(gpu::internal::vk::submit_wait_stages_to_vk(
            { .transfer },
            graphics_only,
        )!! == vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT,
        "graphics queues support transfer destinations");
    assert(gpu::internal::vk::submit_wait_stages_to_vk(
            { .compute },
            compute_only,
        )!! == vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
        "compute submission waits map exactly");
    assert(gpu::internal::vk::submit_wait_stages_to_vk(
            { .vertex_shader },
            graphics_only,
        )!! == vk::PipelineStageFlagBits2.STAGE_2_VERTEX_SHADER_BIT,
        "vertex submission waits map exactly");
    assert(gpu::internal::vk::submit_wait_stages_to_vk(
            { .fragment_shader },
            graphics_only,
        )!! == vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT,
        "fragment submission waits map exactly");
    assert(gpu::internal::vk::submit_wait_stages_to_vk(
            { .color_output },
            graphics_only,
        )!! == vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        "color-output submission waits map exactly");
    ulong depth =
        (ulong)vk::PipelineStageFlagBits2.STAGE_2_EARLY_FRAGMENT_TESTS_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    assert((ulong)gpu::internal::vk::submit_wait_stages_to_vk(
            { .depth_output },
            graphics_only,
        )!! == depth,
        "depth-output submission waits map to both test stages");

    assert(@catch(gpu::internal::vk::submit_wait_stages_to_vk({}, graphics_only))
            == gpu::INVALID_ARGUMENT,
        "empty submission wait stages are invalid");
    assert(@catch(gpu::internal::vk::submit_wait_stages_to_vk(
            { .all = true, .compute = true },
            graphics_only,
        )) == gpu::INVALID_ARGUMENT,
        "all cannot be combined with a specific submission wait stage");
    assert(@catch(gpu::internal::vk::submit_wait_stages_to_vk(
            { .host },
            graphics_only,
        )) == gpu::INVALID_ARGUMENT,
        "host is not a submission wait destination stage");
    assert(@catch(gpu::internal::vk::submit_wait_stages_to_vk(
            { .present },
            graphics_only,
        )) == gpu::INVALID_ARGUMENT,
        "present is not a submission wait destination stage");
    assert(@catch(gpu::internal::vk::submit_wait_stages_to_vk(
            { .compute },
            transfer_only,
        )) == gpu::INVALID_ARGUMENT,
        "submission wait stages must be supported by the target queue");
    assert(@catch(gpu::internal::vk::submit_wait_stages_to_vk(
            { .compute },
            graphics_only,
        )) == gpu::INVALID_ARGUMENT,
        "graphics-only queues reject compute destinations");
    assert(@catch(gpu::internal::vk::submit_wait_stages_to_vk(
            { .vertex_shader },
            compute_only,
        )) == gpu::INVALID_ARGUMENT,
        "compute-only queues reject graphics destinations");
    gpu::StageMask unknown = (gpu::StageMask)(1u << 9);
    assert(@catch(gpu::internal::vk::submit_wait_stages_to_vk(
            unknown,
            graphics_only,
        )) == gpu::INVALID_ARGUMENT,
        "unknown submission wait stage bits are invalid");
}

fn void submit_wait_consumer_scopes_validate_and_map_exactly() @test {
    vk::QueueFlags transfer_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.TRANSFER_BIT;
    vk::QueueFlags compute_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.COMPUTE_BIT;
    vk::QueueFlags graphics_only =
        (vk::QueueFlags)(uint)vk::QueueFlagBits.GRAPHICS_BIT;
    gpu::CompletionConsumerFlags draw_arguments = { .draw_arguments };
    ulong indirect =
        (ulong)vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT;

    assert((ulong)gpu::internal::vk::submit_wait_scope_to_vk(
            stages:         {},
            consumers:      draw_arguments,
            flags:          graphics_only,
            generated_work_enabled: false,
        )!! == indirect,
        "graphics queues accept draw-argument-only completion waits");
    assert((ulong)gpu::internal::vk::submit_wait_scope_to_vk(
            stages:         {},
            consumers:      draw_arguments,
            flags:          compute_only,
            generated_work_enabled: false,
        )!! == indirect,
        "compute queues accept draw-argument-only completion waits");
    ulong generated = indirect
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMMAND_PREPROCESS_BIT_NV;
    assert((ulong)gpu::internal::vk::submit_wait_scope_to_vk(
            stages:         {},
            consumers:      draw_arguments,
            flags:          graphics_only,
            generated_work_enabled: true,
        )!! == generated,
        "generated work adds the command-preprocess completion-wait stage");
    ulong mixed = indirect
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_VERTEX_SHADER_BIT;
    assert((ulong)gpu::internal::vk::submit_wait_scope_to_vk(
            stages:         { .vertex_shader },
            consumers:      draw_arguments,
            flags:          graphics_only,
            generated_work_enabled: false,
        )!! == mixed,
        "ordinary and draw-argument consumers lower to their exact union");
    assert(@catch(gpu::internal::vk::submit_wait_scope_to_vk(
            stages:         {},
            consumers:      draw_arguments,
            flags:          transfer_only,
            generated_work_enabled: false,
        )) == gpu::INVALID_ARGUMENT,
        "transfer-only queues reject draw-argument consumers");
    assert(@catch(gpu::internal::vk::submit_wait_scope_to_vk(
            stages:         { .all },
            consumers:      draw_arguments,
            flags:          graphics_only,
            generated_work_enabled: false,
        )) == gpu::INVALID_ARGUMENT,
        "all cannot be combined with a special completion consumer");
    assert(@catch(gpu::internal::vk::submit_wait_scope_to_vk(
            stages:         {},
            consumers:      (gpu::CompletionConsumerFlags)(1u << 1),
            flags:          graphics_only,
            generated_work_enabled: false,
        )) == gpu::INVALID_ARGUMENT,
        "unknown completion consumer bits are invalid");
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
    assert((ulong)gpu::internal::vk::barrier_before_access_to_vk(stages) == before_expected,
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
    assert((ulong)gpu::internal::vk::barrier_after_access_to_vk(stages, hazards, true)
            == after_expected,
        "consumer stages and special hazards map to required access scopes");
    ulong indexing_expected = after_expected
        & ~(ulong)vk::AccessFlagBits2.ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT;
    assert((ulong)gpu::internal::vk::barrier_after_access_to_vk(stages, hazards, false)
            == indexing_expected,
        "descriptor-buffer access is omitted for the indexing implementation");

    ulong generated_expected = after_expected
        | (ulong)vk::AccessFlagBits2.ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV;
    assert((ulong)gpu::internal::vk::barrier_after_access_to_vk(stages, hazards, true, true)
            == generated_expected,
        "generated work adds command-preprocess reads without dropping indirect reads");

    gpu::StageMask present = { .present = true };
    assert(gpu::internal::vk::barrier_after_access_to_vk(present, {}, false)
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
    vk::MemoryBarrier2 native = gpu::internal::vk::barrier_to_vk(&barrier, true);

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

    vk::MemoryBarrier2 generated = gpu::internal::vk::barrier_to_vk(&barrier, true, true);
    assert((ulong)generated.dst_stage_mask
            == expected_dst_stages
                | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMMAND_PREPROCESS_BIT_NV,
        "generated work extends the wired destination stage scope");
    assert((ulong)generated.dst_access_mask
            == expected_dst_access
                | (ulong)vk::AccessFlagBits2.ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV,
        "generated work extends the wired destination access scope");
}

fn void semantic_barrier_masks_reject_empty_and_contradictory_values() @test {
    gpu::Barrier valid = {
        .before = { .compute = true },
        .after = { .fragment_shader = true },
        .hazards = { .descriptors = true },
    };
    assert(gpu::internal::vk::barrier_masks_valid(&valid),
        "nonempty semantic masks are valid");

    gpu::Barrier empty_before = valid;
    empty_before.before = {};
    assert(!gpu::internal::vk::barrier_masks_valid(&empty_before),
        "empty producer masks are invalid");

    gpu::Barrier empty_after = valid;
    empty_after.after = {};
    assert(!gpu::internal::vk::barrier_masks_valid(&empty_after),
        "empty consumer masks are invalid");

    gpu::Barrier all_and_specific = valid;
    all_and_specific.before.all = true;
    assert(!gpu::internal::vk::barrier_masks_valid(&all_and_specific),
        "all cannot be combined with a specific stage");

    gpu::Barrier present_and_specific = valid;
    present_and_specific.after.present = true;
    assert(!gpu::internal::vk::barrier_masks_valid(&present_and_specific),
        "present cannot be combined with a device stage");

    gpu::Barrier present_with_hazard = {
        .before = { .color_output = true },
        .after = { .present = true },
        .hazards = { .draw_arguments = true },
    };
    assert(!gpu::internal::vk::barrier_masks_valid(&present_with_hazard),
        "presentation cannot carry a special hazard");

    gpu::Barrier unknown_stage = valid;
    unknown_stage.before = (gpu::StageMask)((uint)valid.before | 0x200u);
    assert(!gpu::internal::vk::barrier_masks_valid(&unknown_stage),
        "mixed known and unknown stage-mask bits are invalid");

    gpu::Barrier unknown_hazard = valid;
    unknown_hazard.hazards = (gpu::HazardFlags)0x8u;
    assert(!gpu::internal::vk::barrier_masks_valid(&unknown_hazard),
        "unknown hazard bits are invalid");

    gpu::Barrier descriptor_without_shader = {
        .before = { .transfer = true },
        .after = { .transfer = true },
        .hazards = { .descriptors = true },
    };
    assert(!gpu::internal::vk::barrier_masks_valid(&descriptor_without_shader),
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
    assert(gpu::internal::vk::barrier_queue_flags_valid(&transfer, transfer_only),
        "transfer-only queues accept host and transfer scopes");

    gpu::Barrier compute = {
        .before = { .transfer = true },
        .after = { .compute = true },
    };
    assert(!gpu::internal::vk::barrier_queue_flags_valid(&compute, transfer_only),
        "transfer-only queues reject compute scopes");
    assert(gpu::internal::vk::barrier_queue_flags_valid(&compute, compute_only),
        "compute-capable families accept transfer and compute scopes");

    gpu::Barrier color = {
        .before = { .compute = true },
        .after = { .color_output = true },
    };
    assert(!gpu::internal::vk::barrier_queue_flags_valid(&color, compute_only),
        "compute-only queues reject graphics scopes");
    assert(gpu::internal::vk::barrier_queue_flags_valid(&color, graphics_compute),
        "graphics-compute families accept both scopes");

    gpu::Barrier draw_arguments = {
        .before = { .transfer = true },
        .after = { .transfer = true },
        .hazards = { .draw_arguments = true },
    };
    assert(!gpu::internal::vk::barrier_queue_flags_valid(
            &draw_arguments,
            transfer_only,
        ),
        "transfer-only queues reject indirect-command hazards");
    assert(gpu::internal::vk::barrier_queue_flags_valid(
            &draw_arguments,
            compute_only,
        ),
        "compute-capable families accept indirect-command hazards");

    gpu::Barrier depth_stencil = {
        .before = { .transfer = true },
        .after = { .transfer = true },
        .hazards = { .depth_stencil = true },
    };
    assert(!gpu::internal::vk::barrier_queue_flags_valid(
            &depth_stencil,
            compute_only,
        ),
        "compute-only queues reject depth/stencil hazards");
    assert(gpu::internal::vk::barrier_queue_flags_valid(
            &depth_stencil,
            graphics_only,
        ),
        "graphics-capable families accept depth/stencil hazards");
}

fn void index_type_validation_rejects_unknown_values() @test {
    assert(gpu::internal::vk::index_element_size(gpu::IndexType.U16)!! == 2);
    assert(gpu::internal::vk::index_element_size(gpu::IndexType.U32)!! == 4);
    gpu::IndexType unknown_type;
    *(int*)&unknown_type = 99;
    fault unknown = @catch(gpu::internal::vk::index_element_size(unknown_type));
    assert(unknown == gpu::INVALID_ARGUMENT, "unknown index format faults");
}

fn void strict_presentation_requires_completion_fences() @test {
    assert(gpu::internal::vk::presentation_maintenance_supported(
        extension_supported: true,
        feature_enabled:     vk::TRUE,
    ));
    assert(!gpu::internal::vk::presentation_maintenance_supported(
        extension_supported: false,
        feature_enabled:     vk::TRUE,
    ));
    assert(!gpu::internal::vk::presentation_maintenance_supported(
        extension_supported: true,
        feature_enabled:     vk::FALSE,
    ));
}
