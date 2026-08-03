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

fn gpu::internal::vk::RayQueryCandidateSupport
ray_query_candidate_with_required() {
    return {
        .acceleration_structure_extension = true,
        .ray_query_extension = true,
        .deferred_host_operations_extension = true,
        .acceleration_structure = true,
        .ray_query = true,
        .descriptor_update_after_bind = true,
        .buffer_device_address = true,
        .runtime_descriptor_array = true,
        .descriptor_partially_bound = true,
        .descriptor_update_unused_while_pending = true,
        .spirv_1_4 = true,
        .required_commands = true,
        .triangle_vertex_format = true,
        .max_geometry_count = 1,
        .max_primitive_count = 1,
        .max_instance_count = 1,
        .max_descriptor_count = 1,
        .scratch_alignment = 256,
    };
}

fn void ray_query_candidate_gate_requires_the_complete_family() @test {
    gpu::internal::vk::RayQueryCandidateSupport support =
        ray_query_candidate_with_required();
    assert(gpu::internal::vk::ray_query_candidate_supported(&support));

    support.acceleration_structure_extension = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.ray_query_extension = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.deferred_host_operations_extension = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.acceleration_structure = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.ray_query = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.descriptor_update_after_bind = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.buffer_device_address = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.runtime_descriptor_array = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.descriptor_partially_bound = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.descriptor_update_unused_while_pending = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.spirv_1_4 = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.required_commands = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.triangle_vertex_format = false;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
}

fn void ray_query_candidate_gate_requires_actionable_limits() @test {
    gpu::internal::vk::RayQueryCandidateSupport support =
        ray_query_candidate_with_required();
    support.max_geometry_count = 0;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.max_primitive_count = 0;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.max_instance_count = 0;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.max_descriptor_count = 0;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
    support = ray_query_candidate_with_required();
    support.scratch_alignment = 0;
    assert(!gpu::internal::vk::ray_query_candidate_supported(&support));
}

fn void ray_query_caps_are_exact_saturating_and_opt_in_only() @test {
    gpu::internal::vk::RayQueryCandidateSupport support =
        ray_query_candidate_with_required();
    support.max_geometry_count = (ulong)uint::max + 17;
    support.max_primitive_count = 29;
    support.max_instance_count = (ulong)uint::max + 1;
    support.max_descriptor_count = 31;
    support.scratch_alignment = 512;

    gpu::RayQueryCaps disabled =
        gpu::internal::vk::normalize_ray_query_caps(false, 7, &support);
    assert(!disabled.enabled
            && disabled.acceleration_structure_heap_capacity == 0
            && disabled.max_geometry_count == 0
            && disabled.max_primitive_count == 0
            && disabled.max_instance_count == 0
            && disabled.scratch_alignment == 0,
        "the disabled path publishes a false zero capability record");

    gpu::RayQueryCaps enabled =
        gpu::internal::vk::normalize_ray_query_caps(true, 7, &support);
    assert(enabled.enabled
            && enabled.acceleration_structure_heap_capacity == 7
            && enabled.max_geometry_count == uint::max
            && enabled.max_primitive_count == 29
            && enabled.max_instance_count == uint::max
            && enabled.scratch_alignment == 512,
        "enabled public caps preserve exact limits and saturate wide values");

    assert(!gpu::internal::vk::normalize_ray_query_caps(
            true,
            support.max_descriptor_count + 1,
            &support,
        ).enabled,
        "the configured binding capacity must fit the descriptor path");
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

fn vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT
dynamic_color_features_with_required() {
    return vk::physical_device_extended_dynamic_state3features_ext()
        .set_extended_dynamic_state3color_blend_enable(vk::TRUE)
        .set_extended_dynamic_state3color_blend_equation(vk::TRUE)
        .set_extended_dynamic_state3color_write_mask(vk::TRUE);
}

fn void required_dynamic_color_features_require_the_complete_set() @test {
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT features =
        dynamic_color_features_with_required();
    assert(
        gpu::internal::vk::required_dynamic_color_features_supported(&features),
        "all three color features satisfy the device baseline"
    );

    features.extended_dynamic_state3color_blend_enable = vk::FALSE;
    assert(!gpu::internal::vk::required_dynamic_color_features_supported(&features));
    features = dynamic_color_features_with_required();
    features.extended_dynamic_state3color_blend_equation = vk::FALSE;
    assert(!gpu::internal::vk::required_dynamic_color_features_supported(&features));
    features = dynamic_color_features_with_required();
    features.extended_dynamic_state3color_write_mask = vk::FALSE;
    assert(!gpu::internal::vk::required_dynamic_color_features_supported(&features));
    assert(!gpu::internal::vk::required_dynamic_color_features_supported(null));
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
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_color =
        dynamic_color_features_with_required();
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        dynamic_properties_with_required();

    assert(!@catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_color:      &dynamic_color,
        dynamic_properties: &dynamic_properties,
    )), "all required features present must pass the gate");
}

fn void required_features_reject_only_missing_dynamic_color() @test {
    vk::PhysicalDeviceFeatures base = base_with_required();
    vk::PhysicalDeviceVulkan11Features v11 = v11_with_required();
    vk::PhysicalDeviceVulkan12Features v12 =
        vk::physical_device_vulkan12features()
            .set_buffer_device_address(vk::TRUE)
            .set_timeline_semaphore(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features v13 =
        vk::physical_device_vulkan13features()
            .set_synchronization2(vk::TRUE)
            .set_dynamic_rendering(vk::TRUE);
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_color =
        vk::physical_device_extended_dynamic_state3features_ext();
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT
        dynamic_properties = dynamic_properties_with_required();

    assert(@catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_color:      &dynamic_color,
        dynamic_properties: &dynamic_properties,
    )) == gpu::UNSUPPORTED_FEATURE,
        "the required-feature gate rejects an otherwise complete device without dynamic color");
}

fn bool? dynamic_state3_extension_absent(
    vk::PhysicalDevice physical_device,
    ZString extension,
) {
    return extension.str_view() != "VK_EXT_extended_dynamic_state3";
}

fn bool? dynamic_state3_extension_query_host_oom(
    vk::PhysicalDevice physical_device,
    ZString extension,
) {
    return vk::ERROR_OUT_OF_HOST_MEMORY~;
}

fn bool? dynamic_state3_extension_query_device_oom(
    vk::PhysicalDevice physical_device,
    ZString extension,
) {
    return vk::ERROR_OUT_OF_DEVICE_MEMORY~;
}

fn bool? dynamic_state3_extension_query_layer_missing(
    vk::PhysicalDevice physical_device,
    ZString extension,
) {
    return vk::ERROR_LAYER_NOT_PRESENT~;
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

fn void required_dynamic_state3_extension_query_faults_are_mapped() @test {
    vk::PhysicalDevice physical_device = (vk::PhysicalDevice)(uptr)0xA0;
    assert(
        @catch(gpu::internal::vk::check_required_dynamic_state3_extension_with_query(
            physical_device,
            &dynamic_state3_extension_query_host_oom,
        )) == gpu::OUT_OF_HOST_MEMORY,
        "extension-query host OOM maps into the public fault taxonomy"
    );
    assert(
        @catch(gpu::internal::vk::check_required_dynamic_state3_extension_with_query(
            physical_device,
            &dynamic_state3_extension_query_device_oom,
        )) == gpu::OUT_OF_DEVICE_MEMORY,
        "extension-query device OOM maps into the public fault taxonomy"
    );
    assert(
        @catch(gpu::internal::vk::check_required_dynamic_state3_extension_with_query(
            physical_device,
            &dynamic_state3_extension_query_layer_missing,
        )) == gpu::UNSUPPORTED_FEATURE,
        "a missing extension layer maps through the public unsupported-feature taxonomy"
    );
}

fn void missing_feature_rejected() @test {
    // All-zero structs: no required feature set.
    vk::PhysicalDeviceFeatures base;
    vk::PhysicalDeviceVulkan11Features v11 = vk::physical_device_vulkan11features();
    vk::PhysicalDeviceVulkan12Features v12 = vk::physical_device_vulkan12features();
    vk::PhysicalDeviceVulkan13Features v13 = vk::physical_device_vulkan13features();
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_color =
        vk::physical_device_extended_dynamic_state3features_ext();
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        vk::physical_device_extended_dynamic_state3properties_ext();

    fault gate_fault = @catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_color:      &dynamic_color,
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
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_color =
        dynamic_color_features_with_required();
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        dynamic_properties_with_required();

    fault gate_fault = @catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_color:      &dynamic_color,
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
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_color =
        dynamic_color_features_with_required();
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        dynamic_properties_with_required();

    fault gate_fault = @catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_color:      &dynamic_color,
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
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_color =
        dynamic_color_features_with_required();
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
        dynamic_color:      &dynamic_color,
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
        dynamic_color:      &dynamic_color,
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
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_color =
        dynamic_color_features_with_required();
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        dynamic_properties_with_required();

    base.independent_blend = vk::FALSE;
    assert(@catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_color:      &dynamic_color,
        dynamic_properties: &dynamic_properties,
    )) == gpu::UNSUPPORTED_FEATURE, "missing independentBlend rejects the candidate");

    base = base_with_required();
    base.depth_bias_clamp = vk::FALSE;
    assert(@catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_color:      &dynamic_color,
        dynamic_properties: &dynamic_properties,
    )) == gpu::UNSUPPORTED_FEATURE, "missing depthBiasClamp rejects the candidate");

    base = base_with_required();
    dynamic_properties.dynamic_primitive_topology_unrestricted = vk::FALSE;
    assert(@catch(gpu::internal::vk::check_required_features(
        base:               &base,
        v11:                &v11,
        v12:                &v12,
        v13:                &v13,
        dynamic_color:      &dynamic_color,
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
            gpu::internal::vk::TextureDescriptorKind.SAMPLED
        ) == vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
        "sampled color descriptors use the shader read-only layout");

    gpu::internal::vk::TextureSlot depth = {
        .format = gpu::Format.D32_FLOAT,
        .usage  = { .sampled },
    };
    assert(gpu::internal::vk::texture_descriptor_layout(
            &depth,
            gpu::internal::vk::TextureDescriptorKind.SAMPLED
        ) == vk::ImageLayout.DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        "sampled depth descriptors use the depth-stencil read-only layout");

    gpu::internal::vk::TextureSlot combined = {
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled, .storage },
    };
    assert(gpu::internal::vk::texture_descriptor_layout(
            &combined,
            gpu::internal::vk::TextureDescriptorKind.SAMPLED
        ) == vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
        "sampled bindings retain read-only layout on storage-capable textures");
    assert(gpu::internal::vk::texture_descriptor_layout(
            &combined,
            gpu::internal::vk::TextureDescriptorKind.STORAGE
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
        heap:              &heap,
        slot:              3,
        texture_slot:      &texture,
        view:              {},
        image_infos:       image_infos[..],
        writes:            writes[..],
        write_count:       0,
    );
    assert(write_count == 2, "combined textures publish both descriptor bindings");
    assert(image_infos[0].image_layout == vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
        "batched sampled descriptors retain the read-only layout");
    assert(image_infos[1].image_layout == vk::ImageLayout.GENERAL,
        "batched storage descriptors use the general layout");
}

fn void texture_descriptor_writes_route_by_dimension() @test {
    gpu::internal::vk::DescriptorHeapState heap;
    gpu::internal::vk::TextureSlot two_dimensional = {
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled, .storage },
    };
    gpu::internal::vk::TextureSlot three_dimensional = {
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled, .storage },
        .depth  = 4,
    };
    vk::DescriptorImageInfo[4] image_infos;
    vk::WriteDescriptorSet[4] writes;

    uint write_count = gpu::internal::vk::accumulate_texture_descriptor_write(
        heap:         &heap,
        slot:         3,
        texture_slot: &two_dimensional,
        view:         {},
        image_infos:  image_infos[..],
        writes:       writes[..],
        write_count:  0,
    );
    assert(write_count == 2,
        "one 2D view emits at most sampled and storage writes");
    assert(writes[0].dst_binding == gpu::internal::vk::HEAP_BINDING_SAMPLED_2D
            && writes[1].dst_binding
                == gpu::internal::vk::HEAP_BINDING_STORAGE_2D,
        "2D writes target the 2D bindings");
    assert(writes[0].dst_array_element == 3
            && writes[1].dst_array_element == 3,
        "2D sampled and storage writes share one logical slot");

    write_count = gpu::internal::vk::accumulate_texture_descriptor_write(
        heap:         &heap,
        slot:         7,
        texture_slot: &three_dimensional,
        view:         {},
        image_infos:  image_infos[..],
        writes:       writes[..],
        write_count:  write_count,
    );
    assert(write_count == 4,
        "a mixed-dimensional batch emits at most two writes per view");
    assert(writes[2].dst_binding == gpu::internal::vk::HEAP_BINDING_SAMPLED_3D
            && writes[3].dst_binding
                == gpu::internal::vk::HEAP_BINDING_STORAGE_3D,
        "3D writes target the 3D bindings");
    assert(writes[2].dst_array_element == 7
            && writes[3].dst_array_element == 7,
        "3D sampled and storage writes share one logical slot");
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
        .depth_output = true,
        .indirect = true,
        .acceleration_structure_build = true,
    };
    ulong expected = (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_EARLY_FRAGMENT_TESTS_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT
        | (ulong)vk::PipelineStageFlagBits2
            .STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    assert((ulong)gpu::internal::vk::barrier_before_stages_to_vk(stages) == expected,
        "combined semantic producer stages map exactly");

    ulong generated_expected = expected
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMMAND_PREPROCESS_BIT_NV;
    assert((ulong)gpu::internal::vk::barrier_before_stages_to_vk(stages, true)
            == generated_expected,
        "generated work adds command preprocessing to indirect producer stages");
    assert((ulong)gpu::internal::vk::barrier_after_stages_to_vk(stages) == expected,
        "destination stages use the same semantic stage vocabulary");

    assert((ulong)gpu::internal::vk::barrier_after_stages_to_vk(stages, true)
            == generated_expected,
        "generated work adds the command-preprocess destination stage");

    gpu::StageMask all = { .all = true };
    assert(gpu::internal::vk::barrier_before_stages_to_vk(all)
            == vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        "all maps to the broad execution scope");
    gpu::StageMask present = { .present = true };
    assert(gpu::internal::vk::barrier_after_stages_to_vk(present)
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
    assert(gpu::internal::vk::submit_wait_stages_to_vk(
            { .indirect },
            graphics_only,
        )!! == vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT,
        "graphics queues accept indirect completion waits");
    assert(gpu::internal::vk::submit_wait_stages_to_vk(
            { .indirect },
            compute_only,
        )!! == vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT,
        "compute queues accept indirect completion waits");
    assert(gpu::internal::vk::submit_wait_stages_to_vk(
            { .acceleration_structure_build },
            graphics_only,
        )!! == vk::PipelineStageFlagBits2
            .STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        "graphics queues accept acceleration-structure build waits");
    assert(gpu::internal::vk::submit_wait_stages_to_vk(
            { .acceleration_structure_build },
            compute_only,
        )!! == vk::PipelineStageFlagBits2
            .STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        "compute queues accept acceleration-structure build waits");
    ulong generated_indirect =
        (ulong)vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMMAND_PREPROCESS_BIT_NV;
    assert((ulong)gpu::internal::vk::submit_wait_stages_to_vk(
            { .indirect },
            graphics_only,
            true,
        )!! == generated_indirect,
        "generated work adds command preprocessing to indirect waits");
    ulong shader_indirect = (ulong)vk::PipelineStageFlagBits2.STAGE_2_VERTEX_SHADER_BIT
        | (ulong)vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT;
    assert((ulong)gpu::internal::vk::submit_wait_stages_to_vk(
            { .vertex_shader, .indirect },
            graphics_only,
        )!! == shader_indirect,
        "ordinary and indirect completion stages compose exactly");

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
    assert(@catch(gpu::internal::vk::submit_wait_stages_to_vk(
            { .indirect },
            transfer_only,
        )) == gpu::INVALID_ARGUMENT,
        "transfer-only queues reject indirect destinations");
    assert(@catch(gpu::internal::vk::submit_wait_stages_to_vk(
            { .acceleration_structure_build },
            transfer_only,
        )) == gpu::INVALID_ARGUMENT,
        "transfer-only queues reject acceleration-structure build waits");
    gpu::StageMask unknown = (gpu::StageMask)(1u << 11);
    assert(@catch(gpu::internal::vk::submit_wait_stages_to_vk(
            unknown,
            graphics_only,
        )) == gpu::INVALID_ARGUMENT,
        "unknown submission wait stage bits are invalid");
}

fn void semantic_barrier_access_masks_map_exactly() @test {
    gpu::StageMask stages = {
        .host = true,
        .transfer = true,
        .compute = true,
        .color_output = true,
        .depth_output = true,
        .indirect = true,
    };
    ulong before_expected = (ulong)vk::AccessFlagBits2.ACCESS_2_HOST_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    assert((ulong)gpu::internal::vk::barrier_before_access_to_vk(stages) == before_expected,
        "producer stages make their writes available");

    ulong after_expected = (ulong)vk::AccessFlagBits2.ACCESS_2_HOST_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_TRANSFER_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_INDIRECT_COMMAND_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    assert((ulong)gpu::internal::vk::barrier_after_access_to_vk(stages)
            == after_expected,
        "consumer stages map to required access scopes");

    ulong generated_expected = after_expected
        | (ulong)vk::AccessFlagBits2.ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV;
    assert((ulong)gpu::internal::vk::barrier_after_access_to_vk(stages, true)
            == generated_expected,
        "generated work adds command-preprocess reads without dropping indirect reads");

    gpu::StageMask present = { .present = true };
    assert(gpu::internal::vk::barrier_after_access_to_vk(present)
            == vk::AccessFlagBits2.ACCESS_2_MEMORY_READ_BIT,
        "presentation consumes memory");
}

fn void semantic_barrier_native_wiring_preserves_direction() @test {
    gpu::Barrier barrier = {
        .before = { .host = true },
        .after = { .compute = true, .indirect = true },
    };
    vk::MemoryBarrier2 native = gpu::internal::vk::barrier_to_vk(&barrier);

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
        "the destination stages come from Barrier.after");
    ulong expected_dst_access =
        (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    assert((ulong)native.dst_access_mask == expected_dst_access,
        "the destination access comes from Barrier.after");

    vk::MemoryBarrier2 generated = gpu::internal::vk::barrier_to_vk(&barrier, true);
    assert((ulong)generated.dst_stage_mask
            == expected_dst_stages
                | (ulong)vk::PipelineStageFlagBits2.STAGE_2_COMMAND_PREPROCESS_BIT_NV,
        "generated work extends the wired destination stage scope");
    assert((ulong)generated.dst_access_mask
            == expected_dst_access
                | (ulong)vk::AccessFlagBits2.ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV,
        "generated work extends the wired destination access scope");
}

fn void acceleration_structure_barrier_scopes_map_exactly() @test {
    ulong acceleration_structure_access =
        (ulong)vk::AccessFlagBits2
            .ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
        | (ulong)vk::AccessFlagBits2
            .ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    gpu::Barrier build_to_build = {
        .before = { .acceleration_structure_build },
        .after = { .acceleration_structure_build },
    };
    vk::MemoryBarrier2 native_build_to_build =
        gpu::internal::vk::barrier_to_vk(&build_to_build, false, true);
    assert(native_build_to_build.src_stage_mask
            == vk::PipelineStageFlagBits2
                .STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
            && native_build_to_build.dst_stage_mask
                == vk::PipelineStageFlagBits2
                    .STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
            && (ulong)native_build_to_build.src_access_mask
                == acceleration_structure_access
            && (ulong)native_build_to_build.dst_access_mask
                == acceleration_structure_access,
        "build-to-build barriers expose exact AS read/write scopes");

    gpu::Barrier build_to_query = {
        .before = { .acceleration_structure_build },
        .after = { .compute },
    };
    vk::MemoryBarrier2 native_build_to_query =
        gpu::internal::vk::barrier_to_vk(&build_to_query, false, true);
    ulong query_access =
        (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2
            .ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    assert((ulong)native_build_to_query.src_access_mask
            == acceleration_structure_access
            && native_build_to_query.dst_stage_mask
                == vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT
            && (ulong)native_build_to_query.dst_access_mask == query_access,
        "build-to-query adds AS traversal reads to the calling shader stage");
    gpu::StageMask graphics_query = {
        .vertex_shader,
        .fragment_shader,
    };
    ulong graphics_query_access =
        (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
        | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT
        | (ulong)vk::AccessFlagBits2
            .ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    assert((ulong)gpu::internal::vk::barrier_after_access_to_vk(
            graphics_query,
            false,
            true,
        ) == graphics_query_access,
        "vertex and fragment query scopes include AS traversal reads");
    assert((ulong)gpu::internal::vk::barrier_after_access_to_vk(
            { .compute },
            false,
            false,
        ) == (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
                | (ulong)vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT,
        "baseline shader scopes remain free of AS traversal access");
}

fn void semantic_barrier_masks_reject_empty_and_contradictory_values() @test {
    gpu::Barrier valid = {
        .before = { .compute = true },
        .after = { .fragment_shader = true },
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

    gpu::Barrier unknown_stage = valid;
    unknown_stage.before = (gpu::StageMask)((uint)valid.before | 0x800u);
    assert(!gpu::internal::vk::barrier_masks_valid(&unknown_stage),
        "mixed known and unknown stage-mask bits are invalid");
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
        .after = { .indirect = true },
    };
    assert(!gpu::internal::vk::barrier_queue_flags_valid(
            &draw_arguments,
            transfer_only,
        ),
        "transfer-only queues reject indirect-command stages");
    assert(gpu::internal::vk::barrier_queue_flags_valid(
            &draw_arguments,
            compute_only,
        ),
        "compute-capable families accept indirect-command stages");

    gpu::Barrier depth_stencil = {
        .before = { .transfer = true },
        .after = { .depth_output = true },
    };
    assert(!gpu::internal::vk::barrier_queue_flags_valid(
            &depth_stencil,
            compute_only,
        ),
        "compute-only queues reject depth-output stages");
    assert(gpu::internal::vk::barrier_queue_flags_valid(
            &depth_stencil,
            graphics_only,
        ),
        "graphics-capable families accept depth-output stages");
}

fn void index_type_validation_rejects_unknown_values() @test {
    assert(gpu::internal::vk::index_element_size(gpu::IndexType.U16)!! == 2);
    assert(gpu::internal::vk::index_element_size(gpu::IndexType.U32)!! == 4);
    gpu::IndexType unknown_type;
    *(int*)&unknown_type = 99;
    fault unknown = @catch(gpu::internal::vk::index_element_size(unknown_type));
    assert(unknown == gpu::INVALID_ARGUMENT, "unknown index format faults");
}

fn void presentation_requires_completion_fences() @test {
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

struct SparseFeaturePlanCase {
    bool requested;
    uint sparse_binding;
    uint image_2d;
    uint image_3d;
    bool supported;
    bool expected_2d;
    bool expected_3d;
}

fn void sparse_texture_feature_plan_is_exact() @test {
    SparseFeaturePlanCase[6] cases = {
        { false, vk::FALSE, vk::FALSE, vk::FALSE, true,  false, false },
        { true,  vk::FALSE, vk::TRUE,  vk::FALSE, false, false, false },
        { true,  vk::TRUE,  vk::FALSE, vk::FALSE, false, false, false },
        { true,  vk::TRUE,  vk::TRUE,  vk::FALSE, true,  true,  false },
        { true,  vk::TRUE,  vk::FALSE, vk::TRUE,  true,  false, true  },
        { true,  vk::TRUE,  vk::TRUE,  vk::TRUE,  true,  true,  true  },
    };

    foreach (test_case : cases) {
        vk::PhysicalDeviceFeatures features = {
            .sparse_binding             = test_case.sparse_binding,
            .sparse_residency_image2d   = test_case.image_2d,
            .sparse_residency_image3d   = test_case.image_3d,
        };
        vk::SparseTextureFeaturePlan? result =
            vk::resolve_sparse_texture_feature_plan(
                test_case.requested,
                &features,
            );
        if (!test_case.supported) {
            assert(@catch(result) == gpu::UNSUPPORTED_FEATURE);
            continue;
        }

        vk::SparseTextureFeaturePlan plan = result!!;
        assert(plan.enabled == test_case.requested);
        assert(plan.image_2d == test_case.expected_2d);
        assert(plan.image_3d == test_case.expected_3d);
    }
}

fn void sparse_texture_feature_plan_rejects_null_requested_features() @test {
    assert(@catch(vk::resolve_sparse_texture_feature_plan(true, null))
        == gpu::INVALID_ARGUMENT);
    vk::SparseTextureFeaturePlan plan =
        vk::resolve_sparse_texture_feature_plan(false, null)!!;
    assert(!plan.enabled && !plan.image_2d && !plan.image_3d);
}
