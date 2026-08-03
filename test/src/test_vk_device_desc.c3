module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::atomic;
import std::core::mem;
import std::io;

struct BaselineSupportDiagnosticCapture {
    uint calls;
    ZString expected_field;
    ZString expected_invariant;
    bool matched;
}

struct BaselineSupportDiagnosticCase {
    gpu::internal::vk::IndexingHeapLimits limits;
    ZString                               expected_field;
    ZString                               expected_invariant;
}

struct NativeDeviceCreateCapture {
    uint calls;
}

struct DeviceFeatureChainCapture {
    uint calls;
    bool found_base_features;
    bool sparse_binding;
    bool sparse_image_2d;
    bool sparse_image_3d;
    bool sparse_uncontracted_features;
    bool found_features12;
    bool indexing_features_enabled;
    bool found_dynamic_color;
    bool dynamic_color_features_enabled;
    bool found_acceleration_structure;
    bool acceleration_structure_enabled;
    bool acceleration_structure_descriptor_update_after_bind;
    bool found_ray_query;
    bool ray_query_enabled;
    bool acceleration_structure_extension;
    bool ray_query_extension;
    bool deferred_host_operations_extension;
}

fn void capture_baseline_support_diagnostic(
    gpu::DebugMessage* message,
    void* user_data,
) {
    BaselineSupportDiagnosticCapture* capture =
        (BaselineSupportDiagnosticCapture*)user_data;
    bool matched = message.severity == gpu::DebugMessageSeverity.ERROR
        && message.category.public_contract
        && message.has_fault
        && message.public_fault == gpu::UNSUPPORTED_FEATURE
        && message.operation.str_view() == "create_device"
        && message.rejected_field.str_view() == capture.expected_field.str_view()
        && message.invariant.str_view() == capture.expected_invariant.str_view();
    if (matched) {
        capture.calls++;
        capture.matched = true;
    }
}

fn vk::Result count_native_device_create(
    void* user_data,
    vk::PhysicalDevice physical_device,
    vk::DeviceCreateInfo* create_info,
    vk::Device* device,
) {
    (void)physical_device;
    (void)create_info;
    (void)device;
    NativeDeviceCreateCapture* capture = (NativeDeviceCreateCapture*)user_data;
    capture.calls++;
    return vk::Result.ERROR_INITIALIZATION_FAILED;
}

fn vk::Result capture_device_feature_chain(
    void* user_data,
    vk::PhysicalDevice physical_device,
    vk::DeviceCreateInfo* create_info,
    vk::Device* device,
) {
    (void)physical_device;
    (void)device;
    DeviceFeatureChainCapture* capture =
        (DeviceFeatureChainCapture*)user_data;
    capture.calls++;

    for (uint i = 0; i < create_info.enabled_extension_count; i++) {
        String extension = ((ZString)create_info
            .pp_enabled_extension_names[i]).str_view();
        if (extension == "VK_KHR_acceleration_structure") {
            capture.acceleration_structure_extension = true;
        } else if (extension == "VK_KHR_ray_query") {
            capture.ray_query_extension = true;
        } else if (extension == "VK_KHR_deferred_host_operations") {
            capture.deferred_host_operations_extension = true;
        }
    }

    vk::BaseInStructure* feature =
        (vk::BaseInStructure*)create_info.p_next;
    while (feature != null) {
        if (feature.s_type == vk::StructureType.PHYSICAL_DEVICE_FEATURES_2) {
            vk::PhysicalDeviceFeatures2* features2 =
                (vk::PhysicalDeviceFeatures2*)feature;
            capture.found_base_features = true;
            capture.sparse_binding =
                features2.features.sparse_binding == vk::TRUE;
            capture.sparse_image_2d =
                features2.features.sparse_residency_image2d == vk::TRUE;
            capture.sparse_image_3d =
                features2.features.sparse_residency_image3d == vk::TRUE;
            capture.sparse_uncontracted_features =
                features2.features.sparse_residency_buffer != vk::FALSE
                || features2.features.sparse_residency2samples != vk::FALSE
                || features2.features.sparse_residency4samples != vk::FALSE
                || features2.features.sparse_residency8samples != vk::FALSE
                || features2.features.sparse_residency16samples != vk::FALSE
                || features2.features.sparse_residency_aliased != vk::FALSE;
        } else if (feature.s_type
            == vk::StructureType.PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
            vk::PhysicalDeviceVulkan12Features* features12 =
                (vk::PhysicalDeviceVulkan12Features*)feature;
            capture.found_features12 = true;
            capture.indexing_features_enabled =
                features12.runtime_descriptor_array == vk::TRUE
                && features12.shader_sampled_image_array_non_uniform_indexing
                    == vk::TRUE
                && features12.shader_storage_image_array_non_uniform_indexing
                    == vk::TRUE
                && features12.descriptor_binding_partially_bound == vk::TRUE
                && features12.descriptor_binding_sampled_image_update_after_bind
                    == vk::TRUE
                && features12.descriptor_binding_storage_image_update_after_bind
                    == vk::TRUE
                && features12.descriptor_binding_update_unused_while_pending
                    == vk::TRUE;
        } else if (feature.s_type
            == vk::StructureType.PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT) {
            vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT* dynamic_color =
                (vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT*)feature;
            capture.found_dynamic_color = true;
            capture.dynamic_color_features_enabled =
                dynamic_color.extended_dynamic_state3color_blend_enable
                    == vk::TRUE
                && dynamic_color.extended_dynamic_state3color_blend_equation
                    == vk::TRUE
                && dynamic_color.extended_dynamic_state3color_write_mask
                    == vk::TRUE;
        } else if (feature.s_type
            == vk::StructureType
                .PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR) {
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR*
                acceleration_structure =
                    (vk::PhysicalDeviceAccelerationStructureFeaturesKHR*)feature;
            capture.found_acceleration_structure = true;
            capture.acceleration_structure_enabled =
                acceleration_structure.acceleration_structure == vk::TRUE;
            capture.acceleration_structure_descriptor_update_after_bind =
                acceleration_structure
                    .descriptor_binding_acceleration_structure_update_after_bind
                    == vk::TRUE;
        } else if (feature.s_type
            == vk::StructureType.PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR) {
            vk::PhysicalDeviceRayQueryFeaturesKHR* ray_query =
                (vk::PhysicalDeviceRayQueryFeaturesKHR*)feature;
            capture.found_ray_query = true;
            capture.ray_query_enabled = ray_query.ray_query == vk::TRUE;
        }
        feature = feature.p_next;
    }
    return vk::Result.ERROR_INITIALIZATION_FAILED;
}

fn uint live_device_slot_count() @private {
    uint live_count;
    for (uint i = 0; i < gpu::internal::MAX_DEVICES; i++) {
        gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[i];
        gpu::internal::DeviceSlotControl control =
            (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
        if (control.state == gpu::internal::DEVICE_SLOT_LIVE) live_count++;
    }
    return live_count;
}

fn fault run_injected_device_create(
    gpu::Runtime* runtime,
    gpu::Adapter* adapter,
    gpu::DeviceDesc* device_desc,
    gpu::internal::vk::DeviceCreationProbe* probe,
    gpu::internal::vk::DeviceCreationOps* creation_ops = null,
) @private {
    gpu::Runtime retained_runtime = *runtime;
    gpu::internal::runtime_retain_device(&retained_runtime)!!;
    gpu::Device? creation_result =
        gpu::internal::vk::create_vk_device_with_probe(
            adapter:          adapter,
            device_desc:      device_desc,
            retained_runtime: &retained_runtime,
            probe:            probe,
            creation_ops:     creation_ops,
        );
    fault creation_fault = @catch(creation_result);
    if (creation_fault) {
        gpu::internal::runtime_release_device(&retained_runtime)!!;
        return creation_fault;
    }
    gpu::Device device = creation_result!!;
    gpu::destroy_device(&device)!!;
    return creation_fault;
}

fn vk::Result reject_native_device_create(
    void* user_data,
    vk::PhysicalDevice physical_device,
    vk::DeviceCreateInfo* requested_info,
    vk::Device* device,
) @private {
    (void)user_data;
    (void)physical_device;
    (void)requested_info;
    (void)device;
    return vk::Result.ERROR_INITIALIZATION_FAILED;
}

fn void destroy_registry_blockers(gpu::Device[] blockers) @private {
    foreach (&blocker : blockers) {
        if (blocker.is_valid()) (void)gpu::destroy_device(blocker);
    }
}

fn uptr device_physical_adapter(gpu::Device* device) @private {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::vk::VkDeviceState* state = operation.state;
    return (uptr)state.physical_device;
}

fn gpu::internal::vk::IndexingHeapLimits permissive_indexing_heap_limits() @private {
    return {
        .max_sampled_images           = uint::max,
        .max_storage_images           = uint::max,
        .max_samplers                 = uint::max,
        .max_per_stage_sampled_images = uint::max,
        .max_per_stage_storage_images = uint::max,
        .max_per_stage_samplers       = uint::max,
        .max_per_stage_resources      = uint::max,
        .max_all_pools_descriptors    = uint::max,
    };
}

fn void shader_heap_state_preserves_configured_capacities() @test {
    gpu::internal::vk::RuntimeConfig config = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
    };
    gpu::internal::vk::VkShaderHeapState indexing =
        gpu::internal::vk::resolve_shader_heap_state(&config)!!;
    assert(indexing.texture_capacity == 7, "the requested texture capacity is preserved");
    assert(indexing.sampler_capacity == 9, "the requested sampler capacity is preserved");
}

fn void shader_heap_defaults_fit_required_indexing_limits() @test {
    gpu::internal::vk::IndexingHeapLimits required_limits = {
        .max_sampled_images           = 500_000,
        .max_storage_images           = 500_000,
        .max_samplers                 = 500_000,
        .max_per_stage_sampled_images = 500_000,
        .max_per_stage_storage_images = 500_000,
        .max_per_stage_samplers       = 500_000,
        .max_per_stage_resources      = 500_000,
        .max_all_pools_descriptors    = 500_000,
    };
    assert(
        gpu::internal::vk::indexing_heap_limit_failure(
            gpu::DEFAULT_TEXTURE_HEAP_CAPACITY,
            gpu::DEFAULT_SAMPLER_HEAP_CAPACITY,
            required_limits,
        ) == gpu::internal::vk::IndexingHeapLimitFailure.NONE,
        "the shipped defaults fit Vulkan's required update-after-bind limits"
    );
}

fn void shader_heap_support_reports_exact_failure() @test {
    gpu::internal::vk::RuntimeConfig config = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
    };
    gpu::internal::vk::VkShaderHeapState heap =
        gpu::internal::vk::resolve_shader_heap_state(&config)!!;
    BaselineSupportDiagnosticCapture capture = {
        .expected_field = "device baseline",
        .expected_invariant =
            "the device baseline requires descriptor indexing with update-after-bind support",
    };
    gpu::internal::vk::VkDeviceState device = {
        .debug_callback  = &capture_baseline_support_diagnostic,
        .debug_user_data = &capture,
    };
    gpu::internal::vk::ShaderHeapSupport support = {
        .limits = permissive_indexing_heap_limits(),
    };
    assert(@catch(gpu::internal::vk::validate_shader_heap_support(
        &device,
        &heap,
        &support,
    )) == gpu::UNSUPPORTED_FEATURE);
    assert(capture.calls == 1 && capture.matched,
        "missing indexing support reports the exact baseline requirement");

    BaselineSupportDiagnosticCase[5] cases;
    gpu::internal::vk::IndexingHeapLimits limits =
        permissive_indexing_heap_limits();
    limits.max_sampled_images = 6;
    cases[0] = {
        .limits          = limits,
        .expected_field = "RuntimeDesc.texture_heap_capacity",
        .expected_invariant =
            "texture heap capacity must fit the sampled-image update-after-bind limit",
    };
    limits = permissive_indexing_heap_limits();
    limits.max_storage_images = 6;
    cases[1] = {
        .limits          = limits,
        .expected_field = "RuntimeDesc.texture_heap_capacity",
        .expected_invariant =
            "texture heap capacity must fit the storage-image update-after-bind limit",
    };
    limits = permissive_indexing_heap_limits();
    limits.max_samplers = 8;
    cases[2] = {
        .limits          = limits,
        .expected_field = "RuntimeDesc.sampler_heap_capacity",
        .expected_invariant =
            "sampler heap capacity must fit the sampler update-after-bind limit",
    };
    limits = permissive_indexing_heap_limits();
    limits.max_per_stage_resources = 13;
    cases[3] = {
        .limits          = limits,
        .expected_field =
            "RuntimeDesc.texture_heap_capacity/sampler_heap_capacity",
        .expected_invariant =
            "combined descriptor capacity must fit the per-stage update-after-bind resource limit",
    };
    limits = permissive_indexing_heap_limits();
    limits.max_all_pools_descriptors = 22;
    cases[4] = {
        .limits          = limits,
        .expected_field =
            "RuntimeDesc.texture_heap_capacity/sampler_heap_capacity",
        .expected_invariant =
            "combined descriptor capacity must fit the all-pools update-after-bind descriptor limit",
    };

    support.indexing = true;
    foreach (&diagnostic_case : cases) {
        capture = {
            .expected_field     = diagnostic_case.expected_field,
            .expected_invariant = diagnostic_case.expected_invariant,
        };
        support.limits = diagnostic_case.limits;
        assert(@catch(gpu::internal::vk::validate_shader_heap_support(
            &device,
            &heap,
            &support,
        )) == gpu::UNSUPPORTED_FEATURE);
        assert(capture.calls == 1 && capture.matched,
            "each baseline capacity failure reports its exact diagnostic mapping");
    }
}

fn void runtime_owned_configuration_reaches_device_creation() @test {
    char[4] cache_data = { 1, 2, 3, 4 };
    gpu::RuntimeDesc desc = {
        .texture_heap_capacity = 17,
        .sampler_heap_capacity = 19,
        .texture_capacity      = 23,
        .pipeline_capacity     = 29,
        .pipeline_cache_data   = cache_data[..],
        .application_name      = "gpu_vk_runtime_defaults_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    desc = {};
    cache_data[0] = 9;
    gpu::internal::vk::VkRuntimeState* runtime_state =
        gpu::internal::resolve_runtime(&runtime).state;
    assert(
        runtime_state.config.pipeline_cache_data.len == cache_data.len
            && runtime_state.config.pipeline_cache_data[0] == 1,
        "caller mutation cannot change the runtime cache copy"
    );
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter)!!.supported) continue;

        gpu::Device device = gpu::create_device(&adapter)!!;
        gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
        assert(
            caps.texture_heap_capacity == 17
                && caps.sampler_heap_capacity == 19,
            "the published device retains explicit runtime heap capacities"
        );
        gpu::destroy_device(&device)!!;
        return;
    }
    assert(false, "the controlled native target supports the device baseline");
}

fn void adapter_support_and_creation_share_runtime_capacities() @test {
    gpu::RuntimeDesc desc = {
        .texture_heap_capacity = gpu::MAX_SHADER_HEAP_CAPACITY,
        .sampler_heap_capacity = gpu::MAX_SHADER_HEAP_CAPACITY,
        .application_name      = "gpu_vk_adapter_support_config_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    assert(adapters.count > 0, "the controlled target exposes an adapter");
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        gpu::DeviceSupport support =
            gpu::supports_device_desc(&adapter)!!;
        if (!support.supported) {
            assert(
                @catch(gpu::create_device(&adapter)) == gpu::UNSUPPORTED_FEATURE,
                "creation rejects a configuration reported as unsupported"
            );
            continue;
        }
        gpu::Device device = gpu::create_device(&adapter)!!;
        gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
        assert(
            caps.texture_heap_capacity == gpu::MAX_SHADER_HEAP_CAPACITY,
            "creation preserves the supported runtime texture capacity"
        );
        assert(
            caps.sampler_heap_capacity == gpu::MAX_SHADER_HEAP_CAPACITY,
            "creation preserves the supported runtime sampler capacity"
        );
        gpu::destroy_device(&device)!!;
    }
}

fn void public_creation_reports_cached_baseline_capacity_failure() @test {
    BaselineSupportDiagnosticCapture capture = {
        .expected_field = "RuntimeDesc.texture_heap_capacity",
        .expected_invariant =
            "texture heap capacity must fit the sampled-image update-after-bind limit",
    };
    gpu::RuntimeDesc desc = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
        .debug_callback        = &capture_baseline_support_diagnostic,
        .debug_user_data       = &capture,
        .application_name      = "gpu_vk_cached_baseline_support_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::internal::RuntimeData* runtime_data =
        gpu::internal::resolve_runtime(&runtime);
    gpu::internal::vk::VkRuntimeState* runtime_state = runtime_data.state;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter)!!.supported) {
            continue;
        }
        gpu::internal::vk::VkAdapterState* adapter_state =
            &runtime_state.adapters[adapter.adapter_index];
        gpu::internal::vk::ShaderHeapSupport saved_support =
            adapter_state.shader_heap;
        adapter_state.shader_heap = {
            .indexing = true,
            .limits   = permissive_indexing_heap_limits(),
        };
        adapter_state.shader_heap.limits.max_sampled_images = 6;

        fault rejected = @catch(gpu::create_device(&adapter));
        adapter_state.shader_heap = saved_support;

        assert(rejected == gpu::UNSUPPORTED_FEATURE,
            "public creation rejects the cached baseline capacity failure");
        assert(capture.calls == 1 && capture.matched,
            "public creation reports the exact cached capacity diagnostic");
        return;
    }
    assert(false, "the pinned Vulkan adapter supports the device baseline");
}

fn void cached_baseline_failure_precedes_native_device_creation() @test {
    gpu::RuntimeDesc desc = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
        .application_name      = "gpu_vk_cached_baseline_native_gate_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::internal::RuntimeData* runtime_data =
        gpu::internal::resolve_runtime(&runtime);
    gpu::internal::vk::VkRuntimeState* runtime_state = runtime_data.state;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter)!!.supported) {
            continue;
        }
        gpu::internal::vk::VkAdapterState* adapter_state =
            &runtime_state.adapters[adapter.adapter_index];
        gpu::internal::vk::ShaderHeapSupport saved_support =
            adapter_state.shader_heap;
        adapter_state.shader_heap = {
            .indexing = true,
            .limits   = permissive_indexing_heap_limits(),
        };
        adapter_state.shader_heap.limits.max_sampled_images = 6;

        NativeDeviceCreateCapture create_capture;
        gpu::internal::vk::DeviceCreationProbe probe;
        gpu::internal::vk::DeviceCreationOps creation_ops = {
            .user_data = &create_capture,
            .create    = &count_native_device_create,
        };
        gpu::Runtime retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!!;
        fault rejected =
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                device_desc:      null,
                retained_runtime: &retained_runtime,
                probe:            &probe,
                creation_ops:     &creation_ops,
            ));
        gpu::internal::runtime_release_device(&retained_runtime)!!;
        adapter_state.shader_heap = saved_support;

        assert(rejected == gpu::UNSUPPORTED_FEATURE,
            "cached baseline capacity failure rejects backend creation");
        assert(create_capture.calls == 0,
            "cached baseline capacity failure never reaches vkCreateDevice");
        assert(
            probe.reached == gpu::internal::vk::DeviceCreationCheckpoint.NONE
                && probe.rollback_count == 1,
            "cached baseline rejection precedes the first creation checkpoint"
        );
        return;
    }
    assert(false, "the pinned Vulkan adapter supports the device baseline");
}

fn void device_creation_enables_required_native_features() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_vk_required_indexing_features_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter)!!.supported) {
            continue;
        }
        DeviceFeatureChainCapture capture;
        gpu::internal::vk::DeviceCreationProbe probe;
        gpu::internal::vk::DeviceCreationOps creation_ops = {
            .user_data = &capture,
            .create    = &capture_device_feature_chain,
        };
        uint live_slots_before = live_device_slot_count();
        gpu::Runtime retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!!;
        fault rejected =
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                device_desc:      null,
                retained_runtime: &retained_runtime,
                probe:            &probe,
                creation_ops:     &creation_ops,
            ));
        gpu::internal::runtime_release_device(&retained_runtime)!!;

        assert(rejected == gpu::UNSUPPORTED_BACKEND,
            "the native hook aborts creation after inspecting the feature chain");
        assert(capture.calls == 1 && capture.found_features12,
            "device creation chains Vulkan 1.2 features into vkCreateDevice");
        assert(capture.indexing_features_enabled,
            "device creation enables all seven required indexing feature bits");
        assert(capture.found_base_features,
            "device creation chains core base features");
        assert(!capture.sparse_binding
                && !capture.sparse_image_2d
                && !capture.sparse_image_3d
                && !capture.sparse_uncontracted_features,
            "default creation enables no sparse feature");
        assert(capture.found_dynamic_color
                && capture.dynamic_color_features_enabled,
            "device creation chains and enables all required color features");
        assert(!capture.found_acceleration_structure
                && !capture.found_ray_query
                && !capture.acceleration_structure_extension
                && !capture.ray_query_extension
                && !capture.deferred_host_operations_extension,
            "baseline creation requests no ray-query feature family");
        assert(
            probe.reached == gpu::internal::vk::DeviceCreationCheckpoint.NONE
                && probe.rollback_count == 1
                && live_device_slot_count() == live_slots_before,
            "the injected native creation failure rolls back without publication"
        );
        return;
    }
    assert(false, "the pinned Vulkan adapter supports the device baseline");
}

fn void ray_query_device_creation_enables_the_complete_optional_family() @test {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "gpu_vk_ray_query_feature_chain_test",
        .acceleration_structure_heap_capacity = 8,
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceDesc device_desc = { .enable_ray_queries = true };

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        gpu::DeviceSupport support =
            gpu::supports_device_desc(&adapter, &device_desc)!!;
        if (!support.supported) {
            assert(@catch(gpu::create_device(&adapter, &device_desc))
                == gpu::UNSUPPORTED_FEATURE);
            continue;
        }
        DeviceFeatureChainCapture capture;
        gpu::internal::vk::DeviceCreationProbe probe;
        gpu::internal::vk::DeviceCreationOps creation_ops = {
            .user_data = &capture,
            .create    = &capture_device_feature_chain,
        };
        gpu::Runtime retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!!;
        fault rejected =
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                device_desc:      &device_desc,
                retained_runtime: &retained_runtime,
                probe:            &probe,
                creation_ops:     &creation_ops,
            ));
        gpu::internal::runtime_release_device(&retained_runtime)!!;

        assert(rejected == gpu::UNSUPPORTED_BACKEND);
        assert(capture.found_acceleration_structure
                && capture.acceleration_structure_enabled
                && capture.acceleration_structure_descriptor_update_after_bind,
            "opt-in enables the complete acceleration-structure feature node");
        assert(capture.found_ray_query && capture.ray_query_enabled,
            "opt-in enables the ray-query feature node");
        assert(capture.acceleration_structure_extension
                && capture.ray_query_extension
                && capture.deferred_host_operations_extension,
            "opt-in requests the complete extension family together");
        assert(probe.reached == gpu::internal::vk::DeviceCreationCheckpoint.NONE
                && probe.rollback_count == 1,
            "injected native failure rolls optional feature state back");
        return;
    }
    assert(false, "the controlled native target supports ray queries");
}

fn void requested_sparse_creation_enables_only_planned_features() @test {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "gpu_vk_sparse_feature_plan_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceDesc device_desc = {
        .enable_sparse_textures = true,
    };
    bool checked_unsupported;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter, &device_desc)!!.supported) {
            assert(@catch(gpu::create_device(&adapter, &device_desc))
                == gpu::UNSUPPORTED_FEATURE,
                "unsupported sparse creation takes the verified negative path");
            checked_unsupported = true;
            continue;
        }
        DeviceFeatureChainCapture capture;
        gpu::internal::vk::DeviceCreationProbe probe;
        gpu::internal::vk::DeviceCreationOps creation_ops = {
            .user_data = &capture,
            .create    = &capture_device_feature_chain,
        };
        gpu::Runtime retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!!;
        fault rejected =
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                device_desc:      &device_desc,
                retained_runtime: &retained_runtime,
                probe:            &probe,
                creation_ops:     &creation_ops,
            ));
        gpu::internal::runtime_release_device(&retained_runtime)!!;

        assert(rejected == gpu::UNSUPPORTED_BACKEND);
        assert(capture.found_base_features && capture.sparse_binding);
        assert(capture.sparse_image_2d || capture.sparse_image_3d);
        assert(!capture.sparse_uncontracted_features);
        return;
    }
    assert(checked_unsupported,
        "the controlled native target exposes a sparse positive or negative path");
}

fn void default_device_caps_hide_sparse_support() @test {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "gpu_vk_default_sparse_caps_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceDesc device_desc;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter, &device_desc)!!.supported) {
            continue;
        }
        gpu::Device device =
            gpu::create_device(&adapter, &device_desc)!!;
        gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
        assert(!caps.sparse_textures.image_2d
                && !caps.sparse_textures.image_3d
                && !caps.sparse_textures.nonresident_strict
                && (uint)caps.sparse_textures.binding_queues == 0,
            "a zero DeviceDesc publishes entirely zero sparse capabilities");
        assert(!caps.ray_queries.enabled
                && caps.acceleration_structures.heap_capacity == 0
                && caps.acceleration_structures.max_geometry_count == 0
                && caps.acceleration_structures.max_primitive_count == 0
                && caps.acceleration_structures.max_instance_count == 0
                && caps.acceleration_structures.scratch_alignment == 0,
            "a zero DeviceDesc publishes entirely zero ray-query capabilities");
        gpu::destroy_device(&device)!!;
        return;
    }
    assert(false, "the controlled native target supports the device baseline");
}

fn void ray_query_support_and_created_caps_are_consistent() @test {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "gpu_vk_ray_query_caps_test",
        .acceleration_structure_heap_capacity = 7,
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceDesc device_desc = { .enable_ray_queries = true };

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        gpu::DeviceSupport support =
            gpu::supports_device_desc(&adapter, &device_desc)!!;
        if (!support.supported) {
            assert(@catch(gpu::create_device(&adapter, &device_desc))
                == gpu::UNSUPPORTED_FEATURE);
            continue;
        }

        gpu::Device device = gpu::create_device(&adapter, &device_desc)!!;
        gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
        gpu::internal::vk::VkDeviceState* state =
            gpu::internal::device_slots[device.index].data.state;
        gpu::AccelerationStructureCaps expected =
            gpu::internal::vk::normalize_acceleration_structure_caps(
                true,
                runtime_desc.acceleration_structure_heap_capacity,
                state.shader_heap.texture_capacity,
                state.shader_heap.sampler_capacity,
                &state.ray_query_support,
            );
        assert(caps.ray_queries.enabled
                && caps.acceleration_structures.heap_capacity == 7
                && caps.acceleration_structures.max_geometry_count
                    == expected.max_geometry_count
                && caps.acceleration_structures.max_primitive_count
                    == expected.max_primitive_count
                && caps.acceleration_structures.max_instance_count
                    == expected.max_instance_count
                && caps.acceleration_structures.scratch_alignment
                    == expected.scratch_alignment,
            "created caps exactly normalize the selected device properties");
        gpu::destroy_device(&device)!!;
        return;
    }
    assert(false, "the controlled native target supports ray queries");
}

fn void sparse_support_and_created_caps_are_consistent() @test {
    gpu::RuntimeDesc runtime_desc = {
        .application_name = "gpu_vk_sparse_caps_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceDesc device_desc = {
        .enable_sparse_textures = true,
    };

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        gpu::DeviceSupport support =
            gpu::supports_device_desc(&adapter, &device_desc)!!;
        if (!support.supported) {
            assert(@catch(gpu::create_device(&adapter, &device_desc))
                == gpu::UNSUPPORTED_FEATURE);
            continue;
        }

        gpu::Device device =
            gpu::create_device(&adapter, &device_desc)!!;
        gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
        gpu::internal::vk::VkDeviceState* state =
            gpu::internal::device_slots[device.index].data.state;
        vk::PhysicalDeviceFeatures2 features2 =
            vk::physical_device_features2();
        vk::get_physical_device_features2(
            state.physical_device,
            &features2,
        );
        vk::PhysicalDeviceProperties properties;
        vk::get_physical_device_properties(
            state.physical_device,
            &properties,
        );

        assert(caps.sparse_textures.image_2d
            == (features2.features.sparse_residency_image2d != 0));
        assert(caps.sparse_textures.image_3d
            == (features2.features.sparse_residency_image3d != 0));
        assert(caps.sparse_textures.nonresident_strict
            == (properties.sparse_properties.residency_non_resident_strict
                != 0));
        assert((uint)caps.sparse_textures.binding_queues
            == (uint)state.queues.sparse_binding_queues);
        assert((uint)caps.sparse_textures.binding_queues != 0);

        gpu::destroy_device(&device)!!;
    }
}

fn void adapter_desc_creation_reuses_runtime_discovery() @test {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_vk_device_desc_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    if (adapters.count == 0) {
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        gpu::DeviceSupport support =
            gpu::supports_device_desc(&adapter)!!;
        if (!support.supported) continue;

        gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_adapter(&adapter);
        gpu::internal::vk::VkRuntimeState* runtime_state =
            runtime_data.state;
        gpu::Device device = gpu::create_device(&adapter)!!;
        gpu::internal::vk::VkDeviceState* device_state =
            gpu::internal::device_slots[device.index].data.state;

        assert(device_state.instance == runtime_state.instance, "the device reuses the runtime instance");
        assert(
            device_state.physical_device
                == runtime_state.adapters[adapter.adapter_index].physical_device,
            "the device uses the exact requested physical adapter"
        );
        assert(
            device_state.validation_policy.contract
                == gpu::ContractValidation.FULL,
            "the device inherits runtime contract validation"
        );
        assert(
            device_state.validation_policy.vulkan_layers,
            "the device inherits runtime Vulkan validation layers"
        );
        gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
        assert(
            gpu::internal::vk::device_dispatch_supports_dynamic_color(
                &device_state.device_dispatch,
            ),
            "every device loads the required color dispatch"
        );
        assert(
            device_state.shader_heap.descriptors.texture_cells.ptr != null,
            "the device owns texture descriptor cells"
        );
        assert(
            device_state.samplers.slots.ptr != null,
            "the device owns an independent sampler identity table"
        );
        assert(
            device_state.shader_heap.descriptors.sampler_next_new == 0,
            "the sampler heap begins unpublished"
        );
        assert(
            device_state.shader_heap.descriptors.set_layout != null,
            "the device owns its descriptor layout"
        );
        assert(
            caps.texture_heap_capacity == device_state.shader_heap.texture_capacity,
            "published texture capacity matches shader heap state"
        );
        assert(
            caps.sampler_heap_capacity == device_state.shader_heap.sampler_capacity,
            "published sampler capacity matches shader heap state"
        );
        assert(!device_state.presentation.enabled, "headless devices create no presentation state");
        assert(device_state.presentation.queue == null, "headless devices retrieve no presentation queue");
        assert(!device_state.presentation.queue_mutex.is_initialized(), "headless devices initialize no presentation mutex");
        assert(device_state.presentation.swapchains.slots.ptr == null, "headless devices allocate no swapchain table");
        assert(
            !gpu::internal::vk::instance_dispatch_supports_presentation(&device_state.instance_dispatch),
            "headless devices retain no surface procedures"
        );
        assert(
            !gpu::internal::vk::device_dispatch_supports_presentation(&device_state.device_dispatch),
            "headless devices load no presentation procedures"
        );
        assert(
            @catch(gpu::destroy_runtime(&runtime)) == gpu::RESOURCE_IN_USE,
            "the live device retains its runtime"
        );

        gpu::destroy_device(&device)!!;
        assert(
            gpu::get_adapter_info(&adapter)!!.name.len > 0,
            "device destruction leaves runtime discovery live"
        );
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::Adapter unsupported_adapter = adapters.get(0)!!;
    gpu::DeviceSupport unsupported =
        gpu::supports_device_desc(&unsupported_adapter)!!;
    assert(!unsupported.supported, "no adapter bypasses the device baseline");
    assert(
        @catch(gpu::create_device(&unsupported_adapter)) == gpu::UNSUPPORTED_FEATURE,
        "unsupported native creation publishes no device"
    );
    gpu::destroy_runtime(&runtime)!!;
}

fn void malformed_native_desc_preserves_runtime_state() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_vk_device_desc_failure_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    if (adapters.count == 0) {
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::Adapter adapter = adapters.get(0)!!;
    gpu::DeviceDesc invalid = {
        .queues = {
            .required = { .graphics },
            .distinct = { .compute },
        },
    };
    assert(
        @catch(gpu::create_device(&adapter, &invalid)) == gpu::INVALID_ARGUMENT,
        "an invalid descriptor is rejected before native creation"
    );
    assert(
        gpu::get_adapter_info(&adapter)!!.name.len > 0,
        "failed creation leaves runtime-owned adapter state intact"
    );
    gpu::destroy_runtime(&runtime)!!;
}

fn void presentation_device_creation_requires_instance_maintenance() @test {
    gpu::internal::vk::VkRuntimeState* runtime_state =
        mem::new(gpu::internal::vk::VkRuntimeState);
    runtime_state.adapters = mem::new_array(
        gpu::internal::vk::VkAdapterState,
        1,
    );
    gpu::Runtime runtime = gpu::internal::register_runtime(
        state:         runtime_state,
        adapter_count: 1,
    )!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::Surface surface = gpu::internal::register_surface(&runtime, (void*)1)!!;
    defer gpu::destroy_surface(&surface)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::Adapter adapter = adapters.get(0)!!;
    gpu::DeviceDesc device_desc = { .surface = surface };
    gpu::internal::vk::DeviceCreationProbe probe;

    assert(
        @catch(gpu::internal::vk::create_vk_device_with_probe(
            adapter:          &adapter,
            device_desc:      &device_desc,
            retained_runtime: &runtime,
            probe:            &probe,
        )) == gpu::UNSUPPORTED_FEATURE,
        "presentation creation requires instance maintenance support"
    );
    assert(
        probe.reached == gpu::internal::vk::DeviceCreationCheckpoint.NONE
            && probe.rollback_count == 0,
        "the instance capability gate precedes backend device mutation"
    );
}

fn void command_policy_selection_covers_both_contract_modes() @test {
    gpu::internal::vk::DeviceValidationPolicy policy;
    uptr trusted_address = (uptr)gpu::internal::vk::select_command_ops(policy);
    assert(
        gpu::internal::vk::select_command_ops(policy)
            == &gpu::internal::vk::TRUSTED_COMMAND_OPS,
        "trusted validation selects the trusted table"
    );

    policy.contract = gpu::ContractValidation.FULL;
    uptr full_address = (uptr)gpu::internal::vk::select_command_ops(policy);
    assert(
        gpu::internal::vk::select_command_ops(policy)
            == &gpu::internal::vk::FULL_COMMAND_OPS,
        "full validation selects the full table"
    );

    assert(trusted_address != full_address);
    policy.contract = bitcast(99, gpu::ContractValidation);
    assert(
        gpu::internal::vk::select_command_ops(policy)
            == &gpu::internal::vk::FULL_COMMAND_OPS,
        "an unknown policy falls back to full command recording"
    );
    assert(
        gpu::internal::vk::contract_uses_full_validation(policy),
        "an unknown policy preserves full validation and reference storage"
    );
    assert(
        gpu::internal::vk::TRUSTED_COMMAND_OPS.begin_label
            == gpu::internal::vk::FULL_COMMAND_OPS.begin_label,
        "both command policies share the label-begin entry"
    );
    assert(
        gpu::internal::vk::TRUSTED_COMMAND_OPS.end_label
            == gpu::internal::vk::FULL_COMMAND_OPS.end_label,
        "both command policies share the label-end entry"
    );
}

fn void device_creation_rolls_back_each_checkpoint() @test {
    gpu::RuntimeDesc desc = {
        .application_name  = "gpu_vk_device_desc_rollback_test",
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter)!!.supported) continue;

        gpu::internal::vk::DeviceCreationCheckpoint[6] checkpoints = {
            gpu::internal::vk::DeviceCreationCheckpoint.LOGICAL_DEVICE,
            gpu::internal::vk::DeviceCreationCheckpoint.ALLOCATOR,
            gpu::internal::vk::DeviceCreationCheckpoint.COMMAND_STORAGE,
            gpu::internal::vk::DeviceCreationCheckpoint.DESCRIPTOR_HEAP,
            gpu::internal::vk::DeviceCreationCheckpoint.PIPELINE_SHARED,
            gpu::internal::vk::DeviceCreationCheckpoint.BEFORE_PUBLICATION,
        };
        foreach (checkpoint : checkpoints) {
            gpu::internal::vk::DeviceCreationProbe probe = { .fail_at = checkpoint };
            uint live_slots_before = live_device_slot_count();

            assert(
                run_injected_device_create(
                    runtime:     &runtime,
                    adapter:     &adapter,
                    device_desc: null,
                    probe:       &probe,
                ) == gpu::BACKEND_ERROR,
                "a checkpoint failure returns the backend fault"
            );
            assert(probe.reached == checkpoint, "creation reaches the requested checkpoint");
            assert(probe.rollback_count == 1, "creation rolls back exactly once");
            assert(
                live_device_slot_count() == live_slots_before,
                "failed creation publishes no device"
            );
            assert(
                gpu::get_adapter_info(&adapter)!!.name.len > 0,
                "failed creation leaves its adapter usable"
            );
        }

        gpu::Device device = gpu::create_device(&adapter)!!;
        gpu::destroy_device(&device)!!;
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::destroy_runtime(&runtime)!!;
}

fn void native_device_creation_failure_rolls_back() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_vk_native_device_failure_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter)!!.supported) continue;
        uint live_slots_before = live_device_slot_count();

        gpu::internal::vk::DeviceCreationProbe probe;
        gpu::internal::vk::DeviceCreationOps creation_ops = {
            .create = &reject_native_device_create,
        };
        assert(
            run_injected_device_create(
                runtime:      &runtime,
                adapter:      &adapter,
                device_desc:  null,
                probe:        &probe,
                creation_ops: &creation_ops,
            ) == gpu::UNSUPPORTED_BACKEND,
            "an injected native failure aborts device creation"
        );
        assert(
            probe.reached == gpu::internal::vk::DeviceCreationCheckpoint.NONE
                && probe.rollback_count == 1
                && live_device_slot_count() == live_slots_before,
            "native failure rolls back without publishing a device"
        );
        gpu::Device device = gpu::create_device(&adapter)!!;
        gpu::destroy_device(&device)!!;
        return;
    }
}

fn void device_loads_required_dynamic_color_dispatch() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_vk_required_dynamic_color_dispatch_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter)!!.supported) continue;
        gpu::Device device = gpu::create_device(&adapter)!!;
        gpu::internal::vk::VkDeviceState* state =
            gpu::internal::device_slots[device.index].data.state;
        assert(
            gpu::internal::vk::device_dispatch_supports_dynamic_color(
                &state.device_dispatch,
            ),
            "successful device creation loads the complete color dispatch"
        );
        assert(state.acceleration_structures.slots.ptr == null
                && state.shader_heap.descriptors
                    .acceleration_structure_cells.ptr == null
                && state.shader_heap.descriptors
                    .acceleration_structure_free.ptr == null
                && state.ray_query_operations == null,
            "baseline creation allocates no optional ray-query state");
        gpu::destroy_device(&device)!!;
        return;
    }
}

fn void opted_in_ray_query_state_is_transactional_and_prepublication() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_vk_ray_query_state_test",
        .acceleration_structure_heap_capacity = 7,
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceDesc device_desc = { .enable_ray_queries = true };

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter, &device_desc)!!.supported) {
            continue;
        }

        gpu::internal::vk::DeviceCreationProbe probe = {
            .fail_at = gpu::internal::vk::DeviceCreationCheckpoint
                .RAY_QUERY_STATE,
        };
        uint live_slots_before = live_device_slot_count();
        assert(run_injected_device_create(
                runtime:     &runtime,
                adapter:     &adapter,
                device_desc: &device_desc,
                probe:       &probe,
            ) == gpu::BACKEND_ERROR,
            "an optional-state checkpoint failure aborts creation");
        assert(probe.reached
                == gpu::internal::vk::DeviceCreationCheckpoint
                    .RAY_QUERY_STATE
                && probe.rollback_count == 1
                && live_device_slot_count() == live_slots_before,
            "optional-state failure unwinds without public registration");

        gpu::Device device = gpu::create_device(&adapter, &device_desc)!!;
        gpu::internal::vk::VkDeviceState* state =
            gpu::internal::device_slots[device.index].data.state;
        assert(state.acceleration_structures.slots.ptr != null
                && state.ray_query_operations
                    == &state.device_dispatch.ray_queries,
            "opt-in establishes the resource and operation tables");
        assert(state.shader_heap.descriptors
                    .acceleration_structure_cells.len == 7
                && state.shader_heap.descriptors
                    .acceleration_structure_free.len == 7,
            "opt-in establishes independently sized descriptor storage");
        gpu::destroy_device(&device)!!;
        return;
    }
    assert(false, "the controlled native target supports ray queries");
}

fn void publication_failure_rolls_back_native_state() @test {
    gpu::RuntimeDesc desc = {
        .application_name  = "gpu_vk_publication_rollback_test",
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer {
        if (runtime.is_valid()) (void)gpu::destroy_runtime(&runtime);
    }
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    bool exercised;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter)!!.supported) continue;
        exercised = true;

        gpu::Device[gpu::internal::MAX_DEVICES] blockers;
        defer destroy_registry_blockers(blockers[..]);
        foreach (&blocker : blockers) {
            gpu::internal::vk::VkDeviceState* blocker_state =
                mem::new(gpu::internal::vk::VkDeviceState);
            *blocker = gpu::internal::register_device(
                state:  blocker_state,
                caps:   {},
            )!!;
        }

        gpu::Runtime retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!!;
        bool retain_live = true;
        defer {
            if (retain_live) (void)gpu::internal::runtime_release_device(&retained_runtime);
        }
        gpu::internal::vk::DeviceCreationProbe probe;
        uint live_slots_before = live_device_slot_count();

        assert(
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                device_desc:      null,
                retained_runtime: &retained_runtime,
                probe:            &probe,
            )) == gpu::SLOT_TABLE_FULL,
            "registration reports a full device registry"
        );
        gpu::internal::runtime_release_device(&retained_runtime)!!;
        retain_live = false;
        assert(
            probe.reached == gpu::internal::vk::DeviceCreationCheckpoint.BEFORE_PUBLICATION,
            "registry exhaustion occurs after native initialization"
        );
        assert(probe.rollback_count == 1, "registry exhaustion rolls back once");
        assert(
            live_device_slot_count() == live_slots_before,
            "registry exhaustion publishes no additional device"
        );
        assert(
            @catch(gpu::create_device(&adapter)) == gpu::SLOT_TABLE_FULL,
            "public creation reports a full registry without retaining the runtime"
        );

        destroy_registry_blockers(blockers[..]);
        gpu::Device device = gpu::create_device(&adapter)!!;
        gpu::destroy_device(&device)!!;
        assert(
            gpu::get_adapter_info(&adapter)!!.name.len > 0,
            "registry recovery leaves the adapter usable"
        );
        gpu::destroy_runtime(&runtime)!!;
        break;
    }

    assert(exercised, "the controlled native target supports the device baseline");
}

fn void runtime_devices_destroy_independently() @test {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_vk_multi_device_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    bool exercised;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter)!!.supported) continue;

        gpu::Adapter second_adapter = adapter;
        bool distinct_adapters;
        for (uint j = 0; j < adapters.count; j++) {
            if (j == i) continue;
            gpu::Adapter candidate = adapters.get(j)!!;
            if (!gpu::supports_device_desc(&candidate)!!.supported) continue;
            second_adapter = candidate;
            distinct_adapters = true;
            break;
        }
        if (!distinct_adapters) {
            io::eprintfn("runtime_devices: distinct-adapter=N/A (one baseline-compatible adapter)");
        }

        gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_adapter(&adapter);
        gpu::internal::vk::VkRuntimeState* runtime_state =
            runtime_data.state;
        gpu::Device first = gpu::create_device(&adapter)!!;
        gpu::Device second = gpu::create_device(&second_adapter)!!;
        uptr first_physical = device_physical_adapter(&first);
        uptr second_physical = device_physical_adapter(&second);

        assert((ulong)first != (ulong)second, "same-runtime devices have distinct identities");
        assert(
            first_physical
                == (uptr)runtime_state.adapters[adapter.adapter_index].physical_device,
            "the first device owns the first requested physical adapter"
        );
        assert(
            second_physical
                == (uptr)runtime_state.adapters[second_adapter.adapter_index].physical_device,
            "the second device owns the second requested physical adapter"
        );
        if (distinct_adapters) {
            assert(
                first_physical != second_physical,
                "distinct adapter tokens create devices on distinct physical adapters"
            );
        }

        gpu::AllocationDesc allocation_desc = {
            .size         = 256,
            .alignment    = 16,
            .memory_class = gpu::MemoryClass.GPU_PRIVATE,
            .access       = { .graphics, .compute, .transfer },
        };
        gpu::GpuAllocation first_allocation = gpu::allocate_memory(&first, &allocation_desc)!!;
        gpu::GpuAllocation second_allocation = gpu::allocate_memory(&second, &allocation_desc)!!;
        gpu::free_allocation(&first, &first_allocation)!!;
        gpu::free_allocation(&second, &second_allocation)!!;

        gpu::destroy_device(&first)!!;
        assert(
            gpu::get_device_caps(&second)!!.texture_heap_capacity > 0,
            "destroying one device preserves its peer"
        );
        assert(
            device_physical_adapter(&second) == second_physical,
            "destroying one device preserves its peer's physical adapter"
        );

        gpu::GpuAllocation allocation = gpu::allocate_memory(&second, &allocation_desc)!!;
        gpu::free_allocation(&second, &allocation)!!;
        gpu::destroy_device(&second)!!;
        exercised = true;
        break;
    }

    gpu::destroy_runtime(&runtime)!!;
    assert(exercised, "the controlled native target supports the device baseline");
}
