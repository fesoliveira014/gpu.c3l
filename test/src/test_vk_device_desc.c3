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

struct IndexingFeatureChainCapture {
    uint calls;
    bool found_features12;
    bool all_required_bits_enabled;
}

fn void capture_baseline_support_diagnostic(
    gpu::DebugMessage* message,
    void* user_data,
) {
    BaselineSupportDiagnosticCapture* capture =
        (BaselineSupportDiagnosticCapture*)user_data;
    capture.calls++;
    capture.matched = message.severity == gpu::DebugMessageSeverity.ERROR
        && message.category.public_contract
        && message.has_fault
        && message.public_fault == gpu::UNSUPPORTED_FEATURE
        && message.operation.str_view() == "create_device"
        && message.rejected_field.str_view() == capture.expected_field.str_view()
        && message.invariant.str_view() == capture.expected_invariant.str_view();
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

fn vk::Result capture_indexing_feature_chain(
    void* user_data,
    vk::PhysicalDevice physical_device,
    vk::DeviceCreateInfo* create_info,
    vk::Device* device,
) {
    (void)physical_device;
    (void)device;
    IndexingFeatureChainCapture* capture =
        (IndexingFeatureChainCapture*)user_data;
    capture.calls++;

    vk::BaseInStructure* feature =
        (vk::BaseInStructure*)create_info.p_next;
    while (feature != null) {
        if (feature.s_type
            == vk::StructureType.PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
            vk::PhysicalDeviceVulkan12Features* features12 =
                (vk::PhysicalDeviceVulkan12Features*)feature;
            capture.found_features12 = true;
            capture.all_required_bits_enabled =
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
            break;
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
    gpu::internal::vk::VkDeviceConfig desc = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
    };
    gpu::internal::vk::VkShaderHeapState indexing =
        gpu::internal::vk::resolve_shader_heap_state(&desc)!!;
    assert(indexing.texture_capacity == 7, "the requested texture capacity is preserved");
    assert(indexing.sampler_capacity == 9, "the requested sampler capacity is preserved");
}

fn void shader_heap_support_reports_exact_failure() @test {
    gpu::internal::vk::VkDeviceConfig desc = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
    };
    gpu::internal::vk::VkShaderHeapState heap =
        gpu::internal::vk::resolve_shader_heap_state(&desc)!!;
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

fn void runtime_copies_device_creation_defaults() @test {
    char[4] cache_data = { 1, 2, 3, 4 };
    gpu::RuntimeDesc desc = {
        .enable_debug_names    = true,
        .texture_heap_capacity = 17,
        .sampler_heap_capacity = 19,
        .texture_capacity      = 23,
        .pipeline_capacity     = 29,
        .pipeline_cache_data   = cache_data[..],
        .application_name      = "gpu_vk_runtime_defaults_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::internal::vk::VkRuntimeState* state =
        gpu::internal::resolve_runtime(&runtime).state;

    assert(state != null, "the runtime publishes its Vulkan state");
    assert(state.debug_names, "the runtime retains the debug-name default");
    assert(
        state.texture_heap_capacity == 17,
        "the runtime retains the texture heap capacity"
    );
    assert(
        state.sampler_heap_capacity == 19,
        "the runtime retains the sampler heap capacity"
    );
    assert(
        state.texture_capacity == 23,
        "the runtime retains the texture identity capacity"
    );
    assert(
        state.pipeline_capacity == 29,
        "the runtime retains the pipeline capacity"
    );
    assert(
        state.pipeline_cache_data.ptr != &cache_data[0],
        "the runtime owns an independent cache copy"
    );
    assert(
        state.pipeline_cache_data.len == cache_data.len,
        "the runtime retains the full cache blob"
    );
    assert(state.pipeline_cache_data[0] == 1, "the runtime copies cache bytes");

    cache_data[0] = 9;
    assert(
        state.pipeline_cache_data[0] == 1,
        "caller mutation cannot change the runtime cache copy"
    );

    gpu::destroy_runtime(&runtime)!!;
}

fn void adapter_baseline_support_inherits_runtime_device_config() @test {
    gpu::RuntimeDesc desc = {
        .texture_heap_capacity = gpu::MAX_SHADER_HEAP_CAPACITY,
        .sampler_heap_capacity = gpu::MAX_SHADER_HEAP_CAPACITY,
        .application_name      = "gpu_vk_adapter_support_config_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    assert(adapters.count > 0, "the controlled target exposes an adapter");
    gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_runtime(&runtime);
    gpu::internal::vk::VkRuntimeState* runtime_state =
        runtime_data.state;
    gpu::internal::vk::VkDeviceConfig config = gpu::internal::vk::runtime_device_config(runtime_state);

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        gpu::DeviceSupport support =
            gpu::supports_device_desc(&adapter)!!;
        gpu::internal::vk::VkAdapterState* adapter_state = &runtime_state.adapters[i];
        bool expected = gpu::internal::vk::device_baseline_supported(
            adapter_state.physical_device,
            &config,
            &adapter_state.shader_heap,
        );
        assert(
            adapter_state.device_baseline_supported == expected,
            "adapter baseline support uses the runtime configuration"
        );
        assert(support.supported == expected);

        gpu::internal::vk::VkDeviceConfig invalid_config = config;
        invalid_config.texture_heap_capacity = gpu::MAX_SHADER_HEAP_CAPACITY + 1;
        gpu::internal::vk::VkAdapterState invalid_adapter;
        gpu::internal::vk::initialize_adapter(
            &invalid_adapter,
            adapter_state.physical_device,
            &invalid_config,
        );
        assert(
            !invalid_adapter.device_baseline_supported,
            "each adapter consumes the supplied configuration"
        );

        if (!expected) {
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
        assert(capture.matched,
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
        gpu::internal::vk::DeviceCreationProbe probe = {
            .native_device_create_hook      = &count_native_device_create,
            .native_device_create_user_data = &create_capture,
        };
        gpu::Runtime retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!!;
        fault rejected =
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                device_desc:      null,
                retained_runtime: &retained_runtime,
                probe:            &probe,
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

fn void device_creation_enables_all_required_indexing_features() @test {
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
        IndexingFeatureChainCapture capture;
        gpu::internal::vk::DeviceCreationProbe probe = {
            .native_device_create_hook      = &capture_indexing_feature_chain,
            .native_device_create_user_data = &capture,
        };
        gpu::Runtime retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!!;
        fault rejected =
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                device_desc:      null,
                retained_runtime: &retained_runtime,
                probe:            &probe,
            ));
        gpu::internal::runtime_release_device(&retained_runtime)!!;

        assert(rejected == gpu::UNSUPPORTED_BACKEND,
            "the native hook aborts creation after inspecting the feature chain");
        assert(capture.calls == 1 && capture.found_features12,
            "device creation chains Vulkan 1.2 features into vkCreateDevice");
        assert(capture.all_required_bits_enabled,
            "device creation enables all seven required indexing feature bits");
        assert(probe.rollback_count == 1,
            "the injected native creation failure rolls back exactly once");
        return;
    }
    assert(false, "the pinned Vulkan adapter supports the device baseline");
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
            .counts   = { .graphics = 1 },
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

fn void device_creation_checkpoint_ordinals_match_exact_schema() @test {
    assert(gpu::internal::vk::DeviceCreationCheckpoint.NONE.ordinal == 0);
    assert(gpu::internal::vk::DeviceCreationCheckpoint.LOGICAL_DEVICE.ordinal == 1);
    assert(gpu::internal::vk::DeviceCreationCheckpoint.ALLOCATOR.ordinal == 2);
    assert(gpu::internal::vk::DeviceCreationCheckpoint.COMMAND_STORAGE.ordinal == 3);
    assert(gpu::internal::vk::DeviceCreationCheckpoint.DESCRIPTOR_HEAP.ordinal == 4);
    assert(gpu::internal::vk::DeviceCreationCheckpoint.PIPELINE_SHARED.ordinal == 5);
    assert(gpu::internal::vk::DeviceCreationCheckpoint.BEFORE_PUBLICATION.ordinal == 6);
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

fn void failed_native_device_creation_rolls_back_each_checkpoint() @test {
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

        for (
            uint checkpoint_ordinal = gpu::internal::vk::DeviceCreationCheckpoint.LOGICAL_DEVICE.ordinal;
            checkpoint_ordinal <= gpu::internal::vk::DeviceCreationCheckpoint.BEFORE_PUBLICATION.ordinal;
            checkpoint_ordinal++
        ) {
            gpu::internal::vk::DeviceCreationCheckpoint checkpoint =
                (gpu::internal::vk::DeviceCreationCheckpoint)checkpoint_ordinal;
            gpu::Runtime retained_runtime = runtime;
            gpu::internal::runtime_retain_device(&retained_runtime)!!;
            gpu::internal::vk::DeviceCreationProbe probe = { .fail_at = checkpoint };
            uint live_slots_before = live_device_slot_count();

            assert(
                @catch(gpu::internal::vk::create_vk_device_with_probe(
                    adapter:          &adapter,
                    device_desc:      null,
                    retained_runtime: &retained_runtime,
                    probe:            &probe,
                )) == gpu::BACKEND_ERROR,
                "a checkpoint failure returns the backend fault"
            );
            gpu::internal::runtime_release_device(&retained_runtime)!!;
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

fn void required_dynamic_color_features_are_enabled_transactionally() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_vk_required_dynamic_color_features_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter)!!.supported) continue;
        uint live_slots_before = live_device_slot_count();

        gpu::Runtime retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!!;
        gpu::internal::vk::DeviceCreationProbe failure_probe = {
            .fail_native_device_creation = true,
        };
        assert(
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                device_desc:      null,
                retained_runtime: &retained_runtime,
                probe:            &failure_probe,
            )) == gpu::BACKEND_ERROR,
            "an injected native failure aborts device creation"
        );
        gpu::internal::runtime_release_device(&retained_runtime)!!;
        assert(
            failure_probe.dynamic_color_feature_chained
                && failure_probe.dynamic_color_blend_enable == vk::TRUE
                && failure_probe.dynamic_color_blend_equation == vk::TRUE
                && failure_probe.dynamic_color_write_mask == vk::TRUE,
            "device creation chains and enables all three color feature bits"
        );
        assert(
            failure_probe.rollback_count == 1
                && live_device_slot_count() == live_slots_before,
            "native failure rolls back once and publishes no device"
        );
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
        gpu::destroy_device(&device)!!;
        return;
    }
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
