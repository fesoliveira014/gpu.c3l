module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::atomic;
import std::core::mem;
import std::io;

struct ResourceAgnosticCreateObservation {
    bool selected;
    uint calls;
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

fn fault run_injected_resource_agnostic_texture_sync_create(
    gpu::Runtime* runtime,
    gpu::Adapter* adapter,
    gpu::DeviceRequest* request,
    gpu::internal::vk::DeviceCreationProbe* probe,
) @private {
    gpu::Runtime retained_runtime = *runtime;
    gpu::internal::runtime_retain_device(&retained_runtime)!!;
    fault creation_fault = @catch(
        gpu::internal::vk::create_vk_device_with_probe(
            adapter:          adapter,
            request:          request,
            retained_runtime: &retained_runtime,
            probe:            probe,
        )
    );
    gpu::internal::runtime_release_device(&retained_runtime)!!;
    return creation_fault;
}

fn vk::Result observe_resource_agnostic_texture_sync_create(
    void* user_data,
    vk::PhysicalDevice physical_device,
    vk::DeviceCreateInfo* requested_info,
    vk::Device* device,
) @private {
    (void)physical_device;
    (void)device;
    ResourceAgnosticCreateObservation* observation =
        (ResourceAgnosticCreateObservation*)user_data;
    observation.calls++;

    bool extension_enabled;
    for (uint i = 0; i < requested_info.enabled_extension_count; i++) {
        ZString extension =
            (ZString)requested_info.pp_enabled_extension_names[i];
        if (extension.str_view()
            == gpu::internal::vk::RESOURCE_AGNOSTIC_TEXTURE_SYNC_EXTENSION
                .str_view()) {
            extension_enabled = true;
            break;
        }
    }
    vk::PhysicalDeviceFeatures2* features =
        (vk::PhysicalDeviceFeatures2*)requested_info.p_next;
    if (!observation.selected) {
        vk::BaseOutStructure* first =
            (vk::BaseOutStructure*)features.p_next;
        assert(
            !extension_enabled
                && first != null
                && first.s_type
                    != vk::StructureType
                        .PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR,
            "unrequested creation omits the unified-layout extension and feature"
        );
        return vk::Result.ERROR_INITIALIZATION_FAILED;
    }

    vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR* unified =
        (vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR*)features.p_next;
    assert(
        extension_enabled
            && unified.s_type
                == vk::StructureType
                    .PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR
            && unified.unified_image_layouts == vk::TRUE
            && unified.unified_image_layouts_video == vk::FALSE,
        "requested creation enables one exact unified feature and no video"
    );
    return vk::Result.ERROR_INITIALIZATION_FAILED;
}

fn void resource_agnostic_texture_sync_support_requires_extension_and_feature() @test {
    assert(
        gpu::internal::vk::RESOURCE_AGNOSTIC_TEXTURE_SYNC_EXTENSION.str_view()
            == "VK_KHR_unified_image_layouts",
        "the backend uses the exact unified-image-layout extension name"
    );
    gpu::internal::vk::ResourceAgnosticTextureSyncSupport support;
    assert(
        !gpu::internal::vk::resource_agnostic_texture_sync_supported(support),
        "missing extension and feature are unsupported"
    );

    support.extension = true;
    assert(
        !gpu::internal::vk::resource_agnostic_texture_sync_supported(support),
        "an extension without unifiedImageLayouts is unsupported"
    );

    support = {
        .extension             = true,
        .unified_image_layouts = true,
    };
    assert(
        gpu::internal::vk::resource_agnostic_texture_sync_supported(support),
        "the extension and unifiedImageLayouts together are supported"
    );
    assert(
        !gpu::internal::vk::select_resource_agnostic_texture_sync(
            false,
            support,
        )!!,
        "capable unrequested selection stays disabled"
    );
    assert(
        gpu::internal::vk::select_resource_agnostic_texture_sync(
            true,
            support,
        )!!,
        "a capable explicit request is selected"
    );

    support.extension = false;
    assert(
        @catch(gpu::internal::vk::select_resource_agnostic_texture_sync(
            true,
            support,
        )) == gpu::UNSUPPORTED_FEATURE,
        "a missing extension rejects the explicit request"
    );
    support.extension = true;
    support.unified_image_layouts = false;
    assert(
        @catch(gpu::internal::vk::select_resource_agnostic_texture_sync(
            true,
            support,
        )) == gpu::UNSUPPORTED_FEATURE,
        "a missing feature rejects the explicit request"
    );
}

fn void resource_agnostic_texture_sync_support_hook_uses_cached_availability() @test {
    gpu::internal::vk::VkAdapterState[1] adapters;
    gpu::internal::vk::VkRuntimeState state = {
        .adapters = adapters[..],
    };
    adapters[0].resource_agnostic_texture_sync = {
        .extension             = true,
        .unified_image_layouts = true,
    };
    assert(
        gpu::internal::vk::vk_supports_resource_agnostic_texture_sync(
            &state,
            0,
        )!!,
        "the backend support hook reports cached complete support"
    );

    adapters[0].resource_agnostic_texture_sync.unified_image_layouts = false;
    assert(
        !gpu::internal::vk::vk_supports_resource_agnostic_texture_sync(
            &state,
            0,
        )!!,
        "the backend support hook reports cached feature absence"
    );
    assert(
        @catch(gpu::internal::vk::vk_supports_resource_agnostic_texture_sync(
            &state,
            1,
        )) == gpu::INVALID_HANDLE,
        "the backend support hook rejects an invalid adapter index"
    );
}

fn void resource_agnostic_texture_sync_device_creation_is_exact_and_transactional() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_vk_resource_agnostic_texture_sync_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceRequest classic_request = gpu::strict_device_request();
    gpu::DeviceRequest selected_request =
        gpu::request_resource_agnostic_texture_sync(classic_request)!!;
    bool exercised;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(
            &adapter,
            &classic_request,
        )!!.supported) continue;
        exercised = true;
        gpu::internal::RuntimeData* runtime_data =
            gpu::internal::resolve_adapter(&adapter);
        gpu::internal::vk::VkRuntimeState* runtime_state =
            runtime_data.state;
        gpu::internal::vk::VkAdapterState* adapter_state =
            &runtime_state.adapters[adapter.adapter_index];
        adapter_state.resource_agnostic_texture_sync = {
            .extension             = true,
            .unified_image_layouts = true,
        };
        assert(
            gpu::supports_device_request(
                &adapter,
                &selected_request,
            )!!.supported,
            "complete cached support satisfies the semantic request query"
        );

        gpu::internal::vk::DeviceCreationProbe missing_extension = {
            .resource_agnostic_texture_sync_override = true,
            .resource_agnostic_texture_sync_feature_available = true,
        };
        assert(
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                request:          &selected_request,
                retained_runtime: &runtime,
                probe:            &missing_extension,
            )) == gpu::UNSUPPORTED_FEATURE,
            "creation rechecks and rejects a missing extension"
        );
        assert(
            !missing_extension.resource_agnostic_texture_sync_selected
                && missing_extension.reached
                    == gpu::internal::vk::DeviceCreationCheckpoint.NONE
                && missing_extension.rollback_count == 0,
            "extension absence rejects before backend state mutation"
        );

        gpu::internal::vk::DeviceCreationProbe missing_feature = {
            .resource_agnostic_texture_sync_override = true,
            .resource_agnostic_texture_sync_extension_available = true,
        };
        assert(
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                request:          &selected_request,
                retained_runtime: &runtime,
                probe:            &missing_feature,
            )) == gpu::UNSUPPORTED_FEATURE,
            "creation rechecks and rejects a missing feature"
        );
        assert(
            !missing_feature.resource_agnostic_texture_sync_selected
                && missing_feature.reached
                    == gpu::internal::vk::DeviceCreationCheckpoint.NONE
                && missing_feature.rollback_count == 0,
            "feature absence rejects before backend state mutation"
        );

        uint live_slots_before = live_device_slot_count();
        ResourceAgnosticCreateObservation unrequested_observation;
        gpu::internal::vk::DeviceCreationProbe unrequested = {
            .resource_agnostic_texture_sync_override = true,
            .resource_agnostic_texture_sync_extension_available = true,
            .resource_agnostic_texture_sync_feature_available = true,
            .native_device_create_hook =
                &observe_resource_agnostic_texture_sync_create,
            .native_device_create_user_data = &unrequested_observation,
        };
        assert(
            run_injected_resource_agnostic_texture_sync_create(
                runtime: &runtime,
                adapter: &adapter,
                request: &classic_request,
                probe:   &unrequested,
            ) == gpu::UNSUPPORTED_BACKEND,
            "the injected native failure maps through device creation"
        );
        assert(
            !unrequested.resource_agnostic_texture_sync_selected
                && unrequested_observation.calls == 1,
            "capable unrequested creation omits the extension and feature"
        );
        assert(
            unrequested.rollback_count == 1
                && live_device_slot_count() == live_slots_before,
            "unrequested native failure rolls back without publication"
        );

        ResourceAgnosticCreateObservation requested_observation = {
            .selected = true,
        };
        gpu::internal::vk::DeviceCreationProbe requested = {
            .resource_agnostic_texture_sync_override = true,
            .resource_agnostic_texture_sync_extension_available = true,
            .resource_agnostic_texture_sync_feature_available = true,
            .native_device_create_hook =
                &observe_resource_agnostic_texture_sync_create,
            .native_device_create_user_data = &requested_observation,
        };
        assert(
            run_injected_resource_agnostic_texture_sync_create(
                runtime: &runtime,
                adapter: &adapter,
                request: &selected_request,
                probe:   &requested,
            ) == gpu::UNSUPPORTED_BACKEND,
            "requested support reaches the injected native create"
        );
        assert(
            requested.resource_agnostic_texture_sync_selected
                && requested_observation.calls == 1,
            "requested support reaches the exact selected native path"
        );
        assert(
            requested.rollback_count == 1
                && live_device_slot_count() == live_slots_before,
            "requested native failure rolls back without publication"
        );
        break;
    }

    assert(exercised, "the controlled native target has a compatible strict adapter");
}

fn void strict_disabled_state_owns_no_sampler_table() @test {
    gpu::internal::vk::VkDeviceState state;
    gpu::internal::vk::create_sampler_table(&state)!!;

    assert(state.samplers.slots.ptr == null,
        "a strict-disabled state allocates no sampler table");
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

fn void strict_heap_state_selects_private_implementation() @test {
    gpu::internal::vk::VkDeviceConfig desc = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
    };
    gpu::internal::vk::IndexingHeapLimits limits = permissive_indexing_heap_limits();

    gpu::internal::vk::VkStrictHeapState disabled =
        gpu::internal::vk::resolve_strict_heap_state(false, &desc)!!;
    gpu::internal::vk::select_strict_heap_implementation(
        state:              &disabled,
        indexing_supported: false,
        buffer_supported:   false,
        indexing_limits:    limits,
    )!!;
    assert(!disabled.enabled, "an unrequested strict heap remains disabled");
    assert(
        disabled.implementation == gpu::internal::vk::VkHeapImplementation.NONE,
        "a disabled strict heap selects no private implementation"
    );
    assert(disabled.texture_capacity == 0, "a disabled strict heap reserves no texture capacity");
    assert(disabled.sampler_capacity == 0, "a disabled strict heap reserves no sampler capacity");
    assert(disabled.descriptors.texture_cells.ptr == null, "a disabled strict heap owns no descriptor cells");

    gpu::internal::vk::VkStrictHeapState indexing =
        gpu::internal::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::internal::vk::select_strict_heap_implementation(
        state:              &indexing,
        indexing_supported: true,
        buffer_supported:   true,
        indexing_limits:    limits,
    )!!;
    assert(indexing.enabled, "a requested strict heap is enabled");
    assert(
        indexing.implementation == gpu::internal::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        "indexing is preferred when it satisfies the semantic capacities"
    );
    assert(indexing.texture_capacity == 7, "the requested texture capacity is preserved");
    assert(indexing.sampler_capacity == 9, "the requested sampler capacity is preserved");
    assert(gpu::internal::vk::strict_heap_uses_descriptor_indexing(&indexing), "the indexing path requests indexing features");
    assert(
        !gpu::internal::vk::strict_heap_uses_descriptor_buffer(&indexing),
        "the indexing path requests no descriptor-buffer dispatch"
    );

    gpu::internal::vk::IndexingHeapLimits constrained = limits;
    constrained.max_sampled_images = 6;
    gpu::internal::vk::VkStrictHeapState fallback =
        gpu::internal::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::internal::vk::select_strict_heap_implementation(
        state:              &fallback,
        indexing_supported: true,
        buffer_supported:   true,
        indexing_limits:    constrained,
    )!!;
    assert(
        fallback.implementation == gpu::internal::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        "descriptor buffers are selected when indexing cannot provide the requested capacity"
    );

    gpu::internal::vk::VkStrictHeapState buffer_only =
        gpu::internal::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::internal::vk::select_strict_heap_implementation(
        state:              &buffer_only,
        indexing_supported: false,
        buffer_supported:   true,
        indexing_limits:    limits,
    )!!;
    assert(
        buffer_only.implementation == gpu::internal::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        "descriptor buffers satisfy strict heaps when indexing is unavailable"
    );

    gpu::internal::vk::VkStrictHeapState unavailable =
        gpu::internal::vk::resolve_strict_heap_state(true, &desc)!!;
    fault selection_fault = @catch(gpu::internal::vk::select_strict_heap_implementation(
        state:              &unavailable,
        indexing_supported: false,
        buffer_supported:   false,
        indexing_limits:    limits,
    ));
    assert(
        selection_fault == gpu::UNSUPPORTED_FEATURE,
        "strict heap selection faults when no private implementation is available"
    );
    assert(
        unavailable.implementation == gpu::internal::vk::VkHeapImplementation.NONE,
        "failed selection leaves the private implementation unpublished"
    );
}

fn void descriptor_buffer_creation_override_is_private_and_exact() @test {
    gpu::internal::vk::VkDeviceConfig desc = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
    };
    gpu::internal::vk::IndexingHeapLimits limits = permissive_indexing_heap_limits();

    gpu::internal::vk::VkStrictHeapState automatic =
        gpu::internal::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::internal::vk::select_strict_heap_implementation_for_creation(
        state:              &automatic,
        indexing_supported: true,
        buffer_supported:   true,
        indexing_limits:    limits,
        probe:              null,
    )!!;
    assert(
        automatic.implementation == gpu::internal::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        "ordinary creation keeps automatic indexing preference"
    );

    gpu::internal::vk::DeviceCreationProbe probe = {
        .descriptor_buffer_override = true,
    };
    gpu::internal::vk::VkStrictHeapState overridden =
        gpu::internal::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::internal::vk::select_strict_heap_implementation_for_creation(
        state:              &overridden,
        indexing_supported: true,
        buffer_supported:   true,
        indexing_limits:    limits,
        probe:              &probe,
    )!!;
    assert(
        overridden.implementation == gpu::internal::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        "the private creation probe can exercise descriptor-buffer runtime paths"
    );
    assert(
        probe.descriptor_buffer_capability_checked
            && probe.descriptor_buffer_capability_available
            && probe.descriptor_buffer_selected,
        "the probe records advertised capability and completed selection"
    );

    gpu::internal::vk::VkStrictHeapState unsupported =
        gpu::internal::vk::resolve_strict_heap_state(true, &desc)!!;
    assert(
        @catch(gpu::internal::vk::select_strict_heap_implementation_for_creation(
            state:              &unsupported,
            indexing_supported: true,
            buffer_supported:   false,
            indexing_limits:    limits,
            probe:              &probe,
        )) == gpu::UNSUPPORTED_FEATURE,
        "the private override never bypasses descriptor-buffer feature support"
    );
    assert(
        probe.descriptor_buffer_capability_checked
            && !probe.descriptor_buffer_capability_available
            && !probe.descriptor_buffer_selected,
        "the probe distinguishes proven capability absence from later failure"
    );
}

fn void descriptor_buffer_query_is_needed_only_for_fallback_or_override() @test {
    gpu::internal::vk::VkDeviceConfig desc = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
    };
    gpu::internal::vk::VkStrictHeapState state =
        gpu::internal::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::internal::vk::IndexingHeapLimits limits = permissive_indexing_heap_limits();
    assert(
        !gpu::internal::vk::strict_heap_needs_descriptor_buffer_query(
            state:              &state,
            indexing_supported: true,
            indexing_limits:    limits,
            probe:              null,
        ),
        "a usable indexing heap does not require a descriptor-buffer query"
    );

    gpu::internal::vk::IndexingHeapLimits constrained = limits;
    constrained.max_sampled_images = 6;
    assert(
        gpu::internal::vk::strict_heap_needs_descriptor_buffer_query(
            state:              &state,
            indexing_supported: true,
            indexing_limits:    constrained,
            probe:              null,
        ),
        "an indexing-capacity shortfall requires descriptor-buffer fallback evidence"
    );
    assert(
        gpu::internal::vk::strict_heap_needs_descriptor_buffer_query(
            state:              &state,
            indexing_supported: false,
            indexing_limits:    limits,
            probe:              null,
        ),
        "missing indexing features require descriptor-buffer fallback evidence"
    );

    gpu::internal::vk::DeviceCreationProbe probe = {
        .descriptor_buffer_override = true,
    };
    assert(
        gpu::internal::vk::strict_heap_needs_descriptor_buffer_query(
            state:              &state,
            indexing_supported: true,
            indexing_limits:    limits,
            probe:              &probe,
        ),
        "the private override always requires descriptor-buffer capability evidence"
    );
}
fn bool? descriptor_buffer_extension_absent(
    vk::PhysicalDevice physical_device,
    ZString extension,
) {
    return false;
}

fn bool? descriptor_buffer_extension_query_host_oom(
    vk::PhysicalDevice physical_device,
    ZString extension,
) {
    return vk::ERROR_OUT_OF_HOST_MEMORY~;
}

fn bool? descriptor_buffer_extension_query_device_oom(
    vk::PhysicalDevice physical_device,
    ZString extension,
) {
    return vk::ERROR_OUT_OF_DEVICE_MEMORY~;
}

fn bool? descriptor_buffer_extension_query_layer_missing(
    vk::PhysicalDevice physical_device,
    ZString extension,
) {
    return vk::ERROR_LAYER_NOT_PRESENT~;
}

fn void descriptor_buffer_capability_query_distinguishes_absence_and_failure() @test {
    vk::PhysicalDevice physical_device = (vk::PhysicalDevice)(uptr)0xA0;
    assert(
        !gpu::internal::vk::query_descriptor_buffer_support_with_query(
            physical_device,
            &descriptor_buffer_extension_absent,
        )!!,
        "confirmed extension absence is unsupported capability"
    );
    assert(
        @catch(gpu::internal::vk::query_descriptor_buffer_support_with_query(
            physical_device,
            &descriptor_buffer_extension_query_host_oom,
        )) == gpu::OUT_OF_HOST_MEMORY,
        "extension-query host OOM maps into the public fault taxonomy"
    );
    assert(
        @catch(gpu::internal::vk::query_descriptor_buffer_support_with_query(
            physical_device,
            &descriptor_buffer_extension_query_device_oom,
        )) == gpu::OUT_OF_DEVICE_MEMORY,
        "extension-query device OOM maps into the public fault taxonomy"
    );
    assert(
        @catch(gpu::internal::vk::query_descriptor_buffer_support_with_query(
            physical_device,
            &descriptor_buffer_extension_query_layer_missing,
        )) == gpu::UNSUPPORTED_FEATURE,
        "a missing extension layer maps through the public unsupported-feature taxonomy"
    );
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

fn void adapter_support_inherits_runtime_device_config() @test {
    gpu::RuntimeDesc desc = {
        .texture_heap_capacity = gpu::MAX_SHADER_HEAP_CAPACITY,
        .sampler_heap_capacity = gpu::MAX_SHADER_HEAP_CAPACITY,
        .application_name      = "gpu_vk_adapter_support_config_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    assert(adapters.count > 0, "the controlled target exposes an adapter");
    gpu::DeviceRequest request = gpu::strict_device_request();
    gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_runtime(&runtime);
    gpu::internal::vk::VkRuntimeState* runtime_state =
        runtime_data.state;
    gpu::internal::vk::VkDeviceConfig config = gpu::internal::vk::runtime_device_config(runtime_state);

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        gpu::AdapterInfo info = gpu::get_adapter_info(&adapter)!!;
        gpu::DeviceRequestSupport support =
            gpu::supports_device_request(&adapter, &request)!!;
        gpu::internal::vk::VkAdapterState* adapter_state = &runtime_state.adapters[i];
        bool expected = gpu::internal::vk::strict_adapter_supported(
            adapter_state.physical_device,
            &config,
        );
        assert(info.strict_supported == expected, "adapter info uses the runtime configuration");
        assert(support.supported == expected, "support and adapter info agree");

        gpu::internal::vk::VkDeviceConfig invalid_config = config;
        invalid_config.texture_heap_capacity = gpu::MAX_SHADER_HEAP_CAPACITY + 1;
        gpu::internal::vk::VkAdapterState invalid_adapter;
        gpu::internal::vk::initialize_adapter(
            &invalid_adapter,
            adapter_state.physical_device,
            &invalid_config,
        );
        assert(!invalid_adapter.info.strict_supported, "each adapter consumes the supplied configuration");

        if (!expected) {
            assert(
                @catch(gpu::create_device(&adapter, &request)) == gpu::UNSUPPORTED_FEATURE,
                "creation rejects a configuration reported as unsupported"
            );
            continue;
        }
        gpu::Device device = gpu::create_device(&adapter, &request)!!;
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

fn void adapter_request_creation_reuses_runtime_discovery() @test {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_vk_device_request_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    if (adapters.count == 0) {
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::DeviceRequest request = gpu::strict_device_request();
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        gpu::DeviceRequestSupport support =
            gpu::supports_device_request(&adapter, &request)!!;
        if (!support.supported) continue;

        gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_adapter(&adapter);
        gpu::internal::vk::VkRuntimeState* runtime_state =
            runtime_data.state;
        gpu::Device device = gpu::create_device(&adapter, &request)!!;
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
        assert(device_state.strict_heap.enabled, "the strict request owns heap state");
        assert(caps.strict_enabled, "strict state is enabled");
        assert(
            !caps.dynamic_color_state
                && !gpu::internal::vk::device_dispatch_supports_dynamic_color(
                    &device_state.device_dispatch,
                ),
            "an unrequested color capability publishes no cap or dispatch"
        );
        assert(
            device_state.strict_heap.descriptors.texture_cells.ptr != null,
            "strict state owns texture descriptor cells"
        );
        assert(
            device_state.samplers.slots.ptr != null,
            "the device owns an independent sampler identity table"
        );
        assert(
            device_state.strict_heap.descriptors.sampler_next_new == 0,
            "the strict sampler heap begins unpublished"
        );
        assert(
            device_state.strict_heap.descriptors.set_layout != null,
            "strict state owns its descriptor layout"
        );
        assert(
            caps.texture_heap_capacity == device_state.strict_heap.texture_capacity,
            "published texture capacity matches strict state"
        );
        assert(
            caps.sampler_heap_capacity == device_state.strict_heap.sampler_capacity,
            "published sampler capacity matches strict state"
        );
        if (gpu::internal::vk::strict_heap_uses_descriptor_buffer(&device_state.strict_heap)) {
            assert(
                gpu::internal::vk::device_dispatch_supports_descriptor_buffer(&device_state.device_dispatch),
                "the selected descriptor-buffer path loads its dispatch"
            );
            assert(
                device_state.strict_heap.descriptors.descriptor_buffer != null,
                "the selected descriptor-buffer path owns its buffer"
            );
        } else {
            assert(
                !gpu::internal::vk::device_dispatch_supports_descriptor_buffer(&device_state.device_dispatch),
                "the indexing path loads no descriptor-buffer dispatch"
            );
            assert(
                device_state.strict_heap.descriptors.descriptor_buffer == null,
                "the indexing path owns no descriptor buffer"
            );
        }
        assert(!device_state.presentation.enabled, "headless requests create no presentation state");
        assert(device_state.presentation.queue == null, "headless requests retrieve no presentation queue");
        assert(!device_state.presentation.queue_mutex.is_initialized(), "headless requests initialize no presentation mutex");
        assert(device_state.presentation.swapchains.slots.ptr == null, "headless requests allocate no swapchain table");
        assert(
            !gpu::internal::vk::instance_dispatch_supports_presentation(&device_state.instance_dispatch),
            "headless requests retain no surface procedures"
        );
        assert(
            !gpu::internal::vk::device_dispatch_supports_presentation(&device_state.device_dispatch),
            "headless requests load no presentation procedures"
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
    gpu::DeviceRequestSupport unsupported =
        gpu::supports_device_request(&unsupported_adapter, &request)!!;
    assert(!unsupported.supported, "no adapter silently falls back from strict semantics");
    assert(
        @catch(gpu::create_device(&unsupported_adapter, &request)) == gpu::UNSUPPORTED_FEATURE,
        "unsupported native creation publishes no device"
    );
    gpu::destroy_runtime(&runtime)!!;
}

fn void malformed_native_request_preserves_runtime_state() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_vk_device_request_failure_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    if (adapters.count == 0) {
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::Adapter adapter = adapters.get(0)!!;
    gpu::DeviceRequest empty = gpu::internal::empty_device_request();
    assert(
        @catch(gpu::create_device(&adapter, &empty)) == gpu::INVALID_ARGUMENT,
        "an empty request is rejected before native creation"
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
    gpu::DeviceRequest request = gpu::request_presentation(
        gpu::strict_device_request(),
        &surface,
    )!!;
    gpu::internal::vk::DeviceCreationProbe probe;

    assert(
        @catch(gpu::internal::vk::create_vk_device_with_probe(
            adapter:          &adapter,
            request:          &request,
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
        .application_name  = "gpu_vk_device_request_rollback_test",
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceRequest request = gpu::strict_device_request();

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) continue;

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
                    request:          &request,
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

        gpu::Device device = gpu::create_device(&adapter, &request)!!;
        gpu::destroy_device(&device)!!;
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::destroy_runtime(&runtime)!!;
}

fn void dynamic_color_activation_rejects_missing_support_and_rolls_back() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_vk_dynamic_color_activation_rollback_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceRequest classic = gpu::strict_device_request();
    gpu::DeviceRequest requested =
        gpu::request_dynamic_color_state(classic)!!;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &classic)!!.supported) continue;
        gpu::internal::RuntimeData* runtime_data =
            gpu::internal::resolve_adapter(&adapter);
        gpu::internal::vk::VkRuntimeState* runtime_state =
            runtime_data.state;
        gpu::internal::vk::VkAdapterState* adapter_state =
            &runtime_state.adapters[adapter.adapter_index];
        bool original_support = adapter_state.dynamic_color_state_supported;
        defer adapter_state.dynamic_color_state_supported = original_support;
        uint live_slots_before = live_device_slot_count();

        adapter_state.dynamic_color_state_supported = false;
        gpu::Runtime retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!!;
        gpu::internal::vk::DeviceCreationProbe missing_probe;
        assert(
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                request:          &requested,
                retained_runtime: &retained_runtime,
                probe:            &missing_probe,
            )) == gpu::UNSUPPORTED_FEATURE,
            "a missing color feature group rejects the request"
        );
        gpu::internal::runtime_release_device(&retained_runtime)!!;
        assert(
            missing_probe.reached
                    == gpu::internal::vk::DeviceCreationCheckpoint.NONE
                && missing_probe.rollback_count == 0
                && live_device_slot_count() == live_slots_before,
            "missing support rejects before native mutation or publication"
        );

        adapter_state.dynamic_color_state_supported = true;
        retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!!;
        gpu::internal::vk::DeviceCreationProbe failure_probe = {
            .fail_native_device_creation = true,
        };
        assert(
            @catch(gpu::internal::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                request:          &requested,
                retained_runtime: &retained_runtime,
                probe:            &failure_probe,
            )) == gpu::BACKEND_ERROR,
            "an injected native failure aborts requested activation"
        );
        gpu::internal::runtime_release_device(&retained_runtime)!!;
        assert(
            failure_probe.dynamic_color_feature_chained
                && failure_probe.dynamic_color_blend_enable == vk::TRUE
                && failure_probe.dynamic_color_blend_equation == vk::TRUE
                && failure_probe.dynamic_color_write_mask == vk::TRUE,
            "requested activation chains all three feature bits together"
        );
        assert(
            failure_probe.rollback_count == 1
                && live_device_slot_count() == live_slots_before,
            "native failure rolls back once and publishes no capability"
        );
        return;
    }
}

fn void supported_dynamic_color_request_publishes_complete_activation() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_vk_dynamic_color_activation_success_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceRequest request =
        gpu::request_dynamic_color_state(gpu::strict_device_request())!!;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) continue;
        gpu::Device device = gpu::create_device(&adapter, &request)!!;
        gpu::internal::vk::VkDeviceState* state =
            gpu::internal::device_slots[device.index].data.state;
        assert(
            gpu::get_device_caps(&device)!!.dynamic_color_state
                && gpu::internal::vk::device_dispatch_supports_dynamic_color(
                    &state.device_dispatch,
                ),
            "successful creation publishes the cap with complete dispatch"
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
    gpu::DeviceRequest request = gpu::strict_device_request();
    bool exercised;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) continue;
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
                request:          &request,
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
            @catch(gpu::create_device(&adapter, &request)) == gpu::SLOT_TABLE_FULL,
            "public creation reports a full registry without retaining the runtime"
        );

        destroy_registry_blockers(blockers[..]);
        gpu::Device device = gpu::create_device(&adapter, &request)!!;
        gpu::destroy_device(&device)!!;
        assert(
            gpu::get_adapter_info(&adapter)!!.name.len > 0,
            "registry recovery leaves the adapter usable"
        );
        gpu::destroy_runtime(&runtime)!!;
        break;
    }

    assert(exercised, "the controlled native target has a compatible strict adapter");
}

fn void runtime_devices_destroy_independently() @test {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_vk_multi_device_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceRequest request = gpu::strict_device_request();
    bool exercised;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) continue;

        gpu::Adapter second_adapter = adapter;
        bool distinct_adapters;
        for (uint j = 0; j < adapters.count; j++) {
            if (j == i) continue;
            gpu::Adapter candidate = adapters.get(j)!!;
            if (!gpu::supports_device_request(&candidate, &request)!!.supported) continue;
            second_adapter = candidate;
            distinct_adapters = true;
            break;
        }
        if (!distinct_adapters) {
            io::eprintfn("runtime_devices: distinct-adapter=N/A (one strict-compatible adapter)");
        }

        gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_adapter(&adapter);
        gpu::internal::vk::VkRuntimeState* runtime_state =
            runtime_data.state;
        gpu::Device first = gpu::create_device(&adapter, &request)!!;
        gpu::Device second = gpu::create_device(&second_adapter, &request)!!;
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
        assert(gpu::get_device_caps(&second)!!.strict_enabled, "destroying one device preserves its peer");
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
    assert(exercised, "the controlled native target has a compatible strict adapter");
}
