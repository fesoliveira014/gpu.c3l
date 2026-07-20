module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::atomic;
import std::io;

const gpu::BackendVTable REGISTRY_BLOCKER_VTABLE @private = {
    .destroy_device = &destroy_registry_blocker,
};
const gpu::RuntimeBackendVTable PRESENTATION_INSTANCE_GATE_VTABLE @private = {
    .destroy_runtime = &destroy_presentation_instance_gate_runtime,
    .destroy_surface = &destroy_presentation_instance_gate_surface,
};
fn uint live_device_slot_count() @private {
    uint live_count;
    for (uint i = 0; i < gpu::MAX_DEVICES; i++) {
        gpu::DeviceSlot* slot = &gpu::device_slots[i];
        gpu::DeviceSlotControl control =
            (gpu::DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
        if (control.state == gpu::DEVICE_SLOT_LIVE) live_count++;
    }
    return live_count;
}

fn void destroy_registry_blocker(void* backend_state) @private {
    (void)backend_state;
}

fn void destroy_presentation_instance_gate_runtime(
    void* backend_state,
) @private {
    (void)backend_state;
}

fn void destroy_presentation_instance_gate_surface(
    void* backend_state,
    void* backend_surface,
) @private {
    (void)backend_state;
    (void)backend_surface;
}

fn void destroy_registry_blockers(gpu::Device[] blockers) @private {
    foreach (&blocker : blockers) {
        if (blocker.is_valid()) (void)gpu::destroy_device(blocker);
    }
}

fn uptr device_physical_adapter(gpu::Device* device) @private {
    gpu::DeviceOperation operation = gpu::begin_device_operation(device)!!;
    defer gpu::end_device_operation(&operation);
    gpu::vk::VkDeviceState* state =
        (gpu::vk::VkDeviceState*)gpu::device_backend_state_ptr(device);
    return (uptr)state.physical_device;
}

fn gpu::vk::IndexingHeapLimits permissive_indexing_heap_limits() @private {
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
    gpu::DeviceDesc desc = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
    };
    gpu::vk::IndexingHeapLimits limits = permissive_indexing_heap_limits();

    gpu::vk::VkStrictHeapState disabled =
        gpu::vk::resolve_strict_heap_state(false, &desc)!!;
    gpu::vk::select_strict_heap_implementation(
        state:              &disabled,
        indexing_supported: false,
        buffer_supported:   false,
        indexing_limits:    limits,
    )!!;
    assert(!disabled.enabled, "an unrequested strict heap remains disabled");
    assert(
        disabled.implementation == gpu::vk::VkHeapImplementation.NONE,
        "a disabled strict heap selects no private implementation"
    );
    assert(disabled.texture_capacity == 0, "a disabled strict heap reserves no texture capacity");
    assert(disabled.sampler_capacity == 0, "a disabled strict heap reserves no sampler capacity");
    assert(disabled.descriptors.texture_cells.ptr == null, "a disabled strict heap owns no descriptor cells");

    gpu::vk::VkStrictHeapState indexing =
        gpu::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::vk::select_strict_heap_implementation(
        state:              &indexing,
        indexing_supported: true,
        buffer_supported:   true,
        indexing_limits:    limits,
    )!!;
    assert(indexing.enabled, "a requested strict heap is enabled");
    assert(
        indexing.implementation == gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        "indexing is preferred when it satisfies the semantic capacities"
    );
    assert(indexing.texture_capacity == 7, "the requested texture capacity is preserved");
    assert(indexing.sampler_capacity == 9, "the requested sampler capacity is preserved");
    assert(gpu::vk::strict_heap_uses_descriptor_indexing(&indexing), "the indexing path requests indexing features");
    assert(
        !gpu::vk::strict_heap_uses_descriptor_buffer(&indexing),
        "the indexing path requests no descriptor-buffer dispatch"
    );

    gpu::vk::IndexingHeapLimits constrained = limits;
    constrained.max_sampled_images = 6;
    gpu::vk::VkStrictHeapState fallback =
        gpu::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::vk::select_strict_heap_implementation(
        state:              &fallback,
        indexing_supported: true,
        buffer_supported:   true,
        indexing_limits:    constrained,
    )!!;
    assert(
        fallback.implementation == gpu::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        "descriptor buffers are selected when indexing cannot provide the requested capacity"
    );

    gpu::vk::VkStrictHeapState buffer_only =
        gpu::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::vk::select_strict_heap_implementation(
        state:              &buffer_only,
        indexing_supported: false,
        buffer_supported:   true,
        indexing_limits:    limits,
    )!!;
    assert(
        buffer_only.implementation == gpu::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        "descriptor buffers satisfy strict heaps when indexing is unavailable"
    );

    gpu::vk::VkStrictHeapState unavailable =
        gpu::vk::resolve_strict_heap_state(true, &desc)!!;
    fault selection_fault = @catch(gpu::vk::select_strict_heap_implementation(
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
        unavailable.implementation == gpu::vk::VkHeapImplementation.NONE,
        "failed selection leaves the private implementation unpublished"
    );
}

fn void descriptor_buffer_creation_override_is_private_and_exact() @test {
    gpu::DeviceDesc desc = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
    };
    gpu::vk::IndexingHeapLimits limits = permissive_indexing_heap_limits();

    gpu::vk::VkStrictHeapState automatic =
        gpu::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::vk::select_strict_heap_implementation_for_creation(
        state:              &automatic,
        indexing_supported: true,
        buffer_supported:   true,
        indexing_limits:    limits,
        probe:              null,
    )!!;
    assert(
        automatic.implementation == gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        "ordinary creation keeps automatic indexing preference"
    );

    gpu::vk::DeviceCreationProbe probe = {
        .descriptor_buffer_override = true,
    };
    gpu::vk::VkStrictHeapState overridden =
        gpu::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::vk::select_strict_heap_implementation_for_creation(
        state:              &overridden,
        indexing_supported: true,
        buffer_supported:   true,
        indexing_limits:    limits,
        probe:              &probe,
    )!!;
    assert(
        overridden.implementation == gpu::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        "the private creation probe can exercise descriptor-buffer runtime paths"
    );
    assert(
        probe.descriptor_buffer_capability_checked
            && probe.descriptor_buffer_capability_available
            && probe.descriptor_buffer_selected,
        "the probe records advertised capability and completed selection"
    );

    gpu::vk::VkStrictHeapState unsupported =
        gpu::vk::resolve_strict_heap_state(true, &desc)!!;
    assert(
        @catch(gpu::vk::select_strict_heap_implementation_for_creation(
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
    gpu::DeviceDesc desc = {
        .texture_heap_capacity = 7,
        .sampler_heap_capacity = 9,
    };
    gpu::vk::VkStrictHeapState state =
        gpu::vk::resolve_strict_heap_state(true, &desc)!!;
    gpu::vk::IndexingHeapLimits limits = permissive_indexing_heap_limits();
    assert(
        !gpu::vk::strict_heap_needs_descriptor_buffer_query(
            state:              &state,
            indexing_supported: true,
            indexing_limits:    limits,
            probe:              null,
        ),
        "a usable indexing heap does not require a descriptor-buffer query"
    );

    gpu::vk::IndexingHeapLimits constrained = limits;
    constrained.max_sampled_images = 6;
    assert(
        gpu::vk::strict_heap_needs_descriptor_buffer_query(
            state:              &state,
            indexing_supported: true,
            indexing_limits:    constrained,
            probe:              null,
        ),
        "an indexing-capacity shortfall requires descriptor-buffer fallback evidence"
    );
    assert(
        gpu::vk::strict_heap_needs_descriptor_buffer_query(
            state:              &state,
            indexing_supported: false,
            indexing_limits:    limits,
            probe:              null,
        ),
        "missing indexing features require descriptor-buffer fallback evidence"
    );

    gpu::vk::DeviceCreationProbe probe = {
        .descriptor_buffer_override = true,
    };
    assert(
        gpu::vk::strict_heap_needs_descriptor_buffer_query(
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
        !gpu::vk::query_descriptor_buffer_support_with_query(
            physical_device,
            &descriptor_buffer_extension_absent,
        )!!,
        "confirmed extension absence is unsupported capability"
    );
    assert(
        @catch(gpu::vk::query_descriptor_buffer_support_with_query(
            physical_device,
            &descriptor_buffer_extension_query_host_oom,
        )) == gpu::OUT_OF_HOST_MEMORY,
        "extension-query host OOM maps into the public fault taxonomy"
    );
    assert(
        @catch(gpu::vk::query_descriptor_buffer_support_with_query(
            physical_device,
            &descriptor_buffer_extension_query_device_oom,
        )) == gpu::OUT_OF_DEVICE_MEMORY,
        "extension-query device OOM maps into the public fault taxonomy"
    );
    assert(
        @catch(gpu::vk::query_descriptor_buffer_support_with_query(
            physical_device,
            &descriptor_buffer_extension_query_layer_missing,
        )) == gpu::UNSUPPORTED_FEATURE,
        "a missing extension layer maps through the public unsupported-feature taxonomy"
    );
}

fn void adapter_request_creation_reuses_runtime_discovery() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
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

        gpu::RuntimeData* runtime_data = gpu::resolve_adapter(&adapter);
        gpu::vk::VkRuntimeState* runtime_state =
            (gpu::vk::VkRuntimeState*)runtime_data.backend_state;
        gpu::Device device = gpu::create_device(&adapter, &request)!!;
        gpu::vk::VkDeviceState* device_state =
            (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);

        assert(device_state.instance == runtime_state.instance, "the device reuses the runtime instance");
        assert(
            device_state.physical_device
                == runtime_state.adapters[adapter.adapter_index].physical_device,
            "the device uses the exact requested physical adapter"
        );
        assert(!device_state.owns_instance, "the adapter device does not own the runtime instance");
        gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
        assert(device_state.strict_heap.enabled, "the strict request owns heap state");
        assert(caps.strict_enabled, "strict state is enabled");
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
        if (gpu::vk::strict_heap_uses_descriptor_buffer(&device_state.strict_heap)) {
            assert(
                gpu::vk::device_dispatch_supports_descriptor_buffer(&device_state.device_dispatch),
                "the selected descriptor-buffer path loads its dispatch"
            );
            assert(
                device_state.strict_heap.descriptors.descriptor_buffer != null,
                "the selected descriptor-buffer path owns its buffer"
            );
        } else {
            assert(
                !gpu::vk::device_dispatch_supports_descriptor_buffer(&device_state.device_dispatch),
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
            !gpu::vk::instance_dispatch_supports_presentation(&device_state.instance_dispatch),
            "headless requests retain no surface procedures"
        );
        assert(
            !gpu::vk::device_dispatch_supports_presentation(&device_state.device_dispatch),
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
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "gpu_vk_device_request_failure_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    if (adapters.count == 0) {
        gpu::destroy_runtime(&runtime)!!;
        return;
    }

    gpu::Adapter adapter = adapters.get(0)!!;
    gpu::DeviceRequest empty = gpu::empty_device_request();
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
    gpu::vk::VkAdapterState[1] adapter_states;
    gpu::vk::VkRuntimeState runtime_state = {
        .adapters = adapter_states[..],
    };
    gpu::Runtime runtime = gpu::register_runtime(
        backend:       gpu::BackendKind.VULKAN,
        vtable:        &PRESENTATION_INSTANCE_GATE_VTABLE,
        backend_state: &runtime_state,
        adapter_count: 1,
    )!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::Surface surface = gpu::register_surface(&runtime, (void*)1)!!;
    defer gpu::destroy_surface(&surface)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::Adapter adapter = adapters.get(0)!!;
    gpu::DeviceRequest request = gpu::request_presentation(
        gpu::strict_device_request(),
        &surface,
    )!!;
    gpu::vk::DeviceCreationProbe probe;

    assert(
        @catch(gpu::vk::create_vk_device_with_probe(
            adapter:          &adapter,
            request:          &request,
            retained_runtime: &runtime,
            probe:            &probe,
        )) == gpu::UNSUPPORTED_FEATURE,
        "presentation creation requires instance maintenance support"
    );
    assert(
        probe.reached == gpu::vk::DeviceCreationCheckpoint.NONE
            && probe.rollback_count == 0,
        "the instance capability gate precedes backend device mutation"
    );
}

fn void device_creation_checkpoint_ordinals_match_exact_schema() @test {
    assert(gpu::vk::DeviceCreationCheckpoint.NONE.ordinal == 0);
    assert(gpu::vk::DeviceCreationCheckpoint.OWNED_INSTANCE.ordinal == 1);
    assert(gpu::vk::DeviceCreationCheckpoint.LOGICAL_DEVICE.ordinal == 2);
    assert(gpu::vk::DeviceCreationCheckpoint.ALLOCATOR.ordinal == 3);
    assert(gpu::vk::DeviceCreationCheckpoint.RECORDING_CONTEXTS.ordinal == 4);
    assert(gpu::vk::DeviceCreationCheckpoint.DESCRIPTOR_HEAP.ordinal == 5);
    assert(gpu::vk::DeviceCreationCheckpoint.PIPELINE_SHARED.ordinal == 6);
    assert(gpu::vk::DeviceCreationCheckpoint.BEFORE_PUBLICATION.ordinal == 7);
}

fn void failed_native_device_creation_rolls_back_each_checkpoint() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .application_name  = "gpu_vk_device_request_rollback_test",
        .enable_validation = true,
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceRequest request = gpu::strict_device_request();

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) continue;

        for (
            uint checkpoint_ordinal = gpu::vk::DeviceCreationCheckpoint.LOGICAL_DEVICE.ordinal;
            checkpoint_ordinal <= gpu::vk::DeviceCreationCheckpoint.BEFORE_PUBLICATION.ordinal;
            checkpoint_ordinal++
        ) {
            gpu::vk::DeviceCreationCheckpoint checkpoint =
                (gpu::vk::DeviceCreationCheckpoint)checkpoint_ordinal;
            gpu::Runtime retained_runtime = runtime;
            gpu::runtime_retain_device(&retained_runtime)!!;
            gpu::vk::DeviceCreationProbe probe = { .fail_at = checkpoint };
            uint live_slots_before = live_device_slot_count();

            assert(
                @catch(gpu::vk::create_vk_device_with_probe(
                    adapter:          &adapter,
                    request:          &request,
                    retained_runtime: &retained_runtime,
                    probe:            &probe,
                )) == gpu::BACKEND_ERROR,
                "a checkpoint failure returns the backend fault"
            );
            gpu::runtime_release_device(&retained_runtime)!!;
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

fn void owned_instance_creation_rolls_back_after_bootstrap() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .application_name  = "gpu_vk_owned_instance_rollback_test",
        .enable_validation = true,
    };
    gpu::vk::DeviceCreationProbe probe = {
        .fail_at = gpu::vk::DeviceCreationCheckpoint.OWNED_INSTANCE,
    };
    uint live_slots_before = live_device_slot_count();

    assert(
        @catch(gpu::vk::create_vk_device_from_desc_with_probe(&desc, &probe))
            == gpu::BACKEND_ERROR,
        "an owned-instance checkpoint failure returns the backend fault"
    );
    assert(
        probe.reached == gpu::vk::DeviceCreationCheckpoint.OWNED_INSTANCE,
        "owned-instance creation reaches its checkpoint"
    );
    assert(probe.rollback_count == 1, "owned-instance creation rolls back once");
    assert(
        probe.owned_instance_teardown_count == 1,
        "owned-instance rollback destroys the instance once"
    );
    assert(
        live_device_slot_count() == live_slots_before,
        "owned-instance failure publishes no device"
    );
}

fn void publication_failure_rolls_back_native_state() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .application_name  = "gpu_vk_publication_rollback_test",
        .enable_validation = true,
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

        uint blocker_state;
        gpu::Device[gpu::MAX_DEVICES] blockers;
        defer destroy_registry_blockers(blockers[..]);
        foreach (&blocker : blockers) {
            *blocker = gpu::register_device(
                backend:       gpu::BackendKind.VULKAN,
                caps:          {},
                vtable:        &REGISTRY_BLOCKER_VTABLE,
                backend_state: &blocker_state,
            )!!;
        }

        gpu::Runtime retained_runtime = runtime;
        gpu::runtime_retain_device(&retained_runtime)!!;
        bool retain_live = true;
        defer {
            if (retain_live) (void)gpu::runtime_release_device(&retained_runtime);
        }
        gpu::vk::DeviceCreationProbe probe;
        uint live_slots_before = live_device_slot_count();

        assert(
            @catch(gpu::vk::create_vk_device_with_probe(
                adapter:          &adapter,
                request:          &request,
                retained_runtime: &retained_runtime,
                probe:            &probe,
            )) == gpu::SLOT_TABLE_FULL,
            "registration reports a full device registry"
        );
        gpu::runtime_release_device(&retained_runtime)!!;
        retain_live = false;
        assert(
            probe.reached == gpu::vk::DeviceCreationCheckpoint.BEFORE_PUBLICATION,
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
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
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

        gpu::RuntimeData* runtime_data = gpu::resolve_adapter(&adapter);
        gpu::vk::VkRuntimeState* runtime_state =
            (gpu::vk::VkRuntimeState*)runtime_data.backend_state;
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
