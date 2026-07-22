module gpu;

import gpu::vk @public;
import std::atomic;
import std::atomic::types;

<* True when the device handle is not the invalid zero sentinel. *>
fn bool Device.is_valid(self)          => (ulong)self != 0;
// Owner-bearing handles require both fields to be nonzero.
fn bool GpuAllocation.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool GpuAllocation.equals(self, GpuAllocation other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool TextureHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool TextureHandle.equals(self, TextureHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool AttachmentViewHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool AttachmentViewHandle.equals(self, AttachmentViewHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool TextureView.is_valid(self) {
    return self.owner != 0 && self.index.is_valid() && self.generation != 0;
}
fn bool TextureView.equals(self, TextureView other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool PipelineHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool PipelineHandle.equals(self, PipelineHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool SwapchainHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool SwapchainHandle.equals(self, SwapchainHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool CommandListHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool CommandListHandle.equals(self, CommandListHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool CommandAllocatorHandle.is_valid(self) =>
    self.owner != 0 && self.generation != 0;
fn bool CommandAllocatorHandle.equals(
    self,
    CommandAllocatorHandle other,
) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}

<* True for a live heap index; the zero value is the invalid sentinel. *>
fn bool TextureIndex.is_valid(self) => (uint)self != 0;
<* True for a live heap index; the zero value is the invalid sentinel. *>
fn bool SamplerIndex.is_valid(self) => (uint)self != 0;

<* True when the token is not the invalid zero value. *>
fn bool Runtime.is_valid(self) => (ulong)self != 0;

<* Return the former all-validation-enabled runtime configuration. *>
fn RuntimeDesc full_validation_runtime_desc() => {
    .backend                  = BackendKind.VULKAN,
    .contract_validation      = ContractValidation.FULL,
    .track_resource_lifetimes = true,
    .enable_vulkan_validation = true,
};

<* Initialize backend discovery. Importing gpu performs no initialization. *>
fn Runtime? create_runtime(RuntimeDesc* desc) {
    validate_runtime_desc(desc)!;
    switch (desc.backend) {
        case BackendKind.VULKAN:
            return gpu::vk::create_vk_runtime(desc);
        default:
            DebugMessage message = {
                .severity       = DebugMessageSeverity.ERROR,
                .category       = { .public_contract },
                .operation      = "create_runtime",
                .has_fault      = true,
                .public_fault   = UNSUPPORTED_BACKEND,
                .rejected_field = "RuntimeDesc.backend",
                .invariant      = "backend must be supported",
                .backend_text   = "unsupported GPU backend",
            };
            dispatch_debug_message(desc.debug_callback, desc.debug_user_data, &message);
            return UNSUPPORTED_BACKEND~;
    }
}

<* Destroy a runtime after its surfaces and devices have been destroyed. *>
fn void? destroy_runtime(Runtime* runtime) {
    RuntimeData* data = resolve_runtime(runtime);
    if (data == null) return INVALID_HANDLE~;
    if (data.live_surface_count > 0 || data.live_device_count.load(AtomicOrdering.ACQUIRE) > 0) {
        return RESOURCE_IN_USE~;
    }

    DestroyRuntimeBackendFn destroy_backend = data.vtable.destroy_runtime;
    void* backend_state = data.backend_state;
    data.vtable = null;
    data.backend_state = null;
    data.adapter_count = 0;
    data.used = false;
    *runtime = RUNTIME_INVALID;
    destroy_backend(backend_state);
}

<* True when the token is not the invalid zero value. *>
fn bool Adapter.is_valid(self) => (ulong)self != 0;

<* Return one borrowed adapter from this runtime-owned view. *>
fn Adapter? AdapterList.get(&self, uint index) {
    RuntimeData* data = resolve_runtime(&self.runtime);
    if (data == null) return INVALID_HANDLE~;
    if (index >= self.count || index >= data.adapter_count || index > ushort::max) {
        return INVALID_ARGUMENT~;
    }
    return (Adapter){
        .adapter_index      = (ushort)index,
        .runtime_index      = (ushort)self.runtime.index,
        .runtime_generation = self.runtime.generation,
    };
}

<* Return an allocation-free view of the runtime's borrowed adapters. *>
fn AdapterList? enumerate_adapters(Runtime* runtime) {
    RuntimeData* data = resolve_runtime(runtime);
    if (data == null) return INVALID_HANDLE~;
    return {
        .runtime = *runtime,
        .count   = data.adapter_count,
    };
}

<* Query backend-neutral information for a borrowed adapter. *>
fn AdapterInfo? get_adapter_info(Adapter* adapter) {
    RuntimeData* data = resolve_adapter(adapter);
    if (data == null) return INVALID_HANDLE~;
    return data.vtable.get_adapter_info(
        data.backend_state,
        adapter.adapter_index,
    );
}

<* Query backend and driver identity for diagnostics and logging. *>
fn AdapterDiagnostics? get_adapter_diagnostics(Adapter* adapter) {
    RuntimeData* data = resolve_adapter(adapter);
    if (data == null) return INVALID_HANDLE~;
    return data.vtable.get_adapter_diagnostics(
        data.backend_state,
        adapter.adapter_index,
    );
}

<* Create a strict semantic device request. *>
fn DeviceRequest strict_device_request() {
    return compose_device_request(
        empty_device_request(),
        DeviceCapability.STRICT,
    )!!;
}

<* Return a new request that requires presentation to the given surface. *>
fn DeviceRequest? request_presentation(
    DeviceRequest request,
    Surface* surface,
) {
    return compose_device_request(
        request,
        DeviceCapability.PRESENTATION,
        surface,
    );
}

<* Return a new request with one explicit semantic queue group. *>
fn DeviceRequest? request_queues(
    DeviceRequest request,
    QueueRequirements requirements,
) {
    if (!device_request_is_well_formed(request)
        || !queue_requirements_are_valid(requirements)) {
        return INVALID_ARGUMENT~;
    }
    if (((ulong)(uint128)request & DEVICE_REQUEST_QUEUES_BIT) != 0) {
        return INVALID_ARGUMENT~;
    }
    if (requested_surface(request).is_valid()
        && requirements.counts.graphics == 0) {
        return INVALID_ARGUMENT~;
    }
    return (DeviceRequest)((uint128)request
        | (uint128)(DEVICE_REQUEST_QUEUES_BIT | encode_queue_requirements(requirements)));
}

<* Query semantic request support without enabling device state.
   unmet_requirement is a borrowed static label. *>
fn DeviceRequestSupport? supports_device_request(
    Adapter* adapter,
    DeviceRequest* request,
) {
    validate_device_request(request)!;
    RuntimeData* data = resolve_adapter(adapter);
    if (data == null) return INVALID_HANDLE~;

    AdapterInfo info = data.vtable.get_adapter_info(
        data.backend_state,
        adapter.adapter_index,
    )!;
    if (!info.strict_supported) {
        return {
            .supported         = false,
            .unmet_requirement = "strict semantics",
        };
    }

    QueueRequirements queues = requested_queue_requirements(*request);
    if (queues.counts.graphics > info.queues.graphics_count) {
        return {
            .supported         = false,
            .unmet_requirement = "graphics queues",
        };
    }
    if (queues.counts.compute > info.queues.compute_count) {
        return {
            .supported         = false,
            .unmet_requirement = "compute queues",
        };
    }
    if (queues.counts.transfer > info.queues.transfer_count) {
        return {
            .supported         = false,
            .unmet_requirement = "transfer queues",
        };
    }

    Surface surface = requested_surface(*request);
    void* backend_surface = null;
    if (surface.is_valid() && !supports_presentation(adapter, &surface)!) {
        return {
            .supported         = false,
            .unmet_requirement = "presentation",
        };
    }
    if (surface.is_valid()) {
        backend_surface = surface_backend_handle(&surface)!;
    }

    if (data.vtable.supports_queue_requirements == null
        || !data.vtable.supports_queue_requirements(
            data.backend_state,
            adapter.adapter_index,
            &queues,
            backend_surface,
        )!) {
        return {
            .supported         = false,
            .unmet_requirement = "queue topology",
        };
    }

    return { .supported = true };
}

fn bool Surface.is_valid(self) => (ulong)self != 0;

<* Return whether an adapter can present to a surface from the same runtime. *>
fn bool? supports_presentation(Adapter* adapter, Surface* surface) {
    RuntimeData* adapter_runtime = resolve_adapter(adapter);
    SurfaceData* surface_data = resolve_surface(surface);
    if (adapter_runtime == null || surface_data == null) return INVALID_HANDLE~;

    RuntimeData* surface_runtime = resolve_runtime(&surface_data.runtime);
    if (surface_runtime != adapter_runtime) return INVALID_HANDLE~;
    if (adapter_runtime.vtable.supports_presentation == null) return false;
    return adapter_runtime.vtable.supports_presentation(
        adapter_runtime.backend_state,
        adapter.adapter_index,
        surface_data.backend_surface,
    );
}

<* Destroy a surface after all swapchains that borrow it. *>
fn void? destroy_surface(Surface* surface) {
    SurfaceData* data = resolve_surface(surface);
    if (data == null) return INVALID_HANDLE~;
    if (data.live_swapchain_count > 0) return RESOURCE_IN_USE~;

    Runtime runtime = data.runtime;
    RuntimeData* runtime_data = resolve_runtime(&runtime);
    if (runtime_data == null) return INVALID_HANDLE~;
    void* backend_surface = data.backend_surface;

    data.runtime = RUNTIME_INVALID;
    data.backend_surface = null;
    data.live_swapchain_count = 0;
    data.used = false;
    *surface = SURFACE_INVALID;

    runtime_data.vtable.destroy_surface(
        runtime_data.backend_state,
        backend_surface,
    );
    runtime_release_surface(&runtime)!!;
}

<* Reset process-wide command instrumentation before a synchronized test or
   benchmark interval. Do not call while another thread records commands. *>
fn void reset_command_resolution_stats() @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS)) {
    command_device_registry_resolutions.store(0, AtomicOrdering.RELAXED);
    command_retained_pin_operations.store(0, AtomicOrdering.RELAXED);
    command_lifecycle_vtable_dispatches.store(0, AtomicOrdering.RELAXED);
    command_table_lookups.store(0, AtomicOrdering.RELAXED);
    command_pipeline_table_lookups.store(0, AtomicOrdering.RELAXED);
    command_pipeline_cache_lookups.store(0, AtomicOrdering.RELAXED);
    command_policy_selections.store(0, AtomicOrdering.RELAXED);
    command_recording_commands.store(0, AtomicOrdering.RELAXED);
    command_native_commands.store(0, AtomicOrdering.RELAXED);
}

<* Snapshot process-wide command instrumentation with relaxed loads.
   recording_commands counts live-encoder entry-point attempts;
   native_commands counts every Vulkan command emitted by those paths. *>
fn CommandResolutionStats command_resolution_stats() @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS)) {
    return {
        .device_registry_resolutions = command_device_registry_resolutions.load(
            AtomicOrdering.RELAXED,
        ),
        .retained_pin_operations = command_retained_pin_operations.load(
            AtomicOrdering.RELAXED,
        ),
        .lifecycle_vtable_dispatches = command_lifecycle_vtable_dispatches.load(
            AtomicOrdering.RELAXED,
        ),
        .command_table_lookups = command_table_lookups.load(
            AtomicOrdering.RELAXED,
        ),
        .pipeline_table_lookups = command_pipeline_table_lookups.load(
            AtomicOrdering.RELAXED,
        ),
        .pipeline_cache_lookups = command_pipeline_cache_lookups.load(
            AtomicOrdering.RELAXED,
        ),
        .command_policy_selections = command_policy_selections.load(
            AtomicOrdering.RELAXED,
        ),
        .recording_commands = command_recording_commands.load(
            AtomicOrdering.RELAXED,
        ),
        .native_commands = command_native_commands.load(
            AtomicOrdering.RELAXED,
        ),
    };
}

<* Return the backend selected for the live device. *>
fn BackendKind? get_device_backend(Device* device) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.data.backend;
}

<* Snapshot the selected device's capabilities. *>
fn DeviceCaps? get_device_caps(Device* device) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.data.caps;
}

<* Create a strict device from one exact borrowed adapter.
   Ownership: the caller must release the returned device with destroy_device. *>
fn Device? create_device(Adapter* adapter, DeviceRequest* request) {
    validate_device_request(request)!;
    RuntimeData* runtime_data = resolve_adapter(adapter);
    if (runtime_data == null) return INVALID_HANDLE~;
    if (!device_slot_available()) return SLOT_TABLE_FULL~;

    DeviceRequestSupport support = supports_device_request(adapter, request)!;
    if (!support.supported) return UNSUPPORTED_FEATURE~;

    Runtime runtime = {
        .index      = adapter.runtime_index,
        .generation = adapter.runtime_generation,
    };
    runtime_retain_device(&runtime)!;

    Device device;
    bool retain_transferred = false;
    defer catch {
        if ((ulong)device != 0) {
            (void)destroy_device(&device);
        }
        if (!retain_transferred) {
            (void)runtime_release_device(&runtime);
        }
    }

    switch (runtime_data.backend) {
        case BackendKind.VULKAN:
            device = gpu::vk::create_vk_device(adapter, request, &runtime)!;
        default:
            return UNSUPPORTED_BACKEND~;
    }
    retain_transferred = true;
    return device;
}

<* Destroy a quiescent device. Live children fault RESOURCE_IN_USE; active
   host or queue work faults DEVICE_BUSY without consuming the token. *>
fn void? destroy_device(Device* device) {
    if (device == null || !device.is_valid()) return INVALID_HANDLE~;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return INVALID_HANDLE~;

    DeviceSlot* slot = &device_slots[device.index];
    DeviceSlotControl closing;
    DeviceData data;
    {
        lock_device_registry();
        defer unlock_device_registry();
        DeviceSlotControl control = (DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
        if (control.generation != device.generation) return INVALID_HANDLE~;
        if (control.state == DEVICE_SLOT_CLOSING) return DEVICE_BUSY~;
        if (control.state != DEVICE_SLOT_LIVE) return INVALID_HANDLE~;
        bool lost = slot.lost.load(AtomicOrdering.ACQUIRE);
        if (!lost && slot.live_children.load(AtomicOrdering.ACQUIRE) > 0) {
            return RESOURCE_IN_USE~;
        }

        DeviceSlotControl state_step = { .state = 1 };
        ulong previous = slot.control.add((ulong)state_step, AtomicOrdering.ACQUIRE_RELEASE);
        closing = (DeviceSlotControl)previous;
        closing.state = DEVICE_SLOT_CLOSING;

        if (closing.pins > 0) {
            (void)slot.control.sub((ulong)state_step, AtomicOrdering.RELEASE);
            return DEVICE_BUSY~;
        }
        if (!lost && slot.live_children.load(AtomicOrdering.ACQUIRE) > 0) {
            (void)slot.control.sub((ulong)state_step, AtomicOrdering.RELEASE);
            return RESOURCE_IN_USE~;
        }
        if (!lost && slot.data.vtable.check_device_destroy != null) {
            if (catch readiness_fault = slot.data.vtable.check_device_destroy(
                slot.data.backend_state,
            )) {
                (void)slot.control.sub((ulong)state_step, AtomicOrdering.RELEASE);
                return readiness_fault~;
            }
            lost = slot.lost.load(AtomicOrdering.ACQUIRE);
            if (!lost && slot.live_children.load(AtomicOrdering.ACQUIRE) > 0) {
                (void)slot.control.sub((ulong)state_step, AtomicOrdering.RELEASE);
                return RESOURCE_IN_USE~;
            }
        }

        data = slot.data;
    }

    data.vtable.destroy_device(data.backend_state);

    {
        lock_device_registry();
        defer unlock_device_registry();
        slot.data = {};
        slot.live_children.store(0, AtomicOrdering.RELAXED);
        slot.lost.store(false, AtomicOrdering.RELAXED);
        closing.pins = 0;
        if (closing.generation == MAX_DEVICE_GENERATION) {
            closing.state = DEVICE_SLOT_RETIRED;
        } else {
            closing.generation++;
            closing.state = DEVICE_SLOT_FREE;
        }
        closing.reserved = 0;
        slot.control.store((ulong)closing, AtomicOrdering.RELEASE);
    }

    *device = DEVICE_INVALID;
    if (data.runtime_retained) runtime_release_device(&data.runtime)!;
}

<* Snapshot per-heap memory usage and live allocation counts. *>
fn MemoryStats? get_memory_stats(Device* device) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_memory_stats(device);
}

<* Build a human-readable memory report. Ownership: the caller frees the
   returned String. *>
fn String? build_memory_report(Device* device, bool detailed) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.build_memory_report(device, detailed);
}

<* Open a debug-utils label region; a no-op when the extension is absent.
   Valid while recording, including inside render passes; balance is the
   caller's responsibility. *>
fn void? cmd_begin_label(CommandList* commands, ZString label, float[4] color = {}) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    if (label == null) return INVALID_ARGUMENT~;
    return encoder.ops.begin_label(encoder, label, color);
}

<* Counterpart of cmd_begin_label. *>
fn void? cmd_end_label(CommandList* commands) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.end_label(encoder);
}
