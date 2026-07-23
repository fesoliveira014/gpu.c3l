module gpu;

import gpu::internal @public;
import gpu::internal::vk @public;
import std::atomic;
import std::atomic::types;
import std::core::mem;

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
    gpu::internal::validate_runtime_desc(desc)!;
    switch (desc.backend) {
        case BackendKind.VULKAN:
            return gpu::internal::vk::create_vk_runtime(desc);
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
            gpu::internal::dispatch_debug_message(desc.debug_callback, desc.debug_user_data, &message);
            return UNSUPPORTED_BACKEND~;
    }
}

<* Destroy a runtime after its surfaces and devices have been destroyed. *>
fn void? destroy_runtime(Runtime* runtime) {
    gpu::internal::RuntimeData* data = gpu::internal::resolve_runtime(runtime);
    if (data == null) return INVALID_HANDLE~;
    if (data.live_surface_count > 0 || data.live_device_count.load(AtomicOrdering.ACQUIRE) > 0) {
        return RESOURCE_IN_USE~;
    }

    gpu::internal::DestroyRuntimeBackendFn destroy_backend = data.vtable.destroy_runtime;
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
    gpu::internal::RuntimeData* data = gpu::internal::resolve_runtime(&self.runtime);
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
    gpu::internal::RuntimeData* data = gpu::internal::resolve_runtime(runtime);
    if (data == null) return INVALID_HANDLE~;
    return {
        .runtime = *runtime,
        .count   = data.adapter_count,
    };
}

<* Query backend-neutral information for a borrowed adapter. *>
fn AdapterInfo? get_adapter_info(Adapter* adapter) {
    gpu::internal::RuntimeData* data = gpu::internal::resolve_adapter(adapter);
    if (data == null) return INVALID_HANDLE~;
    return data.vtable.get_adapter_info(
        data.backend_state,
        adapter.adapter_index,
    );
}

<* Query backend and driver identity for diagnostics and logging. *>
fn AdapterDiagnostics? get_adapter_diagnostics(Adapter* adapter) {
    gpu::internal::RuntimeData* data = gpu::internal::resolve_adapter(adapter);
    if (data == null) return INVALID_HANDLE~;
    return data.vtable.get_adapter_diagnostics(
        data.backend_state,
        adapter.adapter_index,
    );
}

<* Create a strict semantic device request. *>
fn DeviceRequest strict_device_request() {
    return gpu::internal::compose_device_request(
        gpu::internal::empty_device_request(),
        gpu::internal::DeviceCapability.STRICT,
    )!!;
}

<* Return a new request that requires presentation to the given surface. *>
fn DeviceRequest? request_presentation(
    DeviceRequest request,
    Surface* surface,
) {
    return gpu::internal::compose_device_request(
        request,
        gpu::internal::DeviceCapability.PRESENTATION,
        surface,
    );
}

<* Return a new request with one explicit semantic queue group. *>
fn DeviceRequest? request_queues(
    DeviceRequest request,
    QueueRequirements requirements,
) {
    if (!gpu::internal::device_request_is_well_formed(request)
        || !gpu::internal::queue_requirements_are_valid(requirements)) {
        return INVALID_ARGUMENT~;
    }
    if (((ulong)(uint128)request & gpu::internal::DEVICE_REQUEST_QUEUES_BIT) != 0) {
        return INVALID_ARGUMENT~;
    }
    if (gpu::internal::requested_surface(request).is_valid()
        && requirements.counts.graphics == 0) {
        return INVALID_ARGUMENT~;
    }
    return (DeviceRequest)((uint128)request
        | (uint128)(gpu::internal::DEVICE_REQUEST_QUEUES_BIT | gpu::internal::encode_queue_requirements(requirements)));
}

<* Query semantic request support without enabling device state.
   unmet_requirement is a borrowed static label. *>
fn DeviceRequestSupport? supports_device_request(
    Adapter* adapter,
    DeviceRequest* request,
) {
    gpu::internal::validate_device_request(request)!;
    gpu::internal::RuntimeData* data = gpu::internal::resolve_adapter(adapter);
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

    QueueRequirements queues = gpu::internal::requested_queue_requirements(*request);
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

    Surface surface = gpu::internal::requested_surface(*request);
    void* backend_surface = null;
    if (surface.is_valid() && !supports_presentation(adapter, &surface)!) {
        return {
            .supported         = false,
            .unmet_requirement = "presentation",
        };
    }
    if (surface.is_valid()) {
        backend_surface = gpu::internal::surface_backend_handle(&surface)!;
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
    gpu::internal::RuntimeData* adapter_runtime = gpu::internal::resolve_adapter(adapter);
    gpu::internal::SurfaceData* surface_data = gpu::internal::resolve_surface(surface);
    if (adapter_runtime == null || surface_data == null) return INVALID_HANDLE~;

    gpu::internal::RuntimeData* surface_runtime = gpu::internal::resolve_runtime(&surface_data.runtime);
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
    gpu::internal::SurfaceData* data = gpu::internal::resolve_surface(surface);
    if (data == null) return INVALID_HANDLE~;
    if (data.live_swapchain_count > 0) return RESOURCE_IN_USE~;

    Runtime runtime = data.runtime;
    gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_runtime(&runtime);
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
    gpu::internal::runtime_release_surface(&runtime)!!;
}

<* Reset process-wide command instrumentation before a synchronized test or
   benchmark interval. Do not call while another thread records commands. *>
fn void reset_command_resolution_stats() @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS)) {
    gpu::internal::command_device_registry_resolutions.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_retained_pin_operations.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_lifecycle_vtable_dispatches.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_table_lookups.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_pipeline_table_lookups.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_pipeline_cache_lookups.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_policy_selections.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_encoder_cell_computations.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_encoder_lease_comparisons.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_recording_commands.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_native_commands.store(0, AtomicOrdering.RELAXED);
}

<* Snapshot process-wide command instrumentation with relaxed loads.
   recording_commands counts live-encoder entry-point attempts;
   native_commands counts every Vulkan command emitted by those paths. *>
fn CommandResolutionStats command_resolution_stats() @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS)) {
    return {
        .device_registry_resolutions = gpu::internal::command_device_registry_resolutions.load(
            AtomicOrdering.RELAXED,
        ),
        .retained_pin_operations = gpu::internal::command_retained_pin_operations.load(
            AtomicOrdering.RELAXED,
        ),
        .lifecycle_vtable_dispatches = gpu::internal::command_lifecycle_vtable_dispatches.load(
            AtomicOrdering.RELAXED,
        ),
        .command_table_lookups = gpu::internal::command_table_lookups.load(
            AtomicOrdering.RELAXED,
        ),
        .pipeline_table_lookups = gpu::internal::command_pipeline_table_lookups.load(
            AtomicOrdering.RELAXED,
        ),
        .pipeline_cache_lookups = gpu::internal::command_pipeline_cache_lookups.load(
            AtomicOrdering.RELAXED,
        ),
        .command_policy_selections = gpu::internal::command_policy_selections.load(
            AtomicOrdering.RELAXED,
        ),
        .encoder_cell_computations = gpu::internal::command_encoder_cell_computations.load(
            AtomicOrdering.RELAXED,
        ),
        .encoder_lease_comparisons = gpu::internal::command_encoder_lease_comparisons.load(
            AtomicOrdering.RELAXED,
        ),
        .recording_commands = gpu::internal::command_recording_commands.load(
            AtomicOrdering.RELAXED,
        ),
        .native_commands = gpu::internal::command_native_commands.load(
            AtomicOrdering.RELAXED,
        ),
    };
}

<* Return the backend selected for the live device. *>
fn BackendKind? get_device_backend(Device* device) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.data.backend;
}

<* Snapshot the selected device's capabilities. *>
fn DeviceCaps? get_device_caps(Device* device) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.data.caps;
}

<* Create a strict device from one exact borrowed adapter.
   Ownership: the caller must release the returned device with destroy_device. *>
fn Device? create_device(Adapter* adapter, DeviceRequest* request) {
    gpu::internal::validate_device_request(request)!;
    gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_adapter(adapter);
    if (runtime_data == null) return INVALID_HANDLE~;
    if (!gpu::internal::device_slot_available()) return SLOT_TABLE_FULL~;

    DeviceRequestSupport support = supports_device_request(adapter, request)!;
    if (!support.supported) return UNSUPPORTED_FEATURE~;

    Runtime runtime = {
        .index      = adapter.runtime_index,
        .generation = adapter.runtime_generation,
    };
    gpu::internal::runtime_retain_device(&runtime)!;

    Device device;
    bool retain_transferred = false;
    defer catch {
        if ((ulong)device != 0) {
            (void)destroy_device(&device);
        }
        if (!retain_transferred) {
            (void)gpu::internal::runtime_release_device(&runtime);
        }
    }

    switch (runtime_data.backend) {
        case BackendKind.VULKAN:
            device = gpu::internal::vk::create_vk_device(adapter, request, &runtime)!;
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
    if (device.reserved != 0 || device.index >= gpu::internal::MAX_DEVICES) return INVALID_HANDLE~;

    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    gpu::internal::DeviceSlotControl closing;
    gpu::internal::DeviceData data;
    {
        gpu::internal::lock_device_registry();
        defer gpu::internal::unlock_device_registry();
        gpu::internal::DeviceSlotControl control = (gpu::internal::DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
        if (control.generation != device.generation) return INVALID_HANDLE~;
        if (control.state == gpu::internal::DEVICE_SLOT_CLOSING) return DEVICE_BUSY~;
        if (control.state != gpu::internal::DEVICE_SLOT_LIVE) return INVALID_HANDLE~;
        bool lost = slot.lost.load(AtomicOrdering.ACQUIRE);
        if (!lost && slot.live_children.load(AtomicOrdering.ACQUIRE) > 0) {
            return RESOURCE_IN_USE~;
        }

        gpu::internal::DeviceSlotControl state_step = { .state = 1 };
        ulong previous = slot.control.add((ulong)state_step, AtomicOrdering.ACQUIRE_RELEASE);
        closing = (gpu::internal::DeviceSlotControl)previous;
        closing.state = gpu::internal::DEVICE_SLOT_CLOSING;

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
        gpu::internal::lock_device_registry();
        defer gpu::internal::unlock_device_registry();
        slot.data = {};
        slot.live_children.store(0, AtomicOrdering.RELAXED);
        slot.lost.store(false, AtomicOrdering.RELAXED);
        closing.pins = 0;
        if (closing.generation == gpu::internal::MAX_DEVICE_GENERATION) {
            closing.state = gpu::internal::DEVICE_SLOT_RETIRED;
        } else {
            closing.generation++;
            closing.state = gpu::internal::DEVICE_SLOT_FREE;
        }
        closing.reserved = 0;
        slot.control.store((ulong)closing, AtomicOrdering.RELEASE);
    }

    *device = DEVICE_INVALID;
    if (data.runtime_retained) gpu::internal::runtime_release_device(&data.runtime)!;
}

<* Snapshot per-heap memory usage and live allocation counts. *>
fn MemoryStats? get_memory_stats(Device* device) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.vtable.get_memory_stats(device);
}

<* Build a human-readable memory report. Ownership: the caller frees the
   returned String. *>
fn String? build_memory_report(Device* device, bool detailed) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.vtable.build_memory_report(device, detailed);
}

<* Open a debug-utils label region; a no-op when the extension is absent.
   Valid while recording, including inside render passes; balance is the
   caller's responsibility. *>
fn void? cmd_begin_label(CommandList* commands, ZString label, float[4] color = {}) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    if (label == null) return INVALID_ARGUMENT~;
    return encoder.ops.begin_label(encoder, label, color);
}

<* Counterpart of cmd_begin_label. *>
fn void? cmd_end_label(CommandList* commands) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.end_label(encoder);
}

<* Allocate caller-managed GPU storage. Free with `free_allocation`. *>
fn GpuAllocation? allocate_memory(Device* device, AllocationDesc* desc) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null || desc.size == 0) return INVALID_ARGUMENT~;
    if (!gpu::internal::memory_class_is_known(desc.memory_class)) return INVALID_ARGUMENT~;
    gpu::internal::validate_queue_access(&operation, desc.access)!;

    AllocationDesc normalized = *desc;
    normalized.alignment = gpu::internal::normalized_allocation_alignment(desc.alignment)!;
    gpu::internal::validate_texture_allocation(&operation, desc, &normalized.alignment)!;
    GpuAllocation allocation = operation.vtable.allocate_memory(
        device,
        &normalized,
    )!;
    gpu::internal::retain_device_children(&operation);
    return allocation;
}

<* Free quiescent storage; faults preserve the owning token. *>
fn void? free_allocation(Device* device, GpuAllocation* allocation) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (allocation == null || !allocation.is_valid()) return INVALID_HANDLE~;
    gpu::internal::validate_resource_owner(
        &operation,
        allocation.owner,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    )!;
    operation.vtable.free_allocation(device, *allocation)!;
    gpu::internal::release_device_children(&operation);
    *allocation = GPU_ALLOCATION_INVALID;
}

<* Return immutable properties for a live allocation. *>
fn AllocationInfo? get_allocation_info(
    Device* device,
    GpuAllocation allocation,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(
        &operation,
        allocation.owner,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.get_allocation_info(device, allocation);
}

<* Borrow the allocation's complete range. *>
fn GpuSpan? get_allocation_span(Device* device, GpuAllocation allocation) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(
        &operation,
        allocation.owner,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.get_allocation_span(device, allocation);
}

<* Borrow the CPU mapping for a live host-visible span. *>
fn char[]? get_span_mapping(Device* device, GpuSpan span) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.vtable.get_span_mapping(device, span);
}

<* Return the stable nonzero address of a live span. *>
fn GpuAddress? get_span_address(Device* device, GpuSpan span) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.vtable.get_span_address(device, span);
}

<* Make CPU writes to a live mapped allocation range visible to the GPU.
   Coherent ranges are no-ops. *>
fn void? flush_mapped_span(Device* device, GpuSpan span) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(
        &operation,
        span.owner,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.flush_mapped_span(device, span);
}

<* After the relevant GPU completion point, make its writes visible to CPU
   reads. This call does not wait; coherent ranges are no-ops. *>
fn void? invalidate_mapped_span(Device* device, GpuSpan span) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(
        &operation,
        span.owner,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    )!;
    return operation.vtable.invalidate_mapped_span(device, span);
}

<* Derive a sub-range without bounds or overflow checks. Use only when the
   caller has already proved the range, or deliberately needs an invalid span
   for validation testing. *>
fn GpuSpan GpuSpan.unchecked_subspan(self, usz offset, usz size) {
    return {
        .owner      = self.owner,
        .index      = self.index,
        .generation = self.generation,
        .offset     = self.offset + offset,
        .size       = size,
    };
}

<* Derive a non-empty exact-sized range within this immediate parent. *>
fn GpuSpan? GpuSpan.checked_subspan(self, usz offset, usz size) {
    if (size == 0 || !gpu::internal::range_fits(self.size, offset, size)) {
        return INVALID_ARGUMENT~;
    }
    if (offset > usz::max - self.offset) return INVALID_ARGUMENT~;
    return self.unchecked_subspan(offset, size);
}

fn bool TextureCompatibility.is_valid(self) {
    uint128 raw = (uint128)self;
    ulong payload = (ulong)raw;
    return (ulong)(raw >> 64) != 0
        && (uint)payload != 0
        && payload >> 32 == 0;
}

<* Query library-creatable features for a format. Individual feature bits do
   not guarantee that a combination is supported. *>
fn TextureFormatSupport? get_texture_format_support(Device* device, Format format) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.vtable.get_texture_format_support(device, format);
}

<* Return whether an exact descriptor is creatable by the backend and adapter.
   This checks capabilities only; allocation can still fail for lack of memory. *>
fn bool? supports_texture_desc(Device* device, TextureDesc* desc) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    if (!gpu::internal::queue_access_is_valid(&operation, desc.access)) return false;
    return operation.vtable.supports_texture_desc(device, desc);
}

<* Return exact storage requirements for a texture descriptor. *>
fn TextureRequirements? get_texture_requirements(Device* device, TextureDesc* desc) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    gpu::internal::validate_queue_access(&operation, desc.access)!;
    return operation.vtable.get_texture_requirements(device, desc);
}

<* The handle owns the image; shader access goes through a TextureIndex.
   Ownership: release with destroy_texture. *>
fn TextureHandle? create_texture(Device* device, TextureDesc* desc) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    gpu::internal::validate_queue_access(&operation, desc.access)!;
    TextureHandle texture = operation.vtable.create_texture(device, desc)!;
    gpu::internal::retain_device_children(&operation);
    return texture;
}

<* Create a texture in caller-owned texture memory. *>
fn TextureHandle? create_placed_texture(
    Device* device,
    TextureDesc* desc,
    GpuAllocation allocation,
    usz offset,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    gpu::internal::validate_queue_access(&operation, desc.access)!;
    gpu::internal::validate_resource_owner(
        &operation,
        allocation.owner,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    )!;
    TextureHandle texture = operation.vtable.create_placed_texture(
        device:     device,
        desc:       desc,
        allocation: allocation,
        offset:     offset,
    )!;
    gpu::internal::retain_device_children(&operation);
    return texture;
}

<* Create a texture and its independently owned dedicated storage atomically. *>
fn DedicatedTexture? create_dedicated_texture(
    Device* device,
    TextureDesc* desc,
    AllocationDesc* allocation_desc,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null || allocation_desc == null
        || allocation_desc.size == 0
        || allocation_desc.memory_class != MemoryClass.TEXTURE) {
        return INVALID_ARGUMENT~;
    }
    gpu::internal::validate_queue_access(&operation, desc.access)!;
    gpu::internal::validate_queue_access(&operation, allocation_desc.access)!;
    if (!gpu::internal::queue_roles_are_subset(desc.access, allocation_desc.access)) {
        return INVALID_ARGUMENT~;
    }

    AllocationDesc normalized = *allocation_desc;
    normalized.alignment = gpu::internal::normalized_allocation_alignment(
        allocation_desc.alignment,
    )!;
    gpu::internal::validate_texture_allocation(
        operation:       &operation,
        desc:            allocation_desc,
        alignment:       &normalized.alignment,
        allow_dedicated: true,
    )!;
    DedicatedTexture created = operation.vtable.create_dedicated_texture(
        device,
        desc,
        &normalized,
    )!;
    gpu::internal::retain_device_children(&operation, 2);
    return created;
}

<* Destroy a texture after its descriptors. Placed storage remains caller-owned. *>
fn void? destroy_texture(Device* device, TextureHandle texture) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(&operation, texture.owner, gpu::internal::ResourceOwnerKind.TEXTURE)!;
    operation.vtable.destroy_texture(device, texture)!;
    gpu::internal::release_device_children(&operation);
}

<* Intern and publish an immutable device-lifetime sampler index. Identical
   semantic descriptions return the same value; debug_name is not semantic.
   Active anisotropy must be within the exact range reported by DeviceCaps;
   values above the maximum are rejected rather than clamped. *>
fn SamplerIndex? intern_sampler(Device* device, SamplerDesc* desc) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    if (!operation.data.caps.strict_enabled) return UNSUPPORTED_FEATURE~;
    gpu::internal::validate_sampler_desc(&operation, desc)!;
    return operation.vtable.intern_sampler(device, desc);
}

<* Publish one texture view in the strict shader-visible heap. Requires strict
   capability, and the texture must carry sampled or storage usage. The returned
   owner-bearing view contains the raw generation-free index stored in shader data. *>
fn TextureView? create_texture_view(
    Device* device,
    TextureHandle texture,
    TextureViewDesc* desc,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(&operation, texture.owner, gpu::internal::ResourceOwnerKind.TEXTURE)!;
    if (!operation.data.caps.strict_enabled) return UNSUPPORTED_FEATURE~;
    TextureView view = operation.vtable.create_texture_view(device, texture, desc)!;
    gpu::internal::retain_device_children(&operation);
    return view;
}

<* Release one texture view and recycle its shader-visible index immediately.
   GPU references to the index must already be complete. *>
fn void? destroy_texture_view(Device* device, TextureView view) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(&operation, view.owner, gpu::internal::ResourceOwnerKind.TEXTURE_VIEW)!;
    operation.vtable.destroy_texture_view(device, view)!;
    gpu::internal::release_device_children(&operation);
}

<* Batch-publish texture views. Requires strict capability.
   `out_views.len` must equal `descs.len`; an empty batch succeeds. A fault leaves
   heap allocation, generations, cached native views, and outputs unchanged. *>
fn void? create_texture_views(
    Device* device,
    TextureViewCreateDesc[] descs,
    TextureView[] out_views,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (out_views.len != descs.len) {
        return gpu::internal::report_device_contract_failure(
            operation:      &operation,
            operation_name: "create_texture_views",
            public_fault:   INVALID_ARGUMENT,
            resource_kind:  DebugResourceKind.TEXTURE_DESCRIPTOR,
            rejected_field: "out_views",
            invariant:      "output count must match descriptor count",
        )~;
    }
    if (descs.len == 0) return;
    foreach (&desc : descs) {
        gpu::internal::validate_resource_owner(&operation, desc.texture.owner, gpu::internal::ResourceOwnerKind.TEXTURE)!;
    }
    if (!operation.data.caps.strict_enabled) return UNSUPPORTED_FEATURE~;
    operation.vtable.create_texture_views(device, descs, out_views)!;
    gpu::internal::retain_device_children(&operation, (ulong)descs.len);
}

fn bool ShaderCode.is_valid(self) {
    if (self.digest == 0
        || !gpu::internal::shader_stage_valid(self.stage)
        || self.spirv.ptr == null
        || self.spirv.len < gpu::internal::SPIRV_HEADER_SIZE
        || self.spirv.len % uint::size != 0
        || self.entry_point == null) return false;

    uint magic;
    mem::copy(&magic, self.spirv.ptr, uint::size);
    usz entry_point_len = self.entry_point.str_view().len;
    return magic == gpu::internal::SPIRV_MAGIC
        && entry_point_len > 0
        && entry_point_len < gpu::internal::SHADER_ENTRY_POINT_CAP;
}

fn bool ShaderCode.equals(self, ShaderCode other) @operator(==) {
    if (!self.is_valid() || !other.is_valid()) {
        return !self.is_valid()
            && !other.is_valid()
            && self.spirv.ptr == other.spirv.ptr
            && self.spirv.len == other.spirv.len;
    }
    return self.digest == other.digest
        && self.stage == other.stage
        && self.entry_point.str_view() == other.entry_point.str_view()
        && self.spirv.len == other.spirv.len
        && mem::equals(self.spirv, other.spirv);
}

<* Validate borrowed SPIR-V and compute its opaque process-local identity. *>
fn ShaderCode? prepare_shader_code(ShaderDesc* desc) {
    if (desc == null) return INVALID_ARGUMENT~;
    if (!gpu::internal::shader_stage_valid(desc.stage)) return INVALID_ARGUMENT~;
    if (desc.spirv.ptr == null
        || desc.spirv.len < gpu::internal::SPIRV_HEADER_SIZE
        || desc.spirv.len % uint::size != 0) return SHADER_INVALID~;

    uint magic;
    mem::copy(&magic, desc.spirv.ptr, uint::size);
    if (magic != gpu::internal::SPIRV_MAGIC) return SHADER_INVALID~;

    ZString entry_point = desc.entry_point == null ? "main" : desc.entry_point;
    usz entry_point_len = entry_point.str_view().len;
    if (entry_point_len == 0 || entry_point_len >= gpu::internal::SHADER_ENTRY_POINT_CAP) {
        return INVALID_ARGUMENT~;
    }
    return {
        .stage       = desc.stage,
        .spirv       = desc.spirv,
        .entry_point = entry_point,
        .debug_name  = desc.debug_name,
        .digest      = gpu::internal::shader_code_digest(desc.spirv),
    };
}

<* Create a compute pipeline. Identical descriptors may alias one backend
   pipeline under distinct handles. Ownership: release with destroy_pipeline,
   once per create. *>
fn PipelineHandle? create_compute_pipeline(Device* device, ComputePipelineDesc* desc) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    PipelineHandle pipeline = operation.vtable.create_compute_pipeline(device, desc)!;
    gpu::internal::retain_device_children(&operation);
    return pipeline;
}

<* Create a graphics pipeline targeting dynamic rendering. The layout is fixed:
   the global heap set plus a 16-byte vertex/fragment root push range.
   Descriptors identical in immutable state alias one backend pipeline under
   distinct handles. Depth state is bound separately during command recording.
   At most MAX_COLOR_ATTACHMENTS color targets. Ownership:
   release with destroy_pipeline, once per create. *>
fn PipelineHandle? create_graphics_pipeline(Device* device, GraphicsPipelineDesc* desc) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    PipelineHandle pipeline = operation.vtable.create_graphics_pipeline(device, desc)!;
    gpu::internal::retain_device_children(&operation);
    return pipeline;
}

<* Create compute pipelines transactionally. Output count must match descriptor
   count; an empty batch succeeds. A fault leaves every output unchanged. *>
fn void? create_compute_pipelines(
    Device* device,
    ComputePipelineDesc[] descs,
    PipelineHandle[] out_pipelines,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (out_pipelines.len != descs.len) return INVALID_ARGUMENT~;
    if (descs.len == 0) return;
    operation.vtable.create_compute_pipelines(device, descs, out_pipelines)!;
    for (usz i = 0; i < descs.len; i++) gpu::internal::retain_device_children(&operation);
}

<* Create graphics pipelines transactionally. Output count must match descriptor
   count; an empty batch succeeds. A fault leaves every output unchanged. *>
fn void? create_graphics_pipelines(
    Device* device,
    GraphicsPipelineDesc[] descs,
    PipelineHandle[] out_pipelines,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (out_pipelines.len != descs.len) return INVALID_ARGUMENT~;
    if (descs.len == 0) return;
    operation.vtable.create_graphics_pipelines(device, descs, out_pipelines)!;
    for (usz i = 0; i < descs.len; i++) gpu::internal::retain_device_children(&operation);
}

<* Release one returned handle. Descriptions sharing one backend pipeline are
   refcounted. *>
fn void? destroy_pipeline(Device* device, PipelineHandle pipeline) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(&operation, pipeline.owner, gpu::internal::ResourceOwnerKind.PIPELINE)!;
    operation.vtable.destroy_pipeline(device, pipeline)!;
    gpu::internal::release_device_children(&operation);
}

<* Byte size of the driver pipeline cache blob. The cache grows as pipelines
   are built — re-query before each export. Must not be called concurrently
   with pipeline creation. *>
fn usz? get_pipeline_cache_size(Device* device) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.vtable.get_pipeline_cache_size(device);
}

<* Fill `out` with the driver pipeline cache blob and return the bytes
   written; faults INVALID_ARGUMENT if `out` is smaller than the current
   blob. Feed the blob to RuntimeDesc.pipeline_cache_data before creating a
   later runtime and device. Must not be called concurrently with pipeline
   creation. *>
fn usz? get_pipeline_cache_data(Device* device, char[] out) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.vtable.get_pipeline_cache_data(device, out);
}

<* Create one fixed-capacity command allocator for an exact selected queue.
   Zero capacities default to 8 command buffers, 64 tracked references per list,
   and 4 generated-preprocess buffers per list; storage scales with their products.
   Creation may fault for invalid limits, host/device allocation failure, a full
   allocator table, device loss, or a busy/closing device. *>
fn CommandAllocator? create_command_allocator(
    Device* device,
    Queue queue,
    CommandAllocatorDesc* desc = null,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_queue(&operation, device, queue)!;

    CommandAllocatorDesc normalized = desc == null ? {} : *desc;
    if (normalized.command_buffer_capacity == 0) {
        normalized.command_buffer_capacity = DEFAULT_COMMAND_ALLOCATOR_CAPACITY;
    }
    if (normalized.max_resource_references_per_list == 0) {
        normalized.max_resource_references_per_list =
            DEFAULT_COMMAND_REFERENCES_PER_LIST;
    }
    if (normalized.max_generated_preprocess_buffers_per_list == 0) {
        normalized.max_generated_preprocess_buffers_per_list =
            DEFAULT_COMMAND_PREPROCESS_PER_LIST;
    }
    if (normalized.command_buffer_capacity > MAX_COMMAND_ALLOCATOR_CAPACITY
        || normalized.max_resource_references_per_list
            > MAX_COMMAND_REFERENCES_PER_LIST
        || normalized.max_generated_preprocess_buffers_per_list
            > MAX_COMMAND_PREPROCESS_PER_LIST) {
        return INVALID_ARGUMENT~;
    }
    if (normalized.command_buffer_capacity
        > uint::max
            / normalized.max_generated_preprocess_buffers_per_list) {
        return INVALID_ARGUMENT~;
    }

    CommandAllocator allocator = {
        .device = *device,
        .queue  = queue,
    };
    if (operation.vtable.create_command_allocator == null) {
        return BACKEND_ERROR~;
    }
    allocator.handle = operation.vtable.create_command_allocator(
        device,
        queue,
        &normalized,
    )!;
    ulong expected_owner = gpu::internal::resource_owner_tag(
        operation.data.resource_owner,
        gpu::internal::ResourceOwnerKind.COMMAND_ALLOCATOR,
    );
    if (!allocator.handle.is_valid()
        || allocator.handle.owner != expected_owner) {
        if (operation.vtable.destroy_command_allocator != null) {
            (void)operation.vtable.destroy_command_allocator(
                device,
                allocator.handle,
            );
        }
        return BACKEND_ERROR~;
    }
    gpu::internal::retain_device_children(&operation);
    return allocator;
}

<* Destroy a quiescent allocator. RESOURCE_IN_USE or DEVICE_BUSY leaves the
   allocator token unchanged and retryable. *>
fn void? destroy_command_allocator(CommandAllocator* allocator) {
    if (allocator == null || !allocator.handle.is_valid()) {
        return INVALID_HANDLE~;
    }
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(&allocator.device, true)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_command_allocator(&operation, allocator)!;
    if (operation.vtable.destroy_command_allocator == null) {
        return BACKEND_ERROR~;
    }
    operation.vtable.destroy_command_allocator(
        &allocator.device,
        allocator.handle,
    )!;
    *allocator = {};
    gpu::internal::release_device_children(&operation);
}

<* Reserve compatible generated-command preprocess storage on a quiescent
   allocator. The allocator's fixed slot and byte ceilings bound replacement. *>
fn void? reserve_generated_scratch(
    CommandAllocator* allocator,
    GeneratedScratchDesc* desc,
) {
    if (allocator == null) return INVALID_HANDLE~;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(&allocator.device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_command_allocator(&operation, allocator)!;
    if (desc == null) return INVALID_ARGUMENT~;
    gpu::internal::validate_resource_owner(
        &operation,
        desc.pipeline.owner,
        gpu::internal::ResourceOwnerKind.PIPELINE,
    )!;
    if (!gpu::internal::generated_work_kind_valid(desc.kind)
        || desc.max_commands_per_list == 0
        || desc.preprocess_buffer_count == 0
        || desc.max_commands_per_list
            > operation.data.caps.max_generated_work_count) {
        return INVALID_ARGUMENT~;
    }
    if (operation.vtable.reserve_generated_scratch == null) {
        return BACKEND_ERROR~;
    }
    return operation.vtable.reserve_generated_scratch(allocator, desc);
}

<* Release one generated-work reservation from a quiescent allocator. *>
fn void? release_generated_scratch(
    CommandAllocator* allocator,
    PipelineHandle pipeline,
    GeneratedWorkKind kind,
) {
    if (allocator == null) return INVALID_HANDLE~;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(&allocator.device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_command_allocator(&operation, allocator)!;
    gpu::internal::validate_resource_owner(
        &operation,
        pipeline.owner,
        gpu::internal::ResourceOwnerKind.PIPELINE,
    )!;
    if (!gpu::internal::generated_work_kind_valid(kind)) return INVALID_ARGUMENT~;
    if (operation.vtable.release_generated_scratch == null) {
        return BACKEND_ERROR~;
    }
    return operation.vtable.release_generated_scratch(
        allocator,
        pipeline,
        kind,
    );
}

<* Begin recording through an explicit allocator. The token retains its device until
   successful end followed by submit, or until discard. DEVICE_BUSY leaves the
   allocator unchanged when every preallocated command buffer is unavailable. *>
fn CommandList? begin_commands(CommandAllocator* allocator) {
    if (allocator == null) return INVALID_HANDLE~;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(&allocator.device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_command_allocator(&operation, allocator)!;
    gpu::internal::BackendVTable* vtable = operation.vtable;
    if (vtable.begin_commands == null) return BACKEND_ERROR~;
    CommandList commands;
    gpu::internal::CommandBackendContext backend;
    bool backend_created = false;
    bool lease_retained = false;
    defer catch {
        if (lease_retained) {
            (void)gpu::internal::release_command_pin_lease(&operation, &commands);
        }
        if (backend_created) {
            gpu::internal::note_command_lifecycle_vtable_dispatch();
            (void)vtable.discard_commands(
                &commands.device,
                commands.handle,
                false,
                null,
            );
        }
    }
    gpu::internal::note_command_lifecycle_vtable_dispatch();
    vtable.begin_commands(allocator, &commands, &backend)!;
    backend_created = true;
    if (!gpu::internal::retain_command_pin_lease(&operation, &commands)) {
        return INVALID_HANDLE~;
    }
    lease_retained = true;
    gpu::internal::publish_command_encoder(&operation, &commands, &backend)!;
    gpu::internal::retain_device_children(&operation);
    gpu::internal::transfer_device_operation_pin(&operation);
    backend_created = false;
    lease_retained = false;
    return commands;
}

<* Finish and consume a recording token. Failure leaves it recording. *>
fn ExecutableCommandList? end_commands(CommandList* commands) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    gpu::internal::DeviceOperation operation = gpu::internal::command_operation(commands)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::note_command_lifecycle_vtable_dispatch();
    operation.vtable.end_commands(encoder)!;
    gpu::internal::transfer_command_encoder_to_executable(encoder);
    ExecutableCommandList executable = {
        .device = commands.device,
        .handle = commands.handle,
        .encoder = commands.encoder,
    };
    *commands = {};
    return executable;
}

<* Consume an unfinished recording, including after device loss. *>
fn void? discard_commands(CommandList* commands) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(
        commands,
        count_command: false,
    )!;
    gpu::internal::DeviceOperation operation = gpu::internal::command_operation(
        commands,
        allow_lost: true,
    )!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::BackendVTable* vtable = operation.vtable;
    CommandList consumed = *commands;
    gpu::internal::note_command_lifecycle_vtable_dispatch();
    vtable.discard_commands(
        &consumed.device,
        consumed.handle,
        false,
        encoder,
    )!;
    *commands = {};
    bool released_lease = gpu::internal::release_command_pin_lease(&operation, &consumed);
    if (released_lease || !operation.owns_pin) {
        gpu::internal::release_device_children(&operation);
        (void)gpu::internal::release_retained_device_pins(&operation);
    }
}

<* Consume an executable token without submitting it. *>
fn void? discard_executable_commands(ExecutableCommandList* commands) {
    gpu::internal::CommandEncoderState* encoder =
        gpu::internal::executable_encoder(commands)!;
    gpu::internal::DeviceOperation operation = gpu::internal::executable_command_operation(
        commands,
        allow_lost: true,
    )!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::BackendVTable* vtable = operation.vtable;
    ExecutableCommandList consumed = *commands;
    gpu::internal::note_command_lifecycle_vtable_dispatch();
    vtable.discard_commands(
        &consumed.device,
        consumed.handle,
        true,
        encoder,
    )!;
    *commands = {};
    bool released_lease = gpu::internal::release_executable_command_pin_lease(
        &operation,
        &consumed,
    );
    if (released_lease || !operation.owns_pin) {
        gpu::internal::release_device_children(&operation);
        (void)gpu::internal::release_retained_device_pins(&operation);
    }
}

<* Record a buffer-to-buffer copy; transfer-side barriers are the caller's. *>
fn void? cmd_copy_buffer(CommandList* commands, BufferCopyDesc* desc) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    if (desc == null) return INVALID_ARGUMENT~;
    return encoder.ops.copy_buffer(encoder, desc);
}

<* Record a 32-bit-pattern fill over dst; its offset and size must be 4-byte
   multiples. *>
fn void? cmd_fill_buffer(CommandList* commands, GpuSpan dst, uint value) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.fill_buffer(encoder, dst, value);
}

<* The texture's effective layout must be TRANSFER_DST; the caller records
   the surrounding barriers. *>
fn void? cmd_copy_buffer_to_texture(CommandList* commands, BufferTextureCopyDesc* desc) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    if (desc == null) return INVALID_ARGUMENT~;
    return encoder.ops.copy_buffer_to_texture(encoder, desc);
}

<* The texture's effective layout must be TRANSFER_SRC. *>
fn void? cmd_copy_texture_to_buffer(CommandList* commands, TextureBufferCopyDesc* desc) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    if (desc == null) return INVALID_ARGUMENT~;
    return encoder.ops.copy_texture_to_buffer(encoder, desc);
}

<* Record a global semantic execution and memory barrier. *>
fn void? cmd_barrier(CommandList* commands, Barrier* barrier) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    if (barrier == null) return INVALID_ARGUMENT~;
    return encoder.ops.barrier(encoder, barrier);
}

<* Record an explicit semantic texture transition. *>
fn void? cmd_texture_barrier(CommandList* commands, TextureBarrier* barrier) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    if (barrier == null) return INVALID_ARGUMENT~;
    return encoder.ops.texture_barrier(encoder, barrier);
}

<* Bind the active compute or graphics pipeline for later execution commands.
   A failed bind leaves the previous active pipeline unchanged. *>
fn void? cmd_bind_pipeline(CommandList* commands, PipelineHandle pipeline) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.bind_pipeline(encoder, pipeline);
}

<* Dispatch the active compute pipeline with a single nonzero root GPU address
   pushed at offset 0. Every allocation reachable from the root must admit the
   recording queue's role. Counts may be zero and may not exceed
   DeviceCaps.max_compute_work_group_count on any axis. *>
fn void? cmd_dispatch(CommandList* commands, GpuAddress root, Vec3u groups) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.dispatch(encoder, root, groups);
}

<* Set separate dynamic depth state for subsequent draws in the active render
   pass. A failed update leaves the previous depth state unchanged. *>
fn void? cmd_set_depth_state(CommandList* commands, DepthState* depth) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    if (depth == null) return INVALID_ARGUMENT~;
    return encoder.ops.set_depth_state(encoder, depth);
}

<* Set pass-bounded topology, culling, front face, and depth bias for
   subsequent draws. Requires an active render pass. A failed update leaves
   the previous state unchanged. *>
fn void? cmd_set_raster_state(CommandList* commands, DynamicRasterState* raster) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    if (raster == null) return INVALID_ARGUMENT~;
    return encoder.ops.set_raster_state(encoder, raster);
}

<* Set one dynamic viewport for subsequent draws. Requires an active render
   pass. Device-unrepresentable or non-finite values and zero height fault
   INVALID_ARGUMENT; negative height and off-pass overscan are valid. *>
fn void? cmd_set_viewport(CommandList* commands, Viewport* viewport) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    if (viewport == null) return INVALID_ARGUMENT~;
    return encoder.ops.set_viewport(encoder, viewport);
}

<* Set one dynamic scissor for subsequent draws. Requires an active render
   pass. Negative or signed-overflowing fields fault INVALID_ARGUMENT;
   off-pass overscan and zero extent are valid. *>
fn void? cmd_set_scissor(CommandList* commands, ScissorRect* scissor) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    if (scissor == null) return INVALID_ARGUMENT~;
    return encoder.ops.set_scissor(encoder, scissor);
}

<* Draw inside a render pass, pushing the vertex and fragment root addresses.
   Every allocation reachable from either root must admit the recording queue's
   role. *>
fn void? cmd_draw(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, uint vertex_count, uint instance_count) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.draw(
        encoder:        encoder,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        vertex_count:   vertex_count,
        instance_count: instance_count,
    );
}

<* Draw indexed geometry; indices are read from the start of index_span.
   Every allocation reachable from either root must admit the recording queue's
   role. *>
fn void? cmd_draw_indexed(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan index_span, uint index_count, uint instance_count, IndexType index_type = IndexType.U32) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.draw_indexed(
        encoder:        encoder,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        index_span:     index_span,
        index_count:    index_count,
        instance_count: instance_count,
        index_type:     index_type,
    );
}

<* Draw from GPU-written DrawIndirectCommand arguments at the span's offset.
   The args span must support indirect-command reads. The root pair applies to
   every draw in the multi-draw (per-draw variation via gl_DrawID). Ordering
   against argument writes is the caller's barrier. Every allocation reachable
   from either root must admit the recording queue's role. draw_count may not
   exceed DeviceCaps.max_draw_indirect_count. *>
fn void? cmd_draw_indirect(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, uint draw_count) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.draw_indirect(
        encoder:       encoder,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        args:          args,
        draw_count:    draw_count,
    );
}

<* Indexed variant over DrawIndexedIndirectCommand arguments. Index-range
   bounds are read from GPU memory at execution time and are not validated;
   out-of-bounds indices follow device robustness. draw_count may not exceed
   DeviceCaps.max_draw_indirect_count. Every allocation reachable from either
   root must admit the recording queue's role. *>
fn void? cmd_draw_indexed_indirect(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, uint draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.draw_indexed_indirect(
        encoder:       encoder,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        args:          args,
        draw_count:    draw_count,
        index_span:    index_span,
        index_type:    index_type,
    );
}

<* Count-buffer variant: the GPU-written uint at count_span limits execution
   to max_draw_count and may not exceed DeviceCaps.max_draw_indirect_count.
   The args span must hold max_draw_count entries. Requires
   DeviceCaps.draw_indirect_count. Every allocation reachable from either root
   must admit the recording queue's role. *>
fn void? cmd_draw_indexed_indirect_count(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, GpuSpan count_span, uint max_draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.draw_indexed_indirect_count(
        encoder:        encoder,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        args:           args,
        count_span:     count_span,
        max_draw_count: max_draw_count,
        index_span:     index_span,
        index_type:     index_type,
    );
}

<* Dispatch from GPU-written DispatchIndirectCommand arguments. Each group
   count must not exceed the corresponding
   DeviceCaps.max_compute_work_group_count component. Every allocation
   reachable from the root must admit the recording queue's role. *>
fn void? cmd_dispatch_indirect(CommandList* commands, GpuAddress root, GpuSpan args) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.dispatch_indirect(encoder, root, args);
}

<* Execute up to max_draw_count GPU-written draw records. The GPU-written uint
   at count_span selects the actual count. Ordering against producer writes is
   the caller's barrier with hazards.draw_arguments set. *>
fn void? cmd_draw_generated(CommandList* commands, GpuSpan records, GpuSpan count_span, uint max_draw_count) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.draw_generated(
        encoder:        encoder,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
    );
}

<* Indexed generated-work variant. Index bounds remain GPU execution-time data. *>
fn void? cmd_draw_indexed_generated(CommandList* commands, GpuSpan records, GpuSpan count_span, uint max_draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.draw_indexed_generated(
        encoder:        encoder,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
        index_span:     index_span,
        index_type:     index_type,
    );
}

<* Execute up to max_dispatch_count GPU-written dispatch records. *>
fn void? cmd_dispatch_generated(CommandList* commands, GpuSpan records, GpuSpan count_span, uint max_dispatch_count) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.dispatch_generated(
        encoder:            encoder,
        records:            records,
        count_span:         count_span,
        max_dispatch_count: max_dispatch_count,
    );
}

<* Create immutable attachment metadata and any required native subresource
   view before recording. *>
fn AttachmentViewHandle? create_attachment_view(
    Device* device,
    AttachmentViewDesc* desc,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    gpu::internal::validate_resource_owner(
        &operation,
        desc.texture.owner,
        gpu::internal::ResourceOwnerKind.TEXTURE,
    )!;
    AttachmentViewHandle view = operation.vtable.create_attachment_view(
        device,
        desc,
    )!;
    gpu::internal::retain_device_children(&operation);
    return view;
}

<* Destroy an attachment view after every recorded and submitted use completes. *>
fn void? destroy_attachment_view(
    Device* device,
    AttachmentViewHandle view,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(
        &operation,
        view.owner,
        gpu::internal::ResourceOwnerKind.ATTACHMENT_VIEW,
    )!;
    operation.vtable.destroy_attachment_view(device, view)!;
    gpu::internal::release_device_children(&operation);
}

<* Begin a pass without adding synchronization. Viewport and scissor default to
   the full render area. *>
fn void? cmd_begin_render_pass(CommandList* commands, RenderPassDesc* desc) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    if (!gpu::internal::render_pass_desc_host_safe(desc)) return INVALID_ARGUMENT~;
    return encoder.ops.begin_render_pass(encoder, desc);
}

<* End the active render pass without adding synchronization. *>
fn void? cmd_end_render_pass(CommandList* commands) {
    gpu::internal::CommandEncoderState* encoder = gpu::internal::recording_encoder(commands)!;
    return encoder.ops.end_render_pass(encoder);
}

fn bool CompletionPoint.is_valid(self) => self.device.is_valid()
    && gpu::internal::completion_point_sequence(self) != 0;
fn bool CompletionPoint.equals(self, CompletionPoint other) @operator(==) {
    return self.device == other.device && self.payload == other.payload;
}

<* Construct a sampled, read-only texture state at `stages`. This inserts no
   synchronization; pass the result to a texture barrier. *>
fn TextureState sampled_at(StageMask stages) {
    return {
        .layout = TextureLayout.SAMPLED,
        .stages = stages,
        .access = { .read },
    };
}

<* Construct a storage texture state at `stages`, preserving `access`. This
   inserts no synchronization; pass the result to a texture barrier. *>
fn TextureState storage_at(StageMask stages, TextureAccess access) {
    return {
        .layout = TextureLayout.STORAGE,
        .stages = stages,
        .access = access,
    };
}

<* Construct a semantic transition for a selected texture range.
   `UNDEFINED` is source-only. *>
fn TextureBarrier? texture_view_transition(
    TextureHandle texture,
    TextureViewDesc view,
    TextureState before,
    TextureState after,
) {
    if (!gpu::internal::texture_layout_valid(before.layout)
        || !gpu::internal::texture_layout_valid(after.layout)
        || after.layout == TextureLayout.UNDEFINED) {
        return INVALID_ARGUMENT~;
    }
    return {
        .texture = texture,
        .view    = view,
        .before  = before,
        .after   = after,
    };
}

<* Construct a semantic transition for the whole texture.
   `UNDEFINED` is source-only. *>
fn TextureBarrier? texture_transition(
    TextureHandle texture,
    TextureState before,
    TextureState after,
) {
    return texture_view_transition(
        texture: texture,
        view:    {},
        before:  before,
        after:   after,
    );
}

<* Return whether the queue has completed this point. *>
fn bool? poll_completion(CompletionPoint point) {
    Device device = point.device;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(&device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_completion_point(&operation, &device, point)!;
    if (operation.vtable.poll_completion == null) return UNSUPPORTED_FEATURE~;
    return operation.vtable.poll_completion(
        &device,
        gpu::internal::completion_point_queue_id(point),
        gpu::internal::completion_point_sequence(point),
    );
}

<* Wait up to timeout_ns for the queue to complete this point. A timeout leaves
   the point unchanged and faults WAIT_TIMEOUT, so the wait may be retried. *>
fn void? wait_completion(
    CompletionPoint point,
    ulong timeout_ns = TIMEOUT_INFINITE,
) {
    Device device = point.device;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(&device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_completion_point(&operation, &device, point)!;
    if (operation.vtable.wait_completion == null) return UNSUPPORTED_FEATURE~;
    return operation.vtable.wait_completion(
        &device,
        gpu::internal::completion_point_queue_id(point),
        gpu::internal::completion_point_sequence(point),
        timeout_ns,
    );
}

fn bool Queue.is_valid(self) => self.device.is_valid();
fn bool Queue.equals(self, Queue other) @operator(==) {
    return self.device == other.device
        && self.id == other.id
        && (uint)self.roles == (uint)other.roles;
}

<* Return selected queue counts by semantic role; aliased roles count the same
   queue. INVALID_HANDLE if the device is not live. *>
fn QueueCounts? get_queue_counts(Device* device) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return gpu::internal::selected_queue_counts(&operation);
}

<* Return a device-owned queue identity for a semantic role and index. Aliased
   roles return the same token. A dead device faults INVALID_HANDLE; an
   unavailable index faults INVALID_ARGUMENT. *>
fn Queue? get_queue(Device* device, QueueKind kind, uint index = 0) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (index >= gpu::internal::selected_queue_count(&operation, kind)) return INVALID_ARGUMENT~;
    return {
        .device = *device,
        .id     = gpu::internal::selected_queue_id(&operation, kind, index),
        .roles  = gpu::internal::selected_queue_roles(&operation, kind, index),
    };
}

<* Return a live queue's device-local identity and supported roles.
   INVALID_HANDLE if the token is stale, foreign, or malformed. *>
fn QueueInfo? get_queue_info(Device* device, Queue queue) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_queue(&operation, device, queue)!;
    return {
        .id    = queue.id,
        .roles = queue.roles,
    };
}

<* Submit executable command lists to `queue` and return reusable progress.
   Successful submission consumes every list. *>
fn CompletionPoint? submit(Queue queue, SubmitDesc* desc) {
    Device device = queue.device;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(&device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_queue(&operation, &device, queue)!;
    if (desc == null) return INVALID_ARGUMENT~;
    if (desc.command_lists.len > gpu::internal::MAX_SUBMIT_COMMAND_LISTS) {
        return INVALID_ARGUMENT~;
    }
    if (desc.completion_waits.len > gpu::internal::MAX_SUBMIT_COMPLETION_WAITS) {
        return INVALID_ARGUMENT~;
    }
    foreach (&commands : desc.command_lists) {
        if (commands.device != device) return INVALID_HANDLE~;
        gpu::internal::executable_encoder(commands)!;
        gpu::internal::validate_resource_owner(
            &operation,
            commands.handle.owner,
            gpu::internal::ResourceOwnerKind.COMMAND,
        )!;
    }
    foreach (wait : desc.completion_waits) {
        gpu::internal::validate_completion_point(&operation, &device, wait.point)!;
        if (!gpu::internal::completion_wait_scope_valid(
            wait.before,
            wait.consumers,
        )) {
            return INVALID_ARGUMENT~;
        }
    }
    if (desc.readiness != SWAPCHAIN_READINESS_INVALID) {
        if (!desc.readiness.is_valid()
            || desc.readiness.device != device) {
            return INVALID_HANDLE~;
        }
        if (!queue.roles.graphics) return INVALID_ARGUMENT~;
        if (!gpu::internal::submit_wait_stage_mask_valid(desc.readiness_before)) {
            return INVALID_ARGUMENT~;
        }
    } else if ((uint)desc.readiness_before != 0) {
        return INVALID_ARGUMENT~;
    }
    gpu::internal::note_command_lifecycle_vtable_dispatch();
    CompletionPoint point = operation.vtable.submit(queue, desc)!;
    foreach (&commands : desc.command_lists) {
        gpu::internal::release_executable_command_pin_lease(&operation, commands);
        *commands = {};
    }
    gpu::internal::release_device_children(&operation, (ulong)desc.command_lists.len);
    (void)gpu::internal::release_retained_device_pins(
        &operation,
        (uint)desc.command_lists.len,
    );
    return point;
}

fn bool SwapchainReadiness.is_valid(self) => self.device.is_valid()
    && self.payload != 0;
fn bool SwapchainReadiness.equals(
    self,
    SwapchainReadiness other,
) @operator(==) {
    return self.device == other.device && self.payload == other.payload;
}

<* The surface remains owned by its runtime and must outlive the swapchain. *>
fn SwapchainHandle? create_swapchain(Device* device, Surface* surface, SwapchainDesc* desc) {
    if (surface == null || desc == null) return INVALID_ARGUMENT~;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::DeviceData* device_data = operation.data;
    gpu::internal::SurfaceData* surface_data = gpu::internal::resolve_surface(surface);
    if (surface_data == null) return INVALID_HANDLE~;
    if (!device_data.runtime_retained) return INVALID_RESOURCE_STATE~;
    if (!device_data.presentation_surface.is_valid()) return UNSUPPORTED_FEATURE~;
    if ((ulong)*surface != (ulong)device_data.presentation_surface) {
        return INVALID_HANDLE~;
    }

    gpu::internal::RuntimeData* device_runtime = gpu::internal::resolve_runtime(&device_data.runtime);
    gpu::internal::RuntimeData* surface_runtime = gpu::internal::resolve_runtime(&surface_data.runtime);
    if (device_runtime == null || surface_runtime != device_runtime) {
        return INVALID_HANDLE~;
    }
    SwapchainHandle swapchain = device_data.vtable.create_swapchain(device, surface, desc)!;
    gpu::internal::retain_device_children(&operation);
    return swapchain;
}

<* Counterpart of create_swapchain. Immediate: pending acquisition,
   presentation, command, or view use faults without consuming the handle. *>
fn void? destroy_swapchain(Device* device, SwapchainHandle swapchain) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    operation.vtable.destroy_swapchain(device, swapchain)!;
    gpu::internal::release_device_children(&operation);
}

<* Recreate under the same handle at a new extent. A zero clamped extent or
   a failed rebuild leaves the swapchain dormant (acquire faults
   SWAPCHAIN_OUT_OF_DATE) until a successful resize. Prior acquired-image
   handles become stale. Pending acquisition, presentation, command, or view
   use faults immediately and leaves the swapchain unchanged. *>
fn void? resize_swapchain(Device* device, SwapchainHandle swapchain, uint width, uint height) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.vtable.resize_swapchain(device, swapchain, width, height);
}

<* Actual format, extent, image count, and present mode selected for a
   swapchain. Dormant swapchains return the dormant sentinel snapshot. *>
fn SwapchainInfo? get_swapchain_info(Device* device, SwapchainHandle swapchain) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.vtable.get_swapchain_info(device, swapchain);
}

<* Acquire with a nonblocking default. Finite values are nanoseconds;
   TIMEOUT_INFINITE requires platform-guaranteed presentation progress.
   WAIT_TIMEOUT preserves the swapchain for retry. *>
fn AcquiredImage? acquire_next_image(
    Device* device,
    SwapchainHandle swapchain,
    ulong timeout_ns = 0,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.vtable.acquire_next_image(device, swapchain, timeout_ns);
}

<* Present and consume an acquired image after its readiness-consuming submit.
   `render_completion` must be the point returned by that submit. WAIT_TIMEOUT
   preserves the image for retry when private present completion is still busy. *>
fn void? present(
    Device* device,
    AcquiredImage* image,
    CompletionPoint render_completion,
) {
    if (image == null) return INVALID_ARGUMENT~;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_completion_point(&operation, device, render_completion)!;
    operation.vtable.present(device, image, render_completion)!;
    *image = {};
}

<* Present modes the swapchain's surface supports. Unsupported requested
   modes fall back to FIFO at creation; query this to pick deliberately. *>
fn PresentModeSupport? get_present_mode_support(Device* device, SwapchainHandle swapchain) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.vtable.get_present_mode_support(device, swapchain);
}
