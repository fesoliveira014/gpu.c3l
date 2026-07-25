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

<* Tests only whether this value differs from the invalid token sentinel.
   This does not validate live provenance or recording phase. *>
fn bool CommandList.is_valid(&self) =>
    self.record != null
        && self.generation != 0
        && self.device_identity != 0;
<* Tests only whether this value differs from the invalid token sentinel.
   This does not validate live provenance or executable phase. *>
fn bool ExecutableCommandList.is_valid(&self) =>
    self.record != null
        && self.generation != 0
        && self.device_identity != 0;

<* True for a live heap index; the zero value is the invalid sentinel. *>
fn bool TextureIndex.is_valid(self) => (uint)self != 0;
<* True for a live heap index; the zero value is the invalid sentinel. *>
fn bool SamplerIndex.is_valid(self) => (uint)self != 0;

<* True when the token is not the invalid zero value. *>
fn bool Runtime.is_valid(self) => (ulong)self != 0;

<* Return FULL contract diagnostics with Vulkan validation layers enabled. *>
fn RuntimeDesc full_validation_runtime_desc() => {
    .contract_validation = ContractValidation.FULL,
    .enable_vulkan_validation = true,
};

<* Initialize backend discovery. Importing gpu performs no initialization. *>
fn Runtime? create_runtime(RuntimeDesc* desc) {
    gpu::internal::validate_runtime_desc(desc)!;
    return gpu::internal::vk::create_vk_runtime(desc);
}

<* Destroy a runtime.
   Preconditions: every surface and device owned by it is destroyed.
   Always checked: live identity and dependent counts.
   Runtime failures: RESOURCE_IN_USE preserves the runtime for retry. *>
fn void? destroy_runtime(Runtime* runtime) {
    gpu::internal::RuntimeData* data = gpu::internal::resolve_runtime(runtime);
    if (data == null) return INVALID_HANDLE~;
    if (data.live_surface_count > 0 || data.live_device_count.load(AtomicOrdering.ACQUIRE) > 0) {
        return RESOURCE_IN_USE~;
    }

    gpu::internal::vk::VkRuntimeState* state = data.state;
    data.state = null;
    data.adapter_count = 0;
    data.used = false;
    *runtime = RUNTIME_INVALID;
    gpu::internal::vk::destroy_vk_runtime_state(state);
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
    return gpu::internal::vk::vk_get_adapter_info(
        data.state,
        adapter.adapter_index,
    );
}

<* Query backend and driver identity for diagnostics and logging. *>
fn AdapterDiagnostics? get_adapter_diagnostics(Adapter* adapter) {
    gpu::internal::RuntimeData* data = gpu::internal::resolve_adapter(adapter);
    if (data == null) return INVALID_HANDLE~;
    return gpu::internal::vk::vk_get_adapter_diagnostics(
        data.state,
        adapter.adapter_index,
    );
}

<* Query semantic descriptor support without enabling device state.
   unmet_requirement is a borrowed static label. *>
fn DeviceSupport? supports_device_desc(
    Adapter* adapter,
    DeviceDesc* desc = null,
) {
    DeviceDesc normalized = gpu::internal::normalize_device_desc(desc)!;
    gpu::internal::RuntimeData* data = gpu::internal::resolve_adapter(adapter);
    if (data == null) return INVALID_HANDLE~;
    return gpu::internal::evaluate_device_desc_support(
        data,
        adapter,
        normalized,
    );
}

fn bool Surface.is_valid(self) => (ulong)self != 0;

<* Return whether an adapter can present to a surface from the same runtime. *>
fn bool? supports_presentation(Adapter* adapter, Surface* surface) {
    gpu::internal::RuntimeData* adapter_runtime = gpu::internal::resolve_adapter(adapter);
    gpu::internal::SurfaceData* surface_data = gpu::internal::resolve_surface(surface);
    if (adapter_runtime == null || surface_data == null) return INVALID_HANDLE~;

    gpu::internal::RuntimeData* surface_runtime = gpu::internal::resolve_runtime(&surface_data.runtime);
    if (surface_runtime != adapter_runtime) return INVALID_HANDLE~;
    return gpu::internal::vk::vk_supports_presentation(
        adapter_runtime.state,
        adapter.adapter_index,
        surface_data.backend_surface,
    );
}

<* Destroy a surface.
   Preconditions: every borrowing swapchain is destroyed.
   Always checked: live identity, runtime ownership, and dependent count.
   Runtime failures: RESOURCE_IN_USE preserves the surface for retry. *>
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

    gpu::internal::vk::vk_destroy_surface(
        runtime_data.state,
        backend_surface,
    );
    gpu::internal::runtime_release_surface(&runtime)!!;
}

<* Reset process-wide command instrumentation before a synchronized test or
   benchmark interval. Do not call while another thread records commands. *>
fn void reset_command_resolution_stats() @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS)) {
    gpu::internal::command_device_registry_resolutions.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_retained_pin_operations.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_table_lookups.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_pipeline_table_lookups.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_pipeline_cache_lookups.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_recording_commands.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_native_commands.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_pipeline_bind_commands.store(0, AtomicOrdering.RELAXED);
    gpu::internal::command_descriptor_heap_set_bind_commands.store(0, AtomicOrdering.RELAXED);
}

<* Snapshot process-wide command instrumentation with relaxed loads.
   recording_commands counts live-record entry-point attempts;
   native_commands counts every Vulkan command emitted by those paths. *>
fn CommandResolutionStats command_resolution_stats() @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS)) {
    return {
        .device_registry_resolutions = gpu::internal::command_device_registry_resolutions.load(
            AtomicOrdering.RELAXED,
        ),
        .retained_pin_operations = gpu::internal::command_retained_pin_operations.load(
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
        .recording_commands = gpu::internal::command_recording_commands.load(
            AtomicOrdering.RELAXED,
        ),
        .native_commands = gpu::internal::command_native_commands.load(
            AtomicOrdering.RELAXED,
        ),
        .pipeline_bind_commands = gpu::internal::command_pipeline_bind_commands.load(
            AtomicOrdering.RELAXED,
        ),
        .descriptor_heap_set_bind_commands = gpu::internal::command_descriptor_heap_set_bind_commands.load(
            AtomicOrdering.RELAXED,
        ),
    };
}

<* Snapshot the selected device's capabilities. *>
fn DeviceCaps? get_device_caps(Device* device) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return operation.data.caps;
}

<* Create a device from one exact borrowed adapter.
   Ownership: the caller must release the returned device with destroy_device. *>
fn Device? create_device(Adapter* adapter, DeviceDesc* desc = null) {
    DeviceDesc normalized = gpu::internal::normalize_device_desc(desc)!;
    gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_adapter(adapter);
    if (runtime_data == null) return INVALID_HANDLE~;

    DeviceSupport support = gpu::internal::evaluate_device_desc_support(
        data:                 runtime_data,
        adapter:              adapter,
        normalized:           normalized,
        creation_diagnostics: true,
    )!;
    if (!support.supported) return UNSUPPORTED_FEATURE~;
    gpu::internal::vk::validate_device_baseline_support(
        runtime_data.state,
        adapter.adapter_index,
    )!;
    if (!gpu::internal::device_slot_available()) return SLOT_TABLE_FULL~;

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

    device = gpu::internal::vk::create_vk_device(adapter, &normalized, &runtime)!;
    retain_transferred = true;
    return device;
}

<* Destroy a quiescent device.
   Preconditions: no live children, host operations, or incomplete queue work.
   Always checked: device identity, children, pins, closing state, and progress.
   Runtime failures: RESOURCE_IN_USE or DEVICE_BUSY preserves the token. *>
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
        if (!lost) {
            if (catch readiness_fault =
                gpu::internal::vk::vk_check_device_destroy(slot.data.state)) {
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

    gpu::internal::vk::vk_destroy_device(data.state);

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
    return gpu::internal::vk::vk_get_memory_stats(operation.state);
}

<* Build a human-readable memory report. Ownership: the caller frees the
   returned String. *>
fn String? build_memory_report(Device* device, bool detailed) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return gpu::internal::vk::vk_build_memory_report(operation.state, detailed);
}

<* Open a debug-utils label region; a no-op without debug utils.
   Preconditions: calls are balanced while recording.
   Always checked: token identity, authoritative phase, and non-null label. *>
fn void? cmd_begin_label(CommandList* commands, ZString label, float[4] color = {}) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (label == null) return INVALID_ARGUMENT~;
    return record.ops.begin_label(record, label, color);
}

<* Close a debug-utils label region.
   Preconditions: a matching label region is open.
   Always checked: token identity and authoritative recording phase. *>
fn void? cmd_end_label(CommandList* commands) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.end_label(record);
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
    GpuAllocation allocation =
        gpu::internal::vk::vk_allocate_memory(operation.state, &normalized)!;
    gpu::internal::retain_device_children(&operation);
    return allocation;
}

<* Free quiescent storage.
   Preconditions: no command, texture, or descriptor still uses the allocation.
   Always checked: live identity, ownership, and detected dependents.
   Runtime failures: faults preserve the owning token. *>
fn void? free_allocation(Device* device, GpuAllocation* allocation) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (allocation == null || !allocation.is_valid()) return INVALID_HANDLE~;
    gpu::internal::validate_resource_owner(
        &operation,
        allocation.owner,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    )!;
    gpu::internal::vk::vk_free_allocation(operation.state, *allocation)!;
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
    return gpu::internal::vk::vk_get_allocation_info(operation.state, allocation);
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
    return gpu::internal::vk::vk_get_allocation_span(operation.state, allocation);
}

<* Borrow the CPU mapping for a live host-visible span. *>
fn char[]? get_span_mapping(Device* device, GpuSpan span) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return gpu::internal::vk::vk_get_span_mapping(operation.state, span);
}

<* Return the stable nonzero address of a live span. *>
fn GpuAddress? get_span_address(Device* device, GpuSpan span) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return gpu::internal::vk::vk_get_span_address(operation.state, span);
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
    return gpu::internal::vk::vk_flush_mapped_span(operation.state, span);
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
    return gpu::internal::vk::vk_invalidate_mapped_span(operation.state, span);
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
    return gpu::internal::vk::vk_get_texture_format_support(operation.state, format);
}

<* Return whether an exact descriptor is creatable by the backend and adapter.
   This checks capabilities only; allocation can still fail for lack of memory. *>
fn bool? supports_texture_desc(Device* device, TextureDesc* desc) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    if (!gpu::internal::queue_access_is_valid(&operation, desc.access)) return false;
    return gpu::internal::vk::vk_supports_texture_desc(operation.state, desc);
}

<* Return exact storage requirements for a texture descriptor. *>
fn TextureRequirements? get_texture_requirements(Device* device, TextureDesc* desc) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    gpu::internal::validate_queue_access(&operation, desc.access)!;
    return gpu::internal::vk::vk_get_texture_requirements(operation.state, desc);
}

<* The handle owns the image; shader access goes through a TextureIndex.
   Ownership: release with destroy_texture. *>
fn TextureHandle? create_texture(Device* device, TextureDesc* desc) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    gpu::internal::validate_queue_access(&operation, desc.access)!;
    TextureHandle texture =
        gpu::internal::vk::vk_create_texture(operation.state, desc)!;
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
    TextureHandle texture = gpu::internal::vk::vk_create_placed_texture(
        state:     operation.state,
        desc:      desc,
        placement: allocation,
        offset:    offset,
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
    DedicatedTexture created = gpu::internal::vk::vk_create_dedicated_texture(
        operation.state,
        desc,
        &normalized,
    )!;
    gpu::internal::retain_device_children(&operation, 2);
    return created;
}

<* Destroy a texture after its descriptors and command use complete.
   Always checked: live identity, ownership, and detected dependents.
   Runtime failures: faults preserve the handle; placed storage stays caller-owned. *>
fn void? destroy_texture(Device* device, TextureHandle texture) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(&operation, texture.owner, gpu::internal::ResourceOwnerKind.TEXTURE)!;
    gpu::internal::vk::vk_destroy_texture(operation.state, texture)!;
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
    gpu::internal::validate_sampler_desc(&operation, desc)!;
    return gpu::internal::vk::vk_intern_sampler(operation.state, desc);
}

<* Publish one texture view in the shader-visible heap. The texture must carry
   sampled or storage usage. The returned
   owner-bearing view contains the raw generation-free index stored in shader data. *>
fn TextureView? create_texture_view(
    Device* device,
    TextureHandle texture,
    TextureViewDesc* desc,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(&operation, texture.owner, gpu::internal::ResourceOwnerKind.TEXTURE)!;
    TextureView view =
        gpu::internal::vk::vk_create_texture_view(operation.state, texture, desc)!;
    gpu::internal::retain_device_children(&operation);
    return view;
}

<* Release one texture view and recycle its shader-visible index immediately.
   Preconditions: GPU references to the index are complete.
   Always checked: live identity and ownership.
   Runtime failures: faults preserve the view. *>
fn void? destroy_texture_view(Device* device, TextureView view) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(&operation, view.owner, gpu::internal::ResourceOwnerKind.TEXTURE_VIEW)!;
    gpu::internal::vk::vk_destroy_texture_view(operation.state, view)!;
    gpu::internal::release_device_children(&operation);
}

<* Batch-publish texture views.
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
    gpu::internal::vk::vk_create_texture_views(operation.state, descs, out_views)!;
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

<* Construct disabled blending with the selected write mask. *>
fn ColorTargetState color_blend_disabled(
    ColorWriteMask write_mask = COLOR_WRITE_ALL,
) {
    return { .write_mask = write_mask };
}

<* Construct conventional source-alpha blending with RGBA writes. *>
fn ColorTargetState alpha_blend() {
    return {
        .blend = {
            .enable    = true,
            .src_color = BlendFactor.SRC_ALPHA,
            .dst_color = BlendFactor.ONE_MINUS_SRC_ALPHA,
            .color_op  = BlendOp.ADD,
            .src_alpha = BlendFactor.ONE,
            .dst_alpha = BlendFactor.ONE_MINUS_SRC_ALPHA,
            .alpha_op  = BlendOp.ADD,
        },
        .write_mask = COLOR_WRITE_ALL,
    };
}

<* Construct premultiplied-alpha blending with RGBA writes. *>
fn ColorTargetState premultiplied_alpha_blend() {
    return {
        .blend = {
            .enable    = true,
            .src_color = BlendFactor.ONE,
            .dst_color = BlendFactor.ONE_MINUS_SRC_ALPHA,
            .color_op  = BlendOp.ADD,
            .src_alpha = BlendFactor.ONE,
            .dst_alpha = BlendFactor.ONE_MINUS_SRC_ALPHA,
            .alpha_op  = BlendOp.ADD,
        },
        .write_mask = COLOR_WRITE_ALL,
    };
}

<* Construct additive blending with RGBA writes. *>
fn ColorTargetState additive_blend() {
    return {
        .blend = {
            .enable    = true,
            .src_color = BlendFactor.ONE,
            .dst_color = BlendFactor.ONE,
            .color_op  = BlendOp.ADD,
            .src_alpha = BlendFactor.ONE,
            .dst_alpha = BlendFactor.ONE,
            .alpha_op  = BlendOp.ADD,
        },
        .write_mask = COLOR_WRITE_ALL,
    };
}

<* Fill caller-owned targets with one state and return a borrowed packet. *>
fn ColorState uniform_color_state(
    ColorTargetState[] targets,
    ColorTargetState state,
) {
    foreach (&target : targets) *target = state;
    return { .targets = targets };
}

<* Create a compute pipeline. Identical descriptors may alias one backend
   pipeline under distinct handles. Ownership: release with destroy_pipeline,
   once per create. *>
fn PipelineHandle? create_compute_pipeline(Device* device, ComputePipelineDesc* desc) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    PipelineHandle pipeline =
        gpu::internal::vk::vk_create_compute_pipeline(operation.state, desc)!;
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
    PipelineHandle pipeline =
        gpu::internal::vk::vk_create_graphics_pipeline(operation.state, desc)!;
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
    gpu::internal::vk::vk_create_compute_pipelines(
        operation.state,
        descs,
        out_pipelines,
    )!;
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
    gpu::internal::vk::vk_create_graphics_pipelines(
        operation.state,
        descs,
        out_pipelines,
    )!;
    for (usz i = 0; i < descs.len; i++) gpu::internal::retain_device_children(&operation);
}

<* Release one returned pipeline handle.
   Preconditions: recorded and submitted use of this handle is complete.
   Always checked: live identity, ownership, and detected command use.
   Runtime failures: faults preserve this alias; shared backend state is refcounted. *>
fn void? destroy_pipeline(Device* device, PipelineHandle pipeline) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_resource_owner(&operation, pipeline.owner, gpu::internal::ResourceOwnerKind.PIPELINE)!;
    gpu::internal::vk::vk_destroy_pipeline(operation.state, pipeline)!;
    gpu::internal::release_device_children(&operation);
}

<* Byte size of the driver pipeline cache blob. The cache grows as pipelines
   are built — re-query before each export. Must not be called concurrently
   with pipeline creation. *>
fn usz? get_pipeline_cache_size(Device* device) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return gpu::internal::vk::vk_get_pipeline_cache_size(operation.state);
}

<* Fill `out` with the driver pipeline cache blob and return the bytes
   written; faults INVALID_ARGUMENT if `out` is smaller than the current
   blob. Feed the blob to RuntimeDesc.pipeline_cache_data before creating a
   later runtime and device. Must not be called concurrently with pipeline
   creation. *>
fn usz? get_pipeline_cache_data(Device* device, char[] out) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return gpu::internal::vk::vk_get_pipeline_cache_data(operation.state, out);
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
    allocator.handle = gpu::internal::vk::vk_create_command_allocator(
        operation.state,
        queue,
        &normalized,
    )!;
    ulong expected_owner = gpu::internal::resource_owner_tag(
        operation.data.resource_owner,
        gpu::internal::ResourceOwnerKind.COMMAND_ALLOCATOR,
    );
    if (!allocator.handle.is_valid()
        || allocator.handle.owner != expected_owner) {
        (void)gpu::internal::vk::vk_destroy_command_allocator(
            operation.state,
            allocator.handle,
        );
        return BACKEND_ERROR~;
    }
    gpu::internal::retain_device_children(&operation);
    return allocator;
}

<* Destroy a quiescent allocator.
   Preconditions: all recording, executable, and submitted units are retired.
   Always checked: identity, ownership, and allocator lifecycle.
   Runtime failures: RESOURCE_IN_USE or DEVICE_BUSY preserves the token. *>
fn void? destroy_command_allocator(CommandAllocator* allocator) {
    if (allocator == null || !allocator.handle.is_valid()) {
        return INVALID_HANDLE~;
    }
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(&allocator.device, true)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_command_allocator(&operation, allocator)!;
    gpu::internal::vk::vk_destroy_command_allocator(
        operation.state,
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
    return gpu::internal::vk::vk_reserve_generated_scratch(
        operation.state,
        allocator,
        desc,
    );
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
    return gpu::internal::vk::vk_release_generated_scratch(
        state:     operation.state,
        allocator: allocator,
        pipeline:  pipeline,
        kind:      kind,
    );
}

<* Begin recording through an explicit allocator.
   Preconditions: allocator use follows its threading and queue contract.
   Always checked: live identity, ownership, and fixed-unit availability.
   Runtime failures: DEVICE_BUSY preserves the allocator; success retains its device. *>
fn CommandList? begin_commands(CommandAllocator* allocator) {
    if (allocator == null) return INVALID_HANDLE~;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(&allocator.device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_command_allocator(&operation, allocator)!;
    CommandList commands;
    gpu::internal::vk::vk_begin_commands(
        operation.state,
        allocator,
        &commands,
    )!;
    gpu::internal::retain_device_children(&operation);
    gpu::internal::transfer_device_operation_pin(&operation);
    return commands;
}

<* Finish and consume a recording token.
   Preconditions: the command list is recording outside a render pass.
   Always checked: token identity and authoritative phase.
   Runtime failures: failure leaves the token recording. *>
fn ExecutableCommandList? end_commands(CommandList* commands) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    gpu::internal::DeviceOperation operation =
        gpu::internal::command_operation(record)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::vk::vk_end_commands(operation.state, record)!;
    ExecutableCommandList executable;
    gpu::internal::publish_executable_command_list(record, &executable);
    *commands = {};
    return executable;
}

<* Consume an unfinished recording, including after device loss.
   Preconditions: aliases are thread-confined and have not been consumed.
   Always checked: token identity and authoritative recording phase.
   Runtime failures: failure preserves the token. *>
fn void? discard_commands(CommandList* commands) {
    gpu::internal::vk::CommandRecord* record =
        gpu::internal::recording_command(commands, count_command: false)!;
    gpu::internal::DeviceOperation operation = gpu::internal::command_operation(
        record,
        allow_lost: true,
    )!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::vk::vk_discard_commands(
        operation.state,
        record,
        false,
    )!;
    *commands = {};
    gpu::internal::release_device_children(&operation);
    (void)gpu::internal::release_retained_device_pins(&operation);
}

<* Consume an executable token without submitting it.
   Preconditions: aliases are thread-confined and have not been consumed.
   Always checked: token identity and authoritative executable phase.
   Runtime failures: failure preserves the token. *>
fn void? discard_executable_commands(ExecutableCommandList* commands) {
    gpu::internal::vk::CommandRecord* record =
        gpu::internal::executable_command(commands)!;
    gpu::internal::DeviceOperation operation = gpu::internal::command_operation(
        record,
        allow_lost: true,
    )!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::vk::vk_discard_commands(
        operation.state,
        record,
        true,
    )!;
    *commands = {};
    gpu::internal::release_device_children(&operation);
    (void)gpu::internal::release_retained_device_pins(&operation);
}

<* Record a buffer-to-buffer copy; surrounding barriers are caller-owned.
   Preconditions: usage and queue access are valid.
   Always checked: token phase, required pointer, identity, bounds, and safe lowering.
   FULL diagnostics: usage and queue-access misuse. *>
fn void? cmd_copy_buffer(CommandList* commands, BufferCopyDesc* desc) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (desc == null) return INVALID_ARGUMENT~;
    return record.ops.copy_buffer(record, desc);
}

<* Record a 32-bit-pattern fill over dst.
   Preconditions: usage, queue access, and surrounding barriers are valid.
   Always checked: token phase, identity, bounds, alignment, and safe lowering.
   FULL diagnostics: usage and queue-access misuse. *>
fn void? cmd_fill_buffer(CommandList* commands, GpuSpan dst, uint value) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.fill_buffer(record, dst, value);
}

<* Record a buffer-to-texture copy; the effective layout is TRANSFER_DST.
   Preconditions: descriptor resources, usage, queue access, and barriers are valid.
   Always checked: token phase, required pointer, identities, format support,
   subresource/extent bounds, byte-size overflow, alignment, and safe lowering.
   FULL diagnostics: usage and queue-access misuse. *>
fn void? cmd_copy_buffer_to_texture(CommandList* commands, BufferTextureCopyDesc* desc) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (desc == null) return INVALID_ARGUMENT~;
    return record.ops.copy_buffer_to_texture(record, desc);
}

<* Record a texture-to-buffer copy; the effective layout is TRANSFER_SRC.
   Preconditions: descriptor resources, usage, queue access, and barriers are valid.
   Always checked: token phase, required pointer, identities, format support,
   subresource/extent bounds, byte-size overflow, alignment, and safe lowering.
   FULL diagnostics: usage and queue-access misuse. *>
fn void? cmd_copy_texture_to_buffer(CommandList* commands, TextureBufferCopyDesc* desc) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (desc == null) return INVALID_ARGUMENT~;
    return record.ops.copy_texture_to_buffer(record, desc);
}

<* Record a global semantic execution and memory barrier.
   Preconditions: stages and hazards are known, consistent, and queue-supported.
   Always checked: token identity, authoritative phase, and required pointer.
   FULL diagnostics: stage, hazard, and queue-support misuse. *>
fn void? cmd_barrier(CommandList* commands, Barrier* barrier) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (barrier == null) return INVALID_ARGUMENT~;
    return record.ops.barrier(record, barrier);
}

<* Record an explicit semantic texture transition.
   Preconditions: the texture, range, states, usage, and queue access are valid.
   Always checked: token phase, required pointer, texture identity, range, and safe lowering.
   FULL diagnostics: state, usage, queue, format, and presentation misuse. *>
fn void? cmd_texture_barrier(CommandList* commands, TextureBarrier* barrier) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (barrier == null) return INVALID_ARGUMENT~;
    return record.ops.texture_barrier(record, barrier);
}

<* Bind the active compute or graphics pipeline for later commands.
   Preconditions: the pipeline is live and valid for later use on this queue.
   Always checked: token phase, pipeline identity, ownership, descriptor-heap
   binding prerequisites, and a safe native snapshot.
   FULL diagnostics: render-pass compatibility when binding during a pass;
   failure preserves the prior logical bind. *>
fn void? cmd_bind_pipeline(CommandList* commands, PipelineHandle pipeline) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.bind_pipeline(record, pipeline);
}

<* Dispatch with one root passed unchanged at offset 0; zero is valid.
   Preconditions: a compute pipeline is active, group counts fit DeviceCaps,
   root-reachable allocations admit the queue, and shaders branch before
   dereferencing zero unless relying on defined robustness.
   Always checked: token identity, authoritative recording phase, and a
   usable bound-pipeline snapshot.
   FULL diagnostics: active-kind and group-limit misuse. *>
fn void? cmd_dispatch(CommandList* commands, GpuAddress root, Vec3u groups) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.dispatch(record, root, groups);
}

<* Set dynamic depth state before or during a pass.
   Preconditions: the command list is graphics-capable and the enum is valid.
   Always checked: token phase and required pointer.
   FULL diagnostics: queue and enum misuse; failure preserves prior state. *>
fn void? cmd_set_depth_state(CommandList* commands, DepthState* depth) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (depth == null) return INVALID_ARGUMENT~;
    return record.ops.set_depth_state(record, depth);
}

<* Set raster state before or during a pass.
   Preconditions: the queue, enums, polygon mode, and depth-bias values are valid.
   Always checked: token phase and required pointer.
   FULL diagnostics: queue, enum, feature, and finite-value misuse. *>
fn void? cmd_set_raster_state(CommandList* commands, DynamicRasterState* raster) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (raster == null) return INVALID_ARGUMENT~;
    return record.ops.set_raster_state(record, raster);
}

<* Set one dynamic viewport before or during a pass.
   Preconditions: values are finite, representable, within device bounds, and height is nonzero.
   Always checked: token phase and required pointer.
   FULL diagnostics: queue and viewport-value misuse. *>
fn void? cmd_set_viewport(CommandList* commands, Viewport* viewport) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (viewport == null) return INVALID_ARGUMENT~;
    return record.ops.set_viewport(record, viewport);
}

<* Set one dynamic scissor before or during a pass; overscan is valid.
   Preconditions: origin and extent are nonnegative and signed-representable.
   Always checked: token phase and required pointer.
   FULL diagnostics: queue and scissor-value misuse. *>
fn void? cmd_set_scissor(CommandList* commands, ScissorRect* scissor) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (scissor == null) return INVALID_ARGUMENT~;
    return record.ops.set_scissor(record, scissor);
}

<* Draw with vertex and fragment roots passed unchanged; either may be zero.
   Preconditions: a compatible graphics pipeline, render pass, complete state,
   and queue-admitted root data; shaders make zero dereference safe.
   Always checked: token identity, authoritative render-pass phase, and a
   usable bound-pipeline snapshot.
   FULL diagnostics: pipeline-kind and state misuse. *>
fn void? cmd_draw(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, uint vertex_count, uint instance_count) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.draw(
        record:        record,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        vertex_count:   vertex_count,
        instance_count: instance_count,
    );
}

<* Draw indexed geometry with both roots passed unchanged; either may be zero.
   Preconditions: graphics state is valid, index_span and root data admit the
   queue, and shaders make zero dereference safe.
   Always checked: token phase, identities, index bounds, a usable bound-pipeline
   snapshot, and safe lowering.
   FULL diagnostics: pipeline-kind, state, usage, queue, and index-type misuse. *>
fn void? cmd_draw_indexed(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan index_span, uint index_count, uint instance_count, IndexType index_type = IndexType.U32) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.draw_indexed(
        record:        record,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        index_span:     index_span,
        index_count:    index_count,
        instance_count: instance_count,
        index_type:     index_type,
    );
}

<* Draw indirectly with one root pair passed unchanged to every draw; zero is valid.
   Preconditions: graphics state, args, count, queue access, and barriers are valid;
   root data admits the queue and shaders make zero dereference safe.
   Always checked: token phase, identities, bounds, overflow, a usable
   bound-pipeline snapshot, and safe lowering.
   FULL diagnostics: pipeline-kind, state, usage, queue, and count misuse. *>
fn void? cmd_draw_indirect(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, uint draw_count) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.draw_indirect(
        record:       record,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        args:          args,
        draw_count:    draw_count,
    );
}

<* Indexed indirect draw with one unchanged root pair; zero is valid.
   Preconditions: graphics state, spans, count, queue access, and barriers are
   valid; shader zero paths and GPU-written index bounds are caller-owned.
   Always checked: token phase, identities, host-visible bounds, overflow, a
   usable bound-pipeline snapshot, and safe lowering.
   FULL diagnostics: pipeline-kind, state, usage, queue, count, and index-type misuse. *>
fn void? cmd_draw_indexed_indirect(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, uint draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.draw_indexed_indirect(
        record:       record,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        args:          args,
        draw_count:    draw_count,
        index_span:    index_span,
        index_type:    index_type,
    );
}

<* Count-buffer indexed draw with one unchanged root pair; zero is valid.
   Preconditions: graphics state, spans, counts, queue access, barriers, shader
   zero paths, and GPU-written index bounds are valid.
   Always checked: token phase, identities, host-visible bounds, overflow, a
   usable bound-pipeline snapshot, and safe lowering.
   FULL diagnostics: pipeline-kind, state, usage, queue, count, and index-type misuse.
   Runtime failures: unsupported count execution faults UNSUPPORTED_FEATURE. *>
fn void? cmd_draw_indexed_indirect_count(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, GpuSpan count_span, uint max_draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.draw_indexed_indirect_count(
        record:        record,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        args:           args,
        count_span:     count_span,
        max_draw_count: max_draw_count,
        index_span:     index_span,
        index_type:     index_type,
    );
}

<* Dispatch indirectly with the root passed unchanged; zero is valid.
   Preconditions: compute state, args, queue access, barriers, and GPU-written
   group counts are valid; shaders make zero dereference safe.
   Always checked: token phase, identities, bounds, a usable bound-pipeline
   snapshot, and safe lowering.
   FULL diagnostics: pipeline kind, usage, queue, and span misuse. *>
fn void? cmd_dispatch_indirect(CommandList* commands, GpuAddress root, GpuSpan args) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.dispatch_indirect(record, root, args);
}

<* Execute GPU-written draw records, forwarding every embedded root unchanged;
   zero roots are valid.
   Preconditions: graphics state, spans, count, queue access, barriers, and
   shader zero paths are valid.
   Always checked: token phase, identities, bounds, capacity, a usable
   bound-pipeline snapshot, and safe lowering.
   FULL diagnostics: pipeline-kind, state, usage, queue, and count misuse.
   Runtime failures: missing capability faults UNSUPPORTED_FEATURE; unavailable
   reserved scratch faults GENERATED_SCRATCH_EXHAUSTED. *>
fn void? cmd_draw_generated(CommandList* commands, GpuSpan records, GpuSpan count_span, uint max_draw_count) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.draw_generated(
        record:        record,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
    );
}

<* Execute indexed GPU-written records, forwarding every embedded root unchanged;
   zero roots are valid.
   Preconditions: graphics state, spans, count, queue access, barriers, shader
   zero paths, and GPU-written index bounds are valid.
   Always checked: token phase, identities, bounds, capacity, a usable
   bound-pipeline snapshot, and safe lowering.
   FULL diagnostics: pipeline-kind, state, usage, queue, count, and index-type misuse.
   Runtime failures: missing capability faults UNSUPPORTED_FEATURE; unavailable
   reserved scratch faults GENERATED_SCRATCH_EXHAUSTED. *>
fn void? cmd_draw_indexed_generated(CommandList* commands, GpuSpan records, GpuSpan count_span, uint max_draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.draw_indexed_generated(
        record:        record,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
        index_span:     index_span,
        index_type:     index_type,
    );
}

<* Execute GPU-written dispatch records, forwarding every embedded root unchanged;
   zero roots are valid.
   Preconditions: compute state, spans, count, queue access, barriers, and
   shader zero paths are valid.
   Always checked: token phase, identities, bounds, capacity, a usable
   bound-pipeline snapshot, and safe lowering.
   FULL diagnostics: pipeline-kind, usage, queue, and count misuse.
   Runtime failures: missing capability faults UNSUPPORTED_FEATURE; unavailable
   reserved scratch faults GENERATED_SCRATCH_EXHAUSTED. *>
fn void? cmd_dispatch_generated(CommandList* commands, GpuSpan records, GpuSpan count_span, uint max_dispatch_count) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.dispatch_generated(
        record:            record,
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
    AttachmentViewHandle view = gpu::internal::vk::vk_create_attachment_view(
        operation.state,
        desc,
    )!;
    gpu::internal::retain_device_children(&operation);
    return view;
}

<* Destroy an attachment view.
   Preconditions: every recorded and submitted use is complete.
   Always checked: live identity, ownership, and detected command use.
   Runtime failures: faults preserve the view. *>
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
    gpu::internal::vk::vk_destroy_attachment_view(operation.state, view)!;
    gpu::internal::release_device_children(&operation);
}

<* Construct full-area viewport/scissor with zero raster/depth state.
   The returned color packet is empty and must be supplied separately for a
   color pipeline. Dimensions must be in [1, int::max]. *>
fn GraphicsState? render_geometry_state(uint width, uint height) {
    if (width == 0 || height == 0
        || width > (uint)int::max || height > (uint)int::max) {
        return INVALID_ARGUMENT~;
    }
    return {
        .viewport = {
            .width     = (float)width,
            .height    = (float)height,
            .max_depth = 1.0f,
        },
        .scissor = {
            .width  = (int)width,
            .height = (int)height,
        },
        .raster = {},
        .depth  = {},
    };
}

<* Begin a render pass without changing command-buffer graphics state.
   Preconditions: attachments, layouts, queue access, dimensions, and pass phase are valid.
   Always checked: token phase, host-safe descriptor shape, identities, and safe lowering.
   FULL diagnostics: attachment, usage, queue, compatibility, and render-area misuse. *>
fn void? cmd_begin_render_pass(CommandList* commands, RenderPassDesc* desc) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (!gpu::internal::render_pass_desc_host_safe(desc)) return INVALID_ARGUMENT~;
    return record.ops.begin_render_pass(record, desc);
}

<* Begin a render pass and set complete graphics state atomically.
   Preconditions: a compatible graphics pipeline is bound, and the pass and
   complete graphics state are valid.
   Always checked: token phase, host-safe inputs, identities, and safe lowering.
   FULL diagnostics: pipeline, pass, queue, attachment, and graphics-state misuse. *>
fn void? cmd_begin_render_pass_with_state(
    CommandList* commands,
    RenderPassDesc* desc,
    GraphicsState* state,
) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (!gpu::internal::render_pass_desc_host_safe(desc)
        || !gpu::internal::graphics_state_host_safe(state)) {
        return INVALID_ARGUMENT~;
    }
    return record.ops.begin_render_pass_with_state(record, desc, state);
}

<* Replace complete graphics state before or during a pass.
   Preconditions: a compatible graphics pipeline is bound, the queue is
   graphics-capable, and all state values are valid.
   Always checked: token phase, required host-safe input, and safe lowering.
   FULL diagnostics: pipeline, queue, enum, feature, and state-value misuse. *>
fn void? cmd_set_graphics_state(
    CommandList* commands,
    GraphicsState* state,
) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    if (!gpu::internal::graphics_state_host_safe(state)) return INVALID_ARGUMENT~;
    return record.ops.set_graphics_state(record, state);
}

<* Replace complete blend and write-mask state for the bound graphics pipeline.
   Preconditions: a compatible graphics pipeline is selected and values are valid.
   Always checked: token phase, host-safe input, and safe lowering.
   FULL diagnostics: pipeline, target-domain, enum, and blend misuse. *>
fn void? cmd_set_color_state(
    CommandList* commands,
    ColorState* state,
) {
    gpu::internal::vk::CommandRecord* record =
        gpu::internal::recording_command(commands)!;
    if (!gpu::internal::color_state_host_safe(state)) {
        return INVALID_ARGUMENT~;
    }
    return record.ops.set_color_state(record, state);
}

<* End the active render pass without adding synchronization.
   Preconditions: a render pass is active.
   Always checked: token identity and authoritative render-pass phase. *>
fn void? cmd_end_render_pass(CommandList* commands) {
    gpu::internal::vk::CommandRecord* record = gpu::internal::recording_command(commands)!;
    return record.ops.end_render_pass(record);
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

<* Return whether the queue has completed this point.
   Preconditions: the owning device remains live.
   Always checked: point identity, ownership, publication, and device state.
   Runtime failures: device loss or backend failure preserves the point. *>
fn bool? poll_completion(CompletionPoint point) {
    Device device = point.device;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(&device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_completion_point(&operation, &device, point)!;
    return gpu::internal::vk::vk_poll_completion(
        operation.state,
        gpu::internal::completion_point_queue_id(point),
        gpu::internal::completion_point_sequence(point),
    );
}

<* Wait up to timeout_ns for the queue to complete this point.
   Preconditions: the owning device remains live.
   Always checked: point identity, ownership, publication, and device state.
   Runtime failures: WAIT_TIMEOUT, device loss, or backend failure preserves the point. *>
fn void? wait_completion(
    CompletionPoint point,
    ulong timeout_ns = TIMEOUT_INFINITE,
) {
    Device device = point.device;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(&device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_completion_point(&operation, &device, point)!;
    return gpu::internal::vk::vk_wait_completion(
        state:      operation.state,
        queue_id:   gpu::internal::completion_point_queue_id(point),
        sequence:   gpu::internal::completion_point_sequence(point),
        timeout_ns: timeout_ns,
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

<* Submit executable command lists and return reusable queue progress.
   Preconditions: externally synchronize the queue; every list targets it and
   waits/readiness name valid scopes and live same-device values.
   Always checked: queue/point identity, limits, wait scopes, exact record queue,
   authoritative executable phase, duplicates, and all-or-nothing claim/rollback.
   Runtime failures: capacity, busy, device-loss, and backend faults preserve inputs;
   success consumes every list. *>
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
    CompletionPoint point = gpu::internal::vk::vk_submit(
        operation.state,
        queue,
        desc,
    )!;
    foreach (&commands : desc.command_lists) {
        *commands = {};
    }
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
    SwapchainHandle swapchain = gpu::internal::vk::vk_create_swapchain(
        operation.state,
        surface,
        desc,
    )!;
    gpu::internal::retain_device_children(&operation);
    return swapchain;
}

<* Destroy a swapchain without waiting.
   Preconditions: no pending acquisition, presentation, command, or view use.
   Always checked: live identity, ownership, and lifecycle state.
   Runtime failures: detected use faults without consuming the handle. *>
fn void? destroy_swapchain(Device* device, SwapchainHandle swapchain) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::vk::vk_destroy_swapchain(operation.state, swapchain)!;
    gpu::internal::release_device_children(&operation);
}

<* Recreate a quiescent swapchain under the same handle at a new extent.
   Preconditions: no pending acquisition, presentation, command, or view use.
   Always checked: live identity, ownership, lifecycle state, and extent safety.
   Runtime failures: failed or zero-clamped rebuild leaves it dormant; other
   faults preserve it, and successful resize stales prior image handles. *>
fn void? resize_swapchain(Device* device, SwapchainHandle swapchain, uint width, uint height) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return gpu::internal::vk::vk_resize_swapchain(
        state:  operation.state,
        handle: swapchain,
        width:  width,
        height: height,
    );
}

<* Actual format, extent, image count, and present mode selected for a
   swapchain. Dormant swapchains return the dormant sentinel snapshot. *>
fn SwapchainInfo? get_swapchain_info(Device* device, SwapchainHandle swapchain) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return gpu::internal::vk::vk_get_swapchain_info(
        operation.state,
        swapchain,
    );
}

<* Acquire with a nonblocking default; finite values are nanoseconds.
   Preconditions: externally synchronize the swapchain and ensure platform
   progress before using TIMEOUT_INFINITE.
   Always checked: live identity, ownership, and acquisition lifecycle.
   Runtime failures: WAIT_TIMEOUT and surface/swapchain faults preserve it. *>
fn AcquiredImage? acquire_next_image(
    Device* device,
    SwapchainHandle swapchain,
    ulong timeout_ns = 0,
) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return gpu::internal::vk::vk_acquire_next_image(
        operation.state,
        swapchain,
        timeout_ns,
    );
}

<* Present an acquired image after its readiness-consuming submit.
   Preconditions: externally synchronize the image and pass that submit's point.
   Always checked: image/point identity, ownership, publication, and lifecycle.
   Runtime failures: WAIT_TIMEOUT preserves the image; success consumes it. *>
fn void? present(
    Device* device,
    AcquiredImage* image,
    CompletionPoint render_completion,
) {
    if (image == null) return INVALID_ARGUMENT~;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    gpu::internal::validate_completion_point(&operation, device, render_completion)!;
    gpu::internal::vk::vk_present(
        operation.state,
        image,
        render_completion,
    )!;
    *image = {};
}

<* Present modes the swapchain's surface supports. Unsupported requested
   modes fall back to FIFO at creation; query this to pick deliberately. *>
fn PresentModeSupport? get_present_mode_support(Device* device, SwapchainHandle swapchain) {
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device)!;
    defer gpu::internal::end_device_operation(&operation);
    return gpu::internal::vk::vk_get_present_mode_support(
        operation.state,
        swapchain,
    );
}
