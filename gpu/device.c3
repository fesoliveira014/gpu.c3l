module gpu;

import gpu::vk @public;
import std::atomic;
import std::atomic::types;
import std::thread;
import std::core::mem;

// Backend dispatch functions. Command recording entries take the owner-bearing
// CommandList; all other entries take the validated device token.
alias CheckDeviceDestroyFn @private  = fn void? (void*);
alias DestroyDeviceFn @private       = fn void (void*);
alias GetMemoryStatsFn @private      = fn MemoryStats? (Device*);
alias BuildMemoryReportFn @private   = fn String? (Device*, bool detailed);
alias AllocateMemoryFn @private      = fn GpuAllocation? (Device*, AllocationDesc*);
alias FreeAllocationFn @private      = fn void? (Device*, GpuAllocation);
alias GetAllocationInfoFn @private   = fn AllocationInfo? (Device*, GpuAllocation);
alias GetAllocationSpanFn @private   = fn GpuSpan? (Device*, GpuAllocation);
alias GetSpanMappingFn @private      = fn char[]? (Device*, GpuSpan);
alias GetSpanAddressFn @private      = fn GpuAddress? (Device*, GpuSpan);
alias FlushSpanFn @private           = fn void? (Device*, GpuSpan);
alias InvalidateSpanFn @private      = fn void? (Device*, GpuSpan);
// Out-param form avoids a c3c 0.8.0 debug-info codegen failure.
alias BeginCommandsFn @private   = fn void? (Queue, CommandList* out_commands);
alias EndCommandsFn @private     = fn void? (CommandList*);
alias DiscardCommandsFn @private = fn void? (Device*, CommandListHandle, bool executable);
alias SubmitFn @private          = fn CompletionPoint? (Queue, SubmitDesc*);
alias PollCompletionFn @private      = fn bool? (Device*, uint, ulong);
alias WaitCompletionFn @private      = fn void? (Device*, uint, ulong, ulong);
alias CmdCopyBufferFn @private       = fn void? (CommandList*, BufferCopyDesc*);
alias CmdFillBufferFn @private       = fn void? (CommandList*, GpuSpan, uint);
alias CmdCopyBufferToTextureFn @private = fn void? (CommandList*, BufferTextureCopyDesc*);
alias CmdCopyTextureToBufferFn @private = fn void? (CommandList*, TextureBufferCopyDesc*);
alias CmdBufferBarrierFn @private    = fn void? (CommandList*, BufferBarrier*);
alias CmdGlobalBarrierFn @private    = fn void? (CommandList*, GlobalBarrier*);
alias CmdTextureBarrierFn @private   = fn void? (CommandList*, TextureBarrier*);
alias GetTextureFormatSupportFn @private = fn TextureFormatSupport? (Device*, Format);
alias SupportsTextureDescFn @private     = fn bool? (Device*, TextureDesc*);
alias GetTextureRequirementsFn @private  = fn TextureRequirements? (Device*, TextureDesc*);
alias CreateTextureFn @private           = fn TextureHandle? (Device*, TextureDesc*);
alias CreatePlacedTextureFn @private = fn TextureHandle? (
    Device* device,
    TextureDesc* desc,
    GpuAllocation allocation,
    usz offset,
);
alias CreateDedicatedTextureFn @private = fn DedicatedTexture? (
    Device* device,
    TextureDesc* desc,
    AllocationDesc* allocation_desc,
);
alias CreateTextureDescriptorFn @private  = fn TextureIndex? (Device*, TextureHandle, TextureViewDesc*);
alias DestroyTextureFn @private          = fn void? (Device*, TextureHandle);
alias DestroyTextureDescriptorFn @private = fn void? (Device*, TextureIndex);
alias CreateTextureDescriptorsFn @private = fn void? (Device*, TextureDescriptorDesc[], TextureIndex[]);
alias InternSamplerFn @private            = fn Sampler? (Device*, SamplerDesc*);
alias PublishSamplerFn @private           = fn SamplerIndex? (Device*, Sampler);
alias CreateShaderFn @private          = fn ShaderHandle? (Device*, ShaderDesc*);
alias DestroyShaderFn @private         = fn void? (Device*, ShaderHandle);
alias CreateComputePipelineFn @private = fn PipelineHandle? (Device*, ComputePipelineDesc*);
alias CreateGraphicsPipelineFn @private = fn PipelineHandle? (Device*, GraphicsPipelineDesc*);
alias DestroyPipelineFn @private       = fn void? (Device*, PipelineHandle);
alias GetPipelineCacheSizeFn @private  = fn usz? (Device*);
alias GetPipelineCacheDataFn @private  = fn usz? (Device*, char[]);
alias CmdDispatchFn @private           = fn void? (CommandList*, PipelineHandle, GpuAddress, Vec3u);
alias CmdBeginRenderPassFn @private    = fn void? (CommandList*, RenderPassDesc*);
alias CmdSetViewportFn @private        = fn void? (CommandList*, Viewport*);
alias CmdSetScissorFn @private         = fn void? (CommandList*, ScissorRect*);
alias CmdEndRenderPassFn @private      = fn void? (CommandList*);
alias CmdDrawFn @private               = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuAddress, uint, uint);
alias CmdDrawIndexedFn @private        = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuAddress, GpuSpan, uint, uint, IndexType);
alias CmdDrawIndirectFn @private       = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuAddress, GpuSpan, uint);
alias CmdDrawIndexedIndirectFn @private = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuAddress, GpuSpan, uint, GpuSpan, IndexType);
alias CmdDrawIndexedIndirectCountFn @private = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuAddress, GpuSpan, GpuSpan, uint, GpuSpan, IndexType);
alias CmdDispatchIndirectFn @private   = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuSpan);
alias CmdBeginLabelFn @private         = fn void? (CommandList*, ZString, float[4]);
alias CmdEndLabelFn @private           = fn void? (CommandList*);
alias CreateSwapchainFn @private       = fn SwapchainHandle? (Device*, Surface*, SwapchainDesc*);
alias DestroySwapchainFn @private      = fn void? (Device*, SwapchainHandle);
alias ResizeSwapchainFn @private       = fn void? (Device*, SwapchainHandle, uint, uint);
alias GetSwapchainInfoFn @private      = fn SwapchainInfo? (Device*, SwapchainHandle);
alias AcquireNextImageFn @private      = fn AcquiredImage? (Device*, SwapchainHandle);
alias PresentFn @private               = fn void? (Device*, AcquiredImage*, CompletionPoint);
alias GetPresentModeSupportFn @private = fn PresentModeSupport? (Device*, SwapchainHandle);

const uint MAX_DEVICES @private = 64;
const uint MAX_DEVICE_GENERATION @private = 0x00FF_FFFF;
const uint MAX_DEVICE_OPERATION_PINS @private = 0x3FFF_FFFF;
const uint MAX_DEVICE_OPERATION_PIN_ATTEMPTS @private = 64;
const uint MAX_DEVICE_COMMANDS @private = 4096;

const uint DEVICE_SLOT_FREE @private = 0;
const uint DEVICE_SLOT_LIVE @private = 1;
const uint DEVICE_SLOT_CLOSING @private = 2;
const uint DEVICE_SLOT_RETIRED @private = 3;

bitstruct DeviceSlotControl : ulong @private {
    uint state      : 0..1;
    uint pins       : 2..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

// One immutable table per backend.
struct BackendVTable @private {
    CheckDeviceDestroyFn    check_device_destroy;
    DestroyDeviceFn         destroy_device;
    GetMemoryStatsFn        get_memory_stats;
    BuildMemoryReportFn     build_memory_report;
    AllocateMemoryFn        allocate_memory;
    FreeAllocationFn        free_allocation;
    GetAllocationInfoFn     get_allocation_info;
    GetAllocationSpanFn     get_allocation_span;
    GetSpanMappingFn        get_span_mapping;
    GetSpanAddressFn        get_span_address;
    FlushSpanFn             flush_mapped_span;
    InvalidateSpanFn        invalidate_mapped_span;
    BeginCommandsFn         begin_commands;
    EndCommandsFn           end_commands;
    DiscardCommandsFn       discard_commands;
    SubmitFn                submit;
    PollCompletionFn        poll_completion;
    WaitCompletionFn        wait_completion;
    CmdCopyBufferFn         cmd_copy_buffer;
    CmdFillBufferFn         cmd_fill_buffer;
    CmdCopyBufferToTextureFn cmd_copy_buffer_to_texture;
    CmdCopyTextureToBufferFn cmd_copy_texture_to_buffer;
    CmdBufferBarrierFn      cmd_buffer_barrier;
    CmdGlobalBarrierFn      cmd_global_barrier;
    CmdTextureBarrierFn     cmd_texture_barrier;
    GetTextureFormatSupportFn get_texture_format_support;
    SupportsTextureDescFn     supports_texture_desc;
    GetTextureRequirementsFn  get_texture_requirements;
    CreateTextureFn           create_texture;
    CreatePlacedTextureFn     create_placed_texture;
    CreateDedicatedTextureFn  create_dedicated_texture;
    DestroyTextureFn          destroy_texture;
    CreateTextureDescriptorFn  create_texture_descriptor;
    DestroyTextureDescriptorFn destroy_texture_descriptor;
    CreateTextureDescriptorsFn create_texture_descriptors;
    InternSamplerFn            intern_sampler;
    PublishSamplerFn           publish_sampler;
    CreateShaderFn          create_shader;
    DestroyShaderFn         destroy_shader;
    CreateComputePipelineFn create_compute_pipeline;
    CreateGraphicsPipelineFn create_graphics_pipeline;
    DestroyPipelineFn       destroy_pipeline;
    GetPipelineCacheSizeFn  get_pipeline_cache_size;
    GetPipelineCacheDataFn  get_pipeline_cache_data;
    CmdDispatchFn           cmd_dispatch;
    CmdBeginRenderPassFn    cmd_begin_render_pass;
    CmdSetViewportFn        cmd_set_viewport;
    CmdSetScissorFn         cmd_set_scissor;
    CmdEndRenderPassFn      cmd_end_render_pass;
    CmdDrawFn               cmd_draw;
    CmdDrawIndexedFn        cmd_draw_indexed;
    CmdDrawIndirectFn       cmd_draw_indirect;
    CmdDrawIndexedIndirectFn cmd_draw_indexed_indirect;
    CmdDrawIndexedIndirectCountFn cmd_draw_indexed_indirect_count;
    CmdDispatchIndirectFn   cmd_dispatch_indirect;
    CmdBeginLabelFn         cmd_begin_label;
    CmdEndLabelFn           cmd_end_label;
    CreateSwapchainFn       create_swapchain;
    DestroySwapchainFn      destroy_swapchain;
    ResizeSwapchainFn       resize_swapchain;
    GetSwapchainInfoFn      get_swapchain_info;
    AcquireNextImageFn      acquire_next_image;
    PresentFn               present;
    GetPresentModeSupportFn get_present_mode_support;
}

struct DeviceData @private {
    BackendKind    backend;
    DeviceCaps     caps;
    QueueSelection queues;
    BackendVTable* vtable;
    void*          backend_state;
    ulong          resource_owner;
    Runtime        runtime;
    Adapter        adapter;
    Surface        presentation_surface;
    bool           runtime_retained;
}

struct DeviceSlot @private {
    Atomic{ulong}                     control;
    Atomic{ulong}                     live_children;
    Atomic{ulong}                     command_owner;
    Atomic{bool}                      lost;
    Atomic{uint}[MAX_DEVICE_COMMANDS] command_leases;
    DeviceData                        data;
}

struct DeviceOperation @private {
    DeviceSlot*    slot;
    DeviceData*    data;
    BackendVTable* vtable;
    bool           active;
    bool           owns_pin;
}

enum ResourceOwnerKind @private {
    ALLOCATION,
    BUFFER,
    TEXTURE,
    TEXTURE_VIEW,
    SAMPLER,
    PIPELINE,
    SWAPCHAIN,
    COMMAND,
    SHADER,
    COMPLETION_POINT,
}

const ulong RESOURCE_OWNER_KIND_COUNT = (ulong)ResourceOwnerKind.COMPLETION_POINT + 1;
const ulong MAX_DEVICE_RESOURCE_OWNER = ulong::max / RESOURCE_OWNER_KIND_COUNT;

DeviceSlot[MAX_DEVICES] device_slots @private;
Atomic{bool} device_registry_lock @private;
Atomic{ulong} device_resource_owner_counter @private;

fn void lock_device_registry() @private {
    while (device_registry_lock.set(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
}

fn void unlock_device_registry() @private {
    (void)device_registry_lock.clear(AtomicOrdering.RELEASE);
}

fn ulong? allocate_device_resource_owner() @private {
    ulong current = device_resource_owner_counter.load(AtomicOrdering.RELAXED);
    while (true) {
        if (current == MAX_DEVICE_RESOURCE_OWNER) return SLOT_TABLE_FULL~;
        ulong seen = mem::compare_exchange(
            &device_resource_owner_counter.data,
            current,
            current + 1,
            AtomicOrdering.ACQUIRE_RELEASE,
            AtomicOrdering.RELAXED,
        );
        if (seen == current) return current + 1;
        current = seen;
    }
}

fn DeviceOperation? begin_device_operation(
    Device* device,
    bool allow_lost = false,
) @private {
    if (device == null || !device.is_valid()) return INVALID_HANDLE~;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return INVALID_HANDLE~;

    DeviceSlot* slot = &device_slots[device.index];
    ulong packed = slot.control.load(AtomicOrdering.ACQUIRE);
    for (uint attempt = 0; attempt < MAX_DEVICE_OPERATION_PIN_ATTEMPTS; attempt++) {
        DeviceSlotControl control = (DeviceSlotControl)packed;
        if (control.generation != device.generation) return INVALID_HANDLE~;
        if (control.state == DEVICE_SLOT_CLOSING) return DEVICE_BUSY~;
        if (control.state != DEVICE_SLOT_LIVE) return INVALID_HANDLE~;
        if (control.pins == MAX_DEVICE_OPERATION_PINS) return DEVICE_BUSY~;

        DeviceSlotControl pinned = control;
        pinned.pins++;
        ulong seen = mem::compare_exchange(
            &slot.control.data,
            packed,
            (ulong)pinned,
            AtomicOrdering.ACQUIRE_RELEASE,
            AtomicOrdering.ACQUIRE,
        );
        if (seen != packed) {
            packed = seen;
            continue;
        }
        if (!allow_lost && slot.lost.load(AtomicOrdering.ACQUIRE)) {
            DeviceSlotControl pin = { .pins = 1 };
            (void)slot.control.sub((ulong)pin, AtomicOrdering.RELEASE);
            return DEVICE_LOST~;
        }
        return {
            .slot     = slot,
            .data     = &slot.data,
            .vtable   = slot.data.vtable,
            .active   = true,
            .owns_pin = true,
        };
    }
    return DEVICE_BUSY~;
}

fn bool has_retained_command_pin_fields(
    Device device,
    CommandListHandle handle,
) @private {
    if (!device.is_valid()) return false;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return false;
    if (handle.generation == 0 || handle.index >= MAX_DEVICE_COMMANDS) {
        return false;
    }
    DeviceSlot* slot = &device_slots[device.index];
    if (slot.command_owner.load(AtomicOrdering.ACQUIRE) != handle.owner) {
        return false;
    }
    return slot.command_leases[handle.index].load(
        AtomicOrdering.ACQUIRE,
    ) == handle.generation;
}

fn bool has_retained_command_pin(CommandList* commands) @private {
    return commands != null
        && has_retained_command_pin_fields(commands.device, commands.handle);
}

fn bool has_retained_executable_command_pin(
    ExecutableCommandList* commands,
) @private {
    return commands != null
        && has_retained_command_pin_fields(commands.device, commands.handle);
}

fn DeviceOperation? borrow_retained_device_operation(
    Device* device,
    bool allow_lost = false,
) @private {
    if (device == null || !device.is_valid()) return INVALID_HANDLE~;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return INVALID_HANDLE~;

    DeviceSlot* slot = &device_slots[device.index];
    DeviceSlotControl control =
        (DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    if (control.generation != device.generation) return INVALID_HANDLE~;
    if (control.state == DEVICE_SLOT_CLOSING) return DEVICE_BUSY~;
    if (control.state != DEVICE_SLOT_LIVE || control.pins == 0) {
        return INVALID_HANDLE~;
    }
    if (!allow_lost && slot.lost.load(AtomicOrdering.ACQUIRE)) {
        return DEVICE_LOST~;
    }
    return {
        .slot   = slot,
        .data   = &slot.data,
        .vtable = slot.data.vtable,
        .active = true,
    };
}

fn bool retain_command_pin_lease(
    DeviceOperation* operation,
    CommandList* commands,
) @private {
    if (operation == null || operation.slot == null || commands == null) {
        return false;
    }
    if (commands.handle.generation == 0
        || commands.handle.index >= MAX_DEVICE_COMMANDS) {
        return false;
    }
    operation.slot.command_leases[commands.handle.index].store(
        commands.handle.generation,
        AtomicOrdering.RELEASE,
    );
    return true;
}

fn bool release_command_pin_lease_fields(
    DeviceOperation* operation,
    CommandListHandle handle,
) @private {
    if (operation == null || handle.index >= MAX_DEVICE_COMMANDS) return false;
    Atomic{uint}* lease = &operation.slot.command_leases[handle.index];
    uint seen = mem::compare_exchange(
        &lease.data,
        handle.generation,
        (uint)0,
        AtomicOrdering.ACQUIRE_RELEASE,
        AtomicOrdering.ACQUIRE,
    );
    return seen == handle.generation;
}

fn bool release_command_pin_lease(
    DeviceOperation* operation,
    CommandList* commands,
) @private {
    return commands != null
        && release_command_pin_lease_fields(operation, commands.handle);
}

fn bool release_executable_command_pin_lease(
    DeviceOperation* operation,
    ExecutableCommandList* commands,
) @private {
    return commands != null
        && release_command_pin_lease_fields(operation, commands.handle);
}

fn void end_device_operation(DeviceOperation* operation) @private {
    if (operation == null || !operation.active) return;

    if (operation.owns_pin) {
        DeviceSlotControl pin = { .pins = 1 };
        (void)operation.slot.control.sub((ulong)pin, AtomicOrdering.RELEASE);
    }
    operation.active = false;
    operation.owns_pin = false;
}

fn void transfer_device_operation_pin(DeviceOperation* operation) @private {
    if (operation == null || !operation.active) return;
    operation.owns_pin = false;
}

fn bool release_retained_device_pins(
    DeviceOperation* operation,
    uint count = 1,
) @private {
    if (operation == null || !operation.active || count == 0) return false;

    ulong packed = operation.slot.control.load(AtomicOrdering.ACQUIRE);
    while (true) {
        DeviceSlotControl control = (DeviceSlotControl)packed;
        uint operation_pins = operation.owns_pin ? 1 : 0;
        if (control.pins < count
            || control.pins - count < operation_pins) {
            return false;
        }

        DeviceSlotControl released = control;
        released.pins -= count;
        ulong seen = mem::compare_exchange(
            &operation.slot.control.data,
            packed,
            (ulong)released,
            AtomicOrdering.ACQUIRE_RELEASE,
            AtomicOrdering.ACQUIRE,
        );
        if (seen != packed) {
            packed = seen;
            continue;
        }
        return true;
    }
}

fn void retain_device_children(DeviceOperation* operation, ulong count = 1) @private {
    if (count == 0) return;
    (void)operation.slot.live_children.add(count, AtomicOrdering.RELEASE);
}

fn void release_device_children(DeviceOperation* operation, ulong count = 1) @private {
    if (count == 0) return;
    (void)operation.slot.live_children.sub(count, AtomicOrdering.RELEASE);
}

fn ulong resource_owner_tag(ulong device_owner, ResourceOwnerKind kind) @private {
    return (device_owner - 1) * RESOURCE_OWNER_KIND_COUNT + (ulong)kind + 1;
}

fn void? validate_resource_owner(
    DeviceOperation* operation,
    ulong owner,
    ResourceOwnerKind kind,
) @private {
    if (operation == null || !operation.active) return INVALID_HANDLE~;
    if (owner != resource_owner_tag(operation.data.resource_owner, kind)) {
        return INVALID_HANDLE~;
    }
}

<* Caller must hold an operation pin or be this device's destroying thread. *>
fn void mark_device_lost(Device* device) @private {
    if (device == null || !device.is_valid()) return;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return;

    DeviceSlot* slot = &device_slots[device.index];
    DeviceSlotControl control = (DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    if (control.generation != device.generation) return;
    if (control.state != DEVICE_SLOT_LIVE && control.state != DEVICE_SLOT_CLOSING) return;
    slot.lost.store(true, AtomicOrdering.RELEASE);
}

<* Caller must hold an operation pin for this device. *>
fn DeviceData* resolve_pinned_device(Device* device) @private {
    if (device == null || !device.is_valid()) return null;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return null;

    DeviceSlot* slot = &device_slots[device.index];
    DeviceSlotControl control = (DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    if (control.generation != device.generation) return null;
    if (control.state == DEVICE_SLOT_LIVE) return &slot.data;
    if (control.state == DEVICE_SLOT_CLOSING && control.pins > 0) return &slot.data;
    return null;
}

fn bool device_slot_available() @private {
    lock_device_registry();
    defer unlock_device_registry();
    for (uint i = 0; i < MAX_DEVICES; i++) {
        DeviceSlotControl control = (DeviceSlotControl)device_slots[i].control.load(AtomicOrdering.RELAXED);
        if (control.state == DEVICE_SLOT_FREE) return true;
    }
    return false;
}

fn Device? register_device(
    BackendKind backend,
    DeviceCaps caps,
    BackendVTable* vtable,
    void* backend_state,
    ulong resource_owner = 0,
    Runtime* retained_runtime = null,
    Adapter* adapter = null,
    Surface* presentation_surface = null,
    QueueSelection queues = {},
    Device* owner = null,
) @private {
    if (vtable == null || backend_state == null || vtable.destroy_device == null) return INVALID_ARGUMENT~;
    if (resource_owner == 0) resource_owner = allocate_device_resource_owner()!;
    bool has_runtime = retained_runtime != null;
    if (has_runtime != (adapter != null)) return INVALID_ARGUMENT~;

    Runtime runtime;
    Adapter registered_adapter;
    RuntimeData* runtime_data;
    if (has_runtime) {
        runtime_data = resolve_runtime(retained_runtime);
        RuntimeData* adapter_data = resolve_adapter(adapter);
        if (runtime_data == null || adapter_data != runtime_data) {
            return INVALID_HANDLE~;
        }
        if (runtime_data.live_device_count.load(AtomicOrdering.ACQUIRE) == 0) return INVALID_RESOURCE_STATE~;
        runtime = *retained_runtime;
        registered_adapter = *adapter;
    }

    Surface registered_surface;
    if (presentation_surface != null) {
        if (!has_runtime) return INVALID_ARGUMENT~;
        SurfaceData* surface_data = resolve_surface(presentation_surface);
        if (surface_data == null) return INVALID_HANDLE~;
        RuntimeData* surface_runtime = resolve_runtime(&surface_data.runtime);
        if (surface_runtime != runtime_data) return INVALID_HANDLE~;
        registered_surface = *presentation_surface;
    }
    caps.presentation_enabled = registered_surface.is_valid();
    if (queues.counts.graphics == 0
        && queues.counts.compute == 0
        && queues.counts.transfer == 0) {
        queues = default_queue_selection(&caps);
    }
    if (!queue_selection_is_well_formed(&queues)) return INVALID_ARGUMENT~;
    caps.queues = queues.counts;

    lock_device_registry();
    defer unlock_device_registry();
    for (uint i = 0; i < MAX_DEVICES; i++) {
        DeviceSlot* slot = &device_slots[i];
        DeviceSlotControl control = (DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
        if (control.state != DEVICE_SLOT_FREE) continue;

        control.state = DEVICE_SLOT_LIVE;
        control.pins = 0;
        if (control.generation == 0) control.generation = 1;
        control.reserved = 0;
        Device handle = {
            .index      = i,
            .generation = control.generation,
        };
        slot.live_children.store(0, AtomicOrdering.RELAXED);
        slot.command_owner.store(
            resource_owner_tag(resource_owner, ResourceOwnerKind.COMMAND),
            AtomicOrdering.RELAXED,
        );
        slot.lost.store(false, AtomicOrdering.RELAXED);
        slot.data = {
            .backend              = backend,
            .caps                 = caps,
            .queues               = queues,
            .vtable               = vtable,
            .backend_state        = backend_state,
            .resource_owner       = resource_owner,
            .runtime              = runtime,
            .adapter              = registered_adapter,
            .presentation_surface = registered_surface,
            .runtime_retained     = has_runtime,
        };
        if (owner != null) *owner = handle;
        slot.control.store((ulong)control, AtomicOrdering.RELEASE);
        return handle;
    }
    return SLOT_TABLE_FULL~;
}

fn void* device_backend_state_ptr(Device* device) @private {
    DeviceData* data = resolve_pinned_device(device);
    return data == null ? null : data.backend_state;
}

fn void* device_backend_state(Device device) @private {
    return device_backend_state_ptr(&device);
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

<* Create a device for the requested backend.
   Ownership: the caller must release the returned device with destroy_device. *>
fn Device? create_device_from_desc(DeviceDesc* desc) {
    if (desc == null) return INVALID_ARGUMENT~;
    if (!device_slot_available()) return SLOT_TABLE_FULL~;
    switch (desc.backend) {
        case BackendKind.VULKAN:
            return gpu::vk::create_vk_device_from_desc(desc);
        default:
            DebugMessage message = {
                .severity       = DebugMessageSeverity.ERROR,
                .category       = { .public_contract },
                .operation      = "create_device_from_desc",
                .has_fault      = true,
                .public_fault   = UNSUPPORTED_BACKEND,
                .rejected_field = "DeviceDesc.backend",
                .invariant      = "backend must be supported",
                .backend_text   = "unsupported GPU backend",
            };
            dispatch_debug_message(desc.debug_callback, desc.debug_user_data, &message);
            return UNSUPPORTED_BACKEND~;
    }
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
