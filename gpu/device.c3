module gpu;

import gpu::vk @public;
import std::atomic;
import std::atomic::types;
import std::thread;

// Backend dispatch functions. Command recording entries take the owner-bearing
// CommandList; all other entries take the validated device token.
alias DestroyDeviceFn @private       = fn void (void*);
alias GetMemoryStatsFn @private      = fn MemoryStats? (Device*);
alias BuildMemoryReportFn @private   = fn String? (Device*, bool detailed);
alias CreateBufferFn @private        = fn BufferHandle? (Device*, BufferDesc*);
alias DestroyBufferFn @private       = fn void? (Device*, BufferHandle);
alias GetBufferAddressFn @private    = fn GpuAddress? (Device*, BufferHandle);
alias GetBufferSpanFn @private       = fn GpuSpan? (Device*, BufferHandle);
alias BeginFrameFn @private          = fn void? (Device*, FrameToken* out_frame);
alias AllocFrameSpanFn @private      = fn GpuSpan? (FrameToken*, usz size, usz align);
alias EndFrameFn @private            = fn void? (FrameToken*);
alias AllocPersistentSpanFn @private = fn GpuSpan? (Device*, PersistentAllocDesc*);
alias FreePersistentSpanFn @private  = fn void? (Device*, GpuSpan);
alias GetPersistentStatsFn @private  = fn PersistentArenaStats? (Device*);
// Out-param form avoids a c3c 0.8.0 debug-info codegen failure.
alias BeginCommandsFn @private       = fn void? (Device*, QueueKind, RecordingContextHandle, CommandList* out_commands);
alias CreateRecordingContextFn @private  = fn RecordingContextHandle? (Device*);
alias DestroyRecordingContextFn @private = fn void? (Device*, RecordingContextHandle);
alias EndCommandsFn @private         = fn void? (CommandList*);
alias SubmitFn @private              = fn void? (Device*, SubmitDesc*);
alias WaitQueueIdleFn @private       = fn void? (Device*, QueueKind);
alias CmdCopyBufferFn @private       = fn void? (CommandList*, BufferCopyDesc*);
alias CmdFillBufferFn @private       = fn void? (CommandList*, BufferHandle, usz, usz, uint);
alias CmdCopyBufferToTextureFn @private = fn void? (CommandList*, BufferTextureCopyDesc*);
alias CmdCopyTextureToBufferFn @private = fn void? (CommandList*, TextureBufferCopyDesc*);
alias FlushBufferFn @private            = fn void? (Device*, BufferHandle, usz, usz);
alias InvalidateBufferFn @private       = fn void? (Device*, BufferHandle, usz, usz);
alias CmdBufferBarrierFn @private    = fn void? (CommandList*, BufferBarrier*);
alias CmdGlobalBarrierFn @private    = fn void? (CommandList*, GlobalBarrier*);
alias CmdTextureBarrierFn @private   = fn void? (CommandList*, TextureBarrier*);
alias CreateSemaphoreFn @private     = fn SemaphoreHandle? (Device*, SemaphoreDesc*);
alias DestroySemaphoreFn @private    = fn void? (Device*, SemaphoreHandle);
alias WaitSemaphoreFn @private       = fn void? (Device*, SemaphoreHandle, SemaphoreValue, ulong);
alias GetTextureFormatSupportFn @private = fn TextureFormatSupport? (Device*, Format);
alias SupportsTextureDescFn @private     = fn bool? (Device*, TextureDesc*);
alias CreateTextureFn @private           = fn TextureHandle? (Device*, TextureDesc*);
alias DestroyTextureFn @private          = fn void? (Device*, TextureHandle);
alias CreateTextureDescriptorFn @private  = fn TextureIndex? (Device*, TextureHandle, TextureViewDesc*);
alias DestroyTextureDescriptorFn @private = fn void? (Device*, TextureIndex);
alias CreateTextureDescriptorsFn @private = fn void? (Device*, TextureDescriptorDesc[], TextureIndex[]);
alias CreateSamplerFn @private            = fn SamplerIndex? (Device*, SamplerDesc*);
alias DestroySamplerFn @private           = fn void? (Device*, SamplerIndex);
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
alias CmdUploadBufferFn @private       = fn void? (CommandList*, BufferHandle, usz, char[]);
alias CmdUploadTextureFn @private      = fn void? (CommandList*, TextureUploadDesc*, char[]);
alias UploadBufferDataFn @private      = fn void? (Device*, BufferHandle, usz, char[], Stage, Hazard);
alias UploadTextureDataFn @private     = fn void? (Device*, TextureHandle, uint, char[], Stage, Hazard, TextureLayout);
alias ReadbackBufferDataFn @private    = fn void? (Device*, BufferHandle, usz, char[], Stage, Hazard);
alias ReadbackTextureDataFn @private   = fn void? (Device*, TextureHandle, uint, char[], Stage, Hazard, TextureLayout);
alias CmdReadbackBufferFn @private     = fn ReadbackTicket? (CommandList*, BufferHandle, usz, usz);
alias CmdReadbackTextureFn @private    = fn ReadbackTicket? (CommandList*, TextureHandle, uint);
alias PollReadbackFn @private          = fn bool (Device*, ReadbackTicket*);
alias ResolveReadbackFn @private       = fn void? (Device*, ReadbackTicket*, char[]);
alias CreateSwapchainFn @private       = fn SwapchainHandle? (Device*, Surface*, SwapchainDesc*);
alias DestroySwapchainFn @private      = fn void? (Device*, SwapchainHandle);
alias ResizeSwapchainFn @private       = fn void? (Device*, SwapchainHandle, uint, uint);
alias GetSwapchainInfoFn @private      = fn SwapchainInfo? (Device*, SwapchainHandle);
alias AcquireNextImageFn @private      = fn AcquiredImage? (Device*, SwapchainHandle);
alias PresentFn @private               = fn void? (Device*, PresentDesc*);
alias GetPresentModeSupportFn @private = fn PresentModeSupport? (Device*, SwapchainHandle);

const uint MAX_DEVICES @private = 64;
const uint MAX_DEVICE_GENERATION @private = 0x00FF_FFFF;

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
    DestroyDeviceFn         destroy_device;
    GetMemoryStatsFn        get_memory_stats;
    BuildMemoryReportFn     build_memory_report;
    CreateBufferFn          create_buffer;
    DestroyBufferFn         destroy_buffer;
    GetBufferAddressFn      get_buffer_address;
    GetBufferSpanFn         get_buffer_span;
    BeginFrameFn            begin_frame;
    AllocFrameSpanFn        alloc_frame_span;
    EndFrameFn              end_frame;
    AllocPersistentSpanFn   alloc_persistent_span;
    FreePersistentSpanFn    free_persistent_span;
    GetPersistentStatsFn    get_persistent_stats;
    BeginCommandsFn         begin_commands;
    CreateRecordingContextFn  create_recording_context;
    DestroyRecordingContextFn destroy_recording_context;
    EndCommandsFn           end_commands;
    SubmitFn                submit;
    WaitQueueIdleFn         wait_queue_idle;
    CmdCopyBufferFn         cmd_copy_buffer;
    CmdFillBufferFn         cmd_fill_buffer;
    CmdCopyBufferToTextureFn cmd_copy_buffer_to_texture;
    CmdCopyTextureToBufferFn cmd_copy_texture_to_buffer;
    FlushBufferFn            flush_buffer;
    InvalidateBufferFn       invalidate_buffer;
    CmdBufferBarrierFn      cmd_buffer_barrier;
    CmdGlobalBarrierFn      cmd_global_barrier;
    CmdTextureBarrierFn     cmd_texture_barrier;
    CreateSemaphoreFn       create_semaphore;
    DestroySemaphoreFn      destroy_semaphore;
    WaitSemaphoreFn         wait_semaphore;
    GetTextureFormatSupportFn get_texture_format_support;
    SupportsTextureDescFn     supports_texture_desc;
    CreateTextureFn         create_texture;
    DestroyTextureFn        destroy_texture;
    CreateTextureDescriptorFn  create_texture_descriptor;
    DestroyTextureDescriptorFn destroy_texture_descriptor;
    CreateTextureDescriptorsFn create_texture_descriptors;
    CreateSamplerFn            create_sampler;
    DestroySamplerFn           destroy_sampler;
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
    CmdUploadBufferFn       cmd_upload_buffer;
    CmdUploadTextureFn      cmd_upload_texture;
    UploadBufferDataFn      upload_buffer_data;
    UploadTextureDataFn     upload_texture_data;
    ReadbackBufferDataFn    readback_buffer_data;
    ReadbackTextureDataFn   readback_texture_data;
    CmdReadbackBufferFn     cmd_readback_buffer;
    CmdReadbackTextureFn    cmd_readback_texture;
    PollReadbackFn          poll_readback;
    ResolveReadbackFn       resolve_readback;
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
    BackendVTable* vtable;
    void*          backend_state;
    Runtime        runtime;
    Adapter        adapter;
    Surface        presentation_surface;
    bool           runtime_retained;
}

struct DeviceSlot @private {
    Atomic{ulong} control;
    DeviceData    data;
}

DeviceSlot[MAX_DEVICES] device_slots @private;
Atomic{bool} device_registry_lock @private;

fn void lock_device_registry() @private {
    while (device_registry_lock.set(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
}

fn void unlock_device_registry() @private {
    (void)device_registry_lock.clear(AtomicOrdering.RELEASE);
}

fn DeviceData* resolve_device(Device* device) @private {
    if (device == null || !device.is_valid()) return null;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return null;

    DeviceSlot* slot = &device_slots[device.index];
    DeviceSlotControl control = (DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    if (control.state != DEVICE_SLOT_LIVE || control.generation != device.generation) return null;
    return &slot.data;
}

fn bool device_slot_available() @private {
    lock_device_registry();
    defer unlock_device_registry();
    for (uint i = 0; i < MAX_DEVICES; i++) {
        DeviceSlotControl control = (DeviceSlotControl)device_slots[i].control.load(AtomicOrdering.RELAXED);
        if (control.state == DEVICE_SLOT_FREE && control.generation < MAX_DEVICE_GENERATION) return true;
    }
    return false;
}

fn Device? register_device(
    BackendKind backend,
    DeviceCaps caps,
    BackendVTable* vtable,
    void* backend_state,
    Runtime* retained_runtime = null,
    Adapter* adapter = null,
    Surface* presentation_surface = null,
    Device* owner = null,
) @private {
    if (vtable == null || backend_state == null || vtable.destroy_device == null) return INVALID_ARGUMENT~;
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

    lock_device_registry();
    defer unlock_device_registry();
    for (uint i = 0; i < MAX_DEVICES; i++) {
        DeviceSlot* slot = &device_slots[i];
        DeviceSlotControl control = (DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
        if (control.state != DEVICE_SLOT_FREE) continue;
        if (control.generation == MAX_DEVICE_GENERATION) {
            control.state = DEVICE_SLOT_RETIRED;
            slot.control.store((ulong)control, AtomicOrdering.RELAXED);
            continue;
        }

        control.state = DEVICE_SLOT_LIVE;
        control.pins = 0;
        control.generation++;
        control.reserved = 0;
        Device handle = {
            .index      = i,
            .generation = control.generation,
        };
        slot.data = {
            .backend              = backend,
            .caps                 = caps,
            .vtable               = vtable,
            .backend_state        = backend_state,
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

fn BackendVTable* device_vtable(Device* device) @private {
    DeviceData* data = resolve_device(device);
    return data == null ? null : data.vtable;
}

fn void* device_backend_state_ptr(Device* device) @private {
    DeviceData* data = resolve_device(device);
    return data == null ? null : data.backend_state;
}

fn void* device_backend_state(Device device) @private {
    return device_backend_state_ptr(&device);
}

<* Return the backend selected for the live device. *>
fn BackendKind? get_device_backend(Device* device) {
    DeviceData* data = resolve_device(device);
    if (data == null) return INVALID_HANDLE~;
    return data.backend;
}

<* Snapshot the selected device's capabilities. *>
fn DeviceCaps? get_device_caps(Device* device) {
    DeviceData* data = resolve_device(device);
    if (data == null) return INVALID_HANDLE~;
    return data.caps;
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

<* Destroy a device, releasing all backend resources. *>
fn void? destroy_device(Device* device) {
    if (device == null || !device.is_valid()) return INVALID_HANDLE~;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return INVALID_HANDLE~;

    lock_device_registry();
    DeviceSlot* slot = &device_slots[device.index];
    DeviceSlotControl control = (DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
    if (control.state != DEVICE_SLOT_LIVE || control.generation != device.generation) {
        unlock_device_registry();
        return INVALID_HANDLE~;
    }
    control.state = DEVICE_SLOT_CLOSING;
    slot.control.store((ulong)control, AtomicOrdering.RELEASE);
    DeviceData data = slot.data;
    unlock_device_registry();

    data.vtable.destroy_device(data.backend_state);

    lock_device_registry();
    slot.data = {};
    control.pins = 0;
    control.state = control.generation == MAX_DEVICE_GENERATION ? DEVICE_SLOT_RETIRED : DEVICE_SLOT_FREE;
    control.reserved = 0;
    slot.control.store((ulong)control, AtomicOrdering.RELEASE);
    unlock_device_registry();

    *device = DEVICE_INVALID;
    if (data.runtime_retained) runtime_release_device(&data.runtime)!;
}

<* Snapshot per-heap memory usage and live allocation counts. *>
fn MemoryStats? get_memory_stats(Device* device) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.get_memory_stats(device);
}

<* Build a human-readable memory report. Ownership: the caller frees the
   returned String. *>
fn String? build_memory_report(Device* device, bool detailed) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.build_memory_report(device, detailed);
}
