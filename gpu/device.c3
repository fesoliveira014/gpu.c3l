module gpu;

import gpu::vk;

<* Opaque generation token for the process-wide device slot. *>
typedef Device = inline ulong;

// Backend dispatch functions. Command recording entries take the owner-bearing
// CommandList; all other entries take the validated device token.
alias DestroyDeviceFn @private       = fn void (Device*);
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
alias CreateSwapchainFn @private       = fn SwapchainHandle? (Device*, SurfaceDesc*, SwapchainDesc*);
alias DestroySwapchainFn @private      = fn void? (Device*, SwapchainHandle);
alias ResizeSwapchainFn @private       = fn void? (Device*, SwapchainHandle, uint, uint);
alias GetSwapchainInfoFn @private      = fn SwapchainInfo? (Device*, SwapchainHandle);
alias AcquireNextImageFn @private      = fn AcquiredImage? (Device*, SwapchainHandle);
alias PresentFn @private               = fn void? (Device*, PresentDesc*);
alias GetPresentModeSupportFn @private = fn PresentModeSupport? (Device*, SwapchainHandle);

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
    ulong          generation;
    BackendKind    backend;
    DeviceCaps     caps;
    BackendVTable* vtable;
    void*          backend_state;
}

DeviceData active_device @private;
ulong next_device_generation @private;
bool device_active @private;

fn DeviceData* resolve_device(Device* device) @private {
    if (device == null || (ulong)*device == 0 || !device_active) return null;
    if ((ulong)*device != active_device.generation) return null;
    return &active_device;
}

fn bool device_slot_available() @private => !device_active;

fn Device? register_device(
    BackendKind backend,
    DeviceCaps caps,
    BackendVTable* vtable,
    void* backend_state
) @private {
    if (device_active || vtable == null || backend_state == null) return INVALID_RESOURCE_STATE~;
    next_device_generation++;
    if (next_device_generation == 0) next_device_generation++;
    active_device = {
        .generation    = next_device_generation,
        .backend       = backend,
        .caps          = caps,
        .vtable        = vtable,
        .backend_state = backend_state,
    };
    device_active = true;
    return (Device)next_device_generation;
}

fn void? update_device_caps(Device* device, DeviceCaps caps) @private {
    DeviceData* data = resolve_device(device);
    if (data == null) return INVALID_HANDLE~;
    data.caps = caps;
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
fn Device? create_device(DeviceDesc* desc) {
    if (!device_slot_available()) return INVALID_RESOURCE_STATE~;
    switch (desc.backend) {
        case BackendKind.VULKAN:
            return gpu::vk::create_vk_device(desc);
        default:
            DebugMessage message = {
                .severity       = DebugMessageSeverity.ERROR,
                .category       = { .public_contract },
                .operation      = "create_device",
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

<* Destroy a device, releasing all backend resources. *>
fn void? destroy_device(Device* device) {
    DeviceData* data = resolve_device(device);
    if (data == null) return INVALID_HANDLE~;
    BackendVTable* vtable = data.vtable;
    vtable.destroy_device(device);
    active_device = {};
    device_active = false;
    next_device_generation++;
    if (next_device_generation == 0) next_device_generation++;
    *device = {};
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
