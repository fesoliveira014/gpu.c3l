module gpu;

import gpu::vk;

// Backend dispatch vtable. Most fns take the owning Device* so they reach that
// device's backend_state; the cmd_* recording entries instead take a
// CommandList* (which carries its owning Device*).
alias DestroyDeviceFn       = fn void (Device*);
alias GetMemoryStatsFn      = fn MemoryStats? (Device*);
alias BuildMemoryReportFn   = fn String? (Device*, bool detailed);
alias CreateBufferFn        = fn BufferHandle? (Device*, BufferDesc*);
alias DestroyBufferFn       = fn void? (Device*, BufferHandle);
alias GetBufferAddressFn    = fn GpuAddress? (Device*, BufferHandle);
alias GetBufferSpanFn       = fn GpuSpan? (Device*, BufferHandle);
alias BeginFrameFn          = fn void? (Device*);
alias AllocFrameSpanFn      = fn GpuSpan? (Device*, usz size, usz align);
alias EndFrameFn            = fn void? (Device*);
alias AllocPersistentSpanFn = fn GpuSpan? (Device*, PersistentAllocDesc*);
alias FreePersistentSpanFn  = fn void? (Device*, GpuSpan);
alias GetPersistentStatsFn  = fn PersistentArenaStats? (Device*);
// Out-param form works around a c3c 0.8.x debug-info crash: an
// optional-of-struct return in a vtable fn-ptr signature dangles c3c's
// debug-type cache (scripts/c3c_bug_repro/ISSUE.md). Revert to
// `fn CommandList? (Device*, QueueKind)` once the upstream fix lands.
alias BeginCommandsFn       = fn void? (Device*, QueueKind, RecordingContextHandle, CommandList* out_commands);
alias CreateRecordingContextFn  = fn RecordingContextHandle? (Device*);
alias DestroyRecordingContextFn = fn void? (Device*, RecordingContextHandle);
alias EndCommandsFn         = fn void? (Device*, CommandList*);
alias SubmitFn              = fn void? (Device*, SubmitDesc*);
alias WaitQueueIdleFn       = fn void? (Device*, QueueKind);
alias CmdCopyBufferFn       = fn void? (CommandList*, BufferCopyDesc*);
alias CmdFillBufferFn       = fn void? (CommandList*, BufferHandle, usz, usz, uint);
alias CmdCopyBufferToTextureFn = fn void? (CommandList*, BufferTextureCopyDesc*);
alias CmdCopyTextureToBufferFn = fn void? (CommandList*, TextureBufferCopyDesc*);
alias FlushBufferFn            = fn void? (Device*, BufferHandle, usz, usz);
alias InvalidateBufferFn       = fn void? (Device*, BufferHandle, usz, usz);
alias CmdBufferBarrierFn    = fn void? (CommandList*, BufferBarrier*);
alias CmdGlobalBarrierFn    = fn void? (CommandList*, GlobalBarrier*);
alias CmdTextureBarrierFn   = fn void? (CommandList*, TextureBarrier*);
alias CreateSemaphoreFn     = fn SemaphoreHandle? (Device*, SemaphoreDesc*);
alias DestroySemaphoreFn    = fn void? (Device*, SemaphoreHandle);
alias WaitSemaphoreFn       = fn void? (Device*, SemaphoreHandle, SemaphoreValue, ulong);
alias GetTextureFormatSupportFn = fn TextureFormatSupport? (Device*, Format);
alias SupportsTextureDescFn     = fn bool? (Device*, TextureDesc*);
alias CreateTextureFn           = fn TextureHandle? (Device*, TextureDesc*);
alias DestroyTextureFn          = fn void? (Device*, TextureHandle);
alias CreateTextureDescriptorFn  = fn TextureIndex? (Device*, TextureHandle, TextureViewDesc*);
alias DestroyTextureDescriptorFn = fn void? (Device*, TextureIndex);
alias CreateTextureDescriptorsFn = fn void? (Device*, TextureDescriptorDesc[], TextureIndex[]);
alias CreateSamplerFn            = fn SamplerIndex? (Device*, SamplerDesc*);
alias DestroySamplerFn           = fn void? (Device*, SamplerIndex);
alias CreateShaderFn          = fn ShaderHandle? (Device*, ShaderDesc*);
alias DestroyShaderFn         = fn void? (Device*, ShaderHandle);
alias CreateComputePipelineFn = fn PipelineHandle? (Device*, ComputePipelineDesc*);
alias CreateGraphicsPipelineFn = fn PipelineHandle? (Device*, GraphicsPipelineDesc*);
alias DestroyPipelineFn       = fn void? (Device*, PipelineHandle);
alias GetPipelineCacheSizeFn  = fn usz? (Device*);
alias GetPipelineCacheDataFn  = fn usz? (Device*, char[]);
alias CmdDispatchFn           = fn void? (CommandList*, PipelineHandle, GpuAddress, Vec3u);
alias CmdBeginRenderPassFn    = fn void? (CommandList*, RenderPassDesc*);
alias CmdSetViewportFn        = fn void? (CommandList*, Viewport*);
alias CmdSetScissorFn         = fn void? (CommandList*, ScissorRect*);
alias CmdEndRenderPassFn      = fn void? (CommandList*);
alias CmdDrawFn               = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuAddress, uint, uint);
alias CmdDrawIndexedFn        = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuAddress, GpuSpan, uint, uint, IndexType);
alias CmdDrawIndirectFn       = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuAddress, GpuSpan, uint);
alias CmdDrawIndexedIndirectFn = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuAddress, GpuSpan, uint, GpuSpan, IndexType);
alias CmdDrawIndexedIndirectCountFn = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuAddress, GpuSpan, GpuSpan, uint, GpuSpan, IndexType);
alias CmdDispatchIndirectFn   = fn void? (CommandList*, PipelineHandle, GpuAddress, GpuSpan);
alias CmdBeginLabelFn         = fn void? (CommandList*, ZString, float[4]);
alias CmdEndLabelFn           = fn void? (CommandList*);
alias CmdUploadBufferFn       = fn void? (CommandList*, BufferHandle, usz, char[]);
alias CmdUploadTextureFn      = fn void? (CommandList*, TextureUploadDesc*, char[]);
alias UploadBufferDataFn      = fn void? (Device*, BufferHandle, usz, char[], Stage, Hazard);
alias UploadTextureDataFn     = fn void? (Device*, TextureHandle, uint, char[], Stage, Hazard, TextureLayout);
alias ReadbackBufferDataFn    = fn void? (Device*, BufferHandle, usz, char[], Stage, Hazard);
alias ReadbackTextureDataFn   = fn void? (Device*, TextureHandle, uint, char[], Stage, Hazard, TextureLayout);
alias CmdReadbackBufferFn     = fn ReadbackTicket? (CommandList*, BufferHandle, usz, usz);
alias CmdReadbackTextureFn    = fn ReadbackTicket? (CommandList*, TextureHandle, uint);
alias PollReadbackFn          = fn bool (Device*, ReadbackTicket*);
alias ResolveReadbackFn       = fn void? (Device*, ReadbackTicket*, char[]);
alias CreateSwapchainFn       = fn SwapchainHandle? (Device*, SurfaceDesc*, SwapchainDesc*);
alias DestroySwapchainFn      = fn void? (Device*, SwapchainHandle);
alias ResizeSwapchainFn       = fn void? (Device*, SwapchainHandle, uint, uint);
alias GetSwapchainInfoFn      = fn SwapchainInfo? (Device*, SwapchainHandle);
alias AcquireNextImageFn      = fn AcquiredImage? (Device*, SwapchainHandle);
alias PresentFn               = fn void? (Device*, PresentDesc*);
alias GetPresentModeSupportFn = fn PresentModeSupport? (Device*, SwapchainHandle);

// One static table per backend; Device.vtable points at it.
struct BackendVTable {
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

// A GPU device. `vtable` points at one static table shared by all devices of a
// backend; `backend_state` is this device's own opaque backend object.
struct Device {
    BackendKind    backend;
    DeviceCaps     caps;
    BackendVTable* vtable;
    void*          backend_state;
}

<* Create a device for the requested backend.
   Ownership: the caller must release the returned device with destroy_device. *>
fn Device? create_device(DeviceDesc* desc) {
    switch (desc.backend) {
        case BackendKind.VULKAN:
            return gpu::vk::create_vk_device(desc);
        default:
            return UNSUPPORTED_BACKEND~;
    }
}

<* Destroy a device, releasing all backend resources. *>
fn void? destroy_device(Device* device) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    device.vtable.destroy_device(device);
}

<* Snapshot per-heap memory usage and live allocation counts. *>
fn MemoryStats? get_memory_stats(Device* device) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.get_memory_stats(device);
}

<* Build a human-readable memory report. Ownership: the caller frees the
   returned String. *>
fn String? build_memory_report(Device* device, bool detailed) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.build_memory_report(device, detailed);
}
