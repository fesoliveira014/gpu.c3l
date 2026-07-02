module gpu;

import gpu::vk;

// Backend dispatch vtable. Most fns take the owning Device* so they reach that
// device's backend_state; the cmd_* recording entries instead take a
// CommandList* (which carries its owning Device*). The table grows one entry
// per milestone.
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
alias BeginCommandsFn       = fn void? (Device*, QueueKind, CommandList* out_commands);
alias EndCommandsFn         = fn void? (Device*, CommandList*);
alias SubmitFn              = fn void? (Device*, SubmitDesc*);
alias WaitQueueIdleFn       = fn void? (Device*, QueueKind);
alias CmdCopyBufferFn       = fn void? (CommandList*, BufferCopyDesc*);
alias CmdFillBufferFn       = fn void? (CommandList*, BufferHandle, usz, usz, uint);
alias CmdBufferBarrierFn    = fn void? (CommandList*, BufferBarrier*);
alias CmdGlobalBarrierFn    = fn void? (CommandList*, GlobalBarrier*);
alias CmdTextureBarrierFn   = fn void? (CommandList*, TextureBarrier*);
alias CreateSemaphoreFn     = fn SemaphoreHandle? (Device*, SemaphoreDesc*);
alias DestroySemaphoreFn    = fn void? (Device*, SemaphoreHandle);
alias WaitSemaphoreFn       = fn void? (Device*, SemaphoreHandle, SemaphoreValue, ulong);
alias CreateShaderFn          = fn ShaderHandle? (Device*, ShaderDesc*);
alias DestroyShaderFn         = fn void? (Device*, ShaderHandle);
alias CreateComputePipelineFn = fn PipelineHandle? (Device*, ComputePipelineDesc*);
alias DestroyPipelineFn       = fn void? (Device*, PipelineHandle);
alias CmdDispatchFn           = fn void? (CommandList*, PipelineHandle, GpuAddress, Vec3u);

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
    EndCommandsFn           end_commands;
    SubmitFn                submit;
    WaitQueueIdleFn         wait_queue_idle;
    CmdCopyBufferFn         cmd_copy_buffer;
    CmdFillBufferFn         cmd_fill_buffer;
    CmdBufferBarrierFn      cmd_buffer_barrier;
    CmdGlobalBarrierFn      cmd_global_barrier;
    CmdTextureBarrierFn     cmd_texture_barrier;
    CreateSemaphoreFn       create_semaphore;
    DestroySemaphoreFn      destroy_semaphore;
    WaitSemaphoreFn         wait_semaphore;
    CreateShaderFn          create_shader;
    DestroyShaderFn         destroy_shader;
    CreateComputePipelineFn create_compute_pipeline;
    DestroyPipelineFn       destroy_pipeline;
    CmdDispatchFn           cmd_dispatch;
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
