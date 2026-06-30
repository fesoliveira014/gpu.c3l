module gpu;

import gpu::vk;

// Backend dispatch vtable. Each fn takes the owning Device* so it reaches that
// device's backend_state; the table grows one entry per milestone.
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
