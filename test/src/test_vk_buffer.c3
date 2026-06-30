module gpu_test;

import gpu;

// Headless (lavapipe): VMA-backed buffer create/destroy, address, span, handle
// validation.

fn void buffer_create_address_span() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_buffer_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    // Device-local storage buffer.
    gpu::BufferDesc device_desc = {
        .size        = 4096,
        .usage       = { .storage, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "device_buffer",
    };
    gpu::BufferHandle device_buffer = gpu::create_buffer(&device, &device_desc)!!;
    assert(device_buffer.is_valid(), "device buffer handle is valid");

    // Host-visible upload buffer is persistently mapped.
    gpu::BufferDesc upload_desc = {
        .size        = 256,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.FRAME_UPLOAD,
    };
    gpu::BufferHandle upload_buffer = gpu::create_buffer(&device, &upload_desc)!!;
    gpu::GpuSpan upload_span = gpu::get_buffer_span(&device, upload_buffer)!!;
    assert(upload_span.cpu != null, "upload buffer span is mapped");
    assert(upload_span.size == 256, "span reports buffer size");

    // Addressable buffer exposes a non-zero GPU address.
    gpu::BufferDesc addressable_desc = {
        .size        = 1024,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle addressable_buffer = gpu::create_buffer(&device, &addressable_desc)!!;
    gpu::GpuAddress address = gpu::get_buffer_address(&device, addressable_buffer)!!;
    assert((ulong)address != 0, "addressable buffer has a non-zero GPU address");

    // A non-addressable buffer faults on get_buffer_address.
    bool address_faulted = false;
    if (catch gpu::get_buffer_address(&device, device_buffer)) address_faulted = true;
    assert(address_faulted, "non-addressable buffer get_buffer_address faults");

    // Destroying retires the handle.
    gpu::destroy_buffer(&device, device_buffer)!!;
    bool stale_faulted = false;
    if (catch gpu::get_buffer_span(&device, device_buffer)) stale_faulted = true;
    assert(stale_faulted, "destroyed handle returns INVALID_HANDLE");

    gpu::destroy_buffer(&device, upload_buffer)!!;
    gpu::destroy_buffer(&device, addressable_buffer)!!;
}
