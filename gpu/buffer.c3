module gpu;

const usz MAX_BUFFERS = 4096;

// Buffer usage flags. A bitstruct of bools, composed by field-set:
//   BufferUsage u = { .storage, .addressable, .transfer_dst };
bitstruct BufferUsage : uint {
    // Source of copy commands.
    bool transfer_src : 0;
    // Destination of copy/fill commands.
    bool transfer_dst : 1;
    // Uniform buffer binding.
    bool uniform      : 2;
    // Storage buffer access from shaders.
    bool storage      : 3;
    // get_buffer_address / root-pointer access.
    bool addressable  : 4;
    // Indirect draw/dispatch argument source.
    bool indirect     : 5;
    // Index buffer for cmd_draw_indexed.
    bool index        : 6;
    // Reserved for vertex-buffer interop.
    bool vertex       : 7;
    // Concurrent sharing across queue families; required when GRAPHICS and
    // COMPUTE both touch it and caps.async_compute is set.
    bool shared_queues : 8;
}

// Creation parameters for create_buffer. `size` must be non-zero; `usage`
// must include at least one access flag (shared_queues alone maps to none).
struct BufferDesc {
    usz         size;
    BufferUsage usage;
    MemoryKind  memory_kind;
    ZString     debug_name;
}

<* Create a buffer. Ownership: the caller destroys it with destroy_buffer. *>
fn BufferHandle? create_buffer(Device* device, BufferDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    return operation.vtable.create_buffer(device, desc);
}

<* Counterpart of create_buffer; the handle is invalid afterwards. *>
fn void? destroy_buffer(Device* device, BufferHandle buffer) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, buffer.owner, ResourceOwnerKind.BUFFER)!;
    return operation.vtable.destroy_buffer(device, buffer);
}

<* Shader-visible GPU address of an addressable buffer; faults if the buffer was
   not created with the addressable usage flag. *>
fn GpuAddress? get_buffer_address(Device* device, BufferHandle buffer) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, buffer.owner, ResourceOwnerKind.BUFFER)!;
    return operation.vtable.get_buffer_address(device, buffer);
}

<* Host-visible view of a buffer; `cpu` is null for non-mappable memory kinds. *>
fn GpuSpan? get_buffer_span(Device* device, BufferHandle buffer) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, buffer.owner, ResourceOwnerKind.BUFFER)!;
    return operation.vtable.get_buffer_span(device, buffer);
}

<* Make CPU writes to mapped memory visible to the GPU. No-op on coherent
   memory; size 0 means the rest of the allocation. *>
fn void? flush_buffer(Device* device, BufferHandle buffer, usz offset, usz size) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, buffer.owner, ResourceOwnerKind.BUFFER)!;
    return operation.vtable.flush_buffer(device, buffer, offset, size);
}

<* Make GPU writes visible to CPU reads of mapped memory. No-op on coherent
   memory; size 0 means the rest of the allocation. *>
fn void? invalidate_buffer(Device* device, BufferHandle buffer, usz offset, usz size) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, buffer.owner, ResourceOwnerKind.BUFFER)!;
    return operation.vtable.invalidate_buffer(device, buffer, offset, size);
}
