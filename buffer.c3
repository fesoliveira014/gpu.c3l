module gpu;

const usz MAX_BUFFERS = 4096;

// Buffer usage flags. A bitstruct of bools, composed by field-set:
//   BufferUsage u = { .storage, .addressable, .transfer_dst };
bitstruct BufferUsage : uint {
    bool transfer_src : 0;
    bool transfer_dst : 1;
    bool uniform      : 2;
    bool storage      : 3;
    bool addressable  : 4;
    bool indirect     : 5;
    bool index        : 6;
    bool vertex       : 7;
}

struct BufferDesc {
    usz         size;
    BufferUsage usage;
    MemoryKind  memory_kind;
    ZString     debug_name;
}

<* Create a buffer. Ownership: the caller destroys it with destroy_buffer. *>
fn BufferHandle? create_buffer(Device* device, BufferDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.create_buffer(device, desc);
}

fn void? destroy_buffer(Device* device, BufferHandle buffer) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.destroy_buffer(device, buffer);
}

<* Shader-visible GPU address of an addressable buffer; faults if the buffer was
   not created with the addressable usage flag. *>
fn GpuAddress? get_buffer_address(Device* device, BufferHandle buffer) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.get_buffer_address(device, buffer);
}

fn GpuSpan? get_buffer_span(Device* device, BufferHandle buffer) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.get_buffer_span(device, buffer);
}

<* Make CPU writes to mapped memory visible to the GPU. No-op on coherent
   memory; size 0 means the rest of the allocation. *>
fn void? flush_buffer(Device* device, BufferHandle buffer, usz offset, usz size) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.flush_buffer(device, buffer, offset, size);
}

<* Make GPU writes visible to CPU reads of mapped memory. No-op on coherent
   memory; size 0 means the rest of the allocation. *>
fn void? invalidate_buffer(Device* device, BufferHandle buffer, usz offset, usz size) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.invalidate_buffer(device, buffer, offset, size);
}
