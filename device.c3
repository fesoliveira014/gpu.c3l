module gpu;

import gpu::vk;

// Backend dispatch vtable. Each fn takes the owning Device* so it reaches that
// device's backend_state; the table grows one entry per milestone.
alias DestroyDeviceFn = fn void (Device*);

struct BackendVTable {
    DestroyDeviceFn destroy_device;
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
