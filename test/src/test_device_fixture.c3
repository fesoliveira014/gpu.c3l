module gpu_test;

import gpu::internal @public;
import gpu @public;

struct DeviceFixture {
    uint destroy_calls;
}

fn void device_fixture_destroy(void* backend_state) {
    DeviceFixture* fixture = (DeviceFixture*)backend_state;
    fixture.destroy_calls++;
}

gpu::internal::BackendVTable device_fixture_vtable = {
    .destroy_device = &device_fixture_destroy,
};

fn gpu::Device fixture_device(DeviceFixture* fixture) {
    return gpu::internal::register_device(
        backend:       gpu::internal::BackendKind.VULKAN,
        caps:          {},
        vtable:        &device_fixture_vtable,
        backend_state: fixture,
    )!!;
}