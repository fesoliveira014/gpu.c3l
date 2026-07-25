module gpu_test;

import gpu::internal @public;
import gpu::internal::vk @public;
import gpu @public;

struct DeviceFixture {
    gpu::internal::vk::VkDeviceState state;
}

gpu::internal::BackendVTable device_fixture_vtable;

fn gpu::Device fixture_device(DeviceFixture* fixture) {
    return gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
        vtable: &device_fixture_vtable,
    )!!;
}
