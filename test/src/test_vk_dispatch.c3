module gpu_test;

import gpu::vk @public;
import vk;

fn void fake_get_descriptor_a(
    vk::Device device,
    vk::DescriptorGetInfoEXT* info,
    sz data_size,
    void* descriptor,
) {}

fn void fake_get_descriptor_b(
    vk::Device device,
    vk::DescriptorGetInfoEXT* info,
    sz data_size,
    void* descriptor,
) {}

fn vk::PFN_vkVoidFunction fake_device_proc_loader(
    vk::Device device,
    char* name,
) {
    if (((ZString)name).str_view() != "vkGetDescriptorEXT") return null;
    return (vk::PFN_vkVoidFunction)(device == (vk::Device)(uptr)1
        ? &fake_get_descriptor_a
        : &fake_get_descriptor_b);
}

fn void device_dispatch_loads_per_device() @test {
    gpu::vk::VkDeviceDispatch first = gpu::vk::load_vk_device_dispatch(
        (vk::Device)(uptr)1,
        &fake_device_proc_loader,
    );
    gpu::vk::VkDeviceDispatch second = gpu::vk::load_vk_device_dispatch(
        (vk::Device)(uptr)2,
        &fake_device_proc_loader,
    );

    assert(first.get_descriptor == &fake_get_descriptor_a, "first device keeps its loaded entry point");
    assert(second.get_descriptor == &fake_get_descriptor_b, "second device keeps its loaded entry point");
    assert(first.get_descriptor != second.get_descriptor, "device dispatch values do not alias mutable state");
}
