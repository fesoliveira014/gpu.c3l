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

fn vk::Result fake_create_debug_messenger(
    vk::Instance instance,
    vk::DebugUtilsMessengerCreateInfoEXT* info,
    vk::AllocationCallbacks* allocator,
    vk::DebugUtilsMessengerEXT* messenger,
) {
    return vk::Result.SUCCESS;
}

fn void fake_destroy_debug_messenger(
    vk::Instance instance,
    vk::DebugUtilsMessengerEXT messenger,
    vk::AllocationCallbacks* allocator,
) {}

fn void fake_begin_label(
    vk::CommandBuffer command_buffer,
    vk::DebugUtilsLabelEXT* info,
) {}

fn void fake_end_label(vk::CommandBuffer command_buffer) {}

fn void instance_debug_messenger_dispatch_requires_complete_pair() @test {
    gpu::vk::VkInstanceDispatch empty;
    assert(!gpu::vk::instance_dispatch_supports_debug_messenger(&empty), "empty pair is unavailable");

    gpu::vk::VkInstanceDispatch create_only = {
        .create_debug_messenger = &fake_create_debug_messenger,
    };
    assert(!gpu::vk::instance_dispatch_supports_debug_messenger(&create_only), "create-only pair is unavailable");

    gpu::vk::VkInstanceDispatch destroy_only = {
        .destroy_debug_messenger = &fake_destroy_debug_messenger,
    };
    assert(!gpu::vk::instance_dispatch_supports_debug_messenger(&destroy_only), "destroy-only pair is unavailable");

    gpu::vk::VkInstanceDispatch complete = {
        .create_debug_messenger  = &fake_create_debug_messenger,
        .destroy_debug_messenger = &fake_destroy_debug_messenger,
    };
    assert(gpu::vk::instance_dispatch_supports_debug_messenger(&complete), "complete pair is available");
}

fn void device_label_dispatch_requires_complete_pair() @test {
    gpu::vk::VkDeviceDispatch empty;
    assert(!gpu::vk::device_dispatch_supports_labels(&empty), "empty pair is unavailable");

    gpu::vk::VkDeviceDispatch begin_only = {
        .cmd_begin_label = &fake_begin_label,
    };
    assert(!gpu::vk::device_dispatch_supports_labels(&begin_only), "begin-only pair is unavailable");

    gpu::vk::VkDeviceDispatch end_only = {
        .cmd_end_label = &fake_end_label,
    };
    assert(!gpu::vk::device_dispatch_supports_labels(&end_only), "end-only pair is unavailable");

    gpu::vk::VkDeviceDispatch complete = {
        .cmd_begin_label = &fake_begin_label,
        .cmd_end_label   = &fake_end_label,
    };
    assert(gpu::vk::device_dispatch_supports_labels(&complete), "complete pair is available");
}
