module gpu_bench;

import gpu @public;
import gpu::vk @public;
import vk;
import std::io;

fn ZString device_type_name(vk::PhysicalDeviceType device_type) {
    switch (device_type) {
        case vk::PhysicalDeviceType.INTEGRATED_GPU: return "integrated";
        case vk::PhysicalDeviceType.DISCRETE_GPU:   return "discrete";
        case vk::PhysicalDeviceType.VIRTUAL_GPU:    return "virtual";
        case vk::PhysicalDeviceType.CPU:            return "cpu";
        default:                                    return "other";
    }
}

fn int main() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = false,
        .application_name  = "gpu_benchmark_info",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    vk::PhysicalDeviceDriverProperties driver = vk::physical_device_driver_properties();
    vk::PhysicalDeviceProperties2 properties = vk::physical_device_properties2().set_next(&driver);
    vk::get_physical_device_properties2(state.physical_device, &properties);

    bool transfer_distinct = state.transfer_queue != state.graphics_queue;

    io::printfn(
        "adapter: name=\"%s\" type=%s api_version=%d",
        (ZString)&properties.properties.device_name[0],
        device_type_name(properties.properties.device_type),
        properties.properties.api_version,
    );
    io::printfn(
        "driver: name=\"%s\" id=%d version=%d",
        (ZString)&driver.driver_name[0],
        (uint)driver.driver_id,
        properties.properties.driver_version,
    );
    io::printfn("validation: enabled=%s", state.validation ? "true" : "false");
    io::printfn(
        "queues: graphics=%d:0 compute=%d:%d compute_distinct=%s transfer=%d:0 transfer_distinct=%s",
        state.graphics_family,
        state.compute_family,
        state.compute_queue_index,
        state.compute_queue_distinct ? "true" : "false",
        state.transfer_family,
        transfer_distinct ? "true" : "false",
    );
    return 0;
}
