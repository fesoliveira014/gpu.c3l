module gpu_bench;

import vk;

fn void device_type_labels_are_stable() @test {
    assert(upload_device_type_name(vk::PhysicalDeviceType.OTHER).str_view() == "other", "other label");
    assert(
        upload_device_type_name(vk::PhysicalDeviceType.INTEGRATED_GPU).str_view() == "integrated",
        "integrated label"
    );
    assert(upload_device_type_name(vk::PhysicalDeviceType.DISCRETE_GPU).str_view() == "discrete", "discrete label");
    assert(upload_device_type_name(vk::PhysicalDeviceType.VIRTUAL_GPU).str_view() == "virtual", "virtual label");
    assert(upload_device_type_name(vk::PhysicalDeviceType.CPU).str_view() == "cpu", "cpu label");
}

fn void lavapipe_detection_uses_driver_id() @test {
    assert(upload_is_lavapipe_driver(vk::DriverId.MESA_LLVMPIPE), "LLVMpipe identifies lavapipe");
    assert(!upload_is_lavapipe_driver(vk::DriverId.NVIDIA_PROPRIETARY), "NVIDIA is not lavapipe");
    assert(!upload_is_lavapipe_driver(vk::DriverId.MESA_RADV), "RADV is not lavapipe");
}

fn void scaling_is_relative_to_one_worker() @test {
    assert(upload_scaling_vs_one(250.0, 100.0) == 2.5, "scaling is measured rate divided by baseline");
    assert(upload_scaling_vs_one(250.0, 0.0) == 0.0, "zero baseline produces a safe zero ratio");
}
