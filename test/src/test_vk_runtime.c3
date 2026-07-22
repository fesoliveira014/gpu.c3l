module gpu_test;

import gpu;

fn void runtimes_enumerate_vulkan_adapters_independently() @test {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_vk_runtime_test",
    };
    gpu::Runtime first = gpu::create_runtime(&desc)!!;
    gpu::Runtime second = gpu::create_runtime(&desc)!!;

    gpu::AdapterList first_adapters = gpu::enumerate_adapters(&first)!!;
    gpu::AdapterList second_adapters = gpu::enumerate_adapters(&second)!!;
    assert(first_adapters.count > 0, "the configured Vulkan ICD exposes an adapter");
    assert(second_adapters.count == first_adapters.count, "each runtime sees the same native adapters");

    gpu::Adapter first_adapter = first_adapters.get(0)!!;
    gpu::Adapter second_adapter = second_adapters.get(0)!!;
    gpu::AdapterInfo first_info = gpu::get_adapter_info(&first_adapter)!!;
    gpu::AdapterInfo second_info = gpu::get_adapter_info(&second_adapter)!!;
    gpu::AdapterDiagnostics diagnostics = gpu::get_adapter_diagnostics(&first_adapter)!!;
    assert(first_info.name.len > 0, "adapter names are populated");
    assert(second_info.name == first_info.name, "both runtimes report the same adapter");
    assert(first_info.queues.graphics_count > 0, "graphics queue availability is semantic");
    assert(diagnostics.backend_name == "Vulkan", "backend identity stays in diagnostics");
    assert(diagnostics.backend_version.major >= 1, "the native API version is diagnostic data");
    assert(diagnostics.driver_name.len > 0, "driver identity is available for logging");

    gpu::destroy_runtime(&first)!!;
    assert(
        @catch(gpu::get_adapter_info(&first_adapter)) == gpu::INVALID_HANDLE,
        "destroying one runtime invalidates only its borrowed adapters"
    );
    assert(
        gpu::get_adapter_info(&second_adapter)!!.name.len > 0,
        "the second runtime remains valid after the first is destroyed"
    );

    gpu::destroy_runtime(&second)!!;
}
