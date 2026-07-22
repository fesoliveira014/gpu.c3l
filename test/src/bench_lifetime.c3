module gpu_bench;

import gpu @public;
import std::os::env;

fn bool bench_validation_enabled() {
    if (try value = env::tget_var("GPU_C3L_BENCH_VALIDATION")) {
        return value == "1";
    }
    return false;
}

fn gpu::Device? create_bench_device(
    gpu::RuntimeDesc* desc,
    bool explicit_policy = false,
) {
    if (desc == null) return gpu::INVALID_ARGUMENT~;
    gpu::RuntimeDesc runtime_desc = *desc;
    if (bench_validation_enabled() && !explicit_policy) {
        runtime_desc.contract_validation = gpu::ContractValidation.FULL;
        runtime_desc.track_resource_lifetimes = true;
        runtime_desc.enable_vulkan_validation = true;
    }
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
    defer catch (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::DeviceRequest request = gpu::strict_device_request();
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!;
        gpu::DeviceRequestSupport support =
            gpu::supports_device_request(&adapter, &request)!;
        if (!support.supported) continue;
        gpu::Device? device = gpu::create_device(&adapter, &request);
        if (catch err = device) {
            if (err == gpu::UNSUPPORTED_FEATURE) continue;
            return err~;
        }
        return device;
    }
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void? discard_bench_executable(gpu::ExecutableCommandList* commands) {
    if (commands == null || !commands.handle.is_valid()) return;
    if (catch err = gpu::discard_executable_commands(commands)) {
        if (err != gpu::INVALID_HANDLE) return err~;
    }
}

fn gpu::CompletionPoint? submit_bench_queue(
    gpu::Device* device,
    gpu::QueueKind kind,
    gpu::SubmitDesc* desc,
) {
    return gpu::submit(gpu::get_queue(device, kind)!, desc);
}

fn void? wait_bench_queue(gpu::Device* device, gpu::QueueKind queue) {
    gpu::Queue target = gpu::get_queue(device, queue)!;
    gpu::CommandList commands = gpu::begin_commands(target)!;
    defer (void)gpu::discard_commands(&commands);
    gpu::ExecutableCommandList[1] lists = { gpu::end_commands(&commands)! };
    defer (void)gpu::discard_executable_commands(&lists[0]);
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(target, &submit)!;
    gpu::wait_completion(completion)!;
}

fn void? destroy_bench_device(gpu::Device* device) {
    if (device == null || !device.is_valid()) return;
    gpu::Runtime runtime;
    gpu::DeviceOperation operation = gpu::begin_device_operation(device, true)!;
    if (operation.data.runtime_retained) runtime = operation.data.runtime;
    gpu::end_device_operation(&operation);
    gpu::QueueCounts? counts_result = gpu::get_queue_counts(device);
    if (catch err = counts_result) {
        if (err == gpu::DEVICE_LOST) {
            gpu::destroy_device(device)!;
            if (runtime.is_valid()) gpu::destroy_runtime(&runtime)!;
            return;
        }
        return err~;
    }
    gpu::QueueCounts counts = counts_result;
    if (counts.graphics > 0) {
        wait_bench_queue(device, gpu::QueueKind.GRAPHICS)!;
    }
    if (counts.compute > 0) {
        wait_bench_queue(device, gpu::QueueKind.COMPUTE)!;
    }
    if (counts.transfer > 0) {
        wait_bench_queue(device, gpu::QueueKind.TRANSFER)!;
    }
    gpu::destroy_device(device)!;
    if (runtime.is_valid()) gpu::destroy_runtime(&runtime)!;
}
