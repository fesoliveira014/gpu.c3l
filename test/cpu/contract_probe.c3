module contract_probe;

import gpu;
import std::io;

const int EXIT_NO_PANIC = 1;
const int EXIT_USAGE = 2;

fn int main(String[] args) {
    if (args.len != 2) {
        (void)io::eprintn("usage: contract_probe <case>");
        return EXIT_USAGE;
    }
    String probe_case = args[1];
    fault failure;
    switch (probe_case) {
        case "valid":
            failure = @catch(probe_queue_role(gpu::QueueKind.GRAPHICS));
            if (!failure) return 0;
        case "null_runtime_desc":
            failure = @catch(probe_null_runtime_desc());
        case "adapter_index":
            failure = @catch(probe_adapter_index());
        case "queue_role_not_exposed":
            failure = @catch(probe_queue_role(gpu::QueueKind.COMPUTE));
        default:
            (void)io::eprintfn("contract_probe: unknown case %s", probe_case);
            return EXIT_USAGE;
    }
    if (failure) {
        (void)io::eprintfn("contract_probe: %s failed during setup: %s", probe_case, failure);
    } else {
        (void)io::eprintfn("contract_probe: %s broke no contract", probe_case);
    }
    return EXIT_NO_PANIC;
}

fn void? probe_null_runtime_desc() {
    gpu::RuntimeDesc* desc = null;
    gpu::Runtime runtime = gpu::create_runtime(desc)!;
    (void)gpu::destroy_runtime(&runtime);
}

fn void? probe_adapter_index() {
    gpu::RuntimeDesc desc;
    gpu::Runtime runtime = gpu::create_runtime(&desc)!;
    defer (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime);
    gpu::Adapter adapter = adapters.get(adapters.count);
}

fn void? probe_queue_role(gpu::QueueKind kind) {
    gpu::RuntimeDesc runtime_desc;
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
    defer (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime);
    if (adapters.count == 0) return gpu::UNSUPPORTED_FEATURE~;
    gpu::Adapter adapter = adapters.get(0);
    gpu::DeviceDesc device_desc = { .queues = { .required = { .graphics } } };
    gpu::Device device = gpu::create_device(&adapter, &device_desc)!;
    defer (void)gpu::destroy_device(&device);
    gpu::Queue queue = gpu::get_queue(&device, kind);
}
