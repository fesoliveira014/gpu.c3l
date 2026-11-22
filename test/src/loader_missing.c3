module gpu_loader_missing;

import gpu;
import vk;
import std::io;

const ZString MISSING_LOADER = "gpu-c3l-test-missing-vulkan-loader";
const int EXIT_NEEDS_VULKAN = 3;

<* Create a runtime with the loader made unfindable, as on a machine without
   Vulkan, and expect BACKEND_LOADER_MISSING. With --missing-only, stop there
   the way an application would (message, exit EXIT_NEEDS_VULKAN); otherwise
   retry with the system loader and exit 0 when that runtime is created. *>
fn int main(String[] args) {
    if (!vk::set_loader_library(MISSING_LOADER)) return 1;
    gpu::RuntimeDesc desc = { .application_name = "gpu_loader_missing" };
    gpu::Runtime? missing = gpu::create_runtime(&desc);
    if (try runtime = missing) {
        io::printn("gpu.c3l loader_missing: a runtime was created without a loader");
        (void)gpu::destroy_runtime(&runtime);
        return 1;
    }
    fault failure = @catch(missing);
    io::printfn("gpu.c3l loader_missing: create_runtime -> %s", failure);
    if (failure != gpu::BACKEND_LOADER_MISSING) return 1;
    if (args.len > 1 && args[1] == "--missing-only") {
        io::printn("This program needs a graphics driver with Vulkan 1.3.");
        return EXIT_NEEDS_VULKAN;
    }

    if (!vk::set_loader_library(null)) return 1;
    gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::destroy_runtime(&runtime)!!;
    io::printn("gpu.c3l loader_missing: the system loader opened on the retry");
    return 0;
}
