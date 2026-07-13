module gpu_smoke;

import gpu::vk @public;
import std::io;

<* Link the backend bindings and native libraries without adding a backend
   probe to the public gpu module. *>
fn int main() {
    if (!gpu::vk::probe_vma_allocator()) return 1;
    uint version = gpu::vk::probe_vulkan_version();
    io::printfn("gpu.c3l smoke: backend vulkan instance version = %d", version);
    return version != 0 ? 0 : 1;
}
