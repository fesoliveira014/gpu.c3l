module gpu_smoke;

import gpu;
import std::io;

<* Smoke proof: import gpu, call a public facade function that routes into the
   gpu::vk backend, forcing the full gpu -> vma -> vk graph plus native libs to
   resolve and link. Exits zero when the backend reports a non-zero version. *>
fn int main() {
    uint version = gpu::backend_vulkan_version();
    io::printfn("gpu.c3l smoke: backend vulkan instance version = %d", version);
    return version != 0 ? 0 : 1;
}
