module gpu;

import gpu::vk;

<* Vulkan instance version the backend's loader reports, or 0 if unavailable.
   The smoke harness uses it to prove the gpu -> vk/vma graph links. *>
fn uint backend_vulkan_version() {
    if (!gpu::vk::probe_vma_allocator()) return 0;
    return gpu::vk::probe_vulkan_version();
}
