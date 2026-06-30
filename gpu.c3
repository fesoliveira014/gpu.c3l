module gpu;

import gpu::vk;

<* Vulkan instance-level API version the backend's loader reports. Calling this
   exercises the public gpu -> gpu::vk -> vk/vma path with real `vk` and `vma`
   calls, forcing the dependency graph and the native libraries to link.
   Returns 0 if the VMA binding or the Vulkan loader is unavailable. *>
fn uint backend_vulkan_version() {
    if (!gpu::vk::probe_vma_allocator()) return 0;
    return gpu::vk::probe_vulkan_version();
}
