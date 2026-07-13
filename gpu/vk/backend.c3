module gpu::vk @private;

import vk;
import vma;

<* Loader instance version, or 0 if unavailable. A real `vk` call, so it pulls
   libvulkan.so.1 into the link. *>
fn uint probe_vulkan_version() @private {
    uint version = 0;
    if (catch vk::try_enumerate_instance_version(&version)) return 0;
    return version;
}

<* CPU-only VMA virtual-block round-trip. A real `vma` call, so it pulls
   libVulkanMemoryAllocator.a and stdc++ into the link. *>
fn bool probe_vma_allocator() @private {
    vma::VirtualBlockCreateInfo info = { .size = 4096 };
    if (try block = vma::try_create_virtual_block(&info)) {
        defer block.destroy();
        return true;
    }
    return false;
}
