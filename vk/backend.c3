module gpu::vk;

import vk;
import vma;

<* Probe the bound Vulkan loader. This is a real `vk` call, so it pulls
   libvulkan.so.1 into the link. Returns the loader's instance-level API
   version, or 0 if the loader is unavailable. *>
fn uint probe_vulkan_version() {
    uint version = 0;
    if (catch vk::try_enumerate_instance_version(&version)) return 0;
    return version;
}

<* Exercise the VMA binding with a CPU-only virtual block (no Vulkan device
   required). This is a real `vma` call, so it pulls libVulkanMemoryAllocator.a
   and stdc++ into the link. Returns true if the block was created and freed. *>
fn bool probe_vma_allocator() {
    vma::VirtualBlockCreateInfo info = { .size = 4096 };
    if (try block = vma::try_create_virtual_block(&info)) {
        defer block.destroy();
        return true;
    }
    return false;
}
