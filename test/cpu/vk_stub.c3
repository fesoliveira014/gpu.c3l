module gpu::vk;

import gpu;

// CPU-lane stub: satisfies the public module's backend routing so it
// compiles without the real backend or native libraries. Creating a device
// through it faults; the CPU tests never do.
fn gpu::Device? create_vk_device(gpu::DeviceDesc* desc) {
    return gpu::UNSUPPORTED_BACKEND~;
}

fn bool probe_vma_allocator() => false;

fn uint probe_vulkan_version() => 0;
