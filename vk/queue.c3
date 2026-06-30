module gpu::vk;

import gpu;
import vk;
import std::core::mem;

struct QueueFamilies {
    uint graphics;
    uint compute;
    uint transfer;
}

<* A graphics+compute family backs both; a transfer-only family is used for
   transfer when present, else the main family is reused. *>
fn QueueFamilies? select_queue_families(vk::PhysicalDevice physical_device) {
    uint count = 0;
    vk::get_physical_device_queue_family_properties(physical_device, &count, null);
    if (count == 0) return gpu::UNSUPPORTED_FEATURE~;

    @pool() {
        vk::QueueFamilyProperties[] families = mem::talloc_array(vk::QueueFamilyProperties, (sz)count);
        vk::get_physical_device_queue_family_properties(physical_device, &count, families.ptr);

        int main_family = -1;
        int transfer_only = -1;
        foreach (i, family : families) {
            uint flags = (uint)family.queue_flags;
            bool graphics = (flags & (uint)vk::QueueFlagBits.GRAPHICS_BIT) != 0;
            bool compute  = (flags & (uint)vk::QueueFlagBits.COMPUTE_BIT) != 0;
            bool transfer = (flags & (uint)vk::QueueFlagBits.TRANSFER_BIT) != 0;
            if (graphics && compute && main_family < 0) main_family = (int)i;
            if (transfer && !graphics && !compute && transfer_only < 0) transfer_only = (int)i;
        }

        if (main_family < 0) return gpu::UNSUPPORTED_FEATURE~;
        uint main = (uint)main_family;
        uint transfer = transfer_only < 0 ? main : (uint)transfer_only;
        return (QueueFamilies){ .graphics = main, .compute = main, .transfer = transfer };
    };
}
