module gpu::vk;

import gpu;
import vk;

// Vulkan 1.3 in the (variant<<29)|(major<<22)|(minor<<12)|patch layout. The
// binding ships no API_VERSION_1_3 constant.
const uint API_VERSION_1_3 = (1u << 22) | (3u << 12);

<* Map a vk::Result to a gpu fault. SUCCESS is the only non-fault value. *>
fn void? check(vk::Result result) {
    switch (result) {
        case vk::Result.SUCCESS:
            return;
        case vk::Result.ERROR_OUT_OF_HOST_MEMORY:
            return gpu::OUT_OF_HOST_MEMORY~;
        case vk::Result.ERROR_OUT_OF_DEVICE_MEMORY:
            return gpu::OUT_OF_DEVICE_MEMORY~;
        case vk::Result.ERROR_DEVICE_LOST:
            return gpu::DEVICE_LOST~;
        case vk::Result.ERROR_FEATURE_NOT_PRESENT:
        case vk::Result.ERROR_EXTENSION_NOT_PRESENT:
            return gpu::UNSUPPORTED_FEATURE~;
        default:
            return gpu::DEVICE_LOST~;
    }
}
