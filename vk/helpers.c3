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

<* Pure translation of a public execution stage to a synchronization2 stage
   mask. Unit-testable without a device. *>
fn vk::PipelineStageFlags2 stage_to_vk(gpu::Stage stage) {
    switch (stage) {
        case gpu::Stage.HOST:             return vk::PipelineStageFlagBits2.STAGE_2_HOST_BIT;
        case gpu::Stage.TRANSFER:         return vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT;
        case gpu::Stage.COMPUTE_SHADER:   return vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT;
        case gpu::Stage.VERTEX_SHADER:    return vk::PipelineStageFlagBits2.STAGE_2_VERTEX_SHADER_BIT;
        case gpu::Stage.FRAGMENT_SHADER:  return vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT;
        case gpu::Stage.COLOR_ATTACHMENT: return vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        case gpu::Stage.DEPTH_STENCIL:
            return (vk::PipelineStageFlags2)((ulong)vk::PipelineStageFlagBits2.STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                | (ulong)vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT);
        case gpu::Stage.INDIRECT_COMMAND: return vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT;
        case gpu::Stage.PRESENT:          return vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT;
        default:                          return vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT;
    }
}

<* Pure translation of a public hazard to a synchronization2 access mask. *>
fn vk::AccessFlags2 hazard_to_vk_access(gpu::Hazard hazard) {
    switch (hazard) {
        case gpu::Hazard.HOST_WRITE:     return vk::AccessFlagBits2.ACCESS_2_HOST_WRITE_BIT;
        case gpu::Hazard.HOST_READ:      return vk::AccessFlagBits2.ACCESS_2_HOST_READ_BIT;
        case gpu::Hazard.TRANSFER_READ:  return vk::AccessFlagBits2.ACCESS_2_TRANSFER_READ_BIT;
        case gpu::Hazard.TRANSFER_WRITE: return vk::AccessFlagBits2.ACCESS_2_TRANSFER_WRITE_BIT;
        case gpu::Hazard.SHADER_READ:    return vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT;
        case gpu::Hazard.SHADER_WRITE:   return vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT;
        case gpu::Hazard.COLOR_READ:     return vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
        case gpu::Hazard.COLOR_WRITE:    return vk::AccessFlagBits2.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        case gpu::Hazard.DEPTH_READ:     return vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        case gpu::Hazard.DEPTH_WRITE:    return vk::AccessFlagBits2.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        case gpu::Hazard.INDIRECT_READ:  return vk::AccessFlagBits2.ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        case gpu::Hazard.PRESENT_READ:   return vk::AccessFlagBits2.ACCESS_2_MEMORY_READ_BIT;
        default:                         return vk::AccessFlagBits2.ACCESS_2_NONE;
    }
}

<* Pure translation of a public texture layout to a Vulkan image layout. *>
fn vk::ImageLayout layout_to_vk(gpu::TextureLayout layout) {
    switch (layout) {
        case gpu::TextureLayout.UNDEFINED:        return vk::ImageLayout.UNDEFINED;
        case gpu::TextureLayout.GENERAL:          return vk::ImageLayout.GENERAL;
        case gpu::TextureLayout.COLOR_ATTACHMENT: return vk::ImageLayout.COLOR_ATTACHMENT_OPTIMAL;
        case gpu::TextureLayout.DEPTH_STENCIL:    return vk::ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        case gpu::TextureLayout.SHADER_READ:      return vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL;
        case gpu::TextureLayout.TRANSFER_SRC:     return vk::ImageLayout.TRANSFER_SRC_OPTIMAL;
        case gpu::TextureLayout.TRANSFER_DST:     return vk::ImageLayout.TRANSFER_DST_OPTIMAL;
        case gpu::TextureLayout.PRESENT:          return vk::ImageLayout.PRESENT_SRC_KHR;
        default:                                  return vk::ImageLayout.UNDEFINED;
    }
}
