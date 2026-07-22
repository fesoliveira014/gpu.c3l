module gpu::internal @private;

import gpu;
const uint STAGE_MASK_BITS @private = 0x01FF;



fn bool stage_mask_valid(StageMask stages) @private {
    uint raw = (uint)stages;
    if (raw == 0 || (raw & STAGE_MASK_BITS) != raw) return false;
    if (stages.all && raw != 1) return false;
    if (stages.present && raw != 1u << 8) return false;
    return true;
}

fn bool submit_wait_stage_mask_valid(StageMask stages) @private {
    return stage_mask_valid(stages) && !stages.host && !stages.present;
}

fn bool texture_layout_valid(TextureLayout layout) @private {
    switch (layout) {
        case TextureLayout.UNDEFINED:
        case TextureLayout.TRANSFER_SOURCE:
        case TextureLayout.TRANSFER_DESTINATION:
        case TextureLayout.SAMPLED:
        case TextureLayout.STORAGE:
        case TextureLayout.COLOR_ATTACHMENT:
        case TextureLayout.DEPTH_ATTACHMENT:
        case TextureLayout.PRESENT:
            return true;
        default:
            return false;
    }
}
