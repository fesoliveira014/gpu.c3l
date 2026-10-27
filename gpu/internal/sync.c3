module gpu::internal @private;

import gpu;
const uint STAGE_MASK_BITS @private = 0x3FFF;



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

<* True when the submission's slices and wait stages need no state to judge. *>
fn bool submit_desc_valid(SubmitDesc* desc) @private {
    if (desc == null) return false;
    if (desc.command_lists.len > gpu::MAX_SUBMIT_COMMAND_LISTS
        || (desc.command_lists.len > 0 && desc.command_lists.ptr == null)) {
        return false;
    }
    if (desc.completion_waits.len > gpu::MAX_SUBMIT_COMPLETION_WAITS
        || (desc.completion_waits.len > 0 && desc.completion_waits.ptr == null)) {
        return false;
    }
    foreach (wait : desc.completion_waits) {
        if (!submit_wait_stage_mask_valid(wait.before)) return false;
    }
    return true;
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
        case TextureLayout.GENERAL:
            return true;
        default:
            return false;
    }
}
