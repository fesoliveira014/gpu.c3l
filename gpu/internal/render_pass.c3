module gpu::internal @private;

import gpu;

fn bool render_pass_desc_host_safe(RenderPassDesc* desc) @private {
    return desc != null
        && desc.colors.len <= gpu::MAX_COLOR_ATTACHMENTS
        && (desc.colors.len == 0 || desc.colors.ptr != null);
}

fn bool color_state_host_safe(ColorState* state) @private {
    return state != null
        && state.targets.len <= gpu::MAX_COLOR_ATTACHMENTS
        && (state.targets.len == 0 || state.targets.ptr != null);
}

fn bool graphics_state_host_safe(GraphicsState* state) @private {
    return state != null && color_state_host_safe(&state.color);
}
