module gpu::internal @private;

import gpu;

<* True when the render pass description fits the fixed color array, names a
   defined depth resolve mode when it resolves depth, gives a zero extent in
   both dimensions or neither, and has an attachment to take a zero extent
   from. *>
fn bool render_pass_desc_valid(RenderPassDesc* desc) @private {
    if (desc.colors.len > gpu::MAX_COLOR_ATTACHMENTS) return false;
    if (desc.colors.len > 0 && desc.colors.ptr == null) return false;
    if ((desc.width == 0) != (desc.height == 0)) return false;
    if (desc.depth != null && desc.depth.resolve_view.is_valid()) {
        switch (desc.depth.resolve_mode) {
            case DepthResolveMode.SAMPLE_ZERO:
            case DepthResolveMode.MIN:
            case DepthResolveMode.MAX:
                break;
            default:
                return false;
        }
    }
    if (desc.width == 0 && desc.colors.len == 0 && desc.depth == null) return false;
    return true;
}

<* True when the color state fits the fixed lowered arrays. *>
fn bool color_state_valid(ColorState* state) @private {
    return state.targets.len <= gpu::MAX_COLOR_ATTACHMENTS
        && (state.targets.len == 0 || state.targets.ptr != null);
}

<* True when the span can hold an inline root payload: a multiple of four
   bytes, at most INLINE_ROOT_CAPACITY. *>
fn bool inline_root_valid(char[] bytes) @private {
    return bytes.len % 4 == 0 && bytes.len <= gpu::INLINE_ROOT_CAPACITY;
}
