module gpu::internal @private;

import gpu;
import std::math;


fn bool load_op_valid(LoadOp op) @private {
    switch (op) {
        case LoadOp.LOAD:
        case LoadOp.CLEAR:
        case LoadOp.DONT_CARE:
            return true;
        default:
            return false;
    }
}

fn bool store_op_valid(StoreOp op) @private {
    switch (op) {
        case StoreOp.STORE:
        case StoreOp.DONT_CARE:
            return true;
        default:
            return false;
    }
}

fn bool render_pass_desc_host_safe(RenderPassDesc* desc) @private {
    return desc != null
        && desc.colors.len <= gpu::MAX_COLOR_ATTACHMENTS
        && (desc.colors.len == 0 || desc.colors.ptr != null);
}

fn bool render_pass_desc_valid(RenderPassDesc* desc) @private {
    if (!render_pass_desc_host_safe(desc)
        || desc.width == 0
        || desc.height == 0
        || (desc.colors.len == 0 && desc.depth == null)) return false;

    foreach (&target : desc.colors) {
        if (!target.view.is_valid()
            || !load_op_valid(target.load_op)
            || !store_op_valid(target.store_op)) return false;
        if (target.resolve_view.is_valid()
            && target.resolve_view == target.view) return false;
    }

    if (desc.depth != null
        && (!desc.depth.view.is_valid()
            || !load_op_valid(desc.depth.load_op)
            || !store_op_valid(desc.depth.store_op))) return false;
    if (desc.depth != null
        && desc.depth.load_op == LoadOp.CLEAR
        && (!math::is_finite(desc.depth.clear.depth)
            || desc.depth.clear.depth < 0.0f
            || desc.depth.clear.depth > 1.0f)) return false;
    return true;
}
