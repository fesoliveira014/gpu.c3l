module gpu;

import std::math;

<* What happens to an attachment's contents when a render pass begins. *>
enum LoadOp : int {
    LOAD,
    CLEAR,
    DONT_CARE,
}

<* Whether an attachment's contents persist after the render pass ends. *>
enum StoreOp : int {
    STORE,
    DONT_CARE,
}

<* Color clear value. Use rgba for normalized/floating-point formats and
   uint_rgba for unsigned-integer formats. *>
union ClearColor {
    float[4] rgba;
    uint[4]  uint_rgba;
}
$assert ClearColor::size == 16;

<* Depth/stencil clear value; depth in [0, 1]. *>
struct ClearDepthStencil {
    float depth;
    uint  stencil;
}

<* One color attachment and its optional single-sample resolve target. *>
struct ColorTargetDesc {
    TextureHandle texture;
    uint          mip_level;
    uint          array_layer;
    TextureHandle resolve_texture;
    uint          resolve_mip_level;
    uint          resolve_array_layer;
    LoadOp        load_op;
    StoreOp       store_op;
    ClearColor    clear;
}

<* One depth attachment. `clear` applies only with LoadOp.CLEAR. *>
struct DepthTargetDesc {
    TextureHandle     texture;
    LoadOp            load_op;
    StoreOp           store_op;
    ClearDepthStencil clear;
}

<* A dynamic render pass. Every selected attachment must cover the render area. *>
struct RenderPassDesc {
    ColorTargetDesc[] colors;
    DepthTargetDesc*  depth;
    uint              width;
    uint              height;
}

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

fn bool render_pass_desc_valid(RenderPassDesc* desc) @private {
    if (desc == null
        || desc.width == 0
        || desc.height == 0
        || desc.colors.len > MAX_COLOR_ATTACHMENTS
        || (desc.colors.len == 0 && desc.depth == null)) return false;
    if (desc.colors.len > 0 && desc.colors.ptr == null) return false;

    foreach (&target : desc.colors) {
        if (!load_op_valid(target.load_op)
            || !store_op_valid(target.store_op)) return false;
        if (target.resolve_texture != TEXTURE_HANDLE_INVALID
            && target.resolve_texture == target.texture) return false;
    }

    if (desc.depth != null
        && (!load_op_valid(desc.depth.load_op)
            || !store_op_valid(desc.depth.store_op))) return false;
    if (desc.depth != null
        && desc.depth.load_op == LoadOp.CLEAR
        && (!math::is_finite(desc.depth.clear.depth)
            || desc.depth.clear.depth < 0.0f
            || desc.depth.clear.depth > 1.0f)) return false;
    return true;
}

<* Begin a pass without adding synchronization. Viewport and scissor default to
   the full render area. *>
fn void? cmd_begin_render_pass(CommandList* commands, RenderPassDesc* desc) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    if (!render_pass_desc_valid(desc)) return INVALID_ARGUMENT~;
    return encoder.ops.begin_render_pass(encoder, desc);
}

<* End the active render pass without adding synchronization. *>
fn void? cmd_end_render_pass(CommandList* commands) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.end_render_pass(encoder);
}
