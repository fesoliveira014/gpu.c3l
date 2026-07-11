module gpu;

// Command recording: render-target and copy descriptors, the command-list
// recorder, and recording contexts.

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

// Two slots are internal (default + helper contexts); 15 remain for users.
const uint MAX_RECORDING_CONTEXTS = 17;

<* Unsigned 3-vector; dispatches take their workgroup counts as one. *>
struct Vec3u {
    uint x;
    uint y;
    uint z;
}

<* One pass-bounded dynamic viewport. All fields must be finite; width and
   height are positive, while both depth endpoints are in [0, 1]. Set
   max_depth to 1.0f for the conventional full depth range. *>
struct Viewport {
    float x;
    float y;
    float width;
    float height;
    float min_depth;
    float max_depth;
}

<* One pass-bounded dynamic scissor. Zero extent is a valid empty clip. *>
struct ScissorRect {
    int x;
    int y;
    int width;
    int height;
}

// Indirect argument layouts, byte-identical to Vulkan's and shader-writable
// (GLSL twins in include/shaders/generated/shader_abi.glsl).
struct DrawIndirectCommand {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
}
$assert DrawIndirectCommand::size == 16;

<* Arguments for the indexed indirect draws; Vulkan layout, shader-writable. *>
struct DrawIndexedIndirectCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int  vertex_offset;
    uint first_instance;
}
$assert DrawIndexedIndirectCommand::size == 20;

<* Workgroup counts for cmd_dispatch_indirect; Vulkan layout, shader-writable. *>
struct DispatchIndirectCommand {
    uint x;
    uint y;
    uint z;
}
$assert DispatchIndirectCommand::size == 12;

<* Color clear value; rgba is interpreted in the target's format. *>
struct ClearColor {
    float[4] rgba;
}

<* Depth/stencil clear value; depth in [0, 1]. *>
struct ClearDepthStencil {
    float depth;
    uint  stencil;
}

<* One color attachment of a render pass: a single mip/layer of the texture.
   `clear` applies only with LoadOp.CLEAR. *>
struct ColorTargetDesc {
    TextureHandle texture;
    uint          mip_level;
    uint          array_layer;
    LoadOp        load_op;
    StoreOp       store_op;
    ClearColor    clear;
}

<* Depth attachment of a render pass; `clear` applies only with LoadOp.CLEAR. *>
struct DepthTargetDesc {
    TextureHandle     texture;
    LoadOp            load_op;
    StoreOp           store_op;
    ClearDepthStencil clear;
}

<* A dynamic-rendering pass. width/height set the render area and every
   attachment must cover it. *>
struct RenderPassDesc {
    ColorTargetDesc[] colors;
    // Null means no depth attachment.
    DepthTargetDesc*  depth;
    uint              width;
    uint              height;
}

// A small recording token. Mutable lifecycle and backend state are device-owned;
// the owner pointer keeps cmd_* dispatch source-compatible.
struct CommandList {
    Device*           device;
    CommandListHandle handle;
}
$assert CommandList::size <= 16;

<* Byte-range copy for cmd_copy_buffer; src and dst ranges must not overlap. *>
struct BufferCopyDesc {
    BufferHandle src;
    BufferHandle dst;
    usz          src_offset;
    usz          dst_offset;
    usz          size;
}

// Copy regions mirror vk::BufferImageCopy. Zero-value defaults: extent 0 is
// the full mip level, layer_count 0 is one layer, row_length_texels 0 means
// tightly packed rows.
struct BufferTextureCopyDesc {
    GpuSpan       src;
    uint          row_length_texels;
    TextureHandle texture;
    uint          mip;
    uint          base_layer;
    uint          layer_count;
    uint          x;
    uint          y;
    uint          width;
    uint          height;
}

<* Texture-to-buffer readback region; same zero-value defaults as
   BufferTextureCopyDesc. *>
struct TextureBufferCopyDesc {
    TextureHandle texture;
    GpuSpan       dst;
    uint          row_length_texels;
    uint          mip;
    uint          base_layer;
    uint          layer_count;
    uint          x;
    uint          y;
    uint          width;
    uint          height;
}

<* Create a recording context: per-frame command pools confined to one
   thread at a time. Ownership: release with destroy_recording_context. *>
fn RecordingContextHandle? create_recording_context(Device* device) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.create_recording_context(device);
}

<* Counterpart of create_recording_context; the context's command lists must be retired first. *>
fn void? destroy_recording_context(Device* device, RecordingContextHandle ctx) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.destroy_recording_context(device, ctx);
}

<* Begin recording a command list on the given queue. ctx = {} selects the
   device's built-in context; pass a created context to record from another
   thread. Ownership: end with end_commands, then submit; the backing command
   buffer is freed when its frame slot's pool is reset. *>
fn CommandList? begin_commands(Device* device, QueueKind queue, RecordingContextHandle ctx = {}) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    CommandList commands;
    device.vtable.begin_commands(device, queue, ctx, &commands)!;
    return commands;
}

<* Finish recording through the list's owner; it becomes submittable once. *>
fn void? end_commands(CommandList* commands) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.end_commands(commands);
}

<* Record a buffer-to-buffer copy; transfer-side barriers are the caller's. *>
fn void? cmd_copy_buffer(CommandList* commands, BufferCopyDesc* desc) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_copy_buffer(commands, desc);
}

<* Record a 32-bit-pattern fill over [offset, offset + size); offset and size must be 4-byte multiples, size 0 fills to the end. *>
fn void? cmd_fill_buffer(CommandList* commands, BufferHandle buffer, usz offset, usz size, uint value) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_fill_buffer(commands, buffer, offset, size, value);
}

<* The texture's effective layout must be TRANSFER_DST; the caller records
   the surrounding barriers. *>
fn void? cmd_copy_buffer_to_texture(CommandList* commands, BufferTextureCopyDesc* desc) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_copy_buffer_to_texture(commands, desc);
}

<* The texture's effective layout must be TRANSFER_SRC. *>
fn void? cmd_copy_texture_to_buffer(CommandList* commands, TextureBufferCopyDesc* desc) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_copy_texture_to_buffer(commands, desc);
}

<* Record an execution/memory barrier over a buffer range. *>
fn void? cmd_buffer_barrier(CommandList* commands, BufferBarrier* barrier) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_buffer_barrier(commands, barrier);
}

<* Record a device-wide execution/memory barrier with no resource transition. *>
fn void? cmd_global_barrier(CommandList* commands, GlobalBarrier* barrier) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_global_barrier(commands, barrier);
}

<* Record a texture barrier; old_layout must match the texture's tracked layout. *>
fn void? cmd_texture_barrier(CommandList* commands, TextureBarrier* barrier) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_texture_barrier(commands, barrier);
}

<* Dispatch a compute pipeline with a single root GPU address pushed as the
   push constant at offset 0. *>
fn void? cmd_dispatch(CommandList* commands, PipelineHandle pipeline, GpuAddress root, Vec3u groups) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_dispatch(commands, pipeline, root, groups);
}

<* Begin dynamic rendering into the desc's targets. Every selected attachment
   must cover the pass dimensions, and the color count must not exceed
   DeviceCaps.max_color_attachments. Viewport and scissor cover the full
   pass dimensions. *>
fn void? cmd_begin_render_pass(CommandList* commands, RenderPassDesc* desc) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_begin_render_pass(commands, desc);
}

<* Set one pass-bounded dynamic viewport for subsequent draws. Requires an
   active render pass; malformed values fault INVALID_ARGUMENT. *>
fn void? cmd_set_viewport(CommandList* commands, Viewport* viewport) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_set_viewport(commands, viewport);
}

<* Set one pass-bounded dynamic scissor for subsequent draws. Requires an
   active render pass; malformed values fault INVALID_ARGUMENT. *>
fn void? cmd_set_scissor(CommandList* commands, ScissorRect* scissor) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_set_scissor(commands, scissor);
}

<* Counterpart of cmd_begin_render_pass. *>
fn void? cmd_end_render_pass(CommandList* commands) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_end_render_pass(commands);
}

<* Draw inside a render pass, pushing the vertex and fragment root addresses. *>
fn void? cmd_draw(CommandList* commands, PipelineHandle pipeline, GpuAddress vertex_root, GpuAddress fragment_root, uint vertex_count, uint instance_count) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_draw(commands, pipeline, vertex_root, fragment_root, vertex_count, instance_count);
}

<* Draw indexed geometry; indices are read from the start of index_span. *>
fn void? cmd_draw_indexed(CommandList* commands, PipelineHandle pipeline, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan index_span, uint index_count, uint instance_count, IndexType index_type = IndexType.U32) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_draw_indexed(commands, pipeline, vertex_root, fragment_root, index_span, index_count, instance_count, index_type);
}

<* Draw from GPU-written DrawIndirectCommand arguments at the span's offset.
   The args buffer needs indirect usage; the root pair applies to every draw
   in the multi-draw (per-draw variation via gl_DrawID). Ordering against the
   argument writes is the caller's barrier. *>
fn void? cmd_draw_indirect(CommandList* commands, PipelineHandle pipeline, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, uint draw_count) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_draw_indirect(commands, pipeline, vertex_root, fragment_root, args, draw_count);
}

<* Indexed variant over DrawIndexedIndirectCommand arguments. Index-range
   bounds are read from GPU memory at execution time and are not validated;
   out-of-bounds indices follow device robustness. *>
fn void? cmd_draw_indexed_indirect(CommandList* commands, PipelineHandle pipeline, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, uint draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_draw_indexed_indirect(commands, pipeline, vertex_root, fragment_root, args, draw_count, index_span, index_type);
}

<* Count-buffer variant: the GPU-written uint at count_span limits the draw
   count up to max_draw_count. Requires DeviceCaps.draw_indirect_count. *>
fn void? cmd_draw_indexed_indirect_count(CommandList* commands, PipelineHandle pipeline, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, GpuSpan count_span, uint max_draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_draw_indexed_indirect_count(commands, pipeline, vertex_root, fragment_root, args, count_span, max_draw_count, index_span, index_type);
}

<* Dispatch from GPU-written DispatchIndirectCommand arguments. *>
fn void? cmd_dispatch_indirect(CommandList* commands, PipelineHandle pipeline, GpuAddress root, GpuSpan args) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_dispatch_indirect(commands, pipeline, root, args);
}
