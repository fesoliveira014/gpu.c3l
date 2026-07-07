module gpu;

// Command-area descriptor types (render targets and copies). The command-list
// recorder and submit descriptors depend on backend state and arrive with the
// command milestone, not here.

enum LoadOp : int {
    LOAD,
    CLEAR,
    DONT_CARE,
}

enum StoreOp : int {
    STORE,
    DONT_CARE,
}

enum CommandState : int {
    RECORDING,
    RECORDING_RENDER_PASS,
    EXECUTABLE,
    SUBMITTED,
}

const uint MAX_RECORDING_CONTEXTS = 16;

struct Vec3u {
    uint x;
    uint y;
    uint z;
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

struct DrawIndexedIndirectCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int  vertex_offset;
    uint first_instance;
}
$assert DrawIndexedIndirectCommand::size == 20;

struct DispatchIndirectCommand {
    uint x;
    uint y;
    uint z;
}
$assert DispatchIndirectCommand::size == 12;

struct ClearColor {
    float[4] rgba;
}

struct ClearDepthStencil {
    float depth;
    uint  stencil;
}

struct ColorTargetDesc {
    TextureHandle texture;
    uint          mip_level;
    uint          array_layer;
    LoadOp        load_op;
    StoreOp       store_op;
    ClearColor    clear;
}

struct DepthTargetDesc {
    TextureHandle     texture;
    LoadOp            load_op;
    StoreOp           store_op;
    ClearDepthStencil clear;
}

struct RenderPassDesc {
    ColorTargetDesc[] colors;
    DepthTargetDesc*  depth;
    uint              width;
    uint              height;
}

// A recording session. Backend-opaque: the backend command buffer lives behind
// `backend`. Carries its owning Device* so cmd_* reach the vtable without a
// Device parameter.
struct CommandList {
    Device*      device;
    void*        backend;
    QueueKind    queue;
    CommandState state;
    // Backend redundant-bind tracking; callers must not touch.
    PipelineHandle last_pipeline;
    uint           last_entry;
}

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

<* Finish recording; the list becomes submittable exactly once. *>
fn void? end_commands(Device* device, CommandList* commands) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.end_commands(device, commands);
}

<* Record a buffer-to-buffer copy; transfer-side barriers are the caller's. *>
fn void? cmd_copy_buffer(CommandList* commands, BufferCopyDesc* desc) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_copy_buffer(commands, desc);
}

<* Record a 32-bit-pattern fill over [offset, offset + size); size passes through to the backend untouched. *>
fn void? cmd_fill_buffer(CommandList* commands, BufferHandle buffer, usz offset, usz size, uint value) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_fill_buffer(commands, buffer, offset, size, value);
}

<* The texture's tracked layout must be TRANSFER_DST; the caller records the
   surrounding barriers. *>
fn void? cmd_copy_buffer_to_texture(CommandList* commands, BufferTextureCopyDesc* desc) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_copy_buffer_to_texture(commands, desc);
}

<* The texture's tracked layout must be TRANSFER_SRC. *>
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

<* Begin dynamic rendering into the desc's color targets. Viewport and scissor
   are set to the full pass dimensions. *>
fn void? cmd_begin_render_pass(CommandList* commands, RenderPassDesc* desc) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_begin_render_pass(commands, desc);
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
