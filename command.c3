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
    EXECUTABLE,
    SUBMITTED,
}

struct Vec3u {
    uint x;
    uint y;
    uint z;
}

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

<* Begin recording a command list on the given queue.
   Ownership: end with end_commands, then submit; the backing command buffer is
   freed when its frame slot's pool is reset. *>
fn CommandList? begin_commands(Device* device, QueueKind queue) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    CommandList commands;
    device.vtable.begin_commands(device, queue, &commands)!;
    return commands;
}

fn void? end_commands(Device* device, CommandList* commands) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.end_commands(device, commands);
}

fn void? cmd_copy_buffer(CommandList* commands, BufferCopyDesc* desc) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_copy_buffer(commands, desc);
}

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

fn void? cmd_buffer_barrier(CommandList* commands, BufferBarrier* barrier) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_buffer_barrier(commands, barrier);
}

fn void? cmd_global_barrier(CommandList* commands, GlobalBarrier* barrier) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_global_barrier(commands, barrier);
}

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
