module gpu;

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

// Shader-writable indirect argument layouts; GLSL definitions are generated
// in include/shaders/generated/shader_abi.glsl.
struct DrawIndirectCommand {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
}
$assert DrawIndirectCommand::size == 16;

<* Shader-writable arguments for indexed indirect draws. *>
struct DrawIndexedIndirectCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int  vertex_offset;
    uint first_instance;
}
$assert DrawIndexedIndirectCommand::size == 20;

<* Shader-writable workgroup counts for cmd_dispatch_indirect. *>
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

<* A recording command token. Successful end consumes it. *>
struct CommandList {
    Device            device;
    CommandListHandle handle;
}
$assert CommandList::size <= ulong::size * 3;

<* A one-shot executable command token. Submit or discard consumes it. *>
struct ExecutableCommandList {
    Device            device;
    CommandListHandle handle;
}
$assert ExecutableCommandList::size <= ulong::size * 3;

<* Byte-range copy for cmd_copy_buffer; src and dst must have equal size and
   must not overlap. *>
struct BufferCopyDesc {
    GpuSpan src;
    GpuSpan dst;
}

// Zero-value defaults: extent 0 is the full mip level, layer_count 0 is one
// layer, and row_length_texels 0 means tightly packed rows.
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

fn void? validate_dispatch_groups(Vec3u groups, Vec3u limit) @private {
    if (groups.x > limit.x || groups.y > limit.y || groups.z > limit.z) {
        return INVALID_ARGUMENT~;
    }
}

fn usz? indirect_argument_bytes(
    usz argument_size,
    uint count,
    uint max_count,
) @private {
    if (count > max_count) return INVALID_ARGUMENT~;
    if (count == 0) return 0;
    usz size_count = (usz)count;
    if (argument_size > usz::max / size_count) return INVALID_ARGUMENT~;
    return argument_size * size_count;
}

fn DeviceOperation? command_operation(
    CommandList* commands,
    bool allow_lost = false,
) @private {
    if (commands == null) return INVALID_HANDLE~;
    DeviceOperation operation = has_retained_command_pin(commands)
        ? borrow_retained_device_operation(&commands.device, allow_lost)!
        : begin_device_operation(&commands.device, allow_lost)!;
    if (commands.handle.owner != resource_owner_tag(
        operation.data.resource_owner,
        ResourceOwnerKind.COMMAND,
    )) {
        end_device_operation(&operation);
        return INVALID_HANDLE~;
    }
    return operation;
}

fn DeviceOperation? executable_command_operation(
    ExecutableCommandList* commands,
    bool allow_lost = false,
) @private {
    if (commands == null) return INVALID_HANDLE~;
    DeviceOperation operation = has_retained_executable_command_pin(commands)
        ? borrow_retained_device_operation(&commands.device, allow_lost)!
        : begin_device_operation(&commands.device, allow_lost)!;
    if (commands.handle.owner != resource_owner_tag(
        operation.data.resource_owner,
        ResourceOwnerKind.COMMAND,
    )) {
        end_device_operation(&operation);
        return INVALID_HANDLE~;
    }
    return operation;
}

<* Begin recording on a selected queue. The token retains its device until
   successful end followed by submit, or until discard. *>
fn CommandList? begin_commands(Queue queue) {
    Device device = queue.device;
    DeviceOperation operation = begin_device_operation(&device)!;
    defer end_device_operation(&operation);
    validate_queue(&operation, &device, queue)!;
    BackendVTable* vtable = operation.vtable;
    CommandList commands;
    vtable.begin_commands(queue, &commands)!;
    if (!retain_command_pin_lease(&operation, &commands)) {
        vtable.discard_commands(&commands.device, commands.handle, false)!;
        return INVALID_HANDLE~;
    }
    retain_device_children(&operation);
    transfer_device_operation_pin(&operation);
    return commands;
}

<* Finish and consume a recording token. Failure leaves it recording. *>
fn ExecutableCommandList? end_commands(CommandList* commands) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    operation.vtable.end_commands(commands)!;
    ExecutableCommandList executable = {
        .device = commands.device,
        .handle = commands.handle,
    };
    *commands = {};
    return executable;
}

<* Consume an unfinished recording, including after device loss. *>
fn void? discard_commands(CommandList* commands) {
    DeviceOperation operation = command_operation(commands, true)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    CommandList consumed = *commands;
    ulong canceled_tickets = vtable.discard_commands(
        &consumed.device,
        consumed.handle,
        false,
    )!;
    *commands = {};
    bool released_lease = release_command_pin_lease(&operation, &consumed);
    if (released_lease || !operation.owns_pin) {
        release_device_children(&operation, canceled_tickets + 1);
        (void)release_retained_device_pins(&operation);
    }
}

<* Consume an executable token without submitting it. *>
fn void? discard_executable_commands(ExecutableCommandList* commands) {
    DeviceOperation operation = executable_command_operation(commands, true)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    ExecutableCommandList consumed = *commands;
    ulong canceled_tickets = vtable.discard_commands(
        &consumed.device,
        consumed.handle,
        true,
    )!;
    *commands = {};
    bool released_lease = release_executable_command_pin_lease(
        &operation,
        &consumed,
    );
    if (released_lease || !operation.owns_pin) {
        release_device_children(&operation, canceled_tickets + 1);
        (void)release_retained_device_pins(&operation);
    }
}

<* Record a buffer-to-buffer copy; transfer-side barriers are the caller's. *>
fn void? cmd_copy_buffer(CommandList* commands, BufferCopyDesc* desc) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    if (desc == null) return INVALID_ARGUMENT~;
    return vtable.cmd_copy_buffer(commands, desc);
}

<* Record a 32-bit-pattern fill over dst; its offset and size must be 4-byte
   multiples. *>
fn void? cmd_fill_buffer(CommandList* commands, GpuSpan dst, uint value) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    return vtable.cmd_fill_buffer(commands, dst, value);
}

<* The texture's effective layout must be TRANSFER_DST; the caller records
   the surrounding barriers. *>
fn void? cmd_copy_buffer_to_texture(CommandList* commands, BufferTextureCopyDesc* desc) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    if (desc == null) return INVALID_ARGUMENT~;
    return vtable.cmd_copy_buffer_to_texture(commands, desc);
}

<* The texture's effective layout must be TRANSFER_SRC. *>
fn void? cmd_copy_texture_to_buffer(CommandList* commands, TextureBufferCopyDesc* desc) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    if (desc == null) return INVALID_ARGUMENT~;
    return vtable.cmd_copy_texture_to_buffer(commands, desc);
}

<* Record an execution/memory barrier over one GPU span. *>
fn void? cmd_buffer_barrier(CommandList* commands, BufferBarrier* barrier) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    if (barrier == null) return INVALID_ARGUMENT~;
    return vtable.cmd_buffer_barrier(commands, barrier);
}

<* Record a device-wide execution/memory barrier with no resource transition. *>
fn void? cmd_global_barrier(CommandList* commands, GlobalBarrier* barrier) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    if (barrier == null) return INVALID_ARGUMENT~;
    return vtable.cmd_global_barrier(commands, barrier);
}

<* Record a texture barrier; old_layout must match the texture's tracked layout. *>
fn void? cmd_texture_barrier(CommandList* commands, TextureBarrier* barrier) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    if (barrier == null) return INVALID_ARGUMENT~;
    return vtable.cmd_texture_barrier(commands, barrier);
}

<* Dispatch a compute pipeline with a single root GPU address pushed as the
   push constant at offset 0. Every allocation reachable from the root must
   admit the recording queue's role. Counts may be zero and may not exceed
   DeviceCaps.max_compute_work_group_count on any axis. *>
fn void? cmd_dispatch(CommandList* commands, PipelineHandle pipeline, GpuAddress root, Vec3u groups) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    return vtable.cmd_dispatch(commands, pipeline, root, groups);
}

<* Begin dynamic rendering into the desc's targets. Every selected attachment
   must cover the pass dimensions, and the color count must not exceed
   DeviceCaps.max_color_attachments. Viewport and scissor cover the full
   pass dimensions. *>
fn void? cmd_begin_render_pass(CommandList* commands, RenderPassDesc* desc) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    if (desc == null) return INVALID_ARGUMENT~;
    return vtable.cmd_begin_render_pass(commands, desc);
}

<* Set one pass-bounded dynamic viewport for subsequent draws. Requires an
   active render pass; malformed values fault INVALID_ARGUMENT. *>
fn void? cmd_set_viewport(CommandList* commands, Viewport* viewport) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    if (viewport == null) return INVALID_ARGUMENT~;
    return vtable.cmd_set_viewport(commands, viewport);
}

<* Set one pass-bounded dynamic scissor for subsequent draws. Requires an
   active render pass; malformed values fault INVALID_ARGUMENT. *>
fn void? cmd_set_scissor(CommandList* commands, ScissorRect* scissor) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    if (scissor == null) return INVALID_ARGUMENT~;
    return vtable.cmd_set_scissor(commands, scissor);
}

<* Counterpart of cmd_begin_render_pass. *>
fn void? cmd_end_render_pass(CommandList* commands) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    return vtable.cmd_end_render_pass(commands);
}

<* Draw inside a render pass, pushing the vertex and fragment root addresses.
   Every allocation reachable from either root must admit the recording queue's
   role. *>
fn void? cmd_draw(CommandList* commands, PipelineHandle pipeline, GpuAddress vertex_root, GpuAddress fragment_root, uint vertex_count, uint instance_count) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    return vtable.cmd_draw(commands, pipeline, vertex_root, fragment_root, vertex_count, instance_count);
}

<* Draw indexed geometry; indices are read from the start of index_span.
   Every allocation reachable from either root must admit the recording queue's
   role. *>
fn void? cmd_draw_indexed(CommandList* commands, PipelineHandle pipeline, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan index_span, uint index_count, uint instance_count, IndexType index_type = IndexType.U32) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    return vtable.cmd_draw_indexed(commands, pipeline, vertex_root, fragment_root, index_span, index_count, instance_count, index_type);
}

<* Draw from GPU-written DrawIndirectCommand arguments at the span's offset.
   The args buffer needs indirect usage; the root pair applies to every draw
   in the multi-draw (per-draw variation via gl_DrawID). Ordering against the
   argument writes is the caller's barrier. Every allocation reachable from
   either root must admit the recording queue's role. draw_count may not exceed
   DeviceCaps.max_draw_indirect_count. *>
fn void? cmd_draw_indirect(CommandList* commands, PipelineHandle pipeline, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, uint draw_count) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    return vtable.cmd_draw_indirect(commands, pipeline, vertex_root, fragment_root, args, draw_count);
}

<* Indexed variant over DrawIndexedIndirectCommand arguments. Index-range
   bounds are read from GPU memory at execution time and are not validated;
   out-of-bounds indices follow device robustness. draw_count may not exceed
   DeviceCaps.max_draw_indirect_count. Every allocation reachable from either
   root must admit the recording queue's role. *>
fn void? cmd_draw_indexed_indirect(CommandList* commands, PipelineHandle pipeline, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, uint draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    return vtable.cmd_draw_indexed_indirect(commands, pipeline, vertex_root, fragment_root, args, draw_count, index_span, index_type);
}

<* Count-buffer variant: the GPU-written uint at count_span limits execution
   to max_draw_count and may not exceed DeviceCaps.max_draw_indirect_count.
   The args span must hold max_draw_count entries. Requires
   DeviceCaps.draw_indirect_count. Every allocation reachable from either root
   must admit the recording queue's role. *>
fn void? cmd_draw_indexed_indirect_count(CommandList* commands, PipelineHandle pipeline, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, GpuSpan count_span, uint max_draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    return vtable.cmd_draw_indexed_indirect_count(commands, pipeline, vertex_root, fragment_root, args, count_span, max_draw_count, index_span, index_type);
}

<* Dispatch from GPU-written DispatchIndirectCommand arguments. Each group
   count must not exceed the corresponding
   DeviceCaps.max_compute_work_group_count component. Every allocation
   reachable from the root must admit the recording queue's role. *>
fn void? cmd_dispatch_indirect(CommandList* commands, PipelineHandle pipeline, GpuAddress root, GpuSpan args) {
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    return vtable.cmd_dispatch_indirect(commands, pipeline, root, args);
}
