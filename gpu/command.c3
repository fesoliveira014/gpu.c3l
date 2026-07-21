module gpu;

<* Generated-work record kind a scratch reservation serves. *>
enum GeneratedWorkKind : int {
    DRAW,
    DRAW_INDEXED,
    DISPATCH,
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

<* A recording command token. Successful end consumes it. *>
struct CommandList {
    Device            device;
    CommandListHandle handle;
    void*             encoder;
}
$assert CommandList::size <= ulong::size * 4;

<* A one-shot executable command token. Submit or discard consumes it. *>
struct ExecutableCommandList {
    Device            device;
    CommandListHandle handle;
    void*             encoder;
}
$assert ExecutableCommandList::size <= ulong::size * 4;

<* Cold reservation for one generated-work pipeline and command kind on the
   calling thread's device context. Count bounds each generated call; buffer
   count bounds simultaneously retained calls. *>
struct GeneratedScratchDesc {
    PipelineHandle    pipeline;
    GeneratedWorkKind kind;
    uint              max_commands_per_list;
    uint              preprocess_buffer_count;
}

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

fn usz? generated_work_bytes(
    usz record_size,
    uint count,
    uint max_count,
) @private {
    return indirect_argument_bytes(record_size, count, max_count);
}

fn CommandEncoderState*? command_encoder(
    Device device,
    CommandListHandle handle,
    void* opaque,
    CommandEncoderPhase phase,
    bool allow_lost = false,
) @private {
    if (opaque == null || !device.is_valid() || device.reserved != 0) {
        return INVALID_HANDLE~;
    }
    if (device.index >= MAX_DEVICES || handle.index >= MAX_DEVICE_COMMANDS
        || !handle.is_valid()) {
        return INVALID_HANDLE~;
    }
    CommandEncoderState* encoder = (CommandEncoderState*)opaque;
    if (encoder != &device_slots[device.index].command_encoders[handle.index]) {
        return INVALID_HANDLE~;
    }
    if (encoder.device != device || encoder.handle != handle
        || encoder.ops == null
        || encoder.backend_state == null || encoder.backend_command == null
        || encoder.device_lost == null) {
        return INVALID_HANDLE~;
    }
    if (encoder.phase != phase) {
        if ((phase == CommandEncoderPhase.RECORDING
                && encoder.phase == CommandEncoderPhase.EXECUTABLE)
            || (phase == CommandEncoderPhase.EXECUTABLE
                && encoder.phase == CommandEncoderPhase.RECORDING)) {
            return COMMAND_RECORDING_ERROR~;
        }
        return INVALID_HANDLE~;
    }
    if (!allow_lost
        && encoder.device_lost.load(AtomicOrdering.ACQUIRE)) {
        return DEVICE_LOST~;
    }
    return encoder;
}

fn CommandEncoderState*? recording_encoder(
    CommandList* commands,
    bool allow_lost = false,
) @private {
    if (commands == null) return INVALID_HANDLE~;
    CommandEncoderState* encoder = command_encoder(
        commands.device,
        commands.handle,
        commands.encoder,
        CommandEncoderPhase.RECORDING,
        allow_lost,
    )!;
    if (!allow_lost) note_command_recording_call();
    return encoder;
}

fn CommandEncoderState*? executable_encoder(
    ExecutableCommandList* commands,
    bool allow_lost = false,
) @private {
    if (commands == null) return INVALID_HANDLE~;
    return command_encoder(
        commands.device,
        commands.handle,
        commands.encoder,
        CommandEncoderPhase.EXECUTABLE,
        allow_lost,
    );
}

fn CommandEncoderState*? publish_command_encoder(
    DeviceOperation* operation,
    CommandList* commands,
    CommandBackendContext* backend,
) @private {
    if (operation == null || operation.slot == null || operation.data == null
        || commands == null || backend == null || backend.state == null
        || backend.command == null || operation.data.command_ops == null) {
        return BACKEND_ERROR~;
    }
    if (commands.handle.index >= MAX_DEVICE_COMMANDS) return BACKEND_ERROR~;
    CommandEncoderState* encoder =
        &operation.slot.command_encoders[commands.handle.index];
    if (encoder.phase != CommandEncoderPhase.INACTIVE) return BACKEND_ERROR~;
    *encoder = {
        .device          = commands.device,
        .handle          = commands.handle,
        .ops             = operation.data.command_ops,
        .backend_state   = backend.state,
        .backend_command = backend.command,
        .device_lost     = &operation.slot.lost,
        .phase           = CommandEncoderPhase.RECORDING,
    };
    note_command_policy_selection();
    commands.encoder = encoder;
    return encoder;
}

fn void transfer_command_encoder_to_executable(
    CommandEncoderState* encoder,
) @private {
    encoder.phase = CommandEncoderPhase.EXECUTABLE;
}

fn void invalidate_command_encoder(CommandEncoderState* encoder) @private {
    *encoder = {};
}

fn DeviceOperation? command_operation(
    CommandList* commands,
    bool allow_lost = false,
) @private {
    note_command_device_registry_resolution();
    note_command_retained_pin_operation();
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
    note_command_device_registry_resolution();
    note_command_retained_pin_operation();
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

fn bool generated_work_kind_valid(GeneratedWorkKind kind) @private {
    switch (kind) {
        case GeneratedWorkKind.DRAW:
        case GeneratedWorkKind.DRAW_INDEXED:
        case GeneratedWorkKind.DISPATCH:
            return true;
        default:
            return false;
    }
}

<* Reserve driver-compatible generated-command preprocess storage outside
   command recording. Replacement requires a quiescent calling-thread device
   context; insufficient warm capacity faults GENERATED_SCRATCH_EXHAUSTED. *>
fn void? reserve_generated_scratch(
    Queue queue,
    GeneratedScratchDesc* desc,
) {
    Device device = queue.device;
    DeviceOperation operation = begin_device_operation(&device)!;
    defer end_device_operation(&operation);
    validate_queue(&operation, &device, queue)!;
    if (desc == null) return INVALID_ARGUMENT~;
    validate_resource_owner(
        &operation,
        desc.pipeline.owner,
        ResourceOwnerKind.PIPELINE,
    )!;
    if (!generated_work_kind_valid(desc.kind)
        || desc.max_commands_per_list == 0
        || desc.preprocess_buffer_count == 0
        || desc.max_commands_per_list
            > operation.data.caps.max_generated_work_count) {
        return INVALID_ARGUMENT~;
    }
    return operation.vtable.reserve_generated_scratch(queue, desc);
}

<* Release one quiescent generated-work reservation from the calling thread's
   device context. *>
fn void? release_generated_scratch(
    Queue queue,
    PipelineHandle pipeline,
    GeneratedWorkKind kind,
) {
    Device device = queue.device;
    DeviceOperation operation = begin_device_operation(&device)!;
    defer end_device_operation(&operation);
    validate_queue(&operation, &device, queue)!;
    validate_resource_owner(
        &operation,
        pipeline.owner,
        ResourceOwnerKind.PIPELINE,
    )!;
    if (!generated_work_kind_valid(kind)) return INVALID_ARGUMENT~;
    return operation.vtable.release_generated_scratch(queue, pipeline, kind);
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
    CommandBackendContext backend;
    bool backend_created = false;
    bool lease_retained = false;
    defer catch {
        if (lease_retained) {
            (void)release_command_pin_lease(&operation, &commands);
        }
        if (backend_created) {
            note_command_lifecycle_vtable_dispatch();
            (void)vtable.discard_commands(
                &commands.device,
                commands.handle,
                false,
                null,
            );
        }
    }
    note_command_lifecycle_vtable_dispatch();
    vtable.begin_commands(queue, &commands, &backend)!;
    backend_created = true;
    if (!retain_command_pin_lease(&operation, &commands)) {
        return INVALID_HANDLE~;
    }
    lease_retained = true;
    publish_command_encoder(&operation, &commands, &backend)!;
    retain_device_children(&operation);
    transfer_device_operation_pin(&operation);
    backend_created = false;
    lease_retained = false;
    return commands;
}

<* Finish and consume a recording token. Failure leaves it recording. *>
fn ExecutableCommandList? end_commands(CommandList* commands) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    DeviceOperation operation = command_operation(commands)!;
    defer end_device_operation(&operation);
    note_command_lifecycle_vtable_dispatch();
    operation.vtable.end_commands(encoder)!;
    transfer_command_encoder_to_executable(encoder);
    ExecutableCommandList executable = {
        .device = commands.device,
        .handle = commands.handle,
        .encoder = commands.encoder,
    };
    *commands = {};
    return executable;
}

<* Consume an unfinished recording, including after device loss. *>
fn void? discard_commands(CommandList* commands) {
    CommandEncoderState* encoder = recording_encoder(commands, true)!;
    DeviceOperation operation = command_operation(commands, true)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    CommandList consumed = *commands;
    note_command_lifecycle_vtable_dispatch();
    vtable.discard_commands(
        &consumed.device,
        consumed.handle,
        false,
        encoder,
    )!;
    *commands = {};
    bool released_lease = release_command_pin_lease(&operation, &consumed);
    if (released_lease || !operation.owns_pin) {
        release_device_children(&operation);
        (void)release_retained_device_pins(&operation);
    }
}

<* Consume an executable token without submitting it. *>
fn void? discard_executable_commands(ExecutableCommandList* commands) {
    CommandEncoderState* encoder = executable_encoder(commands, true)!;
    DeviceOperation operation = executable_command_operation(commands, true)!;
    defer end_device_operation(&operation);
    BackendVTable* vtable = operation.vtable;
    ExecutableCommandList consumed = *commands;
    note_command_lifecycle_vtable_dispatch();
    vtable.discard_commands(
        &consumed.device,
        consumed.handle,
        true,
        encoder,
    )!;
    *commands = {};
    bool released_lease = release_executable_command_pin_lease(
        &operation,
        &consumed,
    );
    if (released_lease || !operation.owns_pin) {
        release_device_children(&operation);
        (void)release_retained_device_pins(&operation);
    }
}

<* Record a buffer-to-buffer copy; transfer-side barriers are the caller's. *>
fn void? cmd_copy_buffer(CommandList* commands, BufferCopyDesc* desc) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    if (desc == null) return INVALID_ARGUMENT~;
    return encoder.ops.copy_buffer(encoder, desc);
}

<* Record a 32-bit-pattern fill over dst; its offset and size must be 4-byte
   multiples. *>
fn void? cmd_fill_buffer(CommandList* commands, GpuSpan dst, uint value) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.fill_buffer(encoder, dst, value);
}

<* The texture's effective layout must be TRANSFER_DST; the caller records
   the surrounding barriers. *>
fn void? cmd_copy_buffer_to_texture(CommandList* commands, BufferTextureCopyDesc* desc) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    if (desc == null) return INVALID_ARGUMENT~;
    return encoder.ops.copy_buffer_to_texture(encoder, desc);
}

<* The texture's effective layout must be TRANSFER_SRC. *>
fn void? cmd_copy_texture_to_buffer(CommandList* commands, TextureBufferCopyDesc* desc) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    if (desc == null) return INVALID_ARGUMENT~;
    return encoder.ops.copy_texture_to_buffer(encoder, desc);
}

<* Record a global semantic execution and memory barrier. *>
fn void? cmd_barrier(CommandList* commands, Barrier* barrier) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    if (barrier == null) return INVALID_ARGUMENT~;
    return encoder.ops.barrier(encoder, barrier);
}

<* Record an explicit semantic texture transition. *>
fn void? cmd_texture_barrier(CommandList* commands, TextureBarrier* barrier) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    if (barrier == null) return INVALID_ARGUMENT~;
    return encoder.ops.texture_barrier(encoder, barrier);
}

<* Bind the active compute or graphics pipeline for later execution commands.
   A failed bind leaves the previous active pipeline unchanged. *>
fn void? cmd_bind_pipeline(CommandList* commands, PipelineHandle pipeline) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.bind_pipeline(encoder, pipeline);
}

<* Dispatch the active compute pipeline with a single nonzero root GPU address
   pushed at offset 0. Every allocation reachable from the root must admit the
   recording queue's role. Counts may be zero and may not exceed
   DeviceCaps.max_compute_work_group_count on any axis. *>
fn void? cmd_dispatch(CommandList* commands, GpuAddress root, Vec3u groups) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.dispatch(encoder, root, groups);
}

<* Set separate dynamic depth state for subsequent draws in the active render
   pass. A failed update leaves the previous depth state unchanged. *>
fn void? cmd_set_depth_state(CommandList* commands, DepthState* depth) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    if (depth == null) return INVALID_ARGUMENT~;
    return encoder.ops.set_depth_state(encoder, depth);
}

<* Set pass-bounded topology, culling, front face, and depth bias for
   subsequent draws. Requires an active render pass. A failed update leaves
   the previous state unchanged. *>
fn void? cmd_set_raster_state(CommandList* commands, DynamicRasterState* raster) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    if (raster == null) return INVALID_ARGUMENT~;
    return encoder.ops.set_raster_state(encoder, raster);
}

<* Set one pass-bounded dynamic viewport for subsequent draws. Requires an
   active render pass; malformed values fault INVALID_ARGUMENT. *>
fn void? cmd_set_viewport(CommandList* commands, Viewport* viewport) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    if (viewport == null) return INVALID_ARGUMENT~;
    return encoder.ops.set_viewport(encoder, viewport);
}

<* Set one pass-bounded dynamic scissor for subsequent draws. Requires an
   active render pass; malformed values fault INVALID_ARGUMENT. *>
fn void? cmd_set_scissor(CommandList* commands, ScissorRect* scissor) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    if (scissor == null) return INVALID_ARGUMENT~;
    return encoder.ops.set_scissor(encoder, scissor);
}

<* Draw inside a render pass, pushing the vertex and fragment root addresses.
   Every allocation reachable from either root must admit the recording queue's
   role. *>
fn void? cmd_draw(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, uint vertex_count, uint instance_count) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.draw(
        encoder:        encoder,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        vertex_count:   vertex_count,
        instance_count: instance_count,
    );
}

<* Draw indexed geometry; indices are read from the start of index_span.
   Every allocation reachable from either root must admit the recording queue's
   role. *>
fn void? cmd_draw_indexed(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan index_span, uint index_count, uint instance_count, IndexType index_type = IndexType.U32) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.draw_indexed(
        encoder:        encoder,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        index_span:     index_span,
        index_count:    index_count,
        instance_count: instance_count,
        index_type:     index_type,
    );
}

<* Draw from GPU-written DrawIndirectCommand arguments at the span's offset.
   The args span must support indirect-command reads. The root pair applies to
   every draw in the multi-draw (per-draw variation via gl_DrawID). Ordering
   against argument writes is the caller's barrier. Every allocation reachable
   from either root must admit the recording queue's role. draw_count may not
   exceed DeviceCaps.max_draw_indirect_count. *>
fn void? cmd_draw_indirect(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, uint draw_count) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.draw_indirect(
        encoder:       encoder,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        args:          args,
        draw_count:    draw_count,
    );
}

<* Indexed variant over DrawIndexedIndirectCommand arguments. Index-range
   bounds are read from GPU memory at execution time and are not validated;
   out-of-bounds indices follow device robustness. draw_count may not exceed
   DeviceCaps.max_draw_indirect_count. Every allocation reachable from either
   root must admit the recording queue's role. *>
fn void? cmd_draw_indexed_indirect(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, uint draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.draw_indexed_indirect(
        encoder:       encoder,
        vertex_root:   vertex_root,
        fragment_root: fragment_root,
        args:          args,
        draw_count:    draw_count,
        index_span:    index_span,
        index_type:    index_type,
    );
}

<* Count-buffer variant: the GPU-written uint at count_span limits execution
   to max_draw_count and may not exceed DeviceCaps.max_draw_indirect_count.
   The args span must hold max_draw_count entries. Requires
   DeviceCaps.draw_indirect_count. Every allocation reachable from either root
   must admit the recording queue's role. *>
fn void? cmd_draw_indexed_indirect_count(CommandList* commands, GpuAddress vertex_root, GpuAddress fragment_root, GpuSpan args, GpuSpan count_span, uint max_draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.draw_indexed_indirect_count(
        encoder:        encoder,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        args:           args,
        count_span:     count_span,
        max_draw_count: max_draw_count,
        index_span:     index_span,
        index_type:     index_type,
    );
}

<* Dispatch from GPU-written DispatchIndirectCommand arguments. Each group
   count must not exceed the corresponding
   DeviceCaps.max_compute_work_group_count component. Every allocation
   reachable from the root must admit the recording queue's role. *>
fn void? cmd_dispatch_indirect(CommandList* commands, GpuAddress root, GpuSpan args) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.dispatch_indirect(encoder, root, args);
}

<* Execute up to max_draw_count GPU-written draw records. The GPU-written uint
   at count_span selects the actual count. Ordering against producer writes is
   the caller's barrier with hazards.draw_arguments set. *>
fn void? cmd_draw_generated(CommandList* commands, GpuSpan records, GpuSpan count_span, uint max_draw_count) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.draw_generated(
        encoder:        encoder,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
    );
}

<* Indexed generated-work variant. Index bounds remain GPU execution-time data. *>
fn void? cmd_draw_indexed_generated(CommandList* commands, GpuSpan records, GpuSpan count_span, uint max_draw_count, GpuSpan index_span, IndexType index_type = IndexType.U32) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.draw_indexed_generated(
        encoder:        encoder,
        records:        records,
        count_span:     count_span,
        max_draw_count: max_draw_count,
        index_span:     index_span,
        index_type:     index_type,
    );
}

<* Execute up to max_dispatch_count GPU-written dispatch records. *>
fn void? cmd_dispatch_generated(CommandList* commands, GpuSpan records, GpuSpan count_span, uint max_dispatch_count) {
    CommandEncoderState* encoder = recording_encoder(commands)!;
    return encoder.ops.dispatch_generated(
        encoder:            encoder,
        records:            records,
        count_span:         count_span,
        max_dispatch_count: max_dispatch_count,
    );
}
