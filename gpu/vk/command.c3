module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;
import std::core::mem;

alias CreateContextPoolFn @private = fn vk::CommandPool? (void*, vk::Device, uint);
alias DestroyContextPoolFn @private = fn void (void*, vk::Device, vk::CommandPool);

const uint MAX_RECORDING_CONTEXTS = 257;
const uint MAX_THREAD_DEVICE_CONTEXTS = 64;

struct RecordingContextHandle {
    uint index;
    uint generation;
}

struct ThreadRecordingContext {
    gpu::Device            device;
    RecordingContextHandle context;
}

fn bool RecordingContextHandle.is_valid(self) => self.generation != 0;
fn bool RecordingContextHandle.equals(
    self,
    RecordingContextHandle other,
) @operator(==) {
    return self.index == other.index && self.generation == other.generation;
}

tlocal ThreadRecordingContext[MAX_THREAD_DEVICE_CONTEXTS]
    thread_recording_contexts;

// Which of a context's pools hold unreset command buffers for a frame slot.
bitstruct PoolDirtyMask : char {
    bool graphics : 0;
    bool compute  : 1;
    bool transfer : 2;
}

struct ContextPoolOps @private {
    void*                user_data;
    CreateContextPoolFn  create;
    DestroyContextPoolFn destroy;
}

struct RecordingContextState {
    vk::CommandPool[] graphics_pools;
    // Aliases graphics_pools when the compute queue shares the graphics
    // family; owned storage only when distinct (compute_pools_distinct).
    vk::CommandPool[] compute_pools;
    vk::CommandPool[] transfer_pools;
    bool              compute_pools_distinct;
    // One mask per frame slot; begin_frame's reset loop reads and clears it.
    PoolDirtyMask[]   pool_dirty;
}

struct RecordingContextCell {
    RecordingContextState value;
    uint                  generation;
    bool                  used;
}

struct RecordingContextTable {
    RecordingContextCell[] slots;
    uint                   next_new;
}

fn RecordingContextHandle? RecordingContextTable.alloc(&self, RecordingContextState value) {
    if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
    uint index = self.next_new++;
    RecordingContextCell* cell = &self.slots[index];
    cell.value = value;
    cell.generation = next_handle_generation(0)!;
    cell.used = true;
    return {
        .index      = index,
        .generation = cell.generation,
    };
}

fn RecordingContextState*? RecordingContextTable.get(
    &self,
    RecordingContextHandle handle,
) {
    if (handle.index >= self.next_new) return gpu::INVALID_HANDLE~;
    RecordingContextCell* cell = &self.slots[handle.index];
    if (!cell.used || cell.generation != handle.generation) {
        return gpu::INVALID_HANDLE~;
    }
    return &cell.value;
}

fn vk::CommandPool? create_context_pool_real(void* user_data, vk::Device device, uint family) @private {
    VkDeviceState* state = (VkDeviceState*)user_data;
    vk::CommandPoolCreateInfo info = vk::command_pool_create_info()
        .set_queue_family_index(family)
        .set_flags(vk::CommandPoolCreateFlagBits.TRANSIENT_BIT);
    vk::CommandPool pool;
    check_device_result(state, vk::create_command_pool(
        device:        device,
        p_create_info: &info,
        p_allocator:   null,
        p_command_pool: &pool,
    ))!;
    return pool;
}

fn void destroy_context_pool_real(void* user_data, vk::Device device, vk::CommandPool pool) @private {
    vk::destroy_command_pool(device, pool, null);
}

fn ContextPoolOps context_pool_ops_real(VkDeviceState* state) @private {
    return {
        .user_data = state,
        .create    = &create_context_pool_real,
        .destroy   = &destroy_context_pool_real,
    };
}

fn void destroy_context_pools_with_ops(
    VkDeviceState* state,
    RecordingContextState* context,
    ContextPoolOps* ops
) @private {
    if (context.graphics_pools.ptr != null) {
        foreach (pool : context.graphics_pools) {
            if (pool != null) ops.destroy(ops.user_data, state.device, pool);
        }
        alloc::free(state.host_allocator, context.graphics_pools.ptr);
        context.graphics_pools = {};
    }
    if (context.compute_pools_distinct && context.compute_pools.ptr != null) {
        foreach (pool : context.compute_pools) {
            if (pool != null) ops.destroy(ops.user_data, state.device, pool);
        }
        alloc::free(state.host_allocator, context.compute_pools.ptr);
    }
    context.compute_pools = {};
    if (context.transfer_pools.ptr != null) {
        foreach (pool : context.transfer_pools) {
            if (pool != null) ops.destroy(ops.user_data, state.device, pool);
        }
        alloc::free(state.host_allocator, context.transfer_pools.ptr);
        context.transfer_pools = {};
    }
    if (context.pool_dirty.ptr != null) {
        alloc::free(state.host_allocator, context.pool_dirty.ptr);
        context.pool_dirty = {};
    }
    context.compute_pools_distinct = false;
}

<* One command pool per queue family per frame-in-flight, owned by a
   recording context. Construction is all-or-nothing. *>
fn void? create_context_pools_with_ops(
    VkDeviceState* state,
    RecordingContextState* context,
    ContextPoolOps* ops
) @private {
    *context = {};
    defer catch destroy_context_pools_with_ops(state, context, ops);

    if (state.selected_queue_roles.graphics) {
        context.graphics_pools = alloc::new_array(
            state.host_allocator,
            vk::CommandPool,
            (sz)state.frames_in_flight,
        );
    }
    context.compute_pools_distinct = state.selected_queue_roles.compute
        && (!state.selected_queue_roles.graphics
            || state.compute_family != state.graphics_family);
    if (state.selected_queue_roles.compute) {
        context.compute_pools = context.compute_pools_distinct
            ? alloc::new_array(
                state.host_allocator,
                vk::CommandPool,
                (sz)state.frames_in_flight,
            )
            : context.graphics_pools;
    }
    if (state.selected_queue_roles.transfer) {
        context.transfer_pools = alloc::new_array(
            state.host_allocator,
            vk::CommandPool,
            (sz)state.frames_in_flight,
        );
    }
    context.pool_dirty = alloc::new_array(
        state.host_allocator,
        PoolDirtyMask,
        (sz)state.frames_in_flight,
    );

    for (uint i = 0; i < state.frames_in_flight; i++) {
        if (state.selected_queue_roles.graphics) {
            context.graphics_pools[i] = ops.create(ops.user_data, state.device, state.graphics_family)!;
        }
        if (context.compute_pools_distinct) {
            context.compute_pools[i] = ops.create(ops.user_data, state.device, state.compute_family)!;
        }
        if (state.selected_queue_roles.transfer) {
            context.transfer_pools[i] = ops.create(ops.user_data, state.device, state.transfer_family)!;
        }
    }
}

fn void? create_context_pools(VkDeviceState* state, RecordingContextState* context) @private {
    ContextPoolOps ops = context_pool_ops_real(state);
    return create_context_pools_with_ops(state, context, &ops);
}

fn void destroy_context_pools(VkDeviceState* state, RecordingContextState* context) @private {
    ContextPoolOps ops = context_pool_ops_real(state);
    destroy_context_pools_with_ops(state, context, &ops);
}

fn RecordingContextHandle? create_builtin_recording_context_with_ops(
    VkDeviceState* state,
    ContextPoolOps* ops
) @private {
    RecordingContextState context;
    create_context_pools_with_ops(state, &context, ops)!;
    RecordingContextHandle? handle = state.recording_contexts.alloc(context);
    if (catch err = handle) {
        destroy_context_pools_with_ops(state, &context, ops);
        return err~;
    }
    return handle;
}

fn RecordingContextHandle? create_builtin_recording_context(VkDeviceState* state) @private {
    ContextPoolOps ops = context_pool_ops_real(state);
    return create_builtin_recording_context_with_ops(state, &ops);
}

fn RecordingContextHandle? thread_recording_context(
    VkDeviceState* state,
    gpu::Device device,
) @private {
    if (device.index >= MAX_THREAD_DEVICE_CONTEXTS) {
        return gpu::INVALID_HANDLE~;
    }
    ThreadRecordingContext* cached = &thread_recording_contexts[device.index];
    if (cached.device == device && cached.context.is_valid()) {
        return cached.context;
    }

    RecordingContextState context;
    create_context_pools(state, &context)!;
    RecordingContextHandle? handle;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        handle = state.recording_contexts.alloc(context);
    }
    if (catch err = handle) {
        destroy_context_pools(state, &context);
        return err~;
    }
    *cached = {
        .device  = device,
        .context = handle,
    };
    return handle;
}

fn vk::CommandPool pool_for(
    RecordingContextState* context,
    gpu::Queue queue,
    uint frame_index,
) {
    if (queue.roles.graphics) return context.graphics_pools[frame_index];
    if (queue.roles.compute) return context.compute_pools[frame_index];
    return context.transfer_pools[frame_index];
}

fn void mark_pool_dirty(
    RecordingContextState* context,
    gpu::Queue queue,
    uint frame_index,
) @private {
    if (queue.roles.graphics) {
        context.pool_dirty[frame_index].graphics = true;
        return;
    }
    if (queue.roles.compute) {
        if (context.compute_pools_distinct) {
            context.pool_dirty[frame_index].compute = true;
        } else {
            context.pool_dirty[frame_index].graphics = true;
        }
        return;
    }
    context.pool_dirty[frame_index].transfer = true;
}

fn void? vk_begin_commands_with_context(
    VkDeviceState* state,
    gpu::Queue queue,
    RecordingContextHandle context_handle,
    gpu::CommandList* out_commands,
) @private {
    gpu::Device device = queue.device;
    RecordingContextState* context = state.recording_contexts.get(context_handle)!;
    uint frame_index = state.frame_index;
    vk::CommandPool pool = pool_for(context, queue, frame_index);
    mark_pool_dirty(context, queue, frame_index);

    vk::CommandBufferAllocateInfo alloc = vk::command_buffer_allocate_info()
        .set_command_pool(pool)
        .set_level(vk::CommandBufferLevel.PRIMARY)
        .set_command_buffer_count(1);
    vk::CommandBuffer cmd;
    check_device_result(state, vk::allocate_command_buffers(state.device, &alloc, &cmd))!;

    vk::CommandBufferBeginInfo begin = vk::command_buffer_begin_info()
        .set_flags(vk::CommandBufferUsageFlagBits.ONE_TIME_SUBMIT_BIT);
    check_device_result(state, vk::begin_command_buffer(cmd, &begin))!;

    CommandRecord record = {
        .command_buffer = cmd,
        .context        = context_handle,
        .queue          = queue,
        .frame_index    = frame_index,
        .state          = CommandState.RECORDING,
        .last_pipeline  = gpu::PIPELINE_HANDLE_INVALID,
        .last_entry     = PIPELINE_ENTRY_NONE,
    };
    {
        state.command_mutex.lock()!!;
        defer state.command_mutex.unlock();
        if (try handle = state.commands.alloc(record)) {
            *out_commands = {
                .device = device,
                .handle = handle,
            };
            return;
        }
    }
    vk::free_command_buffers(state.device, pool, 1, &cmd);
    return gpu::SLOT_TABLE_FULL~;
}

fn void? vk_begin_commands(
    gpu::Queue queue,
    gpu::CommandList* out_commands,
) @private {
    gpu::Device device = queue.device;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&device);
    enter_tracked(state);
    defer exit_tracked(state);
    RecordingContextHandle context = thread_recording_context(state, device)!;
    return vk_begin_commands_with_context(
        state:          state,
        queue:          queue,
        context_handle: context,
        out_commands:   out_commands,
    );
}

fn void? vk_end_commands(gpu::CommandList* commands) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    check_device_result(state, vk::end_command_buffer(record.command_buffer))!;
    record.state = CommandState.EXECUTABLE;
}

fn ulong? vk_discard_commands(
    gpu::Device* device,
    gpu::CommandListHandle handle,
    bool executable,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    state.transfer_mutex.lock()!!;
    defer state.transfer_mutex.unlock();
    {
        state.command_mutex.lock()!!;
        defer state.command_mutex.unlock();
        CommandRecord* record = state.commands.get(handle)!;
        bool state_matches = executable
            ? record.state == CommandState.EXECUTABLE
            : record.state == CommandState.RECORDING
                || record.state == CommandState.RECORDING_RENDER_PASS;
        if (!state_matches) return gpu::COMMAND_RECORDING_ERROR~;
    }

    ulong canceled_tickets = cancel_command_readback_tickets_locked(
        state,
        handle,
    );
    {
        state.command_mutex.lock()!!;
        defer state.command_mutex.unlock();
        release_command(state, handle);
    }
    return canceled_tickets;
}

fn void? vk_cmd_copy_buffer(gpu::CommandList* commands, gpu::BufferCopyDesc* desc) @private {
    CommandRecord* record = resolve_command(commands)!;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    if (record.state != CommandState.RECORDING) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_copy_buffer",
            public_fault:        gpu::COMMAND_RECORDING_ERROR,
            resource_kind:       gpu::DebugResourceKind.COMMAND_LIST,
            has_public_identity: true,
            index:               commands.handle.index,
            generation:          commands.handle.generation,
            debug_name:          null,
            rejected_field:      "CommandList.state",
            invariant:           "command must be recording outside a render pass",
            backend_text:        null,
        )~;
    }
    BufferSlot* src = state.buffers.get(desc.src)!;
    BufferSlot* dst = state.buffers.get(desc.dst)!;
    validate_buffer_recording_access(
        state:     state,
        record:    record,
        slot:      src,
        handle:    desc.src,
        operation: "cmd_copy_buffer",
    )!;
    validate_buffer_recording_access(
        state:     state,
        record:    record,
        slot:      dst,
        handle:    desc.dst,
        operation: "cmd_copy_buffer",
    )!;
    if (!src.usage.transfer_src) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_copy_buffer",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               desc.src.index,
            generation:          desc.src.generation,
            debug_name:          slot_name(&src.debug_name[0]),
            rejected_field:      "BufferUsage.transfer_src",
            invariant:           "source buffer must include transfer_src usage",
            backend_text:        null,
        )~;
    }
    if (!dst.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;
    if (desc.size == 0) return gpu::INVALID_ARGUMENT~;
    if (!gpu::range_fits(src.size, desc.src_offset, desc.size)) return gpu::INVALID_ARGUMENT~;
    if (!gpu::range_fits(dst.size, desc.dst_offset, desc.size)) return gpu::INVALID_ARGUMENT~;
    // Vulkan forbids overlapping regions within one buffer.
    if (desc.src == desc.dst
        && desc.src_offset < desc.dst_offset + desc.size
        && desc.dst_offset < desc.src_offset + desc.size) {
        return gpu::INVALID_ARGUMENT~;
    }

    vk::BufferCopy region = {
        .src_offset = (vk::DeviceSize)desc.src_offset,
        .dst_offset = (vk::DeviceSize)desc.dst_offset,
        .size       = (vk::DeviceSize)desc.size,
    };
    vk::cmd_copy_buffer(record.command_buffer, src.buffer, dst.buffer, 1, &region);
}

fn void? vk_cmd_fill_buffer(gpu::CommandList* commands, gpu::BufferHandle buffer, usz offset, usz size, uint value) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    BufferSlot* slot = state.buffers.get(buffer)!;
    validate_buffer_recording_access(
        state:     state,
        record:    record,
        slot:      slot,
        handle:    buffer,
        operation: "cmd_fill_buffer",
    )!;
    if (!slot.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;
    // Vulkan: offset and size must be 4-byte multiples; size 0 is the
    // whole-buffer convention (translated to WHOLE_SIZE below).
    if (offset % 4 != 0 || size % 4 != 0) {
        ZString rejected_field = offset % 4 != 0 ? "offset" : "size";
        return report_contract_failure(
            state:               state,
            operation:           "cmd_fill_buffer",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               buffer.index,
            generation:          buffer.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      rejected_field,
            invariant:           "offset and size must be 4-byte aligned",
            backend_text:        null,
        )~;
    }
    if (!gpu::range_fits(slot.size, offset, size)) return gpu::INVALID_ARGUMENT~;
    if (size == 0 && offset >= slot.size) return gpu::INVALID_ARGUMENT~;
    vk::DeviceSize fill_size = size == 0 ? (vk::DeviceSize)vk::WHOLE_SIZE : (vk::DeviceSize)size;
    vk::cmd_fill_buffer(record.command_buffer, slot.buffer, (vk::DeviceSize)offset, fill_size, value);
}

<* Validate a copy region against the texture, the mip's clamped extent,
   Vulkan's buffer-offset alignment, and the span's bounds against its backing
   buffer, then translate it. Zero defaults resolve here: extent 0 = full mip,
   layer_count 0 = 1. *>
fn vk::BufferImageCopy? resolve_texture_copy_region(VkDeviceState* state, gpu::GpuSpan* span, TextureSlot* slot, uint row_length_texels, uint mip, uint base_layer, uint layer_count, uint x, uint y, uint width, uint height) @private {
    if (mip >= slot.mip_levels) return gpu::INVALID_ARGUMENT~;
    uint layers = layer_count == 0 ? 1 : layer_count;
    if ((ulong)base_layer + (ulong)layers > (ulong)slot.array_layers) return gpu::INVALID_ARGUMENT~;

    MipExtent extent = texture_mip_extent(slot, mip);
    if (width == 0)  width = extent.width;
    if (height == 0) height = extent.height;
    if ((ulong)x + (ulong)width > (ulong)extent.width || (ulong)y + (ulong)height > (ulong)extent.height) return gpu::INVALID_ARGUMENT~;
    if (row_length_texels != 0 && row_length_texels < width) return gpu::INVALID_ARGUMENT~;

    usz texel_size = format_texel_size(slot.format);
    if (texel_size == 0) return gpu::INVALID_ARGUMENT~;
    if (span.offset % texel_size != 0 || span.offset % 4 != 0) return gpu::INVALID_ARGUMENT~;

    usz row_texels = row_length_texels == 0 ? width : row_length_texels;
    usz copy_bytes = row_texels * texel_size * height * layers;
    resolve_span_slot(state, span, copy_bytes)!;

    return (vk::BufferImageCopy){
        .buffer_offset      = (vk::DeviceSize)span.offset,
        .buffer_row_length  = row_length_texels,
        .buffer_image_height = 0,
        .image_subresource  = {
            .aspect_mask      = format_aspect(slot.format),
            .mip_level        = mip,
            .base_array_layer = base_layer,
            .layer_count      = layers,
        },
        .image_offset = { .x = (int)x, .y = (int)y, .z = 0 },
        .image_extent = { .width = width, .height = height, .depth = 1 },
    };
}

fn void? vk_cmd_copy_buffer_to_texture(gpu::CommandList* commands, gpu::BufferTextureCopyDesc* desc) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    TextureSlot* texture = state.textures.get(desc.texture)!;
    BufferSlot* buffer = state.buffers.get(desc.src.buffer)!;
    validate_span_recording_access(
        state:     state,
        record:    record,
        slot:      buffer,
        span:      &desc.src,
        operation: "cmd_copy_buffer_to_texture",
    )!;
    validate_texture_recording_access(
        state:     state,
        record:    record,
        slot:      texture,
        handle:    desc.texture,
        operation: "cmd_copy_buffer_to_texture",
    )!;
    if (!buffer.usage.transfer_src) return gpu::INVALID_ARGUMENT~;
    if (!texture.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;
    gpu::TextureLayout layout = effective_layout(record, texture, desc.texture);
    if (layout != gpu::TextureLayout.TRANSFER_DST) return gpu::INVALID_RESOURCE_STATE~;

    vk::BufferImageCopy region = resolve_texture_copy_region(
        state:             state,
        span:              &desc.src,
        slot:              texture,
        row_length_texels: desc.row_length_texels,
        mip:               desc.mip,
        base_layer:        desc.base_layer,
        layer_count:       desc.layer_count,
        x:                 desc.x,
        y:                 desc.y,
        width:             desc.width,
        height:            desc.height,
    )!;
    vk::cmd_copy_buffer_to_image(record.command_buffer, buffer.buffer, texture.image, layout_to_vk(layout), 1, &region);
}

fn void? vk_cmd_copy_texture_to_buffer(gpu::CommandList* commands, gpu::TextureBufferCopyDesc* desc) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    TextureSlot* texture = state.textures.get(desc.texture)!;
    BufferSlot* buffer = state.buffers.get(desc.dst.buffer)!;
    validate_texture_recording_access(
        state:     state,
        record:    record,
        slot:      texture,
        handle:    desc.texture,
        operation: "cmd_copy_texture_to_buffer",
    )!;
    validate_span_recording_access(
        state:     state,
        record:    record,
        slot:      buffer,
        span:      &desc.dst,
        operation: "cmd_copy_texture_to_buffer",
    )!;
    if (!buffer.usage.transfer_dst) return gpu::INVALID_ARGUMENT~;
    if (!texture.usage.transfer_src) return gpu::INVALID_ARGUMENT~;
    gpu::TextureLayout layout = effective_layout(record, texture, desc.texture);
    if (layout != gpu::TextureLayout.TRANSFER_SRC) return gpu::INVALID_RESOURCE_STATE~;

    vk::BufferImageCopy region = resolve_texture_copy_region(
        state:             state,
        span:              &desc.dst,
        slot:              texture,
        row_length_texels: desc.row_length_texels,
        mip:               desc.mip,
        base_layer:        desc.base_layer,
        layer_count:       desc.layer_count,
        x:                 desc.x,
        y:                 desc.y,
        width:             desc.width,
        height:            desc.height,
    )!;
    vk::cmd_copy_image_to_buffer(record.command_buffer, texture.image, layout_to_vk(layout), buffer.buffer, 1, &region);
}

<* Resolve an indirect argument span: indirect usage, 4-byte offset alignment,
   checked argument bytes within the span, and the span within its backing buffer. *>
fn BufferSlot*? resolve_indirect_span(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::GpuSpan* span,
    usz arg_size,
    uint count,
    ZString operation,
    ZString rejected_field,
    uint max_count = uint::max,
) {
    if (count > max_count && operation != null) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      rejected_field,
            invariant:           "count must not exceed DeviceCaps.max_draw_indirect_count",
            backend_text:        null,
        )~;
    }
    usz? required_result = gpu::indirect_argument_bytes(
        argument_size: arg_size,
        count:         count,
        max_count:     max_count,
    );
    if (catch size_fault = required_result) {
        if (operation == null) return size_fault~;
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        size_fault,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      rejected_field,
            invariant:           "indirect argument byte size must not overflow",
            backend_text:        null,
        )~;
    }
    usz required = required_result;
    BufferSlot* slot = resolve_span_slot(
        state:          state,
        span:           span,
        required:       required,
        operation:      operation,
        rejected_field: rejected_field,
    )!;
    validate_span_recording_access(
        state:     state,
        record:    record,
        slot:      slot,
        span:      span,
        operation: operation,
    )!;
    if (!slot.usage.indirect) return gpu::INVALID_ARGUMENT~;
    if (span.offset % 4 != 0) return gpu::INVALID_ARGUMENT~;
    return slot;
}

fn void? bind_compute_and_push(VkDeviceState* state, CommandRecord* record, gpu::PipelineHandle handle, gpu::GpuAddress root) @private {
    validate_descriptor_heap_binding(state)!;
    vk::CommandBuffer cb = record.command_buffer;
    PipelineSlot* slot = state.pipelines.get(handle)!;
    if (handle != record.last_pipeline) {
        vk::cmd_bind_pipeline(cb, vk::PipelineBindPoint.COMPUTE, slot.pipeline);
        bind_descriptor_heap(
            state:      state,
            cb:         cb,
            layout:     slot.layout,
            bind_point: vk::PipelineBindPoint.COMPUTE,
        )!;
        record.last_pipeline = handle;
        record.last_entry = slot.entry;
    }
    gpu::RootPush push = { .root_gpu = root };
    vk::cmd_push_constants(cb, slot.layout, vk::ShaderStageFlagBits.COMPUTE_BIT, 0, gpu::RootPush::size, &push);
}

<* Pipeline and heap binds are skipped when the handle is unchanged since the
   last bind; the heap otherwise rebinds because compute layouts differ in
   push-constant range, which breaks descriptor set compatibility. *>
fn void? vk_cmd_dispatch(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress root, gpu::Vec3u groups) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    if (catch limit_fault = gpu::validate_dispatch_groups(groups, state.caps.max_compute_work_group_count)) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_dispatch",
            public_fault:        limit_fault,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "groups",
            invariant:           "group counts must not exceed DeviceCaps.max_compute_work_group_count",
            backend_text:        null,
        )~;
    }
    validate_pipeline_kind(
        state:     state,
        handle:    handle,
        kind:      PipelineKind.COMPUTE,
        operation: "cmd_dispatch",
    )!;
    vk::CommandBuffer cb = record.command_buffer;
    bind_compute_and_push(
        state:  state,
        record: record,
        handle: handle,
        root:   root,
    )!;
    vk::cmd_dispatch(cb, groups.x, groups.y, groups.z);
}

fn void? vk_cmd_dispatch_indirect(gpu::CommandList* commands, gpu::PipelineHandle handle, gpu::GpuAddress root, gpu::GpuSpan args) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    validate_pipeline_kind(
        state:     state,
        handle:    handle,
        kind:      PipelineKind.COMPUTE,
        operation: "cmd_dispatch_indirect",
    )!;
    BufferSlot* args_buffer = resolve_indirect_span(
        state:          state,
        record:         record,
        span:           &args,
        arg_size:       gpu::DispatchIndirectCommand::size,
        count:          1,
        operation:      "cmd_dispatch_indirect",
        rejected_field: "args",
    )!;
    vk::CommandBuffer cb = record.command_buffer;
    bind_compute_and_push(
        state:  state,
        record: record,
        handle: handle,
        root:   root,
    )!;
    vk::cmd_dispatch_indirect(cb, args_buffer.buffer, (vk::DeviceSize)args.offset);
}
