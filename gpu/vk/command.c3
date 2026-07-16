module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;
import std::core::mem;

alias CreateContextPoolFn @private = fn vk::CommandPool? (void*, vk::Device, uint);
alias DestroyContextPoolFn @private = fn void (void*, vk::Device, vk::CommandPool);

// Concrete (non-generic) slot table for recording contexts. Monomorphized
// twin of gpu::slots, same c3c 0.8.0 rationale as PipelineTable.
const uint RECORDING_CONTEXT_FREE_NONE = 0;

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
    uint                  next_free;
    bool                  used;
}

struct RecordingContextTable {
    RecordingContextCell[] slots;
    ulong                  owner;
    uint                   free_head;
    uint                   next_new;
}

fn gpu::RecordingContextHandle? RecordingContextTable.alloc(&self, RecordingContextState value) {
    uint idx;
    bool found = false;
    while (self.free_head != RECORDING_CONTEXT_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
        if (can_allocate_handle_generation(self.slots[idx].generation)) {
            found = true;
            break;
        }
        self.slots[idx].next_free = RECORDING_CONTEXT_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].used = true;
    return (gpu::RecordingContextHandle){
        .owner      = self.owner,
        .index      = idx,
        .generation = self.slots[idx].generation,
    };
}

fn RecordingContextState*? RecordingContextTable.get(&self, gpu::RecordingContextHandle h) {
    if (h.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? RecordingContextTable.free(&self, gpu::RecordingContextHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    if (!should_recycle_handle_generation(self.slots[idx].generation)) {
        self.slots[idx].generation = HANDLE_GENERATION_MAX;
        self.slots[idx].next_free = RECORDING_CONTEXT_FREE_NONE;
        return;
    }
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
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
        mem::free(context.graphics_pools.ptr);
        context.graphics_pools = {};
    }
    if (context.compute_pools_distinct && context.compute_pools.ptr != null) {
        foreach (pool : context.compute_pools) {
            if (pool != null) ops.destroy(ops.user_data, state.device, pool);
        }
        mem::free(context.compute_pools.ptr);
    }
    context.compute_pools = {};
    if (context.transfer_pools.ptr != null) {
        foreach (pool : context.transfer_pools) {
            if (pool != null) ops.destroy(ops.user_data, state.device, pool);
        }
        mem::free(context.transfer_pools.ptr);
        context.transfer_pools = {};
    }
    if (context.pool_dirty.ptr != null) {
        mem::free(context.pool_dirty.ptr);
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
        context.graphics_pools = mem::new_array(vk::CommandPool, state.frames_in_flight);
    }
    context.compute_pools_distinct = state.selected_queue_roles.compute
        && (!state.selected_queue_roles.graphics
            || state.compute_family != state.graphics_family);
    if (state.selected_queue_roles.compute) {
        context.compute_pools = context.compute_pools_distinct
            ? mem::new_array(vk::CommandPool, state.frames_in_flight)
            : context.graphics_pools;
    }
    if (state.selected_queue_roles.transfer) {
        context.transfer_pools = mem::new_array(vk::CommandPool, state.frames_in_flight);
    }
    context.pool_dirty = mem::new_array(PoolDirtyMask, state.frames_in_flight);

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

fn gpu::RecordingContextHandle? create_builtin_recording_context_with_ops(
    VkDeviceState* state,
    ContextPoolOps* ops
) @private {
    RecordingContextState context;
    create_context_pools_with_ops(state, &context, ops)!;
    gpu::RecordingContextHandle? handle = state.recording_contexts.alloc(context);
    if (catch err = handle) {
        destroy_context_pools_with_ops(state, &context, ops);
        return err~;
    }
    return handle;
}

fn gpu::RecordingContextHandle? create_builtin_recording_context(VkDeviceState* state) @private {
    ContextPoolOps ops = context_pool_ops_real(state);
    return create_builtin_recording_context_with_ops(state, &ops);
}

fn gpu::RecordingContextHandle? vk_create_recording_context(gpu::Device* device) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    RecordingContextState context;
    create_context_pools(state, &context)!;

    gpu::RecordingContextHandle? handle;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        handle = state.recording_contexts.alloc(context);
    }
    if (catch err = handle) {
        destroy_context_pools(state, &context);
        return err~;
    }
    return handle;
}

<* The built-in default and helper contexts cannot be destroyed; a context's
   command lists must be retired first (device teardown sweeps leftovers). *>
fn void? vk_destroy_recording_context(gpu::Device* device, gpu::RecordingContextHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if (handle == state.default_context) return gpu::INVALID_ARGUMENT~;
    if (handle == state.helper_context) return gpu::INVALID_ARGUMENT~;
    RecordingContextState* context = state.recording_contexts.get(handle)!;
    if (context_has_live_commands(state, handle)) return gpu::INVALID_RESOURCE_STATE~;
    destroy_context_pools(state, context);
    state.recording_contexts.free(handle)!;
}

<* The pool whose family matches the queue the list will submit to. *>
fn vk::CommandPool pool_for(RecordingContextState* context, gpu::QueueKind queue, uint frame_index) {
    switch (queue) {
        case gpu::QueueKind.TRANSFER: return context.transfer_pools[frame_index];
        case gpu::QueueKind.COMPUTE:  return context.compute_pools[frame_index];
        default:                      return context.graphics_pools[frame_index];
    }
}

<* Marks the pool pool_for(queue, frame_index) would resolve to as dirty, so
   begin_frame's reset loop knows to reset it. COMPUTE folds into the
   graphics bit when the compute queue shares the graphics pool (mirrors
   pool_for's own aliasing) — one store into the caller's tier-confined
   cell, no locks: Tier C confines a context to one thread and begin_frame
   only reads this under the phase rule's quiescence. *>
fn void mark_pool_dirty(RecordingContextState* context, gpu::QueueKind queue, uint frame_index) @private {
    switch (queue) {
        case gpu::QueueKind.TRANSFER:
            context.pool_dirty[frame_index].transfer = true;
        case gpu::QueueKind.COMPUTE:
            if (context.compute_pools_distinct) {
                context.pool_dirty[frame_index].compute = true;
            } else {
                context.pool_dirty[frame_index].graphics = true;
            }
        default:
            context.pool_dirty[frame_index].graphics = true;
    }
}

fn void? vk_begin_commands(gpu::Device* device, gpu::QueueKind queue, gpu::RecordingContextHandle ctx, gpu::CommandList* out_commands) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    gpu::RecordingContextHandle resolved = ctx.is_valid() ? ctx : state.default_context;
    RecordingContextState* context = state.recording_contexts.get(resolved)!;
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
        .context        = resolved,
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
                .device = *device,
                .handle = handle,
            };
            return;
        }
    }
    vk::free_command_buffers(state.device, pool, 1, &cmd);
    return gpu::SLOT_TABLE_FULL~;
}

fn void? vk_end_commands(gpu::CommandList* commands) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    check_device_result(state, vk::end_command_buffer(record.command_buffer))!;
    record.state = CommandState.EXECUTABLE;
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
