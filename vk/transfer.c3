module gpu::vk;

import gpu;
import vk;
import std::core::mem;

const uint TRANSFER_RETIRE_CAP = 1024;
const uint DEDICATED_RETIRE_CAP = 64;
const usz TRANSFER_ALIGN = 16;

// Ranges live as monotonic virtual offsets (physical = virtual % size), so
// wrap gaps fold into the next entry and head-drain stays a single compare.
struct TransferRange {
    ulong virtual_end;
    ulong timeline_value;
    bool  pinned;
}

struct TransferArenaState {
    gpu::BufferHandle backing;
    gpu::GpuSpan      span;
    usz               size;
    ulong             virtual_head;
    ulong             virtual_tail;
    TransferRange[TRANSFER_RETIRE_CAP] retire;
    uint              retire_head;
    uint              retire_count;
}

struct DedicatedStaging {
    gpu::BufferHandle buffer;
    ulong             timeline_value;
}

struct DedicatedRetireState {
    DedicatedStaging[DEDICATED_RETIRE_CAP] entries;
    uint head;
    uint count;
}

<* The buffer fns used here only dereference backend_state, so a shim Device
   works without the public Device value (which lives by-value with the
   caller and has no stable address). *>
fn gpu::Device device_shim(VkDeviceState* state) @private {
    return { .backend_state = (void*)state };
}

fn void? create_transfer_arenas(VkDeviceState* state, gpu::DeviceDesc* desc) {
    gpu::Device shim = device_shim(state);
    usz staging_size = desc.staging_arena_size == 0 ? gpu::DEFAULT_STAGING_ARENA_SIZE : desc.staging_arena_size;
    usz readback_size = desc.readback_arena_size == 0 ? gpu::DEFAULT_READBACK_ARENA_SIZE : desc.readback_arena_size;

    gpu::BufferDesc staging_desc = {
        .size        = staging_size,
        .usage       = { .transfer_src },
        .memory_kind = gpu::MemoryKind.STAGING,
        .debug_name  = "staging_arena",
    };
    state.staging_arena.backing = vk_create_buffer(&shim, &staging_desc)!;
    state.staging_arena.span = vk_get_buffer_span(&shim, state.staging_arena.backing)!;
    state.staging_arena.size = staging_size;

    gpu::BufferDesc readback_desc = {
        .size        = readback_size,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::MemoryKind.READBACK,
        .debug_name  = "readback_arena",
    };
    state.readback_arena.backing = vk_create_buffer(&shim, &readback_desc)!;
    state.readback_arena.span = vk_get_buffer_span(&shim, state.readback_arena.backing)!;
    state.readback_arena.size = readback_size;
}

fn void drain_transfer_retire(VkDeviceState* state, TransferArenaState* arena) @private {
    if (arena.retire_count == 0) return;
    ulong current = 0;
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &current);
    while (arena.retire_count > 0) {
        TransferRange* entry = &arena.retire[arena.retire_head];
        if (entry.pinned || entry.timeline_value > current) break;
        arena.virtual_head = entry.virtual_end;
        arena.retire_head = (arena.retire_head + 1) % TRANSFER_RETIRE_CAP;
        arena.retire_count--;
    }
}

<* Contiguous ring allocation over virtual offsets; the allocated range
   retires against the current frame's timeline value. Faults ARENA_FULL when
   the request cannot fit the live window (callers fall back to a dedicated
   buffer). *>
fn gpu::GpuSpan? transfer_arena_alloc(VkDeviceState* state, TransferArenaState* arena, usz size, usz align) @private {
    if (size > arena.size) return gpu::ARENA_FULL~;
    drain_transfer_retire(state, arena);
    if (arena.retire_count == TRANSFER_RETIRE_CAP) return gpu::ARENA_FULL~;

    ulong tail = gpu::align_up(arena.virtual_tail, align);
    usz physical = (usz)(tail % arena.size);
    if (physical + size > arena.size) tail += arena.size - physical;
    if (tail + size - arena.virtual_head > arena.size) return gpu::ARENA_FULL~;

    // Always counter+1: ranges hold until the next timeline advance (frame
    // end or a blocking helper), so out-of-frame recordings cannot reuse a
    // range before their command list is submitted.
    arena.retire[(arena.retire_head + arena.retire_count) % TRANSFER_RETIRE_CAP] = {
        .virtual_end    = tail + size,
        .timeline_value = state.timeline_counter + 1,
    };
    arena.retire_count++;
    arena.virtual_tail = tail + size;
    return arena.span.subspan((usz)(tail % arena.size), size);
}

fn void drain_dedicated_transfers(VkDeviceState* state) @private {
    if (state.dedicated_staging.count == 0) return;
    gpu::Device shim = device_shim(state);
    ulong current = 0;
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &current);
    while (state.dedicated_staging.count > 0) {
        DedicatedStaging* entry = &state.dedicated_staging.entries[state.dedicated_staging.head];
        if (entry.timeline_value > current) break;
        (void)vk_destroy_buffer(&shim, entry.buffer);
        state.dedicated_staging.head = (state.dedicated_staging.head + 1) % DEDICATED_RETIRE_CAP;
        state.dedicated_staging.count--;
    }
}

<* Span from the arena, or a dedicated one-shot buffer (destroyed once its
   timeline value retires) when the arena cannot fit the request. *>
fn gpu::GpuSpan? transfer_alloc(VkDeviceState* state, TransferArenaState* arena, usz size, usz align, gpu::BufferDesc* dedicated_desc) @private {
    drain_dedicated_transfers(state);
    if (try span = transfer_arena_alloc(state, arena, size, align)) return span;
    if (state.dedicated_staging.count == DEDICATED_RETIRE_CAP) return gpu::ARENA_FULL~;

    gpu::Device shim = device_shim(state);
    gpu::BufferHandle buffer = vk_create_buffer(&shim, dedicated_desc)!;
    state.dedicated_staging.entries[(state.dedicated_staging.head + state.dedicated_staging.count) % DEDICATED_RETIRE_CAP] = {
        .buffer         = buffer,
        .timeline_value = state.timeline_counter + 1,
    };
    state.dedicated_staging.count++;
    return vk_get_buffer_span(&shim, buffer);
}

fn gpu::GpuSpan? staging_alloc(VkDeviceState* state, usz size, usz align) @private {
    gpu::BufferDesc desc = {
        .size        = size,
        .usage       = { .transfer_src },
        .memory_kind = gpu::MemoryKind.STAGING,
        .debug_name  = "staging_dedicated",
    };
    return transfer_alloc(state, &state.staging_arena, size, align, &desc);
}

fn gpu::GpuSpan? readback_alloc(VkDeviceState* state, usz size, usz align) @private {
    gpu::BufferDesc desc = {
        .size        = size,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::MemoryKind.READBACK,
        .debug_name  = "readback_dedicated",
    };
    return transfer_alloc(state, &state.readback_arena, size, align, &desc);
}

<* Byte size of a texture-upload region with the copy-desc 0-defaults
   resolved. The copy command re-validates independently. *>
fn usz? texture_upload_bytes(VkDeviceState* state, gpu::TextureUploadDesc* desc) @private {
    TextureSlot* slot = state.textures.get(desc.texture)!;
    if (desc.mip >= slot.mip_levels) return gpu::INVALID_ARGUMENT~;
    MipExtent extent = texture_mip_extent(slot, desc.mip);
    uint width  = desc.width == 0 ? extent.width : desc.width;
    uint height = desc.height == 0 ? extent.height : desc.height;
    uint layers = desc.layer_count == 0 ? 1 : desc.layer_count;
    return (usz)width * height * layers * format_texel_size(slot.format);
}

fn void? vk_cmd_upload_buffer(gpu::CommandList* commands, gpu::BufferHandle dst, usz offset, char[] data) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    if (data.len == 0) return gpu::INVALID_ARGUMENT~;
    BufferSlot* dst_slot = state.buffers.get(dst)!;
    if (offset + data.len > dst_slot.size) return gpu::INVALID_ARGUMENT~;

    gpu::GpuSpan staging = staging_alloc(state, data.len, TRANSFER_ALIGN)!;
    mem::copy(staging.cpu, data.ptr, data.len);
    vk_flush_buffer(commands.device, staging.buffer, staging.offset, staging.size)!;

    gpu::BufferCopyDesc copy = {
        .src        = staging.buffer,
        .dst        = dst,
        .src_offset = staging.offset,
        .dst_offset = offset,
        .size       = data.len,
    };
    return vk_cmd_copy_buffer(commands, &copy);
}

fn void? vk_cmd_upload_texture(gpu::CommandList* commands, gpu::TextureUploadDesc* desc, char[] data) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    usz expected = texture_upload_bytes(state, desc)!;
    if (data.len != expected) return gpu::INVALID_ARGUMENT~;

    TextureSlot* slot = state.textures.get(desc.texture)!;
    usz align = format_texel_size(slot.format);
    if (align < 4) align = 4;
    gpu::GpuSpan staging = staging_alloc(state, data.len, align)!;
    mem::copy(staging.cpu, data.ptr, data.len);
    vk_flush_buffer(commands.device, staging.buffer, staging.offset, staging.size)!;

    gpu::BufferTextureCopyDesc copy = {
        .src         = staging,
        .texture     = desc.texture,
        .mip         = desc.mip,
        .base_layer  = desc.base_layer,
        .layer_count = desc.layer_count,
        .x           = desc.x,
        .y           = desc.y,
        .width       = desc.width,
        .height      = desc.height,
    };
    return vk_cmd_copy_buffer_to_texture(commands, &copy);
}

fn void? vk_upload_buffer_data(gpu::Device* device, gpu::BufferHandle dst, usz offset, char[] data, gpu::Stage next_stage, gpu::Hazard next_hazard) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    gpu::CommandList commands;
    vk_begin_commands(device, gpu::QueueKind.GRAPHICS, &commands)!;
    vk_cmd_upload_buffer(&commands, dst, offset, data)!;

    gpu::BufferBarrier next_use = {
        .buffer        = dst,
        .offset        = offset,
        .size          = data.len,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = next_stage,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = next_hazard,
    };
    vk_cmd_buffer_barrier(&commands, &next_use)!;
    return submit_and_wait(state, device, &commands);
}

fn void? vk_upload_texture_data(gpu::Device* device, gpu::TextureHandle dst, uint mip, char[] data, gpu::Stage next_stage, gpu::Hazard next_hazard, gpu::TextureLayout next_layout) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    TextureSlot* slot = state.textures.get(dst)!;
    if (slot.current_layout != gpu::TextureLayout.UNDEFINED
        && slot.current_layout != gpu::TextureLayout.TRANSFER_DST) {
        return gpu::INVALID_RESOURCE_STATE~;
    }
    // Validate before recording: a fault after the entry barrier would leave
    // the tracked layout claiming a transition the GPU never ran.
    gpu::TextureUploadDesc probe = { .texture = dst, .mip = mip };
    usz expected = texture_upload_bytes(state, &probe)!;
    if (data.len != expected) return gpu::INVALID_ARGUMENT~;

    gpu::CommandList commands;
    vk_begin_commands(device, gpu::QueueKind.GRAPHICS, &commands)!;

    if (slot.current_layout == gpu::TextureLayout.UNDEFINED) {
        gpu::TextureBarrier to_dst = {
            .texture       = dst,
            .before_stage  = gpu::Stage.HOST,
            .after_stage   = gpu::Stage.TRANSFER,
            .before_hazard = gpu::Hazard.HOST_WRITE,
            .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
            .old_layout    = gpu::TextureLayout.UNDEFINED,
            .new_layout    = gpu::TextureLayout.TRANSFER_DST,
        };
        vk_cmd_texture_barrier(&commands, &to_dst)!;
    }

    gpu::TextureUploadDesc upload = { .texture = dst, .mip = mip };
    vk_cmd_upload_texture(&commands, &upload, data)!;

    gpu::TextureBarrier next_use = {
        .texture       = dst,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = next_stage,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = next_hazard,
        .old_layout    = gpu::TextureLayout.TRANSFER_DST,
        .new_layout    = next_layout,
    };
    vk_cmd_texture_barrier(&commands, &next_use)!;
    return submit_and_wait(state, device, &commands);
}

fn void? vk_readback_buffer_data(gpu::Device* device, gpu::BufferHandle src, usz offset, char[] out_data, gpu::Stage from_stage, gpu::Hazard from_hazard) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    if (out_data.len == 0) return gpu::INVALID_ARGUMENT~;
    BufferSlot* src_slot = state.buffers.get(src)!;
    if (offset + out_data.len > src_slot.size) return gpu::INVALID_ARGUMENT~;

    gpu::GpuSpan readback = readback_alloc(state, out_data.len, TRANSFER_ALIGN)!;

    gpu::CommandList commands;
    vk_begin_commands(device, gpu::QueueKind.GRAPHICS, &commands)!;

    gpu::BufferBarrier from = {
        .buffer        = src,
        .offset        = offset,
        .size          = out_data.len,
        .before_stage  = from_stage,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = from_hazard,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
    };
    vk_cmd_buffer_barrier(&commands, &from)!;

    gpu::BufferCopyDesc copy = {
        .src        = src,
        .dst        = readback.buffer,
        .src_offset = offset,
        .dst_offset = readback.offset,
        .size       = out_data.len,
    };
    vk_cmd_copy_buffer(&commands, &copy)!;

    gpu::BufferBarrier host_read = {
        .buffer        = readback.buffer,
        .offset        = readback.offset,
        .size          = out_data.len,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::Hazard.HOST_READ,
    };
    vk_cmd_buffer_barrier(&commands, &host_read)!;

    submit_and_wait(state, device, &commands)!;
    vk_invalidate_buffer(device, readback.buffer, readback.offset, readback.size)!;
    mem::copy(out_data.ptr, readback.cpu, out_data.len);
}

fn void? vk_readback_texture_data(gpu::Device* device, gpu::TextureHandle src, uint mip, char[] out_data, gpu::Stage from_stage, gpu::Hazard from_hazard, gpu::TextureLayout from_layout) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    gpu::TextureUploadDesc region = { .texture = src, .mip = mip };
    usz expected = texture_upload_bytes(state, &region)!;
    if (out_data.len != expected) return gpu::INVALID_ARGUMENT~;

    TextureSlot* slot = state.textures.get(src)!;
    usz align = format_texel_size(slot.format);
    if (align < 4) align = 4;
    gpu::GpuSpan readback = readback_alloc(state, out_data.len, align)!;

    gpu::CommandList commands;
    vk_begin_commands(device, gpu::QueueKind.GRAPHICS, &commands)!;

    gpu::TextureBarrier to_src = {
        .texture       = src,
        .before_stage  = from_stage,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = from_hazard,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = from_layout,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    vk_cmd_texture_barrier(&commands, &to_src)!;

    gpu::TextureBufferCopyDesc copy = { .texture = src, .dst = readback, .mip = mip };
    vk_cmd_copy_texture_to_buffer(&commands, &copy)!;

    gpu::TextureBarrier restore = {
        .texture       = src,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = from_stage,
        .before_hazard = gpu::Hazard.TRANSFER_READ,
        .after_hazard  = from_hazard,
        .old_layout    = gpu::TextureLayout.TRANSFER_SRC,
        .new_layout    = from_layout,
    };
    vk_cmd_texture_barrier(&commands, &restore)!;

    gpu::BufferBarrier host_read = {
        .buffer        = readback.buffer,
        .offset        = readback.offset,
        .size          = out_data.len,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::Hazard.HOST_READ,
    };
    vk_cmd_buffer_barrier(&commands, &host_read)!;

    submit_and_wait(state, device, &commands)!;
    vk_invalidate_buffer(device, readback.buffer, readback.offset, readback.size)!;
    mem::copy(out_data.ptr, readback.cpu, out_data.len);
}

<* Blocking helpers act as mini-frames: after the idle-wait the timeline
   advances (host signal is truthful once the queue is drained), retiring
   this call's transfer ranges and any pending dedicated buffers. *>
fn void? submit_and_wait(VkDeviceState* state, gpu::Device* device, gpu::CommandList* commands) @private {
    vk_end_commands(device, commands)!;
    gpu::CommandList[1] lists = { *commands };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    vk_submit(device, &submit)!;
    vk_wait_queue_idle(device, gpu::QueueKind.GRAPHICS)!;

    state.timeline_counter++;
    vk::SemaphoreSignalInfo signal_info = vk::semaphore_signal_info()
        .set_semaphore(state.frame_timeline)
        .set_value(state.timeline_counter);
    return check(vk::signal_semaphore(state.device, &signal_info));
}

struct TicketAlloc @private {
    gpu::GpuSpan span;
    ulong        range_end;
}

<* Arena range for a ticket: allocated like any transfer range but pinned, so
   FIFO drain holds at it until resolve unpins. *>
fn TicketAlloc? ticket_arena_alloc(VkDeviceState* state, TransferArenaState* arena, usz size) @private {
    gpu::GpuSpan span = transfer_arena_alloc(
        state: state,
        arena: arena,
        size:  size,
        align: TRANSFER_ALIGN,
    )!;
    uint idx = (arena.retire_head + arena.retire_count - 1) % TRANSFER_RETIRE_CAP;
    arena.retire[idx].pinned = true;
    return { .span = span, .range_end = arena.retire[idx].virtual_end };
}

fn void unpin_ticket_range(TransferArenaState* arena, ulong range_end) @private {
    for (uint i = 0; i < arena.retire_count; i++) {
        TransferRange* entry = &arena.retire[(arena.retire_head + i) % TRANSFER_RETIRE_CAP];
        if (entry.virtual_end == range_end) {
            entry.pinned = false;
            return;
        }
    }
}

<* Ticket destination: pinned readback-arena range, or a dedicated buffer
   (destroyed at resolve) when the arena cannot fit the request. *>
fn gpu::ReadbackTicket? ticket_alloc(VkDeviceState* state, usz size) @private {
    if (try alloc = ticket_arena_alloc(state, &state.readback_arena, size)) {
        return {
            .span      = alloc.span,
            .value     = (gpu::SemaphoreValue)(state.timeline_counter + 1),
            .range_end = alloc.range_end,
        };
    }
    gpu::BufferDesc desc = {
        .size        = size,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::MemoryKind.READBACK,
        .debug_name  = "readback_ticket",
    };
    gpu::Device shim = device_shim(state);
    gpu::BufferHandle buffer = vk_create_buffer(&shim, &desc)!;
    if (try span = vk_get_buffer_span(&shim, buffer)) {
        return {
            .span      = span,
            .value     = (gpu::SemaphoreValue)(state.timeline_counter + 1),
            .dedicated = buffer,
        };
    }
    (void)vk_destroy_buffer(&shim, buffer);
    return gpu::ARENA_FULL~;
}

fn void release_ticket(VkDeviceState* state, gpu::ReadbackTicket* ticket) @private {
    if (ticket.dedicated.is_valid()) {
        gpu::Device shim = device_shim(state);
        (void)vk_destroy_buffer(&shim, ticket.dedicated);
    } else {
        unpin_ticket_range(&state.readback_arena, ticket.range_end);
    }
}

<* Record the copy plus the transfer→host barrier on the destination range;
   source-side ordering stays with the caller. *>
fn gpu::ReadbackTicket? vk_cmd_readback_buffer(gpu::CommandList* commands, gpu::BufferHandle src, usz offset, usz size) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    if (size == 0) return gpu::INVALID_ARGUMENT~;
    BufferSlot* src_slot = state.buffers.get(src)!;
    if (offset + size > src_slot.size) return gpu::INVALID_ARGUMENT~;

    gpu::ReadbackTicket ticket = ticket_alloc(state, size)!;
    defer catch release_ticket(state, &ticket);

    gpu::BufferCopyDesc copy = {
        .src        = src,
        .dst        = ticket.span.buffer,
        .src_offset = offset,
        .dst_offset = ticket.span.offset,
        .size       = size,
    };
    vk_cmd_copy_buffer(commands, &copy)!;
    record_ticket_host_barrier(commands, &ticket)!;
    return ticket;
}

<* Full-mip texture variant; requires transfer_src usage and the TRANSFER_SRC
   tracked layout (caller's barriers on both sides). *>
fn gpu::ReadbackTicket? vk_cmd_readback_texture(gpu::CommandList* commands, gpu::TextureHandle src, uint mip) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    gpu::TextureUploadDesc region = { .texture = src, .mip = mip };
    usz size = texture_upload_bytes(state, &region)!;

    gpu::ReadbackTicket ticket = ticket_alloc(state, size)!;
    defer catch release_ticket(state, &ticket);

    gpu::TextureBufferCopyDesc copy = {
        .texture = src,
        .dst     = ticket.span,
        .mip     = mip,
    };
    vk_cmd_copy_texture_to_buffer(commands, &copy)!;
    record_ticket_host_barrier(commands, &ticket)!;
    return ticket;
}

fn void? record_ticket_host_barrier(gpu::CommandList* commands, gpu::ReadbackTicket* ticket) @private {
    gpu::BufferBarrier host_read = {
        .buffer        = ticket.span.buffer,
        .offset        = ticket.span.offset,
        .size          = ticket.span.size,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::Hazard.HOST_READ,
    };
    return vk_cmd_buffer_barrier(commands, &host_read);
}

fn bool vk_poll_readback(gpu::Device* device, gpu::ReadbackTicket* ticket) @private {
    if ((ulong)ticket.value == 0) return false;
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    ulong current = 0;
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &current);
    return current >= (ulong)ticket.value;
}

fn void? vk_resolve_readback(gpu::Device* device, gpu::ReadbackTicket* ticket, char[] dest) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    if ((ulong)ticket.value == 0) return gpu::INVALID_ARGUMENT~;
    if (dest.len < ticket.span.size) return gpu::INVALID_ARGUMENT~;
    ulong current = 0;
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &current);
    if (current < (ulong)ticket.value) return gpu::READBACK_NOT_READY~;

    vk_invalidate_buffer(device, ticket.span.buffer, ticket.span.offset, ticket.span.size)!;
    mem::copy(dest.ptr, ticket.span.cpu, (sz)ticket.span.size);
    release_ticket(state, ticket);
    *ticket = {};
}
