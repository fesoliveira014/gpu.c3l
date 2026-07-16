module gpu::vk @private;

import gpu @public;
import vk;
import std::io;
import std::thread;
import std::core::mem;
import std::atomic;
import std::atomic::types;

const uint TRANSFER_RETIRE_CAP = 1024;
const uint READBACK_TICKET_CAPACITY @private = 4096;
const uint READBACK_TICKET_FREE_NONE @private = 0;
const uint DEDICATED_RETIRE_CAP = 64;
const usz TRANSFER_ALIGN = 16;
// Per-round wait; rounds repeat with a warning until the predecessor signals
// (signaling past a slow sibling would retire its in-flight ranges).
const ulong HELPER_TURNSTILE_ROUND_NS = 5_000_000_000;

// FRAME entries retire on frame_timeline, HELPER on helper_timeline;
// disjoint, so a helper's completion never retires another call's resources.
enum TimelineKind : int {
    FRAME,
    HELPER,
}

struct TimelineTag {
    TimelineKind kind;
    ulong        value;
}

// Ranges live as monotonic virtual offsets (physical = virtual % size), so
// wrap gaps fold into the next entry and head-drain stays a single compare.
struct TransferRange {
    ulong        virtual_end;
    ulong        timeline_value;
    TimelineKind timeline_kind;
    bool         pinned;
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

struct TransferArenaRange @private {
    ulong virtual_start;
    ulong virtual_end;
    usz   physical_offset;
}

struct DedicatedStaging {
    gpu::BufferHandle buffer;
    ulong             timeline_value;
    TimelineKind      timeline_kind;
}

struct DedicatedRetireState {
    DedicatedStaging[DEDICATED_RETIRE_CAP] entries;
    uint head;
    uint count;
}
struct ReadbackTicketState @private {
    gpu::GpuSpan        span;
    gpu::SemaphoreValue value;
    gpu::BufferHandle   dedicated;
    ulong               range_end;
}

struct ReadbackTicketCell @private {
    ReadbackTicketState value;
    uint                generation;
    uint                next_free;
    bool                used;
}

struct ReadbackTicketTable @private {
    ReadbackTicketCell[] slots;
    ulong                owner;
    uint                 free_head;
    uint                 next_new;
}

fn gpu::ReadbackTicket? ReadbackTicketTable.alloc(&self, ReadbackTicketState value) {
    uint index;
    bool found = false;
    while (self.free_head != READBACK_TICKET_FREE_NONE) {
        index = self.free_head - 1;
        self.free_head = self.slots[index].next_free;
        if (can_allocate_handle_generation(self.slots[index].generation)) {
            found = true;
            break;
        }
        self.slots[index].next_free = READBACK_TICKET_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        index = self.next_new++;
        self.slots[index].generation = 0;
    }

    self.slots[index].value = value;
    self.slots[index].generation = next_handle_generation(self.slots[index].generation)!;
    self.slots[index].used = true;
    return (gpu::ReadbackTicket){
        .owner      = self.owner,
        .index      = index,
        .generation = self.slots[index].generation,
    };
}

fn ReadbackTicketState*? ReadbackTicketTable.get(&self, gpu::ReadbackTicket ticket) {
    if (ticket.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint index = ticket.index;
    if (index >= self.slots.len || !self.slots[index].used) return gpu::INVALID_HANDLE~;
    if (self.slots[index].generation != ticket.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[index].value;
}

fn void? ReadbackTicketTable.free(&self, gpu::ReadbackTicket ticket) {
    self.get(ticket)!;
    uint index = ticket.index;
    self.slots[index].used = false;
    if (!should_recycle_handle_generation(self.slots[index].generation)) {
        self.slots[index].generation = HANDLE_GENERATION_MAX;
        self.slots[index].next_free = READBACK_TICKET_FREE_NONE;
        return;
    }
    self.slots[index].generation = next_handle_generation(self.slots[index].generation)!;
    self.slots[index].next_free = self.free_head;
    self.free_head = index + 1;
}

fn ReadbackTicketState*? resolve_readback_ticket(VkDeviceState* state, gpu::ReadbackTicket ticket) @private {
    return state.readback_tickets.get(ticket);
}

fn void destroy_readback_tickets(VkDeviceState* state) @private {
    if (state.readback_tickets.slots.ptr == null) return;
    foreach (&cell : state.readback_tickets.slots[:state.readback_tickets.next_new]) {
        if (cell.used) release_readback_state(state, &cell.value);
    }
    mem::free(state.readback_tickets.slots.ptr);
    state.readback_tickets = {};
}

// Immediate destruction follows timeline completion, device idle, or a
// pre-record failure; no fallible work follows a recorded readback copy.
fn void destroy_readback_buffer(VkDeviceState* state, gpu::BufferHandle handle) @private {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if (try slot = state.buffers.get(handle)) {
        state.allocator.destroy_buffer(slot.buffer, slot.allocation);
        state.buffers.free(handle)!!;
    }
}

fn void release_readback_state(VkDeviceState* state, ReadbackTicketState* ticket) @private {
    if (ticket.dedicated.is_valid()) {
        destroy_readback_buffer(state, ticket.dedicated);
    } else {
        unpin_ticket_range(&state.readback_arena, ticket.range_end);
    }
}


fn gpu::Device device_shim(VkDeviceState* state) @private {
    return state.owner;
}

fn gpu::BufferDesc staging_arena_desc(usz size, gpu::QueueRoles access) @private {
    return {
        .size        = size,
        .usage       = { .transfer_src },
        .memory_kind = gpu::MemoryKind.STAGING,
        .access      = access,
        .debug_name  = "staging_arena",
    };
}

fn gpu::BufferDesc readback_arena_desc(usz size, gpu::QueueRoles access) @private {
    return {
        .size        = size,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::MemoryKind.READBACK,
        .access      = access,
        .debug_name  = "readback_arena",
    };
}

fn gpu::BufferDesc staging_dedicated_desc(usz size, gpu::QueueRoles access) @private {
    return {
        .size        = size,
        .usage       = { .transfer_src },
        .memory_kind = gpu::MemoryKind.STAGING,
        .access      = access,
        .debug_name  = "staging_dedicated",
    };
}

fn void? create_transfer_arenas(VkDeviceState* state, gpu::DeviceDesc* desc) {
    usz staging_size = desc.staging_arena_size == 0 ? gpu::DEFAULT_STAGING_ARENA_SIZE : desc.staging_arena_size;
    usz readback_size = desc.readback_arena_size == 0 ? gpu::DEFAULT_READBACK_ARENA_SIZE : desc.readback_arena_size;

    gpu::BufferDesc staging_desc = staging_arena_desc(staging_size, state.selected_queue_roles);
    uint[MAX_QUEUE_FAMILIES] family_storage;
    uint family_count = resource_queue_families(state, staging_desc.access, family_storage[..]);
    state.staging_arena.backing = create_buffer_with_families_for_state(
        state:          state,
        desc:           &staging_desc,
        queue_families: family_storage[:family_count],
    )!;
    mark_buffer_internal(state, state.staging_arena.backing);
    state.staging_arena.span = get_buffer_span_for_state(state, state.staging_arena.backing)!;
    state.staging_arena.size = staging_size;

    gpu::BufferDesc readback_desc = readback_arena_desc(readback_size, state.selected_queue_roles);
    state.readback_arena.backing = create_buffer_for_state(state, &readback_desc)!;
    mark_buffer_internal(state, state.readback_arena.backing);
    state.readback_arena.span = get_buffer_span_for_state(state, state.readback_arena.backing)!;
    state.readback_arena.size = readback_size;
}

<* A ring can mix tags (mid-frame helpers interleave with recordings) but
   stays strict FIFO: the head retires before anything behind it. *>
fn void? drain_transfer_retire(VkDeviceState* state, TransferArenaState* arena) @private {
    if (arena.retire_count == 0) return;
    ulong frame_current = 0;
    ulong helper_current = 0;
    check_device_result(state, vk::get_semaphore_counter_value(state.device, state.frame_timeline, &frame_current))!;
    check_device_result(state, vk::get_semaphore_counter_value(state.device, state.helper_timeline, &helper_current))!;
    while (arena.retire_count > 0) {
        TransferRange* entry = &arena.retire[arena.retire_head];
        ulong current = entry.timeline_kind == TimelineKind.HELPER ? helper_current : frame_current;
        if (entry.pinned || entry.timeline_value > current) break;
        arena.virtual_head = entry.virtual_end;
        arena.retire_head = (arena.retire_head + 1) % TRANSFER_RETIRE_CAP;
        arena.retire_count--;
    }
}

<* Pure range planning. Empty retire queues normalize cursors; malformed input
   faults INVALID_ARGUMENT and exhausted or unrepresentable ranges fault ARENA_FULL. *>
fn TransferArenaRange? transfer_arena_range(
    ulong virtual_head,
    ulong virtual_tail,
    uint retire_count,
    usz capacity,
    usz size,
    usz align
) @private {
    if (size == 0) return gpu::INVALID_ARGUMENT~;
    if (capacity == 0 || size > capacity) return gpu::ARENA_FULL~;
    if (!is_pow2(align)) return gpu::INVALID_ARGUMENT~;
    if (retire_count == 0) {
        virtual_head = 0;
        virtual_tail = 0;
    } else if (virtual_head > virtual_tail) {
        return gpu::ARENA_FULL~;
    }

    ulong capacity_u = (ulong)capacity;
    ulong size_u = (ulong)size;
    ulong align_u = (ulong)align;
    ulong remainder = virtual_tail & (align_u - 1);
    ulong padding = remainder == 0 ? 0 : align_u - remainder;
    if (padding > ulong::max - virtual_tail) return gpu::ARENA_FULL~;
    ulong virtual_start = virtual_tail + padding;

    usz physical_offset = (usz)(virtual_start % capacity_u);
    if (size > capacity - physical_offset) {
        ulong wrap_gap = capacity_u - (ulong)physical_offset;
        if (wrap_gap > ulong::max - virtual_start) return gpu::ARENA_FULL~;
        virtual_start += wrap_gap;
        physical_offset = 0;
    }

    if (virtual_start < virtual_head) return gpu::ARENA_FULL~;
    ulong live = virtual_start - virtual_head;
    if (live > capacity_u) return gpu::ARENA_FULL~;
    if (size_u > capacity_u - live) return gpu::ARENA_FULL~;
    if (size_u > ulong::max - virtual_start) return gpu::ARENA_FULL~;
    return {
        .virtual_start   = virtual_start,
        .virtual_end     = virtual_start + size_u,
        .physical_offset = physical_offset,
    };
}

<* Contiguous ring allocation over virtual offsets; the range holds until
   its tagged timeline passes it — never a different call's advance. Faults
   ARENA_FULL when the request cannot fit (callers fall back to dedicated). *>
fn gpu::GpuSpan? transfer_arena_alloc(VkDeviceState* state, TransferArenaState* arena, usz size, usz align, TimelineTag tag) @private {
    if (size > arena.size) return gpu::ARENA_FULL~;
    drain_transfer_retire(state, arena)!;
    if (arena.retire_count == TRANSFER_RETIRE_CAP) return gpu::ARENA_FULL~;

    TransferArenaRange range = transfer_arena_range(
        virtual_head: arena.virtual_head,
        virtual_tail: arena.virtual_tail,
        retire_count: arena.retire_count,
        capacity:     arena.size,
        size:         size,
        align:        align,
    )!;

    arena.retire[(arena.retire_head + arena.retire_count) % TRANSFER_RETIRE_CAP] = {
        .virtual_end    = range.virtual_end,
        .timeline_value = tag.value,
        .timeline_kind  = tag.kind,
    };
    if (arena.retire_count == 0) arena.virtual_head = 0;
    arena.retire_count++;
    arena.virtual_tail = range.virtual_end;
    return arena.span.unchecked_subspan(range.physical_offset, size);
}

fn void? drain_dedicated_transfers(VkDeviceState* state) @private {
    if (state.dedicated_staging.count == 0) return;
    gpu::Device shim = device_shim(state);
    ulong frame_current = 0;
    ulong helper_current = 0;
    check_device_result(state, vk::get_semaphore_counter_value(state.device, state.frame_timeline, &frame_current))!;
    check_device_result(state, vk::get_semaphore_counter_value(state.device, state.helper_timeline, &helper_current))!;
    while (state.dedicated_staging.count > 0) {
        DedicatedStaging* entry = &state.dedicated_staging.entries[state.dedicated_staging.head];
        ulong current = entry.timeline_kind == TimelineKind.HELPER ? helper_current : frame_current;
        if (entry.timeline_value > current) break;
        (void)vk_destroy_buffer(&shim, entry.buffer);
        state.dedicated_staging.head = (state.dedicated_staging.head + 1) % DEDICATED_RETIRE_CAP;
        state.dedicated_staging.count--;
    }
}

<* Span from the arena, or a dedicated one-shot buffer when it cannot fit.
   The fallback's creation runs outside transfer_mutex; only ring bookkeeping
   is locked, and the full re-check destroys the fresh buffer. *>
fn gpu::GpuSpan? transfer_alloc(VkDeviceState* state, TransferArenaState* arena, usz size, usz align, gpu::BufferDesc* dedicated_desc, TimelineTag tag) @private {
    if (size == 0 || !is_pow2(align)) return gpu::INVALID_ARGUMENT~;
    {
        state.transfer_mutex.lock()!!;
        defer state.transfer_mutex.unlock();
        drain_dedicated_transfers(state)!;
        gpu::GpuSpan? arena_span = transfer_arena_alloc(
            state: state,
            arena: arena,
            size:  size,
            align: align,
            tag:   tag,
        );
        if (catch allocation_fault = arena_span) {
            if (allocation_fault != gpu::ARENA_FULL) return allocation_fault~;
        } else {
            return arena_span;
        }
        if (state.dedicated_staging.count == DEDICATED_RETIRE_CAP) return gpu::ARENA_FULL~;
    }

    gpu::Device shim = device_shim(state);
    gpu::BufferHandle buffer = vk_create_buffer(&shim, dedicated_desc)!;
    defer catch (void)vk_destroy_buffer(&shim, buffer);
    gpu::GpuSpan span = vk_get_buffer_span(&shim, buffer)!;

    state.transfer_mutex.lock()!!;
    defer state.transfer_mutex.unlock();
    drain_dedicated_transfers(state)!;
    if (state.dedicated_staging.count == DEDICATED_RETIRE_CAP) return gpu::ARENA_FULL~;
    mark_buffer_internal(state, buffer);
    state.dedicated_staging.entries[(state.dedicated_staging.head + state.dedicated_staging.count) % DEDICATED_RETIRE_CAP] = {
        .buffer         = buffer,
        .timeline_value = tag.value,
        .timeline_kind  = tag.kind,
    };
    state.dedicated_staging.count++;
    state.dedicated_fallbacks++;
    return span;
}

fn gpu::GpuSpan? staging_alloc(VkDeviceState* state, usz size, usz align, TimelineTag tag) @private {
    gpu::BufferDesc desc = staging_dedicated_desc(size, state.selected_queue_roles);
    return transfer_alloc(
        state:          state,
        arena:          &state.staging_arena,
        size:           size,
        align:          align,
        dedicated_desc: &desc,
        tag:            tag,
    );
}

fn gpu::GpuSpan? readback_alloc(VkDeviceState* state, usz size, usz align, TimelineTag tag) @private {
    gpu::BufferDesc desc = {
        .size        = size,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::MemoryKind.READBACK,
        .access      = state.selected_queue_roles,
        .debug_name  = "readback_dedicated",
    };
    return transfer_alloc(
        state:          state,
        arena:          &state.readback_arena,
        size:           size,
        align:          align,
        dedicated_desc: &desc,
        tag:            tag,
    );
}

fn BufferSlot*? resolve_transfer_buffer(
    VkDeviceState* state,
    gpu::BufferHandle handle,
    ZString operation,
    ZString rejected_field,
    ZString invariant,
) @private {
    BufferSlot*? resolved = state.buffers.get(handle);
    if (catch lookup_fault = resolved) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        lookup_fault,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      rejected_field,
            invariant:           invariant,
            backend_text:        null,
        )~;
    }
    return resolved;
}

fn TextureSlot*? resolve_transfer_texture(
    VkDeviceState* state,
    gpu::TextureHandle handle,
    ZString operation,
    ZString rejected_field,
    ZString invariant,
) @private {
    TextureSlot*? resolved = state.textures.get(handle);
    if (catch lookup_fault = resolved) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        lookup_fault,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      rejected_field,
            invariant:           invariant,
            backend_text:        null,
        )~;
    }
    return resolved;
}

<* Byte size of a texture-upload region with the copy-desc 0-defaults
   resolved. The copy command re-validates independently. *>
fn usz? texture_upload_bytes(VkDeviceState* state, gpu::TextureUploadDesc* desc, ZString operation, ZString rejected_field) @private {
    TextureSlot* slot = resolve_transfer_texture(
        state:          state,
        handle:         desc.texture,
        operation:      operation,
        rejected_field: "TextureUploadDesc.texture",
        invariant:      "texture handle must resolve to a live resource",
    )!;
    if (desc.mip >= slot.mip_levels) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               desc.texture.index,
            generation:          desc.texture.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      rejected_field,
            invariant:           "mip must exist in the texture mip chain",
            backend_text:        null,
        )~;
    }
    MipExtent extent = texture_mip_extent(slot, desc.mip);
    uint width  = desc.width == 0 ? extent.width : desc.width;
    uint height = desc.height == 0 ? extent.height : desc.height;
    uint layers = desc.layer_count == 0 ? 1 : desc.layer_count;
    return (usz)width * height * layers * format_texel_size(slot.format);
}

fn CommandRecord*? resolve_transfer_command(gpu::CommandList* commands, ZString operation) @private {
    if (commands == null || !commands.handle.is_valid()) return gpu::INVALID_HANDLE~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    if (state == null) return gpu::INVALID_HANDLE~;
    CommandRecord*? resolved = state.commands.get(commands.handle);
    if (catch lookup_fault = resolved) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        lookup_fault,
            resource_kind:       gpu::DebugResourceKind.COMMAND_LIST,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "CommandList.handle",
            invariant:           "command list handle must identify a live command list",
            backend_text:        null,
        )~;
    }
    return resolved;
}

fn BufferSlot*? validate_upload_buffer(VkDeviceState* state, gpu::BufferHandle dst, usz offset, char[] data, ZString operation) @private {
    if (data.len == 0) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "data",
            invariant:           "upload data must not be empty",
            backend_text:        null,
        )~;
    }
    BufferSlot* dst_slot = resolve_transfer_buffer(
        state:          state,
        handle:         dst,
        operation:      operation,
        rejected_field: "dst",
        invariant:      "destination buffer handle must resolve to a live resource",
    )!;
    if (!dst_slot.usage.transfer_dst) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               dst.index,
            generation:          dst.generation,
            debug_name:          slot_name(&dst_slot.debug_name[0]),
            rejected_field:      "BufferDesc.usage",
            invariant:           "destination buffer must include transfer_dst",
            backend_text:        null,
        )~;
    }
    if (!gpu::range_fits(dst_slot.size, offset, data.len)) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               dst.index,
            generation:          dst.generation,
            debug_name:          slot_name(&dst_slot.debug_name[0]),
            rejected_field:      "offset/data",
            invariant:           "upload range must fit the destination buffer",
            backend_text:        null,
        )~;
    }
    return dst_slot;
}

<* The caller supplies the timeline tag its staging range retires against. *>
fn void? cmd_upload_buffer_impl(gpu::CommandList* commands, gpu::BufferHandle dst, usz offset, char[] data, TimelineTag tag, ZString operation) @private {
    CommandRecord* record = resolve_transfer_command(commands, operation)!;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    if (record.state != CommandState.RECORDING) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
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
    BufferSlot* dst_slot = validate_upload_buffer(state, dst, offset, data, operation)!;
    validate_buffer_recording_access(
        state:     state,
        record:    record,
        slot:      dst_slot,
        handle:    dst,
        operation: operation,
    )!;

    gpu::GpuSpan? allocated_staging = staging_alloc(
        state: state,
        size:  data.len,
        align: TRANSFER_ALIGN,
        tag:   tag,
    );
    if (catch allocation_fault = allocated_staging) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        allocation_fault,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               dst.index,
            generation:          dst.generation,
            debug_name:          slot_name(&dst_slot.debug_name[0]),
            rejected_field:      "data",
            invariant:           "staging capacity must be available for the upload",
            backend_text:        null,
        )~;
    }
    gpu::GpuSpan staging = allocated_staging;
    mem::copy(staging.cpu, data.ptr, data.len);
    vk_flush_buffer(&commands.device, staging.buffer, staging.offset, staging.size)!;

    gpu::BufferCopyDesc copy = {
        .src        = staging.buffer,
        .dst        = dst,
        .src_offset = staging.offset,
        .dst_offset = offset,
        .size       = data.len,
    };
    return vk_cmd_copy_buffer(commands, &copy);
}

fn void? vk_cmd_upload_buffer(gpu::CommandList* commands, gpu::BufferHandle dst, usz offset, char[] data) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    TimelineTag tag = { .kind = TimelineKind.FRAME, .value = state.timeline_counter.load(AtomicOrdering.RELAXED) + 1 };
    return cmd_upload_buffer_impl(
        commands: commands,
        dst:      dst,
        offset:   offset,
        data:     data,
        tag:      tag,
        operation: "cmd_upload_buffer",
    );
}

fn void? cmd_upload_texture_impl(gpu::CommandList* commands, gpu::TextureUploadDesc* desc, char[] data, TimelineTag tag, ZString operation, ZString mip_field) @private {
    CommandRecord* record = resolve_transfer_command(commands, operation)!;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    if (record.state != CommandState.RECORDING) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
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
    TextureSlot* slot = resolve_transfer_texture(
        state:          state,
        handle:         desc.texture,
        operation:      operation,
        rejected_field: "TextureUploadDesc.texture",
        invariant:      "destination texture handle must resolve to a live resource",
    )!;
    usz expected = texture_upload_bytes(
        state:          state,
        desc:           desc,
        operation:      operation,
        rejected_field: mip_field,
    )!;
    if (data.len != expected) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               desc.texture.index,
            generation:          desc.texture.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "data",
            invariant:           "data length must match the selected texture mip",
            backend_text:        null,
        )~;
    }
    if (!slot.usage.transfer_dst) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               desc.texture.index,
            generation:          desc.texture.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "TextureDesc.usage",
            invariant:           "destination texture must include transfer_dst",
            backend_text:        null,
        )~;
    }
    if (effective_layout(record, slot, desc.texture) != gpu::TextureLayout.TRANSFER_DST) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_RESOURCE_STATE,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               desc.texture.index,
            generation:          desc.texture.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "TextureUploadDesc.texture",
            invariant:           "destination texture must be in TRANSFER_DST layout",
            backend_text:        null,
        )~;
    }
    validate_texture_recording_access(
        state:     state,
        record:    record,
        slot:      slot,
        handle:    desc.texture,
        operation: operation,
    )!;
    usz align = format_texel_size(slot.format);
    if (align < 4) align = 4;
    gpu::GpuSpan? allocated_staging = staging_alloc(
        state: state,
        size:  data.len,
        align: align,
        tag:   tag,
    );
    if (catch allocation_fault = allocated_staging) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        allocation_fault,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               desc.texture.index,
            generation:          desc.texture.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "data",
            invariant:           "staging capacity must be available for the upload",
            backend_text:        null,
        )~;
    }
    gpu::GpuSpan staging = allocated_staging;
    mem::copy(staging.cpu, data.ptr, data.len);
    vk_flush_buffer(&commands.device, staging.buffer, staging.offset, staging.size)!;

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

fn void? vk_cmd_upload_texture(gpu::CommandList* commands, gpu::TextureUploadDesc* desc, char[] data) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    TimelineTag tag = { .kind = TimelineKind.FRAME, .value = state.timeline_counter.load(AtomicOrdering.RELAXED) + 1 };
    return cmd_upload_texture_impl(
        commands:  commands,
        desc:      desc,
        data:      data,
        tag:       tag,
        operation: "cmd_upload_texture",
        mip_field: "TextureUploadDesc.mip",
    );
}

<* Reserved under transfer_mutex, but the ordering guarantee comes from the
   caller's helper_record_mutex window, not this lock: only one thread can
   reserve-and-record at a time, so reservation order matches recording
   order and every tag agrees with the value signal_helper_timeline signals. *>
fn ulong reserve_helper_value(VkDeviceState* state) @private {
    state.transfer_mutex.lock()!!;
    defer state.transfer_mutex.unlock();
    state.helper_counter++;
    return state.helper_counter;
}

<* Route helper Vulkan results; swallowed cleanup omits public-fault metadata. *>
fn void? check_helper_backend_result(
    VkDeviceState* state,
    ZString operation,
    vk::Result result,
    bool fault_is_public,
) {
    if (fault_is_public) return check_backend_result(state, operation, result);
    observe_device_result(state, result);
    bool stderr_fallback = state == null || state.debug_callback == null;
    fault mapped = @catch(map_result(result, stderr_fallback));
    if (!mapped) return;
    if (!stderr_fallback) {
        gpu::DebugMessage message = {
            .severity             = gpu::DebugMessageSeverity.ERROR,
            .category             = { .backend },
            .operation            = operation,
            .backend_text         = backend_result_text(result),
            .validation_id_number = (int)result,
        };
        gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
    }
    return mapped~;
}

<* Wait for `v - 1`, then signal `v`: signals land in reservation order.
   Mutex-free — a mutex would deadlock the predecessor out of its own signal. *>
fn void? signal_helper_timeline(
    VkDeviceState* state,
    ulong v,
    ZString operation,
    bool fault_is_public,
) @private {
    ulong prior = v - 1;
    ulong current = 0;
    check_helper_backend_result(
        state:           state,
        operation:       operation,
        result:          vk::get_semaphore_counter_value(state.device, state.helper_timeline, &current),
        fault_is_public: fault_is_public,
    )!;
    while (current < prior) {
        vk::SemaphoreWaitInfo wait_info = vk::semaphore_wait_info()
            .set_semaphore_count(1)
            .set_semaphores((&state.helper_timeline)[:1])
            .set_values((&prior)[:1]);
        vk::Result wait_result = vk::wait_semaphores(state.device, &wait_info, HELPER_TURNSTILE_ROUND_NS);
        if (wait_result == vk::Result.SUCCESS) break;
        if (wait_result != vk::Result.TIMEOUT) {
            return check_helper_backend_result(
                state:           state,
                operation:       operation,
                result:          wait_result,
                fault_is_public: fault_is_public,
            );
        }
        io::eprintfn("gpu: helper turnstile %d still waiting on %d", v, prior);
        check_helper_backend_result(
            state:           state,
            operation:       operation,
            result:          vk::get_semaphore_counter_value(state.device, state.helper_timeline, &current),
            fault_is_public: fault_is_public,
        )!;
    }
    vk::SemaphoreSignalInfo signal_info = vk::semaphore_signal_info()
        .set_semaphore(state.helper_timeline)
        .set_value(v);
    return check_helper_backend_result(
        state:           state,
        operation:       operation,
        result:          vk::signal_semaphore(state.device, &signal_info),
        fault_is_public: fault_is_public,
    );
}

fn void signal_helper_timeline_cleanup(VkDeviceState* state, ulong v, ZString operation) @private {
    (void)signal_helper_timeline(
        state:           state,
        v:               v,
        operation:       operation,
        fault_is_public: false,
    );
}

fn void? finish_helper_timeline(VkDeviceState* state, ulong* helper_value, ZString operation) @private {
    fault public_fault = @catch(signal_helper_timeline(
        state:           state,
        v:               *helper_value,
        operation:       operation,
        fault_is_public: true,
    ));
    if (!public_fault) {
        *helper_value = 0;
        return;
    }

    ulong retry_value = *helper_value;
    *helper_value = 0;
    if (public_fault != gpu::DEVICE_LOST) {
        signal_helper_timeline_cleanup(state, retry_value, operation);
    }
    return public_fault~;
}

<* Call under helper_record_mutex before this call reserves: every reserved
   value signaled means no helper buffer is live anywhere (helpers end
   recording before releasing the mutex and signal after queue idle). *>
fn void? maybe_reset_helper_pool(VkDeviceState* state, ZString operation) @private {
    RecordingContextState* context = state.recording_contexts.get(state.helper_context)!;
    PoolDirtyMask* dirty = &context.pool_dirty[state.frame_index];
    gpu::QueueKind queue = frame_queue_kind(state);
    bool pool_dirty;
    switch (queue) {
        case gpu::QueueKind.COMPUTE:
            pool_dirty = dirty.compute;
        case gpu::QueueKind.TRANSFER:
            pool_dirty = dirty.transfer;
        default:
            pool_dirty = dirty.graphics;
    }
    if (!pool_dirty) return;
    state.transfer_mutex.lock()!!;
    ulong reserved = state.helper_counter;
    state.transfer_mutex.unlock();
    ulong current = 0;
    check_helper_backend_result(
        state:           state,
        operation:       operation,
        result:          vk::get_semaphore_counter_value(state.device, state.helper_timeline, &current),
        fault_is_public: true,
    )!;
    if (current != reserved) return;
    release_commands_for_context_frame(state, state.helper_context, state.frame_index);
    check_helper_backend_result(
        state:           state,
        operation:       operation,
        result:          vk::reset_command_pool(
            state.device,
            pool_for(context, queue, state.frame_index),
            (vk::CommandPoolResetFlags)0,
        ),
        fault_is_public: true,
    )!;
    switch (queue) {
        case gpu::QueueKind.COMPUTE:
            dirty.compute = false;
        case gpu::QueueKind.TRANSFER:
            dirty.transfer = false;
        default:
            dirty.graphics = false;
    }
    state.helper_pool_resets++;
}

<* Cumulative helper-pool resets, whitebox-exposed for tests. *>
fn ulong helper_pool_reset_count(VkDeviceState* state) {
    return state.helper_pool_resets;
}

<* Cumulative dedicated-fallback registrations (created-then-destroyed
   full-path buffers are not counted), whitebox-exposed for tests. *>
fn ulong dedicated_fallback_count(VkDeviceState* state) {
    return state.dedicated_fallbacks;
}

fn void? vk_upload_buffer_data(gpu::Device* device, gpu::BufferHandle dst, usz offset, char[] data, gpu::Stage next_stage, gpu::Hazard next_hazard) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    BufferSlot* dst_slot = validate_upload_buffer(
        state:     state,
        dst:       dst,
        offset:    offset,
        data:      data,
        operation: "upload_buffer_data",
    )!;
    gpu::QueueKind queue = frame_queue_kind(state);
    validate_buffer_queue_access(
        state:     state,
        queue:     queue,
        slot:      dst_slot,
        handle:    dst,
        operation: "upload_buffer_data",
    )!;
    ulong helper_value = 0;
    defer if (helper_value != 0) signal_helper_timeline_cleanup(state, helper_value, "upload_buffer_data");
    TimelineTag tag = { .kind = TimelineKind.HELPER };

    gpu::CommandList commands;
    {
        state.helper_record_mutex.lock()!!;
        defer state.helper_record_mutex.unlock();
        maybe_reset_helper_pool(state, "upload_buffer_data")!;
        helper_value = reserve_helper_value(state);
        tag.value = helper_value;
        vk_begin_commands(
            device:       device,
            queue:        queue,
            ctx:          state.helper_context,
            out_commands: &commands,
        )!;
        cmd_upload_buffer_impl(
            commands: &commands,
            dst:      dst,
            offset:   offset,
            data:     data,
            tag:      tag,
            operation: "upload_buffer_data",
        )!;

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
        vk_end_commands(&commands)!;
    }
    submit_and_wait(device, &commands, queue)!;
    finish_helper_timeline(state, &helper_value, "upload_buffer_data")!;
}

fn void? vk_upload_texture_data(gpu::Device* device, gpu::TextureHandle dst, uint mip, char[] data, gpu::Stage next_stage, gpu::Hazard next_hazard, gpu::TextureLayout next_layout) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    ulong helper_value = 0;
    defer if (helper_value != 0) signal_helper_timeline_cleanup(state, helper_value, "upload_texture_data");
    TimelineTag tag = { .kind = TimelineKind.HELPER };
    gpu::QueueKind queue = frame_queue_kind(state);

    TextureSlot* slot = resolve_transfer_texture(
        state:          state,
        handle:         dst,
        operation:      "upload_texture_data",
        rejected_field: "dst",
        invariant:      "destination texture handle must resolve to a live resource",
    )!;
    gpu::TextureLayout tracked = texture_layout(slot);
    if (tracked != gpu::TextureLayout.UNDEFINED
        && tracked != gpu::TextureLayout.TRANSFER_DST) {
        return report_contract_failure(
            state:               state,
            operation:           "upload_texture_data",
            public_fault:        gpu::INVALID_RESOURCE_STATE,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               dst.index,
            generation:          dst.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "dst",
            invariant:           "destination texture must be UNDEFINED or TRANSFER_DST before upload",
            backend_text:        null,
        )~;
    }
    // Validate before recording: a fault after the entry barrier would leave
    // the tracked layout claiming a transition the GPU never ran.
    gpu::TextureUploadDesc probe = { .texture = dst, .mip = mip };
    usz expected = texture_upload_bytes(
        state:          state,
        desc:           &probe,
        operation:      "upload_texture_data",
        rejected_field: "mip",
    )!;
    if (data.len != expected) {
        return report_contract_failure(
            state:               state,
            operation:           "upload_texture_data",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               dst.index,
            generation:          dst.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "data",
            invariant:           "data length must match the selected texture mip",
            backend_text:        null,
        )~;
    }
    if (!slot.usage.transfer_dst) {
        return report_contract_failure(
            state:               state,
            operation:           "upload_texture_data",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               dst.index,
            generation:          dst.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "TextureDesc.usage",
            invariant:           "destination texture must include transfer_dst",
            backend_text:        null,
        )~;
    }
    validate_texture_queue_access(
        state:     state,
        queue:     queue,
        slot:      slot,
        handle:    dst,
        operation: "upload_texture_data",
    )!;

    gpu::CommandList commands;
    {
        state.helper_record_mutex.lock()!!;
        defer state.helper_record_mutex.unlock();
        maybe_reset_helper_pool(state, "upload_texture_data")!;
        helper_value = reserve_helper_value(state);
        tag.value = helper_value;
        vk_begin_commands(
            device:       device,
            queue:        queue,
            ctx:          state.helper_context,
            out_commands: &commands,
        )!;

        if (tracked == gpu::TextureLayout.UNDEFINED) {
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
        cmd_upload_texture_impl(
            commands:  &commands,
            desc:      &upload,
            data:      data,
            tag:       tag,
            operation: "upload_texture_data",
            mip_field: "mip",
        )!;

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
        vk_end_commands(&commands)!;
    }
    submit_and_wait(device, &commands, queue)!;
    finish_helper_timeline(state, &helper_value, "upload_texture_data")!;
}

fn void? vk_readback_buffer_data(gpu::Device* device, gpu::BufferHandle src, usz offset, char[] out_data, gpu::Stage from_stage, gpu::Hazard from_hazard) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    ulong helper_value = 0;
    defer if (helper_value != 0) signal_helper_timeline_cleanup(state, helper_value, "readback_buffer_data");
    TimelineTag tag = { .kind = TimelineKind.HELPER };

    gpu::QueueKind queue = frame_queue_kind(state);
    if (out_data.len == 0) {
        return report_contract_failure(
            state:               state,
            operation:           "readback_buffer_data",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "out_data",
            invariant:           "destination must not be empty",
            backend_text:        null,
        )~;
    }
    BufferSlot* src_slot = resolve_transfer_buffer(
        state:          state,
        handle:         src,
        operation:      "readback_buffer_data",
        rejected_field: "src",
        invariant:      "source buffer handle must resolve to a live resource",
    )!;
    if (!src_slot.usage.transfer_src) {
        return report_contract_failure(
            state:               state,
            operation:           "readback_buffer_data",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&src_slot.debug_name[0]),
            rejected_field:      "BufferDesc.usage",
            invariant:           "source buffer must include transfer_src",
            backend_text:        null,
        )~;
    }
    if (!gpu::range_fits(src_slot.size, offset, out_data.len)) {
        return report_contract_failure(
            state:               state,
            operation:           "readback_buffer_data",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&src_slot.debug_name[0]),
            rejected_field:      "offset/out_data",
            invariant:           "readback range must fit the source buffer",
            backend_text:        null,
        )~;
    }
    validate_buffer_queue_access(
        state:     state,
        queue:     queue,
        slot:      src_slot,
        handle:    src,
        operation: "readback_buffer_data",
    )!;

    gpu::CommandList commands;
    gpu::GpuSpan readback;
    fault allocation_fault;
    {
        state.helper_record_mutex.lock()!!;
        defer state.helper_record_mutex.unlock();
        maybe_reset_helper_pool(state, "readback_buffer_data")!;
        helper_value = reserve_helper_value(state);
        tag.value = helper_value;
        gpu::GpuSpan? allocated_readback = readback_alloc(
            state: state,
            size:  out_data.len,
            align: TRANSFER_ALIGN,
            tag:   tag,
        );

        if (catch err = allocated_readback) {
            allocation_fault = err;
        } else {
            readback = allocated_readback;
            vk_begin_commands(
                device:       device,
                queue:        queue,
                ctx:          state.helper_context,
                out_commands: &commands,
            )!;

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
            vk_end_commands(&commands)!;
        }
    }
    if (allocation_fault) {
        return report_contract_failure(
            state:               state,
            operation:           "readback_buffer_data",
            public_fault:        allocation_fault,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&src_slot.debug_name[0]),
            rejected_field:      "out_data",
            invariant:           "readback capacity must be available for the destination",
            backend_text:        null,
        )~;
    }

    submit_and_wait(device, &commands, queue)!;
    vk_invalidate_buffer(device, readback.buffer, readback.offset, readback.size)!;
    mem::copy(out_data.ptr, readback.cpu, out_data.len);
    finish_helper_timeline(state, &helper_value, "readback_buffer_data")!;
}

fn void? vk_readback_texture_data(gpu::Device* device, gpu::TextureHandle src, uint mip, char[] out_data, gpu::Stage from_stage, gpu::Hazard from_hazard, gpu::TextureLayout from_layout) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    ulong helper_value = 0;
    defer if (helper_value != 0) signal_helper_timeline_cleanup(state, helper_value, "readback_texture_data");
    TimelineTag tag = { .kind = TimelineKind.HELPER };

    gpu::QueueKind queue = frame_queue_kind(state);
    gpu::TextureUploadDesc region = { .texture = src, .mip = mip };
    TextureSlot* slot = resolve_transfer_texture(
        state:          state,
        handle:         src,
        operation:      "readback_texture_data",
        rejected_field: "src",
        invariant:      "source texture handle must resolve to a live resource",
    )!;
    usz expected = texture_upload_bytes(
        state:          state,
        desc:           &region,
        operation:      "readback_texture_data",
        rejected_field: "mip",
    )!;
    if (out_data.len != expected) {
        return report_contract_failure(
            state:               state,
            operation:           "readback_texture_data",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "out_data",
            invariant:           "destination length must match the selected texture mip",
            backend_text:        null,
        )~;
    }
    if (!slot.usage.transfer_src) {
        return report_contract_failure(
            state:               state,
            operation:           "readback_texture_data",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "TextureDesc.usage",
            invariant:           "source texture must include transfer_src",
            backend_text:        null,
        )~;
    }
    validate_texture_queue_access(
        state:     state,
        queue:     queue,
        slot:      slot,
        handle:    src,
        operation: "readback_texture_data",
    )!;
    usz align = format_texel_size(slot.format);
    if (align < 4) align = 4;
    gpu::CommandList commands;
    gpu::GpuSpan readback;
    fault allocation_fault;
    {
        state.helper_record_mutex.lock()!!;
        defer state.helper_record_mutex.unlock();
        maybe_reset_helper_pool(state, "readback_texture_data")!;
        helper_value = reserve_helper_value(state);
        tag.value = helper_value;
        gpu::GpuSpan? allocated_readback = readback_alloc(
            state: state,
            size:  out_data.len,
            align: align,
            tag:   tag,
        );

        if (catch err = allocated_readback) {
            allocation_fault = err;
        } else {
            readback = allocated_readback;
            vk_begin_commands(
                device:       device,
                queue:        queue,
                ctx:          state.helper_context,
                out_commands: &commands,
            )!;

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
            vk_end_commands(&commands)!;
        }
    }
    if (allocation_fault) {
        return report_contract_failure(
            state:               state,
            operation:           "readback_texture_data",
            public_fault:        allocation_fault,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "out_data",
            invariant:           "readback capacity must be available for the destination",
            backend_text:        null,
        )~;
    }

    submit_and_wait(device, &commands, queue)!;
    vk_invalidate_buffer(device, readback.buffer, readback.offset, readback.size)!;
    mem::copy(out_data.ptr, readback.cpu, out_data.len);
    finish_helper_timeline(state, &helper_value, "readback_texture_data")!;
}

<* Submits an already-ended command list, blocks until queue idle. Touches
   no timeline — the caller owns its tag's signaling. Runs after the caller
   releases helper_record_mutex, so a successor records while this waits. *>
fn void? submit_and_wait(
    gpu::Device* device,
    gpu::CommandList* commands,
    gpu::QueueKind queue,
) @private {
    gpu::CommandList[1] lists = { *commands };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    vk_submit(device, &submit)!;
    return vk_wait_queue_idle(device, queue);
}

struct TicketAlloc @private {
    gpu::GpuSpan span;
    ulong        range_end;
}

<* Arena range for a ticket: allocated like any transfer range but pinned, so
   FIFO drain holds at it until resolve unpins. *>
fn TicketAlloc? ticket_arena_alloc(VkDeviceState* state, TransferArenaState* arena, usz size, TimelineTag tag) @private {
    gpu::GpuSpan span = transfer_arena_alloc(
        state: state,
        arena: arena,
        size:  size,
        align: TRANSFER_ALIGN,
        tag:   tag,
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
   (destroyed at resolve) when the arena cannot fit the request. Tickets only
   come from Tier C recording (cmd_readback_buffer/cmd_readback_texture),
   never from a blocking helper, so they always tag FRAME. The dedicated
   buffer's VMA creation runs outside transfer_mutex, matching transfer_alloc
   — a ticket's fallback owns its buffer directly (no shared retire ring), so
   there is nothing to re-check on relock beyond the counter bump. *>
fn gpu::ReadbackTicket? ticket_alloc(VkDeviceState* state, usz size) @private {
    TimelineTag tag = { .kind = TimelineKind.FRAME, .value = state.timeline_counter.load(AtomicOrdering.RELAXED) + 1 };
    {
        state.transfer_mutex.lock()!!;
        defer state.transfer_mutex.unlock();
        TicketAlloc? arena_alloc = ticket_arena_alloc(
            state: state,
            arena: &state.readback_arena,
            size:  size,
            tag:   tag,
        );
        if (catch allocation_fault = arena_alloc) {
            if (allocation_fault != gpu::ARENA_FULL) return allocation_fault~;
        } else {
            TicketAlloc alloc = arena_alloc;
            ReadbackTicketState data = {
                .span      = alloc.span,
                .value     = (gpu::SemaphoreValue)tag.value,
                .range_end = alloc.range_end,
            };
            gpu::ReadbackTicket? ticket = state.readback_tickets.alloc(data);
            if (catch ticket_fault = ticket) {
                unpin_ticket_range(&state.readback_arena, alloc.range_end);
                return ticket_fault~;
            }
            return ticket;
        }
    }

    gpu::BufferDesc desc = {
        .size        = size,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::MemoryKind.READBACK,
        .access      = state.selected_queue_roles,
        .debug_name  = "readback_ticket",
    };
    gpu::Device shim = device_shim(state);
    gpu::BufferHandle buffer = vk_create_buffer(&shim, &desc)!;
    if (try span = vk_get_buffer_span(&shim, buffer)) {
        ReadbackTicketState data = {
            .span      = span,
            .value     = (gpu::SemaphoreValue)tag.value,
            .dedicated = buffer,
        };
        state.transfer_mutex.lock()!!;
        mark_buffer_internal(state, buffer);
        state.dedicated_fallbacks++;
        gpu::ReadbackTicket? ticket = state.readback_tickets.alloc(data);
        state.transfer_mutex.unlock();
        if (catch allocation_fault = ticket) {
            (void)vk_destroy_buffer(&shim, buffer);
            return allocation_fault~;
        }
        return ticket;
    }
    (void)vk_destroy_buffer(&shim, buffer);
    return gpu::ARENA_FULL~;
}

fn void? release_ticket(VkDeviceState* state, gpu::ReadbackTicket* ticket) @private {
    state.transfer_mutex.lock()!!;
    defer state.transfer_mutex.unlock();
    ReadbackTicketState* data = resolve_readback_ticket(state, *ticket)!;
    release_readback_state(state, data);
    state.readback_tickets.free(*ticket)!;
    *ticket = {};
}

<* Record the copy plus the transfer→host barrier on the destination range;
   source-side ordering stays with the caller. *>
fn gpu::ReadbackTicket? vk_cmd_readback_buffer(gpu::CommandList* commands, gpu::BufferHandle src, usz offset, usz size) @private {
    CommandRecord* record = resolve_transfer_command(commands, "cmd_readback_buffer")!;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    if (record.state != CommandState.RECORDING) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_readback_buffer",
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
    if (size == 0) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_readback_buffer",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "size",
            invariant:           "readback size must be greater than zero",
            backend_text:        null,
        )~;
    }
    BufferSlot* src_slot = resolve_transfer_buffer(
        state:          state,
        handle:         src,
        operation:      "cmd_readback_buffer",
        rejected_field: "src",
        invariant:      "source buffer handle must resolve to a live resource",
    )!;
    if (!src_slot.usage.transfer_src) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_readback_buffer",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&src_slot.debug_name[0]),
            rejected_field:      "BufferDesc.usage",
            invariant:           "source buffer must include transfer_src",
            backend_text:        null,
        )~;
    }
    if (!gpu::range_fits(src_slot.size, offset, size)) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_readback_buffer",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&src_slot.debug_name[0]),
            rejected_field:      "offset/size",
            invariant:           "readback range must fit the source buffer",
            backend_text:        null,
        )~;
    }
    validate_buffer_recording_access(
        state:     state,
        record:    record,
        slot:      src_slot,
        handle:    src,
        operation: "cmd_readback_buffer",
    )!;

    gpu::ReadbackTicket? allocated_ticket = ticket_alloc(state, size);
    if (catch allocation_fault = allocated_ticket) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_readback_buffer",
            public_fault:        allocation_fault,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&src_slot.debug_name[0]),
            rejected_field:      "size",
            invariant:           "readback capacity must be available for the ticket",
            backend_text:        null,
        )~;
    }
    gpu::ReadbackTicket ticket = allocated_ticket;
    defer catch (void)release_ticket(state, &ticket);
    ReadbackTicketState* ticket_data = resolve_readback_ticket(state, ticket)!;

    gpu::BufferCopyDesc copy = {
        .src        = src,
        .dst        = ticket_data.span.buffer,
        .src_offset = offset,
        .dst_offset = ticket_data.span.offset,
        .size       = size,
    };
    vk_cmd_copy_buffer(commands, &copy)!;
    record_ticket_host_barrier(commands, ticket_data)!;
    return ticket;
}

<* Full-mip texture variant; requires transfer_src usage and the TRANSFER_SRC
   tracked layout (caller's barriers on both sides). *>
fn gpu::ReadbackTicket? vk_cmd_readback_texture(gpu::CommandList* commands, gpu::TextureHandle src, uint mip) @private {
    CommandRecord* record = resolve_transfer_command(commands, "cmd_readback_texture")!;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    if (record.state != CommandState.RECORDING) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_readback_texture",
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
    gpu::TextureUploadDesc region = { .texture = src, .mip = mip };
    TextureSlot* slot = resolve_transfer_texture(
        state:          state,
        handle:         src,
        operation:      "cmd_readback_texture",
        rejected_field: "src",
        invariant:      "source texture handle must resolve to a live resource",
    )!;
    usz size = texture_upload_bytes(
        state:          state,
        desc:           &region,
        operation:      "cmd_readback_texture",
        rejected_field: "mip",
    )!;
    if (!slot.usage.transfer_src) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_readback_texture",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "TextureDesc.usage",
            invariant:           "source texture must include transfer_src",
            backend_text:        null,
        )~;
    }
    if (effective_layout(record, slot, src) != gpu::TextureLayout.TRANSFER_SRC) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_readback_texture",
            public_fault:        gpu::INVALID_RESOURCE_STATE,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "src",
            invariant:           "source texture must be in TRANSFER_SRC layout",
            backend_text:        null,
        )~;
    }
    validate_texture_recording_access(
        state:     state,
        record:    record,
        slot:      slot,
        handle:    src,
        operation: "cmd_readback_texture",
    )!;

    gpu::ReadbackTicket? allocated_ticket = ticket_alloc(state, size);
    if (catch allocation_fault = allocated_ticket) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_readback_texture",
            public_fault:        allocation_fault,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               src.index,
            generation:          src.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "mip",
            invariant:           "readback capacity must be available for the ticket",
            backend_text:        null,
        )~;
    }
    gpu::ReadbackTicket ticket = allocated_ticket;
    defer catch (void)release_ticket(state, &ticket);
    ReadbackTicketState* ticket_data = resolve_readback_ticket(state, ticket)!;

    gpu::TextureBufferCopyDesc copy = {
        .texture = src,
        .dst     = ticket_data.span,
        .mip     = mip,
    };
    vk_cmd_copy_texture_to_buffer(commands, &copy)!;
    record_ticket_host_barrier(commands, ticket_data)!;
    return ticket;
}

fn void? record_ticket_host_barrier(gpu::CommandList* commands, ReadbackTicketState* ticket) @private {
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

fn bool? vk_poll_readback(gpu::Device* device, gpu::ReadbackTicket* ticket) @private {
    if (ticket == null || !ticket.is_valid()) return false;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    gpu::SemaphoreValue value;
    state.transfer_mutex.lock()!!;
    if (try data = resolve_readback_ticket(state, *ticket)) {
        value = data.value;
    } else {
        state.transfer_mutex.unlock();
        return gpu::INVALID_HANDLE~;
    }
    state.transfer_mutex.unlock();
    ulong current = 0;
    check_device_result(state, vk::get_semaphore_counter_value(state.device, state.frame_timeline, &current))!;
    return current >= (ulong)value;
}

fn void? vk_resolve_readback(gpu::Device* device, gpu::ReadbackTicket* ticket, char[] dest) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    if (ticket == null || !ticket.is_valid()) {
        return report_contract_failure(
            state:               state,
            operation:           "resolve_readback",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "ticket",
            invariant:           "ticket must be live and unresolved",
            backend_text:        null,
        )~;
    }

    state.transfer_mutex.lock()!!;
    defer state.transfer_mutex.unlock();
    ReadbackTicketState* data;
    if (try resolved = resolve_readback_ticket(state, *ticket)) {
        data = resolved;
    } else {
        return report_contract_failure(
            state:               state,
            operation:           "resolve_readback",
            public_fault:        gpu::INVALID_HANDLE,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "ticket",
            invariant:           "ticket must name live readback state",
            backend_text:        null,
        )~;
    }
    if (dest.len < data.span.size) {
        return report_contract_failure(
            state:               state,
            operation:           "resolve_readback",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "dest",
            invariant:           "destination must cover the complete ticket span",
            backend_text:        null,
        )~;
    }
    ulong current = 0;
    check_backend_result(
        state:     state,
        operation: "resolve_readback",
        result:    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &current),
    )!;
    if (current < (ulong)data.value) {
        return report_contract_failure(
            state:               state,
            operation:           "resolve_readback",
            public_fault:        gpu::READBACK_NOT_READY,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "ticket",
            invariant:           "ticket timeline value must be signaled before resolve",
            backend_text:        null,
        )~;
    }

    vk_invalidate_buffer(device, data.span.buffer, data.span.offset, data.span.size)!;
    mem::copy(dest.ptr, data.span.cpu, (sz)data.span.size);
    release_readback_state(state, data);
    state.readback_tickets.free(*ticket)!;
    *ticket = {};
}
