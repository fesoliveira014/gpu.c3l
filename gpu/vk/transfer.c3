module gpu::vk;

import gpu;
import vk;
import std::io;
import std::thread;
import std::core::mem;
import std::atomic;
import std::atomic::types;

const uint TRANSFER_RETIRE_CAP = 1024;
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

<* The buffer fns used here only dereference backend_state, so a shim Device
   works without the public Device value (which lives by-value with the
   caller and has no stable address). *>
fn gpu::Device device_shim(VkDeviceState* state) @private {
    return { .backend_state = (void*)state };
}

fn gpu::BufferDesc staging_arena_desc(usz size) @private {
    return {
        .size        = size,
        .usage       = { .transfer_src },
        .memory_kind = gpu::MemoryKind.STAGING,
        .debug_name  = "staging_arena",
    };
}

fn gpu::BufferDesc readback_arena_desc(usz size) @private {
    return {
        .size        = size,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::MemoryKind.READBACK,
        .debug_name  = "readback_arena",
    };
}

fn gpu::BufferDesc staging_dedicated_desc(usz size) @private {
    return {
        .size        = size,
        .usage       = { .transfer_src },
        .memory_kind = gpu::MemoryKind.STAGING,
        .debug_name  = "staging_dedicated",
    };
}

fn void? create_transfer_arenas(VkDeviceState* state, gpu::DeviceDesc* desc) {
    gpu::Device shim = device_shim(state);
    usz staging_size = desc.staging_arena_size == 0 ? gpu::DEFAULT_STAGING_ARENA_SIZE : desc.staging_arena_size;
    usz readback_size = desc.readback_arena_size == 0 ? gpu::DEFAULT_READBACK_ARENA_SIZE : desc.readback_arena_size;

    gpu::BufferDesc staging_desc = staging_arena_desc(staging_size);
    uint[MAX_QUEUE_FAMILIES] family_storage;
    uint family_count = used_queue_families(state, family_storage[..]);
    state.staging_arena.backing = vk_create_buffer_with_families(&shim, &staging_desc, family_storage[:family_count])!;
    mark_buffer_internal(state, state.staging_arena.backing);
    state.staging_arena.span = vk_get_buffer_span(&shim, state.staging_arena.backing)!;
    state.staging_arena.size = staging_size;

    gpu::BufferDesc readback_desc = readback_arena_desc(readback_size);
    state.readback_arena.backing = vk_create_buffer(&shim, &readback_desc)!;
    mark_buffer_internal(state, state.readback_arena.backing);
    state.readback_arena.span = vk_get_buffer_span(&shim, state.readback_arena.backing)!;
    state.readback_arena.size = readback_size;
}

<* A ring can mix tags (mid-frame helpers interleave with recordings) but
   stays strict FIFO: the head retires before anything behind it. *>
fn void drain_transfer_retire(VkDeviceState* state, TransferArenaState* arena) @private {
    if (arena.retire_count == 0) return;
    ulong frame_current = 0;
    ulong helper_current = 0;
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &frame_current);
    vk::get_semaphore_counter_value(state.device, state.helper_timeline, &helper_current);
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
    if (!gpu::is_pow2(align)) return gpu::INVALID_ARGUMENT~;
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
    drain_transfer_retire(state, arena);
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

fn void drain_dedicated_transfers(VkDeviceState* state) @private {
    if (state.dedicated_staging.count == 0) return;
    gpu::Device shim = device_shim(state);
    ulong frame_current = 0;
    ulong helper_current = 0;
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &frame_current);
    vk::get_semaphore_counter_value(state.device, state.helper_timeline, &helper_current);
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
    if (size == 0 || !gpu::is_pow2(align)) return gpu::INVALID_ARGUMENT~;
    {
        state.transfer_mutex.lock()!!;
        defer state.transfer_mutex.unlock();
        drain_dedicated_transfers(state);
        if (try span = transfer_arena_alloc(
            state: state,
            arena: arena,
            size:  size,
            align: align,
            tag:   tag,
        )) return span;
        if (state.dedicated_staging.count == DEDICATED_RETIRE_CAP) return gpu::ARENA_FULL~;
    }

    gpu::Device shim = device_shim(state);
    gpu::BufferHandle buffer = vk_create_buffer(&shim, dedicated_desc)!;

    state.transfer_mutex.lock()!!;
    drain_dedicated_transfers(state);
    if (state.dedicated_staging.count == DEDICATED_RETIRE_CAP) {
        state.transfer_mutex.unlock();
        (void)vk_destroy_buffer(&shim, buffer);
        return gpu::ARENA_FULL~;
    }
    mark_buffer_internal(state, buffer);
    state.dedicated_staging.entries[(state.dedicated_staging.head + state.dedicated_staging.count) % DEDICATED_RETIRE_CAP] = {
        .buffer         = buffer,
        .timeline_value = tag.value,
        .timeline_kind  = tag.kind,
    };
    state.dedicated_staging.count++;
    state.dedicated_fallbacks++;
    state.transfer_mutex.unlock();
    return vk_get_buffer_span(&shim, buffer);
}

fn gpu::GpuSpan? staging_alloc(VkDeviceState* state, usz size, usz align, TimelineTag tag) @private {
    gpu::BufferDesc desc = staging_dedicated_desc(size);
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

<* Byte size of a texture-upload region with the copy-desc 0-defaults
   resolved. The copy command re-validates independently. *>
fn usz? texture_upload_bytes(VkDeviceState* state, gpu::TextureUploadDesc* desc, ZString operation) @private {
    TextureSlot* slot = state.textures.get(desc.texture)!;
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
            rejected_field:      "TextureUploadDesc.mip",
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

<* The caller supplies the timeline tag its staging range retires against. *>
fn void? cmd_upload_buffer_impl(gpu::CommandList* commands, gpu::BufferHandle dst, usz offset, char[] data, TimelineTag tag, ZString operation) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
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
    BufferSlot* dst_slot = state.buffers.get(dst)!;
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

    gpu::GpuSpan staging = staging_alloc(
        state: state,
        size:  data.len,
        align: TRANSFER_ALIGN,
        tag:   tag,
    )!;
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

fn void? vk_cmd_upload_buffer(gpu::CommandList* commands, gpu::BufferHandle dst, usz offset, char[] data) @private {
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
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

fn void? cmd_upload_texture_impl(gpu::CommandList* commands, gpu::TextureUploadDesc* desc, char[] data, TimelineTag tag, ZString operation) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    TextureSlot* slot = state.textures.get(desc.texture)!;
    usz expected = texture_upload_bytes(state, desc, operation)!;
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
    usz align = format_texel_size(slot.format);
    if (align < 4) align = 4;
    gpu::GpuSpan staging = staging_alloc(
        state: state,
        size:  data.len,
        align: align,
        tag:   tag,
    )!;
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

fn void? vk_cmd_upload_texture(gpu::CommandList* commands, gpu::TextureUploadDesc* desc, char[] data) @private {
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    TimelineTag tag = { .kind = TimelineKind.FRAME, .value = state.timeline_counter.load(AtomicOrdering.RELAXED) + 1 };
    return cmd_upload_texture_impl(
        commands: commands,
        desc:     desc,
        data:     data,
        tag:      tag,
        operation: "cmd_upload_texture",
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

<* Wait for `v - 1`, then signal `v`: signals land in reservation order.
   Mutex-free — a mutex would deadlock the predecessor out of its own
   signal. Callers defer this on every exit path once `v` is reserved. *>
fn void? signal_helper_timeline(VkDeviceState* state, ulong v) @private {
    ulong prior = v - 1;
    ulong current = 0;
    vk::get_semaphore_counter_value(state.device, state.helper_timeline, &current);
    while (current < prior) {
        vk::SemaphoreWaitInfo wait_info = vk::semaphore_wait_info()
            .set_semaphore_count(1)
            .set_semaphores((&state.helper_timeline)[:1])
            .set_values((&prior)[:1]);
        if (vk::wait_semaphores(state.device, &wait_info, HELPER_TURNSTILE_ROUND_NS) == vk::Result.SUCCESS) break;
        io::eprintfn("gpu: helper turnstile %d still waiting on %d", v, prior);
        vk::get_semaphore_counter_value(state.device, state.helper_timeline, &current);
    }
    vk::SemaphoreSignalInfo signal_info = vk::semaphore_signal_info()
        .set_semaphore(state.helper_timeline)
        .set_value(v);
    return check(vk::signal_semaphore(state.device, &signal_info));
}

<* Call under helper_record_mutex before this call reserves: every reserved
   value signaled means no helper buffer is live anywhere (helpers end
   recording before releasing the mutex and signal after queue idle). *>
fn void? maybe_reset_helper_pool(VkDeviceState* state) @private {
    RecordingContextState* context = state.recording_contexts.get(state.helper_context)!;
    PoolDirtyMask* dirty = &context.pool_dirty[state.frame_index];
    if (!dirty.graphics) return;
    state.transfer_mutex.lock()!!;
    ulong reserved = state.helper_counter;
    state.transfer_mutex.unlock();
    ulong current = 0;
    vk::get_semaphore_counter_value(state.device, state.helper_timeline, &current);
    if (current != reserved) return;
    release_commands_for_context_frame(state, state.helper_context, state.frame_index);
    check(vk::reset_command_pool(state.device, context.graphics_pools[state.frame_index], (vk::CommandPoolResetFlags)0))!;
    dirty.graphics = false;
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
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    ulong helper_value = 0;
    defer if (helper_value != 0) (void)signal_helper_timeline(state, helper_value);
    TimelineTag tag = { .kind = TimelineKind.HELPER };

    gpu::CommandList commands;
    {
        state.helper_record_mutex.lock()!!;
        defer state.helper_record_mutex.unlock();
        maybe_reset_helper_pool(state)!;
        helper_value = reserve_helper_value(state);
        tag.value = helper_value;
        vk_begin_commands(
            device:       device,
            queue:        gpu::QueueKind.GRAPHICS,
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
    return submit_and_wait(device, &commands);
}

fn void? vk_upload_texture_data(gpu::Device* device, gpu::TextureHandle dst, uint mip, char[] data, gpu::Stage next_stage, gpu::Hazard next_hazard, gpu::TextureLayout next_layout) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    ulong helper_value = 0;
    defer if (helper_value != 0) (void)signal_helper_timeline(state, helper_value);
    TimelineTag tag = { .kind = TimelineKind.HELPER };

    TextureSlot* slot = state.textures.get(dst)!;
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
    usz expected = texture_upload_bytes(state, &probe, "upload_texture_data")!;
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

    gpu::CommandList commands;
    {
        state.helper_record_mutex.lock()!!;
        defer state.helper_record_mutex.unlock();
        maybe_reset_helper_pool(state)!;
        helper_value = reserve_helper_value(state);
        tag.value = helper_value;
        vk_begin_commands(
            device:       device,
            queue:        gpu::QueueKind.GRAPHICS,
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
            commands: &commands,
            desc:     &upload,
            data:     data,
            tag:      tag,
            operation: "upload_texture_data",
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
    return submit_and_wait(device, &commands);
}

fn void? vk_readback_buffer_data(gpu::Device* device, gpu::BufferHandle src, usz offset, char[] out_data, gpu::Stage from_stage, gpu::Hazard from_hazard) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    ulong helper_value = 0;
    defer if (helper_value != 0) (void)signal_helper_timeline(state, helper_value);
    TimelineTag tag = { .kind = TimelineKind.HELPER };

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
    BufferSlot* src_slot = state.buffers.get(src)!;
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

    gpu::CommandList commands;
    gpu::GpuSpan readback;
    {
        state.helper_record_mutex.lock()!!;
        defer state.helper_record_mutex.unlock();
        maybe_reset_helper_pool(state)!;
        helper_value = reserve_helper_value(state);
        tag.value = helper_value;
        readback = readback_alloc(
            state: state,
            size:  out_data.len,
            align: TRANSFER_ALIGN,
            tag:   tag,
        )!;
        vk_begin_commands(
            device:       device,
            queue:        gpu::QueueKind.GRAPHICS,
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

    submit_and_wait(device, &commands)!;
    vk_invalidate_buffer(device, readback.buffer, readback.offset, readback.size)!;
    mem::copy(out_data.ptr, readback.cpu, out_data.len);
}

fn void? vk_readback_texture_data(gpu::Device* device, gpu::TextureHandle src, uint mip, char[] out_data, gpu::Stage from_stage, gpu::Hazard from_hazard, gpu::TextureLayout from_layout) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    ulong helper_value = 0;
    defer if (helper_value != 0) (void)signal_helper_timeline(state, helper_value);
    TimelineTag tag = { .kind = TimelineKind.HELPER };

    gpu::TextureUploadDesc region = { .texture = src, .mip = mip };
    TextureSlot* slot = state.textures.get(src)!;
    usz expected = texture_upload_bytes(state, &region, "readback_texture_data")!;
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
    usz align = format_texel_size(slot.format);
    if (align < 4) align = 4;
    gpu::CommandList commands;
    gpu::GpuSpan readback;
    {
        state.helper_record_mutex.lock()!!;
        defer state.helper_record_mutex.unlock();
        maybe_reset_helper_pool(state)!;
        helper_value = reserve_helper_value(state);
        tag.value = helper_value;
        readback = readback_alloc(
            state: state,
            size:  out_data.len,
            align: align,
            tag:   tag,
        )!;
        vk_begin_commands(
            device:       device,
            queue:        gpu::QueueKind.GRAPHICS,
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

    submit_and_wait(device, &commands)!;
    vk_invalidate_buffer(device, readback.buffer, readback.offset, readback.size)!;
    mem::copy(out_data.ptr, readback.cpu, out_data.len);
}

<* Submits an already-ended command list, blocks until queue idle. Touches
   no timeline — the caller owns its tag's signaling. Runs after the caller
   releases helper_record_mutex, so a successor records while this waits. *>
fn void? submit_and_wait(gpu::Device* device, gpu::CommandList* commands) @private {
    gpu::CommandList[1] lists = { *commands };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    vk_submit(device, &submit)!;
    return vk_wait_queue_idle(device, gpu::QueueKind.GRAPHICS);
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
        if (try alloc = ticket_arena_alloc(
            state: state,
            arena: &state.readback_arena,
            size:  size,
            tag:   tag,
        )) {
            return {
                .span      = alloc.span,
                .value     = (gpu::SemaphoreValue)tag.value,
                .range_end = alloc.range_end,
            };
        }
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
        state.transfer_mutex.lock()!!;
        mark_buffer_internal(state, buffer);
        state.dedicated_fallbacks++;
        state.transfer_mutex.unlock();
        return {
            .span      = span,
            .value     = (gpu::SemaphoreValue)tag.value,
            .dedicated = buffer,
        };
    }
    (void)vk_destroy_buffer(&shim, buffer);
    return gpu::ARENA_FULL~;
}

fn void release_ticket(VkDeviceState* state, gpu::ReadbackTicket* ticket) @private {
    state.transfer_mutex.lock()!!;
    defer state.transfer_mutex.unlock();
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
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
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
    BufferSlot* src_slot = state.buffers.get(src)!;
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
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    gpu::TextureUploadDesc region = { .texture = src, .mip = mip };
    usz size = texture_upload_bytes(state, &region, "cmd_readback_texture")!;
    TextureSlot* slot = state.textures.get(src)!;
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
    enter_tracked(state);
    defer exit_tracked(state);
    if ((ulong)ticket.value == 0) {
        return report_contract_failure(
            state:               state,
            operation:           "resolve_readback",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "ticket.value",
            invariant:           "ticket must be live and unresolved",
            backend_text:        null,
        )~;
    }
    if (dest.len < ticket.span.size) {
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
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &current);
    if (current < (ulong)ticket.value) {
        return report_contract_failure(
            state:               state,
            operation:           "resolve_readback",
            public_fault:        gpu::READBACK_NOT_READY,
            resource_kind:       gpu::DebugResourceKind.NONE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "ticket.value",
            invariant:           "ticket timeline value must be signaled before resolve",
            backend_text:        null,
        )~;
    }

    vk_invalidate_buffer(device, ticket.span.buffer, ticket.span.offset, ticket.span.size)!;
    mem::copy(dest.ptr, ticket.span.cpu, (sz)ticket.span.size);
    release_ticket(state, ticket);
    *ticket = {};
}
