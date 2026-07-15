module gpu::vk @private;

import gpu @public;
import vk;
import vma;
import std::thread;
import std::io;

// Concrete (non-generic) slot table for semaphores. Monomorphized twin of
// gpu::slots to sidestep the c3c 0.8.0 generic-instantiation codegen crash.
const uint SEMAPHORE_FREE_NONE = 0;

struct SemaphoreSlot {
    vk::Semaphore semaphore;
}

struct SemaphoreCell {
    SemaphoreSlot value;
    uint          generation;
    uint          next_free;
    bool          used;
}

struct SemaphoreTable {
    SemaphoreCell[] slots;
    ulong           owner;
    uint            free_head;
    uint            next_new;
}

fn gpu::SemaphoreHandle? SemaphoreTable.alloc(&self, SemaphoreSlot value) {
    uint idx;
    bool found = false;
    while (self.free_head != SEMAPHORE_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
        if (can_allocate_handle_generation(self.slots[idx].generation)) {
            found = true;
            break;
        }
        self.slots[idx].next_free = SEMAPHORE_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].used = true;
    return (gpu::SemaphoreHandle){
        .owner      = self.owner,
        .index      = idx,
        .generation = self.slots[idx].generation,
    };
}

fn SemaphoreSlot*? SemaphoreTable.get(&self, gpu::SemaphoreHandle h) {
    if (h.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? SemaphoreTable.free(&self, gpu::SemaphoreHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    if (!should_recycle_handle_generation(self.slots[idx].generation)) {
        self.slots[idx].generation = HANDLE_GENERATION_MAX;
        self.slots[idx].next_free = SEMAPHORE_FREE_NONE;
        return;
    }
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

fn void? vk_cmd_buffer_barrier(gpu::CommandList* commands, gpu::BufferBarrier* barrier) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    BufferSlot* slot = state.buffers.get(barrier.buffer)!;
    // size 0 is the whole-buffer convention, matching flush/invalidate/fill.
    if (!gpu::range_fits(slot.size, barrier.offset, barrier.size)) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_buffer_barrier",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               barrier.buffer.index,
            generation:          barrier.buffer.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "BufferBarrier.size",
            invariant:           "offset and size must fit the buffer",
            backend_text:        "buffer barrier range exceeds the buffer",
        )~;
    }
    if (barrier.size == 0 && barrier.offset >= slot.size) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_buffer_barrier",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.BUFFER,
            has_public_identity: true,
            index:               barrier.buffer.index,
            generation:          barrier.buffer.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "BufferBarrier.offset",
            invariant:           "whole-buffer barrier offset must lie inside the buffer",
            backend_text:        "whole-buffer barrier offset lies outside the buffer",
        )~;
    }
    vk::DeviceSize barrier_size = barrier.size == 0 ? (vk::DeviceSize)vk::WHOLE_SIZE : (vk::DeviceSize)barrier.size;

    vk::BufferMemoryBarrier2 bar = vk::buffer_memory_barrier2()
        .set_src_stage_mask(stage_to_vk(barrier.before_stage))
        .set_src_access_mask(hazard_to_vk_access(barrier.before_hazard))
        .set_dst_stage_mask(stage_to_vk(barrier.after_stage))
        .set_dst_access_mask(hazard_to_vk_access(barrier.after_hazard))
        .set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_buffer(slot.buffer)
        .set_offset((vk::DeviceSize)barrier.offset)
        .set_size(barrier_size);

    vk::DependencyInfo dep = vk::dependency_info().set_buffer_memory_barriers((&bar)[:1]);
    vk::cmd_pipeline_barrier2(record.command_buffer, &dep);
}

fn void? vk_cmd_global_barrier(gpu::CommandList* commands, gpu::GlobalBarrier* barrier) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    vk::MemoryBarrier2 bar = vk::memory_barrier2()
        .set_src_stage_mask(stage_to_vk(barrier.before_stage))
        .set_src_access_mask(hazard_to_vk_access(barrier.before_hazard))
        .set_dst_stage_mask(stage_to_vk(barrier.after_stage))
        .set_dst_access_mask(hazard_to_vk_access(barrier.after_hazard));
    vk::DependencyInfo dep = vk::dependency_info().set_memory_barriers((&bar)[:1]);
    vk::cmd_pipeline_barrier2(record.command_buffer, &dep);
}

<* Record-time layout for `tex` on `list`: its own pending transition if it
   has recorded one, else the committed tracked layout. *>
fn gpu::TextureLayout effective_layout(CommandRecord* record, TextureSlot* slot, gpu::TextureHandle tex) @private {
    for (uint i = 0; i < record.pending_layout_count; i++) {
        if (record.pending_layouts[i].texture == tex) return record.pending_layouts[i].latest;
    }
    return texture_layout(slot);
}

<* Record or update one texture's pending transition, growing as needed. *>
fn void record_pending_layout(VkDeviceState* state, CommandRecord* record, gpu::TextureHandle tex, gpu::TextureLayout old_layout, gpu::TextureLayout new_layout) @private {
    for (uint i = 0; i < record.pending_layout_count; i++) {
        if (record.pending_layouts[i].texture == tex) {
            record.pending_layouts[i].latest = new_layout;
            return;
        }
    }
    ensure_pending_layout_capacity(state, record);
    record.pending_layouts[record.pending_layout_count] = {
        .texture = tex,
        .first_old = old_layout,
        .latest = new_layout,
    };
    record.pending_layout_count++;
}

<* Replay a submitted list's pending transitions onto tracked state, in
   record order; called under the queue mutex once queue_submit2 succeeds.
   Best-effort drift check: a mismatch against the layout seen at record time
   means a concurrent recorder raced this texture; only validation logs it,
   matching today's relaxed-atomic looseness. *>
fn void commit_pending_layouts(VkDeviceState* state, CommandRecord* record) @private {
    for (uint i = 0; i < record.pending_layout_count; i++) {
        PendingLayout* entry = &record.pending_layouts[i];
        if (try slot = state.textures.get(entry.texture)) {
            if (state.validation && texture_layout(slot) != entry.first_old) {
                io::eprintfn(
                    "gpu: texture layout drift at commit (texture index %d): expected %d, observed %d",
                    entry.texture.index, (int)entry.first_old, (int)texture_layout(slot),
                );
            }
            set_texture_layout(slot, entry.latest);
        }
    }
    record.pending_layout_count = 0;
}

<* Whole-image barrier; a barrier whose old_layout disagrees with the
   effective layout (this list's own pending transition, else the tracked
   layout) faults INVALID_RESOURCE_STATE. The transition is recorded as
   pending, not committed until the list submits. *>
fn void? vk_cmd_texture_barrier(gpu::CommandList* commands, gpu::TextureBarrier* barrier) @private {
    CommandRecord* record = resolve_command(commands)!;
    if (record.state != CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    TextureSlot* slot = state.textures.get(barrier.texture)!;
    if (barrier.old_layout != effective_layout(record, slot, barrier.texture)) {
        return report_contract_failure(
            state:               state,
            operation:           "cmd_texture_barrier",
            public_fault:        gpu::INVALID_RESOURCE_STATE,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               barrier.texture.index,
            generation:          barrier.texture.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      "TextureBarrier.old_layout",
            invariant:           "old_layout must match the command list effective layout",
            backend_text:        "texture barrier old layout disagrees with tracked layout",
        )~;
    }
    // Track the transition before emitting its GPU-visible barrier so the
    // backend record always describes everything in the command buffer.
    record_pending_layout(
        state:      state,
        record:     record,
        tex:        barrier.texture,
        old_layout: barrier.old_layout,
        new_layout: barrier.new_layout,
    );

    vk::ImageMemoryBarrier2 bar = vk::image_memory_barrier2()
        .set_src_stage_mask(stage_to_vk(barrier.before_stage))
        .set_src_access_mask(hazard_to_vk_access(barrier.before_hazard))
        .set_dst_stage_mask(stage_to_vk(barrier.after_stage))
        .set_dst_access_mask(hazard_to_vk_access(barrier.after_hazard))
        .set_old_layout(layout_to_vk(barrier.old_layout))
        .set_new_layout(layout_to_vk(barrier.new_layout))
        .set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_image(slot.image)
        .set_subresource_range({
            .aspect_mask      = format_aspect(slot.format),
            .base_mip_level   = 0,
            .level_count      = slot.mip_levels,
            .base_array_layer = 0,
            .layer_count      = slot.array_layers,
        });

    vk::DependencyInfo dep = vk::dependency_info().set_image_memory_barriers((&bar)[:1]);
    vk::cmd_pipeline_barrier2(record.command_buffer, &dep);
}

fn gpu::SemaphoreHandle? vk_create_semaphore(gpu::Device* device, gpu::SemaphoreDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    vk::SemaphoreTypeCreateInfo type_info = vk::semaphore_type_create_info()
        .set_semaphore_type(vk::SemaphoreType.TIMELINE)
        .set_initial_value((ulong)desc.initial_value);
    vk::SemaphoreCreateInfo info = vk::semaphore_create_info().set_next(&type_info);
    vk::Semaphore semaphore;
    check_device_result(state, vk::create_semaphore(
        device:        state.device,
        p_create_info: &info,
        p_allocator:   null,
        p_semaphore:   &semaphore,
    ))!;
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.SEMAPHORE,
        object_handle: (ulong)semaphore,
        name:          desc.debug_name,
    );

    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        if (try handle = state.semaphores.alloc({ .semaphore = semaphore })) {
            return handle;
        }
    }
    vk::destroy_semaphore(state.device, semaphore, null);
    return gpu::SLOT_TABLE_FULL~;
}

<* Frees the handle now; the vk semaphore release is deferred. The public
   contract already forbids pending waits at destroy. *>
fn void? vk_destroy_semaphore(gpu::Device* device, gpu::SemaphoreHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    SemaphoreSlot* slot = state.semaphores.get(handle)!;
    vk::Semaphore semaphore = slot.semaphore;
    ulong retire = retire_timeline_value(state);
    state.semaphores.free(handle)!;
    enqueue_deferred(
        state:          state,
        kind:           DeferredKind.SEMAPHORE,
        timeline_value: retire,
        object:         (void*)semaphore,
        allocation:     (vma::Allocation)null,
    );
}

fn void? vk_wait_semaphore(gpu::Device* device, gpu::SemaphoreHandle handle, gpu::SemaphoreValue value, ulong timeout_ns) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    SemaphoreSlot* slot = state.semaphores.get(handle)!;
    ulong wait_value = (ulong)value;
    vk::SemaphoreWaitInfo wait_info = vk::semaphore_wait_info()
        .set_semaphore_count(1)
        .set_semaphores((&slot.semaphore)[:1])
        .set_values((&wait_value)[:1]);
    check_wait_backend_result(
        state:     state,
        operation: "wait_semaphore",
        result:    vk::wait_semaphores(state.device, &wait_info, timeout_ns),
    )!;
}
