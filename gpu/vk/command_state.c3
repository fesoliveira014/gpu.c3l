module gpu::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::mem;

const uint MAX_COMMAND_LISTS = gpu::internal::MAX_DEVICE_COMMANDS;
const uint COMMAND_FREE_NONE = 0;

enum CommandState : int {
    RECORDING,
    RECORDING_RENDER_PASS,
    EXECUTABLE,
    SUBMITTING,
    SUBMITTED,
}

struct RenderCompatibility {
    uint                                    color_count;
    gpu::Format[gpu::MAX_COLOR_ATTACHMENTS] color_formats;
    gpu::Format                             depth_format;
    gpu::SampleCount                        sample_count;
}

struct BoundPipeline {
    gpu::PipelineHandle               handle;
    vk::Pipeline                      pipeline;
    vk::PipelineLayout                layout;
    vk::IndirectCommandsLayoutEXT     generated_dispatch_layout;
    uint                              cache_entry;
    PipelineKind                      kind;
    RenderCompatibility               render;
}

struct CommandRecord {
    vk::CommandBuffer            command_buffer;
    gpu::CommandAllocatorHandle  allocator;
    uint                         allocator_buffer;
    gpu::Queue                   queue;
    CommandState                 state;
    ulong                        submit_visit_epoch;
    uint                         active_render_width;
    uint                         active_render_height;
    RenderCompatibility          active_render;
    BoundPipeline                bound_pipeline;
    bool                         depth_state_set;
    CommandScratch*              scratch;
}

struct CommandCell {
    CommandRecord value;
    uint          generation;
    uint          next_free;
    bool          used;
}

struct CommandTable {
    CommandCell[] slots;
    ulong         owner;
    uint          free_head;
    uint          next_new;
}

fn gpu::CommandListHandle? CommandTable.alloc(&self, CommandRecord value) {
    uint idx;
    bool found = false;
    while (self.free_head != COMMAND_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
        if (can_allocate_handle_generation(self.slots[idx].generation)) {
            found = true;
            break;
        }
        self.slots[idx].next_free = COMMAND_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].used = true;
    return (gpu::CommandListHandle){
        .owner = self.owner,
        .index = idx,
        .generation = self.slots[idx].generation,
    };
}

fn CommandRecord*? CommandTable.get(&self, gpu::CommandListHandle handle) {
    gpu::internal::note_command_table_lookup();
    if (handle.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = handle.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != handle.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? CommandTable.free(&self, gpu::CommandListHandle handle) {
    self.get(handle)!;
    uint idx = handle.index;
    self.slots[idx].value = {};
    self.slots[idx].used = false;
    if (!should_recycle_handle_generation(self.slots[idx].generation)) {
        self.slots[idx].generation = HANDLE_GENERATION_MAX;
        self.slots[idx].next_free = COMMAND_FREE_NONE;
        return;
    }
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

fn CommandRecord*? resolve_command(gpu::CommandList* commands) {
    if (commands == null || !commands.handle.is_valid()) return gpu::INVALID_HANDLE~;
    VkDeviceState* state = (VkDeviceState*)gpu::internal::device_backend_state_ptr(&commands.device);
    if (state == null) return gpu::INVALID_HANDLE~;
    return state.commands.get(commands.handle);
}

fn VkDeviceState* encoder_device_state(gpu::internal::CommandEncoderState* encoder) {
    return (VkDeviceState*)encoder.backend_state;
}

fn CommandRecord* encoder_command(gpu::internal::CommandEncoderState* encoder) {
    return (CommandRecord*)encoder.backend_command;
}

fn void? ensure_command_reference_capacity(
    CommandRecord* record,
    uint additional,
) {
    if (additional
            <= record.scratch.references.len
                - record.scratch.reference_count) return;
    return gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED~;
}

fn void? preflight_command_references(
    CommandRecord* record,
    CommandResourceReference[] candidates,
) {
    uint additional = 0;
    foreach (i, &candidate : candidates) {
        bool duplicate = false;
        foreach (&existing : record.scratch.references[
            :record.scratch.reference_count
        ]) {
            if (!same_resource_reference(existing, candidate)) continue;
            duplicate = true;
            break;
        }
        if (duplicate) continue;
        for (uint j = 0; j < i; j++) {
            if (!same_resource_reference(&candidates[j], candidate)) continue;
            duplicate = true;
            break;
        }
        if (!duplicate) additional++;
    }
    ensure_command_reference_capacity(record, additional)!;
}

fn void? track_command_reference(
    VkDeviceState* state,
    CommandRecord* record,
    ulong owner,
    uint index,
    uint generation,
) {
    note_command_tracking_call(state);
    CommandResourceReference reference = {
        .owner      = owner,
        .index      = index,
        .generation = generation,
    };
    CommandScratch* scratch = record.scratch;
    for (uint i = 0; i < scratch.reference_count; i++) {
        if (same_resource_reference(&scratch.references[i], &reference)) return;
    }
    ensure_command_reference_capacity(record, 1)!;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    retain_tracked_command_reference(state, reference)!;
    note_command_reference_increment(state);
    scratch.references[scratch.reference_count++] = reference;
}

fn TextureSlot*? resolve_tracked_texture_command_reference(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::TextureHandle handle,
) {
    note_command_tracking_call(state);
    CommandResourceReference reference = {
        .owner      = handle.owner,
        .index      = handle.index,
        .generation = handle.generation,
    };
    CommandScratch* scratch = record.scratch;
    for (uint i = 0; i < scratch.reference_count; i++) {
        if (same_resource_reference(&scratch.references[i], &reference)) {
            return state.textures.get(handle);
        }
    }

    ensure_command_reference_capacity(record, 1)!;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    TextureSlot* slot = state.textures.get(handle)!;
    reference.tracked_command_references = &slot.tracked_command_references;
    reference.tracked_command_references.add(1, AtomicOrdering.RELAXED);
    note_command_reference_increment(state);
    scratch.references[scratch.reference_count++] = reference;
    return slot;
}

fn void clear_command_references(
    VkDeviceState* state,
    CommandRecord* record,
) {
    release_tracked_command_references(
        state,
        record.scratch.references,
        record.scratch.reference_count,
    );
    record.scratch.reference_count = 0;
}

fn void rollback_command_references(
    VkDeviceState* state,
    CommandRecord* record,
    uint checkpoint,
) {
    while (record.scratch.reference_count > checkpoint) {
        record.scratch.reference_count--;
        release_tracked_command_reference(
            state,
            record.scratch.references[record.scratch.reference_count],
        );
    }
}

fn void release_command(VkDeviceState* state, gpu::CommandListHandle handle) {
    if (try record = state.commands.get(handle)) {
        (void)state.commands.free(handle);
    }
}

fn void destroy_command_table(VkDeviceState* state) {
    if (state.commands.slots.ptr == null) return;
    foreach (&cell : state.commands.slots[:state.commands.next_new]) {
        if (cell.used) {
            clear_command_references(state, &cell.value);
        }
    }
    mem::free(state.commands.slots.ptr);
    state.commands = {};
}
