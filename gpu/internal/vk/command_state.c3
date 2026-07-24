module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::mem;
import std::core::env;
import std::hash::fnv64a;

const uint MAX_COMMAND_LISTS = gpu::internal::MAX_DEVICE_COMMANDS;
const uint COMMAND_FREE_NONE = 0;

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
    GraphicsColorProfile              color_profile;
    RenderCompatibility               render;
}

struct NativePipelineBinding {
    vk::Pipeline                  pipeline;
    vk::PipelineLayout            layout;
    vk::IndirectCommandsLayoutEXT generated_dispatch_layout;
    uint                          cache_entry;
    PipelineKind                  kind;
    GraphicsColorProfile          color_profile;
    RenderCompatibility            render;
}

struct StrictHeapCommandBinding {
    bool descriptor_buffer_bound;
    bool graphics_offsets_set;
    bool compute_offsets_set;
    bool graphics_set_bound;
    bool compute_set_bound;
}

struct VkCommandRecord {
    gpu::internal::CommandRecord* root;
    vk::CommandBuffer            command_buffer;
    gpu::CommandAllocatorHandle  allocator;
    uint                         allocator_buffer;
    gpu::Queue                   queue;
    ulong                        submit_visit_epoch;
    gpu::CompletionPoint         submitted_completion;
    gpu::internal::CommandRecord* next_submitted;
    RenderCompatibility          active_render;
    bool                         graphics_state_initialized;
    BoundPipeline                logical_pipeline;
    NativePipelineBinding        native_graphics_pipeline;
    NativePipelineBinding        native_compute_pipeline;
    StrictHeapCommandBinding     strict_heap_binding;
    CommandScratch*              scratch;
}

struct CommandCell {
    gpu::internal::CommandRecord record;
    VkCommandRecord              value;
    uint                         generation;
    uint                         next_free;
    bool                         used;
}

struct CommandTable {
    CommandCell[] slots;
    ulong         owner;
    uint          free_head;
    uint          next_new;
}

struct CommandReferenceProbe {
    uint cell_index;
    uint reference_index;
    bool found;
}

fn gpu::internal::CommandRecord*? CommandTable.claim(
    &self,
    gpu::Device device,
    gpu::internal::CommandOps* ops,
    void* backend_state,
    VkCommandRecord value,
) {
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
    gpu::CommandListHandle identity = {
        .owner = self.owner,
        .index = idx,
        .generation = self.slots[idx].generation,
    };
    self.slots[idx].record = {
        .ops             = ops,
        .backend_state   = backend_state,
        .backend_command = &self.slots[idx].value,
        .device          = device,
        .identity        = identity,
        .state           = gpu::internal::CommandState.INACTIVE,
    };
    self.slots[idx].value.root = &self.slots[idx].record;
    return &self.slots[idx].record;
}

fn gpu::internal::CommandRecord*? CommandTable.get_record(
    &self,
    gpu::CommandListHandle handle,
) {
    gpu::internal::note_command_table_lookup();
    if (handle.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = handle.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != handle.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].record;
}

fn VkCommandRecord*? CommandTable.get(
    &self,
    gpu::CommandListHandle handle,
) {
    gpu::internal::CommandRecord* record = self.get_record(handle)!;
    return (VkCommandRecord*)record.backend_command;
}

fn void? CommandTable.free(&self, gpu::CommandListHandle handle) {
    self.get(handle)!;
    uint idx = handle.index;
    self.slots[idx].record = {};
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

fn void? CommandTable.cancel_claim(
    &self,
    gpu::CommandListHandle handle,
) {
    return self.free(handle);
}

fn gpu::internal::CommandRecord*? vk_resolve_command(
    void* backend_state,
    gpu::CommandListHandle handle,
) @private {
    if (backend_state == null) return gpu::INVALID_HANDLE~;
    VkDeviceState* state = (VkDeviceState*)backend_state;
    return state.commands.get_record(handle);
}

fn VkCommandRecord*? resolve_command(
    gpu::CommandList* token,
) @if(env::TESTING) {
    gpu::internal::CommandRecord* record =
        gpu::internal::recording_command(
            token,
            count_command: false,
        )!;
    return backend_command(record);
}

fn VkDeviceState* command_device_state(
    gpu::internal::CommandRecord* record,
) {
    return (VkDeviceState*)record.backend_state;
}

fn VkCommandRecord* backend_command(
    gpu::internal::CommandRecord* record,
) {
    return (VkCommandRecord*)record.backend_command;
}

fn void? ensure_command_reference_capacity(
    VkCommandRecord* record,
    uint additional,
) {
    if (additional
            <= record.scratch.references.len
                - record.scratch.reference_count) return;
    return gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED~;
}

fn ulong command_reference_hash(CommandResourceReference* reference) {
    CommandReferenceKey key = {
        .owner      = reference.owner,
        .index      = reference.index,
        .generation = reference.generation,
    };
    return fnv64a::hash(((char*)&key)[:CommandReferenceKey::size]);
}

fn bool indexed_resource_reference_matches(
    VkDeviceState* state,
    CommandReferenceIndexCell* cell,
    CommandResourceReference* reference,
) {
    note_command_reference_equality_check(state);
    return cell.owner == reference.owner
        && cell.index == reference.index
        && cell.generation == reference.generation;
}

fn CommandReferenceProbe? probe_command_reference(
    VkDeviceState* state,
    CommandScratch* scratch,
    CommandResourceReference* reference,
) {
    CommandReferenceIndex* reference_index = &scratch.reference_index;
    if (reference_index.cells.len == 0) {
        return gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED~;
    }
    if (reference_index.epoch == 0) reference_index.epoch = 1;
    note_command_reference_lookup(state);
    uint cell_count = (uint)reference_index.cells.len;
    ulong hash = command_reference_hash(reference);
    uint cell_index = (uint)(
        (hash ^ (hash >> 32))
        & (ulong)(cell_count - 1)
    );
    for (uint probe = 0; probe < cell_count; probe++) {
        note_command_reference_probe(state);
        CommandReferenceIndexCell* cell = &reference_index.cells[cell_index];
        if (cell.epoch != reference_index.epoch) {
            return {
                .cell_index = cell_index,
                .found      = false,
            };
        }
        if (indexed_resource_reference_matches(state, cell, reference)) {
            return {
                .cell_index      = cell_index,
                .reference_index = cell.reference_index,
                .found           = true,
            };
        }
        cell_index = (cell_index + 1) & (cell_count - 1);
    }
    return gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED~;
}

fn void publish_command_reference(
    VkDeviceState* state,
    CommandScratch* scratch,
    CommandResourceReference* reference,
    CommandReferenceProbe* probe,
    uint reference_index,
) {
    scratch.reference_index.cells[probe.cell_index] = {
        .owner           = reference.owner,
        .index           = reference.index,
        .generation      = reference.generation,
        .reference_index = reference_index,
        .epoch           = scratch.reference_index.epoch,
    };
    note_command_reference_publication(state);
}

fn void reset_command_reference_index(
    VkDeviceState* state,
    CommandScratch* scratch,
) {
    CommandReferenceIndex* reference_index = &scratch.reference_index;
    if (reference_index.cells.len == 0) return;
    if (reference_index.epoch == uint::max) {
        foreach (&cell : reference_index.cells) cell.epoch = 0;
        note_command_reference_rollover_clear(state, reference_index.cells.len);
        reference_index.epoch = 1;
        return;
    }
    reference_index.epoch++;
    if (reference_index.epoch == 0) reference_index.epoch = 1;
}

fn void? preflight_command_references(
    VkDeviceState* state,
    VkCommandRecord* record,
    CommandResourceReference[] candidates,
) {
    uint additional = 0;
    foreach (i, &candidate : candidates) {
        CommandReferenceProbe probe = probe_command_reference(
            state,
            record.scratch,
            candidate,
        )!;
        bool duplicate = probe.found;
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
    VkCommandRecord* record,
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
    CommandReferenceProbe probe = probe_command_reference(
        state,
        scratch,
        &reference,
    )!;
    if (probe.found) {
        note_command_reference_duplicate(state);
        return;
    }
    ensure_command_reference_capacity(record, 1)!;
    state.resource_mutex.lock()!!;
    note_command_reference_mutex_acquisition(state);
    defer state.resource_mutex.unlock();
    retain_tracked_command_reference(state, reference)!;
    note_command_reference_increment(state);
    note_command_reference_retain(state);
    uint reference_index = scratch.reference_count++;
    scratch.references[reference_index] = reference;
    publish_command_reference(
        state:           state,
        scratch:         scratch,
        reference:       &reference,
        probe:           &probe,
        reference_index: reference_index,
    );
}

fn TextureSlot*? resolve_tracked_texture_command_reference(
    VkDeviceState* state,
    VkCommandRecord* record,
    gpu::TextureHandle handle,
) {
    note_command_tracking_call(state);
    CommandResourceReference reference = {
        .owner      = handle.owner,
        .index      = handle.index,
        .generation = handle.generation,
    };
    CommandScratch* scratch = record.scratch;
    CommandReferenceProbe probe = probe_command_reference(
        state,
        scratch,
        &reference,
    )!;
    if (probe.found) {
        note_command_reference_duplicate(state);
        return state.textures.get(handle);
    }

    ensure_command_reference_capacity(record, 1)!;
    state.resource_mutex.lock()!!;
    note_command_reference_mutex_acquisition(state);
    defer state.resource_mutex.unlock();
    TextureSlot* slot = state.textures.get(handle)!;
    reference.tracked_command_references = &slot.tracked_command_references;
    reference.tracked_command_references.add(1, AtomicOrdering.RELAXED);
    note_command_reference_increment(state);
    note_command_reference_retain(state);
    uint reference_index = scratch.reference_count++;
    scratch.references[reference_index] = reference;
    publish_command_reference(
        state:           state,
        scratch:         scratch,
        reference:       &reference,
        probe:           &probe,
        reference_index: reference_index,
    );
    return slot;
}

fn void clear_command_scratch_references(
    VkDeviceState* state,
    CommandScratch* scratch,
) {
    release_tracked_command_references(
        state,
        scratch.references,
        scratch.reference_count,
    );
    scratch.reference_count = 0;
    reset_command_reference_index(state, scratch);
}

fn void clear_command_references(
    VkDeviceState* state,
    VkCommandRecord* record,
) {
    clear_command_scratch_references(state, record.scratch);
}

fn void rollback_command_references(
    VkDeviceState* state,
    VkCommandRecord* record,
    uint checkpoint,
) {
    if (record.scratch.reference_count == checkpoint) return;
    while (record.scratch.reference_count > checkpoint) {
        record.scratch.reference_count--;
        release_tracked_command_reference(
            state,
            record.scratch.references[record.scratch.reference_count],
        );
    }
    reset_command_reference_index(state, record.scratch);
    for (uint i = 0; i < record.scratch.reference_count; i++) {
        CommandResourceReference* reference = &record.scratch.references[i];
        CommandReferenceProbe probe = probe_command_reference(
            state,
            record.scratch,
            reference,
        )!!;
        publish_command_reference(
            state:           state,
            scratch:         record.scratch,
            reference:       reference,
            probe:           &probe,
            reference_index: i,
        );
        note_command_reference_rollback_rebuild_publication(state);
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
