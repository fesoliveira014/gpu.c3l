module gpu::vk @private;

import gpu @public;
import vk;
import std::core::mem;

const uint MAX_COMMAND_LISTS = gpu::MAX_DEVICE_COMMANDS;
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
    PipelineCell*                     validation_cell;
    vk::Pipeline                      pipeline;
    vk::PipelineLayout                layout;
    vk::IndirectCommandsLayoutEXT     generated_dispatch_layout;
    uint                              expected_generation;
    uint                              cache_entry;
    PipelineKind                      kind;
    RenderCompatibility               render;
}

struct CommandRecord {
    vk::CommandBuffer           command_buffer;
    vk::CommandPool             command_pool;
    RecordingContextHandle      context;
    gpu::Queue                  queue;
    CommandState                state;
    uint                        active_render_width;
    uint                        active_render_height;
    RenderCompatibility         active_render;
    BoundPipeline               bound_pipeline;
    bool                        depth_state_set;
    CommandResourceReference[]  references;
    uint                        reference_count;
    GeneratedPreprocessBuffer[] generated_preprocess;
    uint                        generated_preprocess_count;
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
    gpu::note_command_table_lookup();
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
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(&commands.device);
    if (state == null) return gpu::INVALID_HANDLE~;
    return state.commands.get(commands.handle);
}

fn VkDeviceState* encoder_device_state(gpu::CommandEncoderState* encoder) {
    return (VkDeviceState*)encoder.backend_state;
}

fn CommandRecord* encoder_command(gpu::CommandEncoderState* encoder) {
    return (CommandRecord*)encoder.backend_command;
}

fn void ensure_command_reference_capacity(
    VkDeviceState* state,
    CommandRecord* record,
) {
    if (record.reference_count < record.references.len) return;
    usz capacity = record.references.len == 0
        ? COMMAND_REFERENCE_INITIAL_CAPACITY
        : record.references.len * 2;
    CommandResourceReference[] grown = alloc::new_array(
        state.host_allocator,
        CommandResourceReference,
        (sz)capacity,
    );
    note_recording_host_allocation(state);
    for (uint i = 0; i < record.reference_count; i++) {
        grown[i] = record.references[i];
    }
    if (record.references.ptr != null) {
        alloc::free(state.host_allocator, record.references.ptr);
    }
    record.references = grown;
}

fn void? track_command_reference(
    VkDeviceState* state,
    CommandRecord* record,
    ulong owner,
    uint index,
    uint generation,
) {
    if (!state.validation && owner != state.attachment_views.owner) return;
    CommandResourceReference reference = {
        .owner = owner, .index = index, .generation = generation,
    };
    for (uint i = 0; i < record.reference_count; i++) {
        if (same_resource_reference(&record.references[i], &reference)) return;
    }
    ensure_command_reference_capacity(state, record);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    retain_validation_reference(state, reference)!;
    record.references[record.reference_count++] = reference;
}

fn TextureSlot*? resolve_texture_command_reference(
    VkDeviceState* state,
    CommandRecord* record,
    gpu::TextureHandle handle,
) {
    if (!state.validation) return state.textures.get(handle);

    CommandResourceReference reference = {
        .owner      = handle.owner,
        .index      = handle.index,
        .generation = handle.generation,
    };
    for (uint i = 0; i < record.reference_count; i++) {
        if (same_resource_reference(&record.references[i], &reference)) {
            return state.textures.get(handle);
        }
    }

    ensure_command_reference_capacity(state, record);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    TextureSlot* slot = state.textures.get(handle)!;
    reference.validation_references = &slot.validation_references;
    reference.validation_references.add(1, AtomicOrdering.RELAXED);
    record.references[record.reference_count++] = reference;
    return slot;
}

fn void free_command_references(
    VkDeviceState* state,
    CommandRecord* record,
) {
    if (record.references.ptr == null) {
        record.reference_count = 0;
        return;
    }
    release_validation_references(
        state,
        record.references,
        record.reference_count,
    );
    alloc::free(state.host_allocator, record.references.ptr);
    record.references = {};
    record.reference_count = 0;
}

fn void rollback_command_references(
    VkDeviceState* state,
    CommandRecord* record,
    uint checkpoint,
) {
    while (record.reference_count > checkpoint) {
        record.reference_count--;
        release_validation_reference(
            state,
            record.references[record.reference_count],
        );
    }
}

fn void release_command(VkDeviceState* state, gpu::CommandListHandle handle) {
    if (try record = state.commands.get(handle)) {
        free_command_references(state, record);
        recycle_generated_preprocess_buffers_locked(
            state,
            record.generated_preprocess,
            record.generated_preprocess_count,
        );
        record.generated_preprocess = {};
        record.generated_preprocess_count = 0;
        (void)state.commands.free(handle);
    }
}

fn void destroy_command_table(VkDeviceState* state) {
    if (state.commands.slots.ptr == null) return;
    foreach (&cell : state.commands.slots[:state.commands.next_new]) {
        if (cell.used) {
            free_command_references(state, &cell.value);
            free_generated_preprocess_buffers(state, cell.value.generated_preprocess);
        }
    }
    mem::free(state.commands.slots.ptr);
    state.commands = {};
}
