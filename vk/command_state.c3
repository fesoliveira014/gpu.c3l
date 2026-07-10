module gpu::vk;

import gpu;
import vk;
import std::core::mem;

const uint MAX_COMMAND_LISTS = 4096;
const uint COMMAND_GENERATION_MAX = 0xFF_FFFF;
const uint COMMAND_FREE_NONE = 0;
const usz PENDING_LAYOUT_INITIAL_CAPACITY = 16;

enum CommandState : int {
    RECORDING,
    RECORDING_RENDER_PASS,
    EXECUTABLE,
    SUBMITTING,
    SUBMITTED,
}

struct PendingLayout {
    gpu::TextureHandle texture;
    gpu::TextureLayout first_old;
    gpu::TextureLayout latest;
}

struct CommandRecord {
    vk::CommandBuffer            command_buffer;
    gpu::RecordingContextHandle  context;
    gpu::QueueKind               queue;
    uint                         frame_index;
    CommandState                 state;
    gpu::PipelineHandle          last_pipeline;
    uint                         last_entry;
    PendingLayout[]              pending_layouts;
    uint                         pending_layout_count;
}

struct CommandCell {
    CommandRecord value;
    uint          generation;
    uint          next_free;
    bool          used;
}

struct CommandTable {
    CommandCell[] slots;
    uint          free_head;
    uint          next_new;
}

fn uint next_command_generation(uint generation) {
    return generation >= COMMAND_GENERATION_MAX ? 1 : generation + 1;
}

fn gpu::CommandListHandle? CommandTable.alloc(&self, CommandRecord value) {
    uint idx;
    if (self.free_head != COMMAND_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
    } else if (self.next_new < self.slots.len) {
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    } else {
        return gpu::SLOT_TABLE_FULL~;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation = next_command_generation(self.slots[idx].generation);
    self.slots[idx].used = true;
    return (gpu::CommandListHandle){
        .index = idx,
        .generation = self.slots[idx].generation,
    };
}

fn CommandRecord*? CommandTable.get(&self, gpu::CommandListHandle handle) {
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
    self.slots[idx].generation = next_command_generation(self.slots[idx].generation);
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

fn CommandRecord*? resolve_command(gpu::CommandList* commands) {
    if (commands.device == null || !commands.handle.is_valid()) return gpu::INVALID_HANDLE~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    if (state == null) return gpu::INVALID_HANDLE~;
    return state.commands.get(commands.handle);
}

fn void free_pending_layouts(VkDeviceState* state, CommandRecord* record) {
    if (record.pending_layouts.ptr != null) {
        alloc::free(state.host_allocator, record.pending_layouts.ptr);
        record.pending_layouts = {};
    }
    record.pending_layout_count = 0;
}

fn void release_command(VkDeviceState* state, gpu::CommandListHandle handle) {
    if (try record = state.commands.get(handle)) {
        free_pending_layouts(state, record);
        (void)state.commands.free(handle);
    }
}

fn void release_commands_for_frame(VkDeviceState* state, uint frame_index) {
    state.command_mutex.lock()!!;
    defer state.command_mutex.unlock();
    for (uint i = 0; i < state.commands.next_new; i++) {
        CommandCell* cell = &state.commands.slots[i];
        if (!cell.used || cell.value.frame_index != frame_index) continue;
        gpu::CommandListHandle handle = {
            .index = i,
            .generation = cell.generation,
        };
        release_command(state, handle);
    }
}

fn void release_commands_for_context_frame(
    VkDeviceState* state,
    gpu::RecordingContextHandle context,
    uint frame_index,
) {
    state.command_mutex.lock()!!;
    defer state.command_mutex.unlock();
    for (uint i = 0; i < state.commands.next_new; i++) {
        CommandCell* cell = &state.commands.slots[i];
        if (!cell.used || cell.value.frame_index != frame_index) continue;
        if ((ulong)cell.value.context != (ulong)context) continue;
        gpu::CommandListHandle handle = {
            .index = i,
            .generation = cell.generation,
        };
        release_command(state, handle);
    }
}

fn bool context_has_live_commands(VkDeviceState* state, gpu::RecordingContextHandle context) {
    state.command_mutex.lock()!!;
    defer state.command_mutex.unlock();
    foreach (&cell : state.commands.slots[:state.commands.next_new]) {
        if (cell.used && (ulong)cell.value.context == (ulong)context) return true;
    }
    return false;
}

fn void destroy_command_table(VkDeviceState* state) {
    if (state.commands.slots.ptr == null) return;
    foreach (&cell : state.commands.slots[:state.commands.next_new]) {
        if (cell.used) free_pending_layouts(state, &cell.value);
    }
    mem::free(state.commands.slots.ptr);
    state.commands = {};
}

fn void ensure_pending_layout_capacity(VkDeviceState* state, CommandRecord* record) {
    if (record.pending_layout_count < record.pending_layouts.len) return;
    usz new_capacity = record.pending_layouts.len == 0
        ? PENDING_LAYOUT_INITIAL_CAPACITY
        : record.pending_layouts.len * 2;
    PendingLayout[] grown = alloc::new_array(
        state.host_allocator,
        PendingLayout,
        (sz)new_capacity,
    );
    for (uint i = 0; i < record.pending_layout_count; i++) {
        grown[i] = record.pending_layouts[i];
    }
    if (record.pending_layouts.ptr != null) {
        alloc::free(state.host_allocator, record.pending_layouts.ptr);
    }
    record.pending_layouts = grown;
}
