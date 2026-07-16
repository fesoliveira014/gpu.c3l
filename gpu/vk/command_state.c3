module gpu::vk @private;

import gpu @public;
import vk;
import std::core::mem;

const uint MAX_COMMAND_LISTS = 4096;
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
    uint                         active_render_width;
    uint                         active_render_height;
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

fn bool frame_has_live_commands(VkDeviceState* state, uint frame_index) {
    state.command_mutex.lock()!!;
    defer state.command_mutex.unlock();
    foreach (&cell : state.commands.slots[:state.commands.next_new]) {
        if (cell.used && cell.value.frame_index == frame_index) return true;
    }
    return false;
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
        if (cell.value.context != context) continue;
        gpu::CommandListHandle handle = {
            .owner = state.commands.owner,
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
        if (cell.used && cell.value.context == context) return true;
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
