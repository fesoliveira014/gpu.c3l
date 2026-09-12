module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::mem;
import std::core::env;

const uint MAX_COMMAND_LISTS = gpu::internal::MAX_DEVICE_COMMANDS;
const uint COMMAND_FREE_NONE = 0;

struct BoundPipeline {
    gpu::PipelineHandle               handle;
    vk::Pipeline                      pipeline;
    vk::PipelineLayout                layout;
    vk::IndirectCommandsLayoutEXT     generated_dispatch_layout;
    uint                              cache_entry;
    PipelineKind                      kind;
    uint                              inline_root_offset;
}

struct NativePipelineBinding {
    vk::Pipeline pipeline;
    uint         cache_entry;
}

struct DescriptorHeapCommandBinding {
    bool graphics_set_bound;
    vk::PipelineLayout graphics_layout;
    bool compute_set_bound;
    bool ray_tracing_set_bound;
}

struct CommandRecord @public {
    VkDeviceState*                   device_state;
    gpu::Device                      device;
    gpu::internal::CommandListHandle identity;
    gpu::internal::CommandState      state;
    vk::CommandBuffer                command_buffer;
    gpu::CommandAllocatorHandle      allocator;
    uint                             allocator_buffer;
    gpu::Queue                       queue;
    ulong                            submit_visit_epoch;
    gpu::CompletionPoint             submitted_completion;
    CommandRecord*                   next_submitted;
    bool                             graphics_state_initialized;
    gpu::PrimitiveTopology           raster_topology;
    bool                             raster_topology_emitted;
    BoundPipeline                    logical_pipeline;
    NativePipelineBinding            native_graphics_pipeline;
    NativePipelineBinding            native_compute_pipeline;
    NativePipelineBinding            native_ray_tracing_pipeline;
    DescriptorHeapCommandBinding     descriptor_heap_binding;
    CommandScratch*                  scratch;
    uint                             pending_acceleration_structure_head;
    char[]                           inline_root;
    bool                             internal;
    gpu::TextureHandle[]             init_textures;
}

struct CommandCell {
    CommandRecord record;
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

fn CommandRecord*? CommandTable.claim(
    &self,
    gpu::Device device,
    VkDeviceState* device_state,
    CommandRecord value,
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
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].used = true;
    gpu::internal::CommandListHandle identity = {
        .owner = self.owner,
        .index = idx,
        .generation = self.slots[idx].generation,
    };
    self.slots[idx].record = value;
    self.slots[idx].record.device_state = device_state;
    self.slots[idx].record.device = device;
    self.slots[idx].record.identity = identity;
    self.slots[idx].record.state =
        gpu::internal::CommandState.INACTIVE;
    return &self.slots[idx].record;
}

fn CommandRecord*? CommandTable.get(
    &self,
    gpu::internal::CommandListHandle handle,
) {
    if (handle.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = handle.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != handle.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].record;
}

fn void? CommandTable.free(&self, gpu::internal::CommandListHandle handle) {
    self.get(handle)!;
    uint idx = handle.index;
    self.slots[idx].record = {};
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
    gpu::internal::CommandListHandle handle,
) {
    return self.free(handle);
}

fn CommandRecord*? resolve_command(
    gpu::CommandList* token,
) @feat(TESTING) {
    CommandRecord* record =
        gpu::internal::recording_command(token)!;
    return record;
}

fn void release_command(
    VkDeviceState* state,
    gpu::internal::CommandListHandle handle,
) {
    if (try record = state.commands.get(handle)) {
        (void)state.commands.free(handle);
    }
}

fn void destroy_command_table(VkDeviceState* state) {
    if (state.commands.slots.ptr == null) return;
    foreach (&cell : state.commands.slots[:state.commands.next_new]) {
        if (cell.used) {
            abandon_pending_acceleration_structure_builds(
                &cell.record,
                true,
            );
        }
    }
    mem::free(state.commands.slots.ptr);
    state.commands = {};
}
