module gpu::vk;

import gpu;
import vk;

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
    uint            free_head;
    uint            next_new;
}

fn gpu::SemaphoreHandle? SemaphoreTable.alloc(&self, SemaphoreSlot value) {
    uint idx;
    if (self.free_head != SEMAPHORE_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
    } else if (self.next_new < self.slots.len) {
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    } else {
        return gpu::SLOT_TABLE_FULL~;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation++;
    self.slots[idx].used = true;
    return (gpu::SemaphoreHandle){ .index = idx, .generation = self.slots[idx].generation };
}

fn SemaphoreSlot*? SemaphoreTable.get(&self, gpu::SemaphoreHandle h) {
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? SemaphoreTable.free(&self, gpu::SemaphoreHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    self.slots[idx].generation++;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

fn void? vk_cmd_buffer_barrier(gpu::CommandList* commands, gpu::BufferBarrier* barrier) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    BufferSlot* slot = state.buffers.get(barrier.buffer)!;

    vk::BufferMemoryBarrier2 bar = vk::buffer_memory_barrier2()
        .set_src_stage_mask(stage_to_vk(barrier.before_stage))
        .set_src_access_mask(hazard_to_vk_access(barrier.before_hazard))
        .set_dst_stage_mask(stage_to_vk(barrier.after_stage))
        .set_dst_access_mask(hazard_to_vk_access(barrier.after_hazard))
        .set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .set_buffer(slot.buffer)
        .set_offset((vk::DeviceSize)barrier.offset)
        .set_size((vk::DeviceSize)barrier.size);

    vk::DependencyInfo dep = vk::dependency_info().set_buffer_memory_barriers((&bar)[:1]);
    vk::cmd_pipeline_barrier2((vk::CommandBuffer)commands.backend, &dep);
}

fn void? vk_cmd_global_barrier(gpu::CommandList* commands, gpu::GlobalBarrier* barrier) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    vk::MemoryBarrier2 bar = vk::memory_barrier2()
        .set_src_stage_mask(stage_to_vk(barrier.before_stage))
        .set_src_access_mask(hazard_to_vk_access(barrier.before_hazard))
        .set_dst_stage_mask(stage_to_vk(barrier.after_stage))
        .set_dst_access_mask(hazard_to_vk_access(barrier.after_hazard));
    vk::DependencyInfo dep = vk::dependency_info().set_memory_barriers((&bar)[:1]);
    vk::cmd_pipeline_barrier2((vk::CommandBuffer)commands.backend, &dep);
}

<* Whole-image barrier; a barrier whose old_layout disagrees with the tracked
   layout faults INVALID_RESOURCE_STATE. *>
fn void? vk_cmd_texture_barrier(gpu::CommandList* commands, gpu::TextureBarrier* barrier) @private {
    if (commands.state != gpu::CommandState.RECORDING) return gpu::COMMAND_RECORDING_ERROR~;
    VkDeviceState* state = (VkDeviceState*)commands.device.backend_state;
    TextureSlot* slot = state.textures.get(barrier.texture)!;
    if (barrier.old_layout != slot.current_layout) return gpu::INVALID_RESOURCE_STATE~;

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
    vk::cmd_pipeline_barrier2((vk::CommandBuffer)commands.backend, &dep);
    slot.current_layout = barrier.new_layout;
}

fn gpu::SemaphoreHandle? vk_create_semaphore(gpu::Device* device, gpu::SemaphoreDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    vk::SemaphoreTypeCreateInfo type_info = vk::semaphore_type_create_info()
        .set_semaphore_type(vk::SemaphoreType.TIMELINE)
        .set_initial_value((ulong)desc.initial_value);
    vk::SemaphoreCreateInfo info = vk::semaphore_create_info().set_next(&type_info);
    vk::Semaphore semaphore;
    check(vk::create_semaphore(state.device, &info, null, &semaphore))!;

    if (try handle = state.semaphores.alloc({ .semaphore = semaphore })) {
        return handle;
    }
    vk::destroy_semaphore(state.device, semaphore, null);
    return gpu::SLOT_TABLE_FULL~;
}

fn void? vk_destroy_semaphore(gpu::Device* device, gpu::SemaphoreHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    SemaphoreSlot* slot = state.semaphores.get(handle)!;
    vk::destroy_semaphore(state.device, slot.semaphore, null);
    state.semaphores.free(handle)!;
}

fn void? vk_wait_semaphore(gpu::Device* device, gpu::SemaphoreHandle handle, gpu::SemaphoreValue value, ulong timeout_ns) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    SemaphoreSlot* slot = state.semaphores.get(handle)!;
    ulong wait_value = (ulong)value;
    vk::SemaphoreWaitInfo wait_info = vk::semaphore_wait_info()
        .set_semaphore_count(1)
        .set_semaphores((&slot.semaphore)[:1])
        .set_values((&wait_value)[:1]);
    check(vk::wait_semaphores(state.device, &wait_info, timeout_ns))!;
}
