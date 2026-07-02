module gpu::vk;

import gpu;
import vk;

// Concrete (non-generic) slot table for pipelines. Mirrors gpu::slots but
// monomorphized to sidestep a c3c 0.8.0 codegen crash tied to generic-module
// instantiation of the slot table (see samples/README.md).
const uint PIPELINE_FREE_NONE = 0;

struct PipelineSlot {
    vk::Pipeline       pipeline;
    vk::PipelineLayout layout;
}

struct PipelineCell {
    PipelineSlot value;
    uint         generation;
    uint         next_free;
    bool         used;
}

struct PipelineTable {
    PipelineCell[] slots;
    uint           free_head;
    uint           next_new;
}

fn gpu::PipelineHandle? PipelineTable.alloc(&self, PipelineSlot value) {
    uint idx;
    if (self.free_head != PIPELINE_FREE_NONE) {
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
    return (gpu::PipelineHandle){ .index = idx, .generation = self.slots[idx].generation };
}

fn PipelineSlot*? PipelineTable.get(&self, gpu::PipelineHandle h) {
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? PipelineTable.free(&self, gpu::PipelineHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    self.slots[idx].generation++;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

<* Layout = root push-constant range + the global heap set-0 layout. The
   ComputePipelineCreateInfo stage/layout are assigned directly rather than
   through the builder setters: set_stage takes the embedded struct by value,
   and that by-value flow crashes c3c 0.8.0 codegen. *>
fn gpu::PipelineHandle? vk_create_compute_pipeline(gpu::Device* device, gpu::ComputePipelineDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    if (desc.push_constant_size < 8) return gpu::INVALID_ARGUMENT~;
    ShaderSlot* shader = state.shaders.get(desc.shader)!;

    vk::PushConstantRange range = {
        .stage_flags = vk::ShaderStageFlagBits.COMPUTE_BIT,
        .offset      = 0,
        .size        = desc.push_constant_size,
    };
    vk::PipelineLayoutCreateInfo layout_info = vk::pipeline_layout_create_info()
        .set_set_layouts((&state.heap.set_layout)[:1])
        .set_push_constant_ranges((&range)[:1]);
    vk::PipelineLayout layout;
    check(vk::create_pipeline_layout(state.device, &layout_info, null, &layout))!;

    vk::ComputePipelineCreateInfo pipe_info = vk::compute_pipeline_create_info();
    pipe_info.stage = vk::pipeline_shader_stage_create_info()
        .set_stage(vk::ShaderStageFlagBits.COMPUTE_BIT)
        .set_mod(shader.shader_module)
        .set_name(shader.entry_point);
    pipe_info.layout = layout;
    if (state.heap_mode == gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER) {
        pipe_info.flags = vk::PipelineCreateFlagBits.DESCRIPTOR_BUFFER_BIT_EXT;
    }

    vk::Pipeline pipeline;
    if (catch check(vk::create_compute_pipelines(state.device, (vk::PipelineCache)null, 1, &pipe_info, null, &pipeline))) {
        vk::destroy_pipeline_layout(state.device, layout, null);
        return gpu::PIPELINE_CREATE_FAILED~;
    }

    if (try handle = state.pipelines.alloc({ .pipeline = pipeline, .layout = layout })) {
        return handle;
    }
    vk::destroy_pipeline(state.device, pipeline, null);
    vk::destroy_pipeline_layout(state.device, layout, null);
    return gpu::SLOT_TABLE_FULL~;
}

fn void? vk_destroy_pipeline(gpu::Device* device, gpu::PipelineHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    PipelineSlot* slot = state.pipelines.get(handle)!;
    vk::destroy_pipeline(state.device, slot.pipeline, null);
    vk::destroy_pipeline_layout(state.device, slot.layout, null);
    state.pipelines.free(handle)!;
}
