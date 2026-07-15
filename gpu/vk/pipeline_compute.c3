module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;

// Concrete slot table avoids a c3c 0.8.0 generic-instantiation codegen failure.
const uint PIPELINE_FREE_NONE = 0;

struct PipelineSlot {
    vk::Pipeline       pipeline;
    vk::PipelineLayout layout;
    uint               entry;
    gpu::RasterState   raster;
    gpu::DepthState    depth;
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
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation);
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
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation);
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

<* Dedup through the descriptor cache keyed on shader + push size. Layout is
   the shared per-push-size compute layout. The ComputePipelineCreateInfo
   stage/layout are assigned directly rather than through the builder setters:
   set_stage takes the embedded struct by value, and that by-value flow
   crashes c3c 0.8.0 codegen. *>
fn gpu::PipelineHandle? vk_create_compute_pipeline(gpu::Device* device, gpu::ComputePipelineDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    if (desc.push_constant_size < gpu::RootPush::size
        || desc.push_constant_size % 4 != 0
        || desc.push_constant_size > state.caps.max_push_constant_size) return gpu::INVALID_ARGUMENT~;
    ShaderSlot* shader = state.shaders.get(desc.shader)!;
    validate_shader_stage(
        state:          state,
        slot:           shader,
        expected:       gpu::ShaderStage.COMPUTE,
        operation:      "create_compute_pipeline",
        rejected_field: "ComputePipelineDesc.shader",
        handle:         desc.shader,
    )!;

    PipelineKey key = build_compute_key(desc);
    ulong hash = hash_key(&key);
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        uint hit = find_entry(&state.pipeline_cache, hash, &key);
        if (hit != PIPELINE_ENTRY_NONE) {
            PipelineCacheEntry* entry = &state.pipeline_cache.entries[hit];
            entry.refcount++;
            if (try handle = state.pipelines.alloc({
                .pipeline = entry.pipeline,
                .layout   = entry.layout,
                .entry    = hit,
            })) {
                return handle;
            }
            entry.refcount--;
            return gpu::SLOT_TABLE_FULL~;
        }
    }

    vk::PipelineLayout layout = get_compute_layout(state, desc.push_constant_size)!;

    vk::ComputePipelineCreateInfo pipe_info = vk::compute_pipeline_create_info();
    pipe_info.stage = vk::pipeline_shader_stage_create_info()
        .set_stage(vk::ShaderStageFlagBits.COMPUTE_BIT)
        .set_mod(shader.shader_module)
        .set_name(&shader.entry_point[0]);
    pipe_info.layout = layout;
    if (strict_heap_uses_descriptor_buffer(&state.strict_heap)) {
        pipe_info.flags = vk::PipelineCreateFlagBits.DESCRIPTOR_BUFFER_BIT_EXT;
    }

    vk::Pipeline pipeline;
    check_pipeline_create_result(
        vk::create_compute_pipelines(
            device:            state.device,
            pipeline_cache:    state.driver_cache,
            create_info_count: 1,
            p_create_infos:    &pipe_info,
            p_allocator:       null,
            p_pipelines:       &pipeline,
        )
    )!;
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.PIPELINE,
        object_handle: (ulong)pipeline,
        name:          desc.debug_name,
    );

    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    uint entry_index;
    uint raced = find_entry(&state.pipeline_cache, hash, &key);
    if (raced != PIPELINE_ENTRY_NONE) {
        vk::destroy_pipeline(state.device, pipeline, null);
        state.pipeline_cache.entries[raced].refcount++;
        entry_index = raced;
        pipeline = state.pipeline_cache.entries[raced].pipeline;
    } else if (try idx = insert_entry(&state.pipeline_cache, {
        .hash     = hash,
        .key      = key,
        .pipeline = pipeline,
        .layout   = layout,
    })) {
        entry_index = idx;
    } else {
        vk::destroy_pipeline(state.device, pipeline, null);
        return gpu::SLOT_TABLE_FULL~;
    }

    if (try handle = state.pipelines.alloc({
        .pipeline = pipeline,
        .layout   = layout,
        .entry    = entry_index,
    })) {
        return handle;
    }
    release_entry(state, entry_index);
    return gpu::SLOT_TABLE_FULL~;
}

fn void? vk_destroy_pipeline(gpu::Device* device, gpu::PipelineHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    PipelineSlot* slot = state.pipelines.get(handle)!;
    uint entry_index = slot.entry;
    state.pipelines.free(handle)!;
    release_entry(state, entry_index);
}
