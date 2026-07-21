module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;
import std::atomic::types;
import std::core::mem;

// Concrete slot table avoids a c3c 0.8.0 generic-instantiation codegen failure.
const uint PIPELINE_FREE_NONE = 0;

struct PipelineSlot {
    vk::Pipeline                    pipeline;
    vk::PipelineLayout              layout;
    vk::IndirectCommandsLayoutEXT   generated_dispatch_layout;
    uint                            entry;
    Atomic{uint}                    validation_references;
}

struct PipelineCell {
    PipelineSlot value;
    uint         generation;
    uint         next_free;
    bool         used;
}

struct PipelineTable {
    PipelineCell[] slots;
    ulong          owner;
    uint           free_head;
    uint           next_new;
}

fn gpu::PipelineHandle? PipelineTable.alloc(&self, PipelineSlot value) {
    uint idx;
    bool found = false;
    while (self.free_head != PIPELINE_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
        if (can_allocate_handle_generation(self.slots[idx].generation)) {
            found = true;
            break;
        }
        self.slots[idx].next_free = PIPELINE_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].used = true;
    return (gpu::PipelineHandle){
        .owner      = self.owner,
        .index      = idx,
        .generation = self.slots[idx].generation,
    };
}

fn PipelineSlot*? PipelineTable.get(&self, gpu::PipelineHandle h) {
    if (h.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? PipelineTable.free(&self, gpu::PipelineHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    if (!should_recycle_handle_generation(self.slots[idx].generation)) {
        self.slots[idx].generation = HANDLE_GENERATION_MAX;
        self.slots[idx].next_free = PIPELINE_FREE_NONE;
        return;
    }
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

fn uint PipelineTable.available_count(&self) {
    uint count = (uint)self.slots.len - self.next_new;
    for (uint next = self.free_head; next != PIPELINE_FREE_NONE; next = self.slots[next - 1].next_free) {
        if (can_allocate_handle_generation(self.slots[next - 1].generation)) count++;
    }
    return count;
}

fn bool pipeline_handle_capacity_available(VkDeviceState* state, usz required) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    return (usz)state.pipelines.available_count() >= required;
}

fn void? validate_compute_pipeline_desc(
    VkDeviceState* state,
    gpu::ComputePipelineDesc* desc,
) {
    if (desc.push_constant_size < gpu::RootPush::size
        || desc.push_constant_size % 4 != 0
        || desc.push_constant_size > state.caps.max_push_constant_size) {
        return gpu::INVALID_ARGUMENT~;
    }
}

fn bool compute_pipeline_is_cached(
    VkDeviceState* state,
    gpu::ComputePipelineDesc* desc,
) {
    PipelineKey key = build_compute_key(desc);
    ulong hash = hash_key(&key);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    return find_entry(
        &state.pipeline_cache,
        hash,
        &key,
        &desc.shader,
        null,
    ) != PIPELINE_ENTRY_NONE;
}

<* Dedup through the descriptor cache keyed on shader + push size. A batch may
   provide one temporary shader module shared by all descriptions using the
   same exact ShaderCode value. *>
fn gpu::PipelineHandle? create_compute_pipeline_from_module(
    gpu::Device* device,
    gpu::ComputePipelineDesc* desc,
    PipelineShaderModule* batch_shader,
) {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    validate_compute_pipeline_desc(state, desc)!;
    PipelineKey key = build_compute_key(desc);
    ulong hash = hash_key(&key);
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        uint hit = find_entry(
            &state.pipeline_cache,
            hash,
            &key,
            &desc.shader,
            null,
        );
        if (hit != PIPELINE_ENTRY_NONE) {
            PipelineCacheEntry* entry = &state.pipeline_cache.entries[hit];
            entry.refcount++;
            if (try handle = state.pipelines.alloc({
                .pipeline                  = entry.pipeline,
                .layout                    = entry.layout,
                .generated_dispatch_layout = entry.generated_dispatch_layout,
                .entry                     = hit,
            })) {
                return handle;
            }
            entry.refcount--;
            return gpu::SLOT_TABLE_FULL~;
        }
    }

    ComputeLayoutRef compute_layout = get_compute_layout(
        state,
        desc.push_constant_size,
    )!;
    PipelineShaderModule owned_shader;
    defer destroy_pipeline_shader_module(state, &owned_shader);
    PipelineShaderModule* shader = batch_shader;
    if (shader == null) {
        owned_shader = create_pipeline_shader_module(
            state:          state,
            code:           &desc.shader,
            expected_stage: gpu::ShaderStage.COMPUTE,
            operation:      "create_compute_pipeline",
            rejected_field: "ComputePipelineDesc.shader",
        )!;
        shader = &owned_shader;
    }

    vk::ComputePipelineCreateInfo pipe_info = vk::compute_pipeline_create_info();
    pipe_info.stage = vk::pipeline_shader_stage_create_info()
        .set_stage(vk::ShaderStageFlagBits.COMPUTE_BIT)
        .set_mod(shader.shader_module)
        .set_name(&shader.entry_point[0]);
    pipe_info.layout = compute_layout.layout;
    if (strict_heap_uses_descriptor_buffer(&state.strict_heap)) {
        pipe_info.flags = vk::PipelineCreateFlagBits.DESCRIPTOR_BUFFER_BIT_EXT;
    }

    vk::Pipeline pipeline;
    check_pipeline_create_device_result(
        state: state,
        result: vk::create_compute_pipelines(
            device:            state.device,
            pipeline_cache:    state.driver_cache,
            create_info_count: 1,
            p_create_infos:    &pipe_info,
            p_allocator:       null,
            p_pipelines:       &pipeline,
        ),
    )!;
    state.pipelines_created++;
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.PIPELINE,
        object_handle: (ulong)pipeline,
        name:          desc.debug_name,
    );

    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    uint entry_index;
    uint raced = find_entry(
        &state.pipeline_cache,
        hash,
        &key,
        &desc.shader,
        null,
    );
    if (raced != PIPELINE_ENTRY_NONE) {
        vk::destroy_pipeline(state.device, pipeline, null);
        state.pipeline_cache.entries[raced].refcount++;
        entry_index = raced;
    } else if (try idx = insert_entry(
        state: state,
        value: {
            .hash                      = hash,
            .key                       = key,
            .pipeline                  = pipeline,
            .layout                    = compute_layout.layout,
            .generated_dispatch_layout = compute_layout.generated_dispatch_layout,
        },
        first_shader:  &desc.shader,
        second_shader: null,
    )) {
        entry_index = idx;
    } else {
        vk::destroy_pipeline(state.device, pipeline, null);
        return gpu::SLOT_TABLE_FULL~;
    }

    PipelineCacheEntry* entry = &state.pipeline_cache.entries[entry_index];
    if (try handle = state.pipelines.alloc({
        .pipeline                  = entry.pipeline,
        .layout                    = entry.layout,
        .generated_dispatch_layout = entry.generated_dispatch_layout,
        .entry                     = entry_index,
    })) {
        return handle;
    }
    release_entry(state, entry_index);
    return gpu::SLOT_TABLE_FULL~;
}

fn gpu::PipelineHandle? vk_create_compute_pipeline(
    gpu::Device* device,
    gpu::ComputePipelineDesc* desc,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    state.pipeline_creation_mutex.lock()!!;
    defer state.pipeline_creation_mutex.unlock();
    return create_compute_pipeline_from_module(device, desc, null);
}

fn void? vk_create_compute_pipelines(
    gpu::Device* device,
    gpu::ComputePipelineDesc[] descs,
    gpu::PipelineHandle[] out_pipelines,
) @private {
    if (out_pipelines.len != descs.len) return gpu::INVALID_ARGUMENT~;
    if (descs.len == 0) return;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    state.pipeline_creation_mutex.lock()!!;
    defer state.pipeline_creation_mutex.unlock();

    foreach (&desc : descs) validate_compute_pipeline_desc(state, desc)!;
    if (!pipeline_handle_capacity_available(state, descs.len)) {
        return gpu::SLOT_TABLE_FULL~;
    }

    @pool() {
        PipelineBatchShader[] shaders = mem::talloc_array(PipelineBatchShader, descs.len);
        PipelineShaderModule*[] native_shaders =
            mem::talloc_array(PipelineShaderModule*, descs.len);
        uint shader_count = 0;
        defer destroy_pipeline_batch_shaders(state, shaders, shader_count);
        foreach (i, &desc : descs) {
            native_shaders[i] = null;
            if (compute_pipeline_is_cached(state, desc)) continue;
            native_shaders[i] = get_pipeline_batch_shader(
                state:          state,
                code:           &desc.shader,
                expected_stage: gpu::ShaderStage.COMPUTE,
                operation:      "create_compute_pipelines",
                rejected_field: "ComputePipelineDesc.shader",
                shaders:        shaders,
                shader_count:   &shader_count,
            )!;
        }

        gpu::PipelineHandle[] created = mem::talloc_array(gpu::PipelineHandle, descs.len);
        usz created_count = 0;
        foreach (i, &desc : descs) {
            if (pipeline_creation_should_fail(state, (usz)i)) {
                for (usz rollback = 0; rollback < created_count; rollback++) {
                    vk_destroy_pipeline(device, created[rollback])!!;
                }
                return gpu::PIPELINE_CREATE_FAILED~;
            }
            gpu::PipelineHandle? created_optional = create_compute_pipeline_from_module(
                device,
                desc,
                native_shaders[i],
            );
            if (catch create_fault = created_optional) {
                for (usz rollback = 0; rollback < created_count; rollback++) {
                    vk_destroy_pipeline(device, created[rollback])!!;
                }
                return create_fault~;
            }
            created[i] = created_optional;
            created_count++;
        }
        foreach (i, pipeline : created) out_pipelines[i] = pipeline;
    };
}
fn void? vk_destroy_pipeline(gpu::Device* device, gpu::PipelineHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    PipelineSlot* slot = state.pipelines.get(handle)!;
    if (slot.validation_references.load(AtomicOrdering.RELAXED) > 0) {
        drain_completed_submitted_commands(state)!;
        if (slot.validation_references.load(AtomicOrdering.RELAXED) > 0) {
            return gpu::RESOURCE_IN_USE~;
        }
    }
    uint entry_index = slot.entry;
    state.pipelines.free(handle)!;
    release_entry(state, entry_index);
}
