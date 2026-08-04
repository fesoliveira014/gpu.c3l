module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::mem;
import std::core::mem::alloc;

alias RayTracingPipelineCreateFn @private = fn vk::Result (
    void* user_data,
    vk::Device device,
    vk::PipelineCache pipeline_cache,
    vk::RayTracingPipelineCreateInfoKHR* create_info,
    vk::Pipeline* pipeline,
);

alias RayTracingGroupHandleGetFn @private = fn vk::Result (
    void* user_data,
    vk::Device device,
    vk::Pipeline pipeline,
    uint first_group,
    uint group_count,
    sz data_size,
    void* data,
);

const uint RAY_TRACING_HIT_CLOSEST_PRESENT = 0x1;
const uint RAY_TRACING_HIT_ANY_PRESENT = 0x2;
const uint RAY_TRACING_HIT_INTERSECTION_PRESENT = 0x4;

struct RayTracingPipelineCreateOps {
    ShaderModuleCreateOps       shader_module;
    void*                       user_data;
    RayTracingPipelineCreateFn  create;
    PipelineCacheInsertFn       insert_cache_entry;
}

struct RayTracingGroupHandleOps {
    void*                       user_data;
    RayTracingGroupHandleGetFn  get;
}

struct RayTracingPipelineShape {
    gpu::RayTracingPipelineInfo info;
    uint                        stage_count;
    uint                        recursion_depth;
}

fn RayTracingPipelineShape? normalize_ray_tracing_pipeline_desc(
    VkDeviceState* state,
    gpu::RayTracingPipelineDesc* desc,
) {
    if (state == null || desc == null) return gpu::INVALID_ARGUMENT~;
    if (!state.ray_tracing_pipelines_enabled
        || !state.caps.ray_tracing_pipelines.enabled) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    if (desc.ray_generation_shaders.len == 0
        || desc.ray_generation_shaders.ptr == null
        || (desc.miss_shaders.len > 0 && desc.miss_shaders.ptr == null)
        || (desc.hit_groups.len > 0 && desc.hit_groups.ptr == null)
        || (desc.callable_shaders.len > 0
            && desc.callable_shaders.ptr == null)
        || desc.ray_generation_shaders.len > uint::max
        || desc.miss_shaders.len > uint::max
        || desc.hit_groups.len > uint::max
        || desc.callable_shaders.len > uint::max) {
        return gpu::INVALID_ARGUMENT~;
    }
    uint recursion_depth = desc.max_recursion_depth == 0
        ? 1
        : desc.max_recursion_depth;
    if (recursion_depth
        > state.caps.ray_tracing_pipelines.max_recursion_depth) {
        return gpu::INVALID_ARGUMENT~;
    }
    gpu::RayTracingPipelineInfo info =
        ray_tracing_pipeline_info_for_counts(
            (uint)desc.ray_generation_shaders.len,
            (uint)desc.miss_shaders.len,
            (uint)desc.hit_groups.len,
            (uint)desc.callable_shaders.len,
        )!;
    ulong stage_count = desc.ray_generation_shaders.len
        + desc.miss_shaders.len
        + desc.callable_shaders.len;
    foreach (&group : desc.hit_groups) {
        switch (group.kind) {
            case gpu::RayTracingHitGroupKind.TRIANGLES:
                if (group.intersection_shader != null) {
                    return gpu::INVALID_ARGUMENT~;
                }
            case gpu::RayTracingHitGroupKind.PROCEDURAL:
                if (group.intersection_shader == null) {
                    return gpu::INVALID_ARGUMENT~;
                }
            default:
                return gpu::INVALID_ARGUMENT~;
        }
        if (group.closest_hit_shader != null) stage_count++;
        if (group.any_hit_shader != null) stage_count++;
        if (group.intersection_shader != null) stage_count++;
        if (stage_count > uint::max) return gpu::INVALID_ARGUMENT~;
    }
    return {
        .info            = info,
        .stage_count     = (uint)stage_count,
        .recursion_depth = recursion_depth,
    };
}

fn vk::Result create_ray_tracing_pipeline_real(
    void* user_data,
    vk::Device device,
    vk::PipelineCache pipeline_cache,
    vk::RayTracingPipelineCreateInfoKHR* create_info,
    vk::Pipeline* pipeline,
) @private {
    VkDeviceState* state = (VkDeviceState*)user_data;
    return state.ray_tracing_pipeline_operations.create_pipelines(
        device,
        (vk::DeferredOperationKHR)null,
        pipeline_cache,
        1,
        create_info,
        null,
        pipeline,
    );
}

fn RayTracingPipelineCreateOps ray_tracing_pipeline_create_ops_real() {
    return {
        .shader_module      = shader_module_create_ops_real(),
        .create             = &create_ray_tracing_pipeline_real,
        .insert_cache_entry = &insert_pipeline_cache_entry_real,
    };
}

fn void? normalize_ray_tracing_shader_inputs(
    VkDeviceState* state,
    gpu::RayTracingPipelineDesc* desc,
    ShaderIdentityInput[] inputs,
) {
    usz index = 0;
    foreach (&shader : desc.ray_generation_shaders) {
        normalize_shader_input(
            state:          state,
            desc:           shader,
            role:           gpu::internal::ShaderRole.RAY_GENERATION,
            operation:      "create_ray_tracing_pipeline",
            rejected_field: "RayTracingPipelineDesc.ray_generation_shaders",
            out_input:      &inputs[index++],
        )!;
    }
    foreach (&shader : desc.miss_shaders) {
        normalize_shader_input(
            state:          state,
            desc:           shader,
            role:           gpu::internal::ShaderRole.MISS,
            operation:      "create_ray_tracing_pipeline",
            rejected_field: "RayTracingPipelineDesc.miss_shaders",
            out_input:      &inputs[index++],
        )!;
    }
    foreach (&group : desc.hit_groups) {
        if (group.closest_hit_shader != null) {
            normalize_shader_input(
                state:          state,
                desc:           group.closest_hit_shader,
                role:           gpu::internal::ShaderRole.CLOSEST_HIT,
                operation:      "create_ray_tracing_pipeline",
                rejected_field: "RayTracingHitGroupDesc.closest_hit_shader",
                out_input:      &inputs[index++],
            )!;
        }
        if (group.any_hit_shader != null) {
            normalize_shader_input(
                state:          state,
                desc:           group.any_hit_shader,
                role:           gpu::internal::ShaderRole.ANY_HIT,
                operation:      "create_ray_tracing_pipeline",
                rejected_field: "RayTracingHitGroupDesc.any_hit_shader",
                out_input:      &inputs[index++],
            )!;
        }
        if (group.intersection_shader != null) {
            normalize_shader_input(
                state:          state,
                desc:           group.intersection_shader,
                role:           gpu::internal::ShaderRole.INTERSECTION,
                operation:      "create_ray_tracing_pipeline",
                rejected_field: "RayTracingHitGroupDesc.intersection_shader",
                out_input:      &inputs[index++],
            )!;
        }
    }
    foreach (&shader : desc.callable_shaders) {
        normalize_shader_input(
            state:          state,
            desc:           shader,
            role:           gpu::internal::ShaderRole.CALLABLE,
            operation:      "create_ray_tracing_pipeline",
            rejected_field: "RayTracingPipelineDesc.callable_shaders",
            out_input:      &inputs[index++],
        )!;
    }
}

fn void build_ray_tracing_hit_keys(
    gpu::RayTracingPipelineDesc* desc,
    ShaderId[] shader_ids,
    RayTracingHitGroupKey[] hit_keys,
    usz first_hit_shader,
) {
    usz shader_index = first_hit_shader;
    foreach (index, &group : desc.hit_groups) {
        RayTracingHitGroupKey key = {
            .kind                = group.kind,
            .closest_hit_shader  = SHADER_ID_INVALID,
            .any_hit_shader      = SHADER_ID_INVALID,
            .intersection_shader = SHADER_ID_INVALID,
        };
        if (group.closest_hit_shader != null) {
            key.presence_mask |= RAY_TRACING_HIT_CLOSEST_PRESENT;
            key.closest_hit_shader = shader_ids[shader_index++];
        }
        if (group.any_hit_shader != null) {
            key.presence_mask |= RAY_TRACING_HIT_ANY_PRESENT;
            key.any_hit_shader = shader_ids[shader_index++];
        }
        if (group.intersection_shader != null) {
            key.presence_mask |= RAY_TRACING_HIT_INTERSECTION_PRESENT;
            key.intersection_shader = shader_ids[shader_index++];
        }
        hit_keys[index] = key;
    }
}

fn void lower_ray_tracing_stages_and_groups(
    gpu::RayTracingPipelineDesc* desc,
    ShaderIdentityInput[] inputs,
    PipelineShaderModule[] modules,
    vk::PipelineShaderStageCreateInfo[] stages,
    vk::RayTracingShaderGroupCreateInfoKHR[] groups,
) {
    foreach (index, &input : inputs) {
        stages[index] = vk::pipeline_shader_stage_create_info()
            .set_stage(shader_role_to_vk(input.role)!!)
            .set_mod(modules[index].shader_module)
            .set_name(&modules[index].entry_point[0]);
    }
    uint shader_index = 0;
    uint group_index = 0;
    foreach (shader : desc.ray_generation_shaders) {
        (void)shader;
        groups[group_index++] = vk::ray_tracing_shader_group_create_info_khr()
            .set_type(vk::RayTracingShaderGroupTypeKHR.GENERAL_KHR)
            .set_general_shader(shader_index++)
            .set_closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .set_any_hit_shader(vk::SHADER_UNUSED_KHR)
            .set_intersection_shader(vk::SHADER_UNUSED_KHR);
    }
    foreach (shader : desc.miss_shaders) {
        (void)shader;
        groups[group_index++] = vk::ray_tracing_shader_group_create_info_khr()
            .set_type(vk::RayTracingShaderGroupTypeKHR.GENERAL_KHR)
            .set_general_shader(shader_index++)
            .set_closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .set_any_hit_shader(vk::SHADER_UNUSED_KHR)
            .set_intersection_shader(vk::SHADER_UNUSED_KHR);
    }
    foreach (&group : desc.hit_groups) {
        uint closest_hit = vk::SHADER_UNUSED_KHR;
        uint any_hit = vk::SHADER_UNUSED_KHR;
        uint intersection = vk::SHADER_UNUSED_KHR;
        if (group.closest_hit_shader != null) closest_hit = shader_index++;
        if (group.any_hit_shader != null) any_hit = shader_index++;
        if (group.intersection_shader != null) intersection = shader_index++;
        vk::RayTracingShaderGroupTypeKHR type =
            group.kind == gpu::RayTracingHitGroupKind.TRIANGLES
                ? vk::RayTracingShaderGroupTypeKHR.TRIANGLES_HIT_GROUP_KHR
                : vk::RayTracingShaderGroupTypeKHR.PROCEDURAL_HIT_GROUP_KHR;
        groups[group_index++] = vk::ray_tracing_shader_group_create_info_khr()
            .set_type(type)
            .set_general_shader(vk::SHADER_UNUSED_KHR)
            .set_closest_hit_shader(closest_hit)
            .set_any_hit_shader(any_hit)
            .set_intersection_shader(intersection);
    }
    foreach (shader : desc.callable_shaders) {
        (void)shader;
        groups[group_index++] = vk::ray_tracing_shader_group_create_info_khr()
            .set_type(vk::RayTracingShaderGroupTypeKHR.GENERAL_KHR)
            .set_general_shader(shader_index++)
            .set_closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .set_any_hit_shader(vk::SHADER_UNUSED_KHR)
            .set_intersection_shader(vk::SHADER_UNUSED_KHR);
    }
}

fn gpu::PipelineHandle? create_ray_tracing_pipeline_from_prepared(
    VkDeviceState* state,
    gpu::RayTracingPipelineDesc* desc,
    RayTracingPipelineShape* shape,
    ShaderIdentityInput[] shader_inputs,
    PreparedShaderIdentity[] prepared,
    ShaderId[] shader_ids,
    RayTracingHitGroupKey[] hit_keys,
    RayTracingPipelineCreateOps* create_ops,
    bool* shader_ids_transferred,
) {
    *shader_ids_transferred = false;
    usz ray_generation_count = desc.ray_generation_shaders.len;
    usz miss_count = desc.miss_shaders.len;
    usz first_hit_shader = ray_generation_count + miss_count;
    build_ray_tracing_hit_keys(
        desc,
        shader_ids,
        hit_keys,
        first_hit_shader,
    );
    usz callable_start = first_hit_shader;
    foreach (&group : desc.hit_groups) {
        if (group.closest_hit_shader != null) callable_start++;
        if (group.any_hit_shader != null) callable_start++;
        if (group.intersection_shader != null) callable_start++;
    }
    RayTracingPipelineKeyInput key_input = {
        .ray_generation_shaders = shader_ids[0:ray_generation_count],
        .miss_shaders = shader_ids[
            ray_generation_count:miss_count
        ],
        .hit_groups = hit_keys,
        .callable_shaders = shader_ids[
            callable_start:desc.callable_shaders.len
        ],
    };
    OwnedRayTracingPipelineKey key = create_ray_tracing_pipeline_key(
        state,
        &key_input,
        shape.recursion_depth,
    )!;
    bool key_owned = true;
    defer if (key_owned) destroy_ray_tracing_pipeline_key(state, &key);

    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        uint hit = find_ray_tracing_entry(state, &key);
        if (hit != PIPELINE_ENTRY_NONE) {
            PipelineCacheEntry* entry = &state.pipeline_cache.entries[hit];
            entry.refcount++;
            if (try handle = state.pipelines.alloc({
                .pipeline         = entry.pipeline,
                .layout           = entry.layout,
                .ray_tracing_info = shape.info,
                .entry            = hit,
            })) {
                foreach (&identity : prepared) {
                    release_prepared_shader_locked(state, identity)!!;
                }
                return handle;
            }
            entry.refcount--;
            return gpu::SLOT_TABLE_FULL~;
        }
    }

    PipelineShaderModule[]? allocated_modules = alloc::new_array_try(
        state.host_allocator,
        PipelineShaderModule,
        shape.stage_count,
    );
    if (catch allocated_modules) return gpu::OUT_OF_HOST_MEMORY~;
    PipelineShaderModule[] modules = allocated_modules;
    defer alloc::free(state.host_allocator, modules.ptr);
    usz module_count = 0;
    defer {
        for (usz i = 0; i < module_count; i++) {
            destroy_pipeline_shader_module(state, &modules[i]);
        }
    };
    foreach (index, &input : shader_inputs) {
        modules[index] = create_pipeline_shader_module_with_ops(
            state:          state,
            code:           input,
            operation:      "create_ray_tracing_pipeline",
            rejected_field: "RayTracingPipelineDesc shaders",
            create_ops:      &create_ops.shader_module,
        )!;
        module_count++;
    }

    vk::PipelineShaderStageCreateInfo[]? allocated_stages =
        alloc::new_array_try(
            state.host_allocator,
            vk::PipelineShaderStageCreateInfo,
            shape.stage_count,
        );
    if (catch allocated_stages) return gpu::OUT_OF_HOST_MEMORY~;
    vk::PipelineShaderStageCreateInfo[] stages = allocated_stages;
    defer alloc::free(state.host_allocator, stages.ptr);
    vk::RayTracingShaderGroupCreateInfoKHR[]? allocated_groups =
        alloc::new_array_try(
            state.host_allocator,
            vk::RayTracingShaderGroupCreateInfoKHR,
            shape.info.total_group_count,
        );
    if (catch allocated_groups) return gpu::OUT_OF_HOST_MEMORY~;
    vk::RayTracingShaderGroupCreateInfoKHR[] groups = allocated_groups;
    defer alloc::free(state.host_allocator, groups.ptr);
    lower_ray_tracing_stages_and_groups(
        desc,
        shader_inputs,
        modules,
        stages,
        groups,
    );

    vk::RayTracingPipelineCreateInfoKHR info =
        vk::ray_tracing_pipeline_create_info_khr();
    info.stage_count = shape.stage_count;
    info.p_stages = stages.ptr;
    info.group_count = shape.info.total_group_count;
    info.p_groups = groups.ptr;
    info.max_pipeline_ray_recursion_depth = shape.recursion_depth;
    info.layout = state.ray_tracing_layout;
    vk::Pipeline pipeline;
    check_pipeline_create_device_result(
        state: state,
        result: create_ops.create(
            create_ops.user_data,
            state.device,
            state.driver_cache,
            &info,
            &pipeline,
        ),
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
    uint raced = find_ray_tracing_entry(state, &key);
    if (raced != PIPELINE_ENTRY_NONE) {
        vk::destroy_pipeline(state.device, pipeline, null);
        state.pipeline_cache.entries[raced].refcount++;
        foreach (&identity : prepared) {
            release_prepared_shader_locked(state, identity)!!;
        }
        entry_index = raced;
    } else if (try index = create_ops.insert_cache_entry(
        create_ops.user_data,
        state,
        {
            .hash                   = key.hash,
            .key                    = {
                .kind = (int)PipelineKind.RAY_TRACING,
            },
            .pipeline               = pipeline,
            .layout                 = state.ray_tracing_layout,
            .ray_tracing_key        = key,
            .ray_tracing_shader_ids = shader_ids,
        },
    )) {
        foreach (&identity : prepared) transfer_prepared_shader(identity);
        key_owned = false;
        *shader_ids_transferred = true;
        entry_index = index;
    } else {
        vk::destroy_pipeline(state.device, pipeline, null);
        return gpu::SLOT_TABLE_FULL~;
    }

    PipelineCacheEntry* entry = &state.pipeline_cache.entries[entry_index];
    if (try handle = state.pipelines.alloc({
        .pipeline         = entry.pipeline,
        .layout           = entry.layout,
        .ray_tracing_info = shape.info,
        .entry            = entry_index,
    })) {
        return handle;
    }
    release_entry(state, entry_index);
    return gpu::SLOT_TABLE_FULL~;
}

fn gpu::PipelineHandle? vk_create_ray_tracing_pipeline_with_ops(
    VkDeviceState* state,
    gpu::RayTracingPipelineDesc* desc,
    RayTracingPipelineCreateOps* create_ops,
) {
    state.pipeline_creation_mutex.lock()!!;
    defer state.pipeline_creation_mutex.unlock();
    RayTracingPipelineShape shape =
        normalize_ray_tracing_pipeline_desc(state, desc)!;
    if (state.ray_tracing_layout == null || create_ops == null) {
        return gpu::BACKEND_ERROR~;
    }
    if (!pipeline_handle_capacity_available(state, 1)) {
        return gpu::SLOT_TABLE_FULL~;
    }

    ShaderIdentityInput[]? allocated_inputs = alloc::new_array_try(
        state.host_allocator,
        ShaderIdentityInput,
        shape.stage_count,
    );
    if (catch allocated_inputs) return gpu::OUT_OF_HOST_MEMORY~;
    ShaderIdentityInput[] shader_inputs = allocated_inputs;
    defer alloc::free(state.host_allocator, shader_inputs.ptr);
    normalize_ray_tracing_shader_inputs(state, desc, shader_inputs)!;

    PreparedShaderIdentity[]? allocated_prepared = alloc::new_array_try(
        state.host_allocator,
        PreparedShaderIdentity,
        shape.stage_count,
    );
    if (catch allocated_prepared) return gpu::OUT_OF_HOST_MEMORY~;
    PreparedShaderIdentity[] prepared = allocated_prepared;
    defer alloc::free(state.host_allocator, prepared.ptr);
    usz prepared_count = 0;
    defer release_prepared_shaders(state, prepared[:prepared_count]);
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        foreach (index, &input : shader_inputs) {
            prepared[index] = prepare_shader_identity_locked(
                state,
                input,
            )!;
            prepared_count++;
        }
    }
    foreach (index, &input : shader_inputs) {
        ZString debug_name = input.debug_name;
        shader_store_input_for_retained_id(
            state:      state,
            id:         prepared[index].id,
            debug_name: debug_name,
            out_input:  input,
        );
    }

    ShaderId[]? allocated_shader_ids = alloc::new_array_try(
        state.host_allocator,
        ShaderId,
        shape.stage_count,
    );
    if (catch allocated_shader_ids) return gpu::OUT_OF_HOST_MEMORY~;
    ShaderId[] shader_ids = allocated_shader_ids;
    bool shader_ids_owned = true;
    defer if (shader_ids_owned) {
        alloc::free(state.host_allocator, shader_ids.ptr);
    }
    foreach (index, &identity : prepared) {
        shader_ids[index] = identity.id;
    }

    RayTracingHitGroupKey[] hit_keys;
    if (desc.hit_groups.len > 0) {
        RayTracingHitGroupKey[]? allocated_hit_keys = alloc::new_array_try(
            state.host_allocator,
            RayTracingHitGroupKey,
            desc.hit_groups.len,
        );
        if (catch allocated_hit_keys) return gpu::OUT_OF_HOST_MEMORY~;
        hit_keys = allocated_hit_keys;
    }
    defer if (hit_keys.ptr != null) {
        alloc::free(state.host_allocator, hit_keys.ptr);
    }
    bool shader_ids_transferred;
    gpu::PipelineHandle? pipeline =
        create_ray_tracing_pipeline_from_prepared(
            state:         state,
            desc:          desc,
            shape:         &shape,
            shader_inputs: shader_inputs,
            prepared:      prepared,
            shader_ids:    shader_ids,
            hit_keys:      hit_keys,
            create_ops:    create_ops,
            shader_ids_transferred: &shader_ids_transferred,
        );
    if (shader_ids_transferred) shader_ids_owned = false;
    return pipeline!;
}

fn gpu::PipelineHandle? vk_create_ray_tracing_pipeline(
    VkDeviceState* state,
    gpu::RayTracingPipelineDesc* desc,
) {
    RayTracingPipelineCreateOps create_ops =
        ray_tracing_pipeline_create_ops_real();
    create_ops.user_data = state;
    return vk_create_ray_tracing_pipeline_with_ops(
        state,
        desc,
        &create_ops,
    );
}

fn gpu::RayTracingPipelineInfo? vk_get_ray_tracing_pipeline_info(
    VkDeviceState* state,
    gpu::PipelineHandle pipeline,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    PipelineSlot* slot = state.pipelines.get(pipeline)!;
    if (!pipeline_kind_matches(state, slot, PipelineKind.RAY_TRACING)) {
        return gpu::INVALID_ARGUMENT~;
    }
    return slot.ray_tracing_info;
}

fn vk::Result get_ray_tracing_group_handles_real(
    void* user_data,
    vk::Device device,
    vk::Pipeline pipeline,
    uint first_group,
    uint group_count,
    sz data_size,
    void* data,
) @private {
    VkDeviceState* state = (VkDeviceState*)user_data;
    return state.ray_tracing_pipeline_operations.get_shader_group_handles(
        device,
        pipeline,
        first_group,
        group_count,
        data_size,
        data,
    );
}

fn void? vk_get_ray_tracing_shader_group_handles_with_ops(
    VkDeviceState* state,
    gpu::PipelineHandle pipeline,
    uint first_group,
    uint group_count,
    char[] out,
    RayTracingGroupHandleOps* ops,
) {
    if (state == null || ops == null || ops.get == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    PipelineSlot* slot = state.pipelines.get(pipeline)!;
    if (!pipeline_kind_matches(state, slot, PipelineKind.RAY_TRACING)) {
        return gpu::INVALID_ARGUMENT~;
    }
    ulong end_group = (ulong)first_group + group_count;
    if (end_group > slot.ray_tracing_info.total_group_count) {
        return gpu::INVALID_ARGUMENT~;
    }
    ulong required = (ulong)group_count
        * state.caps.ray_tracing_pipelines.shader_group_handle_size;
    if (required > usz::max
        || out.len != (usz)required
        || (required > 0 && out.ptr == null)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (group_count == 0) return;
    check_backend_result(
        state:     state,
        operation: "get_ray_tracing_shader_group_handles",
        result:    ops.get(
            ops.user_data,
            state.device,
            slot.pipeline,
            first_group,
            group_count,
            (sz)required,
            out.ptr,
        ),
    )!;
}

fn void? vk_get_ray_tracing_shader_group_handles(
    VkDeviceState* state,
    gpu::PipelineHandle pipeline,
    uint first_group,
    uint group_count,
    char[] out,
) {
    RayTracingGroupHandleOps ops = {
        .user_data = state,
        .get       = &get_ray_tracing_group_handles_real,
    };
    return vk_get_ray_tracing_shader_group_handles_with_ops(
        state,
        pipeline,
        first_group,
        group_count,
        out,
        &ops,
    );
}
