module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::core::mem;
import std::thread;

const char[*] RT_PIPELINE_RGEN_SPIRV =
    $embed("shaders/ray_stage.rgen.spv");
const char[*] RT_PIPELINE_MISS_SPIRV =
    $embed("shaders/ray_stage.rmiss.spv");
const char[*] RT_PIPELINE_CLOSEST_HIT_SPIRV =
    $embed("shaders/ray_stage.rchit.spv");
const char[*] RT_PIPELINE_ALT_CLOSEST_HIT_SPIRV =
    $embed("shaders/ray_trace_functional_triangle.rchit.spv");
const char[*] RT_PIPELINE_ANY_HIT_SPIRV =
    $embed("shaders/ray_stage.rahit.spv");
const char[*] RT_PIPELINE_INTERSECTION_SPIRV =
    $embed("shaders/ray_stage.rint.spv");
const char[*] RT_PIPELINE_CALLABLE_SPIRV =
    $embed("shaders/ray_stage.rcall.spv");

struct RayTracingPipelineCreateCapture {
    vk::PipelineLayout expected_layout;
    vk::Result         result;
    uint               calls;
    bool               matched;
    bool               key_matched;
    bool               expected_dynamic_stack_size;
}

struct RayTracingGroupHandleCapture {
    vk::Pipeline expected_pipeline;
    uint         calls;
    bool         matched;
}

struct RayTracingGroupStackSizeCapture {
    vk::Pipeline          expected_pipeline;
    uint                  expected_group;
    vk::ShaderGroupShaderKHR expected_group_shader;
    vk::DeviceSize        result;
    uint                  calls;
    bool                  matched;
}

struct RayTracingGroupStackQueryArgs {
    vk::VkDeviceState*                state;
    gpu::PipelineHandle               pipeline;
    vk::RayTracingGroupStackSizeOps*  ops;
}

fn int query_ray_tracing_group_stack_size(void* raw_args) {
    RayTracingGroupStackQueryArgs* args =
        (RayTracingGroupStackQueryArgs*)raw_args;
    ulong? size = vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        args.state,
        pipeline:     args.pipeline,
        group:        0,
        group_shader: gpu::RayTracingGroupShader.GENERAL,
        ops:          args.ops,
    );
    if (catch size) return 1;
    return size == 37 ? 0 : 1;
}

struct RayTracingLateSlotFailureCapture {
    RayTracingPipelineCreateCapture create;
    uint saved_free_head;
    uint saved_next_new;
}

fn uint? insert_ray_tracing_cache_then_exhaust_pipeline_slots(
    void* raw_capture,
    vk::VkDeviceState* state,
    vk::PipelineCacheEntry value,
) {
    RayTracingLateSlotFailureCapture* capture =
        (RayTracingLateSlotFailureCapture*)raw_capture;
    uint index = vk::insert_entry(state, value)!;
    capture.saved_free_head = state.pipelines.free_head;
    capture.saved_next_new = state.pipelines.next_new;
    state.pipelines.free_head = vk::PIPELINE_FREE_NONE;
    state.pipelines.next_new = (uint)state.pipelines.slots.len;
    return index;
}

fn uint read_ray_tracing_key_u32(char[] bytes, usz offset) {
    return (uint)bytes[offset]
        | (uint)bytes[offset + 1] << 8
        | (uint)bytes[offset + 2] << 16
        | (uint)bytes[offset + 3] << 24;
}

fn uint? capture_ray_tracing_cache_insert(
    void* raw_capture,
    vk::VkDeviceState* state,
    vk::PipelineCacheEntry value,
) {
    RayTracingPipelineCreateCapture* capture =
        (RayTracingPipelineCreateCapture*)raw_capture;
    capture.key_matched = capture.key_matched
        && read_ray_tracing_key_u32(value.ray_tracing_key.bytes, 44)
            == (uint)value.ray_tracing_shader_ids[2];
    return vk::insert_entry(state, value);
}

fn vk::Result capture_ray_tracing_pipeline_create(
    void* raw_capture,
    vk::Device device,
    vk::PipelineCache pipeline_cache,
    vk::RayTracingPipelineCreateInfoKHR* info,
    vk::Pipeline* out_pipeline,
) {
    (void)device;
    (void)pipeline_cache;
    RayTracingPipelineCreateCapture* capture =
        (RayTracingPipelineCreateCapture*)raw_capture;
    capture.calls++;
    vk::RayTracingShaderGroupCreateInfoKHR* groups = info.p_groups;
    vk::PipelineShaderStageCreateInfo* stages = info.p_stages;
    capture.matched = capture.matched
        && info.stage_count == 6
        && info.group_count == 5
        && info.max_pipeline_ray_recursion_depth == 2
        && (capture.expected_dynamic_stack_size
            ? info.p_dynamic_state != null
                && info.p_dynamic_state.dynamic_state_count == 1
                && info.p_dynamic_state.p_dynamic_states[0]
                    == vk::DynamicState.RAY_TRACING_PIPELINE_STACK_SIZE_KHR
            : info.p_dynamic_state == null)
        && info.layout == capture.expected_layout
        && stages[0].stage == vk::ShaderStageFlagBits.RAYGEN_BIT_KHR
        && stages[1].stage == vk::ShaderStageFlagBits.MISS_BIT_KHR
        && stages[2].stage == vk::ShaderStageFlagBits.CLOSEST_HIT_BIT_KHR
        && stages[3].stage == vk::ShaderStageFlagBits.ANY_HIT_BIT_KHR
        && stages[4].stage == vk::ShaderStageFlagBits.INTERSECTION_BIT_KHR
        && stages[5].stage == vk::ShaderStageFlagBits.CALLABLE_BIT_KHR
        && groups[0].type
            == vk::RayTracingShaderGroupTypeKHR.GENERAL_KHR
        && groups[0].general_shader == 0
        && groups[1].general_shader == 1
        && groups[2].type
            == vk::RayTracingShaderGroupTypeKHR.TRIANGLES_HIT_GROUP_KHR
        && groups[2].general_shader == vk::SHADER_UNUSED_KHR
        && groups[2].closest_hit_shader == 2
        && groups[2].any_hit_shader == 3
        && groups[2].intersection_shader == vk::SHADER_UNUSED_KHR
        && groups[3].type
            == vk::RayTracingShaderGroupTypeKHR.PROCEDURAL_HIT_GROUP_KHR
        && groups[3].general_shader == vk::SHADER_UNUSED_KHR
        && groups[3].intersection_shader == 4
        && groups[4].type
            == vk::RayTracingShaderGroupTypeKHR.GENERAL_KHR
        && groups[4].general_shader == 5;
    if (capture.result == vk::Result.SUCCESS) {
        *out_pipeline = (vk::Pipeline)null;
    }
    return capture.result;
}

fn vk::Result create_null_ray_tracing_shader_module(
    void* user_data,
    vk::Device device,
    vk::ShaderModuleCreateInfo* create_info,
    vk::ShaderModule* out_shader,
) {
    (void)user_data;
    (void)device;
    (void)create_info;
    *out_shader = (vk::ShaderModule)null;
    return vk::Result.SUCCESS;
}

fn vk::Result capture_ray_tracing_group_handles(
    void* raw_capture,
    vk::Device device,
    vk::Pipeline pipeline,
    uint first_group,
    uint group_count,
    sz data_size,
    void* data,
) {
    (void)device;
    RayTracingGroupHandleCapture* capture =
        (RayTracingGroupHandleCapture*)raw_capture;
    capture.calls++;
    capture.matched = capture.matched
        && pipeline == capture.expected_pipeline
        && first_group == 1
        && group_count == 2
        && data_size == 8
        && data != null;
    char* bytes = (char*)data;
    for (usz i = 0; i < (usz)data_size; i++) bytes[i] = (char)(i + 1);
    return vk::Result.SUCCESS;
}

fn vk::DeviceSize capture_ray_tracing_group_stack_size(
    void* raw_capture,
    vk::Device device,
    vk::Pipeline pipeline,
    uint group,
    vk::ShaderGroupShaderKHR group_shader,
) {
    (void)device;
    RayTracingGroupStackSizeCapture* capture =
        (RayTracingGroupStackSizeCapture*)raw_capture;
    capture.calls++;
    capture.matched = capture.matched
        && pipeline == capture.expected_pipeline
        && group == capture.expected_group
        && group_shader == capture.expected_group_shader;
    return capture.result;
}

fn gpu::ShaderDesc ray_tracing_test_shader(char[] spirv) {
    return {
        .spirv       = spirv,
        .entry_point = "main",
    };
}

fn void ray_tracing_hit_key_starts_after_ray_generation_and_miss() @test {
    gpu::ShaderDesc closest_hit =
        ray_tracing_test_shader(RT_PIPELINE_CLOSEST_HIT_SPIRV[..]);
    gpu::RayTracingHitGroupDesc[1] groups = {{
        .kind               = gpu::RayTracingHitGroupKind.TRIANGLES,
        .closest_hit_shader = &closest_hit,
    }};
    gpu::RayTracingPipelineDesc desc = { .hit_groups = groups[..] };
    vk::ShaderId[3] shader_ids = {
        (vk::ShaderId)11,
        (vk::ShaderId)13,
        (vk::ShaderId)17,
    };
    vk::RayTracingHitGroupKey[1] keys;
    vk::build_ray_tracing_hit_keys(
        desc:             &desc,
        shader_ids:       shader_ids[..],
        hit_keys:         keys[..],
        first_hit_shader: 2,
    );
    assert(keys[0].closest_hit_shader == (vk::ShaderId)17);
}

fn void ray_tracing_descriptor_normalizes_depth_ranges_and_stage_count() @test {
    gpu::ShaderDesc[2] ray_generation = {
        ray_tracing_test_shader(RT_PIPELINE_RGEN_SPIRV[..]),
        ray_tracing_test_shader(RT_PIPELINE_RGEN_SPIRV[..]),
    };
    gpu::ShaderDesc[1] miss = {
        ray_tracing_test_shader(RT_PIPELINE_MISS_SPIRV[..]),
    };
    gpu::ShaderDesc closest_hit =
        ray_tracing_test_shader(RT_PIPELINE_CLOSEST_HIT_SPIRV[..]);
    gpu::ShaderDesc any_hit =
        ray_tracing_test_shader(RT_PIPELINE_ANY_HIT_SPIRV[..]);
    gpu::ShaderDesc intersection =
        ray_tracing_test_shader(RT_PIPELINE_INTERSECTION_SPIRV[..]);
    gpu::ShaderDesc[1] callable = {
        ray_tracing_test_shader(RT_PIPELINE_CALLABLE_SPIRV[..]),
    };
    gpu::RayTracingHitGroupDesc[3] hit_groups = {
        { .kind = gpu::RayTracingHitGroupKind.TRIANGLES },
        {
            .kind               = gpu::RayTracingHitGroupKind.TRIANGLES,
            .closest_hit_shader = &closest_hit,
            .any_hit_shader     = &any_hit,
        },
        {
            .kind                = gpu::RayTracingHitGroupKind.PROCEDURAL,
            .intersection_shader = &intersection,
        },
    };
    gpu::RayTracingPipelineDesc desc = {
        .ray_generation_shaders = ray_generation[..],
        .miss_shaders           = miss[..],
        .hit_groups             = hit_groups[..],
        .callable_shaders       = callable[..],
    };
    vk::VkDeviceState state = {
        .ray_tracing_pipelines_enabled = true,
        .caps = {
            .ray_tracing_pipelines = {
                .enabled             = true,
                .max_recursion_depth = 4,
            },
        },
    };

    vk::RayTracingPipelineShape shape =
        vk::normalize_ray_tracing_pipeline_desc(&state, &desc)!!;
    assert(shape.recursion_depth == 1);
    assert(shape.stage_count == 7);
    assert(shape.info.ray_generation.first_group == 0
        && shape.info.ray_generation.group_count == 2);
    assert(shape.info.miss.first_group == 2
        && shape.info.miss.group_count == 1);
    assert(shape.info.hit.first_group == 3
        && shape.info.hit.group_count == 3);
    assert(shape.info.callable.first_group == 6
        && shape.info.callable.group_count == 1);
    assert(shape.info.total_group_count == 7);
    assert(shape.info.max_recursion_depth == 1);
    assert(!shape.info.dynamic_stack_size);
    desc.dynamic_stack_size = true;
    shape = vk::normalize_ray_tracing_pipeline_desc(&state, &desc)!!;
    assert(shape.info.dynamic_stack_size);
}

fn void ray_tracing_descriptor_rejects_unsupported_and_invalid_groups() @test {
    gpu::ShaderDesc ray_generation =
        ray_tracing_test_shader(RT_PIPELINE_RGEN_SPIRV[..]);
    gpu::ShaderDesc closest_hit =
        ray_tracing_test_shader(RT_PIPELINE_CLOSEST_HIT_SPIRV[..]);
    gpu::ShaderDesc intersection =
        ray_tracing_test_shader(RT_PIPELINE_INTERSECTION_SPIRV[..]);
    gpu::RayTracingHitGroupDesc group;
    gpu::RayTracingPipelineDesc desc = {
        .ray_generation_shaders = (&ray_generation)[:1],
        .hit_groups             = (&group)[:1],
        .max_recursion_depth    = 3,
    };
    vk::VkDeviceState state = {
        .ray_tracing_pipelines_enabled = true,
        .caps = {
            .ray_tracing_pipelines = {
                .enabled             = true,
                .max_recursion_depth = 2,
            },
        },
    };

    assert(@catch(vk::normalize_ray_tracing_pipeline_desc(
        &state,
        &desc,
    )) == gpu::INVALID_ARGUMENT);
    desc.max_recursion_depth = 1;
    group = {
        .kind                = gpu::RayTracingHitGroupKind.TRIANGLES,
        .intersection_shader = &intersection,
    };
    assert(@catch(vk::normalize_ray_tracing_pipeline_desc(
        &state,
        &desc,
    )) == gpu::INVALID_ARGUMENT);
    group = {
        .kind               = gpu::RayTracingHitGroupKind.PROCEDURAL,
        .closest_hit_shader = &closest_hit,
    };
    assert(@catch(vk::normalize_ray_tracing_pipeline_desc(
        &state,
        &desc,
    )) == gpu::INVALID_ARGUMENT);
    desc.ray_generation_shaders = {};
    assert(@catch(vk::normalize_ray_tracing_pipeline_desc(
        &state,
        &desc,
    )) == gpu::INVALID_ARGUMENT);
    desc.ray_generation_shaders = (&ray_generation)[:1];
    state.ray_tracing_pipelines_enabled = false;
    state.caps.ray_tracing_pipelines.enabled = false;
    assert(@catch(vk::normalize_ray_tracing_pipeline_desc(
        &state,
        &desc,
    )) == gpu::UNSUPPORTED_FEATURE);
}

fn void ray_tracing_creation_lowers_groups_and_aliases_exact_cache_key() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = cache_test_state(&device);
    state.ray_tracing_pipelines_enabled = true;
    state.acceleration_structures_enabled = true;
    state.caps.ray_tracing_pipelines = {
        .enabled             = true,
        .max_recursion_depth = 4,
        .shader_group_handle_size = 4,
    };
    state.ray_tracing_layout = state.compute_layout;
    defer {
        state.ray_tracing_layout = (vk::PipelineLayout)null;
        state.ray_tracing_pipelines_enabled = false;
        state.acceleration_structures_enabled = false;
        state.caps.ray_tracing_pipelines = {};
    };

    gpu::ShaderDesc ray_generation =
        ray_tracing_test_shader(RT_PIPELINE_RGEN_SPIRV[..]);
    gpu::ShaderDesc miss =
        ray_tracing_test_shader(RT_PIPELINE_MISS_SPIRV[..]);
    gpu::ShaderDesc closest_hit =
        ray_tracing_test_shader(RT_PIPELINE_CLOSEST_HIT_SPIRV[..]);
    gpu::ShaderDesc any_hit =
        ray_tracing_test_shader(RT_PIPELINE_ANY_HIT_SPIRV[..]);
    gpu::ShaderDesc intersection =
        ray_tracing_test_shader(RT_PIPELINE_INTERSECTION_SPIRV[..]);
    gpu::ShaderDesc callable =
        ray_tracing_test_shader(RT_PIPELINE_CALLABLE_SPIRV[..]);
    gpu::RayTracingHitGroupDesc[2] hit_groups = {
        {
            .kind               = gpu::RayTracingHitGroupKind.TRIANGLES,
            .closest_hit_shader = &closest_hit,
            .any_hit_shader     = &any_hit,
        },
        {
            .kind                = gpu::RayTracingHitGroupKind.PROCEDURAL,
            .intersection_shader = &intersection,
        },
    };
    gpu::RayTracingPipelineDesc desc = {
        .ray_generation_shaders = (&ray_generation)[:1],
        .miss_shaders           = (&miss)[:1],
        .hit_groups             = hit_groups[..],
        .callable_shaders       = (&callable)[:1],
        .max_recursion_depth    = 2,
    };
    RayTracingPipelineCreateCapture capture = {
        .expected_layout = state.ray_tracing_layout,
        .matched         = true,
        .key_matched     = true,
    };
    vk::RayTracingPipelineCreateOps ops =
        vk::ray_tracing_pipeline_create_ops_real();
    ops.user_data = &capture;
    ops.create = &capture_ray_tracing_pipeline_create;
    ops.shader_module.create = &create_null_ray_tracing_shader_module;
    ops.insert_cache_entry = &capture_ray_tracing_cache_insert;

    gpu::PipelineHandle first =
        vk::vk_create_ray_tracing_pipeline_with_ops(
            state,
            &desc,
            &ops,
        )!!;
    desc.debug_name = "alias-only-name";
    gpu::PipelineHandle second =
        vk::vk_create_ray_tracing_pipeline_with_ops(
            state,
            &desc,
            &ops,
        )!!;
    assert(capture.matched && capture.key_matched && capture.calls == 1);
    assert(first != second && cache_entry_count(&device) == 1);
    desc.dynamic_stack_size = true;
    capture.expected_dynamic_stack_size = true;
    gpu::PipelineHandle dynamic =
        vk::vk_create_ray_tracing_pipeline_with_ops(
            state,
            &desc,
            &ops,
        )!!;
    assert(capture.matched && capture.key_matched && capture.calls == 2);
    assert(cache_entry_count(&device) == 2);
    gpu::RayTracingPipelineInfo dynamic_info =
        vk::vk_get_ray_tracing_pipeline_info(state, dynamic)!!;
    assert(dynamic_info.dynamic_stack_size
        && dynamic_info.max_recursion_depth == 2);
    desc.dynamic_stack_size = false;
    capture.expected_dynamic_stack_size = false;
    gpu::ShaderDesc alternate_closest_hit =
        ray_tracing_test_shader(RT_PIPELINE_ALT_CLOSEST_HIT_SPIRV[..]);
    hit_groups[0].closest_hit_shader = &alternate_closest_hit;
    gpu::PipelineHandle distinct_hit_pipeline =
        vk::vk_create_ray_tracing_pipeline_with_ops(
            state,
            &desc,
            &ops,
        )!!;
    assert(capture.key_matched
        && capture.calls == 3
        && cache_entry_count(&device) == 3,
        "hit-only shader differences produce distinct cache identities");
    hit_groups[0].closest_hit_shader = &closest_hit;
    assert(vk::vk_get_ray_tracing_pipeline_info(state, first)!!.total_group_count
        == 5);
    RayTracingGroupHandleCapture handles_capture = {
        .expected_pipeline =
            state.pipelines.slots[first.index].value.pipeline,
        .matched = true,
    };
    vk::RayTracingGroupHandleOps handle_ops = {
        .user_data = &handles_capture,
        .get       = &capture_ray_tracing_group_handles,
    };
    char[8] handles;
    vk::vk_get_ray_tracing_shader_group_handles_with_ops(
        state,
        first,
        1,
        2,
        handles[..],
        &handle_ops,
    )!!;
    assert(handles_capture.matched && handles_capture.calls == 1);
    assert(handles[0] == 1 && handles[7] == 8);
    assert(@catch(vk::vk_get_ray_tracing_shader_group_handles_with_ops(
        state,
        first,
        1,
        2,
        handles[:7],
        &handle_ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::vk_get_ray_tracing_shader_group_handles_with_ops(
        state,
        first,
        4,
        2,
        handles[..],
        &handle_ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(handles_capture.calls == 1,
        "invalid handle queries perform no backend call");

    RayTracingGroupStackSizeCapture stack_capture = {
        .expected_pipeline     =
            state.pipelines.slots[first.index].value.pipeline,
        .expected_group        = 0,
        .expected_group_shader = vk::ShaderGroupShaderKHR.GENERAL_KHR,
        .result                = 37,
        .matched               = true,
    };
    vk::RayTracingGroupStackSizeOps stack_ops = {
        .user_data = &stack_capture,
        .get       = &capture_ray_tracing_group_stack_size,
    };
    assert(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     first,
        group:        0,
        group_shader: gpu::RayTracingGroupShader.GENERAL,
        ops:          &stack_ops,
    )!! == 37);
    assert(stack_capture.matched && stack_capture.calls == 1);
    stack_capture.expected_group = 2;
    stack_capture.expected_group_shader =
        vk::ShaderGroupShaderKHR.CLOSEST_HIT_KHR;
    assert(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     first,
        group:        2,
        group_shader: gpu::RayTracingGroupShader.CLOSEST_HIT,
        ops:          &stack_ops,
    )!! == 37);
    stack_capture.expected_group_shader =
        vk::ShaderGroupShaderKHR.ANY_HIT_KHR;
    assert(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     first,
        group:        2,
        group_shader: gpu::RayTracingGroupShader.ANY_HIT,
        ops:          &stack_ops,
    )!! == 37);
    stack_capture.expected_group = 3;
    stack_capture.expected_group_shader =
        vk::ShaderGroupShaderKHR.INTERSECTION_KHR;
    assert(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     first,
        group:        3,
        group_shader: gpu::RayTracingGroupShader.INTERSECTION,
        ops:          &stack_ops,
    )!! == 37);
    stack_capture.expected_group = 4;
    stack_capture.expected_group_shader =
        vk::ShaderGroupShaderKHR.GENERAL_KHR;
    assert(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     first,
        group:        4,
        group_shader: gpu::RayTracingGroupShader.GENERAL,
        ops:          &stack_ops,
    )!! == 37);
    assert(stack_capture.matched && stack_capture.calls == 5);
    stack_capture.expected_group = 0;
    stack_capture.expected_group_shader =
        vk::ShaderGroupShaderKHR.GENERAL_KHR;
    RayTracingGroupStackQueryArgs[2] query_args;
    Thread[2] query_threads;
    foreach (index, &query_arg : query_args) {
        *query_arg = {
            .state    = state,
            .pipeline = first,
            .ops      = &stack_ops,
        };
        query_threads[index].create(
            &query_ray_tracing_group_stack_size,
            query_arg,
        )!!;
    }
    foreach (&query_thread : query_threads) {
        assert(query_thread.join()!! == 0,
            "concurrent immutable stack-size query succeeds");
    }
    assert(stack_capture.matched && stack_capture.calls == 7,
        "concurrent stack-size queries serialize through the resource domain");
    assert(@catch(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     first,
        group:        0,
        group_shader: gpu::RayTracingGroupShader.CLOSEST_HIT,
        ops:          &stack_ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     first,
        group:        2,
        group_shader: gpu::RayTracingGroupShader.GENERAL,
        ops:          &stack_ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     first,
        group:        3,
        group_shader: gpu::RayTracingGroupShader.CLOSEST_HIT,
        ops:          &stack_ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     first,
        group:        2,
        group_shader: gpu::RayTracingGroupShader.INTERSECTION,
        ops:          &stack_ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     first,
        group:        5,
        group_shader: gpu::RayTracingGroupShader.GENERAL,
        ops:          &stack_ops,
    )) == gpu::INVALID_ARGUMENT);
    gpu::RayTracingGroupShader invalid_group_shader;
    int invalid_group_shader_value = 99;
    mem::copy(
        &invalid_group_shader,
        &invalid_group_shader_value,
        int::size,
    );
    assert(@catch(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     first,
        group:        0,
        group_shader: invalid_group_shader,
        ops:          &stack_ops,
    )) == gpu::INVALID_ARGUMENT);
    gpu::PipelineHandle foreign = first;
    foreign.owner++;
    assert(@catch(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     foreign,
        group:        0,
        group_shader: gpu::RayTracingGroupShader.GENERAL,
        ops:          &stack_ops,
    )) == gpu::INVALID_HANDLE);
    gpu::PipelineHandle stale = first;
    stale.generation++;
    assert(@catch(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     stale,
        group:        0,
        group_shader: gpu::RayTracingGroupShader.GENERAL,
        ops:          &stack_ops,
    )) == gpu::INVALID_HANDLE);
    assert(stack_capture.calls == 7);

    gpu::ShaderDesc compute_shader =
        ray_tracing_test_shader(CACHE_COMP_SPIRV[..]);
    gpu::ComputePipelineDesc compute_desc = { .shader = compute_shader };
    gpu::PipelineHandle compute =
        vk::vk_create_compute_pipeline(state, &compute_desc)!!;
    assert(@catch(vk::vk_get_ray_tracing_pipeline_info(state, compute))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::vk_get_ray_tracing_shader_group_stack_size_with_ops(
        state:        state,
        pipeline:     compute,
        group:        0,
        group_shader: gpu::RayTracingGroupShader.GENERAL,
        ops:          &stack_ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::vk_get_ray_tracing_shader_group_handles_with_ops(
        state,
        compute,
        0,
        0,
        {},
        &handle_ops,
    )) == gpu::INVALID_ARGUMENT);
    vk::vk_destroy_pipeline(state, compute)!!;

    vk::vk_destroy_pipeline(state, first)!!;
    assert(cache_entry_count(&device) == 3);
    vk::vk_destroy_pipeline(state, second)!!;
    assert(cache_entry_count(&device) == 2);
    vk::vk_destroy_pipeline(state, dynamic)!!;
    assert(cache_entry_count(&device) == 1);
    vk::vk_destroy_pipeline(state, distinct_hit_pipeline)!!;
    assert(cache_entry_count(&device) == 0);
    assert(vk::shader_store_live_count(state) == 0);

    capture = {
        .expected_layout = state.ray_tracing_layout,
        .result          = vk::Result.ERROR_OUT_OF_DEVICE_MEMORY,
        .matched         = true,
        .key_matched     = true,
    };
    ops.user_data = &capture;
    assert(@catch(vk::vk_create_ray_tracing_pipeline_with_ops(
        state,
        &desc,
        &ops,
    )) == gpu::OUT_OF_DEVICE_MEMORY);
    assert(cache_entry_count(&device) == 0
        && vk::shader_store_live_count(state) == 0);

    capture.result = vk::Result.SUCCESS;
    ops.insert_cache_entry = &fail_pipeline_cache_insert;
    assert(@catch(vk::vk_create_ray_tracing_pipeline_with_ops(
        state,
        &desc,
        &ops,
    )) == gpu::SLOT_TABLE_FULL);
    assert(cache_entry_count(&device) == 0
        && vk::shader_store_live_count(state) == 0);

    RayTracingLateSlotFailureCapture late_failure = {
        .create = {
            .expected_layout = state.ray_tracing_layout,
            .matched         = true,
            .key_matched     = true,
        },
    };
    ops.user_data = &late_failure;
    ops.insert_cache_entry =
        &insert_ray_tracing_cache_then_exhaust_pipeline_slots;
    assert(@catch(vk::vk_create_ray_tracing_pipeline_with_ops(
        state,
        &desc,
        &ops,
    )) == gpu::SLOT_TABLE_FULL);
    state.pipelines.free_head = late_failure.saved_free_head;
    state.pipelines.next_new = late_failure.saved_next_new;
    assert(cache_entry_count(&device) == 0
        && vk::shader_store_live_count(state) == 0);
}
