module gpu_test;

import gpu @public;

fn void ray_tracing_configuration_is_zero_compatible() @test {
    DeviceDesc device_desc;
    DeviceCaps caps;
    StageMask stage = { .ray_tracing = true };

    assert(!device_desc.enable_ray_tracing_pipelines);
    assert(!caps.acceleration_structures.enabled);
    assert(!caps.ray_queries.enabled);
    assert(!caps.ray_tracing_pipelines.enabled);
    assert((uint)stage == (1u << 11));
}

fn void ray_tracing_capabilities_separate_shared_and_pipeline_limits() @test {
    DeviceCaps caps = {
        .acceleration_structures = {
            .enabled             = true,
            .heap_capacity       = 64,
            .max_geometry_count  = 8,
            .max_primitive_count = 1024,
            .max_instance_count  = 32,
            .scratch_alignment   = 256,
        },
        .ray_queries = { .enabled = true },
        .ray_tracing_pipelines = {
            .enabled                           = true,
            .max_recursion_depth               = 4,
            .max_ray_dispatch_invocation_count = 65_536,
            .shader_group_handle_size          = 32,
            .shader_group_handle_alignment     = 32,
            .shader_group_base_alignment       = 64,
            .max_shader_group_stride           = 4096,
            .max_hit_attribute_size            = 32,
        },
    };

    assert(caps.acceleration_structures.heap_capacity == 64);
    assert(caps.acceleration_structures.max_geometry_count == 8);
    assert(caps.acceleration_structures.max_primitive_count == 1024);
    assert(caps.acceleration_structures.max_instance_count == 32);
    assert(caps.acceleration_structures.scratch_alignment == 256);
    assert(caps.ray_queries.enabled);
    assert(caps.ray_tracing_pipelines.max_recursion_depth == 4);
    assert(caps.ray_tracing_pipelines.max_ray_dispatch_invocation_count == 65_536);
    assert(caps.ray_tracing_pipelines.shader_group_handle_size == 32);
    assert(caps.ray_tracing_pipelines.shader_group_handle_alignment == 32);
    assert(caps.ray_tracing_pipelines.shader_group_base_alignment == 64);
    assert(caps.ray_tracing_pipelines.max_shader_group_stride == 4096);
    assert(caps.ray_tracing_pipelines.max_hit_attribute_size == 32);
}

fn void ray_tracing_descriptors_preserve_group_and_sbt_structure() @test {
    ShaderDesc raygen = {};
    ShaderDesc closest_hit = {};
    RayTracingHitGroupDesc[2] groups = {
        { .type = RayTracingHitGroupType.TRIANGLES },
        {
            .type               = RayTracingHitGroupType.PROCEDURAL,
            .closest_hit_shader = &closest_hit,
            .intersection_shader = &closest_hit,
        },
    };
    RayTracingPipelineDesc desc = {
        .ray_generation_shaders = (&raygen)[:1],
        .hit_groups             = groups[..],
        .max_recursion_depth    = 3,
        .debug_name             = "value test",
    };
    RayTracingPipelineInfo info = {
        .ray_generation   = { .first_group = 0, .group_count = 1 },
        .miss             = { .first_group = 1, .group_count = 0 },
        .hit              = { .first_group = 1, .group_count = 2 },
        .callable         = { .first_group = 3, .group_count = 0 },
        .total_group_count = 3,
    };
    GpuSpan raygen_span = {
        .owner      = 1,
        .index      = 2,
        .generation = 3,
        .offset     = 64,
        .size       = 32,
    };
    RayTracingShaderBindingTable table = {
        .ray_generation = { .span = raygen_span, .stride = 32 },
    };

    assert(desc.ray_generation_shaders.len == 1);
    assert(desc.hit_groups.len == 2);
    assert(desc.hit_groups[0].closest_hit_shader == null);
    assert(desc.hit_groups[1].intersection_shader == &closest_hit);
    assert(info.ray_generation.first_group == 0);
    assert(info.hit.first_group == 1 && info.hit.group_count == 2);
    assert(info.total_group_count == 3);
    assert(table.ray_generation.span.size == 32);
    assert(table.ray_generation.stride == 32);
    assert(table.miss.span.size == 0 && table.miss.stride == 0);
}
