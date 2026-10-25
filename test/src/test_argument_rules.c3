module gpu;

import gpu::internal @public;
import gpu::internal::vk @public;
import gpu::util @public;
import gpu_test;

fn Surface? rules_surface_factory(Runtime* runtime, void* user_data) {
    return UNSUPPORTED_FEATURE~;
}

fn Adapter? rules_adapter_selector(
    Runtime* runtime,
    DeviceDesc* requirements,
    void* user_data,
) {
    return UNSUPPORTED_FEATURE~;
}

fn void public_limits_match_backend_storage() @test {
    assert(gpu::MAX_SUBMIT_COMMAND_LISTS == 1024);
    assert(gpu::MAX_SUBMIT_COMPLETION_WAITS == 1024);
    assert(gpu::MAX_SPARSE_TILE_BINDS == 1024);
    assert(gpu::MAX_SPARSE_OPAQUE_BINDS == 1024);
    assert(gpu::MAX_SPARSE_BIND_WAITS == 1024);
    assert(gpu::MAX_SHADER_ENTRY_POINT_BYTES + 1 == gpu::internal::SHADER_ENTRY_POINT_CAP);
}

fn void runtime_desc_rules_reject_each_clause() @test {
    RuntimeDesc zero;
    assert(gpu::internal::runtime_desc_valid(&zero), "the zero descriptor is valid");
    assert(!gpu::internal::runtime_desc_valid(null), "a null descriptor is invalid");

    RuntimeDesc at_limit = {
        .texture_heap_capacity                = MAX_SHADER_HEAP_CAPACITY,
        .sampler_heap_capacity                = MAX_SHADER_HEAP_CAPACITY,
        .texture_capacity                     = MAX_SHADER_HEAP_CAPACITY,
        .pipeline_capacity                    = MAX_SHADER_HEAP_CAPACITY,
        .acceleration_structure_heap_capacity = MAX_SHADER_HEAP_CAPACITY,
    };
    assert(gpu::internal::runtime_desc_valid(&at_limit), "every capacity may equal the limit");

    uint over = MAX_SHADER_HEAP_CAPACITY + 1;
    RuntimeDesc texture_heap = { .texture_heap_capacity = over };
    RuntimeDesc sampler_heap = { .sampler_heap_capacity = over };
    RuntimeDesc texture_table = { .texture_capacity = over };
    RuntimeDesc pipeline_table = { .pipeline_capacity = over };
    RuntimeDesc acceleration_heap = { .acceleration_structure_heap_capacity = over };
    assert(!gpu::internal::runtime_desc_valid(&texture_heap));
    assert(!gpu::internal::runtime_desc_valid(&sampler_heap));
    assert(!gpu::internal::runtime_desc_valid(&texture_table));
    assert(!gpu::internal::runtime_desc_valid(&pipeline_table));
    assert(!gpu::internal::runtime_desc_valid(&acceleration_heap));

    RuntimeDesc null_cache = { .pipeline_cache_data = ((char*)null)[:4] };
    assert(!gpu::internal::runtime_desc_valid(&null_cache),
        "a cache slice with a length needs a pointer");
    char[4] cache_bytes;
    RuntimeDesc cache = { .pipeline_cache_data = cache_bytes[..] };
    assert(gpu::internal::runtime_desc_valid(&cache), "cache bytes are opaque content");
}

fn void device_desc_rules_reject_each_clause() @test {
    DeviceDesc zero;
    assert(gpu::internal::device_desc_valid(null), "null selects the defaults");
    assert(gpu::internal::device_desc_valid(&zero), "the zero descriptor is valid");

    Surface surface = { .index = 1, .generation = 1 };
    DeviceDesc default_queues_with_surface = { .surface = surface };
    assert(gpu::internal::device_desc_valid(&default_queues_with_surface),
        "the default queue set includes graphics for presentation");
    DeviceDesc explicit_request = {
        .surface = surface,
        .queues  = {
            .required       = { .graphics, .compute },
            .distinct_roles = { .compute },
        },
    };
    assert(gpu::internal::device_desc_valid(&explicit_request));

    DeviceDesc unknown_role = { .queues = { .required = (QueueRoles)8 } };
    DeviceDesc unknown_distinct = {
        .queues = {
            .required       = { .graphics },
            .distinct_roles = (QueueRoles)8,
        },
    };
    DeviceDesc distinct_not_required = {
        .queues = {
            .required       = { .graphics },
            .distinct_roles = { .compute },
        },
    };
    DeviceDesc single_queue_with_distinct = {
        .queues = {
            .required       = { .graphics, .compute, .transfer },
            .distinct_roles = { .transfer },
            .single_queue   = true,
        },
    };
    DeviceDesc single_queue_without_required = { .queues = { .single_queue = true } };
    DeviceDesc distinct_without_required = {
        .queues = { .distinct_roles = { .graphics } },
    };
    DeviceDesc presentation_without_graphics = {
        .surface = surface,
        .queues  = { .required = { .compute } },
    };
    assert(!gpu::internal::device_desc_valid(&unknown_role));
    assert(!gpu::internal::device_desc_valid(&unknown_distinct));
    assert(!gpu::internal::device_desc_valid(&distinct_not_required));
    assert(!gpu::internal::device_desc_valid(&single_queue_with_distinct));
    assert(!gpu::internal::device_desc_valid(&single_queue_without_required));
    assert(!gpu::internal::device_desc_valid(&distinct_without_required));
    assert(!gpu::internal::device_desc_valid(&presentation_without_graphics));
}

fn void ray_features_need_an_acceleration_structure_heap() @test {
    DeviceDesc plain;
    DeviceDesc ray_queries = { .enable_ray_queries = true };
    DeviceDesc ray_tracing = { .enable_ray_tracing_pipelines = true };
    assert(gpu::internal::ray_feature_heap_valid(&plain, 0),
        "a device without ray features needs no heap");
    assert(!gpu::internal::ray_feature_heap_valid(&ray_queries, 0));
    assert(!gpu::internal::ray_feature_heap_valid(&ray_tracing, 0));
    assert(gpu::internal::ray_feature_heap_valid(&ray_queries, 1));
    assert(gpu::internal::ray_feature_heap_valid(&ray_tracing, 1));
}

fn void device_context_desc_rules_reject_each_clause() @test {
    util::DeviceContextDesc defaults = util::default_device_context_desc();
    util::DeviceContextDesc zero;
    assert(util::device_context_desc_valid(&defaults), "the default descriptor is valid");
    assert(util::device_context_desc_valid(&zero), "zero queues select every role");
    assert(!util::device_context_desc_valid(null));

    util::DeviceContextDesc with_callbacks = defaults;
    with_callbacks.create_surface = &rules_surface_factory;
    with_callbacks.surface_user_data = (void*)1;
    with_callbacks.select_adapter = &rules_adapter_selector;
    with_callbacks.adapter_user_data = (void*)1;
    assert(util::device_context_desc_valid(&with_callbacks),
        "user data is valid alongside its callback");

    util::DeviceContextDesc supplied_surface = defaults;
    supplied_surface.device.surface = { .index = 1, .generation = 1 };
    util::DeviceContextDesc surface_data_alone = defaults;
    surface_data_alone.surface_user_data = (void*)1;
    util::DeviceContextDesc adapter_data_alone = defaults;
    adapter_data_alone.adapter_user_data = (void*)1;
    util::DeviceContextDesc undefined_queue = defaults;
    undefined_queue.command_queue = bitcast(99, QueueKind);
    util::DeviceContextDesc queue_not_required = defaults;
    queue_not_required.device.queues.required = { .compute };
    util::DeviceContextDesc surface_without_graphics = defaults;
    surface_without_graphics.device.queues.required = { .compute };
    surface_without_graphics.command_queue = QueueKind.COMPUTE;
    surface_without_graphics.create_surface = &rules_surface_factory;
    assert(!util::device_context_desc_valid(&supplied_surface));
    assert(!util::device_context_desc_valid(&surface_data_alone));
    assert(!util::device_context_desc_valid(&adapter_data_alone));
    assert(!util::device_context_desc_valid(&undefined_queue));
    assert(!util::device_context_desc_valid(&queue_not_required));
    assert(!util::device_context_desc_valid(&surface_without_graphics));
}

fn void allocation_alignment_rule_accepts_zero_and_powers_of_two() @test {
    assert(gpu::internal::allocation_alignment_valid(0), "zero selects the default");
    assert(gpu::internal::allocation_alignment_valid(1));
    assert(gpu::internal::allocation_alignment_valid(64));
    assert(!gpu::internal::allocation_alignment_valid(3));
    assert(!gpu::internal::allocation_alignment_valid(24));
}

fn void queue_access_shape_needs_known_bits() @test {
    assert(gpu::internal::queue_access_shape_valid({ .compute }));
    assert(gpu::internal::queue_access_shape_valid({ .graphics, .transfer }));
    assert(!gpu::internal::queue_access_shape_valid({}), "empty access is invalid");
    assert(!gpu::internal::queue_access_shape_valid((QueueRoles)0x80),
        "unknown bits are invalid");
}

fn void texture_requirement_rules_follow_the_memory_class() @test {
    TextureCompatibility compatibility = (TextureCompatibility)(((uint128)1 << 64) | 3);
    TextureRequirements valid = {
        .size          = 64,
        .alignment     = 64,
        .compatibility = compatibility,
    };
    TextureRequirements[1] one = { valid };
    AllocationDesc texture_memory = {
        .size                 = 128,
        .memory_class         = MemoryClass.TEXTURE,
        .texture_requirements = one[..],
    };
    assert(gpu::internal::texture_requirements_valid(&texture_memory));
    assert(gpu::internal::allocation_covers_texture_requirements(&texture_memory));

    AllocationDesc buffer_memory = { .size = 128, .memory_class = MemoryClass.CPU_WRITE };
    assert(gpu::internal::texture_requirements_valid(&buffer_memory),
        "a buffer class needs no requirements");
    buffer_memory.texture_requirements = one[..];
    assert(!gpu::internal::texture_requirements_valid(&buffer_memory),
        "a buffer class takes no requirements");

    AllocationDesc missing = { .size = 128, .memory_class = MemoryClass.TEXTURE };
    assert(!gpu::internal::texture_requirements_valid(&missing),
        "a TEXTURE class needs a requirement");

    TextureRequirements zero_size = valid;
    zero_size.size = 0;
    TextureRequirements zero_alignment = valid;
    zero_alignment.alignment = 0;
    TextureRequirements odd_alignment = valid;
    odd_alignment.alignment = 48;
    TextureRequirements no_compatibility = valid;
    no_compatibility.compatibility = (TextureCompatibility)0;
    TextureRequirements[4] bad = { zero_size, zero_alignment, odd_alignment, no_compatibility };
    foreach (i, &requirement : bad) {
        TextureRequirements[1] only = { *requirement };
        AllocationDesc desc = {
            .size                 = 128,
            .memory_class         = MemoryClass.TEXTURE,
            .texture_requirements = only[..],
        };
        assert(!gpu::internal::texture_requirements_valid(&desc), "requirement %d is malformed", i);
    }

    AllocationDesc small = texture_memory;
    small.size = 32;
    assert(!gpu::internal::allocation_covers_texture_requirements(&small),
        "the allocation must cover every requirement");
}

fn void span_rules_need_a_nonempty_covering_span() @test {
    GpuSpan empty = {};
    GpuSpan sixteen = { .size = 16 };
    assert(!gpu::internal::span_holds_bytes(empty, 0), "an empty span is never accepted");
    assert(gpu::internal::span_holds_bytes(sixteen, 16));
    assert(!gpu::internal::span_holds_bytes(sixteen, 17));

    assert(gpu::internal::span_holds_records(sixteen, 8, 2));
    assert(!gpu::internal::span_holds_records(sixteen, 8, 3));
    assert(gpu::internal::span_holds_records(sixteen, 8, 0), "a zero count needs only a nonempty span");
    assert(!gpu::internal::span_holds_records(empty, 8, 0));
    assert(!gpu::internal::span_holds_records({ .size = usz::max }, usz::max, 2),
        "the byte product must not overflow");
}

fn void index_span_rule_follows_the_index_type() @test {
    GpuSpan span = { .size = 12 };
    assert(gpu::internal::index_span_holds_indices(span, IndexType.U16, 6));
    assert(!gpu::internal::index_span_holds_indices(span, IndexType.U16, 7));
    assert(gpu::internal::index_span_holds_indices(span, IndexType.U32, 3));
    assert(!gpu::internal::index_span_holds_indices(span, IndexType.U32, 4));
}

fn void generated_span_rule_reads_nothing_for_a_zero_maximum() @test {
    GpuSpan empty = {};
    GpuSpan records = { .size = 48 };
    GpuSpan count = { .size = 4 };
    assert(gpu::internal::generated_spans_hold(empty, empty, 24, 0));
    assert(gpu::internal::generated_spans_hold(records, count, 24, 2));
    assert(!gpu::internal::generated_spans_hold(records, count, 24, 3));
    assert(!gpu::internal::generated_spans_hold(records, { .size = 3 }, 24, 2));
    assert(!gpu::internal::generated_spans_hold(empty, count, 24, 1));
}


fn void texture_desc_rule_rejects_each_malformed_shape() @test {
    TextureDesc valid = {
        .width  = 8,
        .height = 8,
        .format = Format.RGBA8_UNORM,
        .usage  = { .sampled },
        .access = { .graphics },
    };
    assert(gpu::internal::texture_desc_valid(&valid));

    TextureDesc zero_width = valid;
    zero_width.width = 0;
    assert(!gpu::internal::texture_desc_valid(&zero_width), "width must be greater than 0");

    TextureDesc unknown_format = valid;
    unknown_format.format = bitcast(9999, Format);
    assert(!gpu::internal::texture_desc_valid(&unknown_format), "an unknown format is not in the profile");

    TextureDesc long_chain = valid;
    long_chain.mip_levels = 5;
    assert(!gpu::internal::texture_desc_valid(&long_chain), "an 8x8 texture has four mips");
}

fn void texture_view_rules_reject_an_overflowing_range_or_unknown_aspect() @test {
    TextureViewDesc whole = {};
    TextureViewDesc ranged = { .base_mip = 1, .mip_count = 2, .base_layer = 3, .layer_count = 4 };
    TextureViewDesc[2] good = { whole, ranged };
    assert(gpu::internal::texture_view_desc_valid(null), "null selects the default view");
    assert(gpu::internal::texture_view_descs_valid(good[..]));
    assert(gpu::internal::texture_view_descs_valid({}), "an empty batch is valid");

    TextureViewDesc last_mip = { .base_mip = uint::max };
    assert(!gpu::internal::texture_view_desc_valid(&last_mip), "base_mip must leave room for one mip");
    TextureViewDesc mip_overflow = { .base_mip = uint::max - 1, .mip_count = 2 };
    assert(!gpu::internal::texture_view_desc_valid(&mip_overflow));
    TextureViewDesc layer_overflow = { .base_layer = uint::max, .layer_count = 1 };
    assert(!gpu::internal::texture_view_desc_valid(&layer_overflow));
    TextureViewDesc unknown_aspect = { .aspect = bitcast(77, TextureAspect) };
    assert(!gpu::internal::texture_view_desc_valid(&unknown_aspect), "aspect must be defined");

    TextureViewDesc[2] mixed = { whole, last_mip };
    assert(!gpu::internal::texture_view_descs_valid(mixed[..]), "one bad element rejects the slice");

    TextureViewCreateDesc[2] created = { { .view = whole }, { .view = last_mip } };
    assert(gpu::internal::texture_view_create_descs_valid(created[:1]));
    assert(!gpu::internal::texture_view_create_descs_valid(created[..]));
}

fn void sampler_rule_rejects_each_malformed_clause() @test {
    SamplerDesc valid = { .min_lod = 0.0f, .max_lod = 8.0f };
    assert(gpu::internal::sampler_desc_valid(&valid));

    SamplerDesc inverted = valid;
    inverted.min_lod = 4.0f;
    inverted.max_lod = 2.0f;
    assert(!gpu::internal::sampler_desc_valid(&inverted), "min_lod must not exceed max_lod");

    SamplerDesc not_a_number = valid;
    not_a_number.mip_lod_bias = float::nan;
    assert(!gpu::internal::sampler_desc_valid(&not_a_number), "bias must be finite");
    SamplerDesc infinite = valid;
    infinite.max_lod = float::inf;
    assert(!gpu::internal::sampler_desc_valid(&infinite), "max_lod must be finite");

    SamplerDesc bad_filter = valid;
    bad_filter.mag_filter = bitcast(5, Filter);
    assert(!gpu::internal::sampler_desc_valid(&bad_filter), "filters must be defined");
    SamplerDesc bad_compare = valid;
    bad_compare.compare_enable = true;
    bad_compare.compare = bitcast(50, CompareOp);
    assert(!gpu::internal::sampler_desc_valid(&bad_compare), "an enabled compare op must be defined");
    bad_compare.compare_enable = false;
    assert(gpu::internal::sampler_desc_valid(&bad_compare), "a disabled compare op is ignored");

    SamplerDesc weak = valid;
    weak.anisotropy_enable = true;
    weak.max_anisotropy = 0.5f;
    assert(!gpu::internal::sampler_desc_valid(&weak), "active anisotropy must be at least 1");
    weak.max_anisotropy = float::nan;
    assert(!gpu::internal::sampler_desc_valid(&weak), "active anisotropy must be finite");
    weak.anisotropy_enable = false;
    assert(gpu::internal::sampler_desc_valid(&weak), "inactive anisotropy is ignored");
}

fn void shader_entry_point_rule_bounds_the_name() @test {
    assert(gpu::internal::shader_entry_point_valid(null), "a null name selects main");
    assert(gpu::internal::shader_entry_point_valid("main"));
    assert(!gpu::internal::shader_entry_point_valid(""), "an empty name is invalid");

    char[MAX_SHADER_ENTRY_POINT_BYTES + 2] name;
    name[..] = 'a';
    name[MAX_SHADER_ENTRY_POINT_BYTES] = 0;
    assert(gpu::internal::shader_entry_point_valid((ZString)&name[0]), "the longest name is valid");
    name[MAX_SHADER_ENTRY_POINT_BYTES] = 'a';
    name[MAX_SHADER_ENTRY_POINT_BYTES + 1] = 0;
    assert(!gpu::internal::shader_entry_point_valid((ZString)&name[0]), "one byte over the limit is invalid");
}

fn void graphics_attachment_rules_reject_each_clause() @test {
    assert(gpu::internal::min_sample_shading_valid(0.0f));
    assert(gpu::internal::min_sample_shading_valid(1.0f));
    assert(!gpu::internal::min_sample_shading_valid(1.5f), "a fraction above 1 is invalid");
    assert(!gpu::internal::min_sample_shading_valid(-0.5f), "a negative fraction is invalid");
    assert(!gpu::internal::min_sample_shading_valid(float::nan), "NaN is invalid");

    assert(gpu::internal::polygon_mode_valid(PolygonMode.LINE));
    assert(!gpu::internal::polygon_mode_valid(bitcast(9, PolygonMode)), "polygon mode must be defined");

    Format[2] colors = { Format.RGBA8_UNORM, Format.BGRA8_SRGB };
    assert(gpu::internal::color_formats_valid(colors[..]));
    assert(gpu::internal::color_formats_valid({}), "an empty list is valid");
    Format[2] undefined_color = { Format.RGBA8_UNORM, Format.UNDEFINED };
    assert(!gpu::internal::color_formats_valid(undefined_color[..]), "UNDEFINED is not a color format");
    Format[1] depth_color = { Format.D32_FLOAT };
    assert(!gpu::internal::color_formats_valid(depth_color[..]), "a depth format is not a color format");
    Format[1] unknown_color = { bitcast(900, Format) };
    assert(!gpu::internal::color_formats_valid(unknown_color[..]), "an unknown format is not a color format");

    assert(gpu::internal::depth_format_valid(Format.UNDEFINED));
    assert(gpu::internal::depth_format_valid(Format.D24_UNORM_S8_UINT));
    assert(!gpu::internal::depth_format_valid(Format.RGBA8_UNORM), "a color format is not a depth format");
}

fn void ray_tracing_pipeline_rules_reject_each_clause() @test {
    ShaderDesc[1] raygen = { { .spirv = (char[])"x" } };
    ShaderDesc[1] miss = { { .spirv = (char[])"x" } };
    RayTracingPipelineDesc valid = { .ray_generation_shaders = raygen[..] };
    assert(gpu::internal::ray_tracing_pipeline_desc_valid(&valid));

    RayTracingPipelineDesc empty_raygen = {};
    assert(!gpu::internal::ray_tracing_pipeline_desc_valid(&empty_raygen), "ray generation must not be empty");

    RayTracingPipelineDesc null_miss = valid;
    null_miss.miss_shaders = ((ShaderDesc*)null)[:1];
    assert(!gpu::internal::ray_tracing_pipeline_desc_valid(&null_miss), "a null miss slice needs length 0");
    RayTracingPipelineDesc null_hit = valid;
    null_hit.hit_groups = ((RayTracingHitGroupDesc*)null)[:1];
    assert(!gpu::internal::ray_tracing_pipeline_desc_valid(&null_hit), "a null hit slice needs length 0");
    RayTracingPipelineDesc null_callable = valid;
    null_callable.callable_shaders = ((ShaderDesc*)null)[:1];
    assert(!gpu::internal::ray_tracing_pipeline_desc_valid(&null_callable), "a null callable slice needs length 0");

    RayTracingPipelineDesc huge_miss = valid;
    huge_miss.miss_shaders = (&miss[0])[:(usz)uint::max + 1];
    assert(!gpu::internal::ray_tracing_pipeline_desc_valid(&huge_miss), "a slice over uint::max is invalid");

    RayTracingPipelineDesc summed = valid;
    summed.ray_generation_shaders = (&raygen[0])[:uint::max];
    summed.miss_shaders = (&miss[0])[:2];
    assert(!gpu::internal::ray_tracing_pipeline_desc_valid(&summed), "the summed count must fit uint");
}

fn void ray_tracing_stage_rules_reject_each_clause() @test {
    ShaderDesc hit = { .spirv = (char[])"x" };
    RayTracingHitGroupDesc[1] triangles = { { .kind = RayTracingHitGroupKind.TRIANGLES, .closest_hit_shader = &hit } };
    assert(gpu::internal::ray_tracing_hit_groups_valid(triangles[..]));
    triangles[0].intersection_shader = &hit;
    assert(!gpu::internal::ray_tracing_hit_groups_valid(triangles[..]), "TRIANGLES must not have an intersection shader");

    RayTracingHitGroupDesc[1] procedural = { { .kind = RayTracingHitGroupKind.PROCEDURAL } };
    assert(!gpu::internal::ray_tracing_hit_groups_valid(procedural[..]), "PROCEDURAL needs an intersection shader");
    procedural[0].intersection_shader = &hit;
    assert(gpu::internal::ray_tracing_hit_groups_valid(procedural[..]));

    RayTracingHitGroupDesc[1] unknown = { { .kind = bitcast(7, RayTracingHitGroupKind) } };
    assert(!gpu::internal::ray_tracing_hit_groups_valid(unknown[..]), "the kind must be defined");

    ShaderDesc[1] raygen = { { .spirv = (char[])"x" } };
    RayTracingPipelineDesc desc = {
        .ray_generation_shaders = raygen[..],
        .hit_groups             = procedural[..],
    };
    assert(gpu::internal::ray_tracing_entry_points_valid(&desc));
    raygen[0].entry_point = "";
    assert(!gpu::internal::ray_tracing_entry_points_valid(&desc), "a ray generation entry must be valid");
    raygen[0].entry_point = null;
    hit.entry_point = "";
    assert(!gpu::internal::ray_tracing_entry_points_valid(&desc), "a hit group entry must be valid");
}

fn void render_pass_rules_reject_each_clause() @test {
    ColorTargetDesc[MAX_COLOR_ATTACHMENTS + 1] colors;
    RenderPassDesc desc = { .colors = colors[:MAX_COLOR_ATTACHMENTS], .width = 4, .height = 4 };
    assert(gpu::internal::render_pass_desc_valid(&desc));

    desc.colors = colors[:MAX_COLOR_ATTACHMENTS + 1];
    assert(!gpu::internal::render_pass_desc_valid(&desc), "more colors than the fixed array holds");

    desc.colors = ((ColorTargetDesc*)null)[:1];
    assert(!gpu::internal::render_pass_desc_valid(&desc), "a null color slice needs length 0");

    desc.colors = colors[:1];
    desc.height = 0;
    assert(!gpu::internal::render_pass_desc_valid(&desc), "exactly one zero dimension");

    desc.width = 0;
    assert(gpu::internal::render_pass_desc_valid(&desc), "a zero extent with a color attachment");
    desc.colors = {};
    assert(!gpu::internal::render_pass_desc_valid(&desc), "a zero extent needs an attachment");

    DepthTargetDesc depth = { .resolve_view = { .owner = 1, .generation = 1 }, .resolve_mode = bitcast(9, DepthResolveMode) };
    desc.depth = &depth;
    assert(!gpu::internal::render_pass_desc_valid(&desc), "a depth resolve needs a defined mode");
    depth.resolve_mode = DepthResolveMode.MIN;
    assert(gpu::internal::render_pass_desc_valid(&desc));
}

fn void color_state_rules_reject_each_clause() @test {
    ColorTargetState[MAX_COLOR_ATTACHMENTS + 1] targets;
    ColorState state = { .targets = targets[:MAX_COLOR_ATTACHMENTS] };
    assert(gpu::internal::color_state_valid(&state));
    state.targets = targets[:MAX_COLOR_ATTACHMENTS + 1];
    assert(!gpu::internal::color_state_valid(&state), "more targets than the fixed array holds");
    state.targets = ((ColorTargetState*)null)[:1];
    assert(!gpu::internal::color_state_valid(&state), "a null target slice needs length 0");
}

fn void inline_root_rules_reject_each_clause() @test {
    char[INLINE_ROOT_CAPACITY + 4] bytes;
    assert(gpu::internal::inline_root_valid({}));
    assert(gpu::internal::inline_root_valid(bytes[:INLINE_ROOT_CAPACITY]));
    assert(!gpu::internal::inline_root_valid(bytes[:INLINE_ROOT_CAPACITY + 4]), "one word over capacity");
    assert(!gpu::internal::inline_root_valid(bytes[:INLINE_ROOT_CAPACITY - 2]), "not a multiple of four");
    assert(!gpu::internal::inline_root_valid(bytes[:3]), "not a multiple of four");
}

fn void command_allocator_rules_reject_each_clause() @test {
    CommandAllocatorDesc desc = { .max_acceleration_structure_geometries_per_build = 4 };
    assert(gpu::internal::command_allocator_geometry_capacity_fits(&desc), "the default capacity fits");
    desc.command_buffer_capacity = uint::max / 4;
    assert(gpu::internal::command_allocator_geometry_capacity_fits(&desc));
    desc.command_buffer_capacity = uint::max / 4 + 1;
    assert(!gpu::internal::command_allocator_geometry_capacity_fits(&desc), "the product overflows uint");
    desc.max_acceleration_structure_geometries_per_build = 0;
    assert(gpu::internal::command_allocator_geometry_capacity_fits(&desc), "no geometry capacity needs no product");
    desc.max_acceleration_structure_geometries_per_build = uint::max;
    desc.command_buffer_capacity = 0;
    assert(!gpu::internal::command_allocator_geometry_capacity_fits(&desc), "the defaulted capacity counts");
}

fn void submit_desc_rules_reject_each_clause() @test {
    assert(!gpu::internal::submit_desc_valid(null), "a null descriptor is invalid");
    SubmitDesc empty;
    assert(gpu::internal::submit_desc_valid(&empty), "the zero descriptor is valid");

    ExecutableCommandList[MAX_SUBMIT_COMMAND_LISTS + 1] lists;
    SubmitDesc at_limit = { .command_lists = lists[:MAX_SUBMIT_COMMAND_LISTS] };
    assert(gpu::internal::submit_desc_valid(&at_limit), "the list limit itself is allowed");
    SubmitDesc too_many_lists = { .command_lists = lists[:MAX_SUBMIT_COMMAND_LISTS + 1] };
    assert(!gpu::internal::submit_desc_valid(&too_many_lists), "one list over the limit");
    SubmitDesc null_lists = { .command_lists = ((ExecutableCommandList*)null)[:1] };
    assert(!gpu::internal::submit_desc_valid(&null_lists), "a null list slice needs length 0");

    CompletionWait[MAX_SUBMIT_COMPLETION_WAITS + 1] waits;
    foreach (&wait : waits) wait.before = { .compute };
    SubmitDesc waits_at_limit = { .completion_waits = waits[:MAX_SUBMIT_COMPLETION_WAITS] };
    assert(gpu::internal::submit_desc_valid(&waits_at_limit), "the wait limit itself is allowed");
    SubmitDesc too_many_waits = { .completion_waits = waits[:MAX_SUBMIT_COMPLETION_WAITS + 1] };
    assert(!gpu::internal::submit_desc_valid(&too_many_waits), "one wait over the limit");
    SubmitDesc null_waits = { .completion_waits = ((CompletionWait*)null)[:1] };
    assert(!gpu::internal::submit_desc_valid(&null_waits), "a null wait slice needs length 0");

    CompletionWait[1] bad = {{ .before = { .host } }};
    SubmitDesc host_wait = { .completion_waits = bad[..] };
    assert(!gpu::internal::submit_desc_valid(&host_wait), "a wait stage excludes host");
    bad[0].before = { .present };
    assert(!gpu::internal::submit_desc_valid(&host_wait), "a wait stage excludes present");
    bad[0].before = {};
    assert(!gpu::internal::submit_desc_valid(&host_wait), "a wait stage must be nonempty");
    bad[0].before = { .compute };
    assert(gpu::internal::submit_desc_valid(&host_wait));
}
