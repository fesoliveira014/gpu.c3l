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
