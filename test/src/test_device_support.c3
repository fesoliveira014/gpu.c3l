module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;

fn RuntimeDesc device_support_runtime_desc() {
    return {
        .application_name = "gpu_device_desc_support_test",
    };
}

fn Adapter first_support_adapter(Runtime* runtime) {
    return gpu::enumerate_adapters(runtime)!!.get(0)!!;
}

fn void descriptor_support_is_semantic_and_does_not_enable_state() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceDesc desc;
    DeviceDesc original = desc;

    DeviceSupport support = gpu::supports_device_desc(&adapter, &desc)!!;

    assert(support.supported, "the device baseline is supported");
    assert(support.unmet_requirement.len == 0);
    assert(
        device_desc_equal(desc, original),
        "support detection does not mutate the descriptor"
    );
    assert(gpu::internal::device_slot_available());
    assert(gpu::internal::vk::stub_device_support_query_count() == 0);
}

fn void unsupported_baseline_names_the_semantic_requirement() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(false);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);

    DeviceSupport support = gpu::supports_device_desc(&adapter)!!;

    assert(!support.supported);
    assert(support.unmet_requirement == "device baseline");
}

fn void invalid_desc_and_stale_adapter_fault_before_backend_work() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceDesc invalid = {
        .queues = {
            .required = { .graphics },
            .distinct = { .compute },
        },
    };

    assert(@catch(gpu::supports_device_desc(&adapter, &invalid))
        == gpu::INVALID_ARGUMENT);
    assert(gpu::internal::vk::stub_device_support_query_count() == 0);

    gpu::destroy_runtime(&runtime)!!;
    assert(@catch(gpu::supports_device_desc(&adapter)) == gpu::INVALID_HANDLE);
    assert(gpu::internal::vk::stub_device_support_query_count() == 0);
}

fn void ray_query_opt_in_requires_a_configured_runtime_heap() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceDesc desc = { .enable_ray_queries = true };

    assert(@catch(gpu::supports_device_desc(&adapter, &desc))
        == gpu::INVALID_ARGUMENT);
    assert(gpu::internal::vk::stub_device_support_query_count() == 0,
        "incoherent opt-in faults before backend capability work");
}

fn void coherent_ray_query_opt_in_reaches_capability_evaluation() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    runtime_desc.acceleration_structure_heap_capacity = 17;
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceDesc desc = { .enable_ray_queries = true };

    DeviceSupport support = gpu::supports_device_desc(&adapter, &desc)!!;

    assert(support.supported);

    Device device = gpu::create_device(&adapter, &desc)!!;
    DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(caps.ray_queries.enabled
            && caps.acceleration_structures.heap_capacity == 17
            && caps.acceleration_structures.max_geometry_count == uint::max
            && caps.acceleration_structures.max_primitive_count == uint::max
            && caps.acceleration_structures.max_instance_count == uint::max
            && caps.acceleration_structures.scratch_alignment == 256,
        "the public device fixture exposes the enabled actionable caps");
    gpu::destroy_device(&device)!!;
}

fn void ray_tracing_pipeline_opt_in_is_independent_and_actionable() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_set_ray_query_support(false);
    gpu::internal::vk::stub_set_ray_tracing_pipeline_support(true);
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    runtime_desc.acceleration_structure_heap_capacity = 9;
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceDesc desc = { .enable_ray_tracing_pipelines = true };

    assert(gpu::supports_device_desc(&adapter, &desc)!!.supported);
    Device device = gpu::create_device(&adapter, &desc)!!;
    DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(caps.acceleration_structures.enabled
            && caps.acceleration_structures.heap_capacity == 9);
    assert(!caps.ray_queries.enabled);
    assert(caps.ray_tracing_pipelines.enabled
            && caps.ray_tracing_pipelines.max_recursion_depth == 4);
    Queue queue = gpu::get_queue(&device, QueueKind.GRAPHICS)!!;
    CommandAllocatorDesc allocator_desc = {
        .max_acceleration_structure_geometries_per_build = 1,
    };
    CommandAllocator allocator = gpu::create_command_allocator(
        device: &device,
        queue:  queue,
        desc:   &allocator_desc,
    )!!;
    gpu::destroy_command_allocator(&allocator)!!;
    gpu::destroy_device(&device)!!;
    gpu::internal::vk::stub_set_ray_query_support(true);
}

fn void ray_tracing_pipeline_support_reports_frontend_and_queue_failures() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    runtime_desc.acceleration_structure_heap_capacity = 1;
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceDesc desc = { .enable_ray_tracing_pipelines = true };

    gpu::internal::vk::stub_set_ray_tracing_pipeline_support(false);
    DeviceSupport unsupported = gpu::supports_device_desc(&adapter, &desc)!!;
    assert(!unsupported.supported
            && unsupported.unmet_requirement == "ray tracing pipelines");

    gpu::internal::vk::stub_set_ray_tracing_pipeline_support(true);
    desc.queues.required = { .transfer };
    DeviceSupport wrong_queue = gpu::supports_device_desc(&adapter, &desc)!!;
    assert(!wrong_queue.supported
            && wrong_queue.unmet_requirement == "ray tracing queue topology");
}

fn void unavailable_ray_queries_are_reported_semantically() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_set_ray_query_support(false);
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    runtime_desc.acceleration_structure_heap_capacity = 1;
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceDesc desc = { .enable_ray_queries = true };

    DeviceSupport support = gpu::supports_device_desc(&adapter, &desc)!!;
    assert(!support.supported);
    assert(support.unmet_requirement == "ray queries");
    assert(@catch(gpu::create_device(&adapter, &desc))
        == gpu::UNSUPPORTED_FEATURE);

    gpu::internal::vk::stub_set_ray_query_support(true);
}

fn void oversized_ray_query_heap_is_reported_semantically() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_set_ray_query_heap_limit(16);
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    runtime_desc.acceleration_structure_heap_capacity = 17;
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceDesc desc = { .enable_ray_queries = true };

    DeviceSupport support = gpu::supports_device_desc(&adapter, &desc)!!;
    assert(!support.supported);
    assert(support.unmet_requirement
        == "acceleration structure heap capacity");
    assert(@catch(gpu::create_device(&adapter, &desc))
        == gpu::UNSUPPORTED_FEATURE);

    gpu::internal::vk::stub_set_ray_query_heap_limit(uint::max);
}

fn void unavailable_queue_topology_is_reported_semantically() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceDesc desc = {
        .queues = {
            .required = { .graphics, .compute },
            .distinct = { .compute },
        },
    };

    DeviceSupport support = gpu::supports_device_desc(&adapter, &desc)!!;

    assert(!support.supported);
    assert(support.unmet_requirement == "queue topology");
}

fn void unavailable_sparse_textures_are_reported_semantically() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceDesc desc = {
        .enable_sparse_textures = true,
    };

    DeviceSupport support = gpu::supports_device_desc(&adapter, &desc)!!;

    assert(!support.supported);
    assert(support.unmet_requirement == "sparse textures");
    assert(@catch(gpu::create_device(&adapter, &desc))
        == gpu::UNSUPPORTED_FEATURE);
}

fn void sparse_queue_requirement_reaches_backend_support_query() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_device_baseline_support(true);
    gpu::internal::vk::stub_set_sparse_texture_support(true);
    gpu::internal::vk::stub_set_sparse_queue_support(false);
    gpu::internal::vk::stub_reset_device_counts();
    RuntimeDesc runtime_desc = device_support_runtime_desc();
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = first_support_adapter(&runtime);
    DeviceDesc sparse = {
        .enable_sparse_textures = true,
    };

    DeviceSupport sparse_support =
        gpu::supports_device_desc(&adapter, &sparse)!!;
    assert(!sparse_support.supported);
    assert(sparse_support.unmet_requirement == "sparse queue topology");

    DeviceDesc ordinary;
    DeviceSupport ordinary_support =
        gpu::supports_device_desc(&adapter, &ordinary)!!;
    assert(ordinary_support.supported,
        "ordinary support does not require a sparse-capable selected queue");
}
