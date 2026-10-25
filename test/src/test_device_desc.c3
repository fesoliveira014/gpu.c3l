module gpu_test;

import gpu::internal @public;
import gpu @public;

fn bool queue_requests_equal(
    gpu::QueueRequest first,
    gpu::QueueRequest second,
) {
    return (uint)first.required == (uint)second.required
        && (uint)first.distinct_roles == (uint)second.distinct_roles
        && first.single_queue == second.single_queue;
}

fn bool device_desc_equal(gpu::DeviceDesc first, gpu::DeviceDesc second) {
    return (ulong)first.surface == (ulong)second.surface
        && queue_requests_equal(first.queues, second.queues)
        && first.enable_sparse_textures == second.enable_sparse_textures
        && first.enable_ray_queries == second.enable_ray_queries
        && first.enable_ray_tracing_pipelines
            == second.enable_ray_tracing_pipelines;
}

fn void null_and_zero_device_desc_use_default_queues() @test {
    gpu::DeviceDesc null_normalized =
        gpu::internal::normalize_device_desc(null);
    gpu::DeviceDesc zero;
    gpu::DeviceDesc zero_normalized =
        gpu::internal::normalize_device_desc(&zero);

    assert(null_normalized.queues.required.graphics);
    assert(null_normalized.queues.required.compute);
    assert(null_normalized.queues.required.transfer);
    assert((uint)null_normalized.queues.distinct_roles == 0);
    assert(!null_normalized.queues.single_queue);
    assert(!zero_normalized.queues.single_queue);
    assert(queue_requests_equal(
        zero_normalized.queues,
        null_normalized.queues,
    ));
    assert(!null_normalized.surface.is_valid());
    assert(!zero_normalized.surface.is_valid());
    assert(!null_normalized.enable_sparse_textures);
    assert(!zero_normalized.enable_sparse_textures);
    assert(!null_normalized.enable_ray_queries);
    assert(!zero_normalized.enable_ray_queries);
    assert(!null_normalized.enable_ray_tracing_pipelines);
    assert(!zero_normalized.enable_ray_tracing_pipelines);
}

fn void normalization_preserves_explicit_desc_without_mutation() @test {
    gpu::DeviceDesc desc = {
        .surface = {
            .index      = 7,
            .generation = 11,
        },
        .queues = {
            .required = { .graphics, .compute },
            .distinct_roles = { .compute },
            .single_queue = false,
        },
        .enable_sparse_textures = true,
        .enable_ray_queries = true,
        .enable_ray_tracing_pipelines = true,
    };
    gpu::DeviceDesc original = desc;

    gpu::DeviceDesc normalized =
        gpu::internal::normalize_device_desc(&desc);

    assert(device_desc_equal(desc, original));
    assert(device_desc_equal(normalized, original));
}
