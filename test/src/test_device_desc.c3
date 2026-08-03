module gpu_test;

import gpu::internal @public;
import gpu @public;

fn bool queue_requests_equal(
    gpu::QueueRequest first,
    gpu::QueueRequest second,
) {
    return (uint)first.required == (uint)second.required
        && (uint)first.distinct == (uint)second.distinct;
}

fn bool device_desc_equal(gpu::DeviceDesc first, gpu::DeviceDesc second) {
    return (ulong)first.surface == (ulong)second.surface
        && queue_requests_equal(first.queues, second.queues)
        && first.enable_sparse_textures == second.enable_sparse_textures
        && first.enable_ray_queries == second.enable_ray_queries;
}

fn void null_and_zero_device_desc_use_default_queues() @test {
    gpu::DeviceDesc null_normalized =
        gpu::internal::normalize_device_desc(null)!!;
    gpu::DeviceDesc zero;
    gpu::DeviceDesc zero_normalized =
        gpu::internal::normalize_device_desc(&zero)!!;

    assert(null_normalized.queues.required.graphics);
    assert(null_normalized.queues.required.compute);
    assert(null_normalized.queues.required.transfer);
    assert((uint)null_normalized.queues.distinct == 0);
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
}

fn void normalization_preserves_explicit_desc_without_mutation() @test {
    gpu::DeviceDesc desc = {
        .surface = {
            .index      = 7,
            .generation = 11,
        },
        .queues = {
            .required = { .graphics, .compute },
            .distinct = { .compute },
        },
        .enable_sparse_textures = true,
        .enable_ray_queries = true,
    };
    gpu::DeviceDesc original = desc;

    gpu::DeviceDesc normalized =
        gpu::internal::normalize_device_desc(&desc)!!;

    assert(device_desc_equal(desc, original));
    assert(device_desc_equal(normalized, original));
}

fn void normalization_rejects_invalid_queue_descriptions() @test {
    gpu::DeviceDesc missing_role = {
        .queues = {
            .required = { .graphics },
            .distinct = { .compute },
        },
    };
    gpu::DeviceDesc unknown_role = {
        .queues = {
            .required = (gpu::QueueRoles)8,
        },
    };
    gpu::DeviceDesc unknown_distinct = {
        .queues = {
            .required = { .graphics },
            .distinct = (gpu::QueueRoles)8,
        },
    };
    gpu::DeviceDesc presentation_without_graphics = {
        .surface = {
            .index      = 1,
            .generation = 1,
        },
        .queues = { .required = { .compute } },
    };

    assert(@catch(gpu::internal::normalize_device_desc(&missing_role))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::internal::normalize_device_desc(&unknown_role))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::internal::normalize_device_desc(&unknown_distinct))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::internal::normalize_device_desc(
        &presentation_without_graphics,
    )) == gpu::INVALID_ARGUMENT);
}
