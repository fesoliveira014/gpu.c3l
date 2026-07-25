module gpu_test;

import gpu::internal @public;
import gpu @public;

fn bool queue_requirements_equal(
    gpu::QueueRequirements first,
    gpu::QueueRequirements second,
) {
    return first.counts.graphics == second.counts.graphics
        && first.counts.compute == second.counts.compute
        && first.counts.transfer == second.counts.transfer
        && (uint)first.distinct == (uint)second.distinct;
}

fn bool device_desc_equal(gpu::DeviceDesc first, gpu::DeviceDesc second) {
    return (ulong)first.surface == (ulong)second.surface
        && queue_requirements_equal(first.queues, second.queues);
}

fn void null_and_zero_device_desc_use_default_queues() @test {
    gpu::DeviceDesc null_normalized =
        gpu::internal::normalize_device_desc(null)!!;
    gpu::DeviceDesc zero;
    gpu::DeviceDesc zero_normalized =
        gpu::internal::normalize_device_desc(&zero)!!;

    assert(null_normalized.queues.counts.graphics == 1);
    assert(null_normalized.queues.counts.compute == 1);
    assert(null_normalized.queues.counts.transfer == 1);
    assert((uint)null_normalized.queues.distinct == 0);
    assert(queue_requirements_equal(
        zero_normalized.queues,
        null_normalized.queues,
    ));
    assert(!null_normalized.surface.is_valid());
    assert(!zero_normalized.surface.is_valid());
}

fn void normalization_preserves_explicit_desc_without_mutation() @test {
    gpu::DeviceDesc desc = {
        .surface = {
            .index      = 7,
            .generation = 11,
        },
        .queues = {
            .counts   = {
                .graphics = 1,
                .compute  = 1,
            },
            .distinct = { .compute },
        },
    };
    gpu::DeviceDesc original = desc;

    gpu::DeviceDesc normalized =
        gpu::internal::normalize_device_desc(&desc)!!;

    assert(device_desc_equal(desc, original));
    assert(device_desc_equal(normalized, original));
}

fn void normalization_rejects_invalid_queue_descriptions() @test {
    gpu::DeviceDesc too_many = {
        .queues = { .counts = { .graphics = 256 } },
    };
    gpu::DeviceDesc missing_role = {
        .queues = {
            .counts   = { .graphics = 1 },
            .distinct = { .compute },
        },
    };
    gpu::DeviceDesc unknown_role = {
        .queues = {
            .counts   = { .graphics = 1 },
            .distinct = (gpu::QueueRoles)8,
        },
    };
    gpu::DeviceDesc presentation_without_graphics = {
        .surface = {
            .index      = 1,
            .generation = 1,
        },
        .queues = { .counts = { .compute = 1 } },
    };

    assert(@catch(gpu::internal::normalize_device_desc(&too_many))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::internal::normalize_device_desc(&missing_role))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::internal::normalize_device_desc(&unknown_role))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::internal::normalize_device_desc(
        &presentation_without_graphics,
    )) == gpu::INVALID_ARGUMENT);
}
