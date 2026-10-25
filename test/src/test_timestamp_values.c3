module gpu_test;

import gpu @public;

struct TimestampDeltaCase {
    uint   bits;
    ulong  begin;
    ulong  end;
    float  period;
    double expected;
}

const TimestampDeltaCase[] TIMESTAMP_DELTA_CASES = {
    { .bits = 64, .begin = 10, .end = 25, .period = 2.0f, .expected = 30.0 },
    { .bits = 8,  .begin = 0x103, .end = 0x108, .period = 1.0f, .expected = 5.0 },
    { .bits = 8,  .begin = 250, .end = 4, .period = 0.5f, .expected = 5.0 },
    {
        .bits     = 36,
        .begin    = ((ulong)1 << 36) - 3,
        .end      = 2,
        .period   = 1.0f,
        .expected = 5.0,
    },
    {
        .bits     = 64,
        .begin    = ulong::max - 2,
        .end      = 2,
        .period   = 1.0f,
        .expected = 5.0,
    },
};

fn void timestamp_public_values_preserve_identity_and_layout() @test {
    TimestampPoolHandle handle = {
        .owner      = 7,
        .index      = 3,
        .generation = 11,
    };
    TimestampPoolHandle different_owner = handle;
    TimestampPoolHandle different_index = handle;
    TimestampPoolHandle different_generation = handle;
    different_owner.owner++;
    different_index.index++;
    different_generation.generation++;

    assert(TimestampPoolHandle::size <= ulong::size * 2,
        "timestamp pool handles fit in two machine words");
    assert(handle.is_valid(), "nonzero owner and generation are valid");
    assert(handle == handle, "identical timestamp pool identity compares equal");
    assert(handle != different_owner, "owner participates in timestamp pool identity");
    assert(handle != different_index, "index participates in timestamp pool identity");
    assert(handle != different_generation,
        "generation participates in timestamp pool identity");
    assert(!gpu::TIMESTAMP_POOL_HANDLE_INVALID.is_valid(),
        "the zero timestamp pool handle is invalid");

    TimestampPoolDesc desc = {
        .capacity   = 8,
        .debug_name = "frame timestamps",
    };
    TimestampCaps timestamps = {
        .queues = {
            .graphics = true,
            .compute  = true,
        },
        .graphics_valid_bits = 64,
        .compute_valid_bits  = 36,
        .period_ns           = 0.5f,
    };
    DeviceCaps caps = { .timestamps = timestamps };
    assert(desc.capacity == 8 && desc.debug_name != null,
        "timestamp pool descriptors expose capacity and debug name");
    assert(caps.timestamps.queues.graphics
        && caps.timestamps.queues.compute
        && caps.timestamps.graphics_valid_bits == 64
        && caps.timestamps.compute_valid_bits == 36
        && caps.timestamps.transfer_valid_bits == 0
        && caps.timestamps.period_ns == 0.5f,
        "device capabilities preserve the complete timestamp schema");
    assert((int)DebugResourceKind.ATTACHMENT_VIEW == 12,
        "the last pre-existing debug kind keeps its value");
    assert((int)DebugResourceKind.TIMESTAMP_POOL == 13,
        "timestamp pools append to the debug resource vocabulary");
}

fn void timestamp_delta_converts_and_wraps() @test {
    foreach (test_case : TIMESTAMP_DELTA_CASES) {
        TimestampCaps caps = {
            .queues.graphics     = true,
            .graphics_valid_bits = test_case.bits,
            .period_ns           = test_case.period,
        };
        double elapsed = gpu::timestamp_delta_ns(
            caps:  &caps,
            queue: QueueKind.GRAPHICS,
            begin: test_case.begin,
            end:   test_case.end,
        )!!;
        assert(elapsed == test_case.expected,
            "timestamp conversion masks, wraps, and scales the counter");
    }
}

fn void timestamp_delta_selects_the_requested_role_width() @test {
    TimestampCaps caps = {
        .queues = {
            .graphics = true,
            .compute  = true,
            .transfer = true,
        },
        .graphics_valid_bits = 8,
        .compute_valid_bits  = 16,
        .transfer_valid_bits = 36,
        .period_ns           = 1.0f,
    };

    double graphics = gpu::timestamp_delta_ns(
        caps:  &caps,
        queue: QueueKind.GRAPHICS,
        begin: 250,
        end:   4,
    )!!;
    double compute = gpu::timestamp_delta_ns(
        caps:  &caps,
        queue: QueueKind.COMPUTE,
        begin: 65_530,
        end:   4,
    )!!;
    double transfer = gpu::timestamp_delta_ns(
        caps:  &caps,
        queue: QueueKind.TRANSFER,
        begin: ((ulong)1 << 36) - 6,
        end:   4,
    )!!;

    assert(graphics == 10.0, "graphics conversion uses graphics valid bits");
    assert(compute == 10.0, "compute conversion uses compute valid bits");
    assert(transfer == 10.0, "transfer conversion uses transfer valid bits");
}

fn void timestamp_delta_rejects_unsupported_roles() @test {
    TimestampCaps caps = {
        .queues.graphics     = true,
        .graphics_valid_bits = 64,
        .period_ns           = 1.0f,
    };

    fault compute_fault = @catch(gpu::timestamp_delta_ns(
        caps:  &caps,
        queue: QueueKind.COMPUTE,
        begin: 0,
        end:   1,
    ));
    fault transfer_fault = @catch(gpu::timestamp_delta_ns(
        caps:  &caps,
        queue: QueueKind.TRANSFER,
        begin: 0,
        end:   1,
    ));

    assert(compute_fault == gpu::UNSUPPORTED_FEATURE,
        "an unsupported compute role returns UNSUPPORTED_FEATURE");
    assert(transfer_fault == gpu::UNSUPPORTED_FEATURE,
        "an unsupported transfer role returns UNSUPPORTED_FEATURE");
}

fn void timestamp_caps_rules_reject_each_clause() @test {
    TimestampCaps zero_width = {
        .queues.graphics = true,
        .period_ns       = 1.0f,
    };
    TimestampCaps wide = {
        .queues.graphics     = true,
        .graphics_valid_bits = 65,
        .period_ns           = 1.0f,
    };
    TimestampCaps zero_period = {
        .queues.graphics     = true,
        .graphics_valid_bits = 64,
    };
    TimestampCaps negative_period = {
        .queues.graphics     = true,
        .graphics_valid_bits = 64,
        .period_ns           = -1.0f,
    };
    TimestampCaps valid = {
        .queues.graphics     = true,
        .graphics_valid_bits = 64,
        .period_ns           = 1.0f,
    };
    int unknown_queue = 99;

    assert(gpu::internal::timestamp_caps_valid(&valid, QueueKind.GRAPHICS));
    assert(!gpu::internal::timestamp_caps_valid(null, QueueKind.GRAPHICS), "null capabilities");
    assert(!gpu::internal::timestamp_caps_valid(&zero_width, QueueKind.GRAPHICS), "zero valid bits");
    assert(!gpu::internal::timestamp_caps_valid(&wide, QueueKind.GRAPHICS), "more than 64 valid bits");
    assert(!gpu::internal::timestamp_caps_valid(&zero_period, QueueKind.GRAPHICS), "zero period");
    assert(!gpu::internal::timestamp_caps_valid(&negative_period, QueueKind.GRAPHICS), "negative period");
    assert(gpu::internal::timestamp_caps_valid(&valid, QueueKind.COMPUTE),
        "an unsupported role is left to its fault");
    assert(!gpu::internal::queue_kind_valid(bitcast(unknown_queue, QueueKind)), "undefined queue kind");
    assert(!gpu::internal::timestamp_caps_valid(&valid, bitcast(unknown_queue, QueueKind)));
}
