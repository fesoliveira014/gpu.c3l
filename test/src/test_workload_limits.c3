module gpu_test;

import gpu::internal @public;
import gpu @public;

fn void dispatch_groups_accept_zero_and_exact_limit() @test {
    gpu::Vec3u limit = { .x = 8, .y = 4, .z = 2 };

    gpu::internal::validate_dispatch_groups({}, limit)!!;
    gpu::internal::validate_dispatch_groups(limit, limit)!!;
}

fn void dispatch_groups_reject_each_over_limit_axis() @test {
    gpu::Vec3u limit = { .x = 8, .y = 4, .z = 2 };

    fault x_fault = @catch(gpu::internal::validate_dispatch_groups({ .x = 9 }, limit));
    fault y_fault = @catch(gpu::internal::validate_dispatch_groups({ .y = 5 }, limit));
    fault z_fault = @catch(gpu::internal::validate_dispatch_groups({ .z = 3 }, limit));

    assert(x_fault == gpu::INVALID_ARGUMENT, "dispatch X above the selected-device limit faults");
    assert(y_fault == gpu::INVALID_ARGUMENT, "dispatch Y above the selected-device limit faults");
    assert(z_fault == gpu::INVALID_ARGUMENT, "dispatch Z above the selected-device limit faults");
}

fn void indirect_argument_bytes_accept_zero_and_exact_limit() @test {
    usz empty = gpu::internal::indirect_argument_bytes(
        argument_size: 16,
        count:         0,
        max_count:     4,
    )!!;
    usz exact = gpu::internal::indirect_argument_bytes(
        argument_size: 16,
        count:         4,
        max_count:     4,
    )!!;

    assert(empty == 0, "zero count needs no argument bytes");
    assert(exact == 64, "exact-limit count returns its required bytes");
}

fn void indirect_argument_bytes_reject_count_and_arithmetic_overflow() @test {
    fault count_fault = @catch(gpu::internal::indirect_argument_bytes(
        argument_size: 16,
        count:         5,
        max_count:     4,
    ));
    fault overflow_fault = @catch(gpu::internal::indirect_argument_bytes(
        argument_size: usz::max,
        count:         2,
        max_count:     2,
    ));

    assert(count_fault == gpu::INVALID_ARGUMENT, "indirect count above the selected-device limit faults");
    assert(overflow_fault == gpu::INVALID_ARGUMENT, "indirect argument byte overflow faults");
}

fn void generated_work_bytes_accept_zero_and_exact_limit() @test {
    usz empty = gpu::internal::generated_work_bytes(
        record_size: gpu::GeneratedDrawRecord::size,
        count:       0,
        max_count:   4,
    )!!;
    usz exact = gpu::internal::generated_work_bytes(
        record_size: gpu::GeneratedDrawIndexedRecord::size,
        count:       4,
        max_count:   4,
    )!!;

    assert(empty == 0, "zero generated-work count needs no record bytes");
    assert(exact == 160, "exact-limit generated-work count returns its required bytes");
}

fn void generated_work_bytes_reject_count_and_arithmetic_overflow() @test {
    fault count_fault = @catch(gpu::internal::generated_work_bytes(
        record_size: gpu::GeneratedDispatchRecord::size,
        count:       5,
        max_count:   4,
    ));
    fault overflow_fault = @catch(gpu::internal::generated_work_bytes(
        record_size: usz::max,
        count:       2,
        max_count:   2,
    ));

    assert(count_fault == gpu::INVALID_ARGUMENT, "generated-work count above the selected-device limit faults");
    assert(overflow_fault == gpu::INVALID_ARGUMENT, "generated-work record byte overflow faults");
}
