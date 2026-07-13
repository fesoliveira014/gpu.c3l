module gpu_test;

import gpu @public;

fn void dispatch_groups_accept_zero_and_exact_limit() @test {
    gpu::Vec3u limit = { .x = 8, .y = 4, .z = 2 };

    gpu::validate_dispatch_groups({}, limit)!!;
    gpu::validate_dispatch_groups(limit, limit)!!;
}

fn void dispatch_groups_reject_each_over_limit_axis() @test {
    gpu::Vec3u limit = { .x = 8, .y = 4, .z = 2 };

    fault x_fault = @catch(gpu::validate_dispatch_groups({ .x = 9 }, limit));
    fault y_fault = @catch(gpu::validate_dispatch_groups({ .y = 5 }, limit));
    fault z_fault = @catch(gpu::validate_dispatch_groups({ .z = 3 }, limit));

    assert(x_fault == gpu::INVALID_ARGUMENT, "dispatch X above the selected-device limit faults");
    assert(y_fault == gpu::INVALID_ARGUMENT, "dispatch Y above the selected-device limit faults");
    assert(z_fault == gpu::INVALID_ARGUMENT, "dispatch Z above the selected-device limit faults");
}

fn void indirect_arguments_accept_zero_and_exact_limit() @test {
    gpu::GpuSpan empty = {};
    gpu::GpuSpan exact = { .size = 64 };

    gpu::validate_indirect_arguments(
        args:          &empty,
        argument_size: 16,
        count:         0,
        max_count:     4,
    )!!;
    gpu::validate_indirect_arguments(
        args:          &exact,
        argument_size: 16,
        count:         4,
        max_count:     4,
    )!!;
}

fn void indirect_arguments_reject_count_size_and_arithmetic_overflow() @test {
    gpu::GpuSpan exact = { .size = 64 };
    gpu::GpuSpan undersized = { .size = 63 };

    fault count_fault = @catch(gpu::validate_indirect_arguments(
        args:          &exact,
        argument_size: 16,
        count:         5,
        max_count:     4,
    ));
    fault size_fault = @catch(gpu::validate_indirect_arguments(
        args:          &undersized,
        argument_size: 16,
        count:         4,
        max_count:     4,
    ));
    fault overflow_fault = @catch(gpu::validate_indirect_arguments(
        args:          &exact,
        argument_size: usz::max,
        count:         2,
        max_count:     2,
    ));

    assert(count_fault == gpu::INVALID_ARGUMENT, "indirect count above the selected-device limit faults");
    assert(size_fault == gpu::INVALID_ARGUMENT, "undersized indirect argument span faults");
    assert(overflow_fault == gpu::INVALID_ARGUMENT, "indirect argument byte overflow faults");
}
