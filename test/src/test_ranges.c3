module gpu_test;

import gpu;

fn void unchecked_subspan_math() @test {
    GpuSpan span = {
        .owner      = 5,
        .index      = 7,
        .generation = 3,
        .size       = 256,
    };

    GpuSpan sub = span.unchecked_subspan(64, 32);
    assert(sub.size == 32, "size is the requested size");
    assert(sub.offset == 64, "offset accumulates");
    assert(sub.owner == span.owner, "owner identity is inherited");
    assert(sub.index == span.index, "slot identity is inherited");
    assert(sub.generation == span.generation, "generation identity is inherited");
}

fn void checked_subspan_bounds_and_nesting() @test {
    GpuSpan span = {
        .owner      = 1,
        .index      = 7,
        .generation = 3,
        .size       = 256,
    };

    GpuSpan parent = span.checked_subspan(32, 128)!!;
    GpuSpan exact = parent.checked_subspan(96, 32)!!;
    fault nested_escape = @catch(parent.checked_subspan(97, 32));
    fault offset_past_parent = @catch(span.checked_subspan(257, 1));
    fault size_past_parent = @catch(span.checked_subspan(200, 57));
    fault overflowing_size = @catch(span.checked_subspan(1, usz::max));

    assert(exact.offset == 128, "nested backing offset advances from each parent");
    assert(exact.size == 32, "exact-fit nested slice keeps the requested size");
    assert(exact.owner == span.owner, "nested slice inherits owner identity");
    assert(exact.index == span.index, "nested slice inherits slot identity");
    assert(exact.generation == span.generation, "nested slice inherits generation identity");
    assert(nested_escape == gpu::INVALID_ARGUMENT,
        "nested slice cannot escape its immediate parent");
    assert(offset_past_parent == gpu::INVALID_ARGUMENT, "offset beyond parent faults");
    assert(size_past_parent == gpu::INVALID_ARGUMENT, "size beyond parent remainder faults");
    assert(overflowing_size == gpu::INVALID_ARGUMENT,
        "maximum size cannot wrap through validation");
}

fn void checked_subspan_rejects_zero_size() @test {
    GpuSpan span = {
        .owner      = 1,
        .generation = 1,
        .size       = 64,
    };

    fault middle = @catch(span.checked_subspan(16, 0));
    fault endpoint = @catch(span.checked_subspan(64, 0));
    assert(middle == gpu::INVALID_ARGUMENT, "zero-size middle slice faults");
    assert(endpoint == gpu::INVALID_ARGUMENT, "zero-size endpoint slice faults");
}

fn void checked_subspan_rejects_metadata_overflow() @test {
    GpuSpan offset_overflow = {
        .owner      = 1,
        .generation = 1,
        .size       = 2,
        .offset     = usz::max,
    };

    fault offset_fault = @catch(offset_overflow.checked_subspan(1, 1));
    assert(offset_fault == gpu::INVALID_ARGUMENT, "backing offset addition cannot wrap");
}
