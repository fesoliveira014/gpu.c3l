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

fn void subspan_bounds_and_nesting() @test {
    GpuSpan span = {
        .owner      = 1,
        .index      = 7,
        .generation = 3,
        .size       = 256,
    };

    GpuSpan parent = span.subspan(32, 128);
    GpuSpan exact = parent.subspan(96, 32);

    assert(exact.offset == 128, "nested backing offset advances from each parent");
    assert(exact.size == 32, "exact-fit nested slice keeps the requested size");
    assert(exact.owner == span.owner, "nested slice inherits owner identity");
    assert(exact.index == span.index, "nested slice inherits slot identity");
    assert(exact.generation == span.generation, "nested slice inherits generation identity");
}

fn void subspan_accepts_the_last_byte_and_the_largest_base_offset() @test {
    GpuSpan span = {
        .owner      = 1,
        .generation = 1,
        .size       = 64,
        .offset     = usz::max - 63,
    };

    GpuSpan last = span.subspan(63, 1);
    assert(last.offset == usz::max && last.size == 1,
        "a subspan may end exactly at the largest representable offset");
}
