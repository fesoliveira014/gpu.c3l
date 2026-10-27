module gpu_test;

import gpu;

fn void mapped_gpu_span_slicing_stays_consistent() @test {
    char[64] storage;
    MappedGpuSpan mapped = {
        .span = {
            .owner      = 5,
            .index      = 7,
            .generation = 3,
            .offset     = 32,
            .size       = storage.len,
        },
        .bytes   = storage[..],
        .address = (gpu::GpuAddress)0x1000,
    };

    MappedGpuSpan child = mapped.subspan(8, 32);
    MappedGpuSpan nested = child.subspan(12, 8);
    MappedGpuSpan exact_end = child.subspan(24, 8);

    assert(child.span.offset == 40 && child.span.size == 32,
        "child span advances from the backing span");
    assert(child.bytes.ptr == &storage[8] && child.bytes.len == 32,
        "child bytes select the exact matching range");
    assert(child.address == (gpu::GpuAddress)0x1008,
        "child address advances by the relative byte offset");
    assert(nested.span.offset == 52 && nested.span.size == 8,
        "nested span advances from its immediate parent");
    assert(nested.bytes.ptr == &storage[20] && nested.bytes.len == 8,
        "nested bytes remain consistent with the nested span");
    assert(nested.address == (gpu::GpuAddress)0x1014,
        "nested address remains consistent with the nested span");
    assert(exact_end.bytes.ptr == &storage[32] && exact_end.bytes.len == 8,
        "an exact-end child is accepted");
}

fn void gpu_span_allocation_accessor_round_trips_identity() @test {
    GpuAllocation allocation = { .owner = 5, .index = 7, .generation = 3 };
    GpuSpan whole = { .owner = 5, .index = 7, .generation = 3, .offset = 0, .size = 64 };
    GpuSpan part = whole.subspan(16, 16);
    assert(whole.allocation() == allocation, "a whole span names its allocation");
    assert(part.allocation() == allocation, "a subspan names the same allocation");
}
