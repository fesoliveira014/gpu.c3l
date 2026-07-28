module gpu_test;

import gpu;
import gpu::utils;

fn void mapped_gpu_span_checked_slicing_stays_aligned() @test {
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

    MappedGpuSpan child = mapped.checked_subspan(8, 32)!!;
    MappedGpuSpan nested = child.checked_subspan(12, 8)!!;
    MappedGpuSpan exact_end = child.checked_subspan(24, 8)!!;

    assert(child.span.offset == 40 && child.span.size == 32,
        "child span advances from the backing span");
    assert(child.bytes.ptr == &storage[8] && child.bytes.len == 32,
        "child bytes select the exact matching range");
    assert(child.address == (gpu::GpuAddress)0x1008,
        "child address advances by the relative byte offset");
    assert(nested.span.offset == 52 && nested.span.size == 8,
        "nested span advances from its immediate parent");
    assert(nested.bytes.ptr == &storage[20] && nested.bytes.len == 8,
        "nested bytes remain aligned with the nested span");
    assert(nested.address == (gpu::GpuAddress)0x1014,
        "nested address remains aligned with the nested span");
    assert(exact_end.bytes.ptr == &storage[32] && exact_end.bytes.len == 8,
        "an exact-end child is accepted");
}

fn void mapped_gpu_span_checked_slicing_propagates_span_faults() @test {
    char[16] storage;
    MappedGpuSpan mapped = {
        .span = {
            .owner      = 1,
            .generation = 1,
            .offset     = usz::max,
            .size       = storage.len,
        },
        .bytes   = storage[..],
        .address = (gpu::GpuAddress)0x2000,
    };

    assert(@catch(mapped.checked_subspan(0, 0)) == gpu::INVALID_ARGUMENT,
        "zero-size child propagates the span fault");
    assert(@catch(mapped.checked_subspan(12, 5)) == gpu::INVALID_ARGUMENT,
        "child escaping its immediate parent propagates the span fault");
    assert(@catch(mapped.checked_subspan(1, 1)) == gpu::INVALID_ARGUMENT,
        "backing offset overflow propagates the span fault");
}
