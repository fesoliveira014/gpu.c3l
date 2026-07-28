module gpu_test;

import gpu;
import gpu::utils;

fn void mapped_gpu_span_factory_uses_exact_public_views() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_mapped_span_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    OwnedSpan owned = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "mapped_span",
    )!!;
    defer free_owned_span(&device, &owned)!!;

    char[] expected_bytes = gpu::get_span_mapping(&device, owned.span)!!;
    gpu::GpuAddress expected_address =
        gpu::get_span_address(&device, owned.span)!!;
    MappedGpuSpan mapped = utils::mapped_gpu_span(&device, owned.span)!!;

    assert(mapped.span.owner == owned.span.owner
        && mapped.span.index == owned.span.index
        && mapped.span.generation == owned.span.generation
        && mapped.span.offset == owned.span.offset
        && mapped.span.size == owned.span.size,
        "factory preserves the exact input span");
    assert(mapped.bytes.ptr == expected_bytes.ptr
        && mapped.bytes.len == expected_bytes.len,
        "factory preserves the exact public mapping");
    assert(mapped.address == expected_address,
        "factory preserves the exact public address");

    mapped.bytes[7] = 0x5A;
    assert(expected_bytes[7] == 0x5A,
        "writes through the utility update the underlying mapping");
}

fn void mapped_gpu_span_factory_propagates_unmapped_fault() @test {
    gpu::RuntimeDesc desc = {
        .application_name = "gpu_unmapped_span_test",
    };
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;

    OwnedSpan owned = allocate_owned_span(
        device:       &device,
        size:         64,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .compute },
        debug_name:   "unmapped_span",
    )!!;
    defer free_owned_span(&device, &owned)!!;

    fault expected = @catch(gpu::get_span_mapping(&device, owned.span));
    fault actual = @catch(utils::mapped_gpu_span(&device, owned.span));
    assert(actual == expected,
        "factory propagates the public mapping query fault");
}
