module gpu_test;

import gpu;

fn void align_up_math() @test {
    assert(gpu::align_up(0, 16) == 0, "0 stays 0");
    assert(gpu::align_up(1, 16) == 16, "rounds up to alignment");
    assert(gpu::align_up(16, 16) == 16, "exact multiple unchanged");
    assert(gpu::align_up(17, 16) == 32, "rounds up to next multiple");
    assert(gpu::align_up(40, 32) == 64, "non-power case within power-of-two align");
}

fn void unchecked_subspan_math() @test {
    char[256] backing;
    GpuSpan s = {
        .gpu    = (GpuAddress)0x1000,
        .cpu    = &backing,
        .size   = 256,
        .offset = 0,
        .kind   = MemoryKind.DEVICE,
    };

    GpuSpan sub = s.unchecked_subspan(64, 32);
    assert((ulong)sub.gpu == 0x1040, "gpu address advances by offset");
    assert(sub.cpu == (char*)&backing + 64, "cpu pointer advances by offset");
    assert(sub.size == 32, "size is the requested size");
    assert(sub.offset == 64, "offset accumulates");
    assert(sub.kind == MemoryKind.DEVICE, "memory kind is inherited");
}

fn void checked_subspan_null_cpu() @test {
    GpuSpan s = { .gpu = (GpuAddress)0x2000, .cpu = null, .size = 128, .offset = 20 };
    GpuSpan sub = s.checked_subspan(16, 16)!!;
    assert(sub.cpu == null, "null cpu pointer stays null");
    assert((ulong)sub.gpu == 0x2010, "gpu address still advances");
    assert(sub.size == 16, "null-CPU slice keeps the requested size");
    assert(sub.offset == 36, "null-CPU slice advances backing offset");
}

fn void checked_subspan_bounds_and_nesting() @test {
    char[256] backing;
    GpuSpan span = {
        .gpu    = (GpuAddress)0x1000,
        .cpu    = &backing,
        .size   = 256,
        .offset = 0,
        .kind   = MemoryKind.PERSISTENT_UPLOAD,
        .buffer = { .index = 7, .generation = 3 },
    };

    GpuSpan parent = span.checked_subspan(32, 128)!!;
    GpuSpan exact = parent.checked_subspan(96, 32)!!;
    fault nested_escape = @catch(parent.checked_subspan(97, 32));
    fault offset_past_parent = @catch(span.checked_subspan(257, 0));
    fault size_past_parent = @catch(span.checked_subspan(200, 57));
    fault overflowing_size = @catch(span.checked_subspan(1, usz::max));

    assert((ulong)exact.gpu == 0x1080, "nested GPU address advances from each parent");
    assert(exact.cpu == (char*)&backing + 128, "nested CPU pointer advances from each parent");
    assert(exact.offset == 128, "nested backing offset advances from each parent");
    assert(exact.size == 32, "exact-fit nested slice keeps the requested size");
    assert(exact.kind == MemoryKind.PERSISTENT_UPLOAD, "nested slice inherits memory kind");
    assert(exact.buffer == span.buffer, "nested slice inherits backing buffer");
    assert(nested_escape == gpu::INVALID_ARGUMENT, "nested slice cannot escape its immediate parent");
    assert(offset_past_parent == gpu::INVALID_ARGUMENT, "offset beyond parent faults");
    assert(size_past_parent == gpu::INVALID_ARGUMENT, "size beyond parent remainder faults");
    assert(overflowing_size == gpu::INVALID_ARGUMENT, "maximum size cannot wrap through validation");
}

fn void checked_subspan_rejects_zero_size() @test {
    char[64] backing;
    GpuSpan span = {
        .gpu    = (GpuAddress)0x2000,
        .cpu    = &backing,
        .size   = 64,
        .offset = 10,
    };

    fault middle = @catch(span.checked_subspan(16, 0));
    fault endpoint = @catch(span.checked_subspan(64, 0));
    assert(middle == gpu::INVALID_ARGUMENT, "zero-size middle slice faults");
    assert(endpoint == gpu::INVALID_ARGUMENT, "zero-size endpoint slice faults");
}

fn void checked_subspan_rejects_metadata_overflow() @test {
    GpuSpan gpu_overflow = {
        .gpu  = (GpuAddress)ulong::max,
        .size = 2,
    };
    GpuSpan offset_overflow = {
        .gpu    = (GpuAddress)1,
        .size   = 2,
        .offset = usz::max,
    };
    GpuSpan cpu_overflow = {
        .gpu  = (GpuAddress)1,
        .cpu  = (void*)(uptr)uptr::max,
        .size = 2,
    };

    fault gpu_fault = @catch(gpu_overflow.checked_subspan(1, 1));
    fault offset_fault = @catch(offset_overflow.checked_subspan(1, 1));
    fault cpu_fault = @catch(cpu_overflow.checked_subspan(1, 1));

    assert(gpu_fault == gpu::INVALID_ARGUMENT, "GPU address addition cannot wrap");
    assert(offset_fault == gpu::INVALID_ARGUMENT, "backing offset addition cannot wrap");
    assert(cpu_fault == gpu::INVALID_ARGUMENT, "CPU pointer addition cannot wrap");
}

fn void pow2_and_range_helpers() @test {
    assert(gpu::is_pow2(1) && gpu::is_pow2(16) && gpu::is_pow2(4096), "powers of two accepted");
    assert(!gpu::is_pow2(0) && !gpu::is_pow2(3) && !gpu::is_pow2(48), "zero and non-powers rejected");

    assert(gpu::range_fits(256, 0, 256), "exact fit passes");
    assert(gpu::range_fits(256, 256, 0), "to-end at limit passes offset check");
    assert(!gpu::range_fits(256, 257, 0), "offset past limit fails");
    assert(!gpu::range_fits(256, 0, 257), "oversized fails");
    assert(!gpu::range_fits(256, 200, usz::max), "overflowing size fails");
    assert(gpu::range_fits(256, 255, 1), "last byte fits");
}
