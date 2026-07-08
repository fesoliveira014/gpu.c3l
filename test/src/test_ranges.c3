module gpu_test;

import gpu;

fn void align_up_math() @test {
    assert(gpu::align_up(0, 16) == 0, "0 stays 0");
    assert(gpu::align_up(1, 16) == 16, "rounds up to alignment");
    assert(gpu::align_up(16, 16) == 16, "exact multiple unchanged");
    assert(gpu::align_up(17, 16) == 32, "rounds up to next multiple");
    assert(gpu::align_up(40, 32) == 64, "non-power case within power-of-two align");
}

fn void subspan_math() @test {
    char[256] backing;
    GpuSpan s = {
        .gpu    = (GpuAddress)0x1000,
        .cpu    = &backing,
        .size   = 256,
        .offset = 0,
        .kind   = MemoryKind.DEVICE,
    };

    GpuSpan sub = s.subspan(64, 32);
    assert((ulong)sub.gpu == 0x1040, "gpu address advances by offset");
    assert(sub.cpu == (char*)&backing + 64, "cpu pointer advances by offset");
    assert(sub.size == 32, "size is the requested size");
    assert(sub.offset == 64, "offset accumulates");
    assert(sub.kind == MemoryKind.DEVICE, "memory kind is inherited");
}

fn void subspan_null_cpu() @test {
    GpuSpan s = { .gpu = (GpuAddress)0x2000, .cpu = null, .size = 128 };
    GpuSpan sub = s.subspan(16, 16);
    assert(sub.cpu == null, "null cpu pointer stays null");
    assert((ulong)sub.gpu == 0x2010, "gpu address still advances");
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
