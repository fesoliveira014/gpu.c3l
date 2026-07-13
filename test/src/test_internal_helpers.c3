module gpu::compat;

fn void range_helper() @test {
    assert(range_fits(256, 0, 256), "exact fit passes");
    assert(range_fits(256, 256, 0), "to-end at limit passes offset check");
    assert(!range_fits(256, 257, 0), "offset past limit fails");
    assert(!range_fits(256, 0, 257), "oversized fails");
    assert(!range_fits(256, 200, usz::max), "overflowing size fails");
    assert(range_fits(256, 255, 1), "last byte fits");
}
