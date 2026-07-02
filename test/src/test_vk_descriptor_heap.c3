module gpu_test;

import gpu;
import gpu::vk;

// Headless (lavapipe): descriptor heap slot bookkeeping through the public API.

fn gpu::Device make_heap_test_device(uint tex_capacity, uint smp_capacity) {
    gpu::DeviceDesc desc = {
        .backend                     = gpu::BackendKind.VULKAN,
        .enable_validation           = true,
        .frames_in_flight            = 2,
        .descriptor_heap_mode        = gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        .texture_descriptor_capacity = tex_capacity,
        .sampler_descriptor_capacity = smp_capacity,
        .application_name            = "gpu_descriptor_heap_test",
    };
    return gpu::create_device(&desc)!!;
}

fn gpu::TextureHandle make_heap_texture(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void texture_descriptor_round_trip() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex index = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(index.is_valid(), "created index is valid");
    gpu::destroy_texture_descriptor(&device, index)!!;

    bool faulted = false;
    if (catch gpu::destroy_texture_descriptor(&device, index)) faulted = true;
    assert(faulted, "destroying a stale index must fault INVALID_HANDLE");
}

fn void texture_descriptor_exhaustion() @test {
    gpu::Device device = make_heap_test_device(2, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex a = gpu::create_texture_descriptor(&device, texture, null)!!;
    gpu::TextureIndex b = gpu::create_texture_descriptor(&device, texture, null)!!;

    bool faulted = false;
    if (catch gpu::create_texture_descriptor(&device, texture, null)) faulted = true;
    assert(faulted, "allocating past capacity must fault DESCRIPTOR_HEAP_FULL");
    assert(a.is_valid() && b.is_valid(), "existing indices stay valid after exhaustion");

    gpu::destroy_texture_descriptor(&device, a)!!;
    gpu::destroy_texture_descriptor(&device, b)!!;
}

fn void texture_descriptor_deferred_reuse() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex first = gpu::create_texture_descriptor(&device, texture, null)!!;
    ushort first_slot = first.slot;

    gpu::begin_frame(&device)!!;
    gpu::destroy_texture_descriptor(&device, first)!!;

    gpu::TextureIndex during = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(during.slot != first_slot, "slot is not reused while its frame is in flight");

    gpu::end_frame(&device)!!;

    gpu::TextureIndex recycled = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(recycled.slot == first_slot, "retired slot is recycled once the frame timeline passes");
    assert(recycled.generation != first.generation, "recycled slot carries a new generation");

    bool faulted = false;
    if (catch gpu::destroy_texture_descriptor(&device, first)) faulted = true;
    assert(faulted, "the pre-recycle index is stale after reuse");

    gpu::destroy_texture_descriptor(&device, during)!!;
    gpu::destroy_texture_descriptor(&device, recycled)!!;
}

fn void texture_descriptor_stale_texture_rejected() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, texture)!!;

    bool faulted = false;
    if (catch gpu::create_texture_descriptor(&device, texture, null)) faulted = true;
    assert(faulted, "registering a destroyed texture must fault INVALID_HANDLE");
}

fn void texture_descriptor_bad_view_rejected() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureViewDesc reinterpret = { .format = gpu::Format.R32_UINT };
    bool faulted = false;
    if (catch gpu::create_texture_descriptor(&device, texture, &reinterpret)) faulted = true;
    assert(faulted, "format-reinterpreting views are outside the slice and must fault");
}

fn void sampler_round_trip_and_exhaustion() @test {
    gpu::Device device = make_heap_test_device(8, 2);
    defer gpu::destroy_device(&device)!!;

    gpu::SamplerDesc desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex a = gpu::create_sampler(&device, &desc)!!;
    gpu::SamplerIndex b = gpu::create_sampler(&device, &desc)!!;
    assert(a.is_valid() && b.is_valid(), "sampler indices valid");

    bool faulted = false;
    if (catch gpu::create_sampler(&device, &desc)) faulted = true;
    assert(faulted, "allocating past sampler capacity must fault DESCRIPTOR_HEAP_FULL");

    gpu::destroy_sampler(&device, a)!!;
    bool stale_faulted = false;
    if (catch gpu::destroy_sampler(&device, a)) stale_faulted = true;
    assert(stale_faulted, "destroying a stale sampler index must fault");

    gpu::destroy_sampler(&device, b)!!;
}

fn void leaked_descriptors_survive_teardown() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    gpu::TextureHandle texture = make_heap_texture(&device);
    gpu::create_texture_descriptor(&device, texture, null)!!;

    gpu::SamplerDesc desc = {};
    gpu::create_sampler(&device, &desc)!!;

    // Leaked on purpose: teardown must report and stay clean.
    gpu::destroy_device(&device)!!;
}
