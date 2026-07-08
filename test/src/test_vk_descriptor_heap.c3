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

    fault stale_fault = @catch(gpu::destroy_texture_descriptor(&device, index));
    assert(stale_fault == gpu::INVALID_HANDLE, "destroying a stale index faults INVALID_HANDLE");
}

fn void texture_descriptor_exhaustion() @test {
    gpu::Device device = make_heap_test_device(2, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex a = gpu::create_texture_descriptor(&device, texture, null)!!;
    gpu::TextureIndex b = gpu::create_texture_descriptor(&device, texture, null)!!;

    fault full_fault = @catch(gpu::create_texture_descriptor(&device, texture, null));
    assert(full_fault == gpu::DESCRIPTOR_HEAP_FULL, "allocating past capacity faults DESCRIPTOR_HEAP_FULL");
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
    // The frame fence signals GPU-side; idle so the timeline has passed.
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::TextureIndex recycled = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(recycled.slot == first_slot, "retired slot is recycled once the frame timeline passes");
    assert(recycled.generation != first.generation, "recycled slot carries a new generation");

    fault recycled_fault = @catch(gpu::destroy_texture_descriptor(&device, first));
    assert(recycled_fault == gpu::INVALID_HANDLE, "the pre-recycle index is stale after reuse");

    gpu::destroy_texture_descriptor(&device, during)!!;
    gpu::destroy_texture_descriptor(&device, recycled)!!;
}

fn void texture_descriptor_stale_texture_rejected() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, texture)!!;

    fault texture_fault = @catch(gpu::create_texture_descriptor(&device, texture, null));
    assert(texture_fault == gpu::INVALID_HANDLE, "registering a destroyed texture faults INVALID_HANDLE");
}

fn void texture_descriptor_bad_view_rejected() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureViewDesc reinterpret = { .format = gpu::Format.R32_UINT };
    fault view_fault = @catch(gpu::create_texture_descriptor(&device, texture, &reinterpret));
    assert(view_fault == gpu::INVALID_ARGUMENT, "format-reinterpreting views fault INVALID_ARGUMENT");
}

fn void destroy_texture_with_live_descriptor_faults_and_recovers() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);

    gpu::TextureIndex index = gpu::create_texture_descriptor(&device, texture, null)!!;

    fault in_use_fault = @catch(gpu::destroy_texture(&device, texture));
    assert(in_use_fault == gpu::RESOURCE_IN_USE, "destroying a texture with a live descriptor faults RESOURCE_IN_USE");

    gpu::TextureIndex second = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(second.is_valid(), "handle stays valid and usable after the faulted destroy");
    gpu::destroy_texture_descriptor(&device, second)!!;

    gpu::destroy_texture_descriptor(&device, index)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_retired_undrained_descriptor_succeeds() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);

    gpu::TextureIndex index = gpu::create_texture_descriptor(&device, texture, null)!!;
    gpu::destroy_texture_descriptor(&device, index)!!;

    // No frame boundary since the descriptor was destroyed: its cell is
    // retired but undrained. destroy_texture must not treat it as live.
    gpu::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_blocks_while_any_descriptor_live() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);

    gpu::TextureIndex a = gpu::create_texture_descriptor(&device, texture, null)!!;
    gpu::TextureIndex b = gpu::create_texture_descriptor(&device, texture, null)!!;

    gpu::destroy_texture_descriptor(&device, a)!!;
    fault still_in_use = @catch(gpu::destroy_texture(&device, texture));
    assert(still_in_use == gpu::RESOURCE_IN_USE, "one remaining live descriptor still blocks destroy_texture");

    gpu::destroy_texture_descriptor(&device, b)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_unaffected_by_unrelated_descriptor() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture_a = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture_a)!!;
    gpu::TextureHandle texture_b = make_heap_texture(&device);

    gpu::TextureIndex index = gpu::create_texture_descriptor(&device, texture_a, null)!!;
    defer gpu::destroy_texture_descriptor(&device, index)!!;

    gpu::destroy_texture(&device, texture_b)!!;
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

    fault full_fault = @catch(gpu::create_sampler(&device, &desc));
    assert(full_fault == gpu::DESCRIPTOR_HEAP_FULL, "allocating past sampler capacity faults DESCRIPTOR_HEAP_FULL");

    gpu::destroy_sampler(&device, a)!!;
    fault stale_fault = @catch(gpu::destroy_sampler(&device, a));
    assert(stale_fault == gpu::INVALID_HANDLE, "destroying a stale sampler index faults INVALID_HANDLE");

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

// Pure: heap_fits_limits takes no device, so these synthesize HeapLimits
// values directly rather than depending on what the driver reports.

fn void heap_fits_limits_indexing_at_limit_fits() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
    };
    bool fits = gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(fits, "texture and sampler capacities at the limit fit");
}

fn void heap_fits_limits_indexing_exceeds_sampled_images() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images = 99,
        .max_storage_images = 100,
        .max_samplers       = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    ),
        "texture capacity over the sampled-image limit is rejected");
}

fn void heap_fits_limits_indexing_exceeds_storage_images() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images = 100,
        .max_storage_images = 99,
        .max_samplers       = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    ),
        "texture capacity over the storage-image limit is rejected");
}

fn void heap_fits_limits_indexing_exceeds_samplers() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images = 100,
        .max_storage_images = 100,
        .max_samplers       = 49,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 49,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    ),
        "sampler capacity over the sampler limit is rejected");
}

fn void heap_fits_limits_buffer_at_limit_fits() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size at the limit fits");
}

fn void heap_fits_limits_buffer_exceeds_resource_range() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 2048,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size over the resource descriptor buffer range is rejected");
}

fn void heap_fits_limits_buffer_exceeds_sampler_range() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 2048,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size over the sampler descriptor buffer range is rejected");
}

fn void heap_fits_limits_buffer_exceeds_address_space() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 2048,
        .sampler_descriptor_address_space_size  = 2048,
.descriptor_buffer_address_space_size = 2048,
        .layout_size                          = 4096,
    };
    assert(!gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size over the descriptor buffer address space is rejected");
}

fn void heap_fits_limits_indexing_exceeds_per_stage() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 99,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
    };
    bool fits = gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(!fits, "per-stage limit binds even when the set-level limit fits");
}

fn void heap_fits_limits_buffer_exceeds_per_type_address_space() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range    = 4096,
        .max_sampler_descriptor_buffer_range     = 4096,
        .resource_descriptor_address_space_size  = 4096,
        .sampler_descriptor_address_space_size   = 2048,
        .descriptor_buffer_address_space_size    = 4096,
        .layout_size                             = 4096,
    };
    bool fits = gpu::vk::heap_fits_limits(
        mode:         gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    );
    assert(!fits, "per-type address space binds even when the combined budget fits");
}

// Integration: a capacity above the library default, but still well within
// lavapipe's reported device limits, creates successfully.
fn void descriptor_heap_raised_capacity_creates() @test {
    gpu::Device device = make_heap_test_device(8192, 512);
    defer gpu::destroy_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureIndex index = gpu::create_texture_descriptor(&device, texture, null)!!;
    assert(index.is_valid(), "raised-capacity heap creates and serves a descriptor");
    gpu::destroy_texture_descriptor(&device, index)!!;
}
