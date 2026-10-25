module gpu_test;

import gpu::internal @public;
import gpu::internal::vk @public;
import gpu @public;

fn TextureCompatibility test_texture_compatibility(ulong owner, uint value) {
    return (TextureCompatibility)(((uint128)owner << 64) | (uint128)value);
}


fn Device create_allocation_stub_device(
    gpu::internal::vk::VkDeviceState* state,
) {
    state.resource_owner =
        gpu::internal::allocate_device_resource_owner()!!;
    QueueSelection queues = gpu::internal::queue_selection_for_request(
        request: {
            .required = { .compute },
        },
        compute_aliases_graphics:  false,
        transfer_aliases_graphics: false,
        transfer_aliases_compute:  false,
    )!!;
    return gpu::internal::register_device(
        state:          state,
        caps:           {},
        resource_owner: state.resource_owner,
        queues:         queues,
    )!!;
}

fn AllocationDesc allocation_stub_desc(
    MemoryClass memory_class = MemoryClass.CPU_WRITE,
) {
    return {
        .size         = 64,
        .alignment    = 1,
        .memory_class = memory_class,
        .access       = { .compute },
    };
}

fn void allocation_descriptor_normalization() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);

    AllocationDesc desc = allocation_stub_desc();
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    assert(state.allocation_desc.alignment == 16,
        "alignment is normalized before dispatch");
    gpu::free_allocation(&device, &allocation)!!;
    gpu::destroy_device(&device)!!;
}

fn void allocation_ownership_queries_and_consumption() @test {
    gpu::internal::vk::VkDeviceState first_state;
    gpu::internal::vk::VkDeviceState second_state;
    Device first = create_allocation_stub_device(&first_state);
    Device second = create_allocation_stub_device(&second_state);
    AllocationDesc desc = allocation_stub_desc();
    GpuAllocation allocation = gpu::allocate_memory(&first, &desc)!!;

    GpuAllocation wrong_device_attempt = allocation;
    assert(wrong_device_attempt == allocation,
        "wrong-device free preserves the owning token");
    assert(first_state.allocation_free_calls == 0
        && second_state.allocation_free_calls == 0,
        "wrong-device free does not dispatch");

    AllocationInfo info = gpu::get_allocation_info(&first, allocation);
    assert(info.size == desc.size, "immutable size round-trips");
    assert(info.alignment == 16, "normalized alignment round-trips");
    GpuSpan span = gpu::get_allocation_span(&first, allocation);
    GpuSpan exact = span.subspan(32, 32);
    assert(gpu::get_span_mapping(&first, exact).len == 32,
        "mapped subspan has exact length");
    assert((ulong)gpu::get_span_address(&first, exact) == 0x1020,
        "address query includes the checked offset");

    first_state.allocation_block_free = true;
    GpuAllocation preserved = allocation;
    assert(@catch(gpu::free_allocation(&first, &allocation)) == gpu::RESOURCE_IN_USE);
    assert(allocation == preserved, "failed free preserves the owning token");
    first_state.allocation_block_free = false;
    gpu::free_allocation(&first, &allocation)!!;
    assert(!allocation.is_valid(), "successful free consumes the owning token");

    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
}

fn void mapped_allocation_composes_views_and_frees_through_span() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    AllocationDesc desc = allocation_stub_desc();
    MappedGpuSpan mapped = gpu::allocate_mapped_memory(&device, &desc)!!;
    assert(mapped.bytes.len == mapped.span.size && mapped.span.size == desc.size,
        "the composed mapping covers the allocation");
    assert((ulong)mapped.address == (ulong)gpu::get_span_address(&device, mapped.span),
        "the composed address matches the public query");
    GpuAllocation allocation = mapped.span.allocation();
    gpu::free_allocation(&device, &allocation)!!;
    assert(!state.allocation_live, "freeing through the span accessor releases the stub allocation");

    state.mapping_unavailable = true;
    assert(@catch(gpu::allocate_mapped_memory(&device, &desc)) == gpu::BACKEND_ERROR,
        "a mapping failure after allocation propagates the mapping fault");
    assert(!state.allocation_live && state.allocation_free_calls == 2,
        "a mapping failure after allocation frees the allocation");
    assert(state.abandoned_allocation_reports == 0,
        "a rollback free that succeeds reports nothing");
    state.mapping_unavailable = false;
    gpu::destroy_device(&device)!!;
}

fn void device_local_cpu_write_class_is_mapped() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    AllocationDesc desc = allocation_stub_desc(MemoryClass.CPU_WRITE_GPU_LOCAL);
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;

    AllocationInfo info = gpu::get_allocation_info(&device, allocation);
    assert(info.memory_class == MemoryClass.CPU_WRITE_GPU_LOCAL
        && info.mapped
        && info.device_local
        && info.addressable,
        "the device-local CPU-write class is mapped and addressable");
    GpuSpan span = gpu::get_allocation_span(&device, allocation);
    assert(gpu::get_span_mapping(&device, span).len == desc.size,
        "the device-local CPU-write class exposes its host mapping");
    gpu::free_allocation(&device, &allocation)!!;
    gpu::destroy_device(&device)!!;
}

fn void mapped_span_visibility_validates_ownership_and_ranges() @test {
    gpu::internal::vk::VkDeviceState first_state;
    gpu::internal::vk::VkDeviceState second_state;
    Device first = create_allocation_stub_device(&first_state);
    Device second = create_allocation_stub_device(&second_state);
    AllocationDesc desc = allocation_stub_desc();
    GpuAllocation allocation = gpu::allocate_memory(&first, &desc)!!;
    GpuSpan span = gpu::get_allocation_span(&first, allocation);
    GpuSpan range = span.subspan(7, 19);

    gpu::flush_mapped_span(&first, range)!!;
    gpu::invalidate_mapped_span(&first, range)!!;
    assert(first_state.allocation_flush_calls == 1
        && first_state.allocation_invalidate_calls == 1,
        "live mapped ranges dispatch each visibility operation once");

    GpuSpan wrong_kind = range;
    wrong_kind.owner = gpu::internal::resource_owner_tag(
        first_state.resource_owner,
        gpu::internal::ResourceOwnerKind.TEXTURE,
    );
    assert(first_state.allocation_visibility_entries == 2,
        "cross-kind spans stop before backend dispatch");
    assert(second_state.allocation_flush_calls == 0
        && first_state.allocation_invalidate_calls == 1,
        "cross-kind spans perform no visibility mutation");

    gpu::free_allocation(&first, &allocation)!!;
    assert(first_state.allocation_flush_calls == 1,
        "stale ranges do not reach native visibility work");

    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
}

fn void live_allocation_blocks_device_destruction() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    AllocationDesc desc = allocation_stub_desc();
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    Device preserved = device;

    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE);
    assert(device == preserved, "failed destroy preserves the device token");
    gpu::free_allocation(&device, &allocation)!!;
    assert(state.allocation_free_calls == 1,
        "successful free dispatches exactly once");
    gpu::free_allocation(&device, &allocation)!!;
    assert(state.allocation_free_calls == 1,
        "freeing the zeroed variable again is a no-op");
    gpu::destroy_device(&device)!!;
}

fn void failed_free_keeps_device_child_live() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    AllocationDesc desc = allocation_stub_desc();
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    GpuAllocation preserved = allocation;
    state.allocation_block_free = true;

    assert(@catch(gpu::free_allocation(&device, &allocation)) == gpu::RESOURCE_IN_USE);
    assert(allocation == preserved, "failed free preserves the allocation token");
    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE);

    state.allocation_block_free = false;
    gpu::free_allocation(&device, &allocation)!!;
    gpu::destroy_device(&device)!!;
}

fn void gpu_private_memory_stays_addressable() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    AllocationDesc desc = allocation_stub_desc(MemoryClass.GPU_PRIVATE);
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    GpuSpan span = gpu::get_allocation_span(&device, allocation);

    assert((ulong)gpu::get_span_address(&device, span) != 0,
        "GPU-private data remains addressable");

    gpu::free_allocation(&device, &allocation)!!;
    gpu::destroy_device(&device)!!;
}

fn void texture_placement_contract_is_strongly_typed() @test {
    TextureCompatibility compatibility = test_texture_compatibility(1, 3);
    TextureRequirements requirements = {
        .size          = 64,
        .alignment     = 64,
        .compatibility = compatibility,
    };
    TextureRequirements[1] queried = { requirements };
    AllocationDesc desc = {
        .size                 = 128,
        .memory_class         = MemoryClass.TEXTURE,
        .access               = { .compute },
        .texture_requirements = queried[..],
    };

    assert(desc.texture_requirements[0].compatibility.is_valid(),
        "texture compatibility is an opaque typed value");
}

fn TextureDesc texture_allocation_desc() {
    return {
        .width     = 4,
        .height    = 4,
        .format    = Format.RGBA8_UNORM,
        .usage     = { .sampled },
        .access    = { .compute },
    };
}

fn void texture_memory_requires_prequeried_compatibility() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    TextureDesc texture_desc = texture_allocation_desc();

    assert(@catch(gpu::get_texture_requirements(&device, null))
        == gpu::INVALID_ARGUMENT);
    assert(state.texture_requirements_calls == 0,
        "null descriptors stop before the backend requirements query");
    TextureRequirements requirements = gpu::get_texture_requirements(
        &device,
        &texture_desc,
    )!!;
    assert(requirements.size == 64 && requirements.alignment == 64,
        "texture requirements expose exact placement size and alignment");
    assert(requirements.compatibility.is_valid() && !requirements.dedicated_only,
        "requirements carry an opaque live compatibility value");

    TextureRequirements[1] queried = { requirements };

    AllocationDesc desc = allocation_stub_desc(MemoryClass.TEXTURE);
    desc.size = 128;
    desc.texture_requirements = queried[..];
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    assert(state.allocation_desc.alignment == requirements.alignment,
        "texture-memory alignment includes every queried requirement");

    TextureHandle texture = gpu::create_placed_texture(
        device:     &device,
        desc:       &texture_desc,
        allocation: allocation,
        offset:     64,
    )!!;
    assert(state.texture_placed_calls == 1,
        "valid placed creation dispatches exactly once");
    gpu::destroy_texture(&device, texture)!!;
    gpu::free_allocation(&device, &allocation)!!;
    gpu::destroy_device(&device)!!;
}

fn void texture_memory_rejects_dedicated_requirements() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    TextureDesc texture_desc = texture_allocation_desc();
    TextureRequirements requirements = gpu::get_texture_requirements(
        &device,
        &texture_desc,
    )!!;
    requirements.dedicated_only = true;
    TextureRequirements[1] queried = { requirements };
    AllocationDesc dedicated = allocation_stub_desc(MemoryClass.TEXTURE);
    dedicated.size = 128;
    dedicated.texture_requirements = queried[..];
    assert(@catch(gpu::allocate_memory(&device, &dedicated))
        == gpu::UNSUPPORTED_FEATURE);
    assert(state.allocation_attempts == 0,
        "dedicated-only requirements perform no allocation");

    gpu::destroy_device(&device)!!;
}

fn void allocation_compatibility_without_shared_memory_type_is_unsupported() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    TextureDesc texture_desc = texture_allocation_desc();
    TextureRequirements requirements = gpu::get_texture_requirements(
        &device,
        &texture_desc,
    )!!;
    ulong owner = gpu::internal::resource_owner_tag(
        state.resource_owner,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    );
    TextureRequirements first = requirements;
    TextureRequirements second = requirements;
    first.compatibility = test_texture_compatibility(owner, 1);
    second.compatibility = test_texture_compatibility(owner, 2);
    TextureRequirements[2] disjoint_requirements = { first, second };
    AllocationDesc desc = allocation_stub_desc(MemoryClass.TEXTURE);
    desc.size = 128;
    desc.texture_requirements = disjoint_requirements[..];

    assert(@catch(gpu::allocate_memory(&device, &desc)) == gpu::UNSUPPORTED_FEATURE,
        "requirements that share no memory type are unsupported");
    assert(state.allocation_attempts == 0,
        "unsupported compatibility performs no allocation");
    gpu::destroy_device(&device)!!;
}

fn void dedicated_texture_creation_publishes_independent_tokens() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    TextureDesc texture_desc = texture_allocation_desc();
    TextureRequirements requirements = gpu::get_texture_requirements(
        &device,
        &texture_desc,
    )!!;
    TextureRequirements[1] queried = { requirements };
    AllocationDesc allocation_desc = {
        .size                 = requirements.size,
        .memory_class         = MemoryClass.TEXTURE,
        .access               = texture_desc.access,
        .texture_requirements = queried[..],
        .debug_name           = "dedicated_texture_memory",
    };

    assert(@catch(gpu::create_dedicated_texture(
        &device,
        null,
        &allocation_desc,
    )) == gpu::INVALID_ARGUMENT);
    assert(state.texture_dedicated_calls == 0,
        "no dedicated backend mutation happens before creation");

    DedicatedTexture created = gpu::create_dedicated_texture(
        &device,
        &texture_desc,
        &allocation_desc,
    )!!;
    assert(created.texture.is_valid() && created.allocation.is_valid(),
        "dedicated creation atomically returns two ownership tokens");
    gpu::destroy_texture(&device, created.texture)!!;
    gpu::free_allocation(&device, &created.allocation)!!;
    gpu::destroy_device(&device)!!;
}
