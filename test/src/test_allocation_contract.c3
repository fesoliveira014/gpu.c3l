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

fn void allocation_descriptor_validation_and_normalization() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);

    assert(@catch(gpu::allocate_memory(&device, null)) == gpu::INVALID_ARGUMENT);
    AllocationDesc zero = allocation_stub_desc();
    zero.size = 0;
    assert(@catch(gpu::allocate_memory(&device, &zero)) == gpu::INVALID_ARGUMENT);
    AllocationDesc bad_alignment = allocation_stub_desc();
    bad_alignment.alignment = 3;
    assert(@catch(gpu::allocate_memory(&device, &bad_alignment)) == gpu::INVALID_ARGUMENT);
    AllocationDesc empty_access = allocation_stub_desc();
    empty_access.access = {};
    assert(@catch(gpu::allocate_memory(&device, &empty_access)) == gpu::INVALID_ARGUMENT);
    AllocationDesc wrong_access = allocation_stub_desc();
    wrong_access.access = { .graphics };
    assert(@catch(gpu::allocate_memory(&device, &wrong_access)) == gpu::INVALID_ARGUMENT);
    assert(state.allocation_attempts == 0,
        "invalid descriptors never reach the backend");

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
    GpuAllocation stale = allocation;

    assert(@catch(gpu::free_allocation(&first, null)) == gpu::INVALID_HANDLE);
    GpuAllocation wrong_device_attempt = allocation;
    assert(@catch(gpu::free_allocation(&second, &wrong_device_attempt))
        == gpu::INVALID_HANDLE);
    assert(wrong_device_attempt == allocation,
        "wrong-device free preserves the owning token");
    assert(first_state.allocation_free_calls == 0
        && second_state.allocation_free_calls == 0,
        "wrong-device free does not dispatch");

    AllocationInfo info = gpu::get_allocation_info(&first, allocation)!!;
    assert(info.size == desc.size, "immutable size round-trips");
    assert(info.alignment == 16, "normalized alignment round-trips");
    GpuSpan span = gpu::get_allocation_span(&first, allocation)!!;
    GpuSpan exact = span.checked_subspan(32, 32)!!;
    assert(gpu::get_span_mapping(&first, exact)!!.len == 32,
        "mapped subspan has exact length");
    assert((ulong)gpu::get_span_address(&first, exact)!! == 0x1020,
        "address query includes the checked offset");
    assert(@catch(gpu::get_allocation_info(&second, allocation)) == gpu::INVALID_HANDLE);

    first_state.allocation_block_free = true;
    GpuAllocation preserved = allocation;
    assert(@catch(gpu::free_allocation(&first, &allocation)) == gpu::RESOURCE_IN_USE);
    assert(allocation == preserved, "failed free preserves the owning token");
    first_state.allocation_block_free = false;
    gpu::free_allocation(&first, &allocation)!!;
    assert(!allocation.is_valid(), "successful free consumes the owning token");
    assert(@catch(gpu::get_allocation_info(&first, stale)) == gpu::INVALID_HANDLE);

    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
}

fn void zero_sized_span_resolvers_are_rejected() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    AllocationDesc desc = allocation_stub_desc();
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    GpuSpan span = gpu::get_allocation_span(&device, allocation)!!;
    span.size = 0;

    assert(@catch(gpu::get_span_mapping(&device, span)) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::get_span_address(&device, span)) == gpu::INVALID_ARGUMENT);

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
    GpuSpan span = gpu::get_allocation_span(&first, allocation)!!;
    GpuSpan range = span.checked_subspan(7, 19)!!;

    gpu::flush_mapped_span(&first, range)!!;
    gpu::invalidate_mapped_span(&first, range)!!;
    assert(first_state.allocation_flush_calls == 1
        && first_state.allocation_invalidate_calls == 1,
        "live mapped ranges dispatch each visibility operation once");

    assert(@catch(gpu::flush_mapped_span(&second, range))
        == gpu::INVALID_HANDLE);
    GpuSpan buffer_owned = range;
    buffer_owned.owner = gpu::internal::resource_owner_tag(
        first_state.resource_owner,
        gpu::internal::ResourceOwnerKind.BUFFER,
    );
    assert(@catch(gpu::flush_mapped_span(&first, buffer_owned))
        == gpu::INVALID_HANDLE);
    assert(first_state.allocation_visibility_entries == 2,
        "cross-kind spans stop before backend dispatch");
    GpuSpan zero = range;
    zero.size = 0;
    assert(@catch(gpu::invalidate_mapped_span(&first, zero))
        == gpu::INVALID_ARGUMENT);
    assert(second_state.allocation_flush_calls == 0
        && first_state.allocation_invalidate_calls == 1,
        "rejected ranges perform no visibility mutation");

    gpu::free_allocation(&first, &allocation)!!;
    assert(@catch(gpu::flush_mapped_span(&first, range))
        == gpu::INVALID_HANDLE);
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
    assert(@catch(gpu::free_allocation(&device, &allocation)) == gpu::INVALID_HANDLE);
    assert(state.allocation_free_calls == 1,
        "consumed allocation cannot release twice");
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

fn void mapping_availability_is_class_specific() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    AllocationDesc desc = allocation_stub_desc(MemoryClass.GPU_PRIVATE);
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    GpuSpan span = gpu::get_allocation_span(&device, allocation)!!;

    assert(@catch(gpu::get_span_mapping(&device, span)) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::flush_mapped_span(&device, span)) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::invalidate_mapped_span(&device, span)) == gpu::INVALID_ARGUMENT);
    assert((ulong)gpu::get_span_address(&device, span)!! != 0,
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

    AllocationDesc missing = allocation_stub_desc(MemoryClass.TEXTURE);
    assert(@catch(gpu::allocate_memory(&device, &missing))
        == gpu::INVALID_ARGUMENT);
    AllocationDesc generic = allocation_stub_desc(MemoryClass.GPU_PRIVATE);
    TextureRequirements[1] queried = { requirements };
    generic.texture_requirements = queried[..];
    assert(@catch(gpu::allocate_memory(&device, &generic))
        == gpu::INVALID_ARGUMENT);
    assert(state.allocation_attempts == 0,
        "malformed texture-memory requests stop before allocation");

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

fn void texture_memory_rejects_foreign_and_dedicated_requirements() @test {
    gpu::internal::vk::VkDeviceState first_state;
    gpu::internal::vk::VkDeviceState second_state;
    Device first = create_allocation_stub_device(&first_state);
    Device second = create_allocation_stub_device(&second_state);
    TextureDesc texture_desc = texture_allocation_desc();
    TextureRequirements requirements = gpu::get_texture_requirements(
        &first,
        &texture_desc,
    )!!;
    TextureRequirements[1] queried = { requirements };

    AllocationDesc foreign = allocation_stub_desc(MemoryClass.TEXTURE);
    foreign.size = 128;
    foreign.texture_requirements = queried[..];
    assert(@catch(gpu::allocate_memory(&second, &foreign))
        == gpu::INVALID_HANDLE);
    TextureRequirements incompatible_first = requirements;
    TextureRequirements incompatible_second = requirements;
    ulong owner = gpu::internal::resource_owner_tag(
        first_state.resource_owner,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    );
    incompatible_first.compatibility = test_texture_compatibility(owner, 1);
    incompatible_second.compatibility = test_texture_compatibility(
        owner,
        2,
    );
    TextureRequirements[2] incompatible = {
        incompatible_first,
        incompatible_second,
    };
    AllocationDesc disjoint = allocation_stub_desc(MemoryClass.TEXTURE);
    disjoint.size = 128;
    disjoint.texture_requirements = incompatible[..];
    assert(@catch(gpu::allocate_memory(&first, &disjoint))
        == gpu::INVALID_ARGUMENT);
    requirements.dedicated_only = true;
    queried[0] = requirements;
    AllocationDesc dedicated = allocation_stub_desc(MemoryClass.TEXTURE);
    dedicated.size = 128;
    dedicated.texture_requirements = queried[..];
    assert(@catch(gpu::allocate_memory(&first, &dedicated))
        == gpu::UNSUPPORTED_FEATURE);
    assert(first_state.allocation_attempts == 0
        && second_state.allocation_attempts == 0,
        "foreign, incompatible, and dedicated requirements perform no allocation");

    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
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
    assert(@catch(gpu::create_dedicated_texture(
        &device,
        &texture_desc,
        null,
    )) == gpu::INVALID_ARGUMENT);
    assert(state.texture_dedicated_calls == 0,
        "null descriptors stop before dedicated backend mutation");

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
