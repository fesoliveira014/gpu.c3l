module gpu_test;

import gpu @public;

struct AllocationStubState {
    ulong          owner;
    GpuAllocation  allocation;
    AllocationDesc seen_desc;
    char[128]      mapping;
    uint           allocate_calls;
    uint           free_calls;
    bool           live;
    bool           block_free;
}

fn void destroy_allocation_stub(void* backend_state) {
    (void)backend_state;
}

fn GpuAllocation? allocate_stub_memory(Device* device, AllocationDesc* desc) {
    AllocationStubState* state = (AllocationStubState*)gpu::device_backend_state_ptr(device);
    state.allocate_calls++;
    if (state.live) return gpu::INVALID_RESOURCE_STATE~;
    state.seen_desc = *desc;
    state.allocation = {
        .owner      = gpu::resource_owner_tag(state.owner, ResourceOwnerKind.ALLOCATION),
        .index      = 0,
        .generation = 1,
    };
    state.live = true;
    return state.allocation;
}

fn void? free_stub_allocation(Device* device, GpuAllocation allocation) {
    AllocationStubState* state = (AllocationStubState*)gpu::device_backend_state_ptr(device);
    state.free_calls++;
    if (!state.live || allocation != state.allocation) return gpu::INVALID_HANDLE~;
    if (state.block_free) return gpu::RESOURCE_IN_USE~;
    state.live = false;
}

fn AllocationInfo? get_stub_allocation_info(Device* device, GpuAllocation allocation) {
    AllocationStubState* state = (AllocationStubState*)gpu::device_backend_state_ptr(device);
    if (!state.live || allocation != state.allocation) return gpu::INVALID_HANDLE~;
    return {
        .size         = state.seen_desc.size,
        .alignment    = state.seen_desc.alignment,
        .memory_class = state.seen_desc.memory_class,
        .access       = state.seen_desc.access,
        .mapped       = state.seen_desc.memory_class != MemoryClass.GPU_PRIVATE,
        .coherent     = false,
        .addressable  = true,
    };
}

fn GpuSpan? get_stub_allocation_span(Device* device, GpuAllocation allocation) {
    AllocationStubState* state = (AllocationStubState*)gpu::device_backend_state_ptr(device);
    if (!state.live || allocation != state.allocation) return gpu::INVALID_HANDLE~;
    return {
        .owner      = allocation.owner,
        .index      = allocation.index,
        .generation = allocation.generation,
        .size       = state.seen_desc.size,
    };
}

fn char[]? get_stub_span_mapping(Device* device, GpuSpan span) {
    AllocationStubState* state = (AllocationStubState*)gpu::device_backend_state_ptr(device);
    if (!state.live || span.owner != state.allocation.owner
        || span.index != state.allocation.index
        || span.generation != state.allocation.generation) {
        return gpu::INVALID_HANDLE~;
    }
    if (span.size == 0) return gpu::INVALID_ARGUMENT~;
    if (state.seen_desc.memory_class == MemoryClass.GPU_PRIVATE) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (!gpu::range_fits(state.seen_desc.size, span.offset, span.size)) {
        return gpu::INVALID_ARGUMENT~;
    }
    return state.mapping[span.offset:span.size];
}

fn GpuAddress? get_stub_span_address(Device* device, GpuSpan span) {
    AllocationStubState* state = (AllocationStubState*)gpu::device_backend_state_ptr(device);
    if (!state.live || span.owner != state.allocation.owner
        || span.index != state.allocation.index
        || span.generation != state.allocation.generation) {
        return gpu::INVALID_HANDLE~;
    }
    if (span.size == 0) return gpu::INVALID_ARGUMENT~;
    if (!gpu::range_fits(state.seen_desc.size, span.offset, span.size)) {
        return gpu::INVALID_ARGUMENT~;
    }
    return (GpuAddress)(0x1000 + span.offset);
}

BackendVTable allocation_stub_vtable = {
    .destroy_device      = &destroy_allocation_stub,
    .allocate_memory     = &allocate_stub_memory,
    .free_allocation     = &free_stub_allocation,
    .get_allocation_info = &get_stub_allocation_info,
    .get_allocation_span = &get_stub_allocation_span,
    .get_span_mapping    = &get_stub_span_mapping,
    .get_span_address    = &get_stub_span_address,
};

fn Device create_allocation_stub_device(AllocationStubState* state) {
    state.owner = gpu::allocate_device_resource_owner()!!;
    QueueSelection queues = gpu::queue_selection_for_requirements(
        requirements: {
            .counts = { .compute = 1 },
        },
        compute_aliases_graphics:  false,
        transfer_aliases_graphics: false,
        transfer_aliases_compute:  false,
    )!!;
    return gpu::register_device(
        backend:        BackendKind.VULKAN,
        caps:           {},
        vtable:         &allocation_stub_vtable,
        backend_state:  state,
        resource_owner: state.owner,
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
    AllocationStubState state;
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
    assert(state.allocate_calls == 0, "invalid descriptors never reach the backend");

    AllocationDesc desc = allocation_stub_desc();
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    assert(state.seen_desc.alignment == 16, "alignment is normalized before dispatch");
    gpu::free_allocation(&device, &allocation)!!;
    gpu::destroy_device(&device)!!;
}

fn void allocation_ownership_queries_and_consumption() @test {
    AllocationStubState first_state;
    AllocationStubState second_state;
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
    assert(first_state.free_calls == 0 && second_state.free_calls == 0,
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

    first_state.block_free = true;
    GpuAllocation preserved = allocation;
    assert(@catch(gpu::free_allocation(&first, &allocation)) == gpu::RESOURCE_IN_USE);
    assert(allocation == preserved, "failed free preserves the owning token");
    first_state.block_free = false;
    gpu::free_allocation(&first, &allocation)!!;
    assert(!allocation.is_valid(), "successful free consumes the owning token");
    assert(@catch(gpu::get_allocation_info(&first, stale)) == gpu::INVALID_HANDLE);

    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
}

fn void zero_sized_span_resolvers_are_rejected() @test {
    AllocationStubState state;
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

fn void live_allocation_blocks_device_destruction() @test {
    AllocationStubState state;
    Device device = create_allocation_stub_device(&state);
    AllocationDesc desc = allocation_stub_desc();
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    Device preserved = device;

    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE);
    assert(device == preserved, "failed destroy preserves the device token");
    gpu::free_allocation(&device, &allocation)!!;
    assert(state.free_calls == 1, "successful free dispatches exactly once");
    assert(@catch(gpu::free_allocation(&device, &allocation)) == gpu::INVALID_HANDLE);
    assert(state.free_calls == 1, "consumed allocation cannot release twice");
    gpu::destroy_device(&device)!!;
}

fn void failed_free_keeps_device_child_live() @test {
    AllocationStubState state;
    Device device = create_allocation_stub_device(&state);
    AllocationDesc desc = allocation_stub_desc();
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    GpuAllocation preserved = allocation;
    state.block_free = true;

    assert(@catch(gpu::free_allocation(&device, &allocation)) == gpu::RESOURCE_IN_USE);
    assert(allocation == preserved, "failed free preserves the allocation token");
    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE);

    state.block_free = false;
    gpu::free_allocation(&device, &allocation)!!;
    gpu::destroy_device(&device)!!;
}

fn void mapping_availability_is_class_specific() @test {
    AllocationStubState state;
    Device device = create_allocation_stub_device(&state);
    AllocationDesc desc = allocation_stub_desc(MemoryClass.GPU_PRIVATE);
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    GpuSpan span = gpu::get_allocation_span(&device, allocation)!!;

    assert(@catch(gpu::get_span_mapping(&device, span)) == gpu::INVALID_ARGUMENT);
    assert((ulong)gpu::get_span_address(&device, span)!! != 0,
        "GPU-private data remains addressable");

    gpu::free_allocation(&device, &allocation)!!;
    gpu::destroy_device(&device)!!;
}