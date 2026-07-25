module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import vma;
import std::core::mem;
import std::atomic::types;

const ulong ALLOCATION_TEST_OWNER = 91;

enum AllocationFailurePoint : int {
    NONE,
    CREATE_BUFFER,
    DEVICE_LOST,
    MISALIGNED_ALLOCATION,
    NULL_MAPPING,
    ZERO_ADDRESS,
    TABLE_FULL,
}

struct AllocationNativeProbe {
    AllocationFailurePoint     failure;
    vk::BufferCreateInfo       buffer_info;
    vma::AllocationCreateInfo  allocation_info;
    vk::DeviceSize             min_alignment;
    uint[3]                    queue_families;
    vk::MemoryRequirements     requirements;
    vk::MemoryRequirements     raw_requirements;
    vma::AllocationCreateInfo  raw_allocation_info;
    vk::MemoryPropertyFlags    memory_properties;
    char[64]                   mapping;
    uint                       create_count;
    uint                       allocate_count;
    uint                       free_count;
    uint                       requirements_count;
    uint                       properties_count;
    uint                       address_count;
    uint                       destroy_count;
}

struct AllocationTransactionFixture {
    vk::VkDeviceState state;
    vk::AllocationCell[1] cells;
    AllocationNativeProbe probe;
}

struct MappedVisibilityProbe {
    vk::Result     result;
    vk::DeviceSize offset;
    vk::DeviceSize size;
    uint           flush_count;
    uint           invalidate_count;
}

struct MappedVisibilityFixture {
    vk::VkDeviceState     state;
    vk::AllocationCell[1] cells;
    char[256]             mapping;
    MappedVisibilityProbe probe;
}

fn vk::Result mapped_visibility_probe_flush(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
    vk::DeviceSize offset,
    vk::DeviceSize size,
) {
    (void)allocator;
    (void)allocation;
    MappedVisibilityProbe* probe = (MappedVisibilityProbe*)user_data;
    probe.flush_count++;
    probe.offset = offset;
    probe.size = size;
    return probe.result;
}

fn vk::Result mapped_visibility_probe_invalidate(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
    vk::DeviceSize offset,
    vk::DeviceSize size,
) {
    (void)allocator;
    (void)allocation;
    MappedVisibilityProbe* probe = (MappedVisibilityProbe*)user_data;
    probe.invalidate_count++;
    probe.offset = offset;
    probe.size = size;
    return probe.result;
}

fn vk::AllocationNativeOps mapped_visibility_probe_ops(
    MappedVisibilityProbe* probe,
) {
    return {
        .user_data             = probe,
        .flush_allocation      = &mapped_visibility_probe_flush,
        .invalidate_allocation = &mapped_visibility_probe_invalidate,
    };
}

fn gpu::GpuSpan init_mapped_visibility_fixture(
    MappedVisibilityFixture* fixture,
    usz logical_size,
    usz native_size,
    bool coherent,
    bool mapped,
) {
    fixture.state.resource_owner = ALLOCATION_TEST_OWNER;
    fixture.state.non_coherent_atom_size = 64;
    fixture.state.allocations.owner = gpu::internal::resource_owner_tag(
        ALLOCATION_TEST_OWNER,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    );
    fixture.state.allocations.slots = fixture.cells[..];
    fixture.state.allocations.next_new = 1;
    fixture.cells[0] = {
        .value = {
            .backing = {
                .allocation = (vma::Allocation)(void*)0x200,
                .allocation_info = { .offset = 4096, .size = (vk::DeviceSize)native_size },
                .cpu_base = mapped ? &fixture.mapping[0] : null,
                .size = logical_size,
                .memory_class = mapped
                    ? gpu::MemoryClass.CPU_WRITE
                    : gpu::MemoryClass.GPU_PRIVATE,
            },
            .alignment = 64,
            .memory_class = mapped
                ? gpu::MemoryClass.CPU_WRITE
                : gpu::MemoryClass.GPU_PRIVATE,
            .coherent = coherent,
        },
        .generation = 3,
        .used = true,
    };
    return {
        .owner      = fixture.state.allocations.owner,
        .index      = 0,
        .generation = 3,
        .size       = logical_size,
    };
}

fn vk::AllocationSlot allocation_slot(uint live_placements = 0) {
    return {
        .backing = {
            .size   = 64,
            .access = { .compute },
        },
        .alignment       = 16,
        .memory_class    = gpu::MemoryClass.CPU_WRITE,
        .live_placements = live_placements,
    };
}

fn vk::AllocationTable allocation_table(vk::AllocationCell[] cells) {
    return {
        .slots = cells,
        .owner = gpu::internal::resource_owner_tag(
            ALLOCATION_TEST_OWNER,
            gpu::internal::ResourceOwnerKind.ALLOCATION,
        ),
    };
}

fn void mapped_memory_range_normalization_is_exact_and_safe() @test {
    vk::NonCoherentRange exact = vk::normalize_mapped_memory_range(
        offset:          64,
        size:            64,
        allocation_size: 150,
        atom_size:       64,
    )!!;
    assert(exact.offset == 64 && exact.size == 64,
        "aligned ranges remain exact");

    vk::NonCoherentRange unaligned = vk::normalize_mapped_memory_range(
        offset:          65,
        size:            2,
        allocation_size: 150,
        atom_size:       64,
    )!!;
    assert(unaligned.offset == 64 && unaligned.size == 64,
        "unaligned ranges expand to complete atoms");

    vk::NonCoherentRange crossing = vk::normalize_mapped_memory_range(
        offset:          63,
        size:            2,
        allocation_size: 130,
        atom_size:       64,
    )!!;
    assert(crossing.offset == 0 && crossing.size == 128,
        "ranges crossing an atom boundary expand from aligned endpoints");

    vk::NonCoherentRange tail = vk::normalize_mapped_memory_range(
        offset:          129,
        size:            21,
        allocation_size: 150,
        atom_size:       64,
    )!!;
    assert(tail.offset == 128 && tail.size == 22,
        "the final atom clamps to the native allocation end");

    vk::NonCoherentRange near_limit = vk::normalize_mapped_memory_range(
        offset:          usz::max - 63,
        size:            63,
        allocation_size: usz::max,
        atom_size:       64,
    )!!;
    assert(near_limit.offset == (vk::DeviceSize)(usz::max - 63)
        && near_limit.size == 63,
        "near-limit normalization never overflows end padding");

    assert(@catch(vk::normalize_mapped_memory_range(
        offset:          0,
        size:            0,
        allocation_size: 150,
        atom_size:       64,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::normalize_mapped_memory_range(
        offset:          149,
        size:            2,
        allocation_size: 150,
        atom_size:       64,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::normalize_mapped_memory_range(
        offset:          usz::max,
        size:            1,
        allocation_size: usz::max,
        atom_size:       64,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::normalize_mapped_memory_range(
        offset:          0,
        size:            1,
        allocation_size: 150,
        atom_size:       0,
    )) == gpu::BACKEND_ERROR);
}

fn void mapped_visibility_validates_and_skips_coherent_memory() @test {
    MappedVisibilityFixture fixture;
    gpu::GpuSpan span = init_mapped_visibility_fixture(
        fixture:      &fixture,
        logical_size: 150,
        native_size:  150,
        coherent:     false,
        mapped:       true,
    );
    vk::AllocationNativeOps ops = mapped_visibility_probe_ops(
        &fixture.probe,
    );

    gpu::GpuSpan unaligned = span;
    unaligned.offset = 65;
    unaligned.size = 2;
    vk::flush_mapped_span_for_state_with_ops(
        &fixture.state,
        unaligned,
        &ops,
    )!!;
    assert(fixture.probe.flush_count == 1
        && fixture.probe.offset == 64
        && fixture.probe.size == 64,
        "non-coherent flush uses the normalized allocation-relative range");

    gpu::GpuSpan tail = span;
    tail.offset = 129;
    tail.size = 21;
    vk::invalidate_mapped_span_for_state_with_ops(
        &fixture.state,
        tail,
        &ops,
    )!!;
    assert(fixture.probe.invalidate_count == 1
        && fixture.probe.offset == 128
        && fixture.probe.size == 22,
        "non-coherent invalidate clamps the final atom");

    fixture.cells[0].value.coherent = true;
    vk::flush_mapped_span_for_state_with_ops(
        &fixture.state,
        span,
        &ops,
    )!!;
    vk::invalidate_mapped_span_for_state_with_ops(
        &fixture.state,
        span,
        &ops,
    )!!;
    assert(fixture.probe.flush_count == 1
        && fixture.probe.invalidate_count == 1,
        "coherent visibility operations are validated native no-ops");

    fixture.cells[0].value.coherent = false;
    fixture.probe.result = vk::Result.ERROR_OUT_OF_HOST_MEMORY;
    assert(@catch(vk::flush_mapped_span_for_state_with_ops(
        &fixture.state,
        span,
        &ops,
    )) == gpu::OUT_OF_HOST_MEMORY,
        "visibility failures use the standard backend result mapping");
}

fn void mapped_visibility_rejects_invalid_spans_before_native_work() @test {
    MappedVisibilityFixture fixture;
    gpu::GpuSpan span = init_mapped_visibility_fixture(
        fixture:      &fixture,
        logical_size: 150,
        native_size:  150,
        coherent:     false,
        mapped:       true,
    );
    vk::AllocationNativeOps ops = mapped_visibility_probe_ops(
        &fixture.probe,
    );

    gpu::GpuSpan zero = span;
    zero.size = 0;
    assert(@catch(vk::flush_mapped_span_for_state_with_ops(
        &fixture.state,
        zero,
        &ops,
    )) == gpu::INVALID_ARGUMENT);

    gpu::GpuSpan overflow = span;
    overflow.offset = 149;
    overflow.size = 2;
    assert(@catch(vk::invalidate_mapped_span_for_state_with_ops(
        &fixture.state,
        overflow,
        &ops,
    )) == gpu::INVALID_ARGUMENT);

    gpu::GpuSpan stale = span;
    stale.generation++;
    assert(@catch(vk::flush_mapped_span_for_state_with_ops(
        &fixture.state,
        stale,
        &ops,
    )) == gpu::INVALID_HANDLE);

    gpu::GpuSpan foreign = span;
    foreign.owner = gpu::internal::resource_owner_tag(
        ALLOCATION_TEST_OWNER,
        gpu::internal::ResourceOwnerKind.BUFFER,
    );
    assert(@catch(vk::invalidate_mapped_span_for_state_with_ops(
        &fixture.state,
        foreign,
        &ops,
    )) == gpu::INVALID_HANDLE);

    fixture.cells[0].value.backing.cpu_base = null;
    assert(@catch(vk::flush_mapped_span_for_state_with_ops(
        &fixture.state,
        span,
        &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(fixture.probe.flush_count == 0
        && fixture.probe.invalidate_count == 0,
        "invalid, stale, foreign, and unmapped spans perform no native work");
}

fn void allocation_table_checks_identity_reuse_and_capacity() @test {
    vk::AllocationCell[2] cells;
    vk::AllocationTable table = allocation_table(cells[..]);

    gpu::GpuAllocation first = table.alloc(allocation_slot())!!;
    gpu::GpuAllocation second = table.alloc(allocation_slot())!!;
    assert(first.owner == table.owner && first.index == 0 && first.generation == 1,
        "first allocation publishes the table identity");
    assert(second.index == 1 && second.generation == 1,
        "new allocations advance through unused slots");
    assert(@catch(table.alloc(allocation_slot())) == gpu::SLOT_TABLE_FULL,
        "a full allocation table faults SLOT_TABLE_FULL");

    table.free(first)!!;
    assert(@catch(table.get(first)) == gpu::INVALID_HANDLE,
        "free invalidates the old allocation identity");
    gpu::GpuAllocation reused = table.alloc(allocation_slot())!!;
    assert(reused.index == first.index && reused.generation > first.generation,
        "a reusable slot receives a fresh generation");
    assert(@catch(table.get(first)) == gpu::INVALID_HANDLE,
        "the original token remains stale after slot reuse");
}

fn void allocation_table_preserves_live_placement_state() @test {
    vk::AllocationCell[1] cells;
    vk::AllocationTable table = allocation_table(cells[..]);
    gpu::GpuAllocation allocation = table.alloc(allocation_slot(1))!!;
    uint generation = cells[0].generation;
    uint free_head = table.free_head;

    assert(@catch(table.free(allocation)) == gpu::RESOURCE_IN_USE,
        "live placements reject allocation release");
    assert(cells[0].used && cells[0].generation == generation,
        "rejected release preserves the allocation cell");
    assert(table.free_head == free_head,
        "rejected release preserves the free list");

    cells[0].value.live_placements = 0;
    table.free(allocation)!!;
}

fn void allocation_table_retires_exhausted_generations() @test {
    vk::AllocationCell[1] cells = {{
        .generation = vk::HANDLE_GENERATION_MAX - 1,
    }};
    vk::AllocationTable table = allocation_table(cells[..]);
    table.free_head = 1;
    table.next_new = 1;
    gpu::GpuAllocation allocation = table.alloc(allocation_slot())!!;
    assert(allocation.generation == vk::HANDLE_GENERATION_MAX,
        "the final valid generation can be published once");

    table.free(allocation)!!;
    assert(cells[0].generation == vk::HANDLE_GENERATION_MAX,
        "the exhausted generation remains retired");
    assert(@catch(table.alloc(allocation_slot())) == gpu::SLOT_TABLE_FULL,
        "a retired slot is never reused");
}

fn void debug_resource_kind_ordinals_match_exact_schema() @test {
    assert((int)gpu::DebugResourceKind.NONE == 0);
    assert((int)gpu::DebugResourceKind.DEVICE == 1);
    assert((int)gpu::DebugResourceKind.GPU_SPAN == 2);
    assert((int)gpu::DebugResourceKind.TEXTURE == 3);
    assert((int)gpu::DebugResourceKind.PIPELINE == 4);
    assert((int)gpu::DebugResourceKind.SWAPCHAIN == 5);
    assert((int)gpu::DebugResourceKind.SHADER == 6);
    assert((int)gpu::DebugResourceKind.COMMAND_ALLOCATOR == 7);
    assert((int)gpu::DebugResourceKind.COMMAND_LIST == 8);
    assert((int)gpu::DebugResourceKind.TEXTURE_DESCRIPTOR == 9);
    assert((int)gpu::DebugResourceKind.SAMPLER == 10);
    assert((int)gpu::DebugResourceKind.ALLOCATION == 11,
        "allocation keeps the exact ordinal 11");
    assert((int)gpu::DebugResourceKind.ATTACHMENT_VIEW == 12,
        "attachment views append after the command allocator kind");
}

fn vk::Result allocation_probe_create_buffer(
    void* user_data,
    vma::Allocator allocator,
    vk::BufferCreateInfo* buffer_info,
    vma::AllocationCreateInfo* allocation_info,
    vk::DeviceSize min_alignment,
    vk::Buffer* out_buffer,
    vma::Allocation* out_allocation,
    vma::AllocationInfo* out_info,
) {
    (void)allocator;
    AllocationNativeProbe* probe = (AllocationNativeProbe*)user_data;
    probe.create_count++;
    probe.buffer_info = *buffer_info;
    probe.allocation_info = *allocation_info;
    probe.min_alignment = min_alignment;
    for (uint i = 0; i < buffer_info.queue_family_index_count; i++) {
        probe.queue_families[i] = buffer_info.p_queue_family_indices[i];
    }
    if (probe.failure == AllocationFailurePoint.CREATE_BUFFER) {
        return vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
    }
    if (probe.failure == AllocationFailurePoint.DEVICE_LOST) {
        return vk::Result.ERROR_DEVICE_LOST;
    }
    *out_buffer = (vk::Buffer)(uptr)0x100;
    *out_allocation = (vma::Allocation)(void*)0x200;
    *out_info = {
        .offset = min_alignment,
        .size = 256,
        .mapped_data = allocation_info.flags.mapped
            && probe.failure != AllocationFailurePoint.NULL_MAPPING
            ? &probe.mapping[0]
            : null,
    };
    return vk::Result.SUCCESS;
}

fn void allocation_probe_get_requirements(
    void* user_data,
    vk::Device device,
    vk::Buffer buffer,
    vk::MemoryRequirements* out_requirements,
) {
    (void)device;
    (void)buffer;
    AllocationNativeProbe* probe = (AllocationNativeProbe*)user_data;
    probe.requirements_count++;
    probe.requirements = {
        .size             = 256,
        .alignment        = probe.failure
            == AllocationFailurePoint.MISALIGNED_ALLOCATION
            ? probe.min_alignment * 2
            : 64,
        .memory_type_bits = uint::max,
    };
    *out_requirements = probe.requirements;
}

fn void allocation_probe_get_properties(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
    vk::MemoryPropertyFlags* out_properties,
) {
    (void)allocator;
    (void)allocation;
    AllocationNativeProbe* probe = (AllocationNativeProbe*)user_data;
    probe.properties_count++;
    uint flags = (uint)vk::MemoryPropertyFlagBits.HOST_VISIBLE_BIT
        | (uint)vk::MemoryPropertyFlagBits.HOST_COHERENT_BIT;
    probe.memory_properties = (vk::MemoryPropertyFlags)flags;
    *out_properties = probe.memory_properties;
}

fn vk::DeviceAddress allocation_probe_get_address(
    void* user_data,
    vk::Device device,
    vk::BufferDeviceAddressInfo* info,
) {
    (void)device;
    (void)info;
    AllocationNativeProbe* probe = (AllocationNativeProbe*)user_data;
    probe.address_count++;
    return probe.failure == AllocationFailurePoint.ZERO_ADDRESS ? 0 : 0x1000;
}

fn void allocation_probe_destroy_buffer(
    void* user_data,
    vma::Allocator allocator,
    vk::Buffer buffer,
    vma::Allocation allocation,
) {
    (void)allocator;
    (void)buffer;
    (void)allocation;
    AllocationNativeProbe* probe = (AllocationNativeProbe*)user_data;
    probe.destroy_count++;
}

fn vk::Result allocation_probe_allocate_memory(
    void* user_data,
    vma::Allocator allocator,
    vk::MemoryRequirements* requirements,
    vma::AllocationCreateInfo* allocation_info,
    vma::Allocation* out_allocation,
    vma::AllocationInfo* out_info,
) {
    (void)allocator;
    AllocationNativeProbe* probe = (AllocationNativeProbe*)user_data;
    probe.allocate_count++;
    probe.raw_requirements = *requirements;
    probe.raw_allocation_info = *allocation_info;
    *out_allocation = (vma::Allocation)(void*)0x500;
    *out_info = {
        .memory_type = 1,
        .offset      = requirements.alignment,
        .size        = requirements.size,
    };
    return vk::Result.SUCCESS;
}

fn void allocation_probe_free_memory(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
) {
    (void)allocator;
    (void)allocation;
    AllocationNativeProbe* probe = (AllocationNativeProbe*)user_data;
    probe.free_count++;
}

fn vk::AllocationNativeOps allocation_probe_ops(AllocationNativeProbe* probe) {
    return {
        .user_data                    = probe,
        .create_buffer                = &allocation_probe_create_buffer,
        .allocate_memory              = &allocation_probe_allocate_memory,
        .free_memory                  = &allocation_probe_free_memory,
        .get_memory_requirements      = &allocation_probe_get_requirements,
        .get_memory_properties        = &allocation_probe_get_properties,
        .get_buffer_device_address    = &allocation_probe_get_address,
        .destroy_buffer               = &allocation_probe_destroy_buffer,
    };
}

fn void init_allocation_transaction_fixture(
    AllocationTransactionFixture* fixture,
    AllocationFailurePoint failure,
) {
    fixture.probe.failure = failure;
    fixture.state.resource_owner = ALLOCATION_TEST_OWNER;
    fixture.state.compute_family = 3;
    fixture.state.transfer_family = 5;
    fixture.state.allocations.owner = gpu::internal::resource_owner_tag(
        ALLOCATION_TEST_OWNER,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    );
    fixture.state.allocations.slots = fixture.cells[..];
    fixture.state.resource_mutex.init()!!;
    if (failure == AllocationFailurePoint.TABLE_FULL) {
        fixture.state.allocations.next_new = 1;
    }
}

fn gpu::AllocationDesc allocation_transaction_desc(
    gpu::MemoryClass memory_class = gpu::MemoryClass.CPU_WRITE,
) {
    return {
        .size         = 128,
        .alignment    = 128,
        .memory_class = memory_class,
        .access       = { .compute, .transfer },
    };
}

fn void check_allocation_failure(
    AllocationFailurePoint failure,
    fault expected_fault,
    uint expected_destroy,
) {
    AllocationTransactionFixture fixture;
    init_allocation_transaction_fixture(&fixture, failure);
    defer (void)fixture.state.resource_mutex.destroy();
    vk::AllocationNativeOps ops = allocation_probe_ops(&fixture.probe);
    gpu::AllocationDesc desc = allocation_transaction_desc();

    fault result = @catch(vk::allocate_memory_for_state_with_ops(
        &fixture.state,
        &desc,
        &ops,
    ));

    assert(result == expected_fault, "failure maps to the expected public fault");
    assert(fixture.probe.create_count == 1);
    assert(fixture.probe.destroy_count == expected_destroy);
    assert(fixture.state.allocations.next_new
        == (failure == AllocationFailurePoint.TABLE_FULL ? 1 : 0),
        "failed transactions never publish an allocation");
}

fn void allocation_transaction_rolls_back_every_failure() @test {
    check_allocation_failure(
        failure:          AllocationFailurePoint.CREATE_BUFFER,
        expected_fault:   gpu::OUT_OF_DEVICE_MEMORY,
        expected_destroy: 0,
    );
    check_allocation_failure(
        failure:          AllocationFailurePoint.DEVICE_LOST,
        expected_fault:   gpu::DEVICE_LOST,
        expected_destroy: 0,
    );
    check_allocation_failure(
        failure:          AllocationFailurePoint.MISALIGNED_ALLOCATION,
        expected_fault:   gpu::BACKEND_ERROR,
        expected_destroy: 1,
    );
    check_allocation_failure(
        failure:          AllocationFailurePoint.NULL_MAPPING,
        expected_fault:   gpu::BACKEND_ERROR,
        expected_destroy: 1,
    );
    check_allocation_failure(
        failure:          AllocationFailurePoint.ZERO_ADDRESS,
        expected_fault:   gpu::BACKEND_ERROR,
        expected_destroy: 1,
    );
    check_allocation_failure(
        failure:          AllocationFailurePoint.TABLE_FULL,
        expected_fault:   gpu::SLOT_TABLE_FULL,
        expected_destroy: 1,
    );
}

fn void repeated_retirement_and_release_destroy_allocation_once() @test {
    AllocationTransactionFixture fixture;
    init_allocation_transaction_fixture(
        &fixture,
        AllocationFailurePoint.NONE,
    );
    defer (void)fixture.state.resource_mutex.destroy();
    vk::AllocationNativeOps ops = allocation_probe_ops(&fixture.probe);
    gpu::AllocationDesc desc = allocation_transaction_desc();
    gpu::GpuAllocation allocation = vk::allocate_memory_for_state_with_ops(
        &fixture.state,
        &desc,
        &ops,
    )!!;
    vk::AllocationSlot* slot = fixture.state.allocations.get(allocation)!!;
    vk::CommandResourceReference[1] references = {{
        .owner      = allocation.owner,
        .index      = allocation.index,
        .generation = allocation.generation,
    }};
    vk::retain_tracked_command_reference(
        &fixture.state,
        &references[0],
    )!!;
    gpu::Queue queue = {
        .device = { .generation = 1 },
        .id     = 0,
    };
    gpu::CompletionPoint completion = gpu::internal::make_completion_point(queue, 1)!!;
    vk::QueueCompletionState[1] completions = {{ .bound = true }};
    fixture.state.queue_completions = completions[..];
    completions[0].retirement_mutex.init()!!;
    defer (void)completions[0].retirement_mutex.destroy();
    completions[0].published_sequence.store(1, AtomicOrdering.RELEASE);
    vk::CommandScratch[1] scratch = {{
        .references      = references[..],
        .reference_count = 1,
        .in_use          = true,
    }};
    uint[1] available;
    vk::CommandAllocatorCell[1] allocator_cells;
    fixture.state.command_allocators = {
        .slots = allocator_cells[..],
        .owner = 1,
    };
    gpu::CommandAllocatorHandle allocator = fixture.state.command_allocators.alloc({
        .scratch           = scratch[..],
        .available         = available[..],
        .in_flight_buffers = 1,
    })!!;
    vk::CommandAllocatorSlot* allocator_slot =
        fixture.state.command_allocators.get(allocator)!!;
    allocator_slot.mutex.init()!!;
    defer (void)allocator_slot.mutex.destroy();
    vk::CommandCell[1] command_cells;
    init_submitted_command_fixture(
        &fixture.state,
        command_cells[..],
    );
    defer (void)fixture.state.command_mutex.destroy();
    append_submitted_command_fixture(
        completion: &completions[0],
        record:     submitted_command_fixture_record(
            &fixture.state,
            allocator,
            0,
        ),
        point:      completion,
    );

    completions[0].retirement_mutex.lock()!!;
    vk::retire_queue_through_locked(&fixture.state, &completions[0], 1);
    completions[0].retirement_mutex.unlock();
    assert(slot.tracked_command_references.load(AtomicOrdering.RELAXED) == 0
        && completions[0].submitted_count == 0,
        "the first retirement releases the ownership record exactly once");
    vk::free_allocation_for_state_with_ops(
        &fixture.state,
        allocation,
        &ops,
    )!!;
    assert(fixture.probe.destroy_count == 1,
        "the released allocation reaches native destruction exactly once");

    completions[0].retirement_mutex.lock()!!;
    vk::retire_queue_through_locked(&fixture.state, &completions[0], 1);
    completions[0].retirement_mutex.unlock();
    assert(completions[0].submitted_count == 0,
        "repeated retirement finds no ownership record to release again");
    assert(@catch(vk::free_allocation_for_state_with_ops(
        &fixture.state,
        allocation,
        &ops,
    )) == gpu::INVALID_HANDLE
        && fixture.probe.destroy_count == 1,
        "repeated release cannot destroy the native allocation twice");
}

fn void check_allocation_memory_policy(gpu::MemoryClass memory_class) {
    AllocationTransactionFixture fixture;
    init_allocation_transaction_fixture(
        &fixture,
        AllocationFailurePoint.NONE,
    );
    defer (void)fixture.state.resource_mutex.destroy();
    vk::AllocationNativeOps ops = allocation_probe_ops(&fixture.probe);
    gpu::AllocationDesc desc = allocation_transaction_desc(memory_class);

    gpu::GpuAllocation allocation = vk::allocate_memory_for_state_with_ops(
        &fixture.state,
        &desc,
        &ops,
    )!!;

    uint expected_usage = (uint)vk::BufferUsageFlagBits.TRANSFER_SRC_BIT
        | (uint)vk::BufferUsageFlagBits.TRANSFER_DST_BIT
        | (uint)vk::BufferUsageFlagBits.UNIFORM_BUFFER_BIT
        | (uint)vk::BufferUsageFlagBits.STORAGE_BUFFER_BIT
        | (uint)vk::BufferUsageFlagBits.INDEX_BUFFER_BIT
        | (uint)vk::BufferUsageFlagBits.VERTEX_BUFFER_BIT
        | (uint)vk::BufferUsageFlagBits.INDIRECT_BUFFER_BIT
        | (uint)vk::BufferUsageFlagBits.SHADER_DEVICE_ADDRESS_BIT;
    assert(fixture.probe.buffer_info.size == desc.size
        && (uint)fixture.probe.buffer_info.usage == expected_usage,
        "atomic creation receives the logical size and generic-data usage set");
    assert(fixture.probe.buffer_info.sharing_mode == vk::SharingMode.CONCURRENT
        && fixture.probe.buffer_info.queue_family_index_count == 2
        && fixture.probe.queue_families[0] == 3
        && fixture.probe.queue_families[1] == 5,
        "allocation sharing follows the requested queue access");
    assert(fixture.probe.min_alignment == (vk::DeviceSize)desc.alignment,
        "atomic creation receives the requested minimum alignment");
    assert((usz)fixture.probe.requirements.alignment == 64,
        "alignment reporting includes the queried native requirement");
    assert(fixture.probe.create_count == 1
        && fixture.probe.requirements_count == 1
        && fixture.probe.properties_count == 1
        && fixture.probe.address_count == 1,
        "a successful transaction executes every native query once");
    assert(fixture.probe.destroy_count == 0,
        "published paired ownership is not rolled back");

    vma::AllocationCreateInfo* info = &fixture.probe.allocation_info;
    switch (memory_class) {
        case gpu::MemoryClass.CPU_WRITE:
            assert(info.usage == vma::MemoryUsage.AUTO_PREFER_HOST
                && info.flags.mapped
                && info.flags.host_access_sequential_write
                && !info.flags.host_access_random,
                "CPU_WRITE requests a persistently mapped sequential host allocation");
        case gpu::MemoryClass.CPU_READ:
            assert(info.usage == vma::MemoryUsage.AUTO_PREFER_HOST
                && info.flags.mapped
                && info.flags.host_access_random
                && !info.flags.host_access_sequential_write,
                "CPU_READ requests a persistently mapped random host allocation");
        case gpu::MemoryClass.GPU_PRIVATE:
            assert(info.usage == vma::MemoryUsage.AUTO_PREFER_DEVICE
                && !info.flags.mapped
                && !info.flags.host_access_random
                && !info.flags.host_access_sequential_write,
                "GPU_PRIVATE requests no host mapping policy");
        default:
            unreachable();
    }
    uint required = (uint)info.required_flags;
    if (memory_class == gpu::MemoryClass.GPU_PRIVATE) {
        assert(required == 0, "GPU_PRIVATE requires no host memory property");
    } else {
        assert((required & (uint)vk::MemoryPropertyFlagBits.HOST_VISIBLE_BIT) != 0,
            "CPU memory classes require HOST_VISIBLE memory");
    }
    assert(fixture.state.allocations.get(allocation)!!.alignment == 128,
        "the published slot retains actual alignment");
}

fn void allocation_transaction_applies_memory_class_policy() @test {
    check_allocation_memory_policy(gpu::MemoryClass.CPU_WRITE);
    check_allocation_memory_policy(gpu::MemoryClass.CPU_READ);
    check_allocation_memory_policy(gpu::MemoryClass.GPU_PRIVATE);
}

fn void texture_memory_uses_raw_non_addressable_allocation() @test {
    AllocationTransactionFixture fixture;
    init_allocation_transaction_fixture(
        &fixture,
        AllocationFailurePoint.NONE,
    );
    defer (void)fixture.state.resource_mutex.destroy();
    vk::AllocationNativeOps ops = allocation_probe_ops(&fixture.probe);
    gpu::TextureRequirements[2] queried = {
        {
            .size      = 128,
            .alignment = 64,
            .compatibility = vk::texture_compatibility(
                fixture.state.allocations.owner,
                0b0110,
            ),
        },
        {
            .size      = 64,
            .alignment = 32,
            .compatibility = vk::texture_compatibility(
                fixture.state.allocations.owner,
                0b0010,
            ),
        },
    };
    gpu::AllocationDesc desc = {
        .size                 = 256,
        .alignment            = 64,
        .memory_class         = gpu::MemoryClass.TEXTURE,
        .access               = { .compute, .transfer },
        .texture_requirements = queried[..],
    };

    gpu::GpuAllocation allocation = vk::allocate_memory_for_state_with_ops(
        &fixture.state,
        &desc,
        &ops,
    )!!;
    assert(fixture.probe.allocate_count == 1
        && fixture.probe.create_count == 0,
        "texture memory uses raw allocation without creating a buffer");
    assert(fixture.probe.raw_requirements.size == desc.size
        && fixture.probe.raw_requirements.alignment == desc.alignment
        && fixture.probe.raw_requirements.memory_type_bits == 0b0010,
        "raw allocation intersects every queried compatibility mask");
    assert(fixture.probe.raw_allocation_info.usage
        == vma::MemoryUsage.GPU_ONLY
        && fixture.probe.raw_allocation_info.flags.can_alias,
        "raw texture memory uses an explicit device-local policy");

    gpu::AllocationInfo info = vk::allocation_info_for_state(
        &fixture.state,
        allocation,
    )!!;
    assert(info.size == desc.size
        && info.memory_class == gpu::MemoryClass.TEXTURE
        && info.access == desc.access
        && !info.mapped && !info.coherent && !info.addressable,
        "texture allocation info exposes no generic data capabilities");
    assert(@catch(vk::allocation_span_for_state(
        &fixture.state,
        allocation,
    )) == gpu::INVALID_ARGUMENT,
        "texture allocation cannot resolve to a generic span");
    gpu::GpuSpan forged = {
        .owner      = allocation.owner,
        .index      = allocation.index,
        .generation = allocation.generation,
        .size       = desc.size,
    };
    assert(@catch(vk::resolve_span(&fixture.state, &forged))
        == gpu::INVALID_ARGUMENT,
        "forged texture spans stop before private-buffer resolution");

    vk::free_allocation_for_state_with_ops(
        &fixture.state,
        allocation,
        &ops,
    )!!;
    assert(fixture.probe.free_count == 1
        && fixture.probe.destroy_count == 0,
        "texture allocation release frees raw memory exactly once");
}

fn void allocation_table_pins_capacity_and_owner_domain() @test {
    assert(vk::ALLOCATION_CAPACITY == 4096,
        "independent allocation capacity is fixed at 4096");
    ulong allocation_owner = gpu::internal::resource_owner_tag(
        ALLOCATION_TEST_OWNER,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    );
    ulong buffer_owner = gpu::internal::resource_owner_tag(
        ALLOCATION_TEST_OWNER,
        gpu::internal::ResourceOwnerKind.BUFFER,
    );
    assert(allocation_owner != buffer_owner,
        "allocation identities occupy a distinct owner domain");

    vk::AllocationCell[] cells = mem::new_array(
        vk::AllocationCell,
        vk::ALLOCATION_CAPACITY,
    );
    defer mem::free(cells.ptr);
    vk::AllocationTable table = allocation_table(cells);
    for (uint i = 0; i < vk::ALLOCATION_CAPACITY; i++) {
        gpu::GpuAllocation allocation = table.alloc(allocation_slot())!!;
        assert(allocation.index == i,
            "allocation capacity is usable through its final slot");
    }
    assert(@catch(table.alloc(allocation_slot())) == gpu::SLOT_TABLE_FULL,
        "the allocation immediately after capacity faults SLOT_TABLE_FULL");
}

fn void allocation_table_rejects_invalid_identities() @test {
    vk::AllocationCell[2] cells;
    vk::AllocationTable table = allocation_table(cells[..]);
    gpu::GpuAllocation live = table.alloc(allocation_slot())!!;

    gpu::GpuAllocation wrong_owner = live;
    wrong_owner.owner = gpu::internal::resource_owner_tag(
        ALLOCATION_TEST_OWNER + 1,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    );
    assert(@catch(table.get(wrong_owner)) == gpu::INVALID_HANDLE,
        "wrong-owner allocation identities are rejected");

    gpu::GpuAllocation stale = live;
    stale.generation++;
    assert(@catch(table.get(stale)) == gpu::INVALID_HANDLE,
        "stale allocation generations are rejected");

    gpu::GpuAllocation unused = live;
    unused.index = 1;
    assert(@catch(table.get(unused)) == gpu::INVALID_HANDLE,
        "unused allocation cells are rejected");

    gpu::GpuAllocation out_of_range = live;
    out_of_range.index = 2;
    assert(@catch(table.get(out_of_range)) == gpu::INVALID_HANDLE,
        "out-of-range allocation indices are rejected");
}

fn void allocation_queries_and_immediate_release() @test {
    AllocationTransactionFixture fixture;
    init_allocation_transaction_fixture(
        &fixture,
        AllocationFailurePoint.NONE,
    );
    defer (void)fixture.state.resource_mutex.destroy();
    vk::AllocationNativeOps ops = allocation_probe_ops(&fixture.probe);
    gpu::AllocationDesc desc = allocation_transaction_desc();
    gpu::GpuAllocation allocation = vk::allocate_memory_for_state_with_ops(
        &fixture.state,
        &desc,
        &ops,
    )!!;

    gpu::AllocationInfo info = vk::allocation_info_for_state(
        &fixture.state,
        allocation,
    )!!;
    assert(info.size == desc.size && info.size < 256,
        "allocation info reports logical rather than native requirement size");
    assert(info.alignment == 128
        && info.memory_class == desc.memory_class
        && info.access == desc.access,
        "allocation info preserves immutable public policy");
    assert(info.mapped && info.coherent && info.addressable,
        "allocation info derives actual mapping, coherence, and addressability");

    gpu::GpuSpan span = vk::allocation_span_for_state(
        &fixture.state,
        allocation,
    )!!;
    assert(span.owner == allocation.owner
        && span.index == allocation.index
        && span.generation == allocation.generation
        && span.offset == 0
        && span.size == desc.size,
        "the whole span preserves allocation identity and logical extent");

    vk::AllocationSlot* slot = fixture.state.allocations.get(allocation)!!;
    slot.live_placements = 1;
    assert(@catch(vk::free_allocation_for_state_with_ops(
        &fixture.state,
        allocation,
        &ops,
    )) == gpu::RESOURCE_IN_USE,
        "live placements reject immediate allocation release");
    assert(fixture.state.allocations.get(allocation)!! == slot
        && fixture.probe.destroy_count == 0,
        "rejected release preserves table and paired native ownership");

    slot.live_placements = 0;
    vk::free_allocation_for_state_with_ops(
        &fixture.state,
        allocation,
        &ops,
    )!!;
    assert(fixture.probe.destroy_count == 1,
        "successful release destroys the paired buffer and VMA allocation");
    assert(@catch(fixture.state.allocations.get(allocation)) == gpu::INVALID_HANDLE,
        "successful release retires the public identity immediately");
}

fn void allocation_table_teardown_destroys_each_live_pair_once() @test {
    vk::VkDeviceState state;
    state.allocations.slots = mem::new_array(vk::AllocationCell, 2);
    state.allocations.next_new = 2;
    state.allocations.slots[0] = {
        .used = true,
        .value = {
            .backing = {
                .buffer     = (vk::Buffer)(uptr)0x100,
                .allocation = (vma::Allocation)(void*)0x200,
            },
        },
    };
    state.allocations.slots[1] = {
        .used = false,
        .value = {
            .backing = {
                .buffer     = (vk::Buffer)(uptr)0x300,
                .allocation = (vma::Allocation)(void*)0x400,
            },
        },
    };
    AllocationNativeProbe probe;
    vk::AllocationNativeOps ops = allocation_probe_ops(&probe);

    vk::destroy_allocation_table_with_ops(&state, &ops);

    assert(probe.destroy_count == 1,
        "device teardown destroys each live allocation pair exactly once");
    assert(state.allocations.slots.ptr == null
        && state.allocations.next_new == 0,
        "device teardown releases and clears the allocation table");
}

fn gpu::Device create_allocation_test_device(ZString name) {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = name,
    };
    return create_test_device(&desc)!!;
}

fn gpu::AllocationDesc native_allocation_desc(
    gpu::MemoryClass memory_class,
    ZString debug_name,
) {
    return {
        .size         = 256,
        .alignment    = 128,
        .memory_class = memory_class,
        .access       = { .graphics, .compute, .transfer },
        .debug_name   = debug_name,
    };
}

fn void native_allocation_classes_map_and_release() @test {
    gpu::Device device = create_allocation_test_device("gpu_allocation_test");
    defer (void)destroy_test_device(&device);
    gpu::MemoryStats baseline = gpu::get_memory_stats(&device)!!;

    gpu::AllocationDesc write_desc = native_allocation_desc(
        gpu::MemoryClass.CPU_WRITE,
        "cpu_write",
    );
    gpu::AllocationDesc private_desc = native_allocation_desc(
        gpu::MemoryClass.GPU_PRIVATE,
        "gpu_private",
    );
    gpu::AllocationDesc read_desc = native_allocation_desc(
        gpu::MemoryClass.CPU_READ,
        "cpu_read",
    );
    gpu::GpuAllocation write = gpu::allocate_memory(&device, &write_desc)!!;
    gpu::GpuAllocation private = gpu::allocate_memory(&device, &private_desc)!!;
    gpu::GpuAllocation read = gpu::allocate_memory(&device, &read_desc)!!;
    gpu::GpuAllocation stale_write = write;

    assert(write != private && write != read && private != read,
        "live allocations have distinct identities");
    gpu::GpuAllocation[3] allocations = { write, private, read };
    foreach (allocation : allocations) {
        gpu::AllocationInfo info = gpu::get_allocation_info(
            &device,
            allocation,
        )!!;
        gpu::GpuSpan span = gpu::get_allocation_span(&device, allocation)!!;
        assert(info.size == 256
            && info.alignment >= 128
            && (info.alignment & (info.alignment - 1)) == 0,
            "native allocation info preserves logical size and valid alignment");
        assert(info.addressable
            && (ulong)gpu::get_span_address(&device, span)!! != 0,
            "every memory class exposes a nonzero GPU address");

        vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
        vk::AllocationSlot* slot = state.allocations.get(allocation)!!;
        vk::MemoryRequirements native_requirements;
        vk::get_buffer_memory_requirements(
            state.device,
            slot.backing.buffer,
            &native_requirements,
        );
        assert(info.alignment >= (usz)native_requirements.alignment,
            "reported alignment includes the native Vulkan requirement");
    }

    gpu::GpuSpan write_span = gpu::get_allocation_span(&device, write)!!;
    char[] write_mapping = gpu::get_span_mapping(&device, write_span)!!;
    char[] write_tail = gpu::get_span_mapping(
        &device,
        write_span.checked_subspan(64, 32)!!,
    )!!;
    write_mapping[0] = 0x2A;
    write_tail[0] = 0x3B;
    assert(write_mapping.len == 256
        && write_tail.len == 32
        && write_mapping[64] == 0x3B,
        "CPU_WRITE exposes exact writable full and subspan mappings");

    gpu::GpuSpan read_span = gpu::get_allocation_span(&device, read)!!;
    char[] read_mapping = gpu::get_span_mapping(&device, read_span)!!;
    read_mapping[0] = 0x4C;
    assert(read_mapping.len == 256 && read_mapping[0] == 0x4C,
        "CPU_READ exposes its exact host mapping without visibility claims");

    gpu::GpuSpan private_span = gpu::get_allocation_span(&device, private)!!;
    assert(@catch(gpu::get_span_mapping(&device, private_span))
        == gpu::INVALID_ARGUMENT,
        "GPU_PRIVATE never exposes a host mapping");
    gpu::MemoryStats live = gpu::get_memory_stats(&device)!!;
    assert(live.live_allocation_count >= baseline.live_allocation_count + 3,
        "independent allocations contribute to VMA allocation statistics");
    assert(live.heap_count == baseline.heap_count
        && live.heap_count > 0
        && live.heaps[0].budget > 0,
        "allocation stats retain a valid heap budget snapshot");

    gpu::free_allocation(&device, &write)!!;
    assert(!write.is_valid()
        && @catch(gpu::get_allocation_info(&device, stale_write))
            == gpu::INVALID_HANDLE,
        "free consumes the token and invalidates copied identities");
    gpu::GpuAllocation double_free = stale_write;
    assert(@catch(gpu::free_allocation(&device, &double_free))
        == gpu::INVALID_HANDLE,
        "double free faults deterministically");
    assert(double_free == stale_write,
        "failed double free preserves the caller token");
    assert(gpu::get_allocation_info(&device, private)!!.size == 256,
        "freeing one allocation preserves its peers");

    gpu::free_allocation(&device, &private)!!;
    gpu::free_allocation(&device, &read)!!;
    gpu::MemoryStats released = gpu::get_memory_stats(&device)!!;
    assert(released.live_allocation_count == baseline.live_allocation_count,
        "immediate release returns VMA allocation statistics to baseline");
}

fn void native_mapped_visibility_round_trip() @test {
    gpu::Device device = create_allocation_test_device(
        "gpu_mapped_visibility_test",
    );
    defer (void)destroy_test_device(&device);
    gpu::AllocationDesc write_desc = native_allocation_desc(
        gpu::MemoryClass.CPU_WRITE,
        "mapped_write",
    );
    gpu::AllocationDesc read_desc = native_allocation_desc(
        gpu::MemoryClass.CPU_READ,
        "mapped_read",
    );
    gpu::GpuAllocation write = gpu::allocate_memory(&device, &write_desc)!!;
    defer (void)gpu::free_allocation(&device, &write);
    gpu::GpuAllocation read = gpu::allocate_memory(&device, &read_desc)!!;
    defer (void)gpu::free_allocation(&device, &read);
    gpu::GpuSpan write_span = gpu::get_allocation_span(&device, write)!!;
    gpu::GpuSpan read_span = gpu::get_allocation_span(&device, read)!!;
    gpu::GpuSpan written = write_span.checked_subspan(3, 127)!!;
    gpu::GpuSpan copied = read_span.checked_subspan(5, 127)!!;
    char[] source = gpu::get_span_mapping(&device, written)!!;
    for (usz i = 0; i < source.len; i++) {
        source[i] = (char)((i * 17 + 11) % 97);
    }
    gpu::flush_mapped_span(&device, written)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer (void)discard_test_commands(&commands);
    gpu::BufferCopyDesc copy = {
        .src = written,
        .dst = copied,
    };
    gpu::cmd_copy_buffer(&commands, &copy)!!;
    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&commands, &host_read)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer (void)discard_test_executable(&executable);
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;

    gpu::invalidate_mapped_span(&device, copied)!!;
    char[] destination = gpu::get_span_mapping(&device, copied)!!;
    assert(destination.len == source.len);
    for (usz i = 0; i < source.len; i++) {
        assert(destination[i] == source[i],
            "completed GPU writes become visible after mapped-span invalidation");
    }
}

fn void native_allocation_spans_support_fill_upload_and_readback() @test {
    gpu::Device device = create_allocation_test_device(
        "gpu_allocation_data_operations_test",
    );
    defer (void)destroy_test_device(&device);
    gpu::AllocationDesc private_desc = native_allocation_desc(
        gpu::MemoryClass.GPU_PRIVATE,
        "data_operations",
    );
    gpu::AllocationDesc upload_desc = native_allocation_desc(
        gpu::MemoryClass.CPU_WRITE,
        "data_upload",
    );
    gpu::AllocationDesc readback_desc = native_allocation_desc(
        gpu::MemoryClass.CPU_READ,
        "data_readback",
    );
    gpu::GpuAllocation allocation = gpu::allocate_memory(
        &device,
        &private_desc,
    )!!;
    defer (void)gpu::free_allocation(&device, &allocation);
    gpu::GpuAllocation upload = gpu::allocate_memory(
        &device,
        &upload_desc,
    )!!;
    defer (void)gpu::free_allocation(&device, &upload);
    gpu::GpuAllocation readback = gpu::allocate_memory(
        &device,
        &readback_desc,
    )!!;
    defer (void)gpu::free_allocation(&device, &readback);
    gpu::GpuSpan span = gpu::get_allocation_span(&device, allocation)!!;
    gpu::GpuSpan upload_span = gpu::get_allocation_span(&device, upload)!!;
    gpu::GpuSpan readback_span = gpu::get_allocation_span(
        &device,
        readback,
    )!!;

    char[256] expected;
    char[] mapped_upload = gpu::get_span_mapping(&device, upload_span)!!;
    for (usz i = 0; i < expected.len; i++) {
        expected[i] = (char)((i * 13 + 7) % 251);
        mapped_upload[i] = expected[i];
    }
    gpu::flush_mapped_span(&device, upload_span)!!;

    const usz FILL_OFFSET = 64;
    const usz FILL_SIZE = 16;
    const char FILL_BYTE = 0x5A;
    gpu::GpuSpan fill_span = span.checked_subspan(
        FILL_OFFSET,
        FILL_SIZE,
    )!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer (void)discard_test_commands(&commands);
    gpu::BufferCopyDesc upload_copy = {
        .src = upload_span,
        .dst = span,
    };
    gpu::cmd_copy_buffer(&commands, &upload_copy)!!;
    gpu::Barrier before_fill = {
        .before = { .transfer = true },
        .after = { .transfer = true },
    };
    gpu::cmd_barrier(&commands, &before_fill)!!;
    gpu::cmd_fill_buffer(&commands, fill_span, 0x5A5A_5A5A)!!;
    gpu::Barrier before_readback = {
        .before = { .transfer = true },
        .after = { .transfer = true },
    };
    gpu::cmd_barrier(&commands, &before_readback)!!;
    gpu::BufferCopyDesc readback_copy = {
        .src = span,
        .dst = readback_span,
    };
    gpu::cmd_copy_buffer(&commands, &readback_copy)!!;
    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after = { .host = true },
    };
    gpu::cmd_barrier(&commands, &host_read)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer (void)discard_test_executable(&executable);
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;

    gpu::invalidate_mapped_span(&device, readback_span)!!;
    char[] actual = gpu::get_span_mapping(&device, readback_span)!!;
    for (usz i = 0; i < actual.len; i++) {
        char wanted = i >= FILL_OFFSET && i < FILL_OFFSET + FILL_SIZE
            ? FILL_BYTE
            : expected[i];
        assert(actual[i] == wanted,
            "allocation-backed copies and fill preserve exact bytes");
    }
}
fn void allocation_span_commands_reject_stale_and_foreign_owners_before_mutation() @test {
    gpu::Device device = create_allocation_test_device(
        "gpu_allocation_span_rejection_test",
    );
    defer (void)destroy_test_device(&device);
    gpu::Device foreign_device = create_allocation_test_device(
        "gpu_allocation_span_foreign_test",
    );
    defer (void)destroy_test_device(&foreign_device);
    gpu::AllocationDesc desc = native_allocation_desc(
        gpu::MemoryClass.GPU_PRIVATE,
        "span_rejection",
    );
    gpu::GpuAllocation local = gpu::allocate_memory(&device, &desc)!!;
    defer (void)gpu::free_allocation(&device, &local);
    gpu::GpuAllocation stale_owner = gpu::allocate_memory(&device, &desc)!!;
    gpu::GpuSpan stale = gpu::get_allocation_span(&device, stale_owner)!!;
    gpu::free_allocation(&device, &stale_owner)!!;
    gpu::GpuAllocation foreign = gpu::allocate_memory(
        &foreign_device,
        &desc,
    )!!;
    defer (void)gpu::free_allocation(&foreign_device, &foreign);

    gpu::GpuSpan local_span = gpu::get_allocation_span(&device, local)!!;
    gpu::GpuSpan foreign_span = gpu::get_allocation_span(
        &foreign_device,
        foreign,
    )!!;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer (void)discard_test_commands(&commands);
    assert(@catch(gpu::cmd_fill_buffer(&commands, stale, 0))
        == gpu::INVALID_HANDLE,
        "fill rejects a stale allocation span");
    gpu::BufferCopyDesc copy = {
        .src = local_span,
        .dst = foreign_span,
    };
    assert(@catch(gpu::cmd_copy_buffer(&commands, &copy))
        == gpu::INVALID_HANDLE,
        "copy rejects a cross-device allocation span");

    gpu::cmd_fill_buffer(&commands, local_span, 0)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer (void)discard_test_executable(&executable);
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;
}

fn void live_allocation_blocks_normal_device_destruction() @test {
    gpu::Device device = create_allocation_test_device(
        "gpu_allocation_lifetime_test",
    );
    gpu::Device original = device;
    gpu::AllocationDesc desc = native_allocation_desc(
        gpu::MemoryClass.GPU_PRIVATE,
        "live_allocation",
    );
    gpu::GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;

    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE,
        "a live allocation blocks normal device destruction");
    assert(device == original,
        "live-child rejection preserves the device token");

    gpu::free_allocation(&device, &allocation)!!;
    destroy_test_device(&device)!!;
}

fn void lost_device_releases_live_allocation_backing() @test {
    gpu::Device device = create_allocation_test_device(
        "gpu_allocation_device_loss_test",
    );
    gpu::AllocationDesc desc = native_allocation_desc(
        gpu::MemoryClass.CPU_WRITE,
        "lost_allocation",
    );
    gpu::GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;
    assert(allocation.is_valid());

    gpu::internal::mark_device_lost(&device);
    destroy_test_device(&device)!!;

    assert(!device.is_valid(),
        "lost-device teardown consumes the device with live allocation backing");
}
