module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;

const ulong SPAN_RESOLVER_DEVICE_OWNER = 71;

struct SpanDiagnosticCapture {
    gpu::DebugResourceKind resource_kind;
    bool                   has_identity;
    uint                   index;
    uint                   generation;
    ZString                debug_name;
}

struct SpanResolverFixture {
    vk::VkDeviceState               state;
    vk::BufferCell[1]               buffer_cells;
    vk::AllocationCell[1]           allocation_cells;
    vk::PersistentAllocationCell[1] persistent_cells;
    char[64]                        mapping;
}

fn void init_span_resolver_fixture(SpanResolverFixture* fixture) {
    fixture.state.resource_owner = SPAN_RESOLVER_DEVICE_OWNER;
    fixture.state.buffers.owner = gpu::resource_owner_tag(
        SPAN_RESOLVER_DEVICE_OWNER,
        gpu::ResourceOwnerKind.BUFFER,
    );
    fixture.state.buffers.slots = fixture.buffer_cells[..];
    fixture.state.buffers.next_new = 1;
    fixture.buffer_cells[0] = {
        .value = {
            .buffer      = (vk::Buffer)(uptr)0x2222,
            .gpu_base    = (gpu::GpuAddress)0x1000,
            .cpu_base    = &fixture.mapping[0],
            .size        = fixture.mapping.len,
            .usage       = { .storage, .addressable },
            .access      = { .compute, .transfer },
            .memory_kind = gpu::MemoryKind.STAGING,
        },
        .generation = 5,
        .used       = true,
    };
    fixture.state.allocations.owner = gpu::resource_owner_tag(
        SPAN_RESOLVER_DEVICE_OWNER,
        gpu::ResourceOwnerKind.ALLOCATION,
    );
    fixture.state.allocations.slots = fixture.allocation_cells[..];
    fixture.state.allocations.next_new = 1;
    fixture.allocation_cells[0] = {
        .value = {
            .backing = {
                .buffer      = (vk::Buffer)(uptr)0x3333,
                .gpu_base    = (gpu::GpuAddress)0x2000,
                .cpu_base    = &fixture.mapping[0],
                .size        = fixture.mapping.len,
                .usage       = { .storage, .addressable },
                .access      = { .compute },
                .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
            },
            .alignment    = 64,
            .memory_class = gpu::MemoryClass.CPU_WRITE,
            .coherent     = true,
        },
        .generation = 7,
        .used       = true,
    };
    vk::copy_debug_name(
        &fixture.allocation_cells[0].value.backing.debug_name[0],
        "independent",
    );
    fixture.state.persistent_arena.backing_buffer = {
        .owner      = fixture.state.buffers.owner,
        .index      = 0,
        .generation = 5,
    };
    fixture.state.persistent_arena.size = fixture.mapping.len;
    fixture.state.persistent_arena.allocations.owner = gpu::resource_owner_tag(
        SPAN_RESOLVER_DEVICE_OWNER,
        gpu::ResourceOwnerKind.PERSISTENT_ALLOCATION,
    );
    fixture.state.persistent_arena.allocations.slots = fixture.persistent_cells[..];
    fixture.state.persistent_arena.allocations.next_new = 1;
    fixture.persistent_cells[0] = {
        .value = {
            .offset = 16,
            .size   = 32,
            .access = { .compute },
        },
        .generation = 9,
        .used       = true,
    };
}

fn gpu::GpuSpan fixture_buffer_span(
    SpanResolverFixture* fixture,
    usz offset,
    usz size,
) {
    return {
        .owner      = fixture.state.buffers.owner,
        .index      = 0,
        .generation = 5,
        .offset     = offset,
        .size       = size,
    };
}

fn void buffer_span_resolves_private_metadata() @test {
    SpanResolverFixture fixture;
    init_span_resolver_fixture(&fixture);
    gpu::GpuSpan span = fixture_buffer_span(
        &fixture,
        8,
        16,
    );

    vk::ResolvedSpan resolved = vk::resolve_span(&fixture.state, &span)!!;
    gpu::BufferHandle expected = {
        .owner      = fixture.state.buffers.owner,
        .index      = 0,
        .generation = 5,
    };

    assert(resolved.slot == &fixture.buffer_cells[0].value,
        "buffer span resolves its native slot");
    assert(resolved.backing_buffer == expected,
        "buffer span resolves its backing handle");
    assert(resolved.offset == 8 && resolved.size == 16,
        "buffer span preserves its checked native range");
    assert(resolved.access == fixture.buffer_cells[0].value.access,
        "buffer span derives immutable access from the slot");
    assert(resolved.memory_kind == gpu::MemoryKind.STAGING,
        "buffer span derives its memory kind from the slot");
    assert(resolved.mapping == &fixture.mapping[8],
        "buffer span resolves its mapped address");
    assert((ulong)resolved.address == 0x1008,
        "buffer span resolves its GPU address");
}

fn gpu::GpuSpan fixture_persistent_span(
    SpanResolverFixture* fixture,
    usz offset,
    usz size,
) {
    return {
        .owner      = fixture.state.persistent_arena.allocations.owner,
        .index      = 0,
        .generation = 9,
        .offset     = offset,
        .size       = size,
    };
}

fn void buffer_span_accepts_full_and_exact_end_ranges() @test {
    SpanResolverFixture fixture;
    init_span_resolver_fixture(&fixture);
    gpu::GpuSpan full = fixture_buffer_span(&fixture, 0, fixture.mapping.len);
    vk::ResolvedSpan full_resolved = vk::resolve_span(&fixture.state, &full)!!;
    assert(full_resolved.offset == 0 && full_resolved.size == fixture.mapping.len,
        "full buffer span resolves");

    gpu::GpuSpan exact_end = fixture_buffer_span(&fixture, 48, 16);
    vk::ResolvedSpan end_resolved = vk::resolve_span(&fixture.state, &exact_end)!!;
    assert(end_resolved.offset == 48 && end_resolved.size == 16,
        "span ending at the buffer extent resolves");
}

fn void persistent_span_composes_relative_and_arena_offsets() @test {
    SpanResolverFixture fixture;
    init_span_resolver_fixture(&fixture);
    gpu::GpuSpan span = fixture_persistent_span(&fixture, 4, 8);

    vk::ResolvedSpan resolved = vk::resolve_span(&fixture.state, &span)!!;

    assert(resolved.slot == &fixture.buffer_cells[0].value,
        "persistent span resolves the arena backing slot");
    assert(resolved.offset == 20 && resolved.size == 8,
        "persistent span adds its relative offset exactly once");
    assert(resolved.access == fixture.persistent_cells[0].value.access,
        "persistent span derives allocation access");
    assert(resolved.memory_kind == gpu::MemoryKind.PERSISTENT_UPLOAD,
        "persistent span derives persistent memory kind");
    assert(resolved.mapping == &fixture.mapping[20],
        "persistent span resolves its mapped address");
    assert((ulong)resolved.address == 0x1014,
        "persistent span resolves its GPU address");
}

fn void span_resolver_rejects_wrong_owner_and_stale_generation() @test {
    SpanResolverFixture fixture;
    init_span_resolver_fixture(&fixture);

    gpu::GpuSpan wrong_owner = fixture_buffer_span(&fixture, 0, 16);
    wrong_owner.owner = gpu::resource_owner_tag(
        SPAN_RESOLVER_DEVICE_OWNER + 1,
        gpu::ResourceOwnerKind.BUFFER,
    );
    fault wrong_owner_fault = @catch(vk::resolve_span(&fixture.state, &wrong_owner));
    assert(wrong_owner_fault == gpu::INVALID_HANDLE,
        "a span from another device faults INVALID_HANDLE");

    gpu::GpuSpan stale = fixture_buffer_span(&fixture, 0, 16);
    stale.generation++;
    fault stale_fault = @catch(vk::resolve_span(&fixture.state, &stale));
    assert(stale_fault == gpu::INVALID_HANDLE,
        "a stale span generation faults INVALID_HANDLE");
}

fn void span_resolver_rejects_empty_and_escaping_ranges() @test {
    SpanResolverFixture fixture;
    init_span_resolver_fixture(&fixture);

    gpu::GpuSpan empty = fixture_buffer_span(&fixture, 0, 0);
    assert(@catch(vk::resolve_span(&fixture.state, &empty)) == gpu::INVALID_ARGUMENT,
        "an empty span faults INVALID_ARGUMENT");

    gpu::GpuSpan one_past = fixture_buffer_span(&fixture, 49, 16);
    assert(@catch(vk::resolve_span(&fixture.state, &one_past)) == gpu::INVALID_ARGUMENT,
        "a span past the buffer extent faults INVALID_ARGUMENT");

    gpu::GpuSpan overflow = fixture_buffer_span(&fixture, usz::max, 2);
    assert(@catch(vk::resolve_span(&fixture.state, &overflow)) == gpu::INVALID_ARGUMENT,
        "an overflowing span range faults INVALID_ARGUMENT");
}

fn void persistent_span_rejects_native_offset_overflow() @test {
    SpanResolverFixture fixture;
    init_span_resolver_fixture(&fixture);
    fixture.persistent_cells[0].value.offset = usz::max - 1;
    fixture.persistent_cells[0].value.size = 3;
    fixture.buffer_cells[0].value.size = usz::max;
    gpu::GpuSpan span = fixture_persistent_span(&fixture, 2, 1);

    fault range_fault = @catch(vk::resolve_span(&fixture.state, &span));

    assert(range_fault == gpu::INVALID_ARGUMENT,
        "persistent native-offset overflow faults INVALID_ARGUMENT");
}

fn void span_resolver_rejects_gpu_address_overflow() @test {
    SpanResolverFixture fixture;
    init_span_resolver_fixture(&fixture);
    fixture.buffer_cells[0].value.gpu_base = (gpu::GpuAddress)(ulong::max - 4);
    gpu::GpuSpan span = fixture_buffer_span(&fixture, 8, 1);

    fault address_fault = @catch(vk::resolve_span(&fixture.state, &span));

    assert(address_fault == gpu::INVALID_ARGUMENT,
        "GPU-address overflow faults INVALID_ARGUMENT");
}

fn void resolved_span_mapping_and_address_require_backing_metadata() @test {
    SpanResolverFixture fixture;
    init_span_resolver_fixture(&fixture);
    gpu::GpuSpan span = fixture_buffer_span(&fixture, 8, 16);

    vk::ResolvedSpan resolved = vk::resolve_span(&fixture.state, &span)!!;
    char[] mapping = vk::resolved_span_mapping(&resolved)!!;
    assert(mapping.ptr == &fixture.mapping[8] && mapping.len == 16,
        "mapping exposes the exact resolved range");
    assert((ulong)vk::resolved_span_address(&resolved)!! == 0x1008,
        "address exposes the exact resolved range start");

    fixture.buffer_cells[0].value.cpu_base = null;
    resolved = vk::resolve_span(&fixture.state, &span)!!;
    assert(@catch(vk::resolved_span_mapping(&resolved)) == gpu::INVALID_ARGUMENT,
        "an unmapped span faults INVALID_ARGUMENT");

    fixture.buffer_cells[0].value.gpu_base = (gpu::GpuAddress)0;
    resolved = vk::resolve_span(&fixture.state, &span)!!;
    assert(@catch(vk::resolved_span_address(&resolved)) == gpu::INVALID_ARGUMENT,
        "a non-addressable span faults INVALID_ARGUMENT");
}

fn void init_independent_span_fixture(SpanResolverFixture* fixture) {
    init_span_resolver_fixture(fixture);
    fixture.state.resource_mutex.init()!!;
}

fn gpu::GpuSpan fixture_allocation_span(
    SpanResolverFixture* fixture,
    usz offset,
    usz size,
) {
    return {
        .owner      = fixture.state.allocations.owner,
        .index      = 0,
        .generation = 7,
        .offset     = offset,
        .size       = size,
    };
}

fn void capture_span_diagnostic(
    gpu::DebugMessage* message,
    void* user_data,
) {
    SpanDiagnosticCapture* capture = (SpanDiagnosticCapture*)user_data;
    capture.resource_kind = message.resource.kind;
    capture.has_identity = message.resource.has_public_identity;
    capture.index = message.resource.index;
    capture.generation = message.resource.generation;
    capture.debug_name = message.resource.debug_name;
}

fn void independent_allocation_span_resolves() @test {
    SpanResolverFixture fixture;
    init_independent_span_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::GpuSpan span = fixture_allocation_span(&fixture, 8, 16);

    vk::ResolvedSpan resolved = vk::resolve_span(&fixture.state, &span)!!;

    assert(resolved.slot == &fixture.allocation_cells[0].value.backing,
        "independent spans resolve their private backing slot");
    assert(resolved.backing_buffer == gpu::BUFFER_HANDLE_INVALID,
        "independent spans never forge a public buffer handle");
    assert(resolved.offset == 8 && resolved.size == 16,
        "independent spans preserve their checked range");
    assert(resolved.access == fixture.allocation_cells[0].value.backing.access
        && resolved.memory_kind == gpu::MemoryKind.PERSISTENT_UPLOAD,
        "independent spans derive immutable backing policy");
    assert(resolved.mapping == &fixture.mapping[8]
        && (ulong)resolved.address == 0x2008,
        "independent spans derive exact mapping and address offsets");

    gpu::GpuSpan exact_end = fixture_allocation_span(&fixture, 48, 16);
    assert(vk::resolve_span(&fixture.state, &exact_end)!!.size == 16,
        "independent spans accept a range ending at the logical extent");
}

fn void independent_allocation_span_rejects_invalid_identity_and_range() @test {
    SpanResolverFixture fixture;
    init_independent_span_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();

    gpu::GpuSpan stale = fixture_allocation_span(&fixture, 0, 16);
    stale.generation++;
    assert(@catch(vk::resolve_span(&fixture.state, &stale)) == gpu::INVALID_HANDLE,
        "stale independent span generations are rejected");

    gpu::GpuSpan escaping = fixture_allocation_span(&fixture, 56, 16);
    assert(@catch(vk::resolve_span(&fixture.state, &escaping)) == gpu::INVALID_ARGUMENT,
        "escaping independent span ranges are rejected");

    fixture.allocation_cells[0].value.backing.cpu_base =
        (void*)(uptr::max - 4);
    gpu::GpuSpan pointer_overflow = fixture_allocation_span(&fixture, 8, 1);
    assert(@catch(vk::resolve_span(&fixture.state, &pointer_overflow))
        == gpu::INVALID_ARGUMENT,
        "independent mapping pointer overflow is rejected");

    fixture.allocation_cells[0].value.backing.cpu_base = null;
    fixture.allocation_cells[0].value.backing.gpu_base =
        (gpu::GpuAddress)(ulong::max - 4);
    gpu::GpuSpan address_overflow = fixture_allocation_span(&fixture, 8, 1);
    assert(@catch(vk::resolve_span(&fixture.state, &address_overflow))
        == gpu::INVALID_ARGUMENT,
        "independent GPU address overflow is rejected");
}

fn void independent_span_diagnostic_uses_allocation_identity() @test {
    SpanResolverFixture fixture;
    init_independent_span_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    SpanDiagnosticCapture capture;
    fixture.state.debug_callback = &capture_span_diagnostic;
    fixture.state.debug_user_data = &capture;
    gpu::GpuSpan escaping = fixture_allocation_span(&fixture, 56, 16);

    fault span_fault = @catch(vk::resolve_span_slot(
        state:          &fixture.state,
        span:           &escaping,
        required:       1,
        operation:      "cmd_dispatch",
        rejected_field: "GpuSpan",
    ));

    assert(span_fault == gpu::INVALID_ARGUMENT);
    assert(capture.resource_kind == gpu::DebugResourceKind.ALLOCATION
        && capture.has_identity
        && capture.index == 0
        && capture.generation == 7,
        "independent span diagnostics preserve allocation identity");
    assert(capture.debug_name != null
        && capture.debug_name.str_view() == "independent",
        "independent span diagnostics preserve the allocation debug name");
}
