module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;

const ulong SPAN_RESOLVER_DEVICE_OWNER = 71;

struct SpanResolverFixture {
    vk::VkDeviceState               state;
    vk::AllocationCell[2]           allocation_cells;
    char[64]                        mapping;
}

fn void init_span_resolver_fixture(SpanResolverFixture* fixture) {
    fixture.state.resource_owner = SPAN_RESOLVER_DEVICE_OWNER;
    fixture.state.allocations.owner = gpu::internal::resource_owner_tag(
        SPAN_RESOLVER_DEVICE_OWNER,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    );
    fixture.state.allocations.slots = fixture.allocation_cells[:1];
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
                .memory_class = gpu::MemoryClass.CPU_WRITE,
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
}

fn void allocation_span_accepts_full_and_exact_end_ranges() @test {
    SpanResolverFixture fixture;
    init_span_resolver_fixture(&fixture);
    gpu::GpuSpan full = fixture_allocation_span(&fixture, 0, fixture.mapping.len);
    vk::ResolvedSpan full_resolved = vk::resolve_span(&fixture.state, &full);
    assert(full_resolved.offset == 0 && full_resolved.size == fixture.mapping.len,
        "full buffer span resolves");

    gpu::GpuSpan exact_end = fixture_allocation_span(&fixture, 48, 16);
    vk::ResolvedSpan end_resolved = vk::resolve_span(&fixture.state, &exact_end);
    assert(end_resolved.offset == 48 && end_resolved.size == 16,
        "span ending at the buffer extent resolves");
}

fn void resolved_span_mapping_and_address_expose_the_resolved_range() @test {
    SpanResolverFixture fixture;
    init_span_resolver_fixture(&fixture);
    gpu::GpuSpan span = fixture_allocation_span(&fixture, 8, 16);

    vk::ResolvedSpan resolved = vk::resolve_span(&fixture.state, &span);
    char[] mapping = vk::resolved_span_mapping(&resolved);
    assert(mapping.ptr == &fixture.mapping[8] && mapping.len == 16,
        "mapping exposes the exact resolved range");
    assert((ulong)vk::resolved_span_address(&resolved) == 0x2008,
        "address exposes the exact resolved range start");
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

fn void independent_allocation_span_resolves() @test {
    SpanResolverFixture fixture;
    init_independent_span_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::GpuSpan span = fixture_allocation_span(&fixture, 8, 16);

    vk::ResolvedSpan resolved = vk::resolve_span(&fixture.state, &span);

    assert(resolved.slot == &fixture.allocation_cells[0].value.backing,
        "independent spans resolve their private backing slot");
    assert(resolved.offset == 8 && resolved.size == 16,
        "independent spans preserve their checked range");
    assert(resolved.memory_class == gpu::MemoryClass.CPU_WRITE,
        "independent spans derive immutable backing policy");
    assert(resolved.mapping == &fixture.mapping[8]
        && (ulong)resolved.address == 0x2008,
        "independent spans derive exact mapping and address offsets");

    gpu::GpuSpan exact_end = fixture_allocation_span(&fixture, 48, 16);
    assert(vk::resolve_span(&fixture.state, &exact_end).size == 16,
        "independent spans accept a range ending at the logical extent");
}
