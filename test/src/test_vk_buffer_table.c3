module gpu::internal::vk;

import gpu @public;

fn void buffer_table_rejects_lower_generation_stale_handle() @test {
    BufferCell[1] cells = {};
    BufferTable table = {
        .slots = cells[..],
        .owner = 1,
    };

    BufferHandle stale = table.alloc({})!!;
    table.free(stale)!!;
    BufferHandle replacement = table.alloc({})!!;

    assert(replacement.index == stale.index
        && replacement.generation > stale.generation,
        "slot reuse advances the live generation");
    assert(@catch(table.get(stale)) == gpu::INVALID_HANDLE,
        "a lower-generation handle cannot resolve the reused slot");
}
