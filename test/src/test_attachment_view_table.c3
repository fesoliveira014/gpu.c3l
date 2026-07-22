module gpu_test;

import gpu @public;
import gpu::internal::vk;
import vk;

fn void attachment_view_table_is_generation_checked() @test {
    vk::AttachmentViewCell[1] cells;
    vk::AttachmentViewTable table = {
        .slots = cells[..],
        .owner = 41,
    };
    vk::AttachmentViewSlot slot = {
        .texture      = { .owner = 7, .index = 3, .generation = 1 },
        .format       = gpu::Format.RGBA8_UNORM,
        .sample_count = gpu::SampleCount.ONE,
        .width        = 16,
        .height       = 8,
    };

    gpu::AttachmentViewHandle first = table.alloc(slot)!!;
    assert(first.owner == 41 && first.index == 0,
        "attachment table publishes its exact owner and slot");
    assert(table.get(first)!!.width == 16,
        "attachment metadata remains immutable while live");
    table.free(first)!!;
    assert(@catch(table.get(first)) == gpu::INVALID_HANDLE,
        "destroyed attachment generations are stale");

    gpu::AttachmentViewHandle second = table.alloc(slot)!!;
    assert(second.index == first.index
        && second.generation != first.generation,
        "recycled attachment slots advance generation");
}

fn void attachment_view_table_rejects_foreign_and_full_handles() @test {
    vk::AttachmentViewCell[1] cells;
    vk::AttachmentViewTable table = {
        .slots = cells[..],
        .owner = 73,
    };
    gpu::AttachmentViewHandle live = table.alloc({})!!;
    assert(@catch(table.alloc({})) == gpu::SLOT_TABLE_FULL,
        "live attachment capacity faults deterministically");

    gpu::AttachmentViewHandle foreign = live;
    foreign.owner++;
    assert(@catch(table.get(foreign)) == gpu::INVALID_HANDLE,
        "foreign attachment owners cannot alias local slots");
}

fn void attachment_view_generation_exhaustion_retires_slot() @test {
    vk::AttachmentViewCell[1] cells;
    cells[0].generation = vk::HANDLE_GENERATION_MAX - 1;
    vk::AttachmentViewTable table = {
        .slots     = cells[..],
        .owner     = 91,
        .free_head = 1,
        .next_new  = 1,
    };

    gpu::AttachmentViewHandle final = table.alloc({})!!;
    assert(final.generation == vk::HANDLE_GENERATION_MAX,
        "the final attachment generation can be published once");
    table.free(final)!!;
    assert(cells[0].generation == vk::HANDLE_GENERATION_MAX
        && table.free_head == 0,
        "an exhausted attachment slot is permanently retired");
    assert(@catch(table.alloc({})) == gpu::SLOT_TABLE_FULL,
        "an exhausted attachment generation has no successor");
}
