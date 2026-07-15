module gpu;

fn void range_helper() @test {
    assert(range_fits(256, 0, 256), "exact fit passes");
    assert(range_fits(256, 256, 0), "to-end at limit passes offset check");
    assert(!range_fits(256, 257, 0), "offset past limit fails");
    assert(!range_fits(256, 0, 257), "oversized fails");
    assert(!range_fits(256, 200, usz::max), "overflowing size fails");
    assert(range_fits(256, 255, 1), "last byte fits");
}

fn void queue_selection_aliases_roles_by_default() @test {
    DeviceCaps caps;
    QueueSelection selection = default_queue_selection(&caps);

    assert(selection.counts.graphics == 1, "default selection exposes graphics role");
    assert(selection.counts.compute == 1, "default selection exposes compute role");
    assert(selection.counts.transfer == 1, "default selection exposes transfer role");
    assert(selection.graphics_id == selection.compute_id, "default compute aliases graphics identity");
    assert(selection.compute_id == selection.transfer_id, "default transfer aliases graphics identity");
    assert(queue_roles_equal(selection.graphics_roles, selection.compute_roles), "aliased roles share one role mask");
    assert(queue_roles_equal(selection.compute_roles, selection.transfer_roles), "transfer role shares the aliased mask");
}

fn void queue_selection_splits_async_compute() @test {
    DeviceCaps caps = { .async_compute = true };
    QueueSelection selection = default_queue_selection(&caps);

    assert(selection.graphics_id != selection.compute_id, "async compute uses a distinct identity");
    assert(selection.transfer_id == selection.graphics_id, "transfer still aliases graphics by default");
    assert(selection.graphics_roles.graphics, "graphics identity supports graphics");
    assert(!selection.graphics_roles.compute, "graphics identity excludes distinct compute");
    assert(selection.compute_roles.compute, "compute identity supports compute");
    assert(!selection.compute_roles.graphics, "compute identity excludes graphics");
}

fn void queue_selection_splits_distinct_transfer() @test {
    QueueSelection selection = queue_selection_for_aliases(
        compute_aliases_graphics:  true,
        transfer_aliases_graphics: false,
        transfer_aliases_compute:  false,
    );

    assert(selection.graphics_id == selection.compute_id, "graphics and compute alias");
    assert(selection.transfer_id != selection.graphics_id, "transfer has a distinct identity");
    assert(
        queue_roles_equal(selection.graphics_roles, selection.compute_roles),
        "aliased roles share canonical metadata"
    );
    assert(selection.graphics_roles.graphics, "shared identity reports graphics");
    assert(selection.graphics_roles.compute, "shared identity reports compute");
    assert(!selection.graphics_roles.transfer, "shared identity excludes transfer");
    assert(selection.transfer_roles.transfer, "transfer identity reports transfer");
}

fn void queue_selection_represents_three_distinct_queues() @test {
    QueueSelection selection = queue_selection_for_aliases(
        compute_aliases_graphics:  false,
        transfer_aliases_graphics: false,
        transfer_aliases_compute:  false,
    );

    assert(selection.graphics_id != selection.compute_id, "graphics and compute differ");
    assert(selection.graphics_id != selection.transfer_id, "graphics and transfer differ");
    assert(selection.compute_id != selection.transfer_id, "compute and transfer differ");
    assert(
        (uint)selection.graphics_roles == (uint)(QueueRoles){ .graphics },
        "graphics identity reports one role"
    );
    assert(
        (uint)selection.compute_roles == (uint)(QueueRoles){ .compute },
        "compute identity reports one role"
    );
    assert(
        (uint)selection.transfer_roles == (uint)(QueueRoles){ .transfer },
        "transfer identity reports one role"
    );
}
