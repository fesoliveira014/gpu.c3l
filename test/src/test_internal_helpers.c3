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
    assert(
        queue_selection_id(&selection, QueueKind.GRAPHICS, 0)
            == queue_selection_id(&selection, QueueKind.COMPUTE, 0),
        "default compute aliases graphics identity"
    );
    assert(
        queue_selection_id(&selection, QueueKind.COMPUTE, 0)
            == queue_selection_id(&selection, QueueKind.TRANSFER, 0),
        "default transfer aliases graphics identity"
    );
    assert(
        queue_roles_equal(
            queue_selection_roles(&selection, QueueKind.GRAPHICS, 0),
            queue_selection_roles(&selection, QueueKind.COMPUTE, 0),
        ),
        "aliased roles share one role mask"
    );
    assert(
        queue_roles_equal(
            queue_selection_roles(&selection, QueueKind.COMPUTE, 0),
            queue_selection_roles(&selection, QueueKind.TRANSFER, 0),
        ),
        "transfer role shares the aliased mask"
    );
}

fn void queue_selection_splits_async_compute() @test {
    DeviceCaps caps = { .async_compute = true };
    QueueSelection selection = default_queue_selection(&caps);

    assert(
        queue_selection_id(&selection, QueueKind.GRAPHICS, 0)
            != queue_selection_id(&selection, QueueKind.COMPUTE, 0),
        "async compute uses a distinct identity"
    );
    assert(
        queue_selection_id(&selection, QueueKind.TRANSFER, 0)
            == queue_selection_id(&selection, QueueKind.GRAPHICS, 0),
        "transfer still aliases graphics by default"
    );
    assert(
        queue_selection_roles(&selection, QueueKind.GRAPHICS, 0).graphics,
        "graphics identity supports graphics"
    );
    assert(
        !queue_selection_roles(&selection, QueueKind.GRAPHICS, 0).compute,
        "graphics identity excludes distinct compute"
    );
    assert(
        queue_selection_roles(&selection, QueueKind.COMPUTE, 0).compute,
        "compute identity supports compute"
    );
    assert(
        !queue_selection_roles(&selection, QueueKind.COMPUTE, 0).graphics,
        "compute identity excludes graphics"
    );
}

fn void queue_selection_splits_distinct_transfer() @test {
    QueueSelection selection = queue_selection_for_aliases(
        compute_aliases_graphics:  true,
        transfer_aliases_graphics: false,
        transfer_aliases_compute:  false,
    );

    assert(
        queue_selection_id(&selection, QueueKind.GRAPHICS, 0)
            == queue_selection_id(&selection, QueueKind.COMPUTE, 0),
        "graphics and compute alias"
    );
    assert(
        queue_selection_id(&selection, QueueKind.TRANSFER, 0)
            != queue_selection_id(&selection, QueueKind.GRAPHICS, 0),
        "transfer has a distinct identity"
    );
    assert(
        queue_roles_equal(
            queue_selection_roles(&selection, QueueKind.GRAPHICS, 0),
            queue_selection_roles(&selection, QueueKind.COMPUTE, 0),
        ),
        "aliased roles share canonical metadata"
    );
    assert(
        queue_selection_roles(&selection, QueueKind.GRAPHICS, 0).graphics,
        "shared identity reports graphics"
    );
    assert(
        queue_selection_roles(&selection, QueueKind.GRAPHICS, 0).compute,
        "shared identity reports compute"
    );
    assert(
        !queue_selection_roles(&selection, QueueKind.GRAPHICS, 0).transfer,
        "shared identity excludes transfer"
    );
    assert(
        queue_selection_roles(&selection, QueueKind.TRANSFER, 0).transfer,
        "transfer identity reports transfer"
    );
}

fn void queue_selection_represents_three_distinct_queues() @test {
    QueueSelection selection = queue_selection_for_aliases(
        compute_aliases_graphics:  false,
        transfer_aliases_graphics: false,
        transfer_aliases_compute:  false,
    );

    assert(
        queue_selection_id(&selection, QueueKind.GRAPHICS, 0)
            != queue_selection_id(&selection, QueueKind.COMPUTE, 0),
        "graphics and compute differ"
    );
    assert(
        queue_selection_id(&selection, QueueKind.GRAPHICS, 0)
            != queue_selection_id(&selection, QueueKind.TRANSFER, 0),
        "graphics and transfer differ"
    );
    assert(
        queue_selection_id(&selection, QueueKind.COMPUTE, 0)
            != queue_selection_id(&selection, QueueKind.TRANSFER, 0),
        "compute and transfer differ"
    );
    assert(
        (uint)queue_selection_roles(&selection, QueueKind.GRAPHICS, 0)
            == (uint)(QueueRoles){ .graphics },
        "graphics identity reports one role"
    );
    assert(
        (uint)queue_selection_roles(&selection, QueueKind.COMPUTE, 0)
            == (uint)(QueueRoles){ .compute },
        "compute identity reports one role"
    );
    assert(
        (uint)queue_selection_roles(&selection, QueueKind.TRANSFER, 0)
            == (uint)(QueueRoles){ .transfer },
        "transfer identity reports one role"
    );
}

fn void queue_selection_stores_multiple_role_identities() @test {
    QueueRequirements requirements = {
        .counts = {
            .graphics = 2,
            .compute  = 2,
            .transfer = 1,
        },
    };
    ulong[2] graphics = { 10, 20 };
    ulong[2] compute = { 20, 30 };
    ulong[1] transfer = { 10 };
    QueueSelection selection = queue_selection_from_native_ids(
        requirements: requirements,
        graphics:     graphics[..],
        compute:      compute[..],
        transfer:     transfer[..],
    )!!;

    uint graphics_0 = queue_selection_id(&selection, QueueKind.GRAPHICS, 0);
    uint graphics_1 = queue_selection_id(&selection, QueueKind.GRAPHICS, 1);
    uint compute_0 = queue_selection_id(&selection, QueueKind.COMPUTE, 0);
    uint compute_1 = queue_selection_id(&selection, QueueKind.COMPUTE, 1);
    uint transfer_0 = queue_selection_id(&selection, QueueKind.TRANSFER, 0);
    assert(selection.identity_count == 3, "three native identities are stored");
    assert(graphics_0 != graphics_1, "one role exposes distinct identities");
    assert(compute_0 == graphics_1, "compute aliases the second graphics queue");
    assert(
        compute_1 != graphics_0 && compute_1 != graphics_1,
        "new native queue gets a new identity"
    );
    assert(transfer_0 == graphics_0, "transfer aliases the first graphics queue");
    assert(
        (uint)queue_selection_roles(&selection, QueueKind.GRAPHICS, 0)
            == (uint)(QueueRoles){ .graphics, .transfer },
        "aliased identities publish the union of their roles"
    );
}

fn void queue_selection_rejects_distinct_role_aliases_across_multiple_ids() @test {
    QueueRequirements requirements = {
        .counts = {
            .graphics = 2,
            .compute  = 1,
        },
        .distinct = { .compute },
    };
    ulong[2] graphics = { 10, 20 };
    ulong[1] compute = { 20 };

    assert(
        @catch(queue_selection_from_native_ids(
            requirements: requirements,
            graphics:     graphics[..],
            compute:      compute[..],
            transfer:     {},
        )) == UNSUPPORTED_FEATURE,
        "a distinct role cannot alias any identity from another role"
    );
}

fn void scalar_queue_selection_ignores_aliases_to_absent_roles() @test {
    QueueRequirements requirements = {
        .counts = {
            .compute  = 1,
            .transfer = 1,
        },
    };
    QueueSelection selection = queue_selection_for_requirements(
        requirements:               requirements,
        compute_aliases_graphics:    true,
        transfer_aliases_graphics:   true,
        transfer_aliases_compute:    false,
    )!!;

    assert(
        queue_selection_id(&selection, QueueKind.COMPUTE, 0)
            != queue_selection_id(&selection, QueueKind.TRANSFER, 0),
        "an absent graphics role cannot cause an alias"
    );
}
