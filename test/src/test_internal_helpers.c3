module gpu;

struct CompletionFixture {
    ulong current;
    fault wait_fault;
    uint  poll_calls;
    uint  wait_calls;
    uint  last_queue_id;
    ulong last_sequence;
    ulong last_timeout_ns;
}

fn void completion_fixture_destroy(void* backend_state) {
    (void)backend_state;
}

fn bool? completion_fixture_poll(
    Device* device,
    uint queue_id,
    ulong sequence,
) {
    CompletionFixture* fixture =
        (CompletionFixture*)device_backend_state_ptr(device);
    if (fixture == null) return INVALID_HANDLE~;
    fixture.poll_calls++;
    fixture.last_queue_id = queue_id;
    fixture.last_sequence = sequence;
    return fixture.current >= sequence;
}

fn void? completion_fixture_wait(
    Device* device,
    uint queue_id,
    ulong sequence,
    ulong timeout_ns,
) {
    CompletionFixture* fixture =
        (CompletionFixture*)device_backend_state_ptr(device);
    if (fixture == null) return INVALID_HANDLE~;
    fixture.wait_calls++;
    fixture.last_queue_id = queue_id;
    fixture.last_sequence = sequence;
    fixture.last_timeout_ns = timeout_ns;
    if (fixture.wait_fault) return fixture.wait_fault~;
}

BackendVTable completion_fixture_vtable = {
    .destroy_device  = &completion_fixture_destroy,
    .poll_completion = &completion_fixture_poll,
    .wait_completion = &completion_fixture_wait,
};

fn void range_helper() @test {
    assert(range_fits(256, 0, 256), "exact fit passes");
    assert(range_fits(256, 256, 0), "to-end at limit passes offset check");
    assert(!range_fits(256, 257, 0), "offset past limit fails");
    assert(!range_fits(256, 0, 257), "oversized fails");
    assert(!range_fits(256, 200, usz::max), "overflowing size fails");
    assert(range_fits(256, 255, 1), "last byte fits");
}

fn void completion_points_pack_queue_identity_and_sequence() @test {
    Device device = { .index = 3, .generation = 7 };
    Queue queue = {
        .device = device,
        .id     = MAX_SELECTED_QUEUE_IDENTITIES - 1,
        .roles  = { .transfer },
    };

    CompletionPoint point = make_completion_point(queue, 42)!!;

    assert(CompletionPoint::size <= ulong::size * 2,
        "a completion point fits in two machine words");
    assert(point.is_valid(), "a non-zero queue sequence is valid");
    assert(completion_point_device(point) == device,
        "the point retains device ownership");
    assert(completion_point_queue_id(point) == queue.id,
        "the point retains the compact queue identity");
    assert(completion_point_sequence(point) == 42,
        "the point retains the submission sequence");
}

fn void completion_points_reject_reserved_values() @test {
    Device device = { .index = 3, .generation = 7 };
    Queue queue = {
        .device = device,
        .id     = 0,
        .roles  = { .graphics },
    };

    assert(!COMPLETION_POINT_INVALID.is_valid(),
        "the zero completion point is invalid");
    assert(@catch(make_completion_point(queue, 0)) == INVALID_ARGUMENT,
        "sequence zero remains reserved");
    queue.id = MAX_SELECTED_QUEUE_IDENTITIES;
    assert(@catch(make_completion_point(queue, 1)) == INVALID_HANDLE,
        "an unencodable queue identity is rejected");
}

fn void completion_sequences_are_monotonic_and_stop_before_wrap() @test {
    Device device = { .index = 3, .generation = 7 };
    Queue queue = {
        .device = device,
        .id     = 0,
        .roles  = { .graphics },
    };
    ulong sequence;

    CompletionPoint first = next_completion_point(queue, &sequence)!!;
    CompletionPoint second = next_completion_point(queue, &sequence)!!;
    assert(completion_point_sequence(first) == 1,
        "the first queue submission uses sequence one");
    assert(completion_point_sequence(second) == 2,
        "queue submission sequences increase monotonically");

    sequence = MAX_COMPLETION_SEQUENCE;
    assert(@catch(next_completion_point(queue, &sequence)) == SLOT_TABLE_FULL,
        "sequence exhaustion faults before wrap");
    assert(sequence == MAX_COMPLETION_SEQUENCE,
        "sequence exhaustion does not mutate queue state");
}

fn void completion_points_poll_and_wait_without_consumption() @test {
    CompletionFixture fixture = { .current = 6 };
    Device device = register_device(
        backend:       BackendKind.VULKAN,
        caps:          {},
        vtable:        &completion_fixture_vtable,
        backend_state: &fixture,
    )!!;
    Queue queue = get_queue(&device, QueueKind.GRAPHICS)!!;
    CompletionPoint point = make_completion_point(queue, 7)!!;
    CompletionPoint original = point;

    assert(!poll_completion(point)!!,
        "poll reports an incomplete queue sequence without blocking");
    fixture.current = 7;
    assert(poll_completion(point)!!,
        "poll reports an already-complete queue sequence");
    fixture.wait_fault = WAIT_TIMEOUT;
    assert(@catch(wait_completion(point, 25)) == WAIT_TIMEOUT,
        "wait timeout remains retryable");
    assert(point == original,
        "poll and wait leave the reusable completion point unchanged");
    fixture.wait_fault = {};
    wait_completion(point, 50)!!;

    assert(fixture.poll_calls == 2,
        "both polls reach the owning backend");
    assert(fixture.wait_calls == 2,
        "a timed-out point can be waited again");
    assert(fixture.last_queue_id == queue.id,
        "host waits resolve the point's queue identity");
    assert(fixture.last_sequence == 7,
        "host waits preserve the point's sequence");
    assert(fixture.last_timeout_ns == 50,
        "host waits preserve the caller's timeout");

    destroy_device(&device)!!;
}

fn void completion_points_reject_stale_and_foreign_ownership() @test {
    CompletionFixture first_fixture;
    CompletionFixture second_fixture;
    Device first = register_device(
        backend:       BackendKind.VULKAN,
        caps:          {},
        vtable:        &completion_fixture_vtable,
        backend_state: &first_fixture,
    )!!;
    Device second = register_device(
        backend:       BackendKind.VULKAN,
        caps:          {},
        vtable:        &completion_fixture_vtable,
        backend_state: &second_fixture,
    )!!;
    Queue first_queue = get_queue(&first, QueueKind.GRAPHICS)!!;
    CompletionPoint point = make_completion_point(first_queue, 1)!!;

    DeviceOperation second_operation = begin_device_operation(&second)!!;
    assert(
        @catch(validate_completion_point(&second_operation, &second, point))
            == INVALID_HANDLE,
        "a point cannot be used with another device"
    );
    end_device_operation(&second_operation);

    CompletionPoint malformed = make_completion_point({
        .device = first,
        .id     = 1,
        .roles  = { .graphics },
    }, 1)!!;
    assert(@catch(poll_completion(malformed)) == INVALID_HANDLE,
        "an unselected queue identity faults before backend work");
    assert(first_fixture.poll_calls == 0,
        "malformed points do not reach the backend");

    destroy_device(&first)!!;
    assert(@catch(poll_completion(point)) == INVALID_HANDLE,
        "a point becomes stale with its device");
    destroy_device(&second)!!;
}

fn void completion_point_creation_allocates_no_public_object() @test {
    CompletionFixture fixture;
    Device device = register_device(
        backend:       BackendKind.VULKAN,
        caps:          {},
        vtable:        &completion_fixture_vtable,
        backend_state: &fixture,
    )!!;
    Queue queue = get_queue(&device, QueueKind.GRAPHICS)!!;
    DeviceSlot* slot = &device_slots[device.index];
    ulong children_before = slot.live_children.load(AtomicOrdering.ACQUIRE);
    ulong sequence;

    for (uint index = 0; index < 1024; index++) {
        (void)next_completion_point(queue, &sequence)!!;
    }

    ulong children_after = slot.live_children.load(AtomicOrdering.ACQUIRE);
    assert(children_after == children_before,
        "creating completion values allocates no device child");
    destroy_device(&device)!!;
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

fn void queue_access_requires_the_recording_role() @test {
    QueueRoles graphics_and_transfer = { .graphics, .transfer };

    assert(queue_access_allows(graphics_and_transfer, QueueKind.GRAPHICS));
    assert(queue_access_allows(graphics_and_transfer, QueueKind.TRANSFER));
    assert(!queue_access_allows(graphics_and_transfer, QueueKind.COMPUTE));
    assert(!queue_access_allows({}, QueueKind.GRAPHICS));
    assert(!queue_access_allows((QueueRoles)0x8, QueueKind.GRAPHICS));
}
