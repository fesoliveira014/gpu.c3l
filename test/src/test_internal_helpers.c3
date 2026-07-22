module gpu;

import gpu::internal @public;
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
        (CompletionFixture*)gpu::internal::device_backend_state_ptr(device);
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
        (CompletionFixture*)gpu::internal::device_backend_state_ptr(device);
    if (fixture == null) return INVALID_HANDLE~;
    fixture.wait_calls++;
    fixture.last_queue_id = queue_id;
    fixture.last_sequence = sequence;
    fixture.last_timeout_ns = timeout_ns;
    if (fixture.wait_fault) return fixture.wait_fault~;
}

gpu::internal::BackendVTable completion_fixture_vtable = {
    .destroy_device  = &completion_fixture_destroy,
    .poll_completion = &completion_fixture_poll,
    .wait_completion = &completion_fixture_wait,
};

fn void range_helper() @test {
    assert(gpu::internal::range_fits(256, 0, 256), "exact fit passes");
    assert(gpu::internal::range_fits(256, 256, 0), "to-end at limit passes offset check");
    assert(!gpu::internal::range_fits(256, 257, 0), "offset past limit fails");
    assert(!gpu::internal::range_fits(256, 0, 257), "oversized fails");
    assert(!gpu::internal::range_fits(256, 200, usz::max), "overflowing size fails");
    assert(gpu::internal::range_fits(256, 255, 1), "last byte fits");
}

fn void completion_points_pack_queue_identity_and_sequence() @test {
    Device device = { .index = 3, .generation = 7 };
    Queue queue = {
        .device = device,
        .id     = gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES - 1,
        .roles  = { .transfer },
    };

    CompletionPoint point = gpu::internal::make_completion_point(queue, 42)!!;

    assert(CompletionPoint::size <= ulong::size * 2,
        "a completion point fits in two machine words");
    assert(point.is_valid(), "a non-zero queue sequence is valid");
    assert(gpu::internal::completion_point_device(point) == device,
        "the point retains device ownership");
    assert(gpu::internal::completion_point_queue_id(point) == queue.id,
        "the point retains the compact queue identity");
    assert(gpu::internal::completion_point_sequence(point) == 42,
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
    assert(@catch(gpu::internal::make_completion_point(queue, 0)) == INVALID_ARGUMENT,
        "sequence zero remains reserved");
    queue.id = gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES;
    assert(@catch(gpu::internal::make_completion_point(queue, 1)) == INVALID_HANDLE,
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

    CompletionPoint first = gpu::internal::next_completion_point(queue, &sequence)!!;
    CompletionPoint second = gpu::internal::next_completion_point(queue, &sequence)!!;
    assert(gpu::internal::completion_point_sequence(first) == 1,
        "the first queue submission uses sequence one");
    assert(gpu::internal::completion_point_sequence(second) == 2,
        "queue submission sequences increase monotonically");

    sequence = gpu::internal::MAX_COMPLETION_SEQUENCE;
    assert(@catch(gpu::internal::next_completion_point(queue, &sequence)) == SLOT_TABLE_FULL,
        "sequence exhaustion faults before wrap");
    assert(sequence == gpu::internal::MAX_COMPLETION_SEQUENCE,
        "sequence exhaustion does not mutate queue state");
}

fn void completion_points_poll_and_wait_without_consumption() @test {
    CompletionFixture fixture = { .current = 6 };
    Device device = gpu::internal::register_device(
        backend:       BackendKind.VULKAN,
        caps:          {},
        vtable:        &completion_fixture_vtable,
        backend_state: &fixture,
    )!!;
    Queue queue = get_queue(&device, QueueKind.GRAPHICS)!!;
    CompletionPoint point = gpu::internal::make_completion_point(queue, 7)!!;
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
    wait_completion(point)!!;

    assert(fixture.poll_calls == 2,
        "both polls reach the owning backend");
    assert(fixture.wait_calls == 3,
        "a timed-out point can be waited again");
    assert(fixture.last_queue_id == queue.id,
        "host waits resolve the point's queue identity");
    assert(fixture.last_sequence == 7,
        "host waits preserve the point's sequence");
    assert(fixture.last_timeout_ns == ulong::max,
        "the default host wait has no timeout");

    destroy_device(&device)!!;
}

fn void completion_points_reject_stale_and_foreign_ownership() @test {
    CompletionFixture first_fixture;
    CompletionFixture second_fixture;
    Device first = gpu::internal::register_device(
        backend:       BackendKind.VULKAN,
        caps:          {},
        vtable:        &completion_fixture_vtable,
        backend_state: &first_fixture,
    )!!;
    Device second = gpu::internal::register_device(
        backend:       BackendKind.VULKAN,
        caps:          {},
        vtable:        &completion_fixture_vtable,
        backend_state: &second_fixture,
    )!!;
    Queue first_queue = get_queue(&first, QueueKind.GRAPHICS)!!;
    CompletionPoint point = gpu::internal::make_completion_point(first_queue, 1)!!;

    gpu::internal::DeviceOperation second_operation = gpu::internal::begin_device_operation(&second)!!;
    assert(
        @catch(gpu::internal::validate_completion_point(&second_operation, &second, point))
            == INVALID_HANDLE,
        "a point cannot be used with another device"
    );
    gpu::internal::end_device_operation(&second_operation);

    CompletionPoint malformed = gpu::internal::make_completion_point({
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
    Device device = gpu::internal::register_device(
        backend:       BackendKind.VULKAN,
        caps:          {},
        vtable:        &completion_fixture_vtable,
        backend_state: &fixture,
    )!!;
    Queue queue = get_queue(&device, QueueKind.GRAPHICS)!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    ulong children_before = slot.live_children.load(AtomicOrdering.ACQUIRE);
    ulong sequence;

    for (uint index = 0; index < 1024; index++) {
        (void)gpu::internal::next_completion_point(queue, &sequence)!!;
    }

    ulong children_after = slot.live_children.load(AtomicOrdering.ACQUIRE);
    assert(children_after == children_before,
        "creating completion values allocates no device child");
    destroy_device(&device)!!;
}

fn void queue_selection_aliases_roles_by_default() @test {
    DeviceCaps caps;
    gpu::internal::QueueSelection selection = gpu::internal::default_queue_selection(&caps);

    assert(selection.counts.graphics == 1, "default selection exposes graphics role");
    assert(selection.counts.compute == 1, "default selection exposes compute role");
    assert(selection.counts.transfer == 1, "default selection exposes transfer role");
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS, 0)
            == gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE, 0),
        "default compute aliases graphics identity"
    );
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE, 0)
            == gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER, 0),
        "default transfer aliases graphics identity"
    );
    assert(
        gpu::internal::queue_roles_equal(
            gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS, 0),
            gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE, 0),
        ),
        "aliased roles share one role mask"
    );
    assert(
        gpu::internal::queue_roles_equal(
            gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE, 0),
            gpu::internal::queue_selection_roles(&selection, QueueKind.TRANSFER, 0),
        ),
        "transfer role shares the aliased mask"
    );
}

fn void queue_selection_splits_async_compute() @test {
    DeviceCaps caps = { .async_compute = true };
    gpu::internal::QueueSelection selection = gpu::internal::default_queue_selection(&caps);

    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS, 0)
            != gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE, 0),
        "async compute uses a distinct identity"
    );
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER, 0)
            == gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS, 0),
        "transfer still aliases graphics by default"
    );
    assert(
        gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS, 0).graphics,
        "graphics identity supports graphics"
    );
    assert(
        !gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS, 0).compute,
        "graphics identity excludes distinct compute"
    );
    assert(
        gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE, 0).compute,
        "compute identity supports compute"
    );
    assert(
        !gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE, 0).graphics,
        "compute identity excludes graphics"
    );
}

fn void queue_selection_splits_distinct_transfer() @test {
    gpu::internal::QueueSelection selection = gpu::internal::queue_selection_for_aliases(
        compute_aliases_graphics:  true,
        transfer_aliases_graphics: false,
        transfer_aliases_compute:  false,
    );

    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS, 0)
            == gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE, 0),
        "graphics and compute alias"
    );
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER, 0)
            != gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS, 0),
        "transfer has a distinct identity"
    );
    assert(
        gpu::internal::queue_roles_equal(
            gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS, 0),
            gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE, 0),
        ),
        "aliased roles share canonical metadata"
    );
    assert(
        gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS, 0).graphics,
        "shared identity reports graphics"
    );
    assert(
        gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS, 0).compute,
        "shared identity reports compute"
    );
    assert(
        !gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS, 0).transfer,
        "shared identity excludes transfer"
    );
    assert(
        gpu::internal::queue_selection_roles(&selection, QueueKind.TRANSFER, 0).transfer,
        "transfer identity reports transfer"
    );
}

fn void queue_selection_represents_three_distinct_queues() @test {
    gpu::internal::QueueSelection selection = gpu::internal::queue_selection_for_aliases(
        compute_aliases_graphics:  false,
        transfer_aliases_graphics: false,
        transfer_aliases_compute:  false,
    );

    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS, 0)
            != gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE, 0),
        "graphics and compute differ"
    );
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS, 0)
            != gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER, 0),
        "graphics and transfer differ"
    );
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE, 0)
            != gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER, 0),
        "compute and transfer differ"
    );
    assert(
        (uint)gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS, 0)
            == (uint)(QueueRoles){ .graphics },
        "graphics identity reports one role"
    );
    assert(
        (uint)gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE, 0)
            == (uint)(QueueRoles){ .compute },
        "compute identity reports one role"
    );
    assert(
        (uint)gpu::internal::queue_selection_roles(&selection, QueueKind.TRANSFER, 0)
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
    gpu::internal::QueueSelection selection = gpu::internal::queue_selection_from_native_ids(
        requirements: requirements,
        graphics:     graphics[..],
        compute:      compute[..],
        transfer:     transfer[..],
    )!!;

    uint graphics_0 = gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS, 0);
    uint graphics_1 = gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS, 1);
    uint compute_0 = gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE, 0);
    uint compute_1 = gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE, 1);
    uint transfer_0 = gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER, 0);
    assert(selection.identity_count == 3, "three native identities are stored");
    assert(graphics_0 != graphics_1, "one role exposes distinct identities");
    assert(compute_0 == graphics_1, "compute aliases the second graphics queue");
    assert(
        compute_1 != graphics_0 && compute_1 != graphics_1,
        "new native queue gets a new identity"
    );
    assert(transfer_0 == graphics_0, "transfer aliases the first graphics queue");
    assert(
        (uint)gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS, 0)
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
        @catch(gpu::internal::queue_selection_from_native_ids(
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
    gpu::internal::QueueSelection selection = gpu::internal::queue_selection_for_requirements(
        requirements:               requirements,
        compute_aliases_graphics:    true,
        transfer_aliases_graphics:   true,
        transfer_aliases_compute:    false,
    )!!;

    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE, 0)
            != gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER, 0),
        "an absent graphics role cannot cause an alias"
    );
}

fn void queue_access_requires_the_recording_role() @test {
    QueueRoles graphics_and_transfer = { .graphics, .transfer };

    assert(gpu::internal::queue_access_allows(graphics_and_transfer, QueueKind.GRAPHICS));
    assert(gpu::internal::queue_access_allows(graphics_and_transfer, QueueKind.TRANSFER));
    assert(!gpu::internal::queue_access_allows(graphics_and_transfer, QueueKind.COMPUTE));
    assert(!gpu::internal::queue_access_allows({}, QueueKind.GRAPHICS));
    assert(!gpu::internal::queue_access_allows((QueueRoles)0x8, QueueKind.GRAPHICS));
}

fn void submit_wait_mask_shape_is_backend_independent() @test {
    assert(gpu::internal::submit_wait_stage_mask_valid({ .all }));
    assert(gpu::internal::submit_wait_stage_mask_valid({ .transfer, .compute }));
    assert(!gpu::internal::submit_wait_stage_mask_valid({}));
    assert(!gpu::internal::submit_wait_stage_mask_valid({ .host }));
    assert(!gpu::internal::submit_wait_stage_mask_valid({ .present }));
    assert(!gpu::internal::submit_wait_stage_mask_valid({ .all, .compute }));
    assert(!gpu::internal::submit_wait_stage_mask_valid((StageMask)(1u << 9)));
}
