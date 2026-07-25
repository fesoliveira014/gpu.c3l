module gpu;

import gpu::internal @public;
import gpu::internal::vk @public;

struct CompletionFixture {
    gpu::internal::vk::VkDeviceState state;
}


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
    CompletionFixture fixture = {
        .state = { .completion_current = 6 },
    };
    Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
    )!!;
    Queue queue = get_queue(&device, QueueKind.GRAPHICS)!!;
    CompletionPoint point = gpu::internal::make_completion_point(queue, 7)!!;
    CompletionPoint original = point;

    assert(!poll_completion(point)!!,
        "poll reports an incomplete queue sequence without blocking");
    fixture.state.completion_current = 7;
    assert(poll_completion(point)!!,
        "poll reports an already-complete queue sequence");
    fixture.state.completion_wait_fault = WAIT_TIMEOUT;
    assert(@catch(wait_completion(point, 25)) == WAIT_TIMEOUT,
        "wait timeout remains retryable");
    assert(point == original,
        "poll and wait leave the reusable completion point unchanged");
    fixture.state.completion_wait_fault = {};
    wait_completion(point, 50)!!;
    wait_completion(point)!!;

    assert(fixture.state.completion_poll_calls == 2,
        "both polls reach the owning backend");
    assert(fixture.state.completion_wait_calls == 3,
        "a timed-out point can be waited again");
    assert(fixture.state.completion_last_queue_id == queue.id,
        "host waits resolve the point's queue identity");
    assert(fixture.state.completion_last_sequence == 7,
        "host waits preserve the point's sequence");
    assert(fixture.state.completion_last_timeout_ns == ulong::max,
        "the default host wait has no timeout");

    destroy_device(&device)!!;
}

fn void completion_points_reject_stale_and_foreign_ownership() @test {
    CompletionFixture first_fixture;
    CompletionFixture second_fixture;
    Device first = gpu::internal::register_device(
        state:  &first_fixture.state,
        caps:   {},
    )!!;
    Device second = gpu::internal::register_device(
        state:  &second_fixture.state,
        caps:   {},
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
    assert(first_fixture.state.completion_poll_calls == 0,
        "malformed points do not reach the backend");

    destroy_device(&first)!!;
    assert(@catch(poll_completion(point)) == INVALID_HANDLE,
        "a point becomes stale with its device");
    destroy_device(&second)!!;
}

fn void completion_point_creation_allocates_no_public_object() @test {
    CompletionFixture fixture;
    Device device = gpu::internal::register_device(
        state:  &fixture.state,
        caps:   {},
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

    assert(selection.selected.graphics, "default selection exposes graphics role");
    assert(selection.selected.compute, "default selection exposes compute role");
    assert(selection.selected.transfer, "default selection exposes transfer role");
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS)
            == gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE),
        "default compute aliases graphics identity"
    );
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE)
            == gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER),
        "default transfer aliases graphics identity"
    );
    assert(
        gpu::internal::queue_roles_equal(
            gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS),
            gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE),
        ),
        "aliased roles share one role mask"
    );
    assert(
        gpu::internal::queue_roles_equal(
            gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE),
            gpu::internal::queue_selection_roles(&selection, QueueKind.TRANSFER),
        ),
        "transfer role shares the aliased mask"
    );
}

fn void queue_selection_splits_async_compute() @test {
    DeviceCaps caps = { .async_compute = true };
    gpu::internal::QueueSelection selection = gpu::internal::default_queue_selection(&caps);

    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS)
            != gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE),
        "async compute uses a distinct identity"
    );
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER)
            == gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS),
        "transfer still aliases graphics by default"
    );
    assert(
        gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS).graphics,
        "graphics identity supports graphics"
    );
    assert(
        !gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS).compute,
        "graphics identity excludes distinct compute"
    );
    assert(
        gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE).compute,
        "compute identity supports compute"
    );
    assert(
        !gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE).graphics,
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
        gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS)
            == gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE),
        "graphics and compute alias"
    );
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER)
            != gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS),
        "transfer has a distinct identity"
    );
    assert(
        gpu::internal::queue_roles_equal(
            gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS),
            gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE),
        ),
        "aliased roles share canonical metadata"
    );
    assert(
        gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS).graphics,
        "shared identity reports graphics"
    );
    assert(
        gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS).compute,
        "shared identity reports compute"
    );
    assert(
        !gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS).transfer,
        "shared identity excludes transfer"
    );
    assert(
        gpu::internal::queue_selection_roles(&selection, QueueKind.TRANSFER).transfer,
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
        gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS)
            != gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE),
        "graphics and compute differ"
    );
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.GRAPHICS)
            != gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER),
        "graphics and transfer differ"
    );
    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE)
            != gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER),
        "compute and transfer differ"
    );
    assert(
        (uint)gpu::internal::queue_selection_roles(&selection, QueueKind.GRAPHICS)
            == (uint)(QueueRoles){ .graphics },
        "graphics identity reports one role"
    );
    assert(
        (uint)gpu::internal::queue_selection_roles(&selection, QueueKind.COMPUTE)
            == (uint)(QueueRoles){ .compute },
        "compute identity reports one role"
    );
    assert(
        (uint)gpu::internal::queue_selection_roles(&selection, QueueKind.TRANSFER)
            == (uint)(QueueRoles){ .transfer },
        "transfer identity reports one role"
    );
}

fn void queue_selection_rejects_distinct_role_aliases() @test {
    QueueRequest request = {
        .required = { .graphics, .compute },
        .distinct = { .compute },
    };

    assert(
        @catch(gpu::internal::queue_selection_from_native_ids(
            request:  request,
            graphics: 10,
            compute:  10,
            transfer: 0,
        )) == UNSUPPORTED_FEATURE,
        "a distinct role cannot alias another requested role"
    );
}

fn void queue_selection_ignores_aliases_to_absent_roles() @test {
    QueueRequest request = {
        .required = { .compute, .transfer },
    };
    gpu::internal::QueueSelection selection =
        gpu::internal::queue_selection_for_request(
        request:                    request,
        compute_aliases_graphics:    true,
        transfer_aliases_graphics:   true,
        transfer_aliases_compute:    false,
    )!!;

    assert(
        gpu::internal::queue_selection_id(&selection, QueueKind.COMPUTE)
            != gpu::internal::queue_selection_id(&selection, QueueKind.TRANSFER),
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

fn void completion_wait_scope_shape_is_backend_independent() @test {
    CompletionConsumerFlags draw_arguments = { .draw_arguments };
    assert(gpu::internal::completion_wait_scope_valid({}, draw_arguments));
    assert(gpu::internal::completion_wait_scope_valid(
        { .vertex_shader },
        draw_arguments,
    ));
    assert(gpu::internal::completion_wait_scope_valid({ .all }, {}));
    assert(!gpu::internal::completion_wait_scope_valid({}, {}));
    assert(!gpu::internal::completion_wait_scope_valid(
        { .all },
        draw_arguments,
    ));
    assert(!gpu::internal::completion_wait_scope_valid({ .host }, {}));
    assert(!gpu::internal::completion_wait_scope_valid({ .present }, {}));
    assert(!gpu::internal::completion_wait_scope_valid(
        (StageMask)(1u << 9),
        {},
    ));
    assert(!gpu::internal::completion_wait_scope_valid(
        {},
        (CompletionConsumerFlags)(1u << 1),
    ));
}
