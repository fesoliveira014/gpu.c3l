module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;

fn RuntimeDesc queue_runtime_desc() {
    return {
        .backend          = BackendKind.VULKAN,
        .application_name = "gpu_queue_test",
    };
}

fn Device create_queue_test_device(Runtime* runtime) {
    Adapter adapter = gpu::enumerate_adapters(runtime)!!.get(0)!!;
    DeviceRequest request = gpu::strict_device_request();
    return gpu::create_device(&adapter, &request)!!;
}

fn void queue_test_destroy(void* backend_state) {
    (void)backend_state;
}

gpu::internal::BackendVTable queue_test_vtable = {
    .destroy_device = &queue_test_destroy,
};

fn bool queue_roles_are_all(gpu::QueueRoles roles) {
    return roles.graphics && roles.compute && roles.transfer;
}

fn bool queue_roles_are_graphics_transfer(gpu::QueueRoles roles) {
    return roles.graphics && !roles.compute && roles.transfer;
}

fn bool queue_roles_are_compute(gpu::QueueRoles roles) {
    return !roles.graphics && roles.compute && !roles.transfer;
}

fn bool queue_counts_equal(gpu::QueueCounts first, gpu::QueueCounts second) {
    return first.graphics == second.graphics
        && first.compute == second.compute
        && first.transfer == second.transfer;
}

fn void queue_tokens_are_compact_device_owned_tokens() @test {
    assert(gpu::Queue::size <= ulong::size * 2, "gpu::Queue fits in two machine words");
    assert(!gpu::QUEUE_INVALID.is_valid(), "zero queue token is invalid");
}

fn void queue_queries_return_device_owned_identities() @test {
    RuntimeDesc desc = queue_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Device first = create_queue_test_device(&runtime);
    Device second = create_queue_test_device(&runtime);

    gpu::QueueCounts counts = gpu::get_queue_counts(&first)!!;
    gpu::QueueCounts caps_counts = gpu::get_device_caps(&first)!!.queues;
    assert(counts.graphics == 1, "stub exposes one graphics queue");
    assert(counts.compute == 1, "stub exposes one compute queue");
    assert(counts.transfer == 1, "stub exposes one transfer queue");
    assert(queue_counts_equal(counts, caps_counts), "queue query mirrors device capabilities");

    gpu::Queue first_graphics = gpu::get_queue(&first, gpu::QueueKind.GRAPHICS, 0)!!;
    gpu::Queue first_compute = gpu::get_queue(&first, gpu::QueueKind.COMPUTE, 0)!!;
    gpu::Queue first_transfer = gpu::get_queue(&first, gpu::QueueKind.TRANSFER, 0)!!;
    gpu::Queue second_graphics = gpu::get_queue(&second, gpu::QueueKind.GRAPHICS, 0)!!;

    assert(first_graphics.is_valid(), "queried queue token is valid");
    assert(first_graphics == first_compute, "aliased graphics and compute roles share one queue identity");
    assert(first_compute == first_transfer, "aliased compute and transfer roles share one queue identity");
    assert(first_graphics != second_graphics, "queue tokens retain exact device ownership");
    assert(first_graphics.device == first, "queue token names its owning device");
    assert(first_graphics.id == 0, "queue token records its device-local index");
    assert(queue_roles_are_all(first_graphics.roles), "queue token reports all supported semantic roles");

    gpu::QueueInfo info = gpu::get_queue_info(&first, first_graphics)!!;
    assert(info.id == 0, "queue info reports the device-local index");
    assert(queue_roles_are_all(info.roles), "queue info reports aliased semantic roles");

    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void queue_queries_preserve_distinct_compute_identity() @test {
    int backend_state;
    gpu::internal::QueueSelection selection = gpu::internal::queue_selection_for_aliases(
        compute_aliases_graphics:  false,
        transfer_aliases_graphics: true,
        transfer_aliases_compute:  false,
    );

    gpu::Device device = gpu::internal::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          { .async_compute = true },
        vtable:        &queue_test_vtable,
        backend_state: &backend_state,
        queues:        selection,
    )!!;

    gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS, 0)!!;
    gpu::Queue compute = gpu::get_queue(&device, gpu::QueueKind.COMPUTE, 0)!!;
    gpu::Queue transfer = gpu::get_queue(&device, gpu::QueueKind.TRANSFER, 0)!!;
    assert(graphics != compute, "distinct compute receives its own queue identity");
    assert(graphics == transfer, "aliased transfer shares the graphics queue identity");
    assert(queue_roles_are_graphics_transfer(graphics.roles), "graphics identity reports aliased transfer role");
    assert(queue_roles_are_compute(compute.roles), "compute identity reports only compute role");

    gpu::Queue forged = compute;
    forged.roles.graphics = true;
    assert(@catch(gpu::get_queue_info(&device, forged)) == gpu::INVALID_HANDLE);

    gpu::destroy_device(&device)!!;
}

fn void queue_queries_reject_unavailable_and_foreign_tokens() @test {
    RuntimeDesc desc = queue_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Device first = create_queue_test_device(&runtime);
    Device second = create_queue_test_device(&runtime);
    gpu::Queue first_graphics = gpu::get_queue(&first, gpu::QueueKind.GRAPHICS, 0)!!;

    assert(@catch(gpu::get_queue(&first, gpu::QueueKind.GRAPHICS, 1)) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::get_queue_info(&second, first_graphics)) == gpu::INVALID_HANDLE);

    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void queue_queries_reject_stale_devices() @test {
    RuntimeDesc desc = queue_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Device device = create_queue_test_device(&runtime);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS, 0)!!;

    gpu::destroy_device(&device)!!;
    assert(@catch(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS, 0)) == gpu::INVALID_HANDLE);
    assert(@catch(gpu::get_queue_info(&device, queue)) == gpu::INVALID_HANDLE);

    gpu::destroy_runtime(&runtime)!!;
}

fn void selected_queue_counts_round_trip_stored_identities() @test {
    int backend_state;
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = 2,
            .compute  = 1,
            .transfer = 1,
        },
    };
    ulong[2] graphics = { 10, 20 };
    ulong[1] compute = { 20 };
    ulong[1] transfer = { 10 };
    gpu::internal::QueueSelection selection = gpu::internal::queue_selection_from_native_ids(
        requirements: requirements,
        graphics:     graphics[..],
        compute:      compute[..],
        transfer:     transfer[..],
    )!!;
    gpu::Device device = gpu::internal::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &queue_test_vtable,
        backend_state: &backend_state,
        queues:        selection,
    )!!;
    defer gpu::destroy_device(&device)!!;

    gpu::QueueCounts counts = gpu::get_queue_counts(&device)!!;
    gpu::Queue graphics_0 = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS, 0)!!;
    gpu::Queue graphics_1 = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS, 1)!!;
    gpu::Queue compute_0 = gpu::get_queue(&device, gpu::QueueKind.COMPUTE, 0)!!;
    gpu::Queue transfer_0 = gpu::get_queue(&device, gpu::QueueKind.TRANSFER, 0)!!;
    assert(counts.graphics == 2, "both graphics identities are exposed");
    assert(counts.compute == 1, "the compute identity is exposed");
    assert(counts.transfer == 1, "the transfer identity is exposed");
    assert(graphics_0 != graphics_1, "same-role queues have distinct identities");
    assert(graphics_1 == compute_0, "compute aliases the second graphics queue");
    assert(graphics_0 == transfer_0, "transfer aliases the first graphics queue");
    assert(gpu::get_queue_info(&device, graphics_0)!!.id == graphics_0.id);
    assert(gpu::get_queue_info(&device, graphics_1)!!.id == graphics_1.id);
    assert(
        @catch(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS, 2))
            == gpu::INVALID_ARGUMENT,
        "only stored indices are queryable"
    );
}

fn void selected_queue_counts_reach_encoded_capacity() @test {
    int backend_state;
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE,
            .compute  = gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE,
            .transfer = gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE,
        },
    };
    ulong[gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE] graphics;
    ulong[gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE] compute;
    ulong[gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE] transfer;
    for (uint index = 0; index < gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE; index++) {
        graphics[index] = (ulong)index;
        compute[index] = (ulong)(gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE + index);
        transfer[index] = (ulong)(2 * gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE + index);
    }
    gpu::internal::QueueSelection selection = gpu::internal::queue_selection_from_native_ids(
        requirements: requirements,
        graphics:     graphics[..],
        compute:      compute[..],
        transfer:     transfer[..],
    )!!;
    gpu::Device device = gpu::internal::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &queue_test_vtable,
        backend_state: &backend_state,
        queues:        selection,
    )!!;
    defer gpu::destroy_device(&device)!!;

    gpu::QueueCounts counts = gpu::get_queue_counts(&device)!!;
    gpu::Queue last = gpu::get_queue(
        &device,
        gpu::QueueKind.TRANSFER,
        gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE - 1,
    )!!;
    gpu::QueueInfo info = gpu::get_queue_info(&device, last)!!;
    assert(counts.graphics == gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE);
    assert(counts.compute == gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE);
    assert(counts.transfer == gpu::internal::MAX_REQUESTED_QUEUES_PER_ROLE);
    assert(last.id == gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES - 1);
    assert(info.id == last.id);
    assert(
        !info.roles.graphics && !info.roles.compute && info.roles.transfer,
        "the final compact identity preserves its semantic role"
    );
}

fn void queue_registration_rejects_duplicate_role_identity() @test {
    int backend_state;
    gpu::QueueRequirements requirements = { .counts = { .graphics = 2 } };
    ulong[2] graphics = { 10, 20 };
    gpu::internal::QueueSelection selection = gpu::internal::queue_selection_from_native_ids(
        requirements: requirements,
        graphics:     graphics[..],
        compute:      {},
        transfer:     {},
    )!!;
    selection.graphics_ids[1] = selection.graphics_ids[0];

    assert(
        @catch(gpu::internal::register_device(
            backend:       gpu::BackendKind.VULKAN,
            caps:          {},
            vtable:        &queue_test_vtable,
            backend_state: &backend_state,
            queues:        selection,
        )) == gpu::INVALID_ARGUMENT,
        "one role cannot publish the same identity twice"
    );
}

fn void device_exposes_only_requested_queue_roles() @test {
    RuntimeDesc runtime_desc = queue_runtime_desc();
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    QueueRequirements requirements = { .counts = { .compute = 1 } };
    DeviceRequest request = gpu::request_queues(
        gpu::strict_device_request(),
        requirements,
    )!!;
    Device device = gpu::create_device(&adapter, &request)!!;
    defer gpu::destroy_device(&device)!!;

    QueueCounts counts = gpu::get_queue_counts(&device)!!;
    Queue compute = gpu::get_queue(&device, QueueKind.COMPUTE)!!;
    QueueInfo info = gpu::get_queue_info(&device, compute)!!;
    assert(counts.graphics == 0, "unrequested graphics queues stay hidden");
    assert(counts.compute == 1, "requested compute queue is exposed");
    assert(counts.transfer == 0, "unrequested transfer queues stay hidden");
    assert((uint)info.roles == (uint)(QueueRoles){ .compute }, "queue roles match the request");
    assert(
        @catch(gpu::get_queue(&device, QueueKind.GRAPHICS)) == gpu::INVALID_ARGUMENT,
        "unrequested roles have no public queue"
    );


}
