module gpu_test;

import gpu @public;
import gpu::vk @public;

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

gpu::BackendVTable queue_test_vtable = {
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

fn void queue_tokens_are_compact_owner_tokens() @test {
    assert(gpu::Queue::size <= ulong::size * 2, "gpu::Queue fits in two machine words");
    assert(!gpu::QUEUE_INVALID.is_valid(), "zero queue token is invalid");
}

fn void queue_queries_return_owner_bearing_identities() @test {
    RuntimeDesc desc = queue_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Device first = create_queue_test_device(&runtime);
    Device second = create_queue_test_device(&runtime);

    gpu::QueueCounts counts = gpu::get_queue_counts(&first)!!;
    assert(counts.graphics == 1, "stub exposes one graphics queue");
    assert(counts.compute == 1, "stub exposes one compute queue");
    assert(counts.transfer == 1, "stub exposes one transfer queue");

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
    gpu::QueueSelection selection = gpu::default_queue_selection();
    selection.graphics_roles = { .graphics, .transfer };
    selection.compute_id = 1;
    selection.compute_roles = { .compute };
    selection.transfer_id = selection.graphics_id;
    selection.transfer_roles = selection.graphics_roles;

    gpu::Device device = gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          { .async_compute = true },
        vtable:        &queue_test_vtable,
        backend_state: &backend_state,
        queues:        selection,
    )!!;

    gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS, 0)!!;
    gpu::Queue compute = gpu::get_queue(&device, gpu::QueueKind.COMPUTE, 0)!!;
    gpu::Queue transfer = gpu::get_queue(&device, gpu::QueueKind.TRANSFER, 0)!!;
    assert(graphics.id != compute.id, "distinct compute receives its own queue identity");
    assert(graphics.id == transfer.id, "aliased transfer shares the graphics queue identity");
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

