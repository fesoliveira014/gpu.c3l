module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;

fn RuntimeDesc queue_runtime_desc() {
    return {
        .application_name = "gpu_queue_test",
    };
}

fn Device create_queue_test_device(Runtime* runtime) {
    Adapter adapter = gpu::enumerate_adapters(runtime)!!.get(0)!!;
    return gpu::create_device(&adapter)!!;
}


fn bool queue_roles_are_all(gpu::QueueRoles roles) {
    return roles.graphics && roles.compute && roles.transfer;
}

fn bool queue_roles_are_graphics_transfer(gpu::QueueRoles roles) {
    return roles.graphics && !roles.compute && roles.transfer;
}

fn bool queue_roles_are_compute(gpu::QueueRoles roles) {
    return !roles.graphics && roles.compute && !roles.transfer;
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

    gpu::QueueRoles selected = gpu::get_device_caps(&first)!!.queues;
    assert(queue_roles_are_all(selected), "default device selects all semantic roles");

    gpu::Queue first_graphics = gpu::get_queue(&first, gpu::QueueKind.GRAPHICS)!!;
    gpu::Queue first_compute = gpu::get_queue(&first, gpu::QueueKind.COMPUTE)!!;
    gpu::Queue first_transfer = gpu::get_queue(&first, gpu::QueueKind.TRANSFER)!!;
    gpu::Queue second_graphics = gpu::get_queue(&second, gpu::QueueKind.GRAPHICS)!!;

    assert(first_graphics.is_valid(), "queried queue token is valid");
    assert(first_graphics == first_compute, "aliased graphics and compute roles share one queue identity");
    assert(first_compute == first_transfer, "aliased compute and transfer roles share one queue identity");
    assert(first_graphics != second_graphics, "queue tokens retain exact device ownership");
    assert(first_graphics.device == first, "queue token names its owning device");
    assert(first_graphics.id == 0, "queue token records its device-local identity");
    assert(queue_roles_are_all(first_graphics.roles), "queue token reports all supported semantic roles");

    gpu::QueueInfo info = gpu::get_queue_info(&first, first_graphics)!!;
    assert(info.id == 0, "queue info reports the device-local identity");
    assert(queue_roles_are_all(info.roles), "queue info reports aliased semantic roles");

    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void queue_queries_preserve_distinct_compute_identity() @test {
    gpu::internal::vk::VkDeviceState state;
    gpu::internal::QueueSelection selection = gpu::internal::queue_selection_for_aliases(
        compute_aliases_graphics:  false,
        transfer_aliases_graphics: true,
        transfer_aliases_compute:  false,
    );

    gpu::Device device = gpu::internal::register_device(
        state:  &state,
        caps:   { .async_compute = true },
        queues: selection,
    )!!;

    gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::Queue compute = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::Queue transfer = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
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
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc compute_desc = {
        .queues = { .required = { .compute } },
    };
    Device first = gpu::create_device(&adapter, &compute_desc)!!;
    Device second = create_queue_test_device(&runtime);
    gpu::Queue compute = gpu::get_queue(&first, gpu::QueueKind.COMPUTE)!!;

    assert(@catch(gpu::get_queue(&first, gpu::QueueKind.GRAPHICS))
        == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::get_queue_info(&second, compute))
        == gpu::INVALID_HANDLE);

    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void queue_queries_reject_stale_devices() @test {
    RuntimeDesc desc = queue_runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Device device = create_queue_test_device(&runtime);
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::destroy_device(&device)!!;
    assert(@catch(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS))
        == gpu::INVALID_HANDLE);
    assert(@catch(gpu::get_queue_info(&device, queue))
        == gpu::INVALID_HANDLE);

    gpu::destroy_runtime(&runtime)!!;
}

fn void selected_queue_roles_round_trip_canonical_identities() @test {
    gpu::internal::vk::VkDeviceState state;
    gpu::QueueRequest request = {
        .required = { .graphics, .compute, .transfer },
    };
    gpu::internal::QueueSelection selection =
        gpu::internal::queue_selection_from_native_ids(
            request:  request,
            graphics: 10,
            compute:  20,
            transfer: 10,
        )!!;
    gpu::Device device = gpu::internal::register_device(
        state:  &state,
        caps:   {},
        queues: selection,
    )!!;
    defer gpu::destroy_device(&device)!!;

    gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::Queue compute = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::Queue transfer = gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    assert(graphics != compute, "distinct native choices publish distinct identities");
    assert(graphics == transfer, "aliased native choices share one identity");
    assert(gpu::get_queue_info(&device, graphics)!!.id == graphics.id);
    assert(gpu::get_device_caps(&device)!!.queues.graphics);
    assert(gpu::get_device_caps(&device)!!.queues.compute);
    assert(gpu::get_device_caps(&device)!!.queues.transfer);
}

fn void queue_registration_rejects_malformed_identity_roles() @test {
    gpu::internal::vk::VkDeviceState state;
    gpu::internal::QueueSelection selection =
        gpu::internal::queue_selection_for_aliases(
            compute_aliases_graphics:  true,
            transfer_aliases_graphics: true,
            transfer_aliases_compute:  true,
        );
    selection.identity_roles[0].compute = false;

    assert(
        @catch(gpu::internal::register_device(
            state:  &state,
            caps:   {},
            queues: selection,
        )) == gpu::INVALID_ARGUMENT,
        "canonical identity roles must match the selected role mapping"
    );
}

fn void device_exposes_only_requested_queue_roles() @test {
    RuntimeDesc runtime_desc = queue_runtime_desc();
    Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    Adapter adapter = gpu::enumerate_adapters(&runtime)!!.get(0)!!;
    DeviceDesc desc = {
        .queues = { .required = { .compute } },
    };
    Device device = gpu::create_device(&adapter, &desc)!!;
    defer gpu::destroy_device(&device)!!;

    QueueRoles selected = gpu::get_device_caps(&device)!!.queues;
    Queue compute = gpu::get_queue(&device, QueueKind.COMPUTE)!!;
    QueueInfo info = gpu::get_queue_info(&device, compute)!!;
    assert(!selected.graphics, "unrequested graphics stays hidden");
    assert(selected.compute, "requested compute is exposed");
    assert(!selected.transfer, "unrequested transfer stays hidden");
    assert((uint)info.roles == (uint)(QueueRoles){ .compute });
    assert(@catch(gpu::get_queue(&device, QueueKind.GRAPHICS))
        == gpu::INVALID_ARGUMENT);
}
