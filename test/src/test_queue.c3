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

fn void queue_tokens_are_compact_owner_tokens() @test {
    assert(gpu::Queue::size <= ulong::size * 2, "gpu::Queue fits in two machine words");
    assert(!gpu::QUEUE_INVALID.is_valid(), "zero queue token is invalid");
}

fn void queue_queries_return_owner_bearing_tokens() @test {
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
    assert(first_graphics.kind == gpu::QueueKind.GRAPHICS, "queue token records its semantic role");
    assert(first_graphics.index == 0, "queue token records its semantic index");
    assert(first_graphics.owner == first_compute.owner, "same-device queue roles share one queue owner");
    assert(first_compute.owner == first_transfer.owner, "same-device transfer role shares one queue owner");
    assert(first_graphics.owner != second_graphics.owner, "queue tokens retain exact device ownership");

    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
    gpu::destroy_runtime(&runtime)!!;
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

