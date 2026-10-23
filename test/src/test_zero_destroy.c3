module gpu_test;

import gpu::internal @public;
import gpu @public;

fn void zero_handles_destroy_without_effect() @test {
    DeviceFixture fixture;
    gpu::Device device = fixture_device(&fixture);
    defer gpu::destroy_device(&device)!!;
    gpu::internal::DeviceSlot* slot = &gpu::internal::device_slots[device.index];
    ulong control_before = slot.control.load(AtomicOrdering.ACQUIRE);
    ulong children_before = slot.live_children.load(AtomicOrdering.ACQUIRE);

    gpu::Runtime runtime;
    gpu::Surface surface;
    gpu::GpuAllocation allocation;
    gpu::CommandAllocator allocator;
    gpu::CommandList commands;
    gpu::ExecutableCommandList executable;
    gpu::destroy_runtime(&runtime)!!;
    gpu::destroy_surface(&surface)!!;
    gpu::free_allocation(&device, &allocation)!!;
    gpu::destroy_texture(&device, (gpu::TextureHandle){})!!;
    gpu::destroy_texture_view(&device, (gpu::TextureView){});
    gpu::release_texture_indices(&device, (gpu::TextureIndexRange){})!!;
    gpu::destroy_acceleration_structure(&device, gpu::ACCELERATION_STRUCTURE_HANDLE_INVALID)!!;
    gpu::destroy_acceleration_structure_view(&device, gpu::ACCELERATION_STRUCTURE_VIEW_INVALID);
    gpu::destroy_pipeline(&device, (gpu::PipelineHandle){})!!;
    gpu::destroy_command_allocator(&allocator)!!;
    gpu::discard_commands(&commands)!!;
    gpu::discard_executable_commands(&executable)!!;
    gpu::destroy_attachment_view(&device, (gpu::AttachmentViewHandle){})!!;
    gpu::destroy_timestamp_pool(&device, (gpu::TimestampPoolHandle){});
    gpu::destroy_swapchain(&device, (gpu::SwapchainHandle){})!!;

    assert(slot.control.load(AtomicOrdering.ACQUIRE) == control_before,
        "zero destroys take and leave no device pin");
    assert(slot.live_children.load(AtomicOrdering.ACQUIRE) == children_before,
        "zero destroys release no child");
}

fn void zero_handles_with_destroyed_device() @test {
    DeviceFixture fixture;
    gpu::Device device = fixture_device(&fixture);
    gpu::Device stale = device;
    gpu::destroy_device(&device)!!;

    gpu::GpuAllocation allocation;
    gpu::destroy_texture(&stale, (gpu::TextureHandle){})!!;
    gpu::free_allocation(&stale, &allocation)!!;
    gpu::destroy_pipeline(&stale, (gpu::PipelineHandle){})!!;
    gpu::destroy_swapchain(&stale, (gpu::SwapchainHandle){})!!;
    gpu::destroy_texture(&device, (gpu::TextureHandle){})!!;
}

fn void zero_device_destroy() @test {
    gpu::Device zero;
    gpu::destroy_device(&zero)!!;
    assert(!zero.is_valid(), "the zero device stays zero");
}

fn void free_allocation_twice_through_variable() @test {
    gpu::internal::vk::VkDeviceState state;
    Device device = create_allocation_stub_device(&state);
    defer gpu::destroy_device(&device)!!;
    AllocationDesc desc = allocation_stub_desc();
    GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!!;

    gpu::free_allocation(&device, &allocation)!!;
    gpu::free_allocation(&device, &allocation)!!;
    assert(state.allocation_free_calls == 1,
        "the second free through the zeroed variable is a no-op");
}

fn void null_destroy_pointers_terminate() @test {
    DeviceFixture fixture;
    gpu::Device device = fixture_device(&fixture);
    defer gpu::destroy_device(&device)!!;

    @assert_panics("gpu::free_allocation: null GpuAllocation") {
        (void)gpu::free_allocation(&device, null);
    };
    @assert_panics("gpu::destroy_device: null Device") {
        (void)gpu::destroy_device(null);
    };
    @assert_panics("gpu::discard_commands: null CommandList") {
        (void)gpu::discard_commands(null);
    };
}
