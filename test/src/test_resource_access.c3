module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;


fn gpu::Device create_compute_only_resource_access_device(
    gpu::internal::vk::VkDeviceState* state,
) {
    state.allocation_out_of_memory = true;
    state.texture_out_of_memory = true;
    gpu::internal::QueueSelection queues = gpu::internal::queue_selection_for_requirements(
        requirements: {
            .counts = { .compute = 1 },
        },
        compute_aliases_graphics:  false,
        transfer_aliases_graphics: false,
        transfer_aliases_compute:  false,
    )!!;
    return gpu::internal::register_device(
        state:  state,
        caps:   {},
        queues: queues,
    )!!;
}

fn gpu::AllocationDesc resource_access_allocation_desc(gpu::QueueRoles access) {
    return {
        .size         = 64,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = access,
    };
}

fn gpu::TextureDesc resource_access_texture_desc(gpu::QueueRoles access) {
    return {
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.R8_UNORM,
        .usage     = { .storage },
        .access    = access,
    };
}

fn void resource_access_rejects_invalid_allocation_domains_before_backend() @test {
    gpu::internal::vk::VkDeviceState state;
    gpu::Device device = create_compute_only_resource_access_device(&state);

    gpu::AllocationDesc empty = resource_access_allocation_desc({});
    assert(@catch(gpu::allocate_memory(&device, &empty)) == gpu::INVALID_ARGUMENT);

    gpu::AllocationDesc unavailable = resource_access_allocation_desc({ .graphics });
    assert(@catch(gpu::allocate_memory(&device, &unavailable)) == gpu::INVALID_ARGUMENT);

    gpu::AllocationDesc unknown = resource_access_allocation_desc((gpu::QueueRoles)0x8);
    assert(@catch(gpu::allocate_memory(&device, &unknown)) == gpu::INVALID_ARGUMENT);
    assert(state.allocation_attempts == 0,
        "invalid domains do not reach allocation");

    gpu::AllocationDesc accepted = resource_access_allocation_desc({ .compute });
    assert(@catch(gpu::allocate_memory(&device, &accepted)) == gpu::OUT_OF_DEVICE_MEMORY);
    assert(state.allocation_attempts == 1,
        "selected compute access reaches allocation");

    gpu::destroy_device(&device)!!;
}

fn void resource_access_rejects_invalid_texture_domains_before_backend() @test {
    gpu::internal::vk::VkDeviceState state;
    gpu::Device device = create_compute_only_resource_access_device(&state);

    gpu::TextureDesc empty = resource_access_texture_desc({});
    assert(!gpu::supports_texture_desc(&device, &empty)!!);
    assert(@catch(gpu::create_texture(&device, &empty)) == gpu::INVALID_ARGUMENT);

    gpu::TextureDesc unavailable = resource_access_texture_desc({ .transfer });
    assert(!gpu::supports_texture_desc(&device, &unavailable)!!);
    assert(@catch(gpu::create_texture(&device, &unavailable)) == gpu::INVALID_ARGUMENT);

    gpu::TextureDesc unknown = resource_access_texture_desc((gpu::QueueRoles)0x80);
    assert(!gpu::supports_texture_desc(&device, &unknown)!!);
    assert(@catch(gpu::create_texture(&device, &unknown)) == gpu::INVALID_ARGUMENT);
    assert(state.texture_support_queries == 0,
        "invalid domains do not reach texture support");
    assert(state.texture_create_attempts == 0,
        "invalid domains do not reach texture creation");

    gpu::TextureDesc accepted = resource_access_texture_desc({ .compute });
    assert(gpu::supports_texture_desc(&device, &accepted)!!);
    assert(@catch(gpu::create_texture(&device, &accepted)) == gpu::OUT_OF_DEVICE_MEMORY);
    assert(state.texture_support_queries == 1,
        "selected compute access reaches support query");
    assert(state.texture_create_attempts == 1,
        "selected compute access reaches texture creation");

    gpu::destroy_device(&device)!!;
}
