module gpu_test;

import gpu @public;
import gpu::vk @public;

uint resource_access_buffer_creates;
uint resource_access_texture_queries;
uint resource_access_texture_creates;
uint resource_access_persistent_allocations;

fn void destroy_resource_access_device(void* backend_state) {
    (void)backend_state;
}

fn gpu::BufferHandle? create_resource_access_buffer(
    gpu::Device* device,
    gpu::BufferDesc* desc,
) {
    (void)device;
    (void)desc;
    resource_access_buffer_creates++;
    return gpu::OUT_OF_DEVICE_MEMORY~;
}

fn bool? supports_resource_access_texture(
    gpu::Device* device,
    gpu::TextureDesc* desc,
) {
    (void)device;
    (void)desc;
    resource_access_texture_queries++;
    return true;
}

fn gpu::TextureHandle? create_resource_access_texture(
    gpu::Device* device,
    gpu::TextureDesc* desc,
) {
    (void)device;
    (void)desc;
    resource_access_texture_creates++;
    return gpu::OUT_OF_DEVICE_MEMORY~;
}

fn gpu::GpuSpan? alloc_resource_access_persistent(
    gpu::Device* device,
    gpu::PersistentAllocDesc* desc,
) {
    (void)device;
    (void)desc;
    resource_access_persistent_allocations++;
    return gpu::ARENA_FULL~;
}

gpu::BackendVTable resource_access_vtable = {
    .destroy_device        = &destroy_resource_access_device,
    .create_buffer         = &create_resource_access_buffer,
    .supports_texture_desc = &supports_resource_access_texture,
    .create_texture        = &create_resource_access_texture,
    .alloc_persistent_span = &alloc_resource_access_persistent,
};

fn gpu::Device create_compute_only_resource_access_device(int* state) {
    gpu::QueueSelection queues = gpu::queue_selection_for_requirements(
        requirements: {
            .counts = { .compute = 1 },
        },
        compute_aliases_graphics:  false,
        transfer_aliases_graphics: false,
        transfer_aliases_compute:  false,
    )!!;
    return gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          {},
        vtable:        &resource_access_vtable,
        backend_state: state,
        queues:        queues,
    )!!;
}

fn gpu::BufferDesc resource_access_buffer_desc(gpu::QueueRoles access) {
    return {
        .size        = 64,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .access      = access,
    };
}

fn gpu::TextureDesc resource_access_texture_desc(gpu::QueueRoles access) {
    return {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.R8_UNORM,
        .usage     = { .storage },
        .access    = access,
    };
}

fn gpu::PersistentAllocDesc resource_access_persistent_desc(gpu::QueueRoles access) {
    return {
        .size        = 64,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .access      = access,
    };
}

fn void resource_access_rejects_invalid_buffer_domains_before_backend() @test {
    int state;
    gpu::Device device = create_compute_only_resource_access_device(&state);
    resource_access_buffer_creates = 0;

    gpu::BufferDesc empty = resource_access_buffer_desc({});
    assert(@catch(gpu::create_buffer(&device, &empty)) == gpu::INVALID_ARGUMENT);

    gpu::BufferDesc unavailable = resource_access_buffer_desc({ .graphics });
    assert(@catch(gpu::create_buffer(&device, &unavailable)) == gpu::INVALID_ARGUMENT);

    gpu::BufferDesc unknown = resource_access_buffer_desc((gpu::QueueRoles)0x8);
    assert(@catch(gpu::create_buffer(&device, &unknown)) == gpu::INVALID_ARGUMENT);
    assert(resource_access_buffer_creates == 0, "invalid domains do not reach buffer creation");

    gpu::BufferDesc accepted = resource_access_buffer_desc({ .compute });
    assert(@catch(gpu::create_buffer(&device, &accepted)) == gpu::OUT_OF_DEVICE_MEMORY);
    assert(resource_access_buffer_creates == 1, "selected compute access reaches the backend");

    gpu::destroy_device(&device)!!;
}

fn void resource_access_rejects_invalid_texture_domains_before_backend() @test {
    int state;
    gpu::Device device = create_compute_only_resource_access_device(&state);
    resource_access_texture_queries = 0;
    resource_access_texture_creates = 0;

    gpu::TextureDesc empty = resource_access_texture_desc({});
    assert(!gpu::supports_texture_desc(&device, &empty)!!);
    assert(@catch(gpu::create_texture(&device, &empty)) == gpu::INVALID_ARGUMENT);

    gpu::TextureDesc unavailable = resource_access_texture_desc({ .transfer });
    assert(!gpu::supports_texture_desc(&device, &unavailable)!!);
    assert(@catch(gpu::create_texture(&device, &unavailable)) == gpu::INVALID_ARGUMENT);

    gpu::TextureDesc unknown = resource_access_texture_desc((gpu::QueueRoles)0x80);
    assert(!gpu::supports_texture_desc(&device, &unknown)!!);
    assert(@catch(gpu::create_texture(&device, &unknown)) == gpu::INVALID_ARGUMENT);
    assert(resource_access_texture_queries == 0, "invalid domains do not reach texture support");
    assert(resource_access_texture_creates == 0, "invalid domains do not reach texture creation");

    gpu::TextureDesc accepted = resource_access_texture_desc({ .compute });
    assert(gpu::supports_texture_desc(&device, &accepted)!!);
    assert(@catch(gpu::create_texture(&device, &accepted)) == gpu::OUT_OF_DEVICE_MEMORY);
    assert(resource_access_texture_queries == 1, "selected compute access reaches support query");
    assert(resource_access_texture_creates == 1, "selected compute access reaches texture creation");

    gpu::destroy_device(&device)!!;
}

fn void resource_access_rejects_invalid_persistent_domains_before_backend() @test {
    int state;
    gpu::Device device = create_compute_only_resource_access_device(&state);
    resource_access_persistent_allocations = 0;

    gpu::PersistentAllocDesc empty = resource_access_persistent_desc({});
    assert(@catch(gpu::alloc_persistent_span(&device, &empty)) == gpu::INVALID_ARGUMENT);

    gpu::PersistentAllocDesc unavailable =
        resource_access_persistent_desc({ .graphics });
    assert(@catch(gpu::alloc_persistent_span(&device, &unavailable))
        == gpu::INVALID_ARGUMENT);

    gpu::PersistentAllocDesc unknown =
        resource_access_persistent_desc((gpu::QueueRoles)0x40);
    assert(@catch(gpu::alloc_persistent_span(&device, &unknown))
        == gpu::INVALID_ARGUMENT);
    assert(resource_access_persistent_allocations == 0,
        "invalid domains do not reach persistent allocation");

    gpu::PersistentAllocDesc accepted =
        resource_access_persistent_desc({ .compute });
    assert(@catch(gpu::alloc_persistent_span(&device, &accepted))
        == gpu::ARENA_FULL);
    assert(resource_access_persistent_allocations == 1,
        "selected compute access reaches persistent allocation");

    gpu::destroy_device(&device)!!;
}
