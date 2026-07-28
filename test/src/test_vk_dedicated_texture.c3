module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import vma;

struct DedicatedTextureProbe {
    vk::Result create_result;
    vk::Result allocate_result;
    vk::Result bind_result;
    vk::Result view_result;
    uint       create_calls;
    uint       allocate_calls;
    uint       bind_calls;
    uint       view_calls;
    uint       destroy_image_calls;
    uint       destroy_view_calls;
    uint       free_calls;
}

struct DedicatedTextureFixture {
    vk::VkDeviceState     state;
    vk::AllocationCell[1] allocations;
    vk::TextureCell[1]    textures;
    DedicatedTextureProbe probe;
}

fn vk::Result dedicated_probe_create_image(
    void* user_data,
    vk::Device device,
    vk::ImageCreateInfo* image_info,
    vk::Image* out_image,
) {
    (void)device;
    (void)image_info;
    DedicatedTextureProbe* probe = (DedicatedTextureProbe*)user_data;
    probe.create_calls++;
    if (probe.create_result != vk::Result.SUCCESS) {
        return probe.create_result;
    }
    *out_image = (vk::Image)(uptr)0x410;
    return vk::Result.SUCCESS;
}

fn vk::Result dedicated_probe_allocate_memory(
    void* user_data,
    vma::Allocator allocator,
    vk::Image image,
    vma::AllocationCreateInfo* allocation_info,
    vma::Allocation* out_allocation,
    vma::AllocationInfo* out_info,
) {
    (void)allocator;
    (void)image;
    (void)allocation_info;
    DedicatedTextureProbe* probe = (DedicatedTextureProbe*)user_data;
    probe.allocate_calls++;
    if (probe.allocate_result != vk::Result.SUCCESS) {
        return probe.allocate_result;
    }
    *out_allocation = (vma::Allocation)(void*)0x420;
    *out_info = {
        .memory_type = 1,
        .size        = 64,
    };
    return vk::Result.SUCCESS;
}

fn vk::Result dedicated_probe_bind_memory(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
    vk::Image image,
) {
    (void)allocator;
    (void)allocation;
    (void)image;
    DedicatedTextureProbe* probe = (DedicatedTextureProbe*)user_data;
    probe.bind_calls++;
    return probe.bind_result;
}

fn vk::Result dedicated_probe_create_view(
    void* user_data,
    vk::Device device,
    vk::ImageViewCreateInfo* view_info,
    vk::ImageView* out_view,
) {
    (void)device;
    (void)view_info;
    DedicatedTextureProbe* probe = (DedicatedTextureProbe*)user_data;
    probe.view_calls++;
    if (probe.view_result != vk::Result.SUCCESS) return probe.view_result;
    *out_view = (vk::ImageView)(uptr)0x430;
    return vk::Result.SUCCESS;
}

fn void dedicated_probe_destroy_image(
    void* user_data,
    vk::Device device,
    vk::Image image,
) {
    (void)device;
    (void)image;
    DedicatedTextureProbe* probe = (DedicatedTextureProbe*)user_data;
    probe.destroy_image_calls++;
}

fn void dedicated_probe_destroy_view(
    void* user_data,
    vk::Device device,
    vk::ImageView view,
) {
    (void)device;
    (void)view;
    DedicatedTextureProbe* probe = (DedicatedTextureProbe*)user_data;
    probe.destroy_view_calls++;
}

fn void dedicated_probe_free_memory(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
) {
    (void)allocator;
    (void)allocation;
    DedicatedTextureProbe* probe = (DedicatedTextureProbe*)user_data;
    probe.free_calls++;
}

fn vk::DedicatedTextureNativeOps dedicated_probe_ops(
    DedicatedTextureProbe* probe,
) {
    return {
        .user_data       = probe,
        .create_image    = &dedicated_probe_create_image,
        .allocate_memory = &dedicated_probe_allocate_memory,
        .bind_memory     = &dedicated_probe_bind_memory,
        .create_view     = &dedicated_probe_create_view,
        .destroy_image   = &dedicated_probe_destroy_image,
        .destroy_view    = &dedicated_probe_destroy_view,
        .free_memory     = &dedicated_probe_free_memory,
    };
}

fn void init_dedicated_fixture(DedicatedTextureFixture* fixture) {
    const ulong OWNER = 131;
    fixture.state.resource_owner = OWNER;
    fixture.state.compute_family = 0;
    fixture.state.allocations.owner = gpu::internal::resource_owner_tag(
        OWNER,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    );
    fixture.state.allocations.slots = fixture.allocations[..];
    fixture.state.textures.owner = gpu::internal::resource_owner_tag(
        OWNER,
        gpu::internal::ResourceOwnerKind.TEXTURE,
    );
    fixture.state.textures.slots = fixture.textures[..];
    fixture.state.resource_mutex.init()!!;
}

fn gpu::TextureDesc dedicated_test_texture_desc() {
    return {
        .access    = { .compute },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled },
    };
}

fn gpu::TextureRequirements dedicated_test_requirements(
    DedicatedTextureFixture* fixture,
) {
    return {
        .size      = 64,
        .alignment = 64,
        .compatibility = vk::texture_compatibility(
            fixture.state.allocations.owner,
            0b0010,
        ),
    };
}

fn void run_dedicated_failure_case(uint stage) {
    DedicatedTextureFixture fixture;
    init_dedicated_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::TextureDesc texture_desc = dedicated_test_texture_desc();
    gpu::TextureRequirements requirements =
        dedicated_test_requirements(&fixture);
    gpu::TextureRequirements[1] queried = { requirements };
    gpu::AllocationDesc allocation_desc = {
        .size                 = requirements.size,
        .alignment            = requirements.alignment,
        .memory_class         = gpu::MemoryClass.TEXTURE,
        .access               = texture_desc.access,
        .texture_requirements = queried[..],
    };
    switch (stage) {
        case 0:
            fixture.probe.create_result =
                vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
        case 1:
            fixture.probe.allocate_result =
                vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
        case 2:
            fixture.probe.bind_result =
                vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
        case 3:
            fixture.probe.view_result =
                vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
        case 4:
            fixture.state.allocations.next_new = 1;
        case 5:
            fixture.state.textures.next_new = 1;
        default:
            unreachable();
    }
    vk::DedicatedTextureNativeOps ops =
        dedicated_probe_ops(&fixture.probe);
    fault actual = @catch(vk::create_dedicated_texture_for_state_with_ops(
        state:           &fixture.state,
        desc:            &texture_desc,
        allocation_desc: &allocation_desc,
        requirements:    requirements,
        ops:             &ops,
    ));
    fault expected = stage < 4
        ? gpu::OUT_OF_DEVICE_MEMORY
        : gpu::SLOT_TABLE_FULL;
    assert(actual == expected, "dedicated failure preserves its public fault");
    assert(!fixture.allocations[0].used && !fixture.textures[0].used,
        "dedicated failure publishes neither token");
    assert(fixture.probe.create_calls == 1);
    assert(fixture.probe.allocate_calls == (stage == 0 ? 0 : 1));
    assert(fixture.probe.bind_calls == (stage <= 1 ? 0 : 1));
    assert(fixture.probe.view_calls == (stage <= 2 ? 0 : 1));
    assert(fixture.probe.destroy_image_calls == (stage == 0 ? 0 : 1));
    assert(fixture.probe.destroy_view_calls == (stage <= 3 ? 0 : 1));
    assert(fixture.probe.free_calls == (stage <= 1 ? 0 : 1));
}

fn void dedicated_texture_validation_precedes_native_mutation() @test {
    DedicatedTextureFixture fixture;
    init_dedicated_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::TextureDesc texture_desc = dedicated_test_texture_desc();
    gpu::TextureRequirements requirements =
        dedicated_test_requirements(&fixture);
    gpu::TextureRequirements[1] queried = { requirements };
    gpu::AllocationDesc allocation_desc = {
        .size                 = requirements.size + 64,
        .alignment            = requirements.alignment,
        .memory_class         = gpu::MemoryClass.TEXTURE,
        .access               = texture_desc.access,
        .texture_requirements = queried[..],
    };
    vk::DedicatedTextureNativeOps ops =
        dedicated_probe_ops(&fixture.probe);

    assert(@catch(vk::create_dedicated_texture_for_state_with_ops(
        state:           &fixture.state,
        desc:            &texture_desc,
        allocation_desc: &allocation_desc,
        requirements:    requirements,
        ops:             &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(fixture.probe.create_calls == 0,
        "invalid dedicated requests stop before native mutation");

    allocation_desc.size = requirements.size;
    gpu::TextureRequirements foreign_requirements = requirements;
    foreign_requirements.compatibility = vk::texture_compatibility(
        gpu::internal::resource_owner_tag(
            132,
            gpu::internal::ResourceOwnerKind.ALLOCATION,
        ),
        0b0010,
    );
    assert(@catch(vk::create_dedicated_texture_for_state_with_ops(
        state:           &fixture.state,
        desc:            &texture_desc,
        allocation_desc: &allocation_desc,
        requirements:    foreign_requirements,
        ops:             &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(fixture.probe.create_calls == 0,
        "foreign direct requirements stop before native mutation");

    queried[0] = foreign_requirements;
    assert(@catch(vk::create_dedicated_texture_for_state_with_ops(
        state:           &fixture.state,
        desc:            &texture_desc,
        allocation_desc: &allocation_desc,
        requirements:    requirements,
        ops:             &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(fixture.probe.create_calls == 0,
        "foreign candidate requirements stop before native mutation");
}

fn void dedicated_texture_transaction_rolls_back_every_step() @test {
    for (uint stage = 0; stage < 6; stage++) {
        run_dedicated_failure_case(stage);
    }
}

fn void dedicated_texture_transaction_publishes_separate_tokens() @test {
    DedicatedTextureFixture fixture;
    init_dedicated_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::TextureDesc texture_desc = dedicated_test_texture_desc();
    gpu::TextureRequirements requirements =
        dedicated_test_requirements(&fixture);
    requirements.dedicated_only = true;
    gpu::TextureRequirements[1] queried = { requirements };
    gpu::AllocationDesc allocation_desc = {
        .size                 = requirements.size,
        .alignment            = requirements.alignment,
        .memory_class         = gpu::MemoryClass.TEXTURE,
        .access               = texture_desc.access,
        .texture_requirements = queried[..],
    };
    vk::DedicatedTextureNativeOps ops =
        dedicated_probe_ops(&fixture.probe);

    gpu::DedicatedTexture created =
        vk::create_dedicated_texture_for_state_with_ops(
            state:           &fixture.state,
            desc:            &texture_desc,
            allocation_desc: &allocation_desc,
            requirements:    requirements,
            ops:             &ops,
        )!!;
    assert(created.texture.is_valid() && created.allocation.is_valid());
    assert(fixture.allocations[0].used && fixture.textures[0].used);
    assert(fixture.allocations[0].value.dedicated_texture
        && fixture.allocations[0].value.live_placements == 1);
    assert(fixture.textures[0].value.placement == created.allocation);
    assert(@catch(fixture.state.allocations.free(created.allocation))
        == gpu::RESOURCE_IN_USE,
        "dedicated storage cannot be released while its texture is live");

    fixture.state.textures.free(created.texture)!!;
    fixture.allocations[0].value.live_placements = 0;
    fixture.state.allocations.free(created.allocation)!!;
    ops.destroy_view(
        ops.user_data,
        fixture.state.device,
        (vk::ImageView)(uptr)0x430,
    );
    ops.destroy_image(
        ops.user_data,
        fixture.state.device,
        (vk::Image)(uptr)0x410,
    );
    ops.free_memory(
        ops.user_data,
        fixture.state.allocator,
        (vma::Allocation)(void*)0x420,
    );
}

fn void dedicated_transfer_only_texture_omits_default_view() @test {
    DedicatedTextureFixture fixture;
    init_dedicated_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::TextureDesc texture_desc = dedicated_test_texture_desc();
    texture_desc.usage = { .transfer_src, .transfer_dst };
    gpu::TextureRequirements requirements =
        dedicated_test_requirements(&fixture);
    gpu::TextureRequirements[1] queried = { requirements };
    gpu::AllocationDesc allocation_desc = {
        .size                 = requirements.size,
        .alignment            = requirements.alignment,
        .memory_class         = gpu::MemoryClass.TEXTURE,
        .access               = texture_desc.access,
        .texture_requirements = queried[..],
    };
    vk::DedicatedTextureNativeOps ops =
        dedicated_probe_ops(&fixture.probe);

    gpu::DedicatedTexture created =
        vk::create_dedicated_texture_for_state_with_ops(
            state:           &fixture.state,
            desc:            &texture_desc,
            allocation_desc: &allocation_desc,
            requirements:    requirements,
            ops:             &ops,
        )!!;
    vk::TextureSlot* slot =
        fixture.state.textures.get(created.texture)!!;
    assert(fixture.probe.create_calls == 1
        && fixture.probe.allocate_calls == 1
        && fixture.probe.bind_calls == 1
        && fixture.probe.view_calls == 0
        && slot.default_view == null,
        "dedicated transfer-only publication owns image memory without a view");

    fixture.state.textures.free(created.texture)!!;
    fixture.allocations[0].value.live_placements = 0;
    fixture.state.allocations.free(created.allocation)!!;
    ops.destroy_image(
        ops.user_data,
        fixture.state.device,
        (vk::Image)(uptr)0x410,
    );
    ops.free_memory(
        ops.user_data,
        fixture.state.allocator,
        (vma::Allocation)(void*)0x420,
    );
}

fn void dedicated_transfer_only_publication_rolls_back_without_view() @test {
    DedicatedTextureFixture fixture;
    init_dedicated_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::TextureDesc texture_desc = dedicated_test_texture_desc();
    texture_desc.usage = { .transfer_dst };
    gpu::TextureRequirements requirements =
        dedicated_test_requirements(&fixture);
    gpu::TextureRequirements[1] queried = { requirements };
    gpu::AllocationDesc allocation_desc = {
        .size                 = requirements.size,
        .alignment            = requirements.alignment,
        .memory_class         = gpu::MemoryClass.TEXTURE,
        .access               = texture_desc.access,
        .texture_requirements = queried[..],
    };
    vk::DedicatedTextureNativeOps ops =
        dedicated_probe_ops(&fixture.probe);
    fixture.state.textures.next_new = 1;

    assert(@catch(vk::create_dedicated_texture_for_state_with_ops(
        state:           &fixture.state,
        desc:            &texture_desc,
        allocation_desc: &allocation_desc,
        requirements:    requirements,
        ops:             &ops,
    )) == gpu::SLOT_TABLE_FULL);
    assert(fixture.probe.create_calls == 1
        && fixture.probe.allocate_calls == 1
        && fixture.probe.bind_calls == 1
        && fixture.probe.view_calls == 0
        && fixture.probe.destroy_view_calls == 0
        && fixture.probe.destroy_image_calls == 1
        && fixture.probe.free_calls == 1,
        "dedicated transfer-only rollback releases image memory without a view");
}

fn void dedicated_texture_native_lifetime_is_independently_owned() @test {
    gpu::Device device = make_texture_test_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc texture_desc = {
        .access    = { .graphics, .compute },
        .width     = 16,
        .height    = 16,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureRequirements requirements =
        gpu::get_texture_requirements(&device, &texture_desc)!!;
    gpu::TextureRequirements[1] queried = { requirements };
    gpu::AllocationDesc allocation_desc = {
        .size                 = requirements.size,
        .alignment            = requirements.alignment,
        .memory_class         = gpu::MemoryClass.TEXTURE,
        .access               = texture_desc.access,
        .texture_requirements = queried[..],
    };

    gpu::DedicatedTexture created = gpu::create_dedicated_texture(
        &device,
        &texture_desc,
        &allocation_desc,
    )!!;
    assert(@catch(gpu::free_allocation(&device, &created.allocation))
        == gpu::RESOURCE_IN_USE,
        "failed early release preserves dedicated allocation ownership");
    gpu::destroy_texture(&device, created.texture)!!;
    gpu::free_allocation(&device, &created.allocation)!!;
}
