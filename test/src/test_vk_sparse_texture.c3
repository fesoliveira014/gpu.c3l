module gpu_test;

import gpu @public;
import gpu::internal::vk @public;
import vk;
import vma;

struct SparseTextureProbe {
    bool                              support_result;
    uint                              sparse_format_count;
    vk::Result                        create_result;
    vk::Result                        view_result;
    vk::ImageCreateFlags              support_flags;
    vk::ImageCreateInfo               image_info;
    vk::MemoryRequirements            memory;
    vk::SparseImageMemoryRequirements[2] native;
    uint                              query_count;
    uint                              fill_count;
    uint                              support_calls;
    uint                              sparse_format_calls;
    uint                              create_calls;
    uint                              memory_calls;
    uint                              sparse_query_calls;
    uint                              sparse_fill_calls;
    uint                              view_calls;
    uint                              destroy_image_calls;
    uint                              destroy_allocator_image_calls;
    uint                              destroy_view_calls;
}

struct SparseTextureFixture {
    vk::VkDeviceState  state;
    vk::TextureCell[1] textures;
    SparseTextureProbe probe;
}

fn bool? sparse_probe_supports_image_desc(
    void* user_data,
    vk::VkDeviceState* state,
    gpu::TextureDesc* desc,
    vk::ImageCreateFlags flags,
) {
    (void)state;
    (void)desc;
    SparseTextureProbe* probe = (SparseTextureProbe*)user_data;
    probe.support_calls++;
    probe.support_flags = flags;
    return probe.support_result;
}

fn uint? sparse_probe_format_property_count(
    void* user_data,
    vk::VkDeviceState* state,
    gpu::TextureDesc* desc,
) {
    (void)state;
    (void)desc;
    SparseTextureProbe* probe = (SparseTextureProbe*)user_data;
    probe.sparse_format_calls++;
    return probe.sparse_format_count;
}

fn vk::Result sparse_probe_create_image(
    void* user_data,
    vk::Device device,
    vk::ImageCreateInfo* image_info,
    vk::Image* out_image,
) {
    (void)device;
    SparseTextureProbe* probe = (SparseTextureProbe*)user_data;
    probe.create_calls++;
    probe.image_info = *image_info;
    if (probe.create_result != vk::Result.SUCCESS) {
        return probe.create_result;
    }
    *out_image = (vk::Image)(uptr)0x510;
    return vk::Result.SUCCESS;
}

fn void sparse_probe_get_memory_requirements(
    void* user_data,
    vk::Device device,
    vk::Image image,
    vk::MemoryRequirements* out_memory,
) {
    (void)device;
    (void)image;
    SparseTextureProbe* probe = (SparseTextureProbe*)user_data;
    probe.memory_calls++;
    *out_memory = probe.memory;
}

fn void sparse_probe_get_sparse_requirements(
    void* user_data,
    vk::Device device,
    vk::Image image,
    uint* count,
    vk::SparseImageMemoryRequirements* out_requirements,
) {
    (void)device;
    (void)image;
    SparseTextureProbe* probe = (SparseTextureProbe*)user_data;
    if (out_requirements == null) {
        probe.sparse_query_calls++;
        *count = probe.query_count;
        return;
    }
    probe.sparse_fill_calls++;
    *count = probe.fill_count;
    for (uint index = 0; index < probe.fill_count; index++) {
        out_requirements[index] = probe.native[index];
    }
}

fn vk::Result sparse_probe_create_view(
    void* user_data,
    vk::Device device,
    vk::ImageViewCreateInfo* view_info,
    vk::ImageView* out_view,
) {
    (void)device;
    (void)view_info;
    SparseTextureProbe* probe = (SparseTextureProbe*)user_data;
    probe.view_calls++;
    if (probe.view_result != vk::Result.SUCCESS) return probe.view_result;
    *out_view = (vk::ImageView)(uptr)0x520;
    return vk::Result.SUCCESS;
}

fn void sparse_probe_destroy_image(
    void* user_data,
    vk::Device device,
    vk::Image image,
) {
    (void)device;
    (void)image;
    SparseTextureProbe* probe = (SparseTextureProbe*)user_data;
    probe.destroy_image_calls++;
}

fn void sparse_probe_destroy_view(
    void* user_data,
    vk::Device device,
    vk::ImageView view,
) {
    (void)device;
    (void)view;
    SparseTextureProbe* probe = (SparseTextureProbe*)user_data;
    probe.destroy_view_calls++;
}

fn void sparse_probe_destroy_allocator_image(
    void* user_data,
    vma::Allocator allocator,
    vk::Image image,
    vma::Allocation allocation,
) {
    (void)allocator;
    (void)image;
    (void)allocation;
    SparseTextureProbe* probe = (SparseTextureProbe*)user_data;
    probe.destroy_allocator_image_calls++;
}

fn vk::SparseTextureNativeOps sparse_probe_ops(SparseTextureProbe* probe) {
    return {
        .user_data                    = probe,
        .supports_image_desc          = &sparse_probe_supports_image_desc,
        .sparse_format_property_count =
            &sparse_probe_format_property_count,
        .create_image                 = &sparse_probe_create_image,
        .get_memory_requirements      =
            &sparse_probe_get_memory_requirements,
        .get_sparse_requirements      =
            &sparse_probe_get_sparse_requirements,
        .create_view                  = &sparse_probe_create_view,
        .destroy_image                = &sparse_probe_destroy_image,
        .destroy_view                 = &sparse_probe_destroy_view,
    };
}

fn void init_sparse_texture_fixture(SparseTextureFixture* fixture) {
    const ulong OWNER = 509;
    fixture.state.resource_owner = OWNER;
    fixture.state.selected_queue_roles = { .compute };
    fixture.state.compute_family = 3;
    fixture.state.caps.sparse_textures = {
        .image_2d       = true,
        .image_3d       = true,
        .binding_queues = { .compute },
    };
    fixture.state.allocations.owner = gpu::internal::resource_owner_tag(
        OWNER,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    );
    fixture.state.textures.owner = gpu::internal::resource_owner_tag(
        OWNER,
        gpu::internal::ResourceOwnerKind.TEXTURE,
    );
    fixture.state.textures.slots = fixture.textures[..];
    fixture.state.resource_mutex.init()!!;
    fixture.probe.support_result = true;
    fixture.probe.sparse_format_count = 1;
    fixture.probe.memory = {
        .size             = 4096,
        .alignment        = 256,
        .memory_type_bits = 0b1010,
    };
    fixture.probe.native[0] = valid_sparse_color_requirement();
    fixture.probe.query_count = 1;
    fixture.probe.fill_count = 1;
}

fn gpu::TextureDesc sparse_test_texture_desc() {
    return {
        .access     = { .compute },
        .width      = 64,
        .height     = 32,
        .mip_levels = 7,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled },
    };
}

fn void sparse_native_ops_exposes_creation_transaction_seam() @test {
    vk::SparseTextureNativeOps ops;

    assert(ops.supports_image_desc == null);
    assert(ops.sparse_format_property_count == null);
    assert(ops.create_image == null);
    assert(ops.get_memory_requirements == null);
    assert(ops.get_sparse_requirements == null);
    assert(ops.create_view == null);
    assert(ops.destroy_image == null);
    assert(ops.destroy_view == null);
}

fn void sparse_texture_slot_stores_cached_requirements() @test {
    vk::TextureSlot slot = {
        .sparse = true,
        .sparse_requirements = {
            .virtual_size = 4096,
            .aspect_count = 1,
        },
    };

    assert(slot.sparse);
    assert(slot.sparse_requirements.virtual_size == 4096);
    assert(slot.sparse_requirements.aspect_count == 1);
    assert(slot.allocation == null);
    assert(!slot.placed);
    assert(slot.allocation_size == 0);
}

fn void texture_image_ownership_distinguishes_sparse_placed_and_vma() @test {
    vk::TextureSlot sparse = { .sparse = true, .placed = true };
    vk::TextureSlot placed = { .placed = true };
    vk::TextureSlot ordinary;
    vk::VkDeviceState state = {
        .allocator = (vma::Allocator)(uptr)0x530,
    };
    SparseTextureProbe probe;
    vk::TextureImageDestroyOps ops = {
        .user_data               = &probe,
        .destroy_image           = &sparse_probe_destroy_image,
        .destroy_allocator_image =
            &sparse_probe_destroy_allocator_image,
    };

    assert(vk::texture_image_ownership(&sparse)
        == vk::TextureImageOwnership.SPARSE);
    assert(vk::texture_image_ownership(&placed)
        == vk::TextureImageOwnership.PLACED);
    assert(vk::texture_image_ownership(&ordinary)
        == vk::TextureImageOwnership.ALLOCATOR);
    vk::destroy_texture_image_with_ops(
        state:      &state,
        ownership:  vk::texture_image_ownership(&sparse),
        image:      sparse.image,
        allocation: sparse.allocation,
        ops:        &ops,
    );
    assert(probe.destroy_image_calls == 1);
    assert(probe.destroy_allocator_image_calls == 0);
    vk::destroy_texture_image_with_ops(
        state:      &state,
        ownership:  vk::texture_image_ownership(&placed),
        image:      placed.image,
        allocation: placed.allocation,
        ops:        &ops,
    );
    assert(probe.destroy_image_calls == 2);
    assert(probe.destroy_allocator_image_calls == 0);
    vk::destroy_texture_image_with_ops(
        state:      &state,
        ownership:  vk::texture_image_ownership(&ordinary),
        image:      ordinary.image,
        allocation: ordinary.allocation,
        ops:        &ops,
    );
    assert(probe.destroy_image_calls == 2);
    assert(probe.destroy_allocator_image_calls == 1);
}

fn void sparse_texture_transaction_publishes_unbound_image() @test {
    SparseTextureFixture fixture;
    init_sparse_texture_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::TextureDesc desc = sparse_test_texture_desc();
    vk::SparseTextureNativeOps ops = sparse_probe_ops(&fixture.probe);

    gpu::TextureHandle handle =
        vk::create_sparse_texture_for_state_with_ops(
            &fixture.state,
            &desc,
            &ops,
        )!!;

    vk::ImageCreateFlags expected_flags =
        (vk::ImageCreateFlags)vk::ImageCreateFlagBits.SPARSE_BINDING_BIT
        | (vk::ImageCreateFlags)vk::ImageCreateFlagBits.SPARSE_RESIDENCY_BIT;
    assert(handle.is_valid());
    assert(fixture.probe.support_calls == 1);
    assert(fixture.probe.sparse_format_calls == 1);
    assert(fixture.probe.support_flags == expected_flags);
    assert(fixture.probe.create_calls == 1);
    assert(fixture.probe.memory_calls == 1);
    assert(fixture.probe.sparse_query_calls == 1);
    assert(fixture.probe.sparse_fill_calls == 1);
    assert(fixture.probe.view_calls == 1);
    assert(fixture.probe.image_info.flags == expected_flags);
    assert(fixture.probe.image_info.image_type == vk::ImageType.TYPE_2D);
    assert(fixture.probe.image_info.extent.width == 64);
    assert(fixture.probe.image_info.extent.height == 32);
    assert(fixture.probe.image_info.extent.depth == 1);
    vk::TextureSlot* slot = fixture.state.textures.get(handle)!!;
    assert(slot.sparse);
    assert(slot.image == (vk::Image)(uptr)0x510);
    assert(slot.default_view == (vk::ImageView)(uptr)0x520);
    assert(slot.allocation == null);
    assert(!slot.placed);
    assert(slot.allocation_size == 0);
    assert(slot.sparse_requirements.virtual_size == 4096);
    assert(slot.sparse_requirements.aspect_count == 1);
}

fn void sparse_three_dimensional_texture_preserves_exact_extent() @test {
    SparseTextureFixture fixture;
    init_sparse_texture_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::TextureDesc desc = sparse_test_texture_desc();
    desc.depth = 8;
    vk::SparseTextureNativeOps ops = sparse_probe_ops(&fixture.probe);

    gpu::TextureHandle handle =
        vk::create_sparse_texture_for_state_with_ops(
            &fixture.state,
            &desc,
            &ops,
        )!!;
    vk::TextureSlot* slot = fixture.state.textures.get(handle)!!;

    assert(fixture.probe.image_info.image_type == vk::ImageType.TYPE_3D);
    assert(fixture.probe.image_info.extent.width == 64);
    assert(fixture.probe.image_info.extent.height == 32);
    assert(fixture.probe.image_info.extent.depth == 8);
    assert(fixture.probe.image_info.array_layers == 1);
    assert(slot.depth == 8);
    assert(slot.array_layers == 1);
}

fn void run_sparse_preflight_case(uint stage) {
    SparseTextureFixture fixture;
    init_sparse_texture_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::TextureDesc desc = sparse_test_texture_desc();
    switch (stage) {
        case 0:
            desc.width = 0;
        case 1:
            desc.array_layers = 2;
        case 2:
            desc.format = gpu::Format.D32_FLOAT;
            desc.usage = { .transfer_src };
        case 3:
            desc.usage = { .color_attach };
        case 4:
            desc.usage = {};
        case 5:
            desc.sample_count = gpu::SampleCount.TWO;
        case 6:
            fixture.state.caps.sparse_textures.image_2d = false;
        case 7:
            desc.depth = 4;
            fixture.state.caps.sparse_textures.image_3d = false;
        case 8:
            fixture.probe.support_result = false;
        case 9:
            fixture.probe.sparse_format_count = 0;
        default:
            unreachable();
    }
    vk::SparseTextureNativeOps ops = sparse_probe_ops(&fixture.probe);

    fault actual = @catch(vk::create_sparse_texture_for_state_with_ops(
        &fixture.state,
        &desc,
        &ops,
    ));
    fault expected = stage <= 5
        ? gpu::INVALID_ARGUMENT
        : gpu::UNSUPPORTED_FEATURE;
    assert(actual == expected);
    assert(fixture.probe.create_calls == 0);
    assert(fixture.probe.support_calls == (stage < 8 ? 0 : 1));
    assert(fixture.probe.sparse_format_calls == (stage == 9 ? 1 : 0));
}

fn void sparse_texture_preflight_precedes_native_mutation() @test {
    for (uint stage = 0; stage < 10; stage++) {
        run_sparse_preflight_case(stage);
    }
}

fn void run_sparse_transaction_failure_case(uint stage) {
    SparseTextureFixture fixture;
    init_sparse_texture_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    switch (stage) {
        case 0:
            fixture.probe.create_result =
                vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
        case 1:
            fixture.probe.query_count = 0;
        case 2:
            fixture.probe.query_count = 3;
        case 3:
            fixture.probe.fill_count = 0;
        case 4:
            fixture.probe.fill_count = 2;
        case 5:
            fixture.probe.memory.alignment = 0;
        case 6:
            fixture.probe.view_result =
                vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
        case 7:
            fixture.state.textures.next_new = 1;
        default:
            unreachable();
    }
    gpu::TextureDesc desc = sparse_test_texture_desc();
    vk::SparseTextureNativeOps ops = sparse_probe_ops(&fixture.probe);

    fault actual = @catch(vk::create_sparse_texture_for_state_with_ops(
        &fixture.state,
        &desc,
        &ops,
    ));
    fault expected = stage == 0 || stage == 6
        ? gpu::OUT_OF_DEVICE_MEMORY
        : stage == 2
            ? gpu::UNSUPPORTED_FEATURE
            : stage == 7
            ? gpu::SLOT_TABLE_FULL
            : gpu::BACKEND_ERROR;
    assert(actual == expected);
    assert(!fixture.textures[0].used);
    assert(fixture.probe.create_calls == 1);
    assert(fixture.probe.memory_calls == (stage == 0 ? 0 : 1));
    assert(fixture.probe.sparse_query_calls == (stage == 0 ? 0 : 1));
    assert(fixture.probe.sparse_fill_calls == (stage < 3 ? 0 : 1));
    assert(fixture.probe.view_calls == (stage < 6 ? 0 : 1));
    assert(fixture.probe.destroy_image_calls == (stage == 0 ? 0 : 1));
    assert(fixture.probe.destroy_view_calls == (stage == 7 ? 1 : 0));
}

fn void sparse_texture_transaction_rolls_back_every_stage() @test {
    for (uint stage = 0; stage < 8; stage++) {
        run_sparse_transaction_failure_case(stage);
    }
}

fn void sparse_storage_texture_creates_default_view() @test {
    SparseTextureFixture fixture;
    init_sparse_texture_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::TextureDesc desc = sparse_test_texture_desc();
    desc.usage = { .storage };
    vk::SparseTextureNativeOps ops = sparse_probe_ops(&fixture.probe);

    gpu::TextureHandle handle =
        vk::create_sparse_texture_for_state_with_ops(
            &fixture.state,
            &desc,
            &ops,
        )!!;
    vk::TextureSlot* slot = fixture.state.textures.get(handle)!!;

    assert(fixture.probe.view_calls == 1);
    assert(slot.default_view == (vk::ImageView)(uptr)0x520);
}

fn void sparse_transfer_only_texture_omits_default_view() @test {
    SparseTextureFixture fixture;
    init_sparse_texture_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::TextureDesc desc = sparse_test_texture_desc();
    desc.usage = { .transfer_src, .transfer_dst };
    vk::SparseTextureNativeOps ops = sparse_probe_ops(&fixture.probe);

    gpu::TextureHandle handle =
        vk::create_sparse_texture_for_state_with_ops(
            &fixture.state,
            &desc,
            &ops,
        )!!;
    vk::TextureSlot* slot = fixture.state.textures.get(handle)!!;

    assert(fixture.probe.view_calls == 0);
    assert(slot.default_view == null);
}

fn void sparse_requirements_query_returns_only_cached_sparse_snapshot() @test {
    SparseTextureFixture fixture;
    init_sparse_texture_fixture(&fixture);
    defer (void)fixture.state.resource_mutex.destroy();
    gpu::TextureDesc desc = sparse_test_texture_desc();
    vk::SparseTextureNativeOps ops = sparse_probe_ops(&fixture.probe);
    gpu::TextureHandle sparse =
        vk::create_sparse_texture_for_state_with_ops(
            &fixture.state,
            &desc,
            &ops,
        )!!;

    gpu::SparseTextureRequirements first =
        vk::vk_get_sparse_texture_requirements(
            &fixture.state,
            sparse,
        )!!;
    fixture.probe.memory.size = 8192;
    fixture.probe.native[0].format_properties.image_granularity.width = 1;
    gpu::SparseTextureRequirements second =
        vk::vk_get_sparse_texture_requirements(
            &fixture.state,
            sparse,
        )!!;

    assert(first.virtual_size == 4096);
    assert(second.virtual_size == first.virtual_size);
    assert(second.aspects[0].tile_extent.x
        == first.aspects[0].tile_extent.x);
    assert(fixture.probe.memory_calls == 1);
    assert(fixture.probe.sparse_query_calls == 1);
    assert(fixture.probe.sparse_fill_calls == 1);

    fixture.state.textures.free(sparse)!!;
    vk::TextureSlot ordinary;
    gpu::TextureHandle ordinary_handle =
        fixture.state.textures.alloc(ordinary)!!;
    assert(@catch(vk::vk_get_sparse_texture_requirements(
        &fixture.state,
        ordinary_handle,
    )) == gpu::INVALID_ARGUMENT);

    fixture.state.textures.free(ordinary_handle)!!;
    assert(@catch(vk::vk_get_sparse_texture_requirements(
        &fixture.state,
        ordinary_handle,
    )) == gpu::INVALID_HANDLE);
}

fn vk::SparseImageMemoryRequirements valid_sparse_color_requirement() {
    return {
        .format_properties = {
            .aspect_mask = (vk::ImageAspectFlags)
                vk::ImageAspectFlagBits.COLOR_BIT,
            .image_granularity = {
                .width  = 64,
                .height = 32,
                .depth  = 16,
            },
        },
        .image_mip_tail_first_lod = 4,
        .image_mip_tail_size      = 512,
        .image_mip_tail_offset    = 3072,
        .image_mip_tail_stride    = 128,
    };
}

fn void assert_sparse_requirement_backend_error(
    vk::MemoryRequirements* memory,
    vk::SparseImageMemoryRequirements* native,
    uint mip_levels,
) {
    assert(@catch(vk::sparse_texture_requirements_from_native(
        allocation_owner: 73,
        memory:           memory,
        native:           native[0..1],
        mip_levels:       mip_levels,
    )) == gpu::BACKEND_ERROR);
}

fn void sparse_color_requirements_translate_to_public_snapshot() @test {
    const ulong ALLOCATION_OWNER = 73;
    vk::MemoryRequirements memory = {
        .size             = 4096,
        .alignment        = 256,
        .memory_type_bits = 0b1010,
    };
    vk::SparseImageMemoryRequirements[1] native = {
        {
            .format_properties = {
                .aspect_mask = (vk::ImageAspectFlags)
                    vk::ImageAspectFlagBits.COLOR_BIT,
                .image_granularity = {
                    .width  = 64,
                    .height = 32,
                    .depth  = 16,
                },
                .flags = (vk::SparseImageFormatFlags)(
                    (uint)vk::SparseImageFormatFlagBits.ALIGNED_MIP_SIZE_BIT
                    | (uint)vk::SparseImageFormatFlagBits
                        .NONSTANDARD_BLOCK_SIZE_BIT
                ),
            },
            .image_mip_tail_first_lod = 4,
            .image_mip_tail_size      = 512,
            .image_mip_tail_offset    = 3072,
            .image_mip_tail_stride    = 128,
        },
    };

    gpu::SparseTextureRequirements requirements =
        vk::sparse_texture_requirements_from_native(
            allocation_owner: ALLOCATION_OWNER,
            memory:           &memory,
            native:           native[..],
            mip_levels:       7,
        )!!;

    assert(requirements.virtual_size == 4096);
    assert(requirements.page_allocation.size == 256);
    assert(requirements.page_allocation.alignment == 256);
    assert(requirements.page_allocation.compatibility
        == vk::texture_compatibility(ALLOCATION_OWNER, 0b1010));
    assert(!requirements.page_allocation.dedicated_only);
    assert(requirements.aspect_count == 1);
    assert(requirements.aspects[0].aspect == gpu::SparseTextureAspect.COLOR);
    assert(requirements.aspects[0].tile_extent.x == 64);
    assert(requirements.aspects[0].tile_extent.y == 32);
    assert(requirements.aspects[0].tile_extent.z == 16);
    assert(requirements.aspects[0].mip_tail_first_lod == 4);
    assert(requirements.aspects[0].mip_tail_size == 512);
    assert(requirements.aspects[0].mip_tail_offset == 3072);
    assert(requirements.aspects[0].mip_tail_stride == 128);
    assert(!requirements.aspects[0].single_mip_tail);
    assert(requirements.aspects[0].aligned_mip_size);
    assert(requirements.aspects[0].nonstandard_block_size);
    assert(requirements.aspects[1].tile_extent.x == 0);
    assert(requirements.aspects[1].tile_extent.y == 0);
    assert(requirements.aspects[1].tile_extent.z == 0);
    assert(requirements.aspects[1].mip_tail_first_lod == 0);
    assert(requirements.aspects[1].mip_tail_size == 0);
    assert(requirements.aspects[1].mip_tail_offset == 0);
    assert(requirements.aspects[1].mip_tail_stride == 0);
    assert(!requirements.aspects[1].single_mip_tail);
    assert(!requirements.aspects[1].aligned_mip_size);
    assert(!requirements.aspects[1].nonstandard_block_size);
}

fn void sparse_metadata_is_canonical_and_accepts_zero_granularity() @test {
    vk::MemoryRequirements memory = {
        .size             = 8192,
        .alignment        = 512,
        .memory_type_bits = 0b11,
    };
    vk::SparseImageMemoryRequirements[2] native = {
        {
            .format_properties = {
                .aspect_mask = (vk::ImageAspectFlags)
                    vk::ImageAspectFlagBits.METADATA_BIT,
                .flags = (vk::SparseImageFormatFlags)
                    vk::SparseImageFormatFlagBits.SINGLE_MIPTAIL_BIT,
            },
            .image_mip_tail_first_lod = 3,
            .image_mip_tail_size      = 512,
            .image_mip_tail_offset    = 7168,
            .image_mip_tail_stride    = 999,
        },
        {
            .format_properties = {
                .aspect_mask = (vk::ImageAspectFlags)
                    vk::ImageAspectFlagBits.COLOR_BIT,
                .image_granularity = {
                    .width  = 32,
                    .height = 16,
                    .depth  = 8,
                },
            },
            .image_mip_tail_first_lod = 5,
            .image_mip_tail_size      = 1024,
            .image_mip_tail_offset    = 6144,
            .image_mip_tail_stride    = 256,
        },
    };

    gpu::SparseTextureRequirements requirements =
        vk::sparse_texture_requirements_from_native(
            allocation_owner: 91,
            memory:           &memory,
            native:           native[..],
            mip_levels:       6,
        )!!;

    assert(requirements.aspect_count == 2);
    assert(requirements.aspects[0].aspect == gpu::SparseTextureAspect.COLOR);
    assert(requirements.aspects[0].tile_extent.x == 32);
    assert(requirements.aspects[0].mip_tail_stride == 256);
    assert(requirements.aspects[1].aspect
        == gpu::SparseTextureAspect.METADATA);
    assert(requirements.aspects[1].tile_extent.x == 0);
    assert(requirements.aspects[1].tile_extent.y == 0);
    assert(requirements.aspects[1].tile_extent.z == 0);
    assert(requirements.aspects[1].mip_tail_first_lod == 3);
    assert(requirements.aspects[1].mip_tail_size == 512);
    assert(requirements.aspects[1].mip_tail_offset == 7168);
    assert(requirements.aspects[1].single_mip_tail);
    assert(requirements.aspects[1].mip_tail_stride == 0);
}

fn void malformed_sparse_numeric_requirements_fault_backend_error() @test {
    vk::MemoryRequirements memory = {
        .size             = 4096,
        .alignment        = 256,
        .memory_type_bits = 0b1010,
    };
    vk::SparseImageMemoryRequirements native =
        valid_sparse_color_requirement();

    memory.size = 0;
    assert_sparse_requirement_backend_error(&memory, &native, 7);
    memory.size = 4096;

    memory.alignment = 0;
    assert_sparse_requirement_backend_error(&memory, &native, 7);
    memory.alignment = 192;
    assert_sparse_requirement_backend_error(&memory, &native, 7);
    memory.alignment = 256;

    memory.memory_type_bits = 0;
    assert_sparse_requirement_backend_error(&memory, &native, 7);
    memory.memory_type_bits = 0b1010;

    native.format_properties.image_granularity.width = 0;
    assert_sparse_requirement_backend_error(&memory, &native, 7);
    native = valid_sparse_color_requirement();
    native.format_properties.image_granularity.height = 0;
    assert_sparse_requirement_backend_error(&memory, &native, 7);
    native = valid_sparse_color_requirement();
    native.format_properties.image_granularity.depth = 0;
    assert_sparse_requirement_backend_error(&memory, &native, 7);
    native = valid_sparse_color_requirement();

    native.image_mip_tail_first_lod = 8;
    assert_sparse_requirement_backend_error(&memory, &native, 7);
    native = valid_sparse_color_requirement();

    native.image_mip_tail_offset = 3840;
    assert_sparse_requirement_backend_error(&memory, &native, 7);
    native = valid_sparse_color_requirement();
    native.image_mip_tail_offset = 4097;
    assert_sparse_requirement_backend_error(&memory, &native, 7);
}

fn void zero_sparse_requirement_count_faults_backend_error() @test {
    vk::MemoryRequirements memory = {
        .size             = 4096,
        .alignment        = 256,
        .memory_type_bits = 0b1010,
    };
    vk::SparseImageMemoryRequirements[] native;

    assert(@catch(vk::sparse_texture_requirements_from_native(
        allocation_owner: 73,
        memory:           &memory,
        native:           native[..],
        mip_levels:       7,
    )) == gpu::BACKEND_ERROR);
}

fn void unknown_sparse_format_flags_fault_unsupported() @test {
    vk::MemoryRequirements memory = {
        .size             = 4096,
        .alignment        = 256,
        .memory_type_bits = 0b1010,
    };
    vk::SparseImageMemoryRequirements native =
        valid_sparse_color_requirement();
    native.format_properties.flags = (vk::SparseImageFormatFlags)0x8;

    assert(@catch(vk::sparse_texture_requirements_from_native(
        allocation_owner: 73,
        memory:           &memory,
        native:           (&native)[:1],
        mip_levels:       7,
    )) == gpu::UNSUPPORTED_FEATURE);
}

fn void duplicate_sparse_color_aspects_fault_unsupported() @test {
    vk::MemoryRequirements memory = {
        .size             = 4096,
        .alignment        = 256,
        .memory_type_bits = 0b1010,
    };
    vk::SparseImageMemoryRequirements[2] native = {
        valid_sparse_color_requirement(),
        valid_sparse_color_requirement(),
    };

    assert(@catch(vk::sparse_texture_requirements_from_native(
        allocation_owner: 73,
        memory:           &memory,
        native:           native[..],
        mip_levels:       7,
    )) == gpu::UNSUPPORTED_FEATURE);
}

fn void assert_sparse_shape_unsupported(
    vk::MemoryRequirements* memory,
    vk::SparseImageMemoryRequirements[] native,
) {
    assert(@catch(vk::sparse_texture_requirements_from_native(
        allocation_owner: 73,
        memory:           memory,
        native:           native,
        mip_levels:       7,
    )) == gpu::UNSUPPORTED_FEATURE);
}

fn void unsupported_sparse_aspect_shapes_fault_unsupported() @test {
    vk::MemoryRequirements memory = {
        .size             = 4096,
        .alignment        = 256,
        .memory_type_bits = 0b1010,
    };
    vk::SparseImageMemoryRequirements color =
        valid_sparse_color_requirement();
    vk::SparseImageMemoryRequirements metadata = color;
    metadata.format_properties.aspect_mask = (vk::ImageAspectFlags)
        vk::ImageAspectFlagBits.METADATA_BIT;
    metadata.format_properties.image_granularity = {};

    vk::SparseImageMemoryRequirements[3] three = {
        color,
        metadata,
        color,
    };
    assert_sparse_shape_unsupported(&memory, three[..]);

    assert_sparse_shape_unsupported(&memory, (&metadata)[:1]);

    vk::SparseImageMemoryRequirements[2] duplicate_metadata = {
        metadata,
        metadata,
    };
    assert_sparse_shape_unsupported(&memory, duplicate_metadata[..]);

    vk::SparseImageMemoryRequirements invalid = color;
    invalid.format_properties.aspect_mask = (vk::ImageAspectFlags)(
        (uint)vk::ImageAspectFlagBits.COLOR_BIT
        | (uint)vk::ImageAspectFlagBits.METADATA_BIT
    );
    assert_sparse_shape_unsupported(&memory, (&invalid)[:1]);

    invalid.format_properties.aspect_mask =
        (vk::ImageAspectFlags)vk::ImageAspectFlagBits.DEPTH_BIT;
    assert_sparse_shape_unsupported(&memory, (&invalid)[:1]);

    invalid.format_properties.aspect_mask = (vk::ImageAspectFlags)0;
    assert_sparse_shape_unsupported(&memory, (&invalid)[:1]);
}

fn void sparse_texture_real_lifecycle_when_supported() @test {
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation = gpu::ContractValidation.FULL,
        .application_name    = "gpu_sparse_texture_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceDesc device_desc = {
        .enable_sparse_textures = true,
    };
    gpu::Format[5] formats = {
        gpu::Format.RGBA8_UNORM,
        gpu::Format.R32_FLOAT,
        gpu::Format.R8_UINT,
        gpu::Format.RG16_FLOAT,
        gpu::Format.RGBA16_FLOAT,
    };

    for (uint adapter_index = 0;
        adapter_index < adapters.count;
        adapter_index++) {
        gpu::Adapter adapter = adapters.get(adapter_index)!!;
        if (!gpu::supports_device_desc(&adapter, &device_desc)!!.supported) {
            continue;
        }
        gpu::Device device = gpu::create_device(
            &adapter,
            &device_desc,
        )!!;
        gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
        gpu::QueueKind queue_kind;
        gpu::QueueRoles access;
        if (caps.queues.graphics) {
            queue_kind = gpu::QueueKind.GRAPHICS;
            access = { .graphics };
        } else if (caps.queues.compute) {
            queue_kind = gpu::QueueKind.COMPUTE;
            access = { .compute };
        } else {
            queue_kind = gpu::QueueKind.TRANSFER;
            access = { .transfer };
        }
        gpu::TextureDesc desc = {
            .access = access,
            .width  = 64,
            .height = 64,
            .depth  = caps.sparse_textures.image_2d ? 0 : 8,
            .format = gpu::Format.RGBA8_UNORM,
            .usage  = { .sampled },
        };

        for (uint format_index = 0;
            format_index < formats.len;
            format_index++) {
            desc.format = formats[format_index];
            gpu::TextureHandle? created =
                gpu::create_sparse_texture(&device, &desc);
            if (catch create_fault = created) {
                assert(create_fault == gpu::UNSUPPORTED_FEATURE);
                continue;
            }
            gpu::TextureHandle texture = created;
            gpu::SparseTextureRequirements requirements =
                gpu::get_sparse_texture_requirements(
                    &device,
                    texture,
                )!!;
            assert(requirements.virtual_size > 0);
            assert(requirements.page_allocation.size > 0);
            assert(requirements.aspect_count > 0);
            gpu::TextureRequirements[1] page_requirements = {
                requirements.page_allocation,
            };
            gpu::AllocationDesc page_desc = {
                .size                 =
                    requirements.page_allocation.size,
                .alignment            =
                    requirements.page_allocation.alignment,
                .memory_class         = gpu::MemoryClass.TEXTURE,
                .access               = desc.access,
                .texture_requirements = page_requirements[..],
            };
            gpu::GpuAllocation pages =
                gpu::allocate_memory(&device, &page_desc)!!;
            gpu::TextureViewDesc view_desc;
            gpu::TextureView view = gpu::create_texture_view(
                &device,
                texture,
                &view_desc,
            )!!;
            assert(@catch(gpu::destroy_texture(&device, texture))
                == gpu::RESOURCE_IN_USE);
            gpu::destroy_texture_view(&device, view)!!;
            gpu::destroy_texture(&device, texture)!!;
            assert(gpu::get_allocation_info(&device, pages)!!.size
                == page_desc.size);
            gpu::free_allocation(&device, &pages)!!;

            gpu::TextureDesc transfer_desc = desc;
            transfer_desc.usage = { .transfer_dst };
            gpu::TextureHandle? transfer_created =
                gpu::create_sparse_texture(&device, &transfer_desc);
            if (try transfer = transfer_created) {
                OwnedSpan upload = allocate_owned_span(
                    device:       &device,
                    size:         64 * 64 * 4,
                    memory_class: gpu::MemoryClass.CPU_WRITE,
                    access:       access,
                    debug_name:   "sparse_transfer_probe",
                )!!;
                gpu::Queue queue = gpu::get_queue(
                    &device,
                    queue_kind,
                )!!;
                gpu::CommandAllocator allocator =
                    gpu::create_command_allocator(&device, queue)!!;
                gpu::CommandList commands =
                    gpu::begin_commands(&allocator)!!;
                gpu::TextureBarrier barrier = {
                    .texture = transfer,
                    .before  = { .layout = gpu::TextureLayout.UNDEFINED },
                    .after   = {
                        .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
                        .stages = { .transfer },
                        .access = { .write },
                    },
                };
                gpu::cmd_texture_barrier(&commands, &barrier)!!;
                gpu::BufferTextureCopyDesc copy = {
                    .src     = upload.span,
                    .texture = transfer,
                };
                gpu::cmd_copy_buffer_to_texture(&commands, &copy)!!;
                assert(@catch(gpu::destroy_texture(&device, transfer))
                    == gpu::RESOURCE_IN_USE);
                gpu::discard_commands(&commands)!!;
                gpu::destroy_texture(&device, transfer)!!;
                gpu::destroy_command_allocator(&allocator)!!;
                free_owned_span(&device, &upload)!!;
            } else {
                assert(@catch(transfer_created)
                    == gpu::UNSUPPORTED_FEATURE);
            }

            gpu::TextureHandle teardown_texture =
                gpu::create_sparse_texture(&device, &desc)!!;
            assert(teardown_texture.is_valid());
            gpu::internal::mark_device_lost(&device);
            gpu::destroy_device(&device)!!;
            return;
        }
        gpu::destroy_device(&device)!!;
    }
}
