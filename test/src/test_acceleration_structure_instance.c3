module gpu_test;

import gpu::internal @public;
import gpu::internal::vk @public;
import gpu @public;

struct InstanceFlagCase {
    gpu::AccelerationStructureInstanceFlags flags;
    uint                                    expected;
}

fn gpu::AccelerationStructureInstanceDesc instance_desc() {
    return {
        .transform_row_0 = { 1.0f, 2.0f, 3.0f, 4.0f },
        .transform_row_1 = { 5.0f, 6.0f, 7.0f, 8.0f },
        .transform_row_2 = { 9.0f, 10.0f, 11.0f, 12.0f },
        .custom_index = 0xA1_B2_C3,
        .mask = 0xD4,
    };
}

fn void instance_packing_matches_the_shader_abi_for_every_flag() @test {
    InstanceFlagCase[5] cases = {
        {},
        { .flags = { .triangle_cull_disable }, .expected = 0x01 },
        { .flags = { .triangle_facing_flip }, .expected = 0x02 },
        { .flags = { .force_opaque }, .expected = 0x04 },
        { .flags = { .force_non_opaque }, .expected = 0x08 },
    };
    gpu::GpuAddress address = (gpu::GpuAddress)0x0123_4567_89AB_CDEF;

    foreach (flag_case : cases) {
        gpu::AccelerationStructureInstanceDesc desc = instance_desc();
        desc.flags = flag_case.flags;

        gpu::AccelerationStructureInstance packed =
            gpu::internal::pack_acceleration_structure_instance(
                &desc,
                address,
            )!!;

        assert(packed.transform_row_0 == desc.transform_row_0);
        assert(packed.transform_row_1 == desc.transform_row_1);
        assert(packed.transform_row_2 == desc.transform_row_2);
        assert(packed.custom_index_and_mask == 0xD4_A1_B2_C3);
        assert(packed.record_offset_and_flags == flag_case.expected << 24);
        assert(packed.acceleration_structure == address);
    }
}

fn void instance_packing_rejects_every_unrepresentable_value() @test {
    gpu::AccelerationStructureInstanceDesc desc = instance_desc();
    gpu::GpuAddress address = (gpu::GpuAddress)1;

    desc.custom_index = 0x01_00_00_00;
    assert(@catch(gpu::internal::pack_acceleration_structure_instance(
        &desc,
        address,
    )) == gpu::INVALID_ARGUMENT);

    desc = instance_desc();
    desc.mask = 0x100;
    assert(@catch(gpu::internal::pack_acceleration_structure_instance(
        &desc,
        address,
    )) == gpu::INVALID_ARGUMENT);

    desc = instance_desc();
    desc.shader_binding_table_record_offset = 1;
    assert(@catch(gpu::internal::pack_acceleration_structure_instance(
        &desc,
        address,
    )) == gpu::INVALID_ARGUMENT);

    desc = instance_desc();
    desc.flags = { .force_opaque, .force_non_opaque };
    assert(@catch(gpu::internal::pack_acceleration_structure_instance(
        &desc,
        address,
    )) == gpu::INVALID_ARGUMENT);

    desc = instance_desc();
    desc.flags = (gpu::AccelerationStructureInstanceFlags)(1u << 4);
    assert(@catch(gpu::internal::pack_acceleration_structure_instance(
        &desc,
        address,
    )) == gpu::INVALID_ARGUMENT);

    desc = instance_desc();
    assert(@catch(gpu::internal::pack_acceleration_structure_instance(
        &desc,
        (gpu::GpuAddress)0,
    )) == gpu::INVALID_ARGUMENT);
}

fn void public_instance_packing_resolves_only_a_live_blas() @test {
    gpu::internal::vk::VkDeviceState state;
    state.resource_owner = gpu::internal::allocate_device_resource_owner()!!;
    gpu::Device device = gpu::internal::register_device(
        state:          &state,
        caps:           {},
        resource_owner: state.resource_owner,
    )!!;
    defer gpu::destroy_device(&device)!!;
    gpu::AccelerationStructureHandle handle = {
        .owner = gpu::internal::resource_owner_tag(
            state.resource_owner,
            gpu::internal::ResourceOwnerKind.ACCELERATION_STRUCTURE,
        ),
        .index      = 3,
        .generation = 7,
    };
    gpu::internal::vk::stub_set_acceleration_structure(
        state:                  &state,
        acceleration_structure: handle,
        kind:                   gpu::AccelerationStructureKind.BOTTOM_LEVEL,
        address:                (gpu::GpuAddress)0x1000,
    );
    gpu::AccelerationStructureInstanceDesc desc = instance_desc();
    desc.bottom_level = handle;

    gpu::AccelerationStructureInstance packed =
        gpu::make_acceleration_structure_instance(&device, &desc)!!;
    assert(packed.acceleration_structure == (gpu::GpuAddress)0x1000);

    state.acceleration_structure_kind =
        gpu::AccelerationStructureKind.TOP_LEVEL;
    assert(@catch(gpu::make_acceleration_structure_instance(&device, &desc))
        == gpu::INVALID_ARGUMENT);

    state.acceleration_structure_kind =
        gpu::AccelerationStructureKind.BOTTOM_LEVEL;
    desc.bottom_level.generation++;
    assert(@catch(gpu::make_acceleration_structure_instance(&device, &desc))
        == gpu::INVALID_HANDLE);
}
