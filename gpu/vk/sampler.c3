module gpu::vk @private;

import gpu @public;
import vk;
import std::core::mem;
import std::math;

struct SamplerCell {
    SamplerKey        key;
    vk::Sampler       native;
    gpu::Sampler      sampler;
    gpu::SamplerIndex published;
}

struct SamplerTable {
    SamplerCell[] slots;
    ulong         owner;
    uint          count;
}

fn SamplerKey? sampler_key_from_desc(
    VkDeviceState* state,
    gpu::SamplerDesc* desc,
) @private {
    ZString invalid_enum_field = sampler_desc_invalid_enum_field(desc);
    if (invalid_enum_field != null) {
        return report_contract_failure(
            state:               state,
            operation:           "intern_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      invalid_enum_field,
            invariant:           "must name a defined enum value",
            backend_text:        null,
        )~;
    }
    if (!math::is_finite(desc.mip_lod_bias)) {
        return report_contract_failure(
            state:               state,
            operation:           "intern_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.mip_lod_bias",
            invariant:           "must be finite",
            backend_text:        null,
        )~;
    }
    if (desc.mip_lod_bias < -state.caps.max_sampler_lod_bias
        || desc.mip_lod_bias > state.caps.max_sampler_lod_bias) {
        return report_contract_failure(
            state:               state,
            operation:           "intern_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.mip_lod_bias",
            invariant:           "absolute value must not exceed DeviceCaps.max_sampler_lod_bias",
            backend_text:        null,
        )~;
    }
    if (!math::is_finite(desc.min_lod)) {
        return report_contract_failure(
            state:               state,
            operation:           "intern_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.min_lod",
            invariant:           "must be finite",
            backend_text:        null,
        )~;
    }
    if (!math::is_finite(desc.max_lod)) {
        return report_contract_failure(
            state:               state,
            operation:           "intern_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.max_lod",
            invariant:           "must be finite",
            backend_text:        null,
        )~;
    }
    if (desc.min_lod > desc.max_lod) {
        return report_contract_failure(
            state:               state,
            operation:           "intern_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.min_lod",
            invariant:           "must not exceed SamplerDesc.max_lod",
            backend_text:        null,
        )~;
    }
    if (desc.anisotropy_enable && !math::is_finite(desc.max_anisotropy)) {
        return report_contract_failure(
            state:               state,
            operation:           "intern_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.max_anisotropy",
            invariant:           "must be finite when anisotropy is enabled",
            backend_text:        null,
        )~;
    }
    if (desc.anisotropy_enable && state.max_sampler_anisotropy <= 0) {
        return report_contract_failure(
            state:               state,
            operation:           "intern_sampler",
            public_fault:        gpu::UNSUPPORTED_FEATURE,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.anisotropy_enable",
            invariant:           "device must support sampler anisotropy",
            backend_text:        null,
        )~;
    }
    if (desc.anisotropy_enable && desc.max_anisotropy < 1.0f) {
        return report_contract_failure(
            state:               state,
            operation:           "intern_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.max_anisotropy",
            invariant:           "must be at least 1.0 when anisotropy is enabled",
            backend_text:        null,
        )~;
    }

    return canonical_sampler_key(desc, state.max_sampler_anisotropy);
}

fn vk::Filter sampler_filter_to_vk(gpu::Filter filter) @private {
    return filter == gpu::Filter.LINEAR ? vk::Filter.LINEAR : vk::Filter.NEAREST;
}

fn vk::SamplerAddressMode sampler_address_mode_to_vk(
    gpu::AddressMode mode,
) @private {
    switch (mode) {
        case gpu::AddressMode.REPEAT:          return vk::SamplerAddressMode.REPEAT;
        case gpu::AddressMode.MIRRORED_REPEAT: return vk::SamplerAddressMode.MIRRORED_REPEAT;
        case gpu::AddressMode.CLAMP_TO_EDGE:   return vk::SamplerAddressMode.CLAMP_TO_EDGE;
        case gpu::AddressMode.CLAMP_TO_BORDER: return vk::SamplerAddressMode.CLAMP_TO_BORDER;
        default:                               return vk::SamplerAddressMode.REPEAT;
    }
}

fn void? create_sampler_table(VkDeviceState* state) @private {
    vk::PhysicalDeviceProperties properties;
    vk::get_physical_device_properties(state.physical_device, &properties);
    uint capacity = properties.limits.max_sampler_allocation_count;
    if (capacity > gpu::MAX_SHADER_HEAP_CAPACITY) {
        capacity = gpu::MAX_SHADER_HEAP_CAPACITY;
    }
    if (capacity == 0) return gpu::UNSUPPORTED_FEATURE~;
    state.samplers.owner = gpu::resource_owner_tag(
        state.resource_owner,
        gpu::ResourceOwnerKind.SAMPLER,
    );
    state.samplers.slots = mem::new_array(SamplerCell, capacity);
}

fn void destroy_sampler_table(VkDeviceState* state) @private {
    SamplerTable* table = &state.samplers;
    if (table.slots.ptr == null) return;
    foreach (&cell : table.slots[:table.count]) {
        if (cell.native != null) {
            vk::destroy_sampler(state.device, cell.native, null);
            cell.native = (vk::Sampler)null;
        }
    }
    mem::free(table.slots.ptr);
    table.slots = {};
    table.count = 0;
}

fn SamplerCell*? sampler_cell_get(
    SamplerTable* table,
    gpu::Sampler sampler,
) @private {
    if (sampler.owner != table.owner
        || sampler.generation != 1
        || sampler.index >= table.count) {
        return gpu::INVALID_HANDLE~;
    }
    SamplerCell* cell = &table.slots[sampler.index];
    if (cell.sampler != sampler || cell.native == null) {
        return gpu::INVALID_HANDLE~;
    }
    return cell;
}

fn gpu::Sampler? vk_intern_sampler(
    gpu::Device* device,
    gpu::SamplerDesc* desc,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    SamplerKey key = sampler_key_from_desc(state, desc)!;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();

    SamplerTable* table = &state.samplers;
    foreach (&cell : table.slots[:table.count]) {
        if (sampler_key_equal(&cell.key, &key)) return cell.sampler;
    }
    if (table.count >= table.slots.len) return gpu::SLOT_TABLE_FULL~;

    vk::SamplerCreateInfo sampler_info = vk::sampler_create_info()
        .set_mag_filter(sampler_filter_to_vk(key.mag_filter))
        .set_min_filter(sampler_filter_to_vk(key.min_filter))
        .set_mipmap_mode(key.mip_filter == gpu::Filter.LINEAR
            ? vk::SamplerMipmapMode.LINEAR
            : vk::SamplerMipmapMode.NEAREST)
        .set_address_mode_u(sampler_address_mode_to_vk(key.address_u))
        .set_address_mode_v(sampler_address_mode_to_vk(key.address_v))
        .set_address_mode_w(sampler_address_mode_to_vk(key.address_w))
        .set_mip_lod_bias(key.mip_lod_bias)
        .set_anisotropy_enable(key.anisotropy_enable ? vk::TRUE : vk::FALSE)
        .set_max_anisotropy(key.max_anisotropy)
        .set_compare_enable(key.compare_enable ? vk::TRUE : vk::FALSE)
        .set_compare_op(compare_op_to_vk(key.compare))
        .set_min_lod(key.min_lod)
        .set_max_lod(key.max_lod);

    vk::Sampler native;
    check_backend_result(
        state:     state,
        operation: "intern_sampler",
        result:    vk::create_sampler(
            device:        state.device,
            p_create_info: &sampler_info,
            p_allocator:   null,
            p_sampler:     &native,
        ),
    )!;
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.SAMPLER,
        object_handle: (ulong)native,
        name:          desc.debug_name,
    );

    uint index = table.count;
    gpu::Sampler sampler = {
        .owner      = table.owner,
        .index      = index,
        .generation = 1,
    };
    table.slots[index] = {
        .key     = key,
        .native  = native,
        .sampler = sampler,
    };
    table.count++;
    return sampler;
}

fn gpu::SamplerIndex? vk_publish_sampler(
    gpu::Device* device,
    gpu::Sampler sampler,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();

    SamplerCell* cell = sampler_cell_get(&state.samplers, sampler)!;
    if (!state.strict_heap.enabled) return gpu::UNSUPPORTED_FEATURE~;
    if (cell.published.is_valid()) return cell.published;

    DescriptorHeapState* heap = &state.strict_heap.descriptors;
    if (heap.sampler_next_new >= state.strict_heap.sampler_capacity) {
        return report_contract_failure(
            state:               state,
            operation:           "publish_sampler",
            public_fault:        gpu::DESCRIPTOR_HEAP_FULL,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: true,
            index:               sampler.index,
            generation:          sampler.generation,
            debug_name:          null,
            rejected_field:      "DeviceDesc.sampler_heap_capacity",
            invariant:           "must exceed published samplers",
            backend_text:        null,
        )~;
    }
    uint slot = heap.sampler_next_new++;
    write_sampler_descriptor(state, slot, cell.native);
    cell.published = { .value = slot + 1 };
    return cell.published;
}
