module gpu::internal::vk @private;

import gpu::internal @public;
import gpu;
import vk;

alias SparseTextureSupportsImageDescFn = fn bool? (
    void*, VkDeviceState*, gpu::TextureDesc*, vk::ImageCreateFlags
);
alias SparseTextureFormatPropertyCountFn = fn uint? (
    void*, VkDeviceState*, gpu::TextureDesc*
);
alias SparseTextureGetMemoryRequirementsFn = fn void (
    void*, vk::Device, vk::Image, vk::MemoryRequirements*
);
alias SparseTextureGetSparseRequirementsFn = fn void (
    void*, vk::Device, vk::Image, uint*, vk::SparseImageMemoryRequirements*
);

struct SparseTextureNativeOps {
    void*                                  user_data;
    SparseTextureSupportsImageDescFn       supports_image_desc;
    SparseTextureFormatPropertyCountFn     sparse_format_property_count;
    TextureCreateImageFn                   create_image;
    SparseTextureGetMemoryRequirementsFn   get_memory_requirements;
    SparseTextureGetSparseRequirementsFn   get_sparse_requirements;
    TextureCreateViewFn                    create_view;
    TextureDestroyImageFn                  destroy_image;
    TextureDestroyViewFn                   destroy_view;
}

struct SparseTextureFeaturePlan {
    bool enabled;
    bool image_2d;
    bool image_3d;
}

fn gpu::SparseTextureAspectRequirements sparse_aspect_requirements_from_native(
    vk::SparseImageMemoryRequirements* native,
    gpu::SparseTextureAspect aspect,
) {
    uint flags = (uint)native.format_properties.flags;
    bool single_mip_tail = (flags
        & (uint)vk::SparseImageFormatFlagBits.SINGLE_MIPTAIL_BIT) != 0;
    return {
        .aspect = aspect,
        .tile_extent = {
            .x = native.format_properties.image_granularity.width,
            .y = native.format_properties.image_granularity.height,
            .z = native.format_properties.image_granularity.depth,
        },
        .mip_tail_first_lod = native.image_mip_tail_first_lod,
        .mip_tail_size      = (usz)native.image_mip_tail_size,
        .mip_tail_offset    = (usz)native.image_mip_tail_offset,
        .mip_tail_stride    = single_mip_tail
            ? 0
            : (usz)native.image_mip_tail_stride,
        .single_mip_tail = single_mip_tail,
        .aligned_mip_size = (flags
            & (uint)vk::SparseImageFormatFlagBits
                .ALIGNED_MIP_SIZE_BIT) != 0,
        .nonstandard_block_size = (flags
            & (uint)vk::SparseImageFormatFlagBits
                .NONSTANDARD_BLOCK_SIZE_BIT) != 0,
    };
}

fn gpu::SparseTextureRequirements? sparse_texture_requirements_from_native(
    ulong allocation_owner,
    vk::MemoryRequirements* memory,
    vk::SparseImageMemoryRequirements[] native,
    uint mip_levels,
) {
    if (memory == null || native.len == 0) return gpu::BACKEND_ERROR~;
    if (native.len > 2) return gpu::UNSUPPORTED_FEATURE~;
    usz virtual_size = (usz)memory.size;
    usz alignment = (usz)memory.alignment;
    if (memory.size == 0
        || (vk::DeviceSize)virtual_size != memory.size
        || memory.alignment == 0
        || (vk::DeviceSize)alignment != memory.alignment
        || (alignment & (alignment - 1)) != 0
        || memory.memory_type_bits == 0) {
        return gpu::BACKEND_ERROR~;
    }
    vk::SparseImageMemoryRequirements* color = null;
    vk::SparseImageMemoryRequirements* metadata = null;
    foreach (&entry : native) {
        switch ((uint)entry.format_properties.aspect_mask) {
            case (uint)vk::ImageAspectFlagBits.COLOR_BIT:
                if (color != null) return gpu::UNSUPPORTED_FEATURE~;
                color = entry;
            case (uint)vk::ImageAspectFlagBits.METADATA_BIT:
                if (metadata != null) return gpu::UNSUPPORTED_FEATURE~;
                metadata = entry;
            default:
                return gpu::UNSUPPORTED_FEATURE~;
        }
        uint flags = (uint)entry.format_properties.flags;
        const uint KNOWN_SPARSE_FORMAT_FLAGS =
            (uint)vk::SparseImageFormatFlagBits.SINGLE_MIPTAIL_BIT
            | (uint)vk::SparseImageFormatFlagBits.ALIGNED_MIP_SIZE_BIT
            | (uint)vk::SparseImageFormatFlagBits
                .NONSTANDARD_BLOCK_SIZE_BIT;
        if ((flags & ~KNOWN_SPARSE_FORMAT_FLAGS) != 0) {
            return gpu::UNSUPPORTED_FEATURE~;
        }
        bool is_color = (uint)entry.format_properties.aspect_mask
            == (uint)vk::ImageAspectFlagBits.COLOR_BIT;
        vk::Extent3D granularity =
            entry.format_properties.image_granularity;
        if ((is_color && (granularity.width == 0
                || granularity.height == 0
                || granularity.depth == 0))
            || entry.image_mip_tail_first_lod > mip_levels) {
            return gpu::BACKEND_ERROR~;
        }
        usz tail_size = (usz)entry.image_mip_tail_size;
        usz tail_offset = (usz)entry.image_mip_tail_offset;
        if ((vk::DeviceSize)tail_size != entry.image_mip_tail_size
            || (vk::DeviceSize)tail_offset != entry.image_mip_tail_offset
            || tail_offset > virtual_size
            || tail_size > virtual_size - tail_offset) {
            return gpu::BACKEND_ERROR~;
        }
        bool single_mip_tail = ((uint)entry.format_properties.flags
            & (uint)vk::SparseImageFormatFlagBits
                .SINGLE_MIPTAIL_BIT) != 0;
        if (!single_mip_tail) {
            usz tail_stride = (usz)entry.image_mip_tail_stride;
            if ((vk::DeviceSize)tail_stride
                != entry.image_mip_tail_stride) {
                return gpu::BACKEND_ERROR~;
            }
        }
    }
    if (color == null) return gpu::UNSUPPORTED_FEATURE~;
    gpu::SparseTextureRequirements requirements = {
        .virtual_size = virtual_size,
        .page_allocation = {
            .size      = alignment,
            .alignment = alignment,
            .compatibility = texture_compatibility(
                allocation_owner,
                memory.memory_type_bits,
            ),
        },
        .aspect_count = metadata == null ? 1 : 2,
    };
    requirements.aspects[0] = sparse_aspect_requirements_from_native(
        color,
        gpu::SparseTextureAspect.COLOR,
    );
    if (metadata != null) {
        requirements.aspects[1] = sparse_aspect_requirements_from_native(
            metadata,
            gpu::SparseTextureAspect.METADATA,
        );
    }
    return requirements;
}

fn bool sparse_texture_desc_valid(gpu::TextureDesc* desc) {
    if (desc == null || !gpu::internal::texture_desc_valid(desc)) {
        return false;
    }
    return gpu::internal::texture_array_layers(desc) == 1
        && desc.sample_count == gpu::SampleCount.ONE
        && desc.format != gpu::Format.D32_FLOAT
        && !desc.usage.color_attach
        && !desc.usage.depth_attach
        && (desc.usage.sampled
            || desc.usage.storage
            || desc.usage.transfer_src
            || desc.usage.transfer_dst);
}

fn gpu::TextureHandle? create_sparse_texture_for_state_with_ops(
    VkDeviceState* state,
    gpu::TextureDesc* desc,
    SparseTextureNativeOps* ops,
) {
    if (!sparse_texture_desc_valid(desc)) return gpu::INVALID_ARGUMENT~;
    bool is_3d = gpu::internal::texture_desc_is_3d(desc);
    if ((is_3d && !state.caps.sparse_textures.image_3d)
        || (!is_3d && !state.caps.sparse_textures.image_2d)) {
        return gpu::UNSUPPORTED_FEATURE~;
    }

    vk::ImageCreateFlags sparse_flags =
        (vk::ImageCreateFlags)vk::ImageCreateFlagBits.SPARSE_BINDING_BIT
        | (vk::ImageCreateFlags)vk::ImageCreateFlagBits.SPARSE_RESIDENCY_BIT;
    if (!ops.supports_image_desc(
        ops.user_data,
        state,
        desc,
        sparse_flags,
    )!) return gpu::UNSUPPORTED_FEATURE~;
    if (ops.sparse_format_property_count(
        ops.user_data,
        state,
        desc,
    )! == 0) return gpu::UNSUPPORTED_FEATURE~;

    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();

    uint[MAX_QUEUE_FAMILIES] families;
    uint family_count = resource_queue_families(
        state,
        desc.access,
        families[..],
    );
    vk::ImageCreateInfo image_info = make_texture_create_info(
        desc,
        families[:family_count],
        sparse_flags,
    );
    uint mip_levels = gpu::internal::texture_mip_levels(desc);
    uint array_layers = gpu::internal::texture_array_layers(desc);
    TextureSlot slot = {
        .format                = desc.format,
        .usage                 = desc.usage,
        .access                = desc.access,
        .sample_count          = desc.sample_count,
        .width                 = desc.width,
        .height                = desc.height,
        .depth                 = desc.depth,
        .mip_levels            = mip_levels,
        .array_layers          = array_layers,
        .live_descriptor_views = 0,
        .sparse                = true,
    };
    bool owns_image;
    bool owns_view;
    defer catch {
        if (owns_view) {
            ops.destroy_view(
                ops.user_data,
                state.device,
                slot.default_view,
            );
        }
        if (owns_image) {
            ops.destroy_image(
                ops.user_data,
                state.device,
                slot.image,
            );
        }
    }

    check_texture_create_device_result(
        state: state,
        result: ops.create_image(
            ops.user_data,
            state.device,
            &image_info,
            &slot.image,
        ),
    )!;
    owns_image = true;

    vk::MemoryRequirements memory;
    ops.get_memory_requirements(
        ops.user_data,
        state.device,
        slot.image,
        &memory,
    );
    uint requirement_count;
    ops.get_sparse_requirements(
        ops.user_data,
        state.device,
        slot.image,
        &requirement_count,
        null,
    );
    if (requirement_count == 0) return gpu::BACKEND_ERROR~;
    if (requirement_count > gpu::MAX_SPARSE_TEXTURE_ASPECTS) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    vk::SparseImageMemoryRequirements[
        gpu::MAX_SPARSE_TEXTURE_ASPECTS
    ] native_requirements;
    uint filled_count = gpu::MAX_SPARSE_TEXTURE_ASPECTS;
    ops.get_sparse_requirements(
        ops.user_data,
        state.device,
        slot.image,
        &filled_count,
        &native_requirements[0],
    );
    if (filled_count != requirement_count) return gpu::BACKEND_ERROR~;
    slot.sparse_requirements = sparse_texture_requirements_from_native(
        state.allocations.owner,
        &memory,
        native_requirements[:filled_count],
        mip_levels,
    )!;

    copy_debug_name(&slot.debug_name[0], desc.debug_name);
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.IMAGE,
        object_handle: (ulong)slot.image,
        name:          desc.debug_name,
    );
    if (texture_needs_default_view(desc.usage)) {
        vk::ImageViewCreateInfo view_info = default_texture_view_create_info(
            desc:         desc,
            image:        slot.image,
            mip_levels:   mip_levels,
            array_layers: array_layers,
        );
        check_device_result(state, ops.create_view(
            ops.user_data,
            state.device,
            &view_info,
            &slot.default_view,
        ))!;
        owns_view = true;
    }

    gpu::TextureHandle handle = state.textures.alloc(slot)!;
    owns_image = false;
    owns_view = false;
    return handle;
}

fn SparseTextureFeaturePlan? resolve_sparse_texture_feature_plan(
    bool requested,
    vk::PhysicalDeviceFeatures* features,
) {
    if (!requested) return {};
    if (features == null) return gpu::INVALID_ARGUMENT~;
    if (features.sparse_binding == 0
        || (features.sparse_residency_image2d == 0
            && features.sparse_residency_image3d == 0)) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    return {
        .enabled  = true,
        .image_2d = features.sparse_residency_image2d != 0,
        .image_3d = features.sparse_residency_image3d != 0,
    };
}

fn bool queue_family_supports_sparse_binding(
    vk::QueueFamilyProperties family,
) {
    return ((uint)family.queue_flags
        & (uint)vk::QueueFlagBits.SPARSE_BINDING_BIT) != 0;
}

fn gpu::QueueRoles sparse_binding_queue_roles(
    QueueFamilies* selection,
    vk::QueueFamilyProperties[] families,
) {
    gpu::QueueRoles roles;
    if (selection.selected.graphics
        && selection.graphics < families.len
        && queue_family_supports_sparse_binding(families[selection.graphics])) {
        roles.graphics = true;
    }
    if (selection.selected.compute
        && selection.compute < families.len
        && queue_family_supports_sparse_binding(families[selection.compute])) {
        roles.compute = true;
    }
    if (selection.selected.transfer
        && selection.transfer < families.len
        && queue_family_supports_sparse_binding(families[selection.transfer])) {
        roles.transfer = true;
    }
    return roles;
}

fn bool? vk_adapter_supports_sparse_textures(
    VkRuntimeState* state,
    uint adapter_index,
) {
    if (state == null || adapter_index >= state.adapters.len) {
        return gpu::INVALID_HANDLE~;
    }
    vk::PhysicalDeviceFeatures2 features2 =
        vk::physical_device_features2();
    vk::get_physical_device_features2(
        state.adapters[adapter_index].physical_device,
        &features2,
    );
    SparseTextureFeaturePlan? plan = resolve_sparse_texture_feature_plan(
        true,
        &features2.features,
    );
    if (catch plan_fault = plan) {
        if (plan_fault == gpu::UNSUPPORTED_FEATURE) return false;
        return plan_fault~;
    }
    return true;
}

fn bool? sparse_texture_supports_image_desc_real(
    void* user_data,
    VkDeviceState* state,
    gpu::TextureDesc* desc,
    vk::ImageCreateFlags flags,
) {
    (void)user_data;
    return texture_desc_supported(state, desc, flags);
}

fn uint? sparse_texture_format_property_count_real(
    void* user_data,
    VkDeviceState* state,
    gpu::TextureDesc* desc,
) {
    (void)user_data;
    uint count;
    vk::get_physical_device_sparse_image_format_properties(
        state.physical_device,
        format_to_vk(desc.format),
        gpu::internal::texture_desc_is_3d(desc)
            ? vk::ImageType.TYPE_3D
            : vk::ImageType.TYPE_2D,
        sample_count_to_vk(desc.sample_count),
        texture_usage_to_vk(desc.usage),
        vk::ImageTiling.OPTIMAL,
        &count,
        null,
    );
    return count;
}

fn void sparse_texture_get_memory_requirements_real(
    void* user_data,
    vk::Device device,
    vk::Image image,
    vk::MemoryRequirements* out_memory,
) {
    (void)user_data;
    vk::get_image_memory_requirements(device, image, out_memory);
}

fn void sparse_texture_get_sparse_requirements_real(
    void* user_data,
    vk::Device device,
    vk::Image image,
    uint* count,
    vk::SparseImageMemoryRequirements* out_requirements,
) {
    (void)user_data;
    vk::get_image_sparse_memory_requirements(
        device,
        image,
        count,
        out_requirements,
    );
}

fn SparseTextureNativeOps sparse_texture_native_ops_real() {
    return {
        .supports_image_desc          =
            &sparse_texture_supports_image_desc_real,
        .sparse_format_property_count =
            &sparse_texture_format_property_count_real,
        .create_image                 = &texture_create_image_real,
        .get_memory_requirements      =
            &sparse_texture_get_memory_requirements_real,
        .get_sparse_requirements      =
            &sparse_texture_get_sparse_requirements_real,
        .create_view                  = &texture_create_view_real,
        .destroy_image                = &texture_destroy_image_real,
        .destroy_view                 = &texture_destroy_view_real,
    };
}

fn gpu::TextureHandle? vk_create_sparse_texture(
    VkDeviceState* state,
    gpu::TextureDesc* desc,
) {
    SparseTextureNativeOps ops = sparse_texture_native_ops_real();
    return create_sparse_texture_for_state_with_ops(state, desc, &ops);
}

fn gpu::SparseTextureRequirements? vk_get_sparse_texture_requirements(
    VkDeviceState* state,
    gpu::TextureHandle texture,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    TextureSlot* slot = state.textures.get(texture)!;
    if (!slot.sparse) return gpu::INVALID_ARGUMENT~;
    return slot.sparse_requirements;
}
