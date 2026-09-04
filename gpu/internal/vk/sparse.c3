module gpu::internal::vk @private;

import gpu::internal @public;
import gpu;
import vk;
import std::core::mem::alloc;

alias QueueBindSparseFn = fn vk::Result (
    vk::Queue queue,
    uint bind_info_count,
    vk::BindSparseInfo* p_bind_info,
    vk::Fence fence,
);
alias GetSemaphoreCounterValueFn = fn vk::Result (
    vk::Device device,
    vk::Semaphore semaphore,
    ulong* p_value,
);

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

const uint KNOWN_SPARSE_FORMAT_FLAGS =
    (uint)vk::SparseImageFormatFlagBits.SINGLE_MIPTAIL_BIT
    | (uint)vk::SparseImageFormatFlagBits.ALIGNED_MIP_SIZE_BIT
    | (uint)vk::SparseImageFormatFlagBits.NONSTANDARD_BLOCK_SIZE_BIT;
const usz MAX_SPARSE_BIND_WAITS   = 1024;
const usz MAX_SPARSE_TILE_BINDS   = 1024;
const usz MAX_SPARSE_OPAQUE_BINDS = 1024;

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

struct SparseBindNativeOps {
    QueueBindSparseFn          queue_bind_sparse;
    GetSemaphoreCounterValueFn get_semaphore_counter_value;
}

struct SparseTextureFeaturePlan {
    bool enabled;
    bool image_2d;
    bool image_3d;
}

struct ResolvedSparseBacking {
    vk::DeviceMemory memory;
    usz              absolute_offset;
    usz              size;
}

struct SparseBackingInterval {
    vk::DeviceMemory memory;
    usz              offset;
    usz              size;
}

fn bool sparse_allocation_is_exact_invalid(
    gpu::GpuAllocation allocation,
) {
    return allocation.owner == 0
        && allocation.index == 0
        && allocation.generation == 0;
}

fn bool intervals_overlap(
    usz a_offset,
    usz a_size,
    usz b_offset,
    usz b_size,
) {
    if (a_size == 0 || b_size == 0) return false;
    if (a_offset <= b_offset) return b_offset - a_offset < a_size;
    return a_offset - b_offset < b_size;
}

fn gpu::Vec3u? sparse_mip_extent(
    bool image_3d,
    gpu::Vec3u base_extent,
    uint mip_level,
) {
    if (base_extent.x == 0
        || base_extent.y == 0
        || base_extent.z == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    gpu::Vec3u extent = {
        .x = mip_level < 32 ? max(1u, base_extent.x >> mip_level) : 1,
        .y = mip_level < 32 ? max(1u, base_extent.y >> mip_level) : 1,
        .z = image_3d && mip_level < 32
            ? max(1u, base_extent.z >> mip_level)
            : 1,
    };
    if (!image_3d) extent.z = 1;
    return extent;
}

fn usz? sparse_tile_backing_size(
    gpu::Vec3u extent,
    gpu::Vec3u tile_extent,
    usz page_size,
) {
    if (extent.x == 0
        || extent.y == 0
        || extent.z == 0
        || tile_extent.x == 0
        || tile_extent.y == 0
        || tile_extent.z == 0
        || page_size == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    usz tile_count_x =
        ((usz)extent.x - 1) / (usz)tile_extent.x + 1;
    usz tile_count_y =
        ((usz)extent.y - 1) / (usz)tile_extent.y + 1;
    usz tile_count_z =
        ((usz)extent.z - 1) / (usz)tile_extent.z + 1;
    if (tile_count_x > usz::max / tile_count_y) {
        return gpu::INVALID_ARGUMENT~;
    }
    usz tile_count = tile_count_x * tile_count_y;
    if (tile_count > usz::max / tile_count_z) {
        return gpu::INVALID_ARGUMENT~;
    }
    tile_count *= tile_count_z;
    if (tile_count > usz::max / page_size) {
        return gpu::INVALID_ARGUMENT~;
    }
    return tile_count * page_size;
}

fn gpu::SparseTextureAspectRequirements*? sparse_aspect_requirements(
    TextureSlot* texture,
    gpu::SparseTextureAspect aspect,
) {
    for (uint i = 0; i < texture.sparse_requirements.aspect_count; i++) {
        if (texture.sparse_requirements.aspects[i].aspect == aspect) {
            return &texture.sparse_requirements.aspects[i];
        }
    }
    return gpu::INVALID_ARGUMENT~;
}

fn bool sparse_tile_boxes_overlap(
    gpu::SparseTextureTileBind* a,
    gpu::SparseTextureTileBind* b,
) {
    if (a.aspect != b.aspect || a.mip_level != b.mip_level) return false;
    return intervals_overlap(a.offset.x, a.extent.x, b.offset.x, b.extent.x)
        && intervals_overlap(a.offset.y, a.extent.y, b.offset.y, b.extent.y)
        && intervals_overlap(a.offset.z, a.extent.z, b.offset.z, b.extent.z);
}

fn void? validate_sparse_tile_geometry(
    TextureSlot* texture,
    gpu::SparseTextureTileBind[] binds,
) {
    gpu::SparseTextureAspectRequirements* requirements =
        sparse_aspect_requirements(
            texture,
            gpu::SparseTextureAspect.COLOR,
        )!;
    bool image_3d = texture_slot_is_3d(texture);
    gpu::Vec3u base_extent = {
        texture.width,
        texture.height,
        texture_slot_native_depth(texture),
    };
    foreach (i, &bind : binds) {
        if (bind.aspect != gpu::SparseTextureAspect.COLOR
            || bind.mip_level >= texture.mip_levels
            || bind.mip_level >= requirements.mip_tail_first_lod) {
            return gpu::INVALID_ARGUMENT~;
        }
        gpu::Vec3u mip_extent = sparse_mip_extent(
            image_3d:   image_3d,
            base_extent: base_extent,
            mip_level:  bind.mip_level,
        )!;
        gpu::Vec3u tile = requirements.tile_extent;
        if (bind.extent.x == 0 || bind.extent.y == 0 || bind.extent.z == 0
            || tile.x == 0 || tile.y == 0 || tile.z == 0
            || bind.offset.x % tile.x != 0
            || bind.offset.y % tile.y != 0
            || bind.offset.z % tile.z != 0
            || bind.offset.x > (uint)int::max
            || bind.offset.y > (uint)int::max
            || bind.offset.z > (uint)int::max
            || !gpu::internal::range_fits(
                mip_extent.x,
                bind.offset.x,
                bind.extent.x,
            )
            || !gpu::internal::range_fits(
                mip_extent.y,
                bind.offset.y,
                bind.extent.y,
            )
            || !gpu::internal::range_fits(
                mip_extent.z,
                bind.offset.z,
                bind.extent.z,
            )) {
            return gpu::INVALID_ARGUMENT~;
        }
        if ((!image_3d && (bind.offset.z != 0 || bind.extent.z != 1))
            || (bind.extent.x % tile.x != 0
                && bind.offset.x + bind.extent.x != mip_extent.x)
            || (bind.extent.y % tile.y != 0
                && bind.offset.y + bind.extent.y != mip_extent.y)
            || (bind.extent.z % tile.z != 0
                && bind.offset.z + bind.extent.z != mip_extent.z)) {
            return gpu::INVALID_ARGUMENT~;
        }
        for (sz prior = 0; prior < i; prior++) {
            if (sparse_tile_boxes_overlap(&binds[prior], bind)) {
                return gpu::INVALID_ARGUMENT~;
            }
        }
    }
}

fn void? validate_sparse_opaque_geometry(
    TextureSlot* texture,
    gpu::SparseTextureOpaqueBind[] binds,
) {
    usz alignment = texture.sparse_requirements.page_allocation.alignment;
    if (alignment == 0) return gpu::INVALID_ARGUMENT~;
    foreach (i, &bind : binds) {
        gpu::SparseTextureAspectRequirements* requirements =
            sparse_aspect_requirements(texture, bind.aspect)!;
        if (bind.size == 0
            || bind.resource_offset % alignment != 0
            || bind.allocation_offset % alignment != 0
            || bind.resource_offset < requirements.mip_tail_offset
            || !gpu::internal::range_fits(
                requirements.mip_tail_size,
                bind.resource_offset - requirements.mip_tail_offset,
                bind.size,
            )) {
            return gpu::INVALID_ARGUMENT~;
        }
        for (sz prior = 0; prior < i; prior++) {
            if (binds[prior].aspect == bind.aspect
                && intervals_overlap(
                    binds[prior].resource_offset,
                    binds[prior].size,
                    bind.resource_offset,
                    bind.size,
                )) {
                return gpu::INVALID_ARGUMENT~;
            }
        }
    }
}

fn ResolvedSparseBacking? resolve_sparse_backing_locked(
    VkDeviceState* state,
    TextureSlot* texture,
    gpu::GpuAllocation allocation,
    usz allocation_offset,
    usz size,
) {
    if (sparse_allocation_is_exact_invalid(allocation)) {
        if (allocation_offset != 0) return gpu::INVALID_ARGUMENT~;
        return {};
    }
    if (!allocation.is_valid()) return gpu::INVALID_HANDLE~;
    AllocationSlot* slot = state.allocations.get(allocation)!;
    gpu::TextureRequirements requirements =
        texture.sparse_requirements.page_allocation;
    if (slot.memory_class != gpu::MemoryClass.TEXTURE
        || !slot.texture_memory
        || slot.dedicated_texture
        || slot.raw_info.device_memory == null
        || slot.raw_info.memory_type >= 32
        || requirements.alignment == 0
        || slot.alignment < requirements.alignment
        || !gpu::internal::queue_roles_are_subset(
            texture.access,
            slot.access,
        )
        || !gpu::internal::range_fits(slot.size, allocation_offset, size)) {
        return gpu::INVALID_ARGUMENT~;
    }
    uint selected_memory_type = 1 << slot.raw_info.memory_type;
    if (texture_compatibility_owner(requirements.compatibility)
            != state.allocations.owner
        || (selected_memory_type
            & texture_compatibility_value(requirements.compatibility)) == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    usz base_offset = (usz)slot.raw_info.offset;
    if ((vk::DeviceSize)base_offset != slot.raw_info.offset
        || allocation_offset > usz::max - base_offset) {
        return gpu::INVALID_ARGUMENT~;
    }
    usz absolute_offset = base_offset + allocation_offset;
    if (absolute_offset % requirements.alignment != 0
        || (vk::DeviceSize)absolute_offset != absolute_offset) {
        return gpu::INVALID_ARGUMENT~;
    }
    return {
        .memory          = slot.raw_info.device_memory,
        .absolute_offset = absolute_offset,
        .size            = size,
    };
}

fn bool sparse_backing_overlaps_placed_locked(
    VkDeviceState* state,
    SparseBackingInterval interval,
) {
    foreach (&cell : state.textures.slots[:state.textures.next_new]) {
        if (!cell.used || !cell.value.placed) continue;
        AllocationSlot*? allocation_result =
            state.allocations.get(cell.value.placement);
        if (catch allocation_result) continue;
        AllocationSlot* allocation = allocation_result;
        if (allocation.raw_info.device_memory != interval.memory) continue;
        usz base = (usz)allocation.raw_info.offset;
        if ((vk::DeviceSize)base != allocation.raw_info.offset
            || cell.value.placement_offset > usz::max - base) {
            continue;
        }
        usz offset = base + cell.value.placement_offset;
        if (intervals_overlap(
            interval.offset,
            interval.size,
            offset,
            cell.value.allocation_size,
        )) return true;
    }
    return false;
}

fn void? add_sparse_backing_interval_locked(
    VkDeviceState* state,
    SparseBackingInterval[] intervals,
    usz count,
    ResolvedSparseBacking resolved,
) {
    if (resolved.memory == null) return;
    SparseBackingInterval interval = {
        .memory = resolved.memory,
        .offset = resolved.absolute_offset,
        .size   = resolved.size,
    };
    foreach (&prior : intervals[:count]) {
        if (prior.memory == interval.memory
            && intervals_overlap(
                prior.offset,
                prior.size,
                interval.offset,
                interval.size,
            )) {
            return gpu::INVALID_ARGUMENT~;
        }
    }
    if (sparse_backing_overlaps_placed_locked(state, interval)) {
        return gpu::INVALID_ARGUMENT~;
    }
    intervals[count] = interval;
}

fn ResolvedSparseBacking? resolve_sparse_bind_backing_locked(
    VkDeviceState* state,
    TextureSlot* texture,
    gpu::SparseTextureBindDesc* desc,
    usz index,
) {
    if (index < desc.tiles.len) {
        gpu::SparseTextureTileBind* bind = &desc.tiles[index];
        gpu::SparseTextureAspectRequirements* requirements =
            sparse_aspect_requirements(texture, bind.aspect)!;
        usz size = sparse_tile_backing_size(
            extent:      bind.extent,
            tile_extent: requirements.tile_extent,
            page_size:   texture.sparse_requirements.page_allocation.size,
        )!;
        return resolve_sparse_backing_locked(
            state:             state,
            texture:           texture,
            allocation:        bind.allocation,
            allocation_offset: bind.allocation_offset,
            size:              size,
        );
    }
    usz opaque_index = index - desc.tiles.len;
    if (opaque_index >= desc.opaque.len) return gpu::INVALID_ARGUMENT~;
    gpu::SparseTextureOpaqueBind* bind = &desc.opaque[opaque_index];
    return resolve_sparse_backing_locked(
        state:             state,
        texture:           texture,
        allocation:        bind.allocation,
        allocation_offset: bind.allocation_offset,
        size:              bind.size,
    );
}

fn void? validate_sparse_backings_locked(
    VkDeviceState* state,
    TextureSlot* texture,
    gpu::SparseTextureBindDesc* desc,
) {
    usz count = desc.tiles.len + desc.opaque.len;
    for (usz index = 0; index < count; index++) {
        ResolvedSparseBacking resolved =
            resolve_sparse_bind_backing_locked(
                state:   state,
                texture: texture,
                desc:    desc,
                index:   index,
            )!;
        if (resolved.memory == null) continue;
        SparseBackingInterval interval = {
            .memory = resolved.memory,
            .offset = resolved.absolute_offset,
            .size   = resolved.size,
        };
        if (sparse_backing_overlaps_placed_locked(state, interval)) {
            return gpu::INVALID_ARGUMENT~;
        }
    }
}

fn void? validate_sparse_wait_publications(
    VkDeviceState* state,
    gpu::CompletionPoint[] waits,
) {
    foreach (point : waits) {
        if (point.device != state.owner || !point.is_valid()) {
            return gpu::INVALID_HANDLE~;
        }
        (void)published_queue_completion_state(
            state,
            gpu::internal::completion_point_queue_id(point),
            gpu::internal::completion_point_sequence(point),
        )!;
    }
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
    if (native.len > gpu::MAX_SPARSE_TEXTURE_ASPECTS) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
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
        && !texture_format_is_depth_stencil(desc.format)
        && !gpu::internal::format_is_block_compressed(desc.format)
        && !desc.usage.color_attach
        && !desc.usage.depth_attach;
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
        .cube_compatible       = desc.cube_compatible,
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

fn SparseBindNativeOps sparse_bind_native_ops_real() {
    return {
        .queue_bind_sparse            = &vk::queue_bind_sparse,
        .get_semaphore_counter_value = &vk::get_semaphore_counter_value,
    };
}

fn gpu::CompletionPoint? vk_bind_sparse_texture_memory_with_ops(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::SparseTextureBindDesc* desc,
    SparseBindNativeOps* ops,
) {
    if (state == null || desc == null || ops == null
        || ops.queue_bind_sparse == null
        || ops.get_semaphore_counter_value == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    usz tile_count = desc.tiles.len;
    usz opaque_count = desc.opaque.len;
    usz wait_count = desc.waits.len;
    if ((tile_count > 0 && desc.tiles.ptr == null)
        || (opaque_count > 0 && desc.opaque.ptr == null)
        || (wait_count > 0 && desc.waits.ptr == null)
        || tile_count > MAX_SPARSE_TILE_BINDS
        || opaque_count > MAX_SPARSE_OPAQUE_BINDS
        || wait_count > MAX_SPARSE_BIND_WAITS
        || tile_count + opaque_count == 0) {
        return gpu::INVALID_ARGUMENT~;
    }

    if (queue.device != state.owner) return gpu::INVALID_HANDLE~;
    QueueCompletionState* target = queue_completion_state(state, queue.id)!;
    if ((uint)target.roles != (uint)queue.roles) {
        return gpu::INVALID_HANDLE~;
    }
    if (((uint)target.roles
        & (uint)state.caps.sparse_textures.binding_queues) == 0) {
        return gpu::INVALID_ARGUMENT~;
    }

    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        TextureSlot* texture = state.textures.get(desc.texture)!;
        if (!texture.sparse) return gpu::INVALID_ARGUMENT~;
        validate_sparse_tile_geometry(texture, desc.tiles)!;
        validate_sparse_opaque_geometry(texture, desc.opaque)!;
        validate_sparse_backings_locked(state, texture, desc)!;
    }
    validate_sparse_wait_publications(state, desc.waits)!;

    vk::Semaphore[] wait_semaphores;
    ulong[] wait_values;
    vk::SparseImageMemoryBind[] tile_binds;
    vk::SparseMemoryBind[] opaque_binds;
    SparseBackingInterval[] backing_intervals;
    usz backing_capacity = tile_count + opaque_count;
    defer if (wait_semaphores.ptr != null) {
        alloc::free(state.host_allocator, wait_semaphores.ptr);
    }
    defer if (wait_values.ptr != null) {
        alloc::free(state.host_allocator, wait_values.ptr);
    }
    defer if (tile_binds.ptr != null) {
        alloc::free(state.host_allocator, tile_binds.ptr);
    }
    defer if (opaque_binds.ptr != null) {
        alloc::free(state.host_allocator, opaque_binds.ptr);
    }
    defer if (backing_intervals.ptr != null) {
        alloc::free(state.host_allocator, backing_intervals.ptr);
    }

    vk::Semaphore[]? allocated_wait_semaphores = alloc::new_array_try(
        state.host_allocator,
        vk::Semaphore,
        (sz)(wait_count + 1),
    );
    if (catch allocated_wait_semaphores) {
        return gpu::OUT_OF_HOST_MEMORY~;
    }
    wait_semaphores = allocated_wait_semaphores;

    ulong[]? allocated_wait_values = alloc::new_array_try(
        state.host_allocator,
        ulong,
        (sz)(wait_count + 1),
    );
    if (catch allocated_wait_values) return gpu::OUT_OF_HOST_MEMORY~;
    wait_values = allocated_wait_values;
    if (tile_count > 0) {
        vk::SparseImageMemoryBind[]? allocated_tile_binds =
            alloc::new_array_try(
                state.host_allocator,
                vk::SparseImageMemoryBind,
                (sz)tile_count,
            );
        if (catch allocated_tile_binds) return gpu::OUT_OF_HOST_MEMORY~;
        tile_binds = allocated_tile_binds;
    }
    if (opaque_count > 0) {
        vk::SparseMemoryBind[]? allocated_opaque_binds =
            alloc::new_array_try(
                state.host_allocator,
                vk::SparseMemoryBind,
                (sz)opaque_count,
            );
        if (catch allocated_opaque_binds) return gpu::OUT_OF_HOST_MEMORY~;
        opaque_binds = allocated_opaque_binds;
    }
    SparseBindReference[]? allocated_reference = alloc::new_array_try(
        state.host_allocator,
        SparseBindReference,
        1,
    );
    if (catch allocated_reference) return gpu::OUT_OF_HOST_MEMORY~;
    SparseBindReference* retained_texture = allocated_reference.ptr;
    bool retained_texture_published;
    defer if (!retained_texture_published) {
        alloc::free(state.host_allocator, retained_texture);
    }

    SparseBackingInterval[]? allocated_intervals = alloc::new_array_try(
        state.host_allocator,
        SparseBackingInterval,
        (sz)backing_capacity,
    );
    if (catch allocated_intervals) return gpu::OUT_OF_HOST_MEMORY~;
    backing_intervals = allocated_intervals;

    bool resource_locked;
    state.resource_mutex.lock()!!;
    resource_locked = true;
    defer if (resource_locked) state.resource_mutex.unlock();
    TextureSlot* texture = state.textures.get(desc.texture)!;
    if (!texture.sparse) return gpu::INVALID_ARGUMENT~;
    validate_sparse_tile_geometry(texture, desc.tiles)!;
    validate_sparse_opaque_geometry(texture, desc.opaque)!;

    usz backing_count;
    foreach (i, &bind : desc.tiles) {
        ResolvedSparseBacking resolved = resolve_sparse_bind_backing_locked(
            state:   state,
            texture: texture,
            desc:    desc,
            index:   (usz)i,
        )!;
        add_sparse_backing_interval_locked(
            state:     state,
            intervals: backing_intervals,
            count:     backing_count,
            resolved:  resolved,
        )!;
        if (resolved.memory != null) backing_count++;
        tile_binds[i] = {
            .subresource = {
                .aspect_mask = (vk::ImageAspectFlags)
                    vk::ImageAspectFlagBits.COLOR_BIT,
                .mip_level   = bind.mip_level,
                .array_layer = 0,
            },
            .offset = {
                .x = (int)bind.offset.x,
                .y = (int)bind.offset.y,
                .z = (int)bind.offset.z,
            },
            .extent = {
                .width  = bind.extent.x,
                .height = bind.extent.y,
                .depth  = bind.extent.z,
            },
            .memory        = resolved.memory,
            .memory_offset = (vk::DeviceSize)resolved.absolute_offset,
            .flags         = (vk::SparseMemoryBindFlags)0,
        };
    }
    foreach (i, &bind : desc.opaque) {
        usz resolved_index = tile_count + (usz)i;
        ResolvedSparseBacking resolved =
            resolve_sparse_bind_backing_locked(
                state:   state,
                texture: texture,
                desc:    desc,
                index:   resolved_index,
            )!;
        add_sparse_backing_interval_locked(
            state:     state,
            intervals: backing_intervals,
            count:     backing_count,
            resolved:  resolved,
        )!;
        if (resolved.memory != null) backing_count++;
        opaque_binds[i] = {
            .resource_offset = (vk::DeviceSize)bind.resource_offset,
            .size            = (vk::DeviceSize)bind.size,
            .memory          = resolved.memory,
            .memory_offset   = (vk::DeviceSize)resolved.absolute_offset,
            .flags = bind.aspect == gpu::SparseTextureAspect.METADATA
                ? (vk::SparseMemoryBindFlags)
                    vk::SparseMemoryBindFlagBits.METADATA_BIT
                : 0,
        };
    }

    target.submission_mutex.lock()!!;
    defer target.submission_mutex.unlock();
    if (((uint)target.roles
        & (uint)state.caps.sparse_textures.binding_queues) == 0) {
        return gpu::INVALID_ARGUMENT~;
    }

    usz native_wait_count = wait_count;
    ulong maximum_same_queue_wait;
    foreach (i, point : desc.waits) {
        if (point.device != state.owner || !point.is_valid()) {
            return gpu::INVALID_HANDLE~;
        }
        ulong sequence = gpu::internal::completion_point_sequence(point);
        QueueCompletionState* source = published_queue_completion_state(
            state,
            gpu::internal::completion_point_queue_id(point),
            sequence,
        )!;
        wait_semaphores[i] = source.timeline;
        wait_values[i] = sequence;
        if (source == target && sequence > maximum_same_queue_wait) {
            maximum_same_queue_wait = sequence;
        }
    }
    ulong prior_target_sequence = target.published_sequence.load(
        AtomicOrdering.ACQUIRE,
    );
    if (prior_target_sequence > maximum_same_queue_wait) {
        wait_semaphores[native_wait_count] = target.timeline;
        wait_values[native_wait_count] = prior_target_sequence;
        native_wait_count++;
    }

    retained_texture.retained_counter =
        &texture.tracked_command_references;
    retained_texture.retained_counter.add(1, AtomicOrdering.RELAXED);
    bool texture_retained = true;
    defer if (texture_retained) {
        retained_texture.retained_counter.sub(
            1,
            AtomicOrdering.RELAXED,
        );
    }
    drain_queue_if_needed_with_query(
        state,
        target,
        ops.get_semaphore_counter_value,
    )!;
    require_queue_completion_headroom_with_query(
        state,
        target,
        ops.get_semaphore_counter_value,
    )!;
    gpu::CompletionPoint point = reserve_queue_completion_locked(
        state,
        queue,
    )!;
    ulong signal_value = gpu::internal::completion_point_sequence(point);
    vk::Semaphore signal_semaphore = target.timeline;

    vk::TimelineSemaphoreSubmitInfo timeline_info =
        vk::timeline_semaphore_submit_info();
    timeline_info.wait_semaphore_value_count = (uint)native_wait_count;
    timeline_info.p_wait_semaphore_values = native_wait_count > 0
        ? wait_values.ptr
        : null;
    timeline_info.signal_semaphore_value_count = 1;
    timeline_info.p_signal_semaphore_values = &signal_value;

    vk::SparseImageMemoryBindInfo image_bind_info = {
        .image      = texture.image,
        .bind_count = (uint)tile_count,
        .p_binds    = tile_count > 0 ? tile_binds.ptr : null,
    };
    vk::SparseImageOpaqueMemoryBindInfo opaque_bind_info = {
        .image      = texture.image,
        .bind_count = (uint)opaque_count,
        .p_binds    = opaque_count > 0 ? opaque_binds.ptr : null,
    };
    vk::BindSparseInfo bind_info = vk::bind_sparse_info();
    bind_info.p_next = &timeline_info;
    bind_info.wait_semaphore_count = (uint)native_wait_count;
    bind_info.p_wait_semaphores = native_wait_count > 0
        ? wait_semaphores.ptr
        : null;
    bind_info.image_bind_count = tile_count > 0 ? 1 : 0;
    bind_info.p_image_binds = tile_count > 0 ? &image_bind_info : null;
    bind_info.image_opaque_bind_count = opaque_count > 0 ? 1 : 0;
    bind_info.p_image_opaque_binds = opaque_count > 0
        ? &opaque_bind_info
        : null;
    bind_info.signal_semaphore_count = 1;
    bind_info.p_signal_semaphores = &signal_semaphore;

    vk::Result result = ops.queue_bind_sparse(
        queue:           target.queue,
        bind_info_count: 1,
        p_bind_info:     &bind_info,
        fence:           (vk::Fence)null,
    );
    if (result != vk::Result.SUCCESS) {
        cancel_queue_completion_locked(state, point)!!;
        check_backend_result(
            state:     state,
            operation: "bind_sparse_texture_memory",
            result:    result,
        )!;
    }
    target.retirement_mutex.lock()!!;
    publish_sparse_bind_reference_locked(
        target,
        retained_texture,
        point,
    );
    target.retirement_mutex.unlock();
    retained_texture_published = true;
    texture_retained = false;
    target.pending_sparse_sequence = signal_value;
    state.resource_mutex.unlock();
    resource_locked = false;
    publish_queue_completion_locked(state, point)!!;
    return point;
}

fn gpu::CompletionPoint? vk_bind_sparse_texture_memory(
    VkDeviceState* state,
    gpu::Queue queue,
    gpu::SparseTextureBindDesc* desc,
) {
    SparseBindNativeOps ops = sparse_bind_native_ops_real();
    return vk_bind_sparse_texture_memory_with_ops(
        state: state,
        queue: queue,
        desc:  desc,
        ops:   &ops,
    );
}
