module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;
import vma;
import std::atomic;

alias TextureCreateAliasingImageFn = fn vk::Result (
    void*, vma::Allocator, vma::Allocation, vk::DeviceSize,
    vk::ImageCreateInfo*, vk::Image*
);
alias TextureCreateViewFn = fn vk::Result (
    void*, vk::Device, vk::ImageViewCreateInfo*, vk::ImageView*
);
alias TextureDestroyImageFn = fn void (
    void*, vk::Device, vk::Image
);
alias TextureDestroyViewFn = fn void (
    void*, vk::Device, vk::ImageView
);
alias TextureCreateImageFn = fn vk::Result (
    void*, vk::Device, vk::ImageCreateInfo*, vk::Image*
);
alias TextureAllocateImageMemoryFn = fn vk::Result (
    void*, vma::Allocator, vk::Image, vma::AllocationCreateInfo*,
    vma::Allocation*, vma::AllocationInfo*
);
alias TextureBindImageMemoryFn = fn vk::Result (
    void*, vma::Allocator, vma::Allocation, vk::Image
);
alias TextureFreeMemoryFn = fn void (
    void*, vma::Allocator, vma::Allocation
);

struct TexturePlacementNativeOps {
    void*                        user_data;
    TextureCreateAliasingImageFn create_aliasing_image;
    TextureCreateViewFn          create_view;
    TextureDestroyImageFn        destroy_image;
    TextureDestroyViewFn         destroy_view;
}

struct DedicatedTextureNativeOps {
    void*                        user_data;
    TextureCreateImageFn         create_image;
    TextureAllocateImageMemoryFn allocate_memory;
    TextureBindImageMemoryFn     bind_memory;
    TextureCreateViewFn          create_view;
    TextureDestroyImageFn        destroy_image;
    TextureDestroyViewFn         destroy_view;
    TextureFreeMemoryFn          free_memory;
}

// Concrete (non-generic) slot table for textures. Monomorphized twin of
// gpu::slots to sidestep the c3c 0.8.0 generic-instantiation codegen crash.
const uint TEXTURE_FREE_NONE = 0;
const sz INITIAL_CACHED_VIEW_CAPACITY = 8;

struct CachedView {
    vk::ImageView     view;
    uint              base_mip;
    uint              mip_count;
    uint              base_layer;
    uint              layer_count;
    vk::ImageViewType view_type;
    bool              used;
}

struct CachedViewCreation {
    CachedView[]* cache;
    CachedView[]  previous_cache;
    sz            index;
    CachedView    previous;
    bool          replaced_cache;
}

struct TextureSlot {
    vk::Image          image;
    vma::Allocation    allocation;
    vk::ImageView      default_view;
    gpu::Format        format;
    gpu::TextureUsage  usage;
    gpu::QueueRoles    access;
    gpu::SampleCount   sample_count;
    uint               width;
    uint               height;
    uint               mip_levels;
    uint               array_layers;
    Atomic{bool}       presented;
    Atomic{uint}        validation_references;
    bool               swapchain_owned;
    bool               placed;
    gpu::GpuAllocation placement;
    usz                placement_offset;
    usz                allocation_size;
    char[DEBUG_NAME_CAP] debug_name;
    CachedView[]        views;
}

struct TextureCell {
    TextureSlot value;
    uint        generation;
    uint        next_free;
    bool        used;
}

struct TextureTable {
    TextureCell[] slots;
    ulong         owner;
    uint          free_head;
    uint          next_new;
}

fn gpu::TextureHandle? TextureTable.alloc(&self, TextureSlot value) {
    uint idx;
    bool found = false;
    while (self.free_head != TEXTURE_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
        if (can_allocate_handle_generation(self.slots[idx].generation)) {
            found = true;
            break;
        }
        self.slots[idx].next_free = TEXTURE_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].used = true;
    return (gpu::TextureHandle){
        .owner      = self.owner,
        .index      = idx,
        .generation = self.slots[idx].generation,
    };
}

fn TextureSlot*? TextureTable.get(&self, gpu::TextureHandle h) {
    if (h.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? TextureTable.free(&self, gpu::TextureHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    if (!should_recycle_handle_generation(self.slots[idx].generation)) {
        self.slots[idx].generation = HANDLE_GENERATION_MAX;
        self.slots[idx].next_free = TEXTURE_FREE_NONE;
        return;
    }
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

fn gpu::TextureCompatibility texture_compatibility(ulong owner, uint value) {
    return (gpu::TextureCompatibility)(((uint128)owner << 64) | (uint128)value);
}

fn ulong texture_compatibility_owner(gpu::TextureCompatibility compatibility) {
    return (ulong)((uint128)compatibility >> 64);
}

fn uint texture_compatibility_value(gpu::TextureCompatibility compatibility) {
    return (uint)(uint128)compatibility;
}

fn vk::Format format_to_vk(gpu::Format format) {
    switch (format) {
        case gpu::Format.R8_UNORM:          return vk::Format.R8_UNORM;
        case gpu::Format.R8_UINT:           return vk::Format.R8_UINT;
        case gpu::Format.RG8_UNORM:         return vk::Format.R8G8_UNORM;
        case gpu::Format.RGBA8_UNORM:       return vk::Format.R8G8B8A8_UNORM;
        case gpu::Format.RGBA8_SRGB:        return vk::Format.R8G8B8A8_SRGB;
        case gpu::Format.BGRA8_UNORM:       return vk::Format.B8G8R8A8_UNORM;
        case gpu::Format.BGRA8_SRGB:        return vk::Format.B8G8R8A8_SRGB;
        case gpu::Format.R16_UINT:          return vk::Format.R16_UINT;
        case gpu::Format.R16_FLOAT:         return vk::Format.R16_SFLOAT;
        case gpu::Format.RG16_FLOAT:        return vk::Format.R16G16_SFLOAT;
        case gpu::Format.RGBA16_FLOAT:      return vk::Format.R16G16B16A16_SFLOAT;
        case gpu::Format.R32_UINT:          return vk::Format.R32_UINT;
        case gpu::Format.R32_FLOAT:         return vk::Format.R32_SFLOAT;
        case gpu::Format.RG32_FLOAT:        return vk::Format.R32G32_SFLOAT;
        case gpu::Format.RGBA32_FLOAT:      return vk::Format.R32G32B32A32_SFLOAT;
        case gpu::Format.D32_FLOAT:         return vk::Format.D32_SFLOAT;
        case gpu::Format.D24_UNORM_S8_UINT: return vk::Format.D24_UNORM_S8_UINT;
        default:                            return vk::Format.UNDEFINED;
    }
}

fn vk::SampleCountFlagBits sample_count_to_vk(
    gpu::SampleCount sample_count,
) @private {
    switch (sample_count) {
        case gpu::SampleCount.ONE:        return vk::SampleCountFlagBits.COUNT_1_BIT;
        case gpu::SampleCount.TWO:        return vk::SampleCountFlagBits.COUNT_2_BIT;
        case gpu::SampleCount.FOUR:       return vk::SampleCountFlagBits.COUNT_4_BIT;
        case gpu::SampleCount.EIGHT:      return vk::SampleCountFlagBits.COUNT_8_BIT;
        case gpu::SampleCount.SIXTEEN:    return vk::SampleCountFlagBits.COUNT_16_BIT;
        case gpu::SampleCount.THIRTY_TWO: return vk::SampleCountFlagBits.COUNT_32_BIT;
        case gpu::SampleCount.SIXTY_FOUR: return vk::SampleCountFlagBits.COUNT_64_BIT;
        default:                          return vk::SampleCountFlagBits.COUNT_1_BIT;
    }
}

fn vk::ImageUsageFlags texture_usage_to_vk(gpu::TextureUsage usage) @private {
    uint flags = 0;
    if (usage.sampled)      flags |= (uint)vk::ImageUsageFlagBits.SAMPLED_BIT;
    if (usage.storage)      flags |= (uint)vk::ImageUsageFlagBits.STORAGE_BIT;
    if (usage.color_attach) flags |= (uint)vk::ImageUsageFlagBits.COLOR_ATTACHMENT_BIT;
    if (usage.depth_attach) flags |= (uint)vk::ImageUsageFlagBits.DEPTH_STENCIL_ATTACHMENT_BIT;
    if (usage.transfer_src) flags |= (uint)vk::ImageUsageFlagBits.TRANSFER_SRC_BIT;
    if (usage.transfer_dst) flags |= (uint)vk::ImageUsageFlagBits.TRANSFER_DST_BIT;
    return (vk::ImageUsageFlags)flags;
}

fn bool? format_supports_sample_count(
    VkDeviceState* state,
    gpu::Format format,
    gpu::TextureUsage usage,
    gpu::SampleCount sample_count,
) @private {
    gpu::TextureDesc desc = {
        .dimension    = gpu::TextureDimension.TEX_2D,
        .width        = 1,
        .height       = 1,
        .format       = format,
        .usage        = usage,
        .access       = state.selected_queue_roles,
        .sample_count = sample_count,
    };
    return texture_desc_supported(state, &desc);
}

fn gpu::TextureFormatSupport? vk_get_texture_format_support(
    gpu::Device* device,
    gpu::Format format,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);

    gpu::TextureFormatSupport support;
    if (format_to_vk(format) == vk::Format.UNDEFINED) return support;

    support.features.sampled = format_supports_usage(state, format, { .sampled })!;
    support.features.storage = format_supports_usage(state, format, { .storage })!;
    support.features.color_attach = format_supports_usage(state, format, { .color_attach })!;
    support.features.depth_attach = format_supports_usage(state, format, { .depth_attach })!;
    support.features.transfer_src = format_supports_usage(state, format, { .transfer_src })!;
    support.features.transfer_dst = format_supports_usage(state, format, { .transfer_dst })!;

    if (support.features.sampled) {
        vk::FormatProperties properties;
        vk::get_physical_device_format_properties(
            state.physical_device,
            format_to_vk(format),
            &properties,
        );
        uint features = (uint)properties.optimal_tiling_features;
        support.features.linear_filter =
            (features & (uint)vk::FormatFeatureFlagBits.SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
    }

    bool creatable = support.features.sampled || support.features.storage
        || support.features.color_attach || support.features.depth_attach
        || support.features.transfer_src || support.features.transfer_dst;
    support.dimensions.tex_2d = creatable;
    support.sample_counts.one = creatable;

    gpu::TextureUsage attachment_usage;
    bool attachment_supported;
    if (format == gpu::Format.D32_FLOAT) {
        attachment_usage = { .depth_attach };
        attachment_supported = support.features.depth_attach;
    } else {
        attachment_usage = { .color_attach };
        attachment_supported = support.features.color_attach;
    }
    if (attachment_supported) {
        support.sample_counts.two = format_supports_sample_count(
            state,
            format,
            attachment_usage,
            gpu::SampleCount.TWO,
        )!;
        support.sample_counts.four = format_supports_sample_count(
            state,
            format,
            attachment_usage,
            gpu::SampleCount.FOUR,
        )!;
        support.sample_counts.eight = format_supports_sample_count(
            state,
            format,
            attachment_usage,
            gpu::SampleCount.EIGHT,
        )!;
        support.sample_counts.sixteen = format_supports_sample_count(
            state,
            format,
            attachment_usage,
            gpu::SampleCount.SIXTEEN,
        )!;
        support.sample_counts.thirty_two =
            format_supports_sample_count(
                state,
                format,
                attachment_usage,
                gpu::SampleCount.THIRTY_TWO,
            )!;
        support.sample_counts.sixty_four =
            format_supports_sample_count(
                state,
                format,
                attachment_usage,
                gpu::SampleCount.SIXTY_FOUR,
            )!;
    }
    return support;
}

fn bool? texture_desc_supported(VkDeviceState* state, gpu::TextureDesc* desc) @private {
    vk::ImageUsageFlags usage = texture_usage_to_vk(desc.usage);
    vk::ImageFormatProperties properties;
    vk::Result result = vk::get_physical_device_image_format_properties(
        state.physical_device,
        format_to_vk(desc.format),
        vk::ImageType.TYPE_2D,
        vk::ImageTiling.OPTIMAL,
        usage,
        (vk::ImageCreateFlags)0,
        &properties,
    );
    switch (result) {
        case vk::Result.ERROR_FORMAT_NOT_SUPPORTED:
            return false;
        case vk::Result.SUCCESS:
            break;
        default:
            check_device_result(state, result)!;
    }

    uint mip_levels = desc.mip_levels == 0 ? 1 : desc.mip_levels;
    uint array_layers = desc.array_layers == 0 ? 1 : desc.array_layers;
    uint depth = desc.depth == 0 ? 1 : desc.depth;
    bool supports_sample_count = ((uint)properties.sample_counts
        & (uint)sample_count_to_vk(desc.sample_count)) != 0;
    return desc.width <= properties.max_extent.width
        && desc.height <= properties.max_extent.height
        && depth <= properties.max_extent.depth
        && mip_levels <= properties.max_mip_levels
        && array_layers <= properties.max_array_layers
        && supports_sample_count;
}

fn bool? format_supports_usage(
    VkDeviceState* state,
    gpu::Format format,
    gpu::TextureUsage usage,
) @private {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 1,
        .height    = 1,
        .format    = format,
        .usage     = usage,
        .access    = state.selected_queue_roles,
    };
    if (!gpu::texture_desc_valid(&desc)) return false;
    return texture_desc_supported(state, &desc);
}

fn bool? vk_supports_texture_desc(
    gpu::Device* device,
    gpu::TextureDesc* desc,
) @private {
    if (!gpu::texture_desc_valid(desc)) return false;
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    return texture_desc_supported(state, desc);
}

fn usz format_texel_size(gpu::Format format) {
    switch (format) {
        case gpu::Format.R8_UNORM:          return 1;
        case gpu::Format.R8_UINT:           return 1;
        case gpu::Format.RG8_UNORM:         return 2;
        case gpu::Format.RGBA8_UNORM:       return 4;
        case gpu::Format.RGBA8_SRGB:        return 4;
        case gpu::Format.BGRA8_UNORM:       return 4;
        case gpu::Format.BGRA8_SRGB:        return 4;
        case gpu::Format.R16_UINT:          return 2;
        case gpu::Format.R16_FLOAT:         return 2;
        case gpu::Format.RG16_FLOAT:        return 4;
        case gpu::Format.RGBA16_FLOAT:      return 8;
        case gpu::Format.R32_UINT:          return 4;
        case gpu::Format.R32_FLOAT:         return 4;
        case gpu::Format.RG32_FLOAT:        return 8;
        case gpu::Format.RGBA32_FLOAT:      return 16;
        case gpu::Format.D32_FLOAT:         return 4;
        case gpu::Format.D24_UNORM_S8_UINT: return 4;
        default:                            return 0;
    }
}

struct MipExtent {
    uint width;
    uint height;
}

fn MipExtent texture_mip_extent(TextureSlot* slot, uint mip) {
    uint width  = slot.width >> mip;
    uint height = slot.height >> mip;
    return {
        .width  = width == 0 ? 1 : width,
        .height = height == 0 ? 1 : height,
    };
}

fn vk::ImageAspectFlags format_aspect(gpu::Format format) @private {
    switch (format) {
        case gpu::Format.D32_FLOAT:
            return (vk::ImageAspectFlags)vk::ImageAspectFlagBits.DEPTH_BIT;
        case gpu::Format.D24_UNORM_S8_UINT:
            return (vk::ImageAspectFlags)((uint)vk::ImageAspectFlagBits.DEPTH_BIT | (uint)vk::ImageAspectFlagBits.STENCIL_BIT);
        default:
            return (vk::ImageAspectFlags)vk::ImageAspectFlagBits.COLOR_BIT;
    }
}

fn gpu::TextureUse swapchain_prior_use(TextureSlot* slot) {
    return slot.presented.load(AtomicOrdering.RELAXED)
        ? gpu::TextureUse.PRESENT
        : gpu::TextureUse.UNDEFINED;
}

fn void mark_texture_presented(TextureSlot* slot) {
    slot.presented.store(true, AtomicOrdering.RELAXED);
}

fn void? check_texture_view_backend_result(
    VkDeviceState* state,
    ZString operation,
    gpu::TextureHandle texture,
    TextureSlot* slot,
    vk::Result result,
) {
    observe_device_result(state, result);
    if (operation == null) return check(result);
    bool stderr_fallback = state.debug_callback == null;
    fault public_fault = @catch(map_result(result, stderr_fallback));
    if (!public_fault) return;
    if (!stderr_fallback) {
        gpu::DebugMessage message = {
            .severity             = gpu::DebugMessageSeverity.ERROR,
            .category             = { .backend },
            .operation            = operation,
            .has_fault            = true,
            .public_fault         = public_fault,
            .resource             = {
                .kind                = gpu::DebugResourceKind.TEXTURE,
                .has_public_identity = texture.is_valid(),
                .index               = texture.index,
                .generation          = texture.generation,
                .debug_name          = slot_name(&slot.debug_name[0]),
            },
            .backend_text         = backend_result_text(result),
            .validation_id_number = (int)result,
        };
        gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
    }
    return public_fault~;
}

<* Resolve a view desc to an image view: null/zero desc (or the explicit full
   range) is the default view; anything else is normalized, validated, and
   get-or-created in the slot's growable view cache. Cached views die with the
   texture. Format reinterpretation is out of scope. *>
fn vk::ImageView? resolve_texture_view_tracked(
    VkDeviceState* state,
    TextureSlot* slot,
    gpu::TextureHandle texture,
    gpu::TextureViewDesc* view,
    ZString operation,
    CachedViewCreation* creation,
) @private {
    *creation = {};
    if (view == null) return slot.default_view;
    if (view.format != gpu::Format.UNDEFINED && view.format != slot.format) return gpu::INVALID_ARGUMENT~;
    if (view.base_mip >= slot.mip_levels || view.base_layer >= slot.array_layers) return gpu::INVALID_ARGUMENT~;

    uint mip_count   = view.mip_count == 0 ? slot.mip_levels - view.base_mip : view.mip_count;
    uint layer_count = view.layer_count == 0 ? slot.array_layers - view.base_layer : view.layer_count;
    if (mip_count > slot.mip_levels - view.base_mip)       return gpu::INVALID_ARGUMENT~;
    if (layer_count > slot.array_layers - view.base_layer) return gpu::INVALID_ARGUMENT~;

    vk::ImageViewType view_type = layer_count > 1 ? vk::ImageViewType.TYPE_2D_ARRAY : vk::ImageViewType.TYPE_2D;
    bool full_range = view.base_mip == 0 && mip_count == slot.mip_levels
        && view.base_layer == 0 && layer_count == slot.array_layers;
    if (full_range) return slot.default_view;

    foreach (&cached : slot.views) {
        if (cached.used
            && cached.base_mip == view.base_mip && cached.mip_count == mip_count
            && cached.base_layer == view.base_layer && cached.layer_count == layer_count
            && cached.view_type == view_type) {
            return cached.view;
        }
    }

    sz cache_index = slot.views.len;
    foreach (i, &cached : slot.views) {
        if (!cached.used) {
            cache_index = i;
            break;
        }
    }

    vk::ImageViewCreateInfo view_info = vk::image_view_create_info()
        .set_image(slot.image)
        .set_view_type(view_type)
        .set_format(format_to_vk(slot.format))
        .set_subresource_range({
            .aspect_mask      = format_aspect(slot.format),
            .base_mip_level   = view.base_mip,
            .level_count      = mip_count,
            .base_array_layer = view.base_layer,
            .layer_count      = layer_count,
        });
    vk::ImageView image_view;
    check_texture_view_backend_result(
        state:     state,
        operation: operation,
        texture:   texture,
        slot:      slot,
        result:    vk::create_image_view(state.device, &view_info, null, &image_view),
    )!;
    if (cache_index == slot.views.len) {
        sz new_capacity = slot.views.len == 0
            ? INITIAL_CACHED_VIEW_CAPACITY
            : slot.views.len * 2;
        CachedView[]? grown_result = alloc::new_array_try(
            state.host_allocator,
            CachedView,
            new_capacity,
        );
        if (catch allocation_fault = grown_result) {
            vk::destroy_image_view(state.device, image_view, null);
            return gpu::OUT_OF_HOST_MEMORY~;
        }
        CachedView[] grown = grown_result;
        foreach (i, &cached : slot.views) grown[i] = *cached;
        creation.previous_cache = slot.views;
        creation.replaced_cache = true;
        slot.views = grown;
    }
    CachedView* cached = &slot.views[cache_index];
    creation.cache = &slot.views;
    creation.index = cache_index;
    creation.previous = *cached;
    *cached = {
        .view        = image_view,
        .base_mip    = view.base_mip,
        .mip_count   = mip_count,
        .base_layer  = view.base_layer,
        .layer_count = layer_count,
        .view_type   = view_type,
        .used        = true,
    };
    return image_view;
}

fn vk::ImageView? resolve_texture_view(VkDeviceState* state, TextureSlot* slot, gpu::TextureViewDesc* view) {
    CachedViewCreation creation;
    return resolve_texture_view_tracked(
        state:     state,
        slot:      slot,
        texture:   {},
        view:      view,
        operation: null,
        creation:  &creation,
    );
}

fn void commit_texture_view_creations(VkDeviceState* state, CachedViewCreation[] creations) @private {
    foreach (&creation : creations) {
        if (creation.replaced_cache && creation.previous_cache.ptr != null) {
            alloc::free(state.host_allocator, creation.previous_cache.ptr);
        }
    }
}

fn void rollback_texture_view_creations(VkDeviceState* state, CachedViewCreation[] creations) @private {
    sz remaining = creations.len;
    while (remaining > 0) {
        CachedViewCreation* creation = &creations[--remaining];
        CachedView* cached = &(*creation.cache)[creation.index];
        vk::destroy_image_view(state.device, cached.view, null);
        *cached = creation.previous;
        if (!creation.replaced_cache) continue;
        alloc::free(state.host_allocator, (*creation.cache).ptr);
        *creation.cache = creation.previous_cache;
    }
}

fn void destroy_cached_views(VkDeviceState* state, TextureSlot* slot) {
    foreach (&cached : slot.views) {
        if (cached.used) {
            vk::destroy_image_view(state.device, cached.view, null);
            cached.used = false;
        }
    }
    if (slot.views.ptr != null) {
        alloc::free(state.host_allocator, slot.views.ptr);
        slot.views = {};
    }
}


fn vk::ImageCreateInfo make_texture_create_info(
    gpu::TextureDesc* desc,
    uint[] queue_families,
    bool aliasing = false,
) @private {
    vk::ImageCreateInfo info = vk::image_create_info()
        .set_image_type(vk::ImageType.TYPE_2D)
        .set_format(format_to_vk(desc.format))
        .set_extent({ .width = desc.width, .height = desc.height, .depth = 1 })
        .set_mip_levels(gpu::texture_mip_levels(desc))
        .set_array_layers(gpu::texture_array_layers(desc))
        .set_samples(sample_count_to_vk(desc.sample_count))
        .set_tiling(vk::ImageTiling.OPTIMAL)
        .set_usage(texture_usage_to_vk(desc.usage))
        .set_initial_layout(vk::ImageLayout.UNDEFINED);
    if (aliasing) {
        info.flags = (vk::ImageCreateFlags)vk::ImageCreateFlagBits.ALIAS_BIT;
    }
    if (queue_families.len >= 2) {
        info = info
            .set_sharing_mode(vk::SharingMode.CONCURRENT)
            .set_queue_family_indices(queue_families);
    }
    return info;
}

fn vk::ImageViewCreateInfo default_texture_view_create_info(
    gpu::TextureDesc* desc,
    vk::Image image,
    uint mip_levels,
    uint array_layers,
) {
    return vk::image_view_create_info()
        .set_image(image)
        .set_view_type(array_layers > 1
            ? vk::ImageViewType.TYPE_2D_ARRAY
            : vk::ImageViewType.TYPE_2D)
        .set_format(format_to_vk(desc.format))
        .set_subresource_range({
            .aspect_mask      = format_aspect(desc.format),
            .base_mip_level   = 0,
            .level_count      = mip_levels,
            .base_array_layer = 0,
            .layer_count      = array_layers,
        });
}

fn gpu::TextureRequirements? texture_requirements_for_state(
    VkDeviceState* state,
    gpu::TextureDesc* desc,
) {
    if (!gpu::texture_desc_valid(desc)) return gpu::INVALID_ARGUMENT~;
    if (!texture_desc_supported(state, desc)!) return gpu::UNSUPPORTED_FEATURE~;

    uint[MAX_QUEUE_FAMILIES] families;
    uint family_count = resource_queue_families(
        state,
        desc.access,
        families[..],
    );
    vk::ImageCreateInfo image_info = make_texture_create_info(
        desc,
        families[:family_count],
        true,
    );
    vk::DeviceImageMemoryRequirements query =
        vk::device_image_memory_requirements().set_create_info(&image_info);
    vk::MemoryDedicatedRequirements dedicated =
        vk::memory_dedicated_requirements();
    vk::MemoryRequirements2 result =
        vk::memory_requirements2().set_next(&dedicated);
    vk::get_device_image_memory_requirements(
        state.device,
        &query,
        &result,
    );

    vk::MemoryRequirements* requirements = &result.memory_requirements;
    usz size = (usz)requirements.size;
    usz alignment = (usz)requirements.alignment;
    if (requirements.size == 0
        || (vk::DeviceSize)size != requirements.size
        || requirements.alignment == 0
        || (vk::DeviceSize)alignment != requirements.alignment
        || requirements.memory_type_bits == 0) {
        return gpu::BACKEND_ERROR~;
    }
    return {
        .size      = size,
        .alignment = alignment,
        .compatibility = texture_compatibility(
            state.allocations.owner,
            requirements.memory_type_bits,
        ),
        .dedicated_only = dedicated.requires_dedicated_allocation != 0,
    };
}

fn gpu::TextureRequirements? vk_get_texture_requirements(
    gpu::Device* device,
    gpu::TextureDesc* desc,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    return texture_requirements_for_state(state, desc);
}

fn gpu::TextureHandle? vk_create_texture(gpu::Device* device, gpu::TextureDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    if (!gpu::texture_desc_valid(desc)) {
        return report_contract_failure(
            state:               state,
            operation:           "create_texture",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "TextureDesc",
            invariant:           "descriptor must satisfy the texture creation contract",
            backend_text:        "texture descriptor validation failed",
        )~;
    }
    if (!texture_desc_supported(state, desc)!) {
        return report_contract_failure(
            state:               state,
            operation:           "create_texture",
            public_fault:        gpu::UNSUPPORTED_FEATURE,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "TextureDesc",
            invariant:           "descriptor must be supported by selected device capabilities",
            backend_text:        "selected device does not support the texture descriptor",
        )~;
    }
    uint mip_levels   = gpu::texture_mip_levels(desc);
    uint array_layers = gpu::texture_array_layers(desc);
    uint[MAX_QUEUE_FAMILIES] share_families;
    uint share_count = resource_queue_families(state, desc.access, share_families[..]);
    vk::ImageCreateInfo image_info =
        make_texture_create_info(desc, share_families[:share_count]);
    vma::AllocationCreateInfo alloc_info = { .usage = vma::MemoryUsage.AUTO_PREFER_DEVICE };

    TextureSlot slot = {
        .format         = desc.format,
        .usage          = desc.usage,
        .access         = desc.access,
        .sample_count   = desc.sample_count,
        .width          = desc.width,
        .height         = desc.height,
        .mip_levels     = mip_levels,
        .array_layers   = array_layers,
    };
    vma::AllocationInfo allocation_info;
    check_texture_create_device_result(
        state: state,
        result: state.allocator.create_image(
            image_info:     &image_info,
            alloc_info:     &alloc_info,
            out_image:      &slot.image,
            out_allocation: &slot.allocation,
            out_info:       &allocation_info,
        ),
    )!;
    slot.allocation_size = (usz)allocation_info.size;

    copy_debug_name(&slot.debug_name[0], desc.debug_name);
    if (desc.debug_name != null) {
        state.allocator.set_allocation_name(slot.allocation, desc.debug_name);
        set_object_name(
            state:         state,
            object_type:   vk::ObjectType.IMAGE,
            object_handle: (ulong)slot.image,
            name:          desc.debug_name,
        );
    }

    vk::ImageViewCreateInfo view_info = default_texture_view_create_info(
        desc:         desc,
        image:        slot.image,
        mip_levels:   mip_levels,
        array_layers: array_layers,
    );
    if (catch err = check_device_result(state, vk::create_image_view(
        device:        state.device,
        p_create_info: &view_info,
        p_allocator:   null,
        p_view:        &slot.default_view,
    ))) {
        state.allocator.destroy_image(slot.image, slot.allocation);
        return err~;
    }

    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        if (try handle = state.textures.alloc(slot)) {
            return handle;
        }
    }
    vk::destroy_image_view(state.device, slot.default_view, null);
    state.allocator.destroy_image(slot.image, slot.allocation);
    return gpu::SLOT_TABLE_FULL~;
}

fn vk::Result texture_create_aliasing_image_real(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
    vk::DeviceSize offset,
    vk::ImageCreateInfo* image_info,
    vk::Image* out_image,
) {
    (void)user_data;
    return allocator.create_aliasing_image2(
        allocation,
        offset,
        image_info,
        out_image,
    );
}

fn vk::Result texture_create_view_real(
    void* user_data,
    vk::Device device,
    vk::ImageViewCreateInfo* view_info,
    vk::ImageView* out_view,
) {
    (void)user_data;
    return vk::create_image_view(device, view_info, null, out_view);
}

fn void texture_destroy_image_real(
    void* user_data,
    vk::Device device,
    vk::Image image,
) {
    (void)user_data;
    vk::destroy_image(device, image, null);
}

fn void texture_destroy_view_real(
    void* user_data,
    vk::Device device,
    vk::ImageView view,
) {
    (void)user_data;
    vk::destroy_image_view(device, view, null);
}

fn TexturePlacementNativeOps texture_placement_native_ops_real() {
    return {
        .create_aliasing_image = &texture_create_aliasing_image_real,
        .create_view           = &texture_create_view_real,
        .destroy_image         = &texture_destroy_image_real,
        .destroy_view          = &texture_destroy_view_real,
    };
}

fn vk::Result texture_create_image_real(
    void* user_data,
    vk::Device device,
    vk::ImageCreateInfo* image_info,
    vk::Image* out_image,
) {
    (void)user_data;
    return vk::create_image(device, image_info, null, out_image);
}

fn vk::Result texture_allocate_image_memory_real(
    void* user_data,
    vma::Allocator allocator,
    vk::Image image,
    vma::AllocationCreateInfo* allocation_info,
    vma::Allocation* out_allocation,
    vma::AllocationInfo* out_info,
) {
    (void)user_data;
    return allocator.allocate_memory_for_image(
        image,
        allocation_info,
        out_allocation,
        out_info,
    );
}

fn vk::Result texture_bind_image_memory_real(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
    vk::Image image,
) {
    (void)user_data;
    return allocator.bind_image_memory(allocation, image);
}

fn void texture_free_memory_real(
    void* user_data,
    vma::Allocator allocator,
    vma::Allocation allocation,
) {
    (void)user_data;
    allocator.free_memory(allocation);
}

fn DedicatedTextureNativeOps dedicated_texture_native_ops_real() {
    return {
        .create_image    = &texture_create_image_real,
        .allocate_memory = &texture_allocate_image_memory_real,
        .bind_memory     = &texture_bind_image_memory_real,
        .create_view     = &texture_create_view_real,
        .destroy_image   = &texture_destroy_image_real,
        .destroy_view    = &texture_destroy_view_real,
        .free_memory     = &texture_free_memory_real,
    };
}

fn bool placed_texture_overlaps(
    VkDeviceState* state,
    gpu::GpuAllocation placement,
    usz offset,
    usz size,
) {
    foreach (&cell : state.textures.slots[:state.textures.next_new]) {
        if (!cell.used || !cell.value.placed
            || cell.value.placement != placement) {
            continue;
        }
        if (offset < cell.value.placement_offset + cell.value.allocation_size
            && cell.value.placement_offset < offset + size) {
            return true;
        }
    }
    return false;
}

fn gpu::TextureHandle? create_placed_texture_for_state_with_ops(
    VkDeviceState* state,
    gpu::TextureDesc* desc,
    gpu::GpuAllocation placement,
    usz offset,
    gpu::TextureRequirements requirements,
    TexturePlacementNativeOps* ops,
) {
    if (requirements.size == 0
        || requirements.alignment == 0
        || (requirements.alignment & (requirements.alignment - 1)) != 0
        || !requirements.compatibility.is_valid()
        || texture_compatibility_owner(requirements.compatibility)
            != state.allocations.owner) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (requirements.dedicated_only) return gpu::UNSUPPORTED_FEATURE~;
    if (offset % requirements.alignment != 0) return gpu::INVALID_ARGUMENT~;

    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    AllocationSlot* allocation = state.allocations.get(placement)!;
    if (allocation.live_placements == uint::max) {
        return gpu::INVALID_RESOURCE_STATE~;
    }
    if (!allocation.texture_memory
        || requirements.alignment > allocation.alignment
        || allocation.dedicated_texture
        || !gpu::queue_roles_are_subset(desc.access, allocation.access)
        || allocation.raw_info.memory_type >= 32) {
        return gpu::INVALID_ARGUMENT~;
    }
    uint selected_memory_type = 1 << allocation.raw_info.memory_type;
    if ((selected_memory_type
            & texture_compatibility_value(requirements.compatibility)) == 0
        || !gpu::range_fits(allocation.size, offset, requirements.size)
        || placed_texture_overlaps(
            state,
            placement,
            offset,
            requirements.size,
        )) {
        return gpu::INVALID_ARGUMENT~;
    }

    uint[MAX_QUEUE_FAMILIES] families;
    uint family_count = resource_queue_families(
        state,
        desc.access,
        families[..],
    );
    vk::ImageCreateInfo image_info = make_texture_create_info(
        desc,
        families[:family_count],
        true,
    );
    uint mip_levels = gpu::texture_mip_levels(desc);
    uint array_layers = gpu::texture_array_layers(desc);
    TextureSlot slot = {
        .format           = desc.format,
        .usage            = desc.usage,
        .access           = desc.access,
        .sample_count     = desc.sample_count,
        .width            = desc.width,
        .height           = desc.height,
        .mip_levels       = mip_levels,
        .array_layers     = array_layers,
        .placed           = true,
        .placement        = placement,
        .placement_offset = offset,
        .allocation_size  = requirements.size,
    };

    bool retained_placement;
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
            ops.destroy_image(ops.user_data, state.device, slot.image);
        }
        if (retained_placement) allocation.live_placements--;
    }

    allocation.live_placements++;
    retained_placement = true;
    check_texture_create_device_result(
        state: state,
        result: ops.create_aliasing_image(
            ops.user_data,
            state.allocator,
            allocation.raw_allocation,
            (vk::DeviceSize)offset,
            &image_info,
            &slot.image,
        ),
    )!;
    owns_image = true;

    copy_debug_name(&slot.debug_name[0], desc.debug_name);
    if (desc.debug_name != null) {
        set_object_name(
            state:         state,
            object_type:   vk::ObjectType.IMAGE,
            object_handle: (ulong)slot.image,
            name:          desc.debug_name,
        );
    }

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

    gpu::TextureHandle handle = state.textures.alloc(slot)!;
    owns_image = false;
    owns_view = false;
    retained_placement = false;
    return handle;
}

fn gpu::DedicatedTexture? create_dedicated_texture_for_state_with_ops(
    VkDeviceState* state,
    gpu::TextureDesc* desc,
    gpu::AllocationDesc* allocation_desc,
    gpu::TextureRequirements requirements,
    DedicatedTextureNativeOps* ops,
) {
    if (desc == null || allocation_desc == null
        || allocation_desc.memory_class != gpu::MemoryClass.TEXTURE
        || allocation_desc.texture_requirements.len == 0
        || allocation_desc.size != requirements.size
        || allocation_desc.alignment < requirements.alignment
        || !gpu::queue_roles_are_subset(desc.access, allocation_desc.access)
        || requirements.size == 0
        || requirements.alignment == 0
        || (requirements.alignment & (requirements.alignment - 1)) != 0
        || texture_compatibility_owner(requirements.compatibility)
            != state.allocations.owner) {
        return gpu::INVALID_ARGUMENT~;
    }

    uint memory_type_bits = texture_compatibility_value(
        requirements.compatibility,
    );
    foreach (&candidate : allocation_desc.texture_requirements) {
        if (candidate.size == 0
            || candidate.alignment == 0
            || texture_compatibility_owner(candidate.compatibility)
                != state.allocations.owner) {
            return gpu::INVALID_ARGUMENT~;
        }
        memory_type_bits &= texture_compatibility_value(
            candidate.compatibility,
        );
    }
    if (memory_type_bits == 0) return gpu::INVALID_ARGUMENT~;

    uint[MAX_QUEUE_FAMILIES] families;
    uint family_count = resource_queue_families(
        state,
        desc.access,
        families[..],
    );
    vk::ImageCreateInfo image_info = make_texture_create_info(
        desc,
        families[:family_count],
        true,
    );
    vma::AllocationCreateInfo native_allocation_desc = {
        .flags            = { .dedicated_memory },
        .usage            = vma::MemoryUsage.GPU_ONLY,
        .memory_type_bits = memory_type_bits,
    };
    uint mip_levels = gpu::texture_mip_levels(desc);
    uint array_layers = gpu::texture_array_layers(desc);
    TextureSlot texture_slot = {
        .format           = desc.format,
        .usage            = desc.usage,
        .access           = desc.access,
        .sample_count     = desc.sample_count,
        .width            = desc.width,
        .height           = desc.height,
        .mip_levels       = mip_levels,
        .array_layers     = array_layers,
        .placed           = true,
        .placement_offset = 0,
        .allocation_size  = requirements.size,
    };
    vma::AllocationInfo native_info;
    vma::Allocation native_allocation;
    bool owns_image;
    bool owns_allocation;
    bool owns_view;
    defer catch {
        if (owns_view) {
            ops.destroy_view(
                ops.user_data,
                state.device,
                texture_slot.default_view,
            );
        }
        if (owns_image) {
            ops.destroy_image(
                ops.user_data,
                state.device,
                texture_slot.image,
            );
        }
        if (owns_allocation) {
            ops.free_memory(
                ops.user_data,
                state.allocator,
                native_allocation,
            );
        }
    }

    check_texture_create_device_result(
        state: state,
        result: ops.create_image(
            ops.user_data,
            state.device,
            &image_info,
            &texture_slot.image,
        ),
    )!;
    owns_image = true;
    check_device_result(state, ops.allocate_memory(
        ops.user_data,
        state.allocator,
        texture_slot.image,
        &native_allocation_desc,
        &native_allocation,
        &native_info,
    ))!;
    owns_allocation = true;
    if (native_info.size < (vk::DeviceSize)requirements.size
        || native_info.memory_type >= 32
        || native_info.offset
            % (vk::DeviceSize)allocation_desc.alignment != 0) {
        return gpu::BACKEND_ERROR~;
    }
    uint selected_memory_type = 1 << native_info.memory_type;
    if ((selected_memory_type & memory_type_bits) == 0) {
        return gpu::BACKEND_ERROR~;
    }
    check_device_result(state, ops.bind_memory(
        ops.user_data,
        state.allocator,
        native_allocation,
        texture_slot.image,
    ))!;

    vk::ImageViewCreateInfo view_info = default_texture_view_create_info(
        desc:         desc,
        image:        texture_slot.image,
        mip_levels:   mip_levels,
        array_layers: array_layers,
    );
    check_device_result(state, ops.create_view(
        ops.user_data,
        state.device,
        &view_info,
        &texture_slot.default_view,
    ))!;
    owns_view = true;

    char[DEBUG_NAME_CAP] debug_name;
    copy_debug_name(&debug_name[0], allocation_desc.debug_name);
    AllocationSlot allocation_slot = {
        .raw_allocation    = native_allocation,
        .raw_info          = native_info,
        .size              = requirements.size,
        .access            = allocation_desc.access,
        .memory_type_bits  = memory_type_bits,
        .texture_memory    = true,
        .dedicated_texture = true,
        .alignment         = allocation_desc.alignment,
        .memory_class      = gpu::MemoryClass.TEXTURE,
        .debug_name        = debug_name,
    };
    copy_debug_name(&texture_slot.debug_name[0], desc.debug_name);
    if (allocation_desc.debug_name != null) {
        state.allocator.set_allocation_name(
            native_allocation,
            allocation_desc.debug_name,
        );
    }
    if (desc.debug_name != null) {
        set_object_name(
            state:         state,
            object_type:   vk::ObjectType.IMAGE,
            object_handle: (ulong)texture_slot.image,
            name:          desc.debug_name,
        );
    }

    gpu::GpuAllocation allocation;
    gpu::TextureHandle texture;
    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        allocation = state.allocations.alloc(allocation_slot)!;
        texture_slot.placement = allocation;
        gpu::TextureHandle? published =
            state.textures.alloc(texture_slot);
        if (catch err = published) {
            state.allocations.free(allocation)!!;
            return err~;
        } else {
            texture = published;
        }
        state.allocations.get(allocation)!!.live_placements = 1;
    }
    owns_view = false;
    owns_image = false;
    owns_allocation = false;
    return { .texture = texture, .allocation = allocation };
}

fn gpu::TextureHandle? vk_create_placed_texture(
    gpu::Device* device,
    gpu::TextureDesc* desc,
    gpu::GpuAllocation placement,
    usz offset,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    gpu::TextureRequirements requirements =
        texture_requirements_for_state(state, desc)!;
    TexturePlacementNativeOps ops = texture_placement_native_ops_real();
    return create_placed_texture_for_state_with_ops(
        state:        state,
        desc:         desc,
        placement:    placement,
        offset:       offset,
        requirements: requirements,
        ops:          &ops,
    );
}

fn gpu::DedicatedTexture? vk_create_dedicated_texture(
    gpu::Device* device,
    gpu::TextureDesc* desc,
    gpu::AllocationDesc* allocation_desc,
) @private {
    VkDeviceState* state =
        (VkDeviceState*)gpu::device_backend_state_ptr(device);
    gpu::TextureRequirements requirements =
        texture_requirements_for_state(state, desc)!;
    DedicatedTextureNativeOps ops = dedicated_texture_native_ops_real();
    return create_dedicated_texture_for_state_with_ops(
        state:           state,
        desc:            desc,
        allocation_desc: allocation_desc,
        requirements:    requirements,
        ops:             &ops,
    );
}

fn void? vk_destroy_texture(gpu::Device* device, gpu::TextureHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    TextureSlot* slot = state.textures.get(handle)!;
    if (slot.swapchain_owned) return gpu::INVALID_ARGUMENT~;
    DescriptorHeapState* heap = &state.strict_heap.descriptors;
    foreach (&cell : heap.texture_cells[:heap.texture_next_new]) {
        if (cell.used && cell.owner == handle) return gpu::RESOURCE_IN_USE~;
    }
    vk::Image image = slot.image;
    if (slot.validation_references.load(AtomicOrdering.RELAXED) > 0) {
        drain_completed_submitted_commands(state)!;
        if (slot.validation_references.load(AtomicOrdering.RELAXED) > 0) {
            return gpu::RESOURCE_IN_USE~;
        }
    }
    vma::Allocation allocation = slot.allocation;
    vk::ImageView default_view = slot.default_view;
    bool placed = slot.placed;
    gpu::GpuAllocation placement = slot.placement;
    destroy_cached_views(state, slot);
    state.textures.free(handle)!;
    vk::destroy_image_view(state.device, default_view, null);
    if (placed) {
        vk::destroy_image(state.device, image, null);
        state.allocations.get(placement)!!.live_placements--;
    } else {
        state.allocator.destroy_image(image, allocation);
    }
}
