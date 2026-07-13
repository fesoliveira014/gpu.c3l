module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;
import vma;
import std::atomic;

// Concrete (non-generic) slot table for textures. Monomorphized twin of
// gpu::slots to sidestep the c3c 0.8.0 generic-instantiation codegen crash.
const uint TEXTURE_FREE_NONE = 0;
const uint MAX_CACHED_VIEWS = 8;

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
    CachedView* cell;
    CachedView  previous;
}

struct TextureSlot {
    vk::Image          image;
    vma::Allocation    allocation;
    vk::ImageView      default_view;
    gpu::Format        format;
    gpu::TextureUsage  usage;
    uint               width;
    uint               height;
    uint               mip_levels;
    uint               array_layers;
    Atomic{int}        current_layout;
    bool               swapchain_owned;
    usz                allocation_size;
    char[DEBUG_NAME_CAP] debug_name;
    CachedView[MAX_CACHED_VIEWS] views;
}

struct TextureCell {
    TextureSlot value;
    uint        generation;
    uint        next_free;
    bool        used;
}

struct TextureTable {
    TextureCell[] slots;
    uint          free_head;
    uint          next_new;
}

fn gpu::TextureHandle? TextureTable.alloc(&self, TextureSlot value) {
    uint idx;
    if (self.free_head != TEXTURE_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
    } else if (self.next_new < self.slots.len) {
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    } else {
        return gpu::SLOT_TABLE_FULL~;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation);
    self.slots[idx].used = true;
    return (gpu::TextureHandle){ .index = idx, .generation = self.slots[idx].generation };
}

fn TextureSlot*? TextureTable.get(&self, gpu::TextureHandle h) {
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? TextureTable.free(&self, gpu::TextureHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation);
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
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

fn gpu::TextureFormatSupport? vk_get_texture_format_support(
    gpu::Device* device,
    gpu::Format format,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);

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
            check(result)!;
    }

    uint mip_levels = desc.mip_levels == 0 ? 1 : desc.mip_levels;
    uint array_layers = desc.array_layers == 0 ? 1 : desc.array_layers;
    uint depth = desc.depth == 0 ? 1 : desc.depth;
    bool supports_sample_one = ((uint)properties.sample_counts
        & (uint)vk::SampleCountFlagBits.COUNT_1_BIT) != 0;
    return desc.width <= properties.max_extent.width
        && desc.height <= properties.max_extent.height
        && depth <= properties.max_extent.depth
        && mip_levels <= properties.max_mip_levels
        && array_layers <= properties.max_array_layers
        && supports_sample_one;
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
    enter_tracked(state);
    defer exit_tracked(state);
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

<* Tracked layout is a relaxed atomic: exact under the per-texture
   single-recorder discipline, best-effort outside it. *>
fn gpu::TextureLayout texture_layout(TextureSlot* slot) {
    return (gpu::TextureLayout)slot.current_layout.load(AtomicOrdering.RELAXED);
}

fn void set_texture_layout(TextureSlot* slot, gpu::TextureLayout layout) {
    slot.current_layout.store((int)layout, AtomicOrdering.RELAXED);
}

fn void? check_texture_view_backend_result(
    VkDeviceState* state,
    ZString operation,
    gpu::TextureHandle texture,
    TextureSlot* slot,
    vk::Result result,
) {
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
   get-or-created in the slot's view cache. Cached views die with the texture.
   Format reinterpretation is out of scope. *>
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

    foreach (&cached : slot.views) {
        if (cached.used) continue;
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
        creation.cell = cached;
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
    if (operation == null) return gpu::SLOT_TABLE_FULL~;
    return report_contract_failure(
        state:               state,
        operation:           operation,
        public_fault:        gpu::SLOT_TABLE_FULL,
        resource_kind:       gpu::DebugResourceKind.TEXTURE,
        has_public_identity: true,
        index:               texture.index,
        generation:          texture.generation,
        debug_name:          slot_name(&slot.debug_name[0]),
        rejected_field:      "texture view cache",
        invariant:           "texture must have free cached-view capacity",
        backend_text:        null,
    )~;
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

fn void rollback_texture_view_creations(VkDeviceState* state, CachedViewCreation[] creations) @private {
    foreach (&creation : creations) {
        vk::destroy_image_view(state.device, creation.cell.view, null);
        *creation.cell = creation.previous;
    }
}

fn void destroy_cached_views(VkDeviceState* state, TextureSlot* slot) {
    foreach (&cached : slot.views) {
        if (cached.used) {
            vk::destroy_image_view(state.device, cached.view, null);
            cached.used = false;
        }
    }
}

<* destroy_cached_views, but enqueueing instead of destroying. *>
fn void enqueue_cached_views(VkDeviceState* state, TextureSlot* slot, ulong retire) @private {
    foreach (&cached : slot.views) {
        if (cached.used) {
            enqueue_deferred(state, DeferredKind.IMAGE_VIEW, retire, (void*)cached.view, (vma::Allocation)null);
            cached.used = false;
        }
    }
}

fn gpu::TextureHandle? vk_create_texture(gpu::Device* device, gpu::TextureDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
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

    vk::ImageCreateInfo image_info = vk::image_create_info()
        .set_image_type(vk::ImageType.TYPE_2D)
        .set_format(format_to_vk(desc.format))
        .set_extent({ .width = desc.width, .height = desc.height, .depth = 1 })
        .set_mip_levels(mip_levels)
        .set_array_layers(array_layers)
        .set_samples(vk::SampleCountFlagBits.COUNT_1_BIT)
        .set_tiling(vk::ImageTiling.OPTIMAL)
        .set_usage(texture_usage_to_vk(desc.usage))
        .set_initial_layout(vk::ImageLayout.UNDEFINED);
    uint[MAX_QUEUE_FAMILIES] share_families;
    uint share_count = used_queue_families(state, share_families[..]);
    if (desc.usage.shared_queues && share_count >= 2) {
        image_info = image_info
            .set_sharing_mode(vk::SharingMode.CONCURRENT)
            .set_queue_family_indices(share_families[:share_count]);
    }
    vma::AllocationCreateInfo alloc_info = { .usage = vma::MemoryUsage.AUTO_PREFER_DEVICE };

    TextureSlot slot = {
        .format         = desc.format,
        .usage          = desc.usage,
        .width          = desc.width,
        .height         = desc.height,
        .mip_levels     = mip_levels,
        .array_layers   = array_layers,
    };
    vma::AllocationInfo allocation_info;
    check_texture_create_result(
        state.allocator.create_image(
            image_info:     &image_info,
            alloc_info:     &alloc_info,
            out_image:      &slot.image,
            out_allocation: &slot.allocation,
            out_info:       &allocation_info,
        )
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

    vk::ImageViewCreateInfo view_info = vk::image_view_create_info()
        .set_image(slot.image)
        .set_view_type(array_layers > 1 ? vk::ImageViewType.TYPE_2D_ARRAY : vk::ImageViewType.TYPE_2D)
        .set_format(format_to_vk(desc.format))
        .set_subresource_range({
            .aspect_mask      = format_aspect(desc.format),
            .base_mip_level   = 0,
            .level_count      = mip_levels,
            .base_array_layer = 0,
            .layer_count      = array_layers,
        });
    if (catch err = check(vk::create_image_view(state.device, &view_info, null, &slot.default_view))) {
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

<* Frees the handle now; views, image, and allocation release is deferred
   until every referencing frame retires. *>
fn void? vk_destroy_texture(gpu::Device* device, gpu::TextureHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    TextureSlot* slot = state.textures.get(handle)!;
    if (slot.swapchain_owned) return gpu::INVALID_ARGUMENT~;
    DescriptorHeapState* heap = &state.heap;
    foreach (&cell : heap.texture_cells[:heap.texture_next_new]) {
        if (cell.used && (ulong)cell.owner == (ulong)handle) return gpu::RESOURCE_IN_USE~;
    }
    ulong retire = retire_timeline_value(state);
    vk::Image image = slot.image;
    vma::Allocation allocation = slot.allocation;
    vk::ImageView default_view = slot.default_view;
    enqueue_cached_views(state, slot, retire);
    state.textures.free(handle)!;
    enqueue_deferred(state, DeferredKind.IMAGE_VIEW, retire, (void*)default_view, (vma::Allocation)null);
    enqueue_deferred(state, DeferredKind.IMAGE, retire, (void*)image, allocation);
}
