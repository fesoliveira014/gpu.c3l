module gpu::vk;

import gpu;
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
    self.slots[idx].generation++;
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
    self.slots[idx].generation++;
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

fn uint full_mip_chain(uint width, uint height) @private {
    uint levels = 1;
    uint size = width > height ? width : height;
    while (size > 1) {
        size >>= 1;
        levels++;
    }
    return levels;
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

<* Storage-image usage on depth formats is illegal Vulkan. D24_UNORM_S8_UINT
   stays out: attachment support is not universal and no format-properties
   query exists. *>
fn void? check_texture_desc(gpu::TextureDesc* desc) @private {
    if (desc.dimension != gpu::TextureDimension.TEX_2D) return gpu::INVALID_ARGUMENT~;
    if (desc.width == 0 || desc.height == 0)            return gpu::INVALID_ARGUMENT~;
    if (desc.depth > 1)                                 return gpu::INVALID_ARGUMENT~;
    if (desc.mip_levels > full_mip_chain(desc.width, desc.height)) return gpu::INVALID_ARGUMENT~;
    if (format_to_vk(desc.format) == vk::Format.UNDEFINED) return gpu::INVALID_ARGUMENT~;
    if (desc.format == gpu::Format.D24_UNORM_S8_UINT) return gpu::INVALID_ARGUMENT~;
    bool is_depth = desc.format == gpu::Format.D32_FLOAT;
    if (is_depth && (desc.usage.color_attach || desc.usage.storage)) return gpu::INVALID_ARGUMENT~;
    if (!is_depth && desc.usage.depth_attach) return gpu::INVALID_ARGUMENT~;
    if (!desc.usage.sampled && !desc.usage.storage && !desc.usage.color_attach && !desc.usage.depth_attach) return gpu::INVALID_ARGUMENT~;
}

<* Resolve a view desc to an image view: null/zero desc (or the explicit full
   range) is the default view; anything else is normalized, validated, and
   get-or-created in the slot's view cache. Cached views die with the texture.
   Format reinterpretation is out of scope. *>
fn vk::ImageView? resolve_texture_view(VkDeviceState* state, TextureSlot* slot, gpu::TextureViewDesc* view) {
    if (view == null) return slot.default_view;
    if (view.format != gpu::Format.UNDEFINED && view.format != slot.format) return gpu::INVALID_ARGUMENT~;
    if (view.base_mip >= slot.mip_levels || view.base_layer >= slot.array_layers) return gpu::INVALID_ARGUMENT~;

    uint mip_count   = view.mip_count == 0 ? slot.mip_levels - view.base_mip : view.mip_count;
    uint layer_count = view.layer_count == 0 ? slot.array_layers - view.base_layer : view.layer_count;
    if (view.base_mip + mip_count > slot.mip_levels)       return gpu::INVALID_ARGUMENT~;
    if (view.base_layer + layer_count > slot.array_layers) return gpu::INVALID_ARGUMENT~;

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
        check(vk::create_image_view(state.device, &view_info, null, &image_view))!;
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
    return gpu::SLOT_TABLE_FULL~;
}

fn void destroy_cached_views(VkDeviceState* state, TextureSlot* slot) {
    foreach (&cached : slot.views) {
        if (cached.used) {
            vk::destroy_image_view(state.device, cached.view, null);
            cached.used = false;
        }
    }
}

fn gpu::TextureHandle? vk_create_texture(gpu::Device* device, gpu::TextureDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    check_texture_desc(desc)!;
    uint mip_levels   = desc.mip_levels == 0 ? 1 : desc.mip_levels;
    uint array_layers = desc.array_layers == 0 ? 1 : desc.array_layers;

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
    vma::AllocationCreateInfo alloc_info = { .usage = vma::MemoryUsage.AUTO_PREFER_DEVICE };

    TextureSlot slot = {
        .format         = desc.format,
        .usage          = desc.usage,
        .width          = desc.width,
        .height         = desc.height,
        .mip_levels     = mip_levels,
        .array_layers   = array_layers,
    };
    check(state.allocator.create_image(&image_info, &alloc_info, &slot.image, &slot.allocation, null))!;

    if (desc.debug_name != null) {
        state.allocator.set_allocation_name(slot.allocation, desc.debug_name);
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

    if (try handle = state.textures.alloc(slot)) {
        return handle;
    }
    vk::destroy_image_view(state.device, slot.default_view, null);
    state.allocator.destroy_image(slot.image, slot.allocation);
    return gpu::SLOT_TABLE_FULL~;
}

fn void? vk_destroy_texture(gpu::Device* device, gpu::TextureHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    TextureSlot* slot = state.textures.get(handle)!;
    if (slot.swapchain_owned) return gpu::INVALID_ARGUMENT~;
    destroy_cached_views(state, slot);
    vk::destroy_image_view(state.device, slot.default_view, null);
    state.allocator.destroy_image(slot.image, slot.allocation);
    state.textures.free(handle)!;
}
