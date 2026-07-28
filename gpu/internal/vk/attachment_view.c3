module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::mem;
import std::core::env;
import std::atomic;
import std::atomic::types;

const uint ATTACHMENT_VIEW_FREE_NONE = 0;

struct AttachmentViewSlot {
    gpu::TextureHandle texture;
    vk::ImageView      native;
    gpu::Format        format;
    gpu::SampleCount   sample_count;
    uint               width;
    uint               height;
    bool               owns_native;
    bool               swapchain_owned;
    Atomic{uint}       tracked_command_references;
}

struct AttachmentViewCell {
    AttachmentViewSlot value;
    uint               generation;
    uint               next_free;
    bool               used;
}

struct AttachmentViewTable {
    AttachmentViewCell[] slots;
    ulong                owner;
    uint                 free_head;
    uint                 next_new;
}

fn gpu::AttachmentViewHandle? AttachmentViewTable.alloc(
    &self,
    AttachmentViewSlot value,
) {
    uint index;
    bool found;
    while (self.free_head != ATTACHMENT_VIEW_FREE_NONE) {
        index = self.free_head - 1;
        self.free_head = self.slots[index].next_free;
        if (can_allocate_handle_generation(self.slots[index].generation)) {
            found = true;
            break;
        }
        self.slots[index].next_free = ATTACHMENT_VIEW_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        index = self.next_new++;
        self.slots[index].generation = 0;
    }
    AttachmentViewCell* cell = &self.slots[index];
    cell.value = value;
    cell.generation = next_handle_generation(cell.generation)!;
    cell.used = true;
    return {
        .owner      = self.owner,
        .index      = index,
        .generation = cell.generation,
    };
}

fn AttachmentViewSlot*? AttachmentViewTable.get(
    &self,
    gpu::AttachmentViewHandle handle,
) {
    if (handle.owner != self.owner || handle.index >= self.slots.len) {
        return gpu::INVALID_HANDLE~;
    }
    AttachmentViewCell* cell = &self.slots[handle.index];
    if (!cell.used || cell.generation != handle.generation) {
        return gpu::INVALID_HANDLE~;
    }
    return &cell.value;
}

fn void? AttachmentViewTable.free(
    &self,
    gpu::AttachmentViewHandle handle,
) {
    self.get(handle)!;
    AttachmentViewCell* cell = &self.slots[handle.index];
    cell.value = {};
    cell.used = false;
    if (!should_recycle_handle_generation(cell.generation)) {
        cell.generation = HANDLE_GENERATION_MAX;
        cell.next_free = ATTACHMENT_VIEW_FREE_NONE;
        return;
    }
    cell.generation = next_handle_generation(cell.generation)!;
    cell.next_free = self.free_head;
    self.free_head = handle.index + 1;
}

fn gpu::AttachmentViewHandle? vk_create_attachment_view(
    VkDeviceState* state,
    gpu::AttachmentViewDesc* desc,
) {
    if (desc == null) return gpu::INVALID_ARGUMENT~;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    TextureSlot* texture = state.textures.get(desc.texture)!;
    if (texture_slot_is_3d(texture)
        || (!texture.usage.color_attach && !texture.usage.depth_attach)
        || desc.mip_level >= texture.mip_levels
        || desc.array_layer >= texture.array_layers) {
        return gpu::INVALID_ARGUMENT~;
    }

    MipExtent extent = texture_mip_extent(texture, desc.mip_level);
    bool owns_native = texture.mip_levels != 1
        || texture.array_layers != 1
        || desc.mip_level != 0
        || desc.array_layer != 0;
    vk::ImageView native = texture.default_view;
    if (owns_native) {
        vk::ImageViewCreateInfo info = vk::image_view_create_info()
            .set_image(texture.image)
            .set_view_type(vk::ImageViewType.TYPE_2D)
            .set_format(format_to_vk(texture.format))
            .set_subresource_range({
                .aspect_mask      = format_aspect(texture.format),
                .base_mip_level   = desc.mip_level,
                .level_count      = 1,
                .base_array_layer = desc.array_layer,
                .layer_count      = 1,
            });
        check_texture_view_backend_result(
            state:     state,
            operation: "create_attachment_view",
            texture:   desc.texture,
            slot:      texture,
            result:    vk::create_image_view(
                device:        state.device,
                p_create_info: &info,
                p_allocator:   null,
                p_view:        &native,
            ),
        )!;
    }
    defer catch {
        if (owns_native) vk::destroy_image_view(state.device, native, null);
    }

    AttachmentViewSlot slot = {
        .texture      = desc.texture,
        .native       = native,
        .format       = texture.format,
        .sample_count = texture.sample_count,
        .width        = extent.width,
        .height       = extent.height,
        .owns_native  = owns_native,
    };
    gpu::AttachmentViewHandle handle = state.attachment_views.alloc(slot)!;
    (void)texture.tracked_command_references.add(1, AtomicOrdering.RELAXED);
    return handle;
}

fn void? vk_destroy_attachment_view(
    VkDeviceState* state,
    gpu::AttachmentViewHandle handle,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    AttachmentViewSlot* slot = state.attachment_views.get(handle)!;
    if (slot.swapchain_owned) return gpu::INVALID_RESOURCE_STATE~;
    if (slot.tracked_command_references.load(AtomicOrdering.RELAXED) > 0) {
        drain_completed_submitted_commands(state)!;
        if (slot.tracked_command_references.load(AtomicOrdering.RELAXED) > 0) {
            return gpu::RESOURCE_IN_USE~;
        }
    }
    TextureSlot* texture = state.textures.get(slot.texture)!;
    vk::ImageView native = slot.native;
    bool owns_native = slot.owns_native;
    state.attachment_views.free(handle)!;
    (void)texture.tracked_command_references.sub(1, AtomicOrdering.RELAXED);
    if (owns_native) vk::destroy_image_view(state.device, native, null);
}

fn void destroy_attachment_view_table(VkDeviceState* state) {
    if (state.attachment_views.slots.ptr == null) return;
    foreach (&cell : state.attachment_views.slots[:state.attachment_views.next_new]) {
        if (cell.used && cell.value.owns_native) {
            vk::destroy_image_view(state.device, cell.value.native, null);
        }
    }
    mem::free(state.attachment_views.slots.ptr);
    state.attachment_views = {};
}
