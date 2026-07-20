module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;
import std::core::mem;

alias SwapchainCompletionPollFn = fn bool? (
    gpu::Device*,
    gpu::CompletionPoint,
);
alias SwapchainCompletionWaitFn = fn void? (
    gpu::Device*,
    gpu::CompletionPoint,
);

// Concrete (non-generic) slot table for swapchains. Monomorphized twin of
// gpu::slots to sidestep the c3c 0.8.0 generic-instantiation codegen crash.
const uint SWAPCHAIN_FREE_NONE = 0;
const uint MAX_SWAPCHAIN_IMAGES = 8;
const uint READINESS_SWAPCHAIN_INDEX_BITS = 8;
const uint READINESS_SWAPCHAIN_GENERATION_BITS = 24;
const uint READINESS_ACQUISITION_SHIFT = 32;
const ulong READINESS_SWAPCHAIN_INDEX_MASK =
    (1ull << READINESS_SWAPCHAIN_INDEX_BITS) - 1;
const ulong READINESS_SWAPCHAIN_GENERATION_MASK =
    (1ull << READINESS_SWAPCHAIN_GENERATION_BITS) - 1;
const ulong ACQUIRE_TIMEOUT_NS = 1_000_000_000;
const uint SWAPCHAIN_ACQUIRE_SLOT_COUNT = 2;

struct SwapchainSlot {
    gpu::Surface             surface_owner;
    vk::SurfaceKHR           surface;
    vk::SwapchainKHR         swapchain;
    uint                     image_count;
    gpu::TextureHandle[MAX_SWAPCHAIN_IMAGES] textures;
    vk::Semaphore[MAX_SWAPCHAIN_IMAGES]      present_semaphores;
    vk::Semaphore[]          acquire_semaphores;
    gpu::CompletionPoint[]   acquire_retirements;
    gpu::SwapchainDesc       requested;
    gpu::SwapchainInfo       info;
    bool                     acquire_pending;
    bool                     readiness_consumed;
    uint                     acquisition;
    uint                     next_acquire_semaphore;
    uint                     pending_acquire_semaphore_index;
    uint                     pending_image;
    vk::Semaphore            pending_acquire_semaphore;
    gpu::CompletionPoint     render_completion;
}

struct SwapchainCell {
    SwapchainSlot value;
    uint          generation;
    uint          next_free;
    bool          used;
}

struct SwapchainTable {
    SwapchainCell[] slots;
    ulong           owner;
    uint            free_head;
    uint            next_new;
}

fn gpu::SwapchainHandle? SwapchainTable.alloc(&self, SwapchainSlot value) {
    uint idx;
    bool found = false;
    while (self.free_head != SWAPCHAIN_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
        if (can_allocate_handle_generation(self.slots[idx].generation)) {
            found = true;
            break;
        }
        self.slots[idx].next_free = SWAPCHAIN_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].used = true;
    return (gpu::SwapchainHandle){
        .owner      = self.owner,
        .index      = idx,
        .generation = self.slots[idx].generation,
    };
}

fn SwapchainSlot*? SwapchainTable.get(&self, gpu::SwapchainHandle h) {
    if (h.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? SwapchainTable.free(&self, gpu::SwapchainHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    if (!should_recycle_handle_generation(self.slots[idx].generation)) {
        self.slots[idx].generation = HANDLE_GENERATION_MAX;
        self.slots[idx].next_free = SWAPCHAIN_FREE_NONE;
        return;
    }
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation)!;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

<* Match the preferred format (then BGRA/RGBA fallbacks in the requested srgb
   flavor, then any of the four) against the surface's SRGB_NONLINEAR formats.
   Pure. *>
fn gpu::Format? select_surface_format(vk::SurfaceFormatKHR[] available, gpu::Format preferred, bool srgb) {
    gpu::Format[5] candidates;
    uint candidate_count = 0;
    if (preferred != gpu::Format.UNDEFINED) candidates[candidate_count++] = preferred;
    if (srgb) {
        candidates[candidate_count++] = gpu::Format.BGRA8_SRGB;
        candidates[candidate_count++] = gpu::Format.RGBA8_SRGB;
    } else {
        candidates[candidate_count++] = gpu::Format.BGRA8_UNORM;
        candidates[candidate_count++] = gpu::Format.RGBA8_UNORM;
    }
    candidates[candidate_count++] = gpu::Format.BGRA8_UNORM;
    candidates[candidate_count++] = gpu::Format.RGBA8_UNORM;

    foreach (candidate : candidates[:candidate_count]) {
        vk::Format wanted = format_to_vk(candidate);
        foreach (surface_format : available) {
            if (surface_format.format == wanted
                && surface_format.color_space == vk::ColorSpaceKHR.SRGB_NONLINEAR_KHR) {
                return candidate;
            }
        }
    }
    return gpu::UNSUPPORTED_FEATURE~;
}

<* FIFO is spec-guaranteed; anything else falls back to it. Pure. *>
fn vk::PresentModeKHR select_present_mode(vk::PresentModeKHR[] available, gpu::PresentMode requested) {
    vk::PresentModeKHR wanted;
    switch (requested) {
        case gpu::PresentMode.IMMEDIATE: wanted = vk::PresentModeKHR.IMMEDIATE_KHR;
        case gpu::PresentMode.MAILBOX:   wanted = vk::PresentModeKHR.MAILBOX_KHR;
        default:                         return vk::PresentModeKHR.FIFO_KHR;
    }
    foreach (mode : available) {
        if (mode == wanted) return wanted;
    }
    return vk::PresentModeKHR.FIFO_KHR;
}

<* 0 = minimum + 1; a capability max of 0 means unbounded. Pure. *>
fn uint clamp_image_count(uint requested, uint min_count, uint max_count) {
    uint count = requested == 0 ? min_count + 1 : requested;
    if (count < min_count) count = min_count;
    if (max_count != 0 && count > max_count) count = max_count;
    return count;
}

<* A fixed currentExtent (not 0xFFFFFFFF) wins; otherwise the request clamps
   to the min/max bounds. Pure. *>
fn vk::Extent2D clamp_extent(vk::SurfaceCapabilitiesKHR* caps, uint width, uint height) {
    if (caps.current_extent.width != uint::max) return caps.current_extent;
    vk::Extent2D extent = { .width = width, .height = height };
    if (extent.width < caps.min_image_extent.width)   extent.width = caps.min_image_extent.width;
    if (extent.width > caps.max_image_extent.width)   extent.width = caps.max_image_extent.width;
    if (extent.height < caps.min_image_extent.height) extent.height = caps.min_image_extent.height;
    if (extent.height > caps.max_image_extent.height) extent.height = caps.max_image_extent.height;
    return extent;
}

fn gpu::SwapchainReadiness? make_swapchain_readiness(
    gpu::Device device,
    gpu::SwapchainHandle swapchain,
    uint acquisition,
) {
    if (!device.is_valid()
        || swapchain.index > READINESS_SWAPCHAIN_INDEX_MASK
        || swapchain.generation == 0
        || swapchain.generation > READINESS_SWAPCHAIN_GENERATION_MASK
        || acquisition == 0) {
        return gpu::INVALID_HANDLE~;
    }
    return {
        .device = device,
        .payload = (ulong)swapchain.index
            | ((ulong)swapchain.generation
                << READINESS_SWAPCHAIN_INDEX_BITS)
            | ((ulong)acquisition << READINESS_ACQUISITION_SHIFT),
    };
}

fn uint swapchain_readiness_index(gpu::SwapchainReadiness readiness) {
    return (uint)(readiness.payload & READINESS_SWAPCHAIN_INDEX_MASK);
}

fn uint swapchain_readiness_generation(gpu::SwapchainReadiness readiness) {
    return (uint)((readiness.payload >> READINESS_SWAPCHAIN_INDEX_BITS)
        & READINESS_SWAPCHAIN_GENERATION_MASK);
}

fn uint swapchain_readiness_acquisition(
    gpu::SwapchainReadiness readiness,
) {
    return (uint)(readiness.payload >> READINESS_ACQUISITION_SHIFT);
}

fn gpu::SwapchainHandle? swapchain_handle_from_readiness(
    VkDeviceState* state,
    gpu::SwapchainReadiness readiness,
) {
    uint generation = swapchain_readiness_generation(readiness);
    uint acquisition = swapchain_readiness_acquisition(readiness);
    if (!readiness.is_valid() || generation == 0 || acquisition == 0) {
        return gpu::INVALID_HANDLE~;
    }
    return {
        .owner = gpu::resource_owner_tag(
            state.resource_owner,
            gpu::ResourceOwnerKind.SWAPCHAIN,
        ),
        .index      = swapchain_readiness_index(readiness),
        .generation = generation,
    };
}

fn uint? next_swapchain_acquisition(uint acquisition) {
    if (acquisition == uint::max) return gpu::SLOT_TABLE_FULL~;
    return acquisition + 1;
}

fn bool? poll_swapchain_completion(
    gpu::Device* device,
    gpu::CompletionPoint point,
) @private {
    return vk_poll_completion(
        device,
        gpu::completion_point_queue_id(point),
        gpu::completion_point_sequence(point),
    );
}

fn void? wait_swapchain_completion(
    gpu::Device* device,
    gpu::CompletionPoint point,
) @private {
    return vk_wait_completion(
        device,
        gpu::completion_point_queue_id(point),
        gpu::completion_point_sequence(point),
        ulong::max,
    );
}

fn void? wait_pending_swapchain_render_with(
    gpu::Device* device,
    SwapchainSlot* slot,
    SwapchainCompletionWaitFn wait_fn,
) {
    if (!slot.acquire_pending || !slot.readiness_consumed) return;
    if (!slot.render_completion.is_valid()) return gpu::BACKEND_ERROR~;
    wait_fn(device, slot.render_completion)!;
}

fn void? wait_pending_swapchain_render(
    gpu::Device* device,
    SwapchainSlot* slot,
) @private {
    return wait_pending_swapchain_render_with(
        device,
        slot,
        &wait_swapchain_completion,
    );
}

fn void? consume_unsubmitted_swapchain_acquire_with(
    VkDeviceState* state,
    WsiDiagnosticContext* context,
    SwapchainSlot* slot,
    QueueSubmit2Fn submit_fn,
) {
    if (!slot.acquire_pending || slot.readiness_consumed) return;
    if (slot.pending_acquire_semaphore == null) return gpu::BACKEND_ERROR~;

    vk::SemaphoreSubmitInfo[1] waits = {
        vk::semaphore_submit_info()
            .set_semaphore(slot.pending_acquire_semaphore)
            .set_stage_mask(
                vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
            ),
    };
    vk::SubmitInfo2 submit = vk::submit_info2()
        .set_wait_semaphore_infos(waits[..]);
    Mutex* queue_mutex = native_queue_mutex_for(
        state,
        state.graphics_queue,
    );
    queue_mutex.lock()!!;
    vk::Result result = submit_fn(
        state.graphics_queue,
        1,
        &submit,
        (vk::Fence)null,
    );
    queue_mutex.unlock();
    check_wsi_backend_result(
        context: context,
        kind:    WsiResultKind.COMMON,
        result:  result,
    )!;
    clear_swapchain_acquisition(slot);
}

fn uint? select_acquire_semaphore_with(
    gpu::Device* device,
    SwapchainSlot* slot,
    SwapchainCompletionPollFn poll_fn,
) {
    if (slot.acquire_retirements.len == 0) return gpu::BACKEND_ERROR~;
    for (uint offset = 0; offset < slot.acquire_retirements.len; offset++) {
        uint index = (slot.next_acquire_semaphore + offset)
            % (uint)slot.acquire_retirements.len;
        gpu::CompletionPoint retirement = slot.acquire_retirements[index];
        if (!retirement.is_valid() || poll_fn(device, retirement)!) {
            return index;
        }
    }
    return gpu::WAIT_TIMEOUT~;
}

fn uint? select_acquire_semaphore(
    gpu::Device* device,
    SwapchainSlot* slot,
) @private {
    return select_acquire_semaphore_with(
        device,
        slot,
        &poll_swapchain_completion,
    );
}

fn void commit_swapchain_submit(
    SwapchainSlot* slot,
    gpu::CompletionPoint point,
) {
    slot.readiness_consumed = true;
    slot.render_completion = point;
    slot.acquire_retirements[slot.pending_acquire_semaphore_index] = point;
}

fn void? swapchain_resize_guard(bool acquire_pending) {
    if (acquire_pending) return gpu::INVALID_RESOURCE_STATE~;
}

fn void? swapchain_submit_guard(
    bool acquire_pending,
    bool readiness_consumed,
    uint expected_acquisition,
    uint provided_acquisition,
    gpu::QueueRoles roles,
) {
    if (!acquire_pending
        || readiness_consumed
        || expected_acquisition == 0
        || provided_acquisition != expected_acquisition) {
        return gpu::INVALID_RESOURCE_STATE~;
    }
    if (!roles.graphics) return gpu::INVALID_ARGUMENT~;
}

fn void? swapchain_present_guard(
    bool acquire_pending,
    bool readiness_consumed,
    uint expected_acquisition,
    uint provided_acquisition,
    gpu::CompletionPoint expected_completion,
    gpu::CompletionPoint provided_completion,
) {
    if (!acquire_pending
        || !readiness_consumed
        || expected_acquisition == 0
        || provided_acquisition != expected_acquisition) {
        return gpu::INVALID_RESOURCE_STATE~;
    }
    if (provided_completion != expected_completion) {
        return gpu::INVALID_ARGUMENT~;
    }
}

<* Map the VkImageUsageFlags granted at swapchain creation to the wrapped
   TextureSlot's usage: color attach is always requested, transfer_src
   carries over iff the surface granted it. Pure. *>
fn gpu::TextureUsage swapchain_image_usage(uint granted_vk_usage) {
    if ((granted_vk_usage & (uint)vk::ImageUsageFlagBits.TRANSFER_SRC_BIT) != 0) {
        return (gpu::TextureUsage){ .color_attach, .transfer_src };
    }
    return (gpu::TextureUsage){ .color_attach };
}

<* Build an active runtime snapshot. Pure. *>
fn gpu::SwapchainInfo make_swapchain_info(gpu::Format format, vk::Extent2D extent, uint image_count, vk::PresentModeKHR present_mode) {
    return {
        .format       = format,
        .width        = extent.width,
        .height       = extent.height,
        .image_count  = image_count,
        .present_mode = present_mode_from_vk(present_mode),
    };
}

<* Build the queryable zero-extent sentinel. Pure. *>
fn gpu::SwapchainInfo dormant_swapchain_info() {
    return {
        .format       = gpu::Format.UNDEFINED,
        .present_mode = gpu::PresentMode.FIFO,
        .dormant      = true,
    };
}

fn void clear_swapchain_acquisition(SwapchainSlot* slot) {
    slot.acquire_pending = false;
    slot.readiness_consumed = false;
    slot.pending_image = 0;
    slot.pending_acquire_semaphore = (vk::Semaphore)null;
    slot.render_completion = {};
}

fn bool present_result_consumes_acquisition(vk::Result result) {
    return result != vk::Result.ERROR_OUT_OF_HOST_MEMORY
        && result != vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
}

fn void commit_present_result(
    SwapchainSlot* slot,
    gpu::AcquiredImage* image,
    vk::Result result,
) {
    if (!present_result_consumes_acquisition(result)) return;
    clear_swapchain_acquisition(slot);
    *image = {};
}

fn void publish_active_swapchain_info(
    SwapchainSlot* slot,
    gpu::SwapchainInfo info,
) {
    slot.info = info;
    clear_swapchain_acquisition(slot);
}

fn void publish_dormant_swapchain_info(SwapchainSlot* slot) {
    slot.info = dormant_swapchain_info();
    clear_swapchain_acquisition(slot);
}

fn gpu::AcquiredImage make_acquired_image(
    TextureSlot* texture,
    gpu::TextureHandle handle,
    gpu::SwapchainReadiness readiness,
    uint index,
    bool suboptimal,
) {
    return {
        .texture      = handle,
        .readiness    = readiness,
        .index        = index,
        .suboptimal   = suboptimal,
        .prior_use    = swapchain_prior_use(texture),
    };
}

<* Create the vk swapchain for `slot` at the clamped extent and wrap its
   images as swapchain-owned texture slots. `old` is chained and NOT destroyed
   here. A zero clamped extent marks the slot dormant and creates nothing. *>
fn void? build_swapchain(
    VkDeviceState* state,
    SwapchainSlot* slot,
    uint width,
    uint height,
    vk::SwapchainKHR old,
    WsiDiagnosticContext* context,
) @private {
    vk::SurfaceCapabilitiesKHR caps;
    check_wsi_backend_result(
        context: context,
        kind:    WsiResultKind.SURFACE,
        result:  state.instance_dispatch.surface.get_surface_capabilities(
            state.physical_device,
            slot.surface,
            &caps,
        ),
    )!;

    vk::Extent2D extent = clamp_extent(&caps, width, height);
    if (extent.width == 0 || extent.height == 0) {
        publish_dormant_swapchain_info(slot);
        return;
    }

    uint format_count = 0;
    check_wsi_backend_result(
        context: context,
        kind:    WsiResultKind.SURFACE,
        result:  state.instance_dispatch.surface.get_surface_formats(
            state.physical_device,
            slot.surface,
            &format_count,
            null,
        ),
    )!;
    uint mode_count = 0;
    check_wsi_backend_result(
        context: context,
        kind:    WsiResultKind.SURFACE,
        result:  state.instance_dispatch.surface.get_surface_present_modes(
            state.physical_device,
            slot.surface,
            &mode_count,
            null,
        ),
    )!;
    if (format_count == 0 || mode_count == 0) {
        return report_wsi_backend_failure(
            context:      context,
            public_fault: gpu::UNSUPPORTED_FEATURE,
            backend_text: "surface exposes no usable format or present mode",
        )~;
    }

    @pool() {
        vk::SurfaceFormatKHR[] formats = mem::talloc_array(vk::SurfaceFormatKHR, (sz)format_count);
        check_wsi_backend_result(
            context: context,
            kind:    WsiResultKind.SURFACE_ENUMERATION,
            result:  state.instance_dispatch.surface.get_surface_formats(
                state.physical_device,
                slot.surface,
                &format_count,
                formats.ptr,
            ),
        )!;
        vk::PresentModeKHR[] modes = mem::talloc_array(vk::PresentModeKHR, (sz)mode_count);
        check_wsi_backend_result(
            context: context,
            kind:    WsiResultKind.SURFACE_ENUMERATION,
            result:  state.instance_dispatch.surface.get_surface_present_modes(
                state.physical_device,
                slot.surface,
                &mode_count,
                modes.ptr,
            ),
        )!;

        gpu::Format? selected_format = select_surface_format(
            available: formats,
            preferred: slot.requested.preferred_format,
            srgb:      slot.requested.srgb,
        );
        if (catch format_fault = selected_format) {
            return report_wsi_backend_failure(
                context:      context,
                public_fault: format_fault,
                backend_text: "surface exposes no compatible swapchain format",
            )~;
        }
        gpu::Format format = selected_format;
        vk::PresentModeKHR present_mode = select_present_mode(modes, slot.requested.present_mode);
        uint image_count = clamp_image_count(slot.requested.image_count, caps.min_image_count, caps.max_image_count);

        uint usage = (uint)vk::ImageUsageFlagBits.COLOR_ATTACHMENT_BIT;
        if (((uint)caps.supported_usage_flags & (uint)vk::ImageUsageFlagBits.TRANSFER_SRC_BIT) != 0) {
            usage |= (uint)vk::ImageUsageFlagBits.TRANSFER_SRC_BIT;
        }

        uint[MAX_QUEUE_FAMILIES] queue_families;
        uint queue_family_count = swapchain_queue_families(state, queue_families[..]);

        vk::SwapchainCreateInfoKHR create_info = {
            .s_type                   = vk::StructureType.SWAPCHAIN_CREATE_INFO_KHR,
            .surface                  = slot.surface,
            .min_image_count          = image_count,
            .image_format             = format_to_vk(format),
            .image_color_space        = vk::ColorSpaceKHR.SRGB_NONLINEAR_KHR,
            .image_extent             = extent,
            .image_array_layers       = 1,
            .image_usage              = (vk::ImageUsageFlags)usage,
            .image_sharing_mode       = queue_family_count == 1
                ? vk::SharingMode.EXCLUSIVE
                : vk::SharingMode.CONCURRENT,
            .queue_family_index_count = queue_family_count == 1 ? 0 : queue_family_count,
            .p_queue_family_indices   = queue_family_count == 1 ? null : &queue_families[0],
            .pre_transform            = caps.current_transform,
            .composite_alpha          = vk::CompositeAlphaFlagBitsKHR.OPAQUE_BIT_KHR,
            .present_mode             = present_mode,
            .clipped                  = vk::TRUE,
            .old_swapchain            = old,
        };
        check_wsi_backend_result(
            context: context,
            kind:    WsiResultKind.SWAPCHAIN_CREATE,
            result:  state.device_dispatch.presentation.create_swapchain(
                state.device,
                &create_info,
                null,
                &slot.swapchain,
            ),
        )!;

        uint got = 0;
        check_wsi_backend_result(
            context: context,
            kind:    WsiResultKind.ENUMERATION,
            result:  state.device_dispatch.presentation.get_swapchain_images(
                state.device,
                slot.swapchain,
                &got,
                null,
            ),
        )!;
        if (got > MAX_SWAPCHAIN_IMAGES) {
            return report_wsi_backend_failure(
                context:      context,
                public_fault: gpu::UNSUPPORTED_FEATURE,
                backend_text: "driver returned more swapchain images than the backend supports",
            )~;
        }
        vk::Image[MAX_SWAPCHAIN_IMAGES] images;
        check_wsi_backend_result(
            context: context,
            kind:    WsiResultKind.ENUMERATION,
            result:  state.device_dispatch.presentation.get_swapchain_images(
                state.device,
                slot.swapchain,
                &got,
                &images[0],
            ),
        )!;

        for (uint i = 0; i < got; i++) {
            vk::ImageViewCreateInfo view_info = vk::image_view_create_info()
                .set_image(images[i])
                .set_view_type(vk::ImageViewType.TYPE_2D)
                .set_format(format_to_vk(format))
                .set_subresource_range({
                    .aspect_mask      = format_aspect(format),
                    .base_mip_level   = 0,
                    .level_count      = 1,
                    .base_array_layer = 0,
                    .layer_count      = 1,
                });
            vk::ImageView view;
            check_wsi_backend_result(
                context: context,
                kind:    WsiResultKind.COMMON,
                result:  vk::create_image_view(
                    device:        state.device,
                    p_create_info: &view_info,
                    p_allocator:   null,
                    p_view:        &view,
                ),
            )!;

            TextureSlot texture = {
                .image           = images[i],
                .default_view    = view,
                .format          = format,
                .usage           = swapchain_image_usage(usage),
                .access          = { .graphics },
                .width           = extent.width,
                .height          = extent.height,
                .mip_levels      = 1,
                .array_layers    = 1,
                .swapchain_owned = true,
            };
            copy_debug_name(&texture.debug_name[0], slot.requested.debug_name);
            set_object_name(
                state:         state,
                object_type:   vk::ObjectType.IMAGE,
                object_handle: (ulong)images[i],
                name:          slot.requested.debug_name,
            );
            bool published = false;
            {
                state.resource_mutex.lock()!!;
                defer state.resource_mutex.unlock();
                if (try handle = state.textures.alloc(texture)) {
                    slot.textures[i] = handle;
                    published = true;
                }
            }
            if (!published) {
                vk::destroy_image_view(state.device, view, null);
                return report_wsi_backend_failure(
                    context:      context,
                    public_fault: gpu::SLOT_TABLE_FULL,
                    backend_text: "backend texture slot capacity exhausted while wrapping swapchain images",
                )~;
            }
            // Count as we go so a mid-loop failure leaves the slot cleanable
            // by release_swapchain_images (semaphores are null-checked there).
            slot.image_count = i + 1;

            vk::SemaphoreCreateInfo semaphore_info = vk::semaphore_create_info();
            check_wsi_backend_result(
                context: context,
                kind:    WsiResultKind.COMMON,
                result:  vk::create_semaphore(
                    device:        state.device,
                    p_create_info: &semaphore_info,
                    p_allocator:   null,
                    p_semaphore:   &slot.present_semaphores[i],
                ),
            )!;
        }

        publish_active_swapchain_info(
            slot,
            make_swapchain_info(
                format:       format,
                extent:       extent,
                image_count:  slot.image_count,
                present_mode: present_mode,
            ),
        );
    };
}

<* Release the wrapped image slots, views, and present semaphores — everything
   `build_swapchain` created except the vk swapchain itself. *>
fn void release_swapchain_images(VkDeviceState* state, SwapchainSlot* slot) @private {
    for (uint i = 0; i < slot.image_count; i++) {
        if (try texture = state.textures.get(slot.textures[i])) {
            vk::destroy_image_view(state.device, texture.default_view, null);
            destroy_cached_views(state, texture);
            {
                state.resource_mutex.lock()!!;
                defer state.resource_mutex.unlock();
                (void)state.textures.free(slot.textures[i]);
            }
        }
        if (slot.present_semaphores[i] != null) {
            vk::destroy_semaphore(state.device, slot.present_semaphores[i], null);
            slot.present_semaphores[i] = (vk::Semaphore)null;
        }
    }
    slot.image_count = 0;
}

fn gpu::SwapchainHandle? vk_create_swapchain(
    gpu::Device* device,
    gpu::Surface* surface,
    gpu::SwapchainDesc* desc,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    WsiDiagnosticContext context = {
        .state      = state,
        .operation  = "create_swapchain",
        .debug_name = desc.debug_name,
    };
    if (!state.presentation.enabled) {
        return report_wsi_contract(
            context:        &context,
            public_fault:   gpu::UNSUPPORTED_FEATURE,
            rejected_field: "DeviceRequest.presentation",
            invariant:      "must be requested before device creation",
        )~;
    }

    SwapchainSlot slot = { .requested = *desc };
    defer catch destroy_swapchain_slot(state, &slot);
    gpu::surface_retain_swapchain(surface)!;
    slot.surface_owner = *surface;
    slot.surface = (vk::SurfaceKHR)gpu::surface_backend_handle(surface)!;

    uint supported = 0;
    check_wsi_backend_result(
        context: &context,
        kind:    WsiResultKind.SURFACE,
        result:  state.instance_dispatch.surface.get_surface_support(
            state.physical_device,
            state.presentation.family,
            slot.surface,
            &supported,
        ),
    )!;
    if (supported == 0) {
        return report_wsi_backend_failure(
            context:      &context,
            public_fault: gpu::UNSUPPORTED_FEATURE,
            backend_text: "selected presentation queue family cannot present to the surface",
        )~;
    }

    slot.acquire_semaphores = mem::new_array(vk::Semaphore, SWAPCHAIN_ACQUIRE_SLOT_COUNT);
    slot.acquire_retirements = mem::new_array(
        gpu::CompletionPoint,
        SWAPCHAIN_ACQUIRE_SLOT_COUNT,
    );
    for (uint i = 0; i < SWAPCHAIN_ACQUIRE_SLOT_COUNT; i++) {
        vk::SemaphoreCreateInfo semaphore_info = vk::semaphore_create_info();
        check_wsi_backend_result(
            context: &context,
            kind:    WsiResultKind.COMMON,
            result:  vk::create_semaphore(
                device:        state.device,
                p_create_info: &semaphore_info,
                p_allocator:   null,
                p_semaphore:   &slot.acquire_semaphores[i],
            ),
        )!;
    }

    build_swapchain(
        state:   state,
        slot:    &slot,
        width:   desc.width,
        height:  desc.height,
        old:     (vk::SwapchainKHR)null,
        context: &context,
    )!;
    gpu::SwapchainHandle? published = state.presentation.swapchains.alloc(slot);
    if (catch publish_fault = published) {
        return report_wsi_backend_failure(
            context:      &context,
            public_fault: publish_fault,
            backend_text: "backend swapchain slot capacity exhausted",
        )~;
    }
    return published;
}

fn void destroy_swapchain_slot(VkDeviceState* state, SwapchainSlot* slot) @private {
    release_swapchain_images(state, slot);
    if (slot.swapchain != null) {
        state.device_dispatch.presentation.destroy_swapchain(state.device, slot.swapchain, null);
        slot.swapchain = (vk::SwapchainKHR)null;
    }
    if (slot.acquire_semaphores.ptr != null) {
        foreach (semaphore : slot.acquire_semaphores) {
            if (semaphore != null) vk::destroy_semaphore(state.device, semaphore, null);
        }
        mem::free(slot.acquire_semaphores.ptr);
        slot.acquire_semaphores = {};
    }
    if (slot.acquire_retirements.ptr != null) {
        mem::free(slot.acquire_retirements.ptr);
        slot.acquire_retirements = {};
    }
    slot.surface = (vk::SurfaceKHR)null;
    if (slot.surface_owner.is_valid()) {
        gpu::surface_release_swapchain(&slot.surface_owner)!!;
        slot.surface_owner = {};
    }
}

fn void? wait_swapchain_queues_idle(
    VkDeviceState* state,
    WsiDiagnosticContext* context,
) @private {
    vk::Queue[2] queues;
    uint queue_count = swapchain_idle_queues(state, queues[..]);
    for (uint index = 0; index < queue_count; index++) {
        {
            Mutex* queue_mutex = native_queue_mutex_for(state, queues[index]);
            queue_mutex.lock()!!;
            defer queue_mutex.unlock();
            check_wsi_backend_result(
                context: context,
                kind:    WsiResultKind.COMMON,
                result:  vk::queue_wait_idle(queues[index]),
            )!;
        }
    }
}

fn void? vk_destroy_swapchain(gpu::Device* device, gpu::SwapchainHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    WsiDiagnosticContext context = { .state = state, .operation = "destroy_swapchain" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, handle)!;
    consume_unsubmitted_swapchain_acquire_with(
        state:     state,
        context:   &context,
        slot:      slot,
        submit_fn: &real_queue_submit2,
    )!;
    wait_pending_swapchain_render(device, slot)!;
    wait_swapchain_queues_idle(state, &context)!;
    destroy_swapchain_slot(state, slot);
    state.presentation.swapchains.free(handle)!;
}

fn void? vk_resize_swapchain(gpu::Device* device, gpu::SwapchainHandle handle, uint width, uint height) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    WsiDiagnosticContext context = { .state = state, .operation = "resize_swapchain" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, handle)!;
    check_swapchain_resize_lifecycle(&context, slot)!;
    wait_swapchain_queues_idle(state, &context)!;

    vk::SwapchainKHR old = slot.swapchain;
    release_swapchain_images(state, slot);
    slot.swapchain = (vk::SwapchainKHR)null;

    fault build_fault = @catch(build_swapchain(
        state:   state,
        slot:    slot,
        width:   width,
        height:  height,
        old:     old,
        context: &context,
    ));
    if (old != null) state.device_dispatch.presentation.destroy_swapchain(state.device, old, null);
    if (build_fault) {
        // A failed rebuild remains dormant and queryable.
        release_swapchain_images(state, slot);
        if (slot.swapchain != null) {
            state.device_dispatch.presentation.destroy_swapchain(state.device, slot.swapchain, null);
            slot.swapchain = (vk::SwapchainKHR)null;
        }
        publish_dormant_swapchain_info(slot);
        return build_fault~;
    }
}

fn gpu::SwapchainInfo? vk_get_swapchain_info(gpu::Device* device, gpu::SwapchainHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    WsiDiagnosticContext context = { .state = state, .operation = "get_swapchain_info" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, handle)!;
    return slot.info;
}

fn gpu::AcquiredImage? vk_acquire_next_image(
    gpu::Device* device,
    gpu::SwapchainHandle handle,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    WsiDiagnosticContext context = { .state = state, .operation = "acquire_next_image" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, handle)!;
    check_swapchain_acquire_lifecycle(&context, slot)!;
    uint acquisition = next_swapchain_acquisition(slot.acquisition)!;

    uint acquire_semaphore_index = select_acquire_semaphore(device, slot)!;
    vk::Semaphore acquire_semaphore =
        slot.acquire_semaphores[acquire_semaphore_index];
    uint image_index = 0;
    vk::Result result = state.device_dispatch.presentation.acquire_next_image(
        state.device,
        slot.swapchain,
        ACQUIRE_TIMEOUT_NS,
        acquire_semaphore,
        (vk::Fence)null,
        &image_index,
    );
    check_wsi_backend_result(
        context: &context,
        kind:    WsiResultKind.ACQUIRE,
        result:  result,
    )!;

    TextureSlot* texture = resolve_swapchain_texture_for_operation(
        context: &context,
        handle:  slot.textures[image_index],
    )!;
    gpu::SwapchainReadiness readiness = make_swapchain_readiness(
        *device,
        handle,
        acquisition,
    )!;
    slot.acquire_pending = true;
    slot.readiness_consumed = false;
    slot.acquisition = acquisition;
    slot.next_acquire_semaphore = (acquire_semaphore_index + 1)
        % (uint)slot.acquire_semaphores.len;
    slot.pending_acquire_semaphore_index = acquire_semaphore_index;
    slot.pending_image = image_index;
    slot.pending_acquire_semaphore = acquire_semaphore;
    slot.render_completion = {};
    return make_acquired_image(
        texture:    texture,
        handle:     slot.textures[image_index],
        readiness:  readiness,
        index:      image_index,
        suboptimal: result == vk::Result.SUBOPTIMAL_KHR,
    );
}

fn void? vk_present_with(
    gpu::Device* device,
    gpu::AcquiredImage* image,
    gpu::CompletionPoint render_completion,
    vk::PFN_vkQueuePresentKHR present_fn,
) {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    if (!image.readiness.is_valid() || image.readiness.device != *device) {
        return gpu::INVALID_HANDLE~;
    }
    gpu::SwapchainHandle handle = swapchain_handle_from_readiness(
        state,
        image.readiness,
    )!;
    WsiDiagnosticContext context = { .state = state, .operation = "present" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, handle)!;
    check_swapchain_present_contract(
        context:           &context,
        slot:              slot,
        image:             image,
        render_completion: render_completion,
    )!;
    TextureSlot* image_slot = resolve_swapchain_texture_for_operation(
        context: &context,
        handle:  image.texture,
    )!;
    vk::PresentInfoKHR info = {
        .s_type               = vk::StructureType.PRESENT_INFO_KHR,
        .wait_semaphore_count = 1,
        .p_wait_semaphores    = &slot.present_semaphores[slot.pending_image],
        .swapchain_count      = 1,
        .p_swapchains         = &slot.swapchain,
        .p_image_indices      = &slot.pending_image,
    };

    Mutex* queue_mutex = present_queue_mutex_for(state);
    queue_mutex.lock()!!;
    vk::Result result = present_fn(state.presentation.queue, &info);
    queue_mutex.unlock();
    if (present_result_consumes_acquisition(result)) {
        mark_texture_presented(image_slot);
    }
    commit_present_result(slot, image, result);
    return check_wsi_backend_result(
        context: &context,
        kind:    WsiResultKind.PRESENT,
        result:  result,
    );
}

fn void? vk_present(
    gpu::Device* device,
    gpu::AcquiredImage* image,
    gpu::CompletionPoint render_completion,
) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    return vk_present_with(
        device:            device,
        image:             image,
        render_completion: render_completion,
        present_fn:        state.device_dispatch.presentation.queue_present,
    );
}

fn gpu::PresentModeSupport? vk_get_present_mode_support(gpu::Device* device, gpu::SwapchainHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    WsiDiagnosticContext context = { .state = state, .operation = "get_present_mode_support" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, handle)!;

    uint mode_count = 0;
    check_wsi_backend_result(
        context: &context,
        kind:    WsiResultKind.SURFACE,
        result:  state.instance_dispatch.surface.get_surface_present_modes(
            state.physical_device,
            slot.surface,
            &mode_count,
            null,
        ),
    )!;
    gpu::PresentModeSupport support;
    if (mode_count == 0) return support;

    @pool() {
        vk::PresentModeKHR[] modes = mem::talloc_array(vk::PresentModeKHR, (sz)mode_count);
        check_wsi_backend_result(
            context: &context,
            kind:    WsiResultKind.SURFACE_ENUMERATION,
            result:  state.instance_dispatch.surface.get_surface_present_modes(
                state.physical_device,
                slot.surface,
                &mode_count,
                modes.ptr,
            ),
        )!;
        foreach (mode : modes[:mode_count]) {
            switch (mode) {
                case vk::PresentModeKHR.FIFO_KHR:      support.fifo = true;
                case vk::PresentModeKHR.IMMEDIATE_KHR: support.immediate = true;
                case vk::PresentModeKHR.MAILBOX_KHR:   support.mailbox = true;
                default:                               break;
            }
        }
    };
    return support;
}
