module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::thread;
import std::core::mem;
import std::atomic::types;

alias SwapchainCompletionPollFn = fn bool? (
    VkDeviceState*,
    gpu::CompletionPoint,
);
alias SwapchainAcquireFn = vk::PFN_vkAcquireNextImageKHR;
alias SwapchainFenceStatusFn = fn vk::Result (
    vk::Device,
    vk::Fence,
);
alias SwapchainFenceResetFn = fn vk::Result (
    vk::Device,
    uint,
    vk::Fence*,
);
alias SwapchainFenceWaitFn = fn vk::Result (
    vk::Device,
    uint,
    vk::Fence*,
    vk::Bool32,
    ulong,
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
const uint SWAPCHAIN_ACQUIRE_SLOT_COUNT = 2;

struct SwapchainSlot {
    gpu::Surface             surface_owner;
    vk::SurfaceKHR           surface;
    vk::SwapchainKHR         swapchain;
    uint                     image_count;
    gpu::TextureHandle[MAX_SWAPCHAIN_IMAGES]        textures;
    gpu::AttachmentViewHandle[MAX_SWAPCHAIN_IMAGES] attachment_views;
    vk::Semaphore[MAX_SWAPCHAIN_IMAGES]             present_semaphores;
    vk::Fence[MAX_SWAPCHAIN_IMAGES]                 present_fences;
    bool[MAX_SWAPCHAIN_IMAGES]                      present_fence_pending;
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

struct SwapchainAcquisitionCandidate {
    gpu::SwapchainReadiness readiness;
    uint                    acquisition;
    uint                    semaphore_index;
    vk::Semaphore           semaphore;
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
        .owner = gpu::internal::resource_owner_tag(
            state.resource_owner,
            gpu::internal::ResourceOwnerKind.SWAPCHAIN,
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
    VkDeviceState* state,
    gpu::CompletionPoint point,
) @private {
    return vk_poll_completion(
        state,
        gpu::internal::completion_point_queue_id(point),
        gpu::internal::completion_point_sequence(point),
    );
}

fn vk::Result real_swapchain_fence_status(
    vk::Device device,
    vk::Fence fence,
) @private {
    return vk::get_fence_status(device, fence);
}

fn vk::Result real_swapchain_fence_reset(
    vk::Device device,
    uint fence_count,
    vk::Fence* fences,
) @private {
    return vk::reset_fences(device, fence_count, fences);
}
fn vk::Result real_swapchain_fence_wait(
    vk::Device device,
    uint fence_count,
    vk::Fence* fences,
    vk::Bool32 wait_all,
    ulong timeout_ns,
) @private {
    return vk::wait_for_fences(
        device,
        fence_count,
        fences,
        wait_all,
        timeout_ns,
    );
}

fn void? swapchain_release_guard(
    bool acquire_pending,
    bool images_referenced,
    bool presentations_pending,
) {
    if (acquire_pending) return gpu::INVALID_RESOURCE_STATE~;
    if (images_referenced || presentations_pending) {
        return gpu::RESOURCE_IN_USE~;
    }
}

fn bool? poll_present_fences_with(
    VkDeviceState* state,
    SwapchainSlot* slot,
    SwapchainFenceStatusFn status_fn,
) {
    bool ready = true;
    for (uint i = 0; i < slot.image_count; i++) {
        if (!slot.present_fence_pending[i]) continue;
        vk::Result result = status_fn(
            state.device,
            slot.present_fences[i],
        );
        if (result == vk::Result.SUCCESS) {
            slot.present_fence_pending[i] = false;
        } else if (result == vk::Result.NOT_READY) {
            ready = false;
        } else {
            check_backend_result(
                state,
                "poll swapchain presentation retirement",
                result,
            )!;
        }
    }
    return ready;
}
fn void? wait_present_fences_with(
    VkDeviceState* state,
    SwapchainSlot* slot,
    ulong timeout_ns,
    SwapchainFenceWaitFn wait_fn,
    SwapchainFenceResetFn reset_fn,
) {
    vk::Fence[MAX_SWAPCHAIN_IMAGES] fences;
    uint[MAX_SWAPCHAIN_IMAGES] indices;
    uint fence_count = 0;
    for (uint i = 0; i < slot.image_count; i++) {
        if (!slot.present_fence_pending[i]) continue;
        fences[fence_count] = slot.present_fences[i];
        indices[fence_count] = i;
        fence_count++;
    }
    if (fence_count == 0) return;

    check_wait_backend_result(
        state:     state,
        operation: "wait_swapchain_presentations",
        result:    wait_fn(
            state.device,
            fence_count,
            &fences[0],
            (vk::Bool32)1,
            timeout_ns,
        ),
    )!;
    check_backend_result(
        state:     state,
        operation: "reset_swapchain_presentations",
        result:    reset_fn(
            state.device,
            fence_count,
            &fences[0],
        ),
    )!;
    for (uint i = 0; i < fence_count; i++) {
        slot.present_fence_pending[indices[i]] = false;
    }
}

fn void? wait_present_fences(
    VkDeviceState* state,
    SwapchainSlot* slot,
    ulong timeout_ns,
) @private {
    return wait_present_fences_with(
        state:     state,
        slot:      slot,
        timeout_ns: timeout_ns,
        wait_fn:   &real_swapchain_fence_wait,
        reset_fn:  &real_swapchain_fence_reset,
    );
}

fn void? prepare_present_fence_with(
    VkDeviceState* state,
    SwapchainSlot* slot,
    uint image_index,
    SwapchainFenceStatusFn status_fn,
    SwapchainFenceResetFn reset_fn,
) {
    if (image_index >= slot.image_count
        || slot.present_fences[image_index] == null) {
        return gpu::BACKEND_ERROR~;
    }
    if (!slot.present_fence_pending[image_index]) return;

    vk::Fence fence = slot.present_fences[image_index];
    vk::Result status = status_fn(state.device, fence);
    if (status == vk::Result.NOT_READY) return gpu::WAIT_TIMEOUT~;
    check_backend_result(
        state,
        "poll swapchain presentation fence",
        status,
    )!;
    check_backend_result(
        state,
        "reset swapchain presentation fence",
        reset_fn(state.device, 1, &fence),
    )!;
    slot.present_fence_pending[image_index] = false;
}

fn void? prepare_present_fence(
    VkDeviceState* state,
    SwapchainSlot* slot,
    uint image_index,
) @private {
    return prepare_present_fence_with(
        state:       state,
        slot:        slot,
        image_index: image_index,
        status_fn:   &real_swapchain_fence_status,
        reset_fn:    &real_swapchain_fence_reset,
    );
}

fn uint? select_acquire_semaphore_with(
    VkDeviceState* state,
    SwapchainSlot* slot,
    SwapchainCompletionPollFn poll_fn,
) {
    if (slot.acquire_retirements.len == 0) return gpu::BACKEND_ERROR~;
    for (uint offset = 0; offset < slot.acquire_retirements.len; offset++) {
        uint index = (slot.next_acquire_semaphore + offset)
            % (uint)slot.acquire_retirements.len;
        gpu::CompletionPoint retirement = slot.acquire_retirements[index];
        if (!retirement.is_valid() || poll_fn(state, retirement)!) {
            return index;
        }
    }
    return gpu::WAIT_TIMEOUT~;
}

fn void commit_swapchain_submit(
    SwapchainSlot* slot,
    gpu::CompletionPoint point,
) {
    slot.readiness_consumed = true;
    slot.render_completion = point;
    slot.acquire_retirements[slot.pending_acquire_semaphore_index] = point;
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
    gpu::AttachmentViewHandle attachment_view,
    gpu::SwapchainReadiness readiness,
    uint index,
    bool suboptimal,
) {
    return {
        .texture         = handle,
        .attachment_view = attachment_view,
        .readiness       = readiness,
        .index           = index,
        .suboptimal      = suboptimal,
        .prior_state     = swapchain_prior_state(texture),
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
                .image                 = images[i],
                .default_view          = view,
                .format                = format,
                .usage                 = swapchain_image_usage(usage),
                .access                = { .graphics },
                .sample_count          = gpu::SampleCount.ONE,
                .width                 = extent.width,
                .height                = extent.height,
                .mip_levels            = 1,
                .array_layers          = 1,
                .live_descriptor_views = 0,
                .swapchain_owned       = true,
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
                    AttachmentViewSlot attachment_view = {
                        .texture         = handle,
                        .native          = view,
                        .format          = format,
                        .sample_count    = gpu::SampleCount.ONE,
                        .width           = extent.width,
                        .height          = extent.height,
                        .swapchain_owned = true,
                    };
                    if (try attachment_handle =
                        state.attachment_views.alloc(attachment_view)) {
                        slot.attachment_views[i] = attachment_handle;
                        published = true;
                    }
                    if (!published) (void)state.textures.free(handle);
                }
            }
            if (!published) {
                vk::destroy_image_view(state.device, view, null);
                return report_wsi_backend_failure(
                    context:      context,
                    public_fault: gpu::SLOT_TABLE_FULL,
                    backend_text: "backend resource slot capacity exhausted while wrapping swapchain images",
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
            vk::FenceCreateInfo fence_info = vk::fence_create_info();
            check_wsi_backend_result(
                context: context,
                kind:    WsiResultKind.COMMON,
                result:  vk::create_fence(
                    device:        state.device,
                    p_create_info: &fence_info,
                    p_allocator:   null,
                    p_fence:       &slot.present_fences[i],
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

<* Release the wrapped image slots, views, semaphores, and fences created by
   `build_swapchain`. The caller holds `resource_mutex`. *>
fn void release_swapchain_images_locked(
    VkDeviceState* state,
    SwapchainSlot* slot,
) @private {
    for (uint i = 0; i < slot.image_count; i++) {
        if (try state.attachment_views.get(slot.attachment_views[i])) {
            (void)state.attachment_views.free(slot.attachment_views[i]);
        }
        if (try texture = state.textures.get(slot.textures[i])) {
            vk::destroy_image_view(state.device, texture.default_view, null);
            destroy_cached_views(state, texture);
            (void)state.textures.free(slot.textures[i]);
        }
        if (slot.present_semaphores[i] != null) {
            vk::destroy_semaphore(
                state.device,
                slot.present_semaphores[i],
                null,
            );
            slot.present_semaphores[i] = (vk::Semaphore)null;
        }
        if (slot.present_fences[i] != null) {
            vk::destroy_fence(
                state.device,
                slot.present_fences[i],
                null,
            );
            slot.present_fences[i] = (vk::Fence)null;
            slot.present_fence_pending[i] = false;
        }
        slot.attachment_views[i] = {};
        slot.textures[i] = {};
    }
    slot.image_count = 0;
}

fn void release_swapchain_images(
    VkDeviceState* state,
    SwapchainSlot* slot,
) @private {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    release_swapchain_images_locked(state, slot);
}

fn bool? swapchain_images_referenced_locked(
    VkDeviceState* state,
    SwapchainSlot* slot,
) @private {
    for (uint i = 0; i < slot.image_count; i++) {
        gpu::TextureHandle handle = slot.textures[i];
        TextureSlot* texture = state.textures.get(handle)!;
        if (texture_has_live_descriptor_views(texture)) return true;
        if (texture.tracked_command_references.load(AtomicOrdering.RELAXED) > 0) {
            return true;
        }
        // A missing borrowed slot carries no references; a missing texture
        // is real inconsistency, so its lookup above still faults.
        if (try attachment_view = state.attachment_views.get(
            slot.attachment_views[i],
        )) {
            if (attachment_view.tracked_command_references.load(
                AtomicOrdering.RELAXED,
            ) > 0) {
                return true;
            }
        }
    }
    return false;
}

fn void? release_swapchain_images_for_lifecycle_with(
    VkDeviceState* state,
    WsiDiagnosticContext* context,
    SwapchainSlot* slot,
    SwapchainFenceStatusFn status_fn,
    SwapchainFenceWaitFn wait_fn,
    SwapchainFenceResetFn reset_fn,
) {
    (void)wait_fn;
    (void)reset_fn;
    if (slot.acquire_pending) {
        return report_wsi_contract(
            context:        context,
            public_fault:   gpu::INVALID_RESOURCE_STATE,
            rejected_field: "swapchain acquire lifecycle",
            invariant:      "the acquired image must be submitted and presented before release",
        )~;
    }

    bool presentations_pending = !poll_present_fences_with(
        state,
        slot,
        status_fn,
    )!;
    if (presentations_pending) {
        return report_wsi_contract(
            context:        context,
            public_fault:   gpu::RESOURCE_IN_USE,
            rejected_field: "presentation resource retirement",
            invariant:      "all prior presentations must complete before release is retried",
        )~;
    }

    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    drain_completed_submitted_commands(state)!;
    bool images_referenced = swapchain_images_referenced_locked(
        state,
        slot,
    )!;
    if (catch guard_fault = swapchain_release_guard(
        acquire_pending:       false,
        images_referenced:     images_referenced,
        presentations_pending: false,
    )) {
        return report_wsi_contract(
            context:        context,
            public_fault:   guard_fault,
            rejected_field: "swapchain image references",
            invariant:      "all command and texture-view references must be released first",
        )~;
    }
    release_swapchain_images_locked(state, slot);
}

fn void? release_swapchain_images_for_lifecycle(
    VkDeviceState* state,
    WsiDiagnosticContext* context,
    SwapchainSlot* slot,
) @private {
    return release_swapchain_images_for_lifecycle_with(
        state:     state,
        context:   context,
        slot:      slot,
        status_fn: &real_swapchain_fence_status,
        wait_fn:   &real_swapchain_fence_wait,
        reset_fn:  &real_swapchain_fence_reset,
    );
}

fn gpu::SwapchainHandle? vk_create_swapchain(
    VkDeviceState* state,
    gpu::Surface* surface,
    gpu::SwapchainDesc* desc,
) {
    WsiDiagnosticContext context = {
        .state      = state,
        .operation  = "create_swapchain",
        .debug_name = desc.debug_name,
    };
    if (!state.presentation.enabled) {
        return report_wsi_contract(
            context:        &context,
            public_fault:   gpu::UNSUPPORTED_FEATURE,
            rejected_field: "DeviceDesc.surface",
            invariant:      "must be requested before device creation",
        )~;
    }

    SwapchainSlot slot = { .requested = *desc };
    defer catch destroy_swapchain_slot(state, &slot);
    gpu::internal::surface_retain_swapchain(surface)!;
    slot.surface_owner = *surface;
    slot.surface = (vk::SurfaceKHR)gpu::internal::surface_backend_handle(surface)!;

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
        gpu::internal::surface_release_swapchain(&slot.surface_owner)!!;
        slot.surface_owner = {};
    }
}

fn void? vk_wait_swapchain_presentations(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
    ulong timeout_ns,
) {
    WsiDiagnosticContext context = {
        .state     = state,
        .operation = "wait_swapchain_presentations",
    };
    SwapchainSlot* slot = resolve_swapchain_for_operation(
        &context,
        handle,
    )!;
    return wait_present_fences(
        state:     state,
        slot:      slot,
        timeout_ns: timeout_ns,
    );
}

fn void? vk_destroy_swapchain(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
) {
    WsiDiagnosticContext context = {
        .state     = state,
        .operation = "destroy_swapchain",
    };
    SwapchainSlot* slot = resolve_swapchain_for_operation(
        &context,
        handle,
    )!;
    release_swapchain_images_for_lifecycle(
        state,
        &context,
        slot,
    )!;
    if (slot.swapchain != null) {
        state.device_dispatch.presentation.destroy_swapchain(
            state.device,
            slot.swapchain,
            null,
        );
        slot.swapchain = (vk::SwapchainKHR)null;
    }
    destroy_swapchain_slot(state, slot);
    state.presentation.swapchains.free(handle)!;
}

fn void? vk_resize_swapchain(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
    uint width,
    uint height,
) {
    WsiDiagnosticContext context = {
        .state     = state,
        .operation = "resize_swapchain",
    };
    SwapchainSlot* slot = resolve_swapchain_for_operation(
        &context,
        handle,
    )!;
    release_swapchain_images_for_lifecycle(
        state,
        &context,
        slot,
    )!;

    vk::SwapchainKHR old = slot.swapchain;
    slot.swapchain = (vk::SwapchainKHR)null;

    fault build_fault = @catch(build_swapchain(
        state:   state,
        slot:    slot,
        width:   width,
        height:  height,
        old:     old,
        context: &context,
    ));
    if (old != null) {
        state.device_dispatch.presentation.destroy_swapchain(
            state.device,
            old,
            null,
        );
    }
    if (build_fault) {
        // A failed rebuild remains dormant and queryable.
        release_swapchain_images(state, slot);
        if (slot.swapchain != null) {
            state.device_dispatch.presentation.destroy_swapchain(
                state.device,
                slot.swapchain,
                null,
            );
            slot.swapchain = (vk::SwapchainKHR)null;
        }
        publish_dormant_swapchain_info(slot);
        return build_fault~;
    }
}

fn gpu::SwapchainInfo? vk_get_swapchain_info(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
) {
    WsiDiagnosticContext context = { .state = state, .operation = "get_swapchain_info" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, handle)!;
    return slot.info;
}

fn gpu::AcquiredImage? vk_acquire_next_image(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
    ulong timeout_ns,
) {
    return vk_acquire_next_image_with(
        state:       state,
        handle:      handle,
        timeout_ns:  timeout_ns,
        poll_fn:     &poll_swapchain_completion,
        acquire_fn:  state.device_dispatch.presentation.acquire_next_image,
    );
}

fn SwapchainAcquisitionCandidate? prepare_swapchain_acquisition(
    WsiDiagnosticContext* context,
    SwapchainSlot* slot,
    gpu::SwapchainHandle handle,
    SwapchainCompletionPollFn poll_fn,
) {
    check_swapchain_acquire_lifecycle(context, slot)!;
    uint acquisition = next_swapchain_acquisition(slot.acquisition)!;
    gpu::SwapchainReadiness readiness = make_swapchain_readiness(
        context.state.owner,
        handle,
        acquisition,
    )!;
    if (slot.acquire_semaphores.len == 0
        || slot.acquire_semaphores.len != slot.acquire_retirements.len) {
        return report_wsi_backend_failure(
            context:      context,
            public_fault: gpu::BACKEND_ERROR,
            backend_text: "swapchain acquire semaphore ring invariant failed",
        )~;
    }
    uint semaphore_index = select_acquire_semaphore_with(
        context.state,
        slot,
        poll_fn,
    )!;
    vk::Semaphore semaphore = slot.acquire_semaphores[semaphore_index];
    if (semaphore == null) {
        return report_wsi_backend_failure(
            context:      context,
            public_fault: gpu::BACKEND_ERROR,
            backend_text: "swapchain acquire semaphore is null",
        )~;
    }
    return {
        .readiness       = readiness,
        .acquisition     = acquisition,
        .semaphore_index = semaphore_index,
        .semaphore       = semaphore,
    };
}

fn gpu::AcquiredImage? commit_acquired_image(
    WsiDiagnosticContext* context,
    SwapchainSlot* slot,
    gpu::SwapchainHandle handle,
    SwapchainAcquisitionCandidate candidate,
    uint image_index,
    bool suboptimal,
) {
    if (image_index >= slot.image_count
        || image_index >= MAX_SWAPCHAIN_IMAGES) {
        return report_wsi_backend_failure(
            context:      context,
            public_fault: gpu::BACKEND_ERROR,
            backend_text: "native acquire returned an invalid image index",
        )~;
    }
    gpu::AttachmentViewHandle attachment_view =
        slot.attachment_views[image_index];
    if (!attachment_view.is_valid()) {
        return report_wsi_backend_failure(
            context:      context,
            public_fault: gpu::BACKEND_ERROR,
            backend_text: "swapchain attachment view invariant failed",
        )~;
    }
    gpu::TextureHandle texture_handle = slot.textures[image_index];
    TextureSlot* texture = resolve_swapchain_texture_for_operation(
        context: context,
        handle:  texture_handle,
    )!;
    gpu::AcquiredImage image = make_acquired_image(
        texture:         texture,
        handle:          texture_handle,
        attachment_view: attachment_view,
        readiness:       candidate.readiness,
        index:           image_index,
        suboptimal:      suboptimal,
    );

    slot.acquire_pending = true;
    slot.readiness_consumed = false;
    slot.acquisition = candidate.acquisition;
    slot.next_acquire_semaphore = (candidate.semaphore_index + 1)
        % (uint)slot.acquire_semaphores.len;
    slot.pending_acquire_semaphore_index = candidate.semaphore_index;
    slot.pending_image = image_index;
    slot.pending_acquire_semaphore = candidate.semaphore;
    slot.render_completion = {};
    return image;
}

fn gpu::AcquiredImage? vk_acquire_next_image_with(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
    ulong timeout_ns,
    SwapchainCompletionPollFn poll_fn,
    SwapchainAcquireFn acquire_fn,
) {
    WsiDiagnosticContext context = { .state = state, .operation = "acquire_next_image" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, handle)!;
    SwapchainAcquisitionCandidate candidate = prepare_swapchain_acquisition(
        context:  &context,
        slot:     slot,
        handle:   handle,
        poll_fn:  poll_fn,
    )!;
    uint image_index = 0;
    vk::Result result = acquire_fn(
        state.device,
        slot.swapchain,
        timeout_ns,
        candidate.semaphore,
        (vk::Fence)null,
        &image_index,
    );
    check_wsi_backend_result(
        context: &context,
        kind:    WsiResultKind.ACQUIRE,
        result:  result,
    )!;
    return commit_acquired_image(
        context:     &context,
        slot:        slot,
        handle:      handle,
        candidate:   candidate,
        image_index: image_index,
        suboptimal:  result == vk::Result.SUBOPTIMAL_KHR,
    );
}

fn void? vk_present_with(
    VkDeviceState* state,
    gpu::AcquiredImage* image,
    gpu::CompletionPoint render_completion,
    vk::PFN_vkQueuePresentKHR present_fn,
) {
    if (!image.readiness.is_valid()
        || image.readiness.device != state.owner) {
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
    uint image_index = slot.pending_image;
    prepare_present_fence(state, slot, image_index)!;
    vk::Fence present_fence = slot.present_fences[image_index];
    vk::SwapchainPresentFenceInfoEXT fence_info =
        vk::swapchain_present_fence_info_ext()
            .set_fences((&present_fence)[:1]);
    vk::PresentInfoKHR info = {
        .s_type               = vk::StructureType.PRESENT_INFO_KHR,
        .p_next               = &fence_info,
        .wait_semaphore_count = 1,
        .p_wait_semaphores    = &slot.present_semaphores[image_index],
        .swapchain_count      = 1,
        .p_swapchains         = &slot.swapchain,
        .p_image_indices      = &image_index,
    };

    Mutex* queue_mutex = present_queue_mutex_for(state);
    queue_mutex.lock()!!;
    vk::Result result = present_fn(state.presentation.queue, &info);
    queue_mutex.unlock();
    if (present_result_consumes_acquisition(result)) {
        slot.present_fence_pending[image_index] = true;
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
    VkDeviceState* state,
    gpu::AcquiredImage* image,
    gpu::CompletionPoint render_completion,
) {
    return vk_present_with(
        state:             state,
        image:             image,
        render_completion: render_completion,
        present_fn:        state.device_dispatch.presentation.queue_present,
    );
}

fn gpu::PresentModeSupport? vk_get_present_mode_support(
    VkDeviceState* state,
    gpu::SwapchainHandle handle,
) {
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
