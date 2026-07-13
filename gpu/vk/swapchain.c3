module gpu::vk;

import gpu @public;
import vk;
import std::thread;
import std::core::mem;

// Concrete (non-generic) slot table for swapchains. Monomorphized twin of
// gpu::slots to sidestep the c3c 0.8.0 generic-instantiation codegen crash.
const uint SWAPCHAIN_FREE_NONE = 0;
const uint MAX_SWAPCHAIN_IMAGES = 8;
const ulong ACQUIRE_TIMEOUT_NS = 1_000_000_000;

struct SwapchainSlot {
    vk::SurfaceKHR      surface;
    vk::SwapchainKHR    swapchain;
    uint                image_count;
    gpu::TextureHandle[MAX_SWAPCHAIN_IMAGES] textures;
    vk::Semaphore[MAX_SWAPCHAIN_IMAGES]      present_semaphores;
    vk::Semaphore[]     acquire_semaphores;
    gpu::SwapchainDesc  requested;
    gpu::SwapchainInfo  info;
    bool                acquire_pending;
    // Set once the pending acquire's coupled submit has consumed its binary
    // acquire semaphore; present requires this to avoid waiting it unsignaled.
    bool                acquire_consumed;
    uint                pending_image;
    vk::Semaphore       pending_acquire_semaphore;
}

struct SwapchainCell {
    SwapchainSlot value;
    uint          generation;
    uint          next_free;
    bool          used;
}

struct SwapchainTable {
    SwapchainCell[] slots;
    uint            free_head;
    uint            next_new;
}

fn gpu::SwapchainHandle? SwapchainTable.alloc(&self, SwapchainSlot value) {
    uint idx;
    if (self.free_head != SWAPCHAIN_FREE_NONE) {
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
    return (gpu::SwapchainHandle){ .index = idx, .generation = self.slots[idx].generation };
}

fn SwapchainSlot*? SwapchainTable.get(&self, gpu::SwapchainHandle h) {
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? SwapchainTable.free(&self, gpu::SwapchainHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    self.slots[idx].generation++;
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

<* Coupled-submit guard: the acquire must be pending and not yet consumed, and
   the coupling must run on GRAPHICS — the only queue these binary semaphores
   wait/signal on. Pure. *>
fn void? swapchain_submit_guard(bool acquire_pending, bool acquire_consumed, gpu::QueueKind kind) {
    if (!acquire_pending || acquire_consumed) return gpu::INVALID_RESOURCE_STATE~;
    if (kind != gpu::QueueKind.GRAPHICS) return gpu::INVALID_ARGUMENT~;
}

<* Present guard: the acquired image must have been consumed by exactly one
   coupled submit before its present semaphore is safe to wait. Pure. *>
fn void? swapchain_present_guard(bool acquire_pending, bool acquire_consumed) {
    if (!acquire_pending || !acquire_consumed) return gpu::INVALID_RESOURCE_STATE~;
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

<* Publish an active snapshot and reset the acquire lifecycle. *>
fn void publish_active_swapchain_info(SwapchainSlot* slot, gpu::SwapchainInfo info) {
    slot.info = info;
    slot.acquire_pending = false;
    slot.acquire_consumed = false;
}

<* Publish the dormant sentinel and reset the acquire lifecycle. *>
fn void publish_dormant_swapchain_info(SwapchainSlot* slot) {
    slot.info = dormant_swapchain_info();
    slot.acquire_pending = false;
    slot.acquire_consumed = false;
}

<* Build the public acquire result from committed texture state. Pure. *>
fn gpu::AcquiredImage make_acquired_image(TextureSlot* texture, gpu::TextureHandle handle, uint index, bool suboptimal) {
    return {
        .texture      = handle,
        .index        = index,
        .suboptimal   = suboptimal,
        .prior_layout = texture_layout(texture),
    };
}

fn vk::SurfaceKHR? create_platform_surface(
    VkDeviceState* state,
    gpu::SurfaceDesc* surface,
    WsiDiagnosticContext* context,
) @private {
    vk::SurfaceKHR vk_surface;
    switch (surface.platform) {
        case gpu::PlatformKind.WAYLAND:
            vk::WaylandSurfaceCreateInfoKHR info = {
                .s_type  = vk::StructureType.WAYLAND_SURFACE_CREATE_INFO_KHR,
                .display = surface.native_display,
                .surface = surface.native_window,
            };
            if (vk::extensions.vk_create_wayland_surface_khr == null) {
                return report_wsi_backend_failure(
                    context:      context,
                    public_fault: gpu::UNSUPPORTED_FEATURE,
                    backend_text: "required platform surface entry point is unavailable",
                )~;
            }
            check_wsi_backend_result(
                context: context,
                kind:    WsiResultKind.SURFACE,
                result:  vk::extensions.vk_create_wayland_surface_khr(
                    state.instance,
                    &info,
                    null,
                    &vk_surface,
                ),
            )!;
        case gpu::PlatformKind.X11:
            vk::XlibSurfaceCreateInfoKHR info = {
                .s_type = vk::StructureType.XLIB_SURFACE_CREATE_INFO_KHR,
                .dpy    = surface.native_display,
                .window = (ulong)surface.native_window,
            };
            if (vk::extensions.vk_create_xlib_surface_khr == null) {
                return report_wsi_backend_failure(
                    context:      context,
                    public_fault: gpu::UNSUPPORTED_FEATURE,
                    backend_text: "required platform surface entry point is unavailable",
                )~;
            }
            check_wsi_backend_result(
                context: context,
                kind:    WsiResultKind.SURFACE,
                result:  vk::extensions.vk_create_xlib_surface_khr(
                    state.instance,
                    &info,
                    null,
                    &vk_surface,
                ),
            )!;
        case gpu::PlatformKind.WIN32:
            vk::Win32SurfaceCreateInfoKHR info = {
                .s_type    = vk::StructureType.WIN32_SURFACE_CREATE_INFO_KHR,
                .hinstance = surface.native_display,
                .hwnd      = surface.native_window,
            };
            if (vk::extensions.vk_create_win32surface_khr == null) {
                return report_wsi_backend_failure(
                    context:      context,
                    public_fault: gpu::UNSUPPORTED_FEATURE,
                    backend_text: "required platform surface entry point is unavailable",
                )~;
            }
            check_wsi_backend_result(
                context: context,
                kind:    WsiResultKind.SURFACE,
                result:  vk::extensions.vk_create_win32surface_khr(
                    state.instance,
                    &info,
                    null,
                    &vk_surface,
                ),
            )!;
    }
    return vk_surface;
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
        result:  vk::extensions.vk_get_physical_device_surface_capabilities_khr(
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
        result:  vk::extensions.vk_get_physical_device_surface_formats_khr(
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
        result:  vk::extensions.vk_get_physical_device_surface_present_modes_khr(
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
            result:  vk::extensions.vk_get_physical_device_surface_formats_khr(
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
            result:  vk::extensions.vk_get_physical_device_surface_present_modes_khr(
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

        vk::SwapchainCreateInfoKHR create_info = {
            .s_type             = vk::StructureType.SWAPCHAIN_CREATE_INFO_KHR,
            .surface            = slot.surface,
            .min_image_count    = image_count,
            .image_format       = format_to_vk(format),
            .image_color_space  = vk::ColorSpaceKHR.SRGB_NONLINEAR_KHR,
            .image_extent       = extent,
            .image_array_layers = 1,
            .image_usage        = (vk::ImageUsageFlags)usage,
            .image_sharing_mode = vk::SharingMode.EXCLUSIVE,
            .pre_transform      = caps.current_transform,
            .composite_alpha    = vk::CompositeAlphaFlagBitsKHR.OPAQUE_BIT_KHR,
            .present_mode       = present_mode,
            .clipped            = vk::TRUE,
            .old_swapchain      = old,
        };
        check_wsi_backend_result(
            context: context,
            kind:    WsiResultKind.SWAPCHAIN_CREATE,
            result:  vk::extensions.vk_create_swapchain_khr(
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
            result:  vk::extensions.vk_get_swapchain_images_khr(
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
            result:  vk::extensions.vk_get_swapchain_images_khr(
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

fn gpu::SwapchainHandle? vk_create_swapchain(gpu::Device* device, gpu::SurfaceDesc* surface, gpu::SwapchainDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    WsiDiagnosticContext context = {
        .state      = state,
        .operation  = "create_swapchain",
        .debug_name = desc.debug_name,
    };
    if (!state.presentation_enabled) {
        return report_wsi_contract(
            context:        &context,
            public_fault:   gpu::UNSUPPORTED_FEATURE,
            rejected_field: "DeviceDesc.enable_presentation",
            invariant:      "must be enabled before creating a swapchain",
        )~;
    }

    SwapchainSlot slot = { .requested = *desc };
    defer catch destroy_swapchain_slot(state, &slot);
    slot.surface = create_platform_surface(state, surface, &context)!;

    uint supported = 0;
    check_wsi_backend_result(
        context: &context,
        kind:    WsiResultKind.SURFACE,
        result:  vk::extensions.vk_get_physical_device_surface_support_khr(
            state.physical_device,
            state.graphics_family,
            slot.surface,
            &supported,
        ),
    )!;
    if (supported == 0) {
        return report_wsi_backend_failure(
            context:      &context,
            public_fault: gpu::UNSUPPORTED_FEATURE,
            backend_text: "selected graphics queue family cannot present to the surface",
        )~;
    }

    slot.acquire_semaphores = mem::new_array(vk::Semaphore, state.frames_in_flight);
    for (uint i = 0; i < state.frames_in_flight; i++) {
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
    gpu::SwapchainHandle? published = state.swapchains.alloc(slot);
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
        vk::extensions.vk_destroy_swapchain_khr(state.device, slot.swapchain, null);
        slot.swapchain = (vk::SwapchainKHR)null;
    }
    if (slot.acquire_semaphores.ptr != null) {
        foreach (semaphore : slot.acquire_semaphores) {
            if (semaphore != null) vk::destroy_semaphore(state.device, semaphore, null);
        }
        mem::free(slot.acquire_semaphores.ptr);
        slot.acquire_semaphores = {};
    }
    if (slot.surface != null) {
        vk::extensions.vk_destroy_surface_khr(state.instance, slot.surface, null);
        slot.surface = (vk::SurfaceKHR)null;
    }
}

fn void? vk_destroy_swapchain(gpu::Device* device, gpu::SwapchainHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    WsiDiagnosticContext context = { .state = state, .operation = "destroy_swapchain" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, handle)!;
    {
        state.graphics_queue_mutex.lock()!!;
        defer state.graphics_queue_mutex.unlock();
        check_wsi_backend_result(
            context: &context,
            kind:    WsiResultKind.COMMON,
            result:  vk::queue_wait_idle(state.graphics_queue),
        )!;
    }
    destroy_swapchain_slot(state, slot);
    state.swapchains.free(handle)!;
}

fn void? vk_resize_swapchain(gpu::Device* device, gpu::SwapchainHandle handle, uint width, uint height) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    WsiDiagnosticContext context = { .state = state, .operation = "resize_swapchain" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, handle)!;
    {
        state.graphics_queue_mutex.lock()!!;
        defer state.graphics_queue_mutex.unlock();
        check_wsi_backend_result(
            context: &context,
            kind:    WsiResultKind.COMMON,
            result:  vk::queue_wait_idle(state.graphics_queue),
        )!;
    }

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
    if (old != null) vk::extensions.vk_destroy_swapchain_khr(state.device, old, null);
    if (build_fault) {
        // Never leave a half-built slot live: tear down whatever the failed
        // build produced and park the slot dormant so the next acquire gets
        // SWAPCHAIN_OUT_OF_DATE instead of undefined state.
        release_swapchain_images(state, slot);
        if (slot.swapchain != null) {
            vk::extensions.vk_destroy_swapchain_khr(state.device, slot.swapchain, null);
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

fn gpu::AcquiredImage? vk_acquire_next_image(gpu::Device* device, gpu::SwapchainHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    WsiDiagnosticContext context = { .state = state, .operation = "acquire_next_image" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, handle)!;
    check_swapchain_acquire_lifecycle(&context, slot)!;

    vk::Semaphore acquire_semaphore = slot.acquire_semaphores[state.frame_index];
    uint image_index = 0;
    vk::Result result = vk::extensions.vk_acquire_next_image_khr(
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
    slot.acquire_pending = true;
    slot.acquire_consumed = false;
    slot.pending_image = image_index;
    slot.pending_acquire_semaphore = acquire_semaphore;
    return make_acquired_image(
        texture:    texture,
        handle:     slot.textures[image_index],
        index:      image_index,
        suboptimal: result == vk::Result.SUBOPTIMAL_KHR,
    );
}

fn void? vk_present(gpu::Device* device, gpu::PresentDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    WsiDiagnosticContext context = { .state = state, .operation = "present" };
    SwapchainSlot* slot = resolve_swapchain_for_operation(&context, desc.swapchain)!;
    check_swapchain_present_lifecycle(&context, slot)!;
    // The public contract requires the image in PRESENT layout; enforce it
    // here instead of leaving it to the driver.
    TextureSlot* image_slot = resolve_swapchain_texture_for_operation(
        context: &context,
        handle:  slot.textures[slot.pending_image],
    )!;
    check_swapchain_present_layout(&context, image_slot)!;

    vk::PresentInfoKHR info = {
        .s_type               = vk::StructureType.PRESENT_INFO_KHR,
        .wait_semaphore_count = 1,
        .p_wait_semaphores    = &slot.present_semaphores[slot.pending_image],
        .swapchain_count      = 1,
        .p_swapchains         = &slot.swapchain,
        .p_image_indices      = &slot.pending_image,
    };
    slot.acquire_pending = false;
    slot.acquire_consumed = false;

    state.graphics_queue_mutex.lock()!!;
    defer state.graphics_queue_mutex.unlock();
    vk::Result result = vk::extensions.vk_queue_present_khr(state.graphics_queue, &info);
    return check_wsi_backend_result(
        context: &context,
        kind:    WsiResultKind.PRESENT,
        result:  result,
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
        result:  vk::extensions.vk_get_physical_device_surface_present_modes_khr(
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
            result:  vk::extensions.vk_get_physical_device_surface_present_modes_khr(
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
