module gpu::vk;

import gpu;
import vk;
import std::core::mem;

// Per-device backend state, reached through Device.backend_state.
struct VkDeviceState {
    vk::Instance               instance;
    vk::DebugUtilsMessengerEXT messenger;
    vk::PhysicalDevice         physical_device;
    vk::Device                 device;
    uint                       graphics_family;
    uint                       compute_family;
    uint                       transfer_family;
    vk::Queue                  graphics_queue;
    vk::Queue                  compute_queue;
    vk::Queue                  transfer_queue;
    gpu::DeviceCaps            caps;
    bool                       validation;
}

// One static vtable shared by every Vulkan device.
const gpu::BackendVTable VK_VTABLE = { .destroy_device = &vk_destroy_device };

<* Build a Vulkan device end to end. On failure the partial state is torn down
   and freed before the fault propagates, so nothing leaks. *>
fn gpu::Device? create_vk_device(gpu::DeviceDesc* desc) {
    VkDeviceState* state = mem::new(VkDeviceState);
    defer catch destroy_state(state);

    state.validation = desc.enable_validation;
    state.instance = create_vk_instance(desc)!;
    if (desc.enable_validation) setup_debug_messenger(state);

    state.physical_device = pick_physical_device(state.instance)!;

    vk::PhysicalDeviceVulkan13Features features13 = vk::physical_device_vulkan13features();
    vk::PhysicalDeviceVulkan12Features features12 = vk::physical_device_vulkan12features().set_next(&features13);
    vk::PhysicalDeviceFeatures2 features2 = vk::physical_device_features2().set_next(&features12);
    vk::get_physical_device_features2(state.physical_device, &features2);
    check_required_features(&features12, &features13)!;

    QueueFamilies families = select_queue_families(state.physical_device)!;
    state.graphics_family = families.graphics;
    state.compute_family  = families.compute;
    state.transfer_family = families.transfer;

    create_logical_device(state, &families)!;
    vk::get_device_queue(state.device, state.graphics_family, 0, &state.graphics_queue);
    state.compute_queue = state.graphics_queue;
    vk::get_device_queue(state.device, state.transfer_family, 0, &state.transfer_queue);

    state.caps = build_caps(&features12, &features13, state.physical_device);

    return (gpu::Device){
        .backend       = gpu::BackendKind.VULKAN,
        .caps          = state.caps,
        .vtable        = &VK_VTABLE,
        .backend_state = (void*)state,
    };
}

fn void vk_destroy_device(gpu::Device* device) @private {
    destroy_state((VkDeviceState*)device.backend_state);
    device.backend_state = null;
}

<* Reverse-order teardown; safe on a partial or zeroed state. *>
fn void destroy_state(VkDeviceState* state) @private {
    if (state == null) return;
    if (state.device != null) {
        vk::device_wait_idle(state.device);
        vk::destroy_device(state.device, null);
        state.device = (vk::Device)null;
    }
    destroy_debug_messenger(state);
    if (state.instance != null) {
        vk::destroy_instance(state.instance, null);
        state.instance = (vk::Instance)null;
    }
    mem::free(state);
}

<* Gate on the required 1.2/1.3 features. Pure, so it is unit-testable without
   a device. *>
fn void? check_required_features(vk::PhysicalDeviceVulkan12Features* v12, vk::PhysicalDeviceVulkan13Features* v13) {
    if (v12.buffer_device_address == 0) return gpu::UNSUPPORTED_FEATURE~;
    if (v12.timeline_semaphore == 0)    return gpu::UNSUPPORTED_FEATURE~;
    if (v13.synchronization2 == 0)      return gpu::UNSUPPORTED_FEATURE~;
    if (v13.dynamic_rendering == 0)     return gpu::UNSUPPORTED_FEATURE~;
}

<* Enumerate physical devices, require Vulkan 1.3, return the highest-scored. *>
fn vk::PhysicalDevice? pick_physical_device(vk::Instance instance) {
    uint count = 0;
    check(vk::enumerate_physical_devices(instance, &count, null))!;
    if (count == 0) return gpu::UNSUPPORTED_FEATURE~;

    @pool() {
        vk::PhysicalDevice[] devices = mem::talloc_array(vk::PhysicalDevice, (sz)count);
        check(vk::enumerate_physical_devices(instance, &count, devices.ptr))!;

        vk::PhysicalDevice best = (vk::PhysicalDevice)null;
        int best_score = -1;
        foreach (candidate : devices) {
            vk::PhysicalDeviceProperties props;
            vk::get_physical_device_properties(candidate, &props);
            if (props.api_version < API_VERSION_1_3) continue;
            int score = score_device((int)props.device_type);
            if (score > best_score) {
                best_score = score;
                best = candidate;
            }
        }
        if (best == null) return gpu::UNSUPPORTED_FEATURE~;
        return best;
    };
}

// VkPhysicalDeviceType: 0 other, 1 integrated, 2 discrete, 3 virtual, 4 cpu.
fn int score_device(int device_type) @private {
    switch (device_type) {
        case 2:  return 100;
        case 1:  return 50;
        case 3:  return 20;
        case 4:  return 10;
        default: return 1;
    }
}

<* Create the logical device with one graphics+compute queue and the required
   feature chain enabled. maintenance4 is enabled for SPIR-V 1.6 compute later. *>
fn void? create_logical_device(VkDeviceState* state, QueueFamilies* families) @private {
    float priority = 1.0;
    vk::DeviceQueueCreateInfo queue_info = vk::device_queue_create_info()
        .set_queue_family_index(families.graphics)
        .set_queue_priorities((&priority)[:1]);

    vk::PhysicalDeviceVulkan13Features features13 = vk::physical_device_vulkan13features()
        .set_dynamic_rendering(vk::TRUE)
        .set_synchronization2(vk::TRUE)
        .set_maintenance4(vk::TRUE);
    vk::PhysicalDeviceVulkan12Features features12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE)
        .set_next(&features13);

    vk::DeviceCreateInfo device_info = vk::device_create_info()
        .set_next(&features12)
        .set_queue_create_infos((&queue_info)[:1]);

    check(vk::create_device(state.physical_device, &device_info, null, &state.device))!;
}

<* Populate public caps from queried features and limits. *>
fn gpu::DeviceCaps build_caps(vk::PhysicalDeviceVulkan12Features* v12, vk::PhysicalDeviceVulkan13Features* v13, vk::PhysicalDevice physical_device) @private {
    vk::PhysicalDeviceProperties props;
    vk::get_physical_device_properties(physical_device, &props);
    return (gpu::DeviceCaps){
        .buffer_device_address      = v12.buffer_device_address != 0,
        .synchronization2           = v13.synchronization2 != 0,
        .dynamic_rendering          = v13.dynamic_rendering != 0,
        .timeline_semaphore         = v12.timeline_semaphore != 0,
        .descriptor_buffer          = false,
        .descriptor_indexing        = v12.descriptor_indexing != 0,
        .max_texture_descriptors    = 0,
        .max_sampler_descriptors    = 0,
        .min_uniform_alignment      = (usz)props.limits.min_uniform_buffer_offset_alignment,
        .min_storage_alignment      = (usz)props.limits.min_storage_buffer_offset_alignment,
        .min_texel_buffer_alignment = (usz)props.limits.min_texel_buffer_offset_alignment,
    };
}
