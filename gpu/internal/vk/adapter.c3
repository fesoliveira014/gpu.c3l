module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::core::mem;

const uint API_VERSION_1_2 @private = (1u << 22) | (2u << 12);

struct ShaderHeapSupport {
    bool               indexing;
    IndexingHeapLimits limits;
}

struct VkAdapterState {
    vk::PhysicalDevice                 physical_device;
    vk::PhysicalDeviceProperties       properties;
    vk::PhysicalDeviceDriverProperties driver_properties;
    ShaderHeapSupport                  shader_heap;
    bool                               device_baseline_supported;
    gpu::AdapterInfo                   info;
    gpu::AdapterDiagnostics            diagnostics;
}

fn gpu::AdapterClass adapter_class(vk::PhysicalDeviceType device_type) {
    switch (device_type) {
        case vk::PhysicalDeviceType.INTEGRATED_GPU: return gpu::AdapterClass.INTEGRATED;
        case vk::PhysicalDeviceType.DISCRETE_GPU:   return gpu::AdapterClass.DISCRETE;
        case vk::PhysicalDeviceType.VIRTUAL_GPU:    return gpu::AdapterClass.VIRTUAL;
        case vk::PhysicalDeviceType.CPU:            return gpu::AdapterClass.SOFTWARE;
        default:                                    return gpu::AdapterClass.OTHER;
    }
}

fn gpu::BackendVersion backend_version(uint version) {
    return {
        .major = (version >> 22) & 0x7Fu,
        .minor = (version >> 12) & 0x3FFu,
        .patch = version & 0xFFFu,
    };
}

fn gpu::AdapterMemoryInfo query_adapter_memory(vk::PhysicalDevice physical_device) {
    vk::PhysicalDeviceMemoryProperties properties;
    vk::get_physical_device_memory_properties(physical_device, &properties);
    bool[vk::MAX_MEMORY_HEAPS] host_visible_heaps;
    for (uint i = 0; i < properties.memory_type_count; i++) {
        vk::MemoryType* memory_type = &properties.memory_types[i];
        if (((uint)memory_type.property_flags
            & (uint)vk::MemoryPropertyFlagBits.HOST_VISIBLE_BIT) != 0) {
            host_visible_heaps[memory_type.heap_index] = true;
        }
    }

    gpu::AdapterMemoryInfo info;
    for (uint i = 0; i < properties.memory_heap_count; i++) {
        vk::MemoryHeap* heap = &properties.memory_heaps[i];
        if (((uint)heap.flags & (uint)vk::MemoryHeapFlagBits.DEVICE_LOCAL_BIT) != 0) {
            info.device_local_bytes += heap.size;
        }
        if (host_visible_heaps[i]) info.host_visible_bytes += heap.size;
    }
    return info;
}

fn gpu::AdapterQueueInfo query_adapter_queues(vk::PhysicalDevice physical_device) {
    uint count = 0;
    vk::get_physical_device_queue_family_properties(physical_device, &count, null);
    gpu::AdapterQueueInfo info;
    if (count == 0) return info;

    @pool() {
        vk::QueueFamilyProperties[] families =
            mem::talloc_array(vk::QueueFamilyProperties, (sz)count);
        vk::get_physical_device_queue_family_properties(
            physical_device,
            &count,
            families.ptr,
        );
        foreach (&family : families[:count]) {
            if (family.queue_count == 0) continue;
            uint flags = (uint)family.queue_flags;
            if ((flags & (uint)vk::QueueFlagBits.GRAPHICS_BIT) != 0) {
                info.available.graphics = true;
            }
            if ((flags & (uint)vk::QueueFlagBits.COMPUTE_BIT) != 0) {
                info.available.compute = true;
            }
            if ((flags & ((uint)vk::QueueFlagBits.GRAPHICS_BIT
                    | (uint)vk::QueueFlagBits.COMPUTE_BIT
                    | (uint)vk::QueueFlagBits.TRANSFER_BIT)) != 0) {
                info.available.transfer = true;
            }
        }
    };
    return info;
}

fn bool device_baseline_supported(
    vk::PhysicalDevice physical_device,
    VkDeviceConfig* config,
    ShaderHeapSupport* shader_heap,
) {
    if (config == null || shader_heap == null) return false;
    return candidate_satisfies(
        candidate:   physical_device,
        desc:        config,
        shader_heap: shader_heap,
    );
}

fn ShaderHeapSupport query_shader_heap_support(
    vk::PhysicalDevice physical_device,
) {
    vk::PhysicalDeviceVulkan12Features features12 =
        vk::physical_device_vulkan12features();
    vk::PhysicalDeviceFeatures2 features2 =
        vk::physical_device_features2().set_next(&features12);
    vk::get_physical_device_features2(physical_device, &features2);
    return {
        .indexing = heap_indexing_supported(
            &features2.features,
            &features12,
        ),
        .limits = query_indexing_heap_limits(physical_device),
    };
}

fn void initialize_adapter(
    VkAdapterState* adapter,
    vk::PhysicalDevice physical_device,
    VkDeviceConfig* config,
) {
    adapter.physical_device = physical_device;
    vk::get_physical_device_properties(physical_device, &adapter.properties);
    adapter.shader_heap = query_shader_heap_support(physical_device);
    adapter.device_baseline_supported = device_baseline_supported(
        physical_device,
        config,
        &adapter.shader_heap,
    );

    bool has_driver_properties = adapter.properties.api_version >= API_VERSION_1_2
        || device_supports_extension(physical_device, "VK_KHR_driver_properties");
    if (has_driver_properties) {
        adapter.driver_properties = vk::physical_device_driver_properties();
        vk::PhysicalDeviceProperties2 properties =
            vk::physical_device_properties2().set_next(&adapter.driver_properties);
        vk::get_physical_device_properties2(physical_device, &properties);
        adapter.properties = properties.properties;
    }

    vk::PhysicalDeviceLimits* limits = &adapter.properties.limits;
    adapter.info = {
        .name             = ((ZString)&adapter.properties.device_name[0]).str_view(),
        .vendor_id        = adapter.properties.vendor_id,
        .device_id        = adapter.properties.device_id,
        .device_class     = adapter_class(adapter.properties.device_type),
        .memory           = query_adapter_memory(physical_device),
        .queues           = query_adapter_queues(physical_device),
        .limits           = {
            .max_texture_dimension_2d  = limits.max_image_dimension2d,
            .max_texture_array_layers  = limits.max_image_array_layers,
            .max_color_attachments     = limits.max_color_attachments < gpu::MAX_COLOR_ATTACHMENTS
                ? limits.max_color_attachments : gpu::MAX_COLOR_ATTACHMENTS,
            .max_push_constant_size    = limits.max_push_constants_size,
            .max_compute_work_group_count = {
                limits.max_compute_work_group_count[0],
                limits.max_compute_work_group_count[1],
                limits.max_compute_work_group_count[2],
            },
            .max_draw_indirect_count = limits.max_draw_indirect_count,
        },
    };
    adapter.diagnostics = {
        .backend_name    = "Vulkan",
        .backend_version = backend_version(adapter.properties.api_version),
        .driver_name     = has_driver_properties
            ? ((ZString)&adapter.driver_properties.driver_name[0]).str_view()
            : "unknown",
        .driver_info     = has_driver_properties
            ? ((ZString)&adapter.driver_properties.driver_info[0]).str_view()
            : "",
        .driver_version  = adapter.properties.driver_version,
    };
}

fn bool? vk_adapter_supports_device_baseline(
    VkRuntimeState* state,
    uint adapter_index,
) {
    if (state == null || adapter_index >= state.adapters.len) {
        return gpu::INVALID_ARGUMENT~;
    }
    return state.adapters[adapter_index].device_baseline_supported;
}

fn void? enumerate_vk_adapters(VkRuntimeState* state) {
    uint count = 0;
    check_physical_device_enumeration_result(
        vk::enumerate_physical_devices(state.instance, &count, null),
    )!;
    if (count == 0) return;
    if (count > gpu::internal::MAX_ADAPTERS_PER_RUNTIME) return gpu::SLOT_TABLE_FULL~;

    vk::PhysicalDevice[] physical_devices =
        mem::new_array(vk::PhysicalDevice, (usz)count);
    defer mem::free(physical_devices.ptr);
    check_physical_device_enumeration_result(
        vk::enumerate_physical_devices(
            state.instance,
            &count,
            physical_devices.ptr,
        ),
    )!;

    VkDeviceConfig config = runtime_device_config(state);
    state.adapters = mem::new_array(VkAdapterState, (usz)count);
    foreach (i, physical_device : physical_devices[:count]) {
        initialize_adapter(&state.adapters[i], physical_device, &config);
    }
}

fn gpu::AdapterInfo? vk_get_adapter_info(VkRuntimeState* state, uint index) {
    if (state == null || index >= state.adapters.len) return gpu::INVALID_HANDLE~;
    return state.adapters[index].info;
}

fn gpu::AdapterDiagnostics? vk_get_adapter_diagnostics(
    VkRuntimeState* state,
    uint index,
) {
    if (state == null || index >= state.adapters.len) return gpu::INVALID_HANDLE~;
    return state.adapters[index].diagnostics;
}
