module gpu::vk @private;

import gpu @public;
import vk;
import vma;
import std::core::mem;
import std::atomic;
import std::atomic::types;
import std::thread;

// Disambiguates from vma::Allocator: std::core::mem's host-memory interface.
alias HostAllocator = std::core::mem::alloc::Allocator;
alias DeviceExtensionQueryFn = fn bool?(vk::PhysicalDevice physical_device, ZString extension);
alias DescriptorBufferCapacityProbeFn @private = fn bool (
    void*,
    vk::PhysicalDevice,
    QueueFamilies*,
    bool,
    uint,
    uint,
);
alias SemaphoreCounterQueryFn @private = fn vk::Result (
    vk::Device,
    vk::Semaphore,
    ulong*,
);

const ZString SWAPCHAIN_EXTENSION             = "VK_KHR_swapchain";
const ZString SWAPCHAIN_MAINTENANCE_EXTENSION = "VK_EXT_swapchain_maintenance1";
const ZString GENERATED_WORK_EXTENSION        = "VK_EXT_device_generated_commands";
const ZString GENERATED_WORK_USAGE_EXTENSION = "VK_KHR_maintenance5";
const uint GENERATED_WORK_TOKEN_COUNT  = 2;
const uint GENERATED_WORK_TOKEN_OFFSET = 16;
const uint GENERATED_WORK_RECORD_SIZE  = 40;

enum DeviceCreationCheckpoint : uint {
    NONE,
    OWNED_INSTANCE,
    LOGICAL_DEVICE,
    ALLOCATOR,
    RECORDING_CONTEXTS,
    DESCRIPTOR_HEAP,
    PIPELINE_SHARED,
    BEFORE_PUBLICATION,
}

struct StandaloneDeviceConfig {
    bool                      enable_validation;
    bool                      enable_debug_names;
    uint                      texture_heap_capacity;
    uint                      sampler_heap_capacity;
    uint                      texture_capacity;
    uint                      pipeline_capacity;
    char[]                    pipeline_cache_data;
    ZString                   application_name;
    gpu::DebugMessageCallback debug_callback;
    void*                     debug_user_data;
}

struct DeviceCreationProbe {
    DeviceCreationCheckpoint fail_at;
    DeviceCreationCheckpoint reached;
    uint                     rollback_count;
    uint                     owned_instance_teardown_count;
    bool                     descriptor_buffer_override;
    bool                     descriptor_buffer_capability_checked;
    bool                     descriptor_buffer_capability_available;
    bool                     descriptor_buffer_selected;
}

struct DescriptorBufferCapacityProbe {
    void*                           user_data;
    DescriptorBufferCapacityProbeFn fits;
}

struct VkPresentationState {
    bool           enabled;
    uint           family;
    uint           queue_index;
    vk::Queue      queue;
    Mutex          queue_mutex;
    SwapchainTable swapchains;
}

struct VkStrictHeapState {
    bool                    enabled;
    VkHeapImplementation    implementation;
    uint                    texture_capacity;
    uint                    sampler_capacity;
    DescriptorHeapState     descriptors;
}

struct VkDeviceState {
    gpu::Device                 owner;
    ulong                       resource_owner;
    vk::Instance               instance;
    VkInstanceDispatch         instance_dispatch;
    vk::DebugUtilsMessengerEXT messenger;
    bool                       owns_instance;
    vk::PhysicalDevice         physical_device;
    vk::Device                 device;
    VkDeviceDispatch           device_dispatch;
    gpu::QueueRoles            selected_queue_roles;
    vma::Allocator             allocator;
    uint                       graphics_family;
    uint                       graphics_queue_index;
    uint                       compute_family;
    uint                       compute_queue_index;
    uint                       transfer_family;
    uint                       transfer_queue_index;
    QueueFamilies              queues;
    vk::Queue                  graphics_queue;
    vk::Queue                  compute_queue;
    vk::Queue                  transfer_queue;
    QueueCompletionState[]     queue_completions;
    uint                       graphics_completion_id;
    uint                       compute_completion_id;
    uint                       transfer_completion_id;
    gpu::DeviceCaps            caps;
    ulong                      max_timeline_semaphore_value_difference;
    usz                        non_coherent_atom_size;
    bool                       validation;
    bool                       debug_names;
    gpu::DebugMessageCallback  debug_callback;
    void*                      debug_user_data;
    bool                       memory_budget_enabled;
    bool                       draw_indirect_count;
    GeneratedWorkSupport       generated_work;
    float                      max_sampler_anisotropy;
    BufferTable                buffers;
    AllocationTable            allocations;
    Mutex                      resource_mutex;
    Mutex                      pipeline_creation_mutex;
    Mutex                      texture_view_cache_mutex;
    Mutex                      command_mutex;
    bool                       compute_queue_distinct;
    RecordingContextTable      recording_contexts;
    CommandTable               commands;
    PipelineTable              pipelines;
    PipelineCacheTable         pipeline_cache;
    vk::PipelineLayout         graphics_layout;
    vk::IndirectCommandsLayoutEXT generated_draw_layout;
    vk::IndirectCommandsLayoutEXT generated_indexed_draw_layout;
    ComputeLayoutCache         compute_layout_cache;
    vk::PipelineCache          driver_cache;
    TextureTable               textures;
    SamplerTable               samplers;
    VkPresentationState        presentation;
    VkStrictHeapState          strict_heap;
    SubmittedCommandBatch[]    submitted_commands;
    uint                       submitted_command_count;
    uint                       submitted_command_reserved;
    ulong                      command_buffers_allocated;
    ulong                      command_buffers_freed;
    ulong                      pipeline_shader_modules_created;
    ulong                      pipelines_created;
    bool                       pipeline_creation_failure_enabled;
    usz                        pipeline_creation_failure_index;
    HostAllocator              host_allocator;
    // Advisory, best-effort across queue mutexes — mirrors get_memory_stats.
    SubmitStats                submit_stats;
}

struct PhysicalDeviceSelection {
    vk::PhysicalDevice physical_device;
    QueueFamilies      families;
}

struct PhysicalDeviceCandidate {
    PhysicalDeviceSelection selection;
    int                     score;
    bool                    suitable;
}

struct GeneratedWorkSupport {
    bool enabled;
    uint max_count;
}

struct CapsInputs {
    bool                 strict_enabled;
    bool                 presentation_enabled;
    bool                 async_compute;
    bool                 line_polygon_mode;
    uint                 texture_capacity;
    uint                 sampler_capacity;
    GeneratedWorkSupport generated_work;
}

fn bool presentation_maintenance_supported(
    bool extension_supported,
    uint feature_enabled,
) {
    return extension_supported && feature_enabled != 0;
}

fn bool device_supports_swapchain_maintenance(
    vk::PhysicalDevice physical_device,
) @private {
    if (!device_supports_extension(
        physical_device,
        SWAPCHAIN_MAINTENANCE_EXTENSION,
    )) return false;

    vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT features =
        vk::physical_device_swapchain_maintenance1features_ext();
    vk::PhysicalDeviceFeatures2 features2 =
        vk::physical_device_features2().set_next(&features);
    vk::get_physical_device_features2(physical_device, &features2);
    return presentation_maintenance_supported(
        true,
        features.swapchain_maintenance1,
    );
}

fn GeneratedWorkSupport resolve_generated_work_support(
    bool extension_supported,
    bool maintenance5_supported,
    vk::PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT* features,
    vk::PhysicalDeviceDeviceGeneratedCommandsPropertiesEXT* properties,
) {
    if (!extension_supported
        || !maintenance5_supported
        || features == null
        || properties == null
        || features.device_generated_commands == 0) return {};

    vk::ShaderStageFlags required_stages =
        vk::ShaderStageFlagBits.VERTEX_BIT
        | vk::ShaderStageFlagBits.FRAGMENT_BIT
        | vk::ShaderStageFlagBits.COMPUTE_BIT;
    if ((properties.supported_indirect_commands_shader_stages
            & required_stages) != required_stages
        || properties.max_indirect_sequence_count == 0
        || properties.max_indirect_commands_token_count
            < GENERATED_WORK_TOKEN_COUNT
        || properties.max_indirect_commands_token_offset
            < GENERATED_WORK_TOKEN_OFFSET
        || properties.max_indirect_commands_indirect_stride
            < GENERATED_WORK_RECORD_SIZE) return {};

    return {
        .enabled   = true,
        .max_count = properties.max_indirect_sequence_count,
    };
}

fn GeneratedWorkSupport query_generated_work_support(
    vk::PhysicalDevice physical_device,
    bool extension_supported,
    bool maintenance5_supported,
    vk::PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT* features,
) @private {
    if (!extension_supported || !maintenance5_supported) return {};
    vk::PhysicalDeviceDeviceGeneratedCommandsPropertiesEXT properties =
        vk::physical_device_device_generated_commands_properties_ext();
    vk::PhysicalDeviceProperties2 properties2 =
        vk::physical_device_properties2().set_next(&properties);
    vk::get_physical_device_properties2(physical_device, &properties2);
    return resolve_generated_work_support(
        extension_supported:    extension_supported,
        maintenance5_supported: maintenance5_supported,
        features:               features,
        properties:             &properties,
    );
}

fn vk::Result real_semaphore_counter_query(
    vk::Device device,
    vk::Semaphore semaphore,
    ulong* value,
) @private {
    return vk::get_semaphore_counter_value(device, semaphore, value);
}

fn void? reach_device_creation_checkpoint(
    DeviceCreationProbe* probe,
    DeviceCreationCheckpoint checkpoint,
) @private {
    if (probe == null) return;
    probe.reached = checkpoint;
    if (probe.fail_at == checkpoint) return gpu::BACKEND_ERROR~;
}

// One static vtable shared by every Vulkan device.
const gpu::BackendVTable VK_VTABLE @private = {
    .check_device_destroy    = &vk_check_device_destroy,
    .destroy_device          = &vk_destroy_device,
    .report_contract_failure = &vk_report_root_contract_failure,
    .get_memory_stats     = &vk_get_memory_stats,
    .build_memory_report  = &vk_build_memory_report,
    .allocate_memory        = &vk_allocate_memory,
    .free_allocation        = &vk_free_allocation,
    .get_allocation_info    = &vk_get_allocation_info,
    .get_allocation_span    = &vk_get_allocation_span,
    .get_span_mapping       = &vk_get_span_mapping,
    .get_span_address       = &vk_get_span_address,
    .flush_mapped_span      = &vk_flush_mapped_span,
    .invalidate_mapped_span = &vk_invalidate_mapped_span,
    .begin_commands      = &vk_begin_commands,
    .end_commands        = &vk_end_commands,
    .discard_commands    = &vk_discard_commands,
    .submit              = &vk_submit,
    .poll_completion     = &vk_poll_completion,
    .wait_completion     = &vk_wait_completion,
    .cmd_copy_buffer     = &vk_cmd_copy_buffer,
    .cmd_fill_buffer     = &vk_cmd_fill_buffer,
    .cmd_copy_buffer_to_texture = &vk_cmd_copy_buffer_to_texture,
    .cmd_copy_texture_to_buffer = &vk_cmd_copy_texture_to_buffer,
    .cmd_barrier         = &vk_cmd_barrier,
    .cmd_texture_barrier = &vk_cmd_texture_barrier,
    .get_texture_format_support = &vk_get_texture_format_support,
    .supports_texture_desc      = &vk_supports_texture_desc,
    .get_texture_requirements   = &vk_get_texture_requirements,
    .create_texture             = &vk_create_texture,
    .create_placed_texture      = &vk_create_placed_texture,
    .create_dedicated_texture   = &vk_create_dedicated_texture,
    .destroy_texture            = &vk_destroy_texture,
    .create_texture_view        = &vk_create_texture_view,
    .destroy_texture_view       = &vk_destroy_texture_view,
    .create_texture_views       = &vk_create_texture_views,
    .intern_sampler             = &vk_intern_sampler,
    .publish_sampler            = &vk_publish_sampler,
    .create_compute_pipeline   = &vk_create_compute_pipeline,
    .create_graphics_pipeline  = &vk_create_graphics_pipeline,
    .create_compute_pipelines  = &vk_create_compute_pipelines,
    .create_graphics_pipelines = &vk_create_graphics_pipelines,
    .destroy_pipeline          = &vk_destroy_pipeline,
    .get_pipeline_cache_size   = &vk_get_pipeline_cache_size,
    .get_pipeline_cache_data   = &vk_get_pipeline_cache_data,
    .cmd_bind_pipeline               = &vk_cmd_bind_pipeline,
    .cmd_set_depth_state             = &vk_cmd_set_depth_state,
    .cmd_dispatch                    = &vk_cmd_dispatch,
    .cmd_begin_render_pass           = &vk_cmd_begin_render_pass,
    .cmd_set_viewport                = &vk_cmd_set_viewport,
    .cmd_set_scissor                 = &vk_cmd_set_scissor,
    .cmd_end_render_pass             = &vk_cmd_end_render_pass,
    .cmd_draw                        = &vk_cmd_draw,
    .cmd_draw_indexed                = &vk_cmd_draw_indexed,
    .cmd_draw_indirect               = &vk_cmd_draw_indirect,
    .cmd_draw_indexed_indirect       = &vk_cmd_draw_indexed_indirect,
    .cmd_draw_indexed_indirect_count = &vk_cmd_draw_indexed_indirect_count,
    .cmd_dispatch_indirect           = &vk_cmd_dispatch_indirect,
    .cmd_draw_generated              = &vk_cmd_draw_generated,
    .cmd_draw_indexed_generated      = &vk_cmd_draw_indexed_generated,
    .cmd_dispatch_generated          = &vk_cmd_dispatch_generated,
    .cmd_begin_label                 = &vk_cmd_begin_label,
    .cmd_end_label                   = &vk_cmd_end_label,

    .create_swapchain        = &vk_create_swapchain,
    .destroy_swapchain       = &vk_destroy_swapchain,
    .resize_swapchain        = &vk_resize_swapchain,
    .get_swapchain_info      = &vk_get_swapchain_info,
    .acquire_next_image      = &vk_acquire_next_image,
    .present                 = &vk_present,
    .get_present_mode_support = &vk_get_present_mode_support,
};

fn gpu::Device? create_standalone_device_with_probe(
    StandaloneDeviceConfig* desc,
    DeviceCreationProbe* probe,
) {
    return create_vk_device_with_selection(
        desc:             desc,
        instance:         (vk::Instance)null,
        physical_device:  (vk::PhysicalDevice)null,
        owns_instance:    true,
        strict_requested: true,
        probe:            probe,
    );
}

fn gpu::Device? create_vk_device(
    gpu::Adapter* adapter,
    gpu::DeviceRequest* request,
    gpu::Runtime* retained_runtime,
) {
    return create_vk_device_with_probe(
        adapter:          adapter,
        request:          request,
        retained_runtime: retained_runtime,
        probe:            null,
    );
}

fn gpu::Device? create_vk_device_with_probe(
    gpu::Adapter* adapter,
    gpu::DeviceRequest* request,
    gpu::Runtime* retained_runtime,
    DeviceCreationProbe* probe,
) {
    gpu::validate_device_request(request)!;
    gpu::RuntimeData* runtime_data = gpu::resolve_adapter(adapter);
    if (runtime_data == null) return gpu::INVALID_HANDLE~;
    VkRuntimeState* runtime_state = (VkRuntimeState*)runtime_data.backend_state;
    if (runtime_state == null) return gpu::INVALID_HANDLE~;

    StandaloneDeviceConfig desc = {
        .enable_validation     = runtime_state.validation,
        .enable_debug_names    = runtime_state.debug_names,
        .texture_heap_capacity = runtime_state.texture_heap_capacity,
        .sampler_heap_capacity = runtime_state.sampler_heap_capacity,
        .texture_capacity      = runtime_state.texture_capacity,
        .pipeline_capacity     = runtime_state.pipeline_capacity,
        .pipeline_cache_data   = runtime_state.pipeline_cache_data,
        .debug_callback        = runtime_state.debug_callback,
        .debug_user_data       = runtime_state.debug_user_data,
    };
    VkAdapterState* adapter_state = &runtime_state.adapters[adapter.adapter_index];
    gpu::Surface requested_surface = gpu::requested_surface(*request);
    if (requested_surface.is_valid()
        && !runtime_state.instance_capabilities.surface_maintenance) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    gpu::QueueRequirements queue_requirements =
        gpu::requested_queue_requirements(*request);
    vk::SurfaceKHR presentation_surface = requested_surface.is_valid()
        ? (vk::SurfaceKHR)gpu::surface_backend_handle(&requested_surface)!
        : (vk::SurfaceKHR)null;
    return create_vk_device_with_selection(
        desc:                       &desc,
        instance:                   runtime_state.instance,
        physical_device:            adapter_state.physical_device,
        owns_instance:              false,
        strict_requested:           gpu::requests_strict(*request),
        retained_runtime:           retained_runtime,
        adapter:                    adapter,
        runtime_state:              runtime_state,
        presentation_surface:       presentation_surface,
        presentation_surface_owner: requested_surface.is_valid()
            ? &requested_surface
            : null,
        queue_requirements:         &queue_requirements,
        probe:                      probe,
    );
}

fn gpu::Device? create_vk_device_with_selection(
    StandaloneDeviceConfig* desc,
    vk::Instance instance,
    vk::PhysicalDevice physical_device,
    bool owns_instance,
    bool strict_requested,
    gpu::Runtime* retained_runtime = null,
    gpu::Adapter* adapter = null,
    VkRuntimeState* runtime_state = null,
    vk::SurfaceKHR presentation_surface = (vk::SurfaceKHR)null,
    gpu::Surface* presentation_surface_owner = null,
    gpu::QueueRequirements* queue_requirements = null,
    DeviceCreationProbe* probe = null,
) @private {
    VkDeviceState* state = mem::new(VkDeviceState);
    state.resource_owner = gpu::allocate_device_resource_owner()!;
    defer catch {
        destroy_state(state, probe);
        if (probe != null) probe.rollback_count++;
    }
    state.owns_instance = owns_instance;
    state.host_allocator = &allocators::LIBC_ALLOCATOR;
    state.debug_callback = desc.debug_callback;
    state.debug_user_data = desc.debug_user_data;
    state.debug_names = desc.enable_debug_names;
    state.resource_mutex.init()!;
    state.pipeline_creation_mutex.init()!;
    state.texture_view_cache_mutex.init()!;
    state.command_mutex.init()!;

    state.strict_heap = resolve_strict_heap_state(strict_requested, desc)!;

    state.validation = desc.enable_validation;
    state.presentation.enabled = presentation_surface != null;
    if (state.presentation.enabled) state.presentation.queue_mutex.init()!;
    bool debug_utils_enabled;
    QueueFamilies families;
    if (owns_instance) {
        vk::DebugUtilsMessengerCreateInfoEXT messenger_info =
            debug_messenger_create_info(state);
        VkInstanceDesc instance_desc = {
            .enable_validation      = desc.enable_validation,
            .enable_debug_utils     = desc.enable_validation
                || desc.enable_debug_names
                || desc.debug_callback != null,
            .application_name       = desc.application_name,
            .debug_messenger_info   = desc.enable_validation
                ? &messenger_info
                : null,
        };
        VkInstanceCreateResult instance_result =
            create_vk_instance(&instance_desc)!;
        state.instance = instance_result.instance;
        debug_utils_enabled = instance_result.capabilities.debug_utils;
        VkInstanceDispatchPlan dispatch_plan = {
            .debug = debug_utils_enabled,
        };
        state.instance_dispatch = load_vk_instance_dispatch(
            state.instance,
            &vk::get_instance_proc_addr,
            dispatch_plan,
        );
        if (should_create_debug_messenger(
            instance_result.capabilities,
            state.validation || state.debug_callback != null,
        )) {
            setup_debug_messenger(state);
        } else if (should_report_debug_messenger_unavailable(
            instance_result.capabilities,
            state.validation || state.debug_callback != null,
        )) {
            report_debug_messenger_failure(state, "debug-utils messenger unavailable");
        }

        PhysicalDeviceSelection selection = pick_physical_device(state.instance, desc)!;
        state.physical_device = selection.physical_device;
        families = selection.families;
        reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.OWNED_INSTANCE)!;
    } else {
        if (instance == null || physical_device == null || runtime_state == null) return gpu::INVALID_ARGUMENT~;
        state.instance = instance;
        debug_utils_enabled =
            runtime_state.instance_capabilities.debug_utils;
        if (state.presentation.enabled) {
            state.instance_dispatch.surface = runtime_state.dispatch.surface;
        }
        state.physical_device = physical_device;
        families = presentation_surface == null
            ? select_queue_families(
                physical_device: state.physical_device,
                requirements:    queue_requirements,
            )!
            : select_queue_families_for_surface(
                runtime:         runtime_state,
                physical_device: state.physical_device,
                surface:         presentation_surface,
                requirements:    queue_requirements,
            )!;
    }

    bool swapchain_maintenance_extension = state.presentation.enabled
        && device_supports_extension(
            state.physical_device,
            SWAPCHAIN_MAINTENANCE_EXTENSION,
        );
    bool generated_work_extension = state.strict_heap.enabled
        && device_supports_extension(
            state.physical_device,
            GENERATED_WORK_EXTENSION,
        );
    bool generated_work_usage_extension = generated_work_extension
        && device_supports_extension(
            state.physical_device,
            GENERATED_WORK_USAGE_EXTENSION,
        );
    vk::PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT generated_work_features =
        vk::physical_device_device_generated_commands_features_ext();
    vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5_features =
        vk::physical_device_maintenance5features_khr();
    vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT presentation_features =
        vk::physical_device_swapchain_maintenance1features_ext();
    vk::PhysicalDeviceVulkan13Features features13 =
        vk::physical_device_vulkan13features();
    if (generated_work_usage_extension) {
        maintenance5_features = maintenance5_features.set_next(&generated_work_features);
        features13 = features13.set_next(&maintenance5_features);
    }
    if (swapchain_maintenance_extension) {
        presentation_features = presentation_features.set_next(
            features13.p_next,
        );
        features13 = features13.set_next(&presentation_features);
    }
    vk::PhysicalDeviceVulkan12Features features12 =
        vk::physical_device_vulkan12features().set_next(&features13);
    vk::PhysicalDeviceVulkan11Features features11 =
        vk::physical_device_vulkan11features().set_next(&features12);
    vk::PhysicalDeviceFeatures2 features2 =
        vk::physical_device_features2().set_next(&features11);
    vk::get_physical_device_features2(state.physical_device, &features2);
    check_required_features(
        base: &features2.features,
        v11:  &features11,
        v12:  &features12,
        v13:  &features13,
    )!;
    if (state.presentation.enabled
        && !presentation_maintenance_supported(
            swapchain_maintenance_extension,
            presentation_features.swapchain_maintenance1,
        )) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    state.draw_indirect_count = features12.draw_indirect_count != 0;
    state.generated_work = query_generated_work_support(
        physical_device:        state.physical_device,
        extension_supported:    generated_work_extension,
        maintenance5_supported: generated_work_usage_extension
            && maintenance5_features.maintenance5 != 0,
        features:               &generated_work_features,
    );
    bool line_polygon_mode = line_polygon_mode_supported(&features2.features);

    if (features2.features.sampler_anisotropy != 0) {
        vk::PhysicalDeviceProperties anisotropy_props;
        vk::get_physical_device_properties(state.physical_device, &anisotropy_props);
        state.max_sampler_anisotropy = anisotropy_props.limits.max_sampler_anisotropy;
    }
    if (state.strict_heap.enabled) {
        bool indexing_supported = heap_indexing_supported(&features2.features, &features12);
        IndexingHeapLimits indexing_limits = query_indexing_heap_limits(state.physical_device);
        bool buffer_supported;
        if (strict_heap_needs_descriptor_buffer_query(
            state:              &state.strict_heap,
            indexing_supported: indexing_supported,
            indexing_limits:    indexing_limits,
            probe:              probe,
        ) && heap_shared_features_supported(&features2.features, &features12)) {
            buffer_supported = query_descriptor_buffer_support(state.physical_device, state)!;
        }
        select_strict_heap_implementation_for_creation(
            state:              &state.strict_heap,
            indexing_supported: indexing_supported,
            buffer_supported:   buffer_supported,
            indexing_limits:    indexing_limits,
            probe:              probe,
        )!;
    }

    state.selected_queue_roles = families.selected;
    if (families.selected.graphics) {
        state.graphics_family = families.graphics;
        state.graphics_queue_index = families.graphics_index;
    }
    if (families.selected.compute) {
        state.compute_family = families.compute;
        state.compute_queue_index = families.compute_index;
    }
    if (families.selected.transfer) {
        state.transfer_family = families.transfer;
        state.transfer_queue_index = families.transfer_index;
    }
    state.queues = families;
    if (state.presentation.enabled) {
        state.presentation.family = families.present;
        state.presentation.queue_index = families.present_index;
    }

    create_logical_device(state, &families, line_polygon_mode)!;
    reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.LOGICAL_DEVICE)!;
    VkDeviceDispatchPlan dispatch_plan = {
        .debug             = debug_utils_enabled,
        .presentation      = state.presentation.enabled,
        .descriptor_buffer = strict_heap_uses_descriptor_buffer(&state.strict_heap),
        .generated_work    = state.generated_work.enabled,
    };
    state.device_dispatch = load_vk_device_dispatch(
        state.device,
        &vk::get_device_proc_addr,
        dispatch_plan,
    );
    if (state.presentation.enabled
        && (!instance_dispatch_supports_presentation(&state.instance_dispatch)
            || !device_dispatch_supports_presentation(&state.device_dispatch))) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    if (strict_heap_uses_descriptor_buffer(&state.strict_heap)
        && !device_dispatch_supports_descriptor_buffer(&state.device_dispatch)) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    if (state.generated_work.enabled
        && !device_dispatch_supports_generated_work(&state.device_dispatch)) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    if (families.selected.graphics) {
        vk::get_device_queue(
            device:             state.device,
            queue_family_index: state.graphics_family,
            queue_index:        state.graphics_queue_index,
            p_queue:            &state.graphics_queue,
        );
    }
    state.compute_queue_distinct = families.async_compute;
    if (families.selected.compute) {
        if (graphics_compute_alias(families)) {
            state.compute_queue = state.graphics_queue;
        } else {
            vk::get_device_queue(
                device:             state.device,
                queue_family_index: families.compute,
                queue_index:        families.compute_index,
                p_queue:            &state.compute_queue,
            );
        }
    }
    if (families.selected.transfer) {
        if (graphics_transfer_alias(families)) {
            state.transfer_queue = state.graphics_queue;
        } else if (compute_transfer_alias(families)) {
            state.transfer_queue = state.compute_queue;
        } else {
            vk::get_device_queue(
                device:             state.device,
                queue_family_index: state.transfer_family,
                queue_index:        state.transfer_queue_index,
                p_queue:            &state.transfer_queue,
            );
        }
    }
    if (state.presentation.enabled) {
        vk::get_device_queue(
            device:             state.device,
            queue_family_index: state.presentation.family,
            queue_index:        state.presentation.queue_index,
            p_queue:            &state.presentation.queue,
        );
    }

    create_vk_allocator(state)!;
    reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.ALLOCATOR)!;
    state.buffers.owner = gpu::resource_owner_tag(state.resource_owner, gpu::ResourceOwnerKind.BUFFER);
    state.buffers.slots = mem::new_array(BufferCell, MAX_BUFFERS);
    state.allocations.owner = gpu::resource_owner_tag(
        state.resource_owner,
        gpu::ResourceOwnerKind.ALLOCATION,
    );
    state.allocations.slots = mem::new_array(
        AllocationCell,
        ALLOCATION_CAPACITY,
    );

    state.recording_contexts.slots = mem::new_array(
        RecordingContextCell,
        MAX_RECORDING_CONTEXTS,
    );
    state.commands.owner = gpu::resource_owner_tag(
        state.resource_owner,
        gpu::ResourceOwnerKind.COMMAND,
    );
    state.commands.slots = mem::new_array(CommandCell, MAX_COMMAND_LISTS);
    reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.RECORDING_CONTEXTS)!;
    uint pipeline_capacity;
    if (state.strict_heap.enabled) {
        pipeline_capacity = resolve_descriptor_capacity(desc.pipeline_capacity, gpu::MAX_PIPELINES)!;
        state.pipelines.owner = gpu::resource_owner_tag(state.resource_owner, gpu::ResourceOwnerKind.PIPELINE);
        state.pipelines.slots = mem::new_array(PipelineCell, pipeline_capacity);
    }
    state.textures.owner = gpu::resource_owner_tag(state.resource_owner, gpu::ResourceOwnerKind.TEXTURE);
    state.textures.slots = mem::new_array(TextureCell, resolve_descriptor_capacity(desc.texture_capacity, gpu::DEFAULT_TEXTURE_CAPACITY)!);
    create_sampler_table(state)!;
    if (state.presentation.enabled) {
        state.presentation.swapchains.owner = gpu::resource_owner_tag(state.resource_owner, gpu::ResourceOwnerKind.SWAPCHAIN);
        state.presentation.swapchains.slots = mem::new_array(SwapchainCell, gpu::MAX_SWAPCHAINS);
    }
    if (state.strict_heap.enabled) {
        create_descriptor_heap(state)!;
        reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.DESCRIPTOR_HEAP)!;
        create_pipeline_shared(state, desc.pipeline_cache_data, pipeline_capacity)!;
        reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.PIPELINE_SHARED)!;
    }
    CapsInputs caps_inputs = {
        .strict_enabled       = state.strict_heap.enabled,
        .presentation_enabled = state.presentation.enabled,
        .async_compute        = state.compute_queue_distinct,
        .line_polygon_mode    = line_polygon_mode,
        .texture_capacity     = state.strict_heap.texture_capacity,
        .sampler_capacity     = state.strict_heap.sampler_capacity,
        .generated_work      = state.generated_work,
    };
    state.caps = build_caps(
        base:                    &features2.features,
        v12:                     &features12,
        v13:                     &features13,
        physical_device:         state.physical_device,
        inputs:                  &caps_inputs,
        max_timeline_difference: &state.max_timeline_semaphore_value_difference,
        non_coherent_atom_size:  &state.non_coherent_atom_size,
    );

    gpu::QueueRequirements requirements = queue_requirements == null
        ? gpu::default_queue_requirements()
        : *queue_requirements;
    gpu::QueueSelection queue_selection =
        gpu::queue_selection_from_native_ids(
            requirements: requirements,
            graphics:     families.graphics_ids[:families.counts.graphics],
            compute:      families.compute_ids[:families.counts.compute],
            transfer:     families.transfer_ids[:families.counts.transfer],
        )!;
    state.caps.queues = queue_selection.counts;
    create_queue_completions(state, &queue_selection)!;

    reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.BEFORE_PUBLICATION)!;
    return gpu::register_device(
        backend:              gpu::BackendKind.VULKAN,
        caps:                 state.caps,
        vtable:               &VK_VTABLE,
        backend_state:        (void*)state,
        resource_owner:       state.resource_owner,
        retained_runtime:     retained_runtime,
        adapter:              adapter,
        presentation_surface: presentation_surface_owner,
        queues:               queue_selection,
        owner:                &state.owner,
    );
}

fn bool? device_destroy_ready_with_query(
    VkDeviceState* state,
    SemaphoreCounterQueryFn query_fn,
) @private {
    foreach (&completion : state.queue_completions) {
        ulong expected = completion.published_sequence.load(
            AtomicOrdering.ACQUIRE,
        );
        if (expected == 0) continue;

        ulong current;
        vk::Result result = query_fn(
            state.device,
            completion.timeline,
            &current,
        );
        if (result == vk::Result.ERROR_DEVICE_LOST) {
            observe_device_result(state, result);
            return true;
        }
        check_device_result(state, result)!;
        if (current < expected) return false;
    }

    return true;
}

fn fault vk_report_root_contract_failure(
    void* backend_state,
    ZString operation,
    fault public_fault,
    gpu::DebugResourceKind resource_kind,
    ZString rejected_field,
    ZString invariant,
) @private {
    return report_contract_failure(
        state:               (VkDeviceState*)backend_state,
        operation:           operation,
        public_fault:        public_fault,
        resource_kind:       resource_kind,
        has_public_identity: false,
        index:               0,
        generation:          0,
        debug_name:          null,
        rejected_field:      rejected_field,
        invariant:           invariant,
        backend_text:        null,
    );
}

fn void? vk_check_device_destroy(void* backend_state) @private {
    VkDeviceState* state = (VkDeviceState*)backend_state;
    if (!device_destroy_ready_with_query(state, &real_semaphore_counter_query)!) {
        return gpu::DEVICE_BUSY~;
    }
}

fn void vk_destroy_device(void* backend_state) @private {
    destroy_state((VkDeviceState*)backend_state);
}

<* Reverse-order cleanup after accepted destruction, device loss, or partial creation. *>
fn void destroy_state(VkDeviceState* state, DeviceCreationProbe* probe = null) @private {
    if (state == null) return;
    if (state.device != null) {
        destroy_submitted_commands(state);
        // Report before cleanup clears the diagnostic tables.
        if (state.validation || state.debug_callback != null) (void)report_resource_leaks(state);
        // Internal buffers and device-loss leftovers remain backend-owned.
        if (state.allocator != null && state.buffers.slots.ptr != null) {
            foreach (&cell : state.buffers.slots[:state.buffers.next_new]) {
                if (cell.used) state.allocator.destroy_buffer(cell.value.buffer, cell.value.allocation);
            }
        }
        if (state.buffers.slots.ptr != null) {
            mem::free(state.buffers.slots.ptr);
            state.buffers.slots = {};
        }
        // Swapchains before textures: their wrapped image slots free here.
        if (state.presentation.swapchains.slots.ptr != null) {
            foreach (&cell : state.presentation.swapchains.slots[:state.presentation.swapchains.next_new]) {
                if (cell.used) destroy_swapchain_slot(state, &cell.value);
            }
            mem::free(state.presentation.swapchains.slots.ptr);
            state.presentation.swapchains.slots = {};
        }
        if (state.validation || state.debug_callback != null) report_descriptor_leaks(state);
        destroy_descriptor_heap(state);
        destroy_sampler_table(state);
        // Internal textures and device-loss leftovers remain backend-owned.
        if (state.textures.slots.ptr != null) {
            foreach (&cell : state.textures.slots[:state.textures.next_new]) {
                if (cell.used) {
                    destroy_cached_views(state, &cell.value);
                    vk::destroy_image_view(
                        state.device,
                        cell.value.default_view,
                        null,
                    );
                    if (cell.value.placed) {
                        vk::destroy_image(state.device, cell.value.image, null);
                    } else if (state.allocator != null) {
                        state.allocator.destroy_image(
                            cell.value.image,
                            cell.value.allocation,
                        );
                    }
                }
            }
            mem::free(state.textures.slots.ptr);
            state.textures.slots = {};
        }
        destroy_allocation_table(state);
        // Cache entries own the pipelines; slots only alias them.
        if (state.pipelines.slots.ptr != null) {
            mem::free(state.pipelines.slots.ptr);
            state.pipelines.slots = {};
        }
        destroy_pipeline_shared(state);
        destroy_command_table(state);
        if (state.recording_contexts.slots.ptr != null) {
            foreach (&cell : state.recording_contexts.slots[:state.recording_contexts.next_new]) {
                if (cell.used) destroy_context_pools(state, &cell.value);
            }
            mem::free(state.recording_contexts.slots.ptr);
            state.recording_contexts.slots = {};
        }
        destroy_queue_completions(state);
        if (state.allocator != null) {
            state.allocator.destroy();
            state.allocator = (vma::Allocator)null;
        }
        vk::destroy_device(state.device, null);
        state.device = (vk::Device)null;
    }
    destroy_debug_messenger(state);
    if (state.owns_instance && state.instance != null) {
        vk::destroy_instance(state.instance, null);
        if (probe != null) probe.owned_instance_teardown_count++;
    }
    state.instance = (vk::Instance)null;
    if (state.resource_mutex.is_initialized())       (void)state.resource_mutex.destroy();
    if (state.pipeline_creation_mutex.is_initialized()) (void)state.pipeline_creation_mutex.destroy();
    if (state.texture_view_cache_mutex.is_initialized()) (void)state.texture_view_cache_mutex.destroy();
    if (state.command_mutex.is_initialized())        (void)state.command_mutex.destroy();
    if (state.presentation.queue_mutex.is_initialized()) (void)state.presentation.queue_mutex.destroy();
    mem::free(state);
}

<* Gate on the required base/1.1/1.2/1.3 features. Pure, so it is unit-testable
   without a device. shaderInt64 backs the root-pointer shader ABI's 64-bit
   address scalars (SPIR-V Int64 capability); multiDrawIndirect and
   shaderDrawParameters back multi-draw indirect execution and the gl_DrawID
   per-draw indexing convention. *>
fn void? check_required_features(vk::PhysicalDeviceFeatures* base, vk::PhysicalDeviceVulkan11Features* v11, vk::PhysicalDeviceVulkan12Features* v12, vk::PhysicalDeviceVulkan13Features* v13) {
    if (base.shader_int64 == 0)         return gpu::UNSUPPORTED_FEATURE~;
    if (base.multi_draw_indirect == 0)  return gpu::UNSUPPORTED_FEATURE~;
    if (v11.shader_draw_parameters == 0) return gpu::UNSUPPORTED_FEATURE~;
    if (v12.buffer_device_address == 0) return gpu::UNSUPPORTED_FEATURE~;
    if (v12.timeline_semaphore == 0)    return gpu::UNSUPPORTED_FEATURE~;
    if (v13.synchronization2 == 0)      return gpu::UNSUPPORTED_FEATURE~;
    if (v13.dynamic_rendering == 0)     return gpu::UNSUPPORTED_FEATURE~;
}

<* Optional non-solid polygon rasterization support. Pure. *>
fn bool line_polygon_mode_supported(vk::PhysicalDeviceFeatures* base) {
    return base.fill_mode_non_solid != 0;
}

<* Features both descriptor paths need — SPIR-V capabilities emitted by the
   heap-access shader helpers. Pure. *>
fn bool heap_shared_features_supported(vk::PhysicalDeviceFeatures* base, vk::PhysicalDeviceVulkan12Features* v12) {
    return base.shader_storage_image_read_without_format != 0
        && base.shader_storage_image_write_without_format != 0
        && v12.runtime_descriptor_array != 0
        && v12.shader_sampled_image_array_non_uniform_indexing != 0
        && v12.shader_storage_image_array_non_uniform_indexing != 0;
}

<* Shared set plus the update-after-bind bits the set path needs. Pure. *>
fn bool heap_indexing_supported(vk::PhysicalDeviceFeatures* base, vk::PhysicalDeviceVulkan12Features* v12) {
    return heap_shared_features_supported(base, v12)
        && v12.descriptor_binding_partially_bound != 0
        && v12.descriptor_binding_sampled_image_update_after_bind != 0
        && v12.descriptor_binding_storage_image_update_after_bind != 0
        && v12.descriptor_binding_update_unused_while_pending != 0;
}

fn VkStrictHeapState? resolve_strict_heap_state(bool requested, StandaloneDeviceConfig* desc) {
    if (desc == null) return gpu::INVALID_ARGUMENT~;
    VkStrictHeapState state;
    if (!requested) return state;
    state.enabled = true;
    state.texture_capacity = resolve_descriptor_capacity(
        desc.texture_heap_capacity,
        gpu::DEFAULT_TEXTURE_HEAP_CAPACITY,
    )!;
    state.sampler_capacity = resolve_descriptor_capacity(
        desc.sampler_heap_capacity,
        gpu::DEFAULT_SAMPLER_HEAP_CAPACITY,
    )!;
    return state;
}

fn bool strict_heap_uses_descriptor_indexing(VkStrictHeapState* state) {
    return state != null
        && state.enabled
        && state.implementation == VkHeapImplementation.DESCRIPTOR_INDEXING;
}

fn bool strict_heap_uses_descriptor_buffer(VkStrictHeapState* state) {
    return state != null
        && state.enabled
        && state.implementation == VkHeapImplementation.DESCRIPTOR_BUFFER;
}

fn bool strict_heap_needs_descriptor_buffer_query(
    VkStrictHeapState* state,
    bool indexing_supported,
    IndexingHeapLimits indexing_limits,
    DeviceCreationProbe* probe,
) {
    if (state == null || !state.enabled) return false;
    if (probe != null && probe.descriptor_buffer_override) return true;
    return !indexing_supported || !indexing_heap_fits_limits(
        state.texture_capacity,
        state.sampler_capacity,
        indexing_limits,
    );
}

fn void? select_strict_heap_implementation_for_creation(
    VkStrictHeapState* state,
    bool indexing_supported,
    bool buffer_supported,
    IndexingHeapLimits indexing_limits,
    DeviceCreationProbe* probe,
) {
    if (probe == null || !probe.descriptor_buffer_override) {
        return select_strict_heap_implementation(
            state:              state,
            indexing_supported: indexing_supported,
            buffer_supported:   buffer_supported,
            indexing_limits:    indexing_limits,
        );
    }
    if (state == null) return gpu::INVALID_ARGUMENT~;
    if (!state.enabled) return;
    probe.descriptor_buffer_capability_checked   = true;
    probe.descriptor_buffer_capability_available = buffer_supported;
    probe.descriptor_buffer_selected             = false;
    if (!buffer_supported) return gpu::UNSUPPORTED_FEATURE~;
    state.implementation = VkHeapImplementation.DESCRIPTOR_BUFFER;
    probe.descriptor_buffer_selected = true;
}

fn bool? query_device_extension_support_mapped(
    VkDeviceState* state,
    vk::PhysicalDevice physical_device,
    ZString extension,
    DeviceExtensionQueryFn extension_query,
) {
    bool? supported = extension_query(physical_device, extension);
    if (catch query_fault = supported) {
        vk::Result result;
        if (query_fault == vk::ERROR_OUT_OF_HOST_MEMORY) {
            result = vk::Result.ERROR_OUT_OF_HOST_MEMORY;
        } else if (query_fault == vk::ERROR_OUT_OF_DEVICE_MEMORY) {
            result = vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
        } else if (query_fault == vk::ERROR_LAYER_NOT_PRESENT) {
            result = vk::Result.ERROR_LAYER_NOT_PRESENT;
        } else {
            return gpu::BACKEND_ERROR~;
        }
        check_backend_result(state, "query_device_extension_support", result)!;
        return false;
    }
    return supported;
}

<* The feature struct is only chained once the extension is confirmed. *>
fn bool? query_descriptor_buffer_support_with_query(
    vk::PhysicalDevice physical_device,
    DeviceExtensionQueryFn extension_query,
    VkDeviceState* state = null,
) {
    if (!query_device_extension_support_mapped(
        state:           state,
        physical_device: physical_device,
        extension:       "VK_EXT_descriptor_buffer",
        extension_query: extension_query,
    )!) return false;
    vk::PhysicalDeviceDescriptorBufferFeaturesEXT db_features = vk::physical_device_descriptor_buffer_features_ext();
    vk::PhysicalDeviceFeatures2 features2 = vk::physical_device_features2().set_next(&db_features);
    vk::get_physical_device_features2(physical_device, &features2);
    return db_features.descriptor_buffer != 0;
}

fn bool? query_descriptor_buffer_support(
    vk::PhysicalDevice physical_device,
    VkDeviceState* state = null,
) @private {
    return query_descriptor_buffer_support_with_query(
        physical_device,
        &query_device_extension_support,
        state,
    );
}

fn IndexingHeapLimits query_indexing_heap_limits(vk::PhysicalDevice physical_device) @private {
    vk::PhysicalDeviceVulkan12Properties props = vk::physical_device_vulkan12properties();
    vk::PhysicalDeviceProperties2 props2 = vk::physical_device_properties2().set_next(&props);
    vk::get_physical_device_properties2(physical_device, &props2);
    return {
        .max_sampled_images           = props.max_descriptor_set_update_after_bind_sampled_images,
        .max_storage_images           = props.max_descriptor_set_update_after_bind_storage_images,
        .max_samplers                 = props.max_descriptor_set_update_after_bind_samplers,
        .max_per_stage_sampled_images = props.max_per_stage_descriptor_update_after_bind_sampled_images,
        .max_per_stage_storage_images = props.max_per_stage_descriptor_update_after_bind_storage_images,
        .max_per_stage_samplers       = props.max_per_stage_descriptor_update_after_bind_samplers,
        .max_per_stage_resources      = props.max_per_stage_update_after_bind_resources,
        .max_all_pools_descriptors    = props.max_update_after_bind_descriptors_in_all_pools,
    };
}

fn bool descriptor_buffer_capacity_probe_real(
    void* user_data,
    vk::PhysicalDevice physical_device,
    QueueFamilies* families,
    bool draw_indirect_count,
    uint texture_capacity,
    uint sampler_capacity,
) @private {
    return descriptor_buffer_candidate_fits(
        physical_device:     physical_device,
        families:            families,
        draw_indirect_count: draw_indirect_count,
        texture_capacity:    texture_capacity,
        sampler_capacity:    sampler_capacity,
    );
}

fn DescriptorBufferCapacityProbe real_descriptor_buffer_capacity_probe() @private {
    return { .fits = &descriptor_buffer_capacity_probe_real };
}

fn VkHeapImplementation? resolve_candidate_heap_implementation_with_probe(
    bool indexing_supported,
    bool descriptor_buffer_supported,
    uint texture_capacity,
    uint sampler_capacity,
    IndexingHeapLimits indexing_limits,
    vk::PhysicalDevice physical_device,
    QueueFamilies* families,
    bool draw_indirect_count,
    DescriptorBufferCapacityProbe* capacity_probe,
) {
    bool indexing_usable = indexing_supported && indexing_heap_fits_limits(
        texture_capacity,
        sampler_capacity,
        indexing_limits,
    );
    bool descriptor_buffer_capacity_usable = false;
    if (!indexing_usable && descriptor_buffer_supported) {
        if (capacity_probe == null || capacity_probe.fits == null) {
            return gpu::INVALID_ARGUMENT~;
        }
        descriptor_buffer_capacity_usable = capacity_probe.fits(
            capacity_probe.user_data,
            physical_device,
            families,
            draw_indirect_count,
            texture_capacity,
            sampler_capacity,
        );
    }
    return resolve_heap_implementation(
        indexing_usable,
        descriptor_buffer_supported && descriptor_buffer_capacity_usable,
    );
}

fn bool candidate_satisfies(
    vk::PhysicalDevice candidate,
    StandaloneDeviceConfig* desc,
    QueueFamilies* families,
) @private {
    vk::PhysicalDeviceProperties props;
    vk::get_physical_device_properties(candidate, &props);
    if (props.api_version < API_VERSION_1_3) return false;

    vk::PhysicalDeviceVulkan13Features features13 = vk::physical_device_vulkan13features();
    vk::PhysicalDeviceVulkan12Features features12 = vk::physical_device_vulkan12features().set_next(&features13);
    vk::PhysicalDeviceVulkan11Features features11 = vk::physical_device_vulkan11features().set_next(&features12);
    vk::PhysicalDeviceFeatures2 features2 = vk::physical_device_features2().set_next(&features11);
    vk::get_physical_device_features2(candidate, &features2);
    if (catch check_required_features(
        base: &features2.features,
        v11:  &features11,
        v12:  &features12,
        v13:  &features13,
    )) return false;

    uint? resolved_tex = resolve_descriptor_capacity(
        desc.texture_heap_capacity,
        gpu::DEFAULT_TEXTURE_HEAP_CAPACITY,
    );
    if (catch capacity_fault = resolved_tex) return false;
    uint? resolved_smp = resolve_descriptor_capacity(
        desc.sampler_heap_capacity,
        gpu::DEFAULT_SAMPLER_HEAP_CAPACITY,
    );
    if (catch capacity_fault = resolved_smp) return false;

    bool indexing_supported =
        heap_indexing_supported(&features2.features, &features12);
    bool descriptor_buffer_supported;
    if (heap_shared_features_supported(&features2.features, &features12)) {
        if (try supported = query_descriptor_buffer_support(candidate)) {
            descriptor_buffer_supported = supported;
        }
    }
    IndexingHeapLimits indexing_limits = query_indexing_heap_limits(candidate);
    DescriptorBufferCapacityProbe capacity_probe =
        real_descriptor_buffer_capacity_probe();
    if (catch implementation_fault = resolve_candidate_heap_implementation_with_probe(
        indexing_supported:          indexing_supported,
        descriptor_buffer_supported: descriptor_buffer_supported,
        texture_capacity:            resolved_tex,
        sampler_capacity:            resolved_smp,
        indexing_limits:             indexing_limits,
        physical_device:             candidate,
        families:                    families,
        draw_indirect_count:         features12.draw_indirect_count != 0,
        capacity_probe:              &capacity_probe,
    )) return false;
    return true;
}

<* Select the highest-scored suitable candidate, preserving enumeration order
   for equal scores. Pure. *>
fn PhysicalDeviceSelection? choose_physical_device(PhysicalDeviceCandidate[] candidates) {
    PhysicalDeviceSelection best;
    int best_score = -1;
    bool found = false;
    foreach (&candidate : candidates) {
        if (!candidate.suitable || (found && candidate.score <= best_score)) continue;
        best = candidate.selection;
        best_score = candidate.score;
        found = true;
    }
    if (!found) return gpu::UNSUPPORTED_FEATURE~;
    return best;
}

<* Enumerate physical devices, drop candidates that cannot satisfy the desc
   (API version, required features, semantic heap support, conditional presentation
   capability, queue topology), and return the highest-scored device with its
   already-resolved queue families. *>
fn PhysicalDeviceSelection? pick_physical_device(vk::Instance instance, StandaloneDeviceConfig* desc) {
    uint count = 0;
    check_physical_device_enumeration_result(vk::enumerate_physical_devices(instance, &count, null))!;
    if (count == 0) return gpu::UNSUPPORTED_FEATURE~;

    @pool() {
        vk::PhysicalDevice[] devices = mem::talloc_array(vk::PhysicalDevice, (sz)count);
        check_physical_device_enumeration_result(vk::enumerate_physical_devices(instance, &count, devices.ptr))!;
        PhysicalDeviceCandidate[] candidates = mem::talloc_array(PhysicalDeviceCandidate, (sz)count);

        foreach (i, physical_device : devices[:count]) {
            PhysicalDeviceCandidate candidate = {
                .selection = { .physical_device = physical_device },
            };
            if (try families = select_queue_families(physical_device)) {
                if (candidate_satisfies(physical_device, desc, &families)) {
                    vk::PhysicalDeviceProperties props;
                    vk::get_physical_device_properties(physical_device, &props);
                    candidate.selection.families = families;
                    candidate.score = score_device((int)props.device_type);
                    candidate.suitable = true;
                }
            }
            candidates[i] = candidate;
        }
        return choose_physical_device(candidates[:count]);
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

<* Enable requested feature groups and deduplicate queue families. *>
fn void? create_logical_device(
    VkDeviceState* state,
    QueueFamilies* families,
    bool line_polygon_mode,
) @private {
    @pool() {
        uint family_capacity = families.counts.graphics
            + families.counts.compute
            + families.counts.transfer
            + (state.presentation.enabled ? 1 : 0);
        uint[] selected_families =
            mem::talloc_array(uint, (sz)family_capacity);
        uint queue_info_count = selected_device_queue_families(
            families,
            state.presentation.enabled,
            selected_families,
        );

        uint max_queue_count;
        foreach (family : selected_families[:queue_info_count]) {
            uint queue_count = requested_queue_count(
                families,
                family,
                state.presentation.enabled,
            );
            if (queue_count > max_queue_count) {
                max_queue_count = queue_count;
            }
        }
        float[] priorities = mem::talloc_array(float, (sz)max_queue_count);
        foreach (&priority : priorities) {
            *priority = 1.0f;
        }

        vk::DeviceQueueCreateInfo[] queue_infos =
            mem::talloc_array(vk::DeviceQueueCreateInfo, (sz)queue_info_count);
        foreach (index, family : selected_families[:queue_info_count]) {
            uint queue_count = requested_queue_count(
                families,
                family,
                state.presentation.enabled,
            );
            queue_infos[index] = vk::device_queue_create_info()
                .set_queue_family_index(family)
                .set_queue_priorities(priorities[:queue_count]);
        }
        return create_logical_device_with_queue_infos(
            state,
            queue_infos,
            line_polygon_mode,
        );
    };
}

fn void? create_logical_device_with_queue_infos(
    VkDeviceState* state,
    vk::DeviceQueueCreateInfo[] queue_infos,
    bool line_polygon_mode,
) @private {

    vk::PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT generated_work_features =
        vk::physical_device_device_generated_commands_features_ext()
            .set_device_generated_commands(vk::TRUE);
    vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5_features =
        vk::physical_device_maintenance5features_khr()
            .set_maintenance5(vk::TRUE)
            .set_next(&generated_work_features);
    vk::PhysicalDeviceDescriptorBufferFeaturesEXT db_features =
        vk::physical_device_descriptor_buffer_features_ext()
            .set_descriptor_buffer(vk::TRUE);
    vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT presentation_features =
        vk::physical_device_swapchain_maintenance1features_ext()
            .set_swapchain_maintenance1(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features features13 =
        vk::physical_device_vulkan13features()
            .set_dynamic_rendering(vk::TRUE)
            .set_synchronization2(vk::TRUE)
            .set_maintenance4(vk::TRUE);
    if (strict_heap_uses_descriptor_buffer(&state.strict_heap)) {
        if (state.generated_work.enabled) {
            db_features = db_features.set_next(&maintenance5_features);
        }
        features13 = features13.set_next(&db_features);
    } else if (state.generated_work.enabled) {
        features13 = features13.set_next(&maintenance5_features);
    }
    if (state.presentation.enabled) {
        presentation_features = presentation_features.set_next(
            features13.p_next,
        );
        features13 = features13.set_next(&presentation_features);
    }
    vk::PhysicalDeviceVulkan12Features features12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE)
        .set_next(&features13);
    if (state.strict_heap.enabled) {
        features12 = features12
            .set_runtime_descriptor_array(vk::TRUE)
            .set_shader_sampled_image_array_non_uniform_indexing(vk::TRUE)
            .set_shader_storage_image_array_non_uniform_indexing(vk::TRUE);
    }
    if (strict_heap_uses_descriptor_indexing(&state.strict_heap)) {
        features12 = features12
            .set_descriptor_binding_partially_bound(vk::TRUE)
            .set_descriptor_binding_sampled_image_update_after_bind(vk::TRUE)
            .set_descriptor_binding_storage_image_update_after_bind(vk::TRUE)
            .set_descriptor_binding_update_unused_while_pending(vk::TRUE);
    }
    if (state.draw_indirect_count) {
        features12 = features12.set_draw_indirect_count(vk::TRUE);
    }
    vk::PhysicalDeviceVulkan11Features features11 = vk::physical_device_vulkan11features()
        .set_shader_draw_parameters(vk::TRUE)
        .set_next(&features12);
    vk::PhysicalDeviceFeatures2 features2 = vk::physical_device_features2().set_next(&features11);
    features2.features.shader_int64 = vk::TRUE;
    if (line_polygon_mode) features2.features.fill_mode_non_solid = vk::TRUE;
    if (state.max_sampler_anisotropy > 0) features2.features.sampler_anisotropy = vk::TRUE;
    features2.features.multi_draw_indirect = vk::TRUE;
    if (state.strict_heap.enabled) {
        features2.features.shader_storage_image_read_without_format  = vk::TRUE;
        features2.features.shader_storage_image_write_without_format = vk::TRUE;
    }

    state.memory_budget_enabled = device_supports_extension(state.physical_device, "VK_EXT_memory_budget");
    ZString[6] extensions;
    uint extension_count = 0;
    if (state.memory_budget_enabled) extensions[extension_count++] = "VK_EXT_memory_budget";
    if (state.generated_work.enabled) {
        extensions[extension_count++] = GENERATED_WORK_EXTENSION;
        extensions[extension_count++] = GENERATED_WORK_USAGE_EXTENSION;
    }
    if (strict_heap_uses_descriptor_buffer(&state.strict_heap)) {
        extensions[extension_count++] = "VK_EXT_descriptor_buffer";
    }
    if (state.presentation.enabled) {
        if (!device_supports_extension(
                state.physical_device,
                SWAPCHAIN_EXTENSION,
            ) || !device_supports_extension(
                state.physical_device,
                SWAPCHAIN_MAINTENANCE_EXTENSION,
            )) {
            return gpu::UNSUPPORTED_FEATURE~;
        }
        extensions[extension_count++] = SWAPCHAIN_EXTENSION;
        extensions[extension_count++] = SWAPCHAIN_MAINTENANCE_EXTENSION;
    }

    vk::DeviceCreateInfo device_info = vk::device_create_info()
        .set_next(&features2)
        .set_queue_create_infos(queue_infos);
    if (extension_count > 0) {
        device_info = device_info.set_enabled_extension_names(extensions[:extension_count]);
    }

    check_backend_create_result(vk::create_device(
        physical_device: state.physical_device,
        p_create_info:   &device_info,
        p_allocator:     null,
        p_device:        &state.device,
    ))!;
}

<* True if `physical_device` advertises the named device extension. *>
fn bool? query_device_extension_support(
    vk::PhysicalDevice physical_device,
    ZString name,
) @private {
    uint count = 0;
    vk::try_enumerate_device_extension_properties(physical_device, null, &count, null)!;
    if (count == 0) return false;

    @pool() {
        vk::ExtensionProperties[] properties = mem::talloc_array(vk::ExtensionProperties, (sz)count);
        vk::try_enumerate_device_extension_properties(
            physical_device,
            null,
            &count,
            properties.ptr,
        )!;
        for (uint i = 0; i < count; i++) {
            if (((ZString)&properties[i].extension_name[0]).str_view() == name.str_view()) return true;
        }
    };
    return false;
}

fn bool device_supports_extension(vk::PhysicalDevice physical_device, ZString name) @private {
    if (try supported = query_device_extension_support(physical_device, name)) return supported;
    return false;
}

<* Report enabled capability state and configured strict heap capacities. *>
fn gpu::DeviceCaps build_caps(
    vk::PhysicalDeviceFeatures* base,
    vk::PhysicalDeviceVulkan12Features* v12,
    vk::PhysicalDeviceVulkan13Features* v13,
    vk::PhysicalDevice physical_device,
    CapsInputs* inputs,
    ulong* max_timeline_difference,
    usz* non_coherent_atom_size,
) @private {
    vk::PhysicalDeviceTimelineSemaphoreProperties timeline_props =
        vk::physical_device_timeline_semaphore_properties();
    vk::PhysicalDeviceProperties2 props2 =
        vk::physical_device_properties2().set_next(&timeline_props);
    vk::get_physical_device_properties2(physical_device, &props2);
    vk::PhysicalDeviceProperties props = props2.properties;
    *max_timeline_difference =
        timeline_props.max_timeline_semaphore_value_difference;
    *non_coherent_atom_size = (usz)props.limits.non_coherent_atom_size;
    return (gpu::DeviceCaps){
        .strict_enabled            = inputs.strict_enabled,
        .presentation_enabled      = inputs.presentation_enabled,
        .buffer_device_address      = v12.buffer_device_address != 0,
        .synchronization2           = v13.synchronization2 != 0,
        .dynamic_rendering          = v13.dynamic_rendering != 0,
        .shader_int64               = base.shader_int64 != 0,
        .draw_indirect_count        = v12.draw_indirect_count != 0,
        .generated_work             = inputs.generated_work.enabled,
        .async_compute              = inputs.async_compute,
        .line_polygon_mode          = inputs.line_polygon_mode,
        .texture_heap_capacity      = inputs.texture_capacity,
        .sampler_heap_capacity      = inputs.sampler_capacity,
        .max_color_attachments      = props.limits.max_color_attachments < gpu::MAX_COLOR_ATTACHMENTS
            ? props.limits.max_color_attachments : gpu::MAX_COLOR_ATTACHMENTS,
        .max_push_constant_size     = props.limits.max_push_constants_size,
        .max_compute_work_group_count = {
            .x = props.limits.max_compute_work_group_count[0],
            .y = props.limits.max_compute_work_group_count[1],
            .z = props.limits.max_compute_work_group_count[2],
        },
        .max_draw_indirect_count    = props.limits.max_draw_indirect_count,
        .max_generated_work_count   = inputs.generated_work.max_count,
        .min_uniform_alignment      = (usz)props.limits.min_uniform_buffer_offset_alignment,
        .min_storage_alignment      = (usz)props.limits.min_storage_buffer_offset_alignment,
        .min_texel_buffer_alignment = (usz)props.limits.min_texel_buffer_offset_alignment,
        .max_sampler_lod_bias       = props.limits.max_sampler_lod_bias,
        .max_sampler_anisotropy     = base.sampler_anisotropy != 0 ? props.limits.max_sampler_anisotropy : 0.0f,
    };
}
