module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import vma;
import std::core::mem;
import std::core::env;
import std::atomic;
import std::atomic::types;
import std::thread;
import std::io;

// Disambiguates from vma::Allocator: std::core::mem's host-memory interface.
alias HostAllocator = std::core::mem::alloc::Allocator;
alias DeviceExtensionQueryFn = fn bool?(vk::PhysicalDevice physical_device, ZString extension);
alias SemaphoreCounterQueryFn @private = fn vk::Result (
    vk::Device,
    vk::Semaphore,
    ulong*,
);
alias GeneratedDispatchLayoutReadHook @private @if(env::TESTING) = fn void (void*);
alias DeviceCreateFn = fn vk::Result (
    void* user_data,
    vk::PhysicalDevice physical_device,
    vk::DeviceCreateInfo* create_info,
    vk::Device* device,
);

const ZString SWAPCHAIN_EXTENSION             = "VK_KHR_swapchain";
const ZString SWAPCHAIN_MAINTENANCE_EXTENSION = "VK_EXT_swapchain_maintenance1";
const ZString DYNAMIC_STATE_3_EXTENSION       = "VK_EXT_extended_dynamic_state3";
const ZString GENERATED_WORK_EXTENSION        = "VK_EXT_device_generated_commands";
const ZString GENERATED_WORK_USAGE_EXTENSION = "VK_KHR_maintenance5";
const uint GENERATED_WORK_TOKEN_COUNT  = 2;
const uint GENERATED_WORK_TOKEN_OFFSET = 16;
const uint GENERATED_WORK_RECORD_SIZE  = 40;

enum DeviceCreationCheckpoint : uint {
    NONE,
    LOGICAL_DEVICE,
    ALLOCATOR,
    COMMAND_STORAGE,
    DESCRIPTOR_HEAP,
    PIPELINE_SHARED,
    BEFORE_PUBLICATION,
}

struct DeviceValidationPolicy {
    gpu::ContractValidation contract;
    bool                    vulkan_layers;
}

struct DeviceCreationProbe {
    DeviceCreationCheckpoint fail_at;
    DeviceCreationCheckpoint reached;
    uint                     rollback_count;
}

struct DeviceCreationOps {
    void*          user_data;
    DeviceCreateFn create;
}

struct VkPresentationState {
    bool           enabled;
    uint           family;
    uint           queue_index;
    vk::Queue      queue;
    Mutex          queue_mutex;
    SwapchainTable swapchains;
}

struct VkShaderHeapState {
    uint                texture_capacity;
    uint                sampler_capacity;
    DescriptorHeapState descriptors;
}

struct ViewportLimits {
    uint  max_width;
    uint  max_height;
    float min_bound;
    float max_bound;
}

struct VkDeviceState @public {
    gpu::Device                 owner;
    ulong                       resource_owner;
    vk::Instance               instance;
    VkInstanceDispatch         instance_dispatch;
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
    ViewportLimits             viewport_limits;
    ulong                      max_timeline_semaphore_value_difference;
    usz                        non_coherent_atom_size;
    DeviceValidationPolicy     validation_policy;
    CommandOps*                command_ops;
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
    ulong                      next_submit_visit_epoch;
    bool                       compute_queue_distinct;
    CommandAllocatorTable      command_allocators;
    CommandTable               commands;
    PipelineTable              pipelines;
    ShaderStore                shader_store;
    PipelineCacheTable         pipeline_cache;
    vk::PipelineLayout         graphics_layout;
    vk::PipelineLayout         compute_layout;
    vk::DescriptorSetLayout    shader_pipeline_set_layout;
    vk::IndirectCommandsLayoutEXT generated_draw_layout;
    vk::IndirectCommandsLayoutEXT generated_indexed_draw_layout;
    vk::IndirectCommandsLayoutEXT generated_dispatch_layout;
    vk::PipelineCache          driver_cache;
    TextureTable               textures;
    AttachmentViewTable        attachment_views;
    SamplerTable               samplers;
    VkPresentationState        presentation;
    VkShaderHeapState          shader_heap;
    Atomic{ulong}              texture_barrier_helper_calls @if(env::TESTING);
    Atomic{ulong}              texture_barrier_access_validations @if(env::TESTING);
    Atomic{ulong}              texture_barrier_range_resolutions @if(env::TESTING);
    Atomic{ulong}              texture_barrier_state_validations @if(env::TESTING);
    Atomic{ulong}              texture_barrier_native_assemblies @if(env::TESTING);
    Atomic{ulong}              texture_barrier_native_emissions @if(env::TESTING);
    bool                       dynamic_state_observation_enabled @if(env::TESTING);
    ulong                      begin_rendering_emissions @if(env::TESTING);
    ulong                      viewport_emissions @if(env::TESTING);
    ulong                      scissor_emissions @if(env::TESTING);
    ulong                      raster_state_emissions @if(env::TESTING);
    ulong                      depth_state_emissions @if(env::TESTING);
    vk::Viewport               last_viewport @if(env::TESTING);
    vk::Rect2D                 last_scissor @if(env::TESTING);
    gpu::DynamicRasterState    last_raster_state @if(env::TESTING);
    gpu::DepthState            last_depth_state @if(env::TESTING);
    CompletionRetirementLockHook completion_retirement_lock_hook @if(env::TESTING);
    void*                      completion_retirement_lock_user_data @if(env::TESTING);
    ulong                      pipeline_shader_modules_created;
    ulong                      pipelines_created;
    ulong                      shader_reflection_validations @if(env::TESTING);
    ulong                      pipeline_shader_create_attempts @if(env::TESTING);
    bool                       root_abi_evidence_enabled @if(env::TESTING);
    vk::Result                 pipeline_shader_create_result @if(env::TESTING);
    vk::Result                 pipeline_native_create_result @if(env::TESTING);
    uint                       pipeline_native_dynamic_state_count @if(env::TESTING);
    uint                       pipeline_native_color_target_count @if(env::TESTING);
    bool                       pipeline_native_color_states @if(env::TESTING);
    bool                       pipeline_native_canonical_color_attachments @if(env::TESTING);
    bool                       pipeline_cache_insert_failure @if(env::TESTING);
    GeneratedDispatchLayoutReadHook generated_dispatch_layout_read_hook @if(env::TESTING);
    void*                       generated_dispatch_layout_read_user_data @if(env::TESTING);
    vk::IndirectCommandsLayoutEXT generated_dispatch_layout_read_observed @if(env::TESTING);
    TextureBarrierEmitHook     texture_barrier_emit_hook @if(env::TESTING);
    void*                      texture_barrier_emit_user_data @if(env::TESTING);
    BarrierEmitHook            barrier_emit_hook @if(env::TESTING);
    void*                      barrier_emit_user_data @if(env::TESTING);
    Atomic{ulong}              barrier_native_emissions @if(env::TESTING);
    SubmissionCheckpointHook  submission_checkpoint_hook @if(env::TESTING);
    void*                     submission_checkpoint_user_data @if(env::TESTING);
    HostAllocator              host_allocator;
    // Advisory, best-effort across queue mutexes — mirrors get_memory_stats.
}

struct GeneratedWorkSupport {
    bool enabled;
    uint max_count;
}

struct CapsInputs {
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

const CommandOps TRUSTED_COMMAND_OPS @private = {
    .copy_buffer                 = &trusted_copy_buffer,
    .fill_buffer                 = &trusted_fill_buffer,
    .copy_buffer_to_texture      = &trusted_copy_buffer_to_texture,
    .copy_texture_to_buffer      = &trusted_copy_texture_to_buffer,
    .barrier                     = &trusted_barrier,
    .texture_barrier             = &trusted_texture_barrier,
    .bind_pipeline               = &trusted_bind_pipeline,
    .set_depth_state             = &trusted_set_depth_state,
    .set_raster_state            = &trusted_set_raster_state,
    .set_graphics_state          = &trusted_set_graphics_state,
    .set_color_state             = &trusted_set_color_state,
    .dispatch                    = &trusted_dispatch,
    .begin_render_pass           = &trusted_begin_render_pass,
    .set_viewport                = &trusted_set_viewport,
    .set_scissor                 = &trusted_set_scissor,
    .end_render_pass             = &vk_cmd_end_render_pass,
    .draw                        = &trusted_draw,
    .draw_indexed                = &trusted_draw_indexed,
    .draw_indirect               = &trusted_draw_indirect,
    .draw_indexed_indirect       = &trusted_draw_indexed_indirect,
    .draw_indexed_indirect_count = &trusted_draw_indexed_indirect_count,
    .dispatch_indirect           = &trusted_dispatch_indirect,
    .draw_generated              = &trusted_draw_generated,
    .draw_indexed_generated      = &trusted_draw_indexed_generated,
    .dispatch_generated          = &trusted_dispatch_generated,
    .begin_label                 = &vk_cmd_begin_label,
    .end_label                   = &vk_cmd_end_label,
};

const CommandOps FULL_COMMAND_OPS @private = {
    .copy_buffer                 = &full_copy_buffer,
    .fill_buffer                 = &full_fill_buffer,
    .copy_buffer_to_texture      = &full_copy_buffer_to_texture,
    .copy_texture_to_buffer      = &full_copy_texture_to_buffer,
    .barrier                     = &full_barrier,
    .texture_barrier             = &full_texture_barrier,
    .bind_pipeline               = &full_bind_pipeline,
    .set_depth_state             = &full_set_depth_state,
    .set_raster_state            = &full_set_raster_state,
    .set_graphics_state          = &full_set_graphics_state,
    .set_color_state             = &full_set_color_state,
    .dispatch                    = &full_dispatch,
    .begin_render_pass           = &full_begin_render_pass,
    .set_viewport                = &full_set_viewport,
    .set_scissor                 = &full_set_scissor,
    .end_render_pass             = &vk_cmd_end_render_pass,
    .draw                        = &full_draw,
    .draw_indexed                = &full_draw_indexed,
    .draw_indirect               = &full_draw_indirect,
    .draw_indexed_indirect       = &full_draw_indexed_indirect,
    .draw_indexed_indirect_count = &full_draw_indexed_indirect_count,
    .dispatch_indirect           = &full_dispatch_indirect,
    .draw_generated              = &full_draw_generated,
    .draw_indexed_generated      = &full_draw_indexed_generated,
    .dispatch_generated          = &full_dispatch_generated,
    .begin_label                 = &vk_cmd_begin_label,
    .end_label                   = &vk_cmd_end_label,
};

fn bool contract_uses_full_validation(DeviceValidationPolicy policy) =>
    policy.contract != gpu::ContractValidation.TRUSTED;

fn CommandOps* select_command_ops(DeviceValidationPolicy policy) {
    return contract_uses_full_validation(policy)
        ? &FULL_COMMAND_OPS
        : &TRUSTED_COMMAND_OPS;
}

fn gpu::Device? create_vk_device(
    gpu::Adapter* adapter,
    gpu::DeviceDesc* device_desc,
    gpu::Runtime* retained_runtime,
) {
    return create_vk_device_with_probe(
        adapter:          adapter,
        device_desc:      device_desc,
        retained_runtime: retained_runtime,
        probe:            null,
        creation_ops:     null,
    );
}

fn gpu::Device? create_vk_device_with_probe(
    gpu::Adapter* adapter,
    gpu::DeviceDesc* device_desc,
    gpu::Runtime* retained_runtime,
    DeviceCreationProbe* probe,
    DeviceCreationOps* creation_ops = null,
) {
    gpu::DeviceDesc normalized =
        gpu::internal::normalize_device_desc(device_desc)!;
    gpu::internal::RuntimeData* runtime_data = gpu::internal::resolve_adapter(adapter);
    if (runtime_data == null) return gpu::INVALID_HANDLE~;
    VkRuntimeState* runtime_state = runtime_data.state;
    if (runtime_state == null) return gpu::INVALID_HANDLE~;

    VkAdapterState* adapter_state = &runtime_state.adapters[adapter.adapter_index];
    gpu::Surface requested_surface = normalized.surface;
    vk::SurfaceKHR presentation_surface = (vk::SurfaceKHR)null;
    if (requested_surface.is_valid()) {
        gpu::internal::SurfaceData* surface_data =
            gpu::internal::resolve_surface(&requested_surface);
        if (surface_data == null) return gpu::INVALID_HANDLE~;
        gpu::internal::RuntimeData* surface_runtime =
            gpu::internal::resolve_runtime(&surface_data.runtime);
        if (surface_runtime != runtime_data) return gpu::INVALID_HANDLE~;
        presentation_surface = (vk::SurfaceKHR)surface_data.backend_surface;
        if (!runtime_state.instance_capabilities.surface_maintenance) {
            return gpu::UNSUPPORTED_FEATURE~;
        }
    }
    return create_vk_device_from_runtime(
        config:                     &runtime_state.config,
        instance:                   runtime_state.instance,
        physical_device:            adapter_state.physical_device,
        retained_runtime:           retained_runtime,
        adapter:                    adapter,
        runtime_state:              runtime_state,
        presentation_surface:       presentation_surface,
        presentation_surface_owner: requested_surface.is_valid()
            ? &requested_surface
            : null,
        queue_request:              &normalized.queues,
        shader_heap_support:        &adapter_state.shader_heap,
        probe:                      probe,
        creation_ops:               creation_ops,
    );
}

fn void? validate_device_baseline_support(
    VkRuntimeState* runtime_state,
    uint adapter_index,
) {
    if (runtime_state == null) return gpu::INVALID_HANDLE~;
    if (adapter_index >= runtime_state.adapters.len) return gpu::INVALID_HANDLE~;

    VkShaderHeapState shader_heap =
        resolve_shader_heap_state(&runtime_state.config)!;
    VkDeviceState diagnostic_state = {
        .debug_callback    = runtime_state.config.debug_callback,
        .debug_user_data   = runtime_state.config.debug_user_data,
        .validation_policy = runtime_state.config.validation,
    };
    validate_shader_heap_support(
        &diagnostic_state,
        &shader_heap,
        &runtime_state.adapters[adapter_index].shader_heap,
    )!;
}

fn gpu::Device? create_vk_device_from_runtime(
    RuntimeConfig* config,
    vk::Instance instance,
    vk::PhysicalDevice physical_device,
    gpu::Runtime* retained_runtime = null,
    gpu::Adapter* adapter = null,
    VkRuntimeState* runtime_state = null,
    vk::SurfaceKHR presentation_surface = (vk::SurfaceKHR)null,
    gpu::Surface* presentation_surface_owner = null,
    gpu::QueueRequest* queue_request = null,
    ShaderHeapSupport* shader_heap_support = null,
    DeviceCreationProbe* probe = null,
    DeviceCreationOps* creation_ops = null,
) @private {
    VkDeviceState* state = mem::new(VkDeviceState);
    state.resource_owner = gpu::internal::allocate_device_resource_owner()!;
    defer catch {
        destroy_state(state);
        if (probe != null) probe.rollback_count++;
    }
    state.host_allocator = &allocators::LIBC_ALLOCATOR;
    state.debug_callback = config.debug_callback;
    state.debug_user_data = config.debug_user_data;
    state.debug_names = config.debug_names;
    state.validation_policy = config.validation;
    state.command_ops = select_command_ops(state.validation_policy);
    state.resource_mutex.init()!;
    state.pipeline_creation_mutex.init()!;
    state.texture_view_cache_mutex.init()!;
    state.command_mutex.init()!;

    state.shader_heap = resolve_shader_heap_state(config)!;

    state.presentation.enabled = presentation_surface != null;
    if (state.presentation.enabled) state.presentation.queue_mutex.init()!;
    bool debug_utils_enabled;
    QueueFamilies families;
    if (instance == null || physical_device == null || runtime_state == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    state.instance = instance;
    debug_utils_enabled = runtime_state.instance_capabilities.debug_utils;
    if (state.presentation.enabled) {
        state.instance_dispatch.surface = runtime_state.dispatch.surface;
    }
    state.physical_device = physical_device;
    families = presentation_surface == null
        ? select_queue_families(
            physical_device: state.physical_device,
            request:         queue_request,
        )!
        : select_queue_families_for_surface(
            runtime:         runtime_state,
            physical_device: state.physical_device,
            surface:         presentation_surface,
            request:         queue_request,
        )!;

    bool swapchain_maintenance_extension = state.presentation.enabled
        && device_supports_extension(
            state.physical_device,
            SWAPCHAIN_MAINTENANCE_EXTENSION,
        );
    bool generated_work_extension = device_supports_extension(
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
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_color_features =
        vk::physical_device_extended_dynamic_state3features_ext();
    vk::PhysicalDeviceVulkan13Features features13 =
        vk::physical_device_vulkan13features();
    if (generated_work_usage_extension) {
        maintenance5_features = maintenance5_features.set_next(&generated_work_features);
        features13 = features13.set_next(&maintenance5_features);
    }
    dynamic_color_features = dynamic_color_features.set_next(features13.p_next);
    features13 = features13.set_next(&dynamic_color_features);
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
    check_required_dynamic_state3_extension(
        state.physical_device,
        state,
    )!;
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        query_dynamic_raster_properties(state.physical_device);
    check_required_features(
        base:               &features2.features,
        v11:                &features11,
        v12:                &features12,
        v13:                &features13,
        dynamic_color:      &dynamic_color_features,
        dynamic_properties: &dynamic_properties,
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
    ShaderHeapSupport queried_support;
    if (shader_heap_support == null) {
        queried_support = {
            .indexing = heap_indexing_supported(
                &features2.features,
                &features12,
            ),
            .limits = query_indexing_heap_limits(state.physical_device),
        };
        shader_heap_support = &queried_support;
    }
    validate_shader_heap_support(
        state,
        &state.shader_heap,
        shader_heap_support,
    )!;

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

    create_logical_device(
        state:             state,
        families:          &families,
        line_polygon_mode: line_polygon_mode,
        creation_ops:      creation_ops,
    )!;
    reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.LOGICAL_DEVICE)!;
    VkDeviceDispatchPlan dispatch_plan = {
        .debug          = debug_utils_enabled,
        .presentation   = state.presentation.enabled,
        .generated_work = state.generated_work.enabled,
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
    if (state.generated_work.enabled
        && !device_dispatch_supports_generated_work(&state.device_dispatch)) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    if (!device_dispatch_supports_dynamic_color(&state.device_dispatch)) {
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
    state.buffers.owner = gpu::internal::resource_owner_tag(state.resource_owner, gpu::internal::ResourceOwnerKind.BUFFER);
    state.buffers.slots = mem::new_array(BufferCell, MAX_BUFFERS);
    state.allocations.owner = gpu::internal::resource_owner_tag(
        state.resource_owner,
        gpu::internal::ResourceOwnerKind.ALLOCATION,
    );
    state.allocations.slots = mem::new_array(
        AllocationCell,
        ALLOCATION_CAPACITY,
    );

    state.command_allocators.owner = gpu::internal::resource_owner_tag(
        state.resource_owner,
        gpu::internal::ResourceOwnerKind.COMMAND_ALLOCATOR,
    );
    state.command_allocators.slots = mem::new_array(
        CommandAllocatorCell,
        MAX_COMMAND_ALLOCATORS,
    );
    state.commands.owner = gpu::internal::resource_owner_tag(
        state.resource_owner,
        gpu::internal::ResourceOwnerKind.COMMAND,
    );
    state.commands.slots = mem::new_array(CommandCell, MAX_COMMAND_LISTS);
    reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.COMMAND_STORAGE)!;
    uint pipeline_capacity =
        resolve_descriptor_capacity(config.pipeline_capacity, gpu::MAX_PIPELINES)!;
    state.pipelines.owner = gpu::internal::resource_owner_tag(
        state.resource_owner,
        gpu::internal::ResourceOwnerKind.PIPELINE,
    );
    state.pipelines.slots = mem::new_array(PipelineCell, pipeline_capacity);
    state.textures.owner = gpu::internal::resource_owner_tag(state.resource_owner, gpu::internal::ResourceOwnerKind.TEXTURE);
    state.textures.slots = mem::new_array(
        TextureCell,
        resolve_descriptor_capacity(
            config.texture_capacity,
            gpu::DEFAULT_TEXTURE_CAPACITY,
        )!,
    );
    state.attachment_views.owner = gpu::internal::resource_owner_tag(
        state.resource_owner,
        gpu::internal::ResourceOwnerKind.ATTACHMENT_VIEW,
    );
    state.attachment_views.slots = mem::new_array(
        AttachmentViewCell,
        gpu::DEFAULT_ATTACHMENT_VIEW_CAPACITY,
    );
    if (state.presentation.enabled) {
        state.presentation.swapchains.owner = gpu::internal::resource_owner_tag(state.resource_owner, gpu::internal::ResourceOwnerKind.SWAPCHAIN);
        state.presentation.swapchains.slots = mem::new_array(SwapchainCell, gpu::MAX_SWAPCHAINS);
    }
    create_sampler_table(state)!;
    create_descriptor_heap(state)!;
    reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.DESCRIPTOR_HEAP)!;
    create_pipeline_shared(
        state,
        config.pipeline_cache_data,
        pipeline_capacity,
    )!;
    reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.PIPELINE_SHARED)!;
    CapsInputs caps_inputs = {
        .presentation_enabled = state.presentation.enabled,
        .async_compute        = state.compute_queue_distinct,
        .line_polygon_mode    = line_polygon_mode,
        .texture_capacity     = state.shader_heap.texture_capacity,
        .sampler_capacity     = state.shader_heap.sampler_capacity,
        .generated_work       = state.generated_work,
    };
    state.caps = build_caps(
        base:                    &features2.features,
        v12:                     &features12,
        physical_device:         state.physical_device,
        inputs:                  &caps_inputs,
        viewport_limits:         &state.viewport_limits,
        max_timeline_difference: &state.max_timeline_semaphore_value_difference,
        non_coherent_atom_size:  &state.non_coherent_atom_size,
    );

    gpu::QueueRequest request = queue_request == null
        ? gpu::internal::default_queue_request()
        : *queue_request;
    gpu::internal::QueueSelection queue_selection =
        gpu::internal::queue_selection_from_native_ids(
            request:  request,
            graphics: native_queue_id(
                families.graphics,
                families.graphics_index,
            ),
            compute: native_queue_id(
                families.compute,
                families.compute_index,
            ),
            transfer: native_queue_id(
                families.transfer,
                families.transfer_index,
            ),
        )!;
    state.caps.queues = queue_selection.selected;
    create_queue_completions(state, &queue_selection)!;

    reach_device_creation_checkpoint(probe, DeviceCreationCheckpoint.BEFORE_PUBLICATION)!;
    return gpu::internal::register_device(
        state:                state,
        caps:                 state.caps,
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
        if (completion.retired_sequence.load(AtomicOrdering.ACQUIRE)
            >= expected) continue;

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
    VkDeviceState* state,
    ZString operation,
    fault public_fault,
    gpu::DebugResourceKind resource_kind,
    ZString rejected_field,
    ZString invariant,
) {
    if (state.validation_policy.contract
        == gpu::ContractValidation.TRUSTED) return public_fault;
    return report_contract_failure(
        state:               state,
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

fn void? vk_check_device_destroy(VkDeviceState* state) {
    if (!device_destroy_ready_with_query(state, &real_semaphore_counter_query)!) {
        return gpu::DEVICE_BUSY~;
    }
}

fn void vk_destroy_device(VkDeviceState* state) {
    destroy_state(state);
}

fn bool report_teardown_leaks(VkDeviceState* state) =>
    contract_uses_full_validation(state.validation_policy);

<* Reverse-order cleanup after accepted destruction, device loss, or partial creation. *>
fn void destroy_state(VkDeviceState* state) @private {
    if (state == null) return;
    if (state.device != null) {
        destroy_submitted_commands(state);
        // Report before cleanup clears the diagnostic tables.
        if (report_teardown_leaks(state)) (void)report_resource_leaks(state);
        destroy_command_table(state);
        if (state.command_allocators.slots.ptr != null) {
            foreach (&cell : state.command_allocators.slots[
                :state.command_allocators.next_new
            ]) {
                if (cell.used) {
                    destroy_command_allocator_slot(state, &cell.value);
                }
            }
            mem::free(state.command_allocators.slots.ptr);
            state.command_allocators.slots = {};
        }
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
        if (report_teardown_leaks(state)) report_descriptor_leaks(state);
        destroy_descriptor_heap(state);
        destroy_sampler_table(state);
        destroy_attachment_view_table(state);
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
        destroy_queue_completions(state);
        if (state.allocator != null) {
            state.allocator.destroy();
            state.allocator = (vma::Allocator)null;
        }
        vk::destroy_device(state.device, null);
        state.device = (vk::Device)null;
    }
    state.instance = (vk::Instance)null;
    if (state.resource_mutex.is_initialized())       (void)state.resource_mutex.destroy();
    if (state.pipeline_creation_mutex.is_initialized()) (void)state.pipeline_creation_mutex.destroy();
    if (state.texture_view_cache_mutex.is_initialized()) (void)state.texture_view_cache_mutex.destroy();
    if (state.command_mutex.is_initialized())        (void)state.command_mutex.destroy();
    if (state.presentation.queue_mutex.is_initialized()) (void)state.presentation.queue_mutex.destroy();
    mem::free(state);
}

<* Gate required device features and unrestricted dynamic topology.
   Pure, so it is unit-testable without a device. *>
fn void? check_required_features(
    vk::PhysicalDeviceFeatures* base,
    vk::PhysicalDeviceVulkan11Features* v11,
    vk::PhysicalDeviceVulkan12Features* v12,
    vk::PhysicalDeviceVulkan13Features* v13,
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT* dynamic_color,
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT* dynamic_properties,
) {
    if (base.shader_int64 == 0)          return gpu::UNSUPPORTED_FEATURE~;
    if (base.multi_draw_indirect == 0)   return gpu::UNSUPPORTED_FEATURE~;
    if (base.independent_blend == 0)     return gpu::UNSUPPORTED_FEATURE~;
    if (base.depth_bias_clamp == 0)      return gpu::UNSUPPORTED_FEATURE~;
    if (v11.shader_draw_parameters == 0) return gpu::UNSUPPORTED_FEATURE~;
    if (v12.buffer_device_address == 0)  return gpu::UNSUPPORTED_FEATURE~;
    if (v12.timeline_semaphore == 0)     return gpu::UNSUPPORTED_FEATURE~;
    if (v13.synchronization2 == 0)       return gpu::UNSUPPORTED_FEATURE~;
    if (v13.dynamic_rendering == 0)      return gpu::UNSUPPORTED_FEATURE~;
    if (!required_dynamic_color_features_supported(dynamic_color)) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    if (dynamic_properties.dynamic_primitive_topology_unrestricted == 0) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
}

fn vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT
query_dynamic_raster_properties(vk::PhysicalDevice physical_device) {
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        vk::physical_device_extended_dynamic_state3properties_ext();
    vk::PhysicalDeviceProperties2 properties2 =
        vk::physical_device_properties2().set_next(&dynamic_properties);
    vk::get_physical_device_properties2(physical_device, &properties2);
    return dynamic_properties;
}

fn bool required_dynamic_color_features_supported(
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT* features,
) {
    return features != null
        && features.extended_dynamic_state3color_blend_enable != 0
        && features.extended_dynamic_state3color_blend_equation != 0
        && features.extended_dynamic_state3color_write_mask != 0;
}

<* Optional non-solid polygon rasterization support. Pure. *>
fn bool line_polygon_mode_supported(vk::PhysicalDeviceFeatures* base) {
    return base.fill_mode_non_solid != 0;
}

<* Features required by the heap-access shader helpers. Pure. *>
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

fn VkShaderHeapState? resolve_shader_heap_state(RuntimeConfig* config) {
    if (config == null) return gpu::INVALID_ARGUMENT~;
    VkShaderHeapState state;
    state.texture_capacity = resolve_descriptor_capacity(
        config.texture_heap_capacity,
        gpu::DEFAULT_TEXTURE_HEAP_CAPACITY,
    )!;
    state.sampler_capacity = resolve_descriptor_capacity(
        config.sampler_heap_capacity,
        gpu::DEFAULT_SAMPLER_HEAP_CAPACITY,
    )!;
    return state;
}

fn void? validate_shader_heap_support(
    VkDeviceState* device,
    VkShaderHeapState* state,
    ShaderHeapSupport* support,
) {
    if (device == null || state == null || support == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (!support.indexing) {
        return report_contract_failure(
            state:               device,
            operation:           "create_device",
            public_fault:        gpu::UNSUPPORTED_FEATURE,
            resource_kind:       gpu::DebugResourceKind.DEVICE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "device baseline",
            invariant:           "the device baseline requires descriptor indexing with update-after-bind support",
            backend_text:        null,
        )~;
    }
    IndexingHeapLimitFailure failure = indexing_heap_limit_failure(
        state.texture_capacity,
        state.sampler_capacity,
        support.limits,
    );
    if (failure == IndexingHeapLimitFailure.NONE) return;
    ZString rejected_field;
    ZString invariant;
    switch (failure) {
        case IndexingHeapLimitFailure.SAMPLED_IMAGES:
            rejected_field = "RuntimeDesc.texture_heap_capacity";
            invariant = "texture heap capacity must fit the sampled-image update-after-bind limit";
        case IndexingHeapLimitFailure.STORAGE_IMAGES:
            rejected_field = "RuntimeDesc.texture_heap_capacity";
            invariant = "texture heap capacity must fit the storage-image update-after-bind limit";
        case IndexingHeapLimitFailure.SAMPLERS:
            rejected_field = "RuntimeDesc.sampler_heap_capacity";
            invariant = "sampler heap capacity must fit the sampler update-after-bind limit";
        case IndexingHeapLimitFailure.PER_STAGE_RESOURCES:
            rejected_field = "RuntimeDesc.texture_heap_capacity/sampler_heap_capacity";
            invariant = "combined descriptor capacity must fit the per-stage update-after-bind resource limit";
        case IndexingHeapLimitFailure.ALL_POOLS_DESCRIPTORS:
            rejected_field = "RuntimeDesc.texture_heap_capacity/sampler_heap_capacity";
            invariant = "combined descriptor capacity must fit the all-pools update-after-bind descriptor limit";
        default:
            return gpu::BACKEND_ERROR~;
    }
    if (device.debug_callback == null
        && contract_uses_full_validation(device.validation_policy)) {
        HeapDescriptorTotals totals = descriptor_heap_totals(
            state.texture_capacity,
            state.sampler_capacity,
        );
        io::eprintfn(
            "gpu: descriptor capacities (tex %d, smp %d) exceed a device update-after-bind limit (set %d/%d/%d, per-stage %d/%d/%d, per-stage resources %d/%d, all-pools descriptors %d/%d)",
            state.texture_capacity,
            state.sampler_capacity,
            support.limits.max_sampled_images,
            support.limits.max_storage_images,
            support.limits.max_samplers,
            support.limits.max_per_stage_sampled_images,
            support.limits.max_per_stage_storage_images,
            support.limits.max_per_stage_samplers,
            totals.per_stage_resources,
            support.limits.max_per_stage_resources,
            totals.all_pools_descriptors,
            support.limits.max_all_pools_descriptors,
        );
    }
    return report_contract_failure(
        state:               device,
        operation:           "create_device",
        public_fault:        gpu::UNSUPPORTED_FEATURE,
        resource_kind:       gpu::DebugResourceKind.DEVICE,
        has_public_identity: false,
        index:               0,
        generation:          0,
        debug_name:          null,
        rejected_field:      rejected_field,
        invariant:           invariant,
        backend_text:        null,
    )~;
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

fn void? check_required_dynamic_state3_extension_with_query(
    vk::PhysicalDevice physical_device,
    DeviceExtensionQueryFn extension_query,
    VkDeviceState* state = null,
) {
    if (!query_device_extension_support_mapped(
        state:           state,
        physical_device: physical_device,
        extension:       DYNAMIC_STATE_3_EXTENSION,
        extension_query: extension_query,
    )!) return gpu::UNSUPPORTED_FEATURE~;
}

fn void? check_required_dynamic_state3_extension(
    vk::PhysicalDevice physical_device,
    VkDeviceState* state = null,
) @private {
    return check_required_dynamic_state3_extension_with_query(
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

fn bool candidate_satisfies(
    vk::PhysicalDevice candidate,
    RuntimeConfig* config,
    ShaderHeapSupport* shader_heap,
) @private {
    vk::PhysicalDeviceProperties props;
    vk::get_physical_device_properties(candidate, &props);
    if (props.api_version < API_VERSION_1_3) return false;
    if (catch select_queue_families(candidate)) return false;
    if (catch check_required_dynamic_state3_extension(candidate)) return false;

    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_color_features =
        vk::physical_device_extended_dynamic_state3features_ext();
    vk::PhysicalDeviceVulkan13Features features13 =
        vk::physical_device_vulkan13features().set_next(&dynamic_color_features);
    vk::PhysicalDeviceVulkan12Features features12 = vk::physical_device_vulkan12features().set_next(&features13);
    vk::PhysicalDeviceVulkan11Features features11 = vk::physical_device_vulkan11features().set_next(&features12);
    vk::PhysicalDeviceFeatures2 features2 = vk::physical_device_features2().set_next(&features11);
    vk::get_physical_device_features2(candidate, &features2);
    vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT dynamic_properties =
        query_dynamic_raster_properties(candidate);
    if (catch check_required_features(
        base:               &features2.features,
        v11:                &features11,
        v12:                &features12,
        v13:                &features13,
        dynamic_color:      &dynamic_color_features,
        dynamic_properties: &dynamic_properties,
    )) return false;

    uint? resolved_tex = resolve_descriptor_capacity(
        config.texture_heap_capacity,
        gpu::DEFAULT_TEXTURE_HEAP_CAPACITY,
    );
    if (catch resolved_tex) return false;
    uint? resolved_smp = resolve_descriptor_capacity(
        config.sampler_heap_capacity,
        gpu::DEFAULT_SAMPLER_HEAP_CAPACITY,
    );
    if (catch resolved_smp) return false;
    if (catch resolve_descriptor_capacity(
        config.texture_capacity,
        gpu::DEFAULT_TEXTURE_CAPACITY,
    )) return false;
    if (catch resolve_descriptor_capacity(
        config.pipeline_capacity,
        gpu::MAX_PIPELINES,
    )) return false;

    return shader_heap != null
        && shader_heap.indexing
        && indexing_heap_fits_limits(
            resolved_tex,
            resolved_smp,
            shader_heap.limits,
        );
}

<* Enable requested feature groups and deduplicate queue families. *>
fn void? create_logical_device(
    VkDeviceState* state,
    QueueFamilies* families,
    bool line_polygon_mode,
    DeviceCreationOps* creation_ops = null,
) @private {
    @pool() {
        uint family_capacity = gpu::internal::MAX_SELECTED_QUEUE_IDENTITIES
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
            state:             state,
            queue_infos:       queue_infos,
            line_polygon_mode: line_polygon_mode,
            creation_ops:      creation_ops,
        );
    };
}

fn void? create_logical_device_with_queue_infos(
    VkDeviceState* state,
    vk::DeviceQueueCreateInfo[] queue_infos,
    bool line_polygon_mode,
    DeviceCreationOps* creation_ops,
) @private {

    vk::PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT generated_work_features =
        vk::physical_device_device_generated_commands_features_ext()
            .set_device_generated_commands(vk::TRUE);
    vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5_features =
        vk::physical_device_maintenance5features_khr()
            .set_maintenance5(vk::TRUE)
            .set_next(&generated_work_features);
    vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT presentation_features =
        vk::physical_device_swapchain_maintenance1features_ext()
            .set_swapchain_maintenance1(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features features13 =
        vk::physical_device_vulkan13features()
            .set_dynamic_rendering(vk::TRUE)
            .set_synchronization2(vk::TRUE)
            .set_maintenance4(vk::TRUE);
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_color_features =
        vk::physical_device_extended_dynamic_state3features_ext()
            .set_extended_dynamic_state3color_blend_enable(vk::TRUE)
            .set_extended_dynamic_state3color_blend_equation(vk::TRUE)
            .set_extended_dynamic_state3color_write_mask(vk::TRUE);
    if (state.generated_work.enabled) {
        features13 = features13.set_next(&maintenance5_features);
    }
    dynamic_color_features = dynamic_color_features.set_next(features13.p_next);
    features13 = features13.set_next(&dynamic_color_features);
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
    features12 = features12
        .set_runtime_descriptor_array(vk::TRUE)
        .set_shader_sampled_image_array_non_uniform_indexing(vk::TRUE)
        .set_shader_storage_image_array_non_uniform_indexing(vk::TRUE)
        .set_descriptor_binding_partially_bound(vk::TRUE)
        .set_descriptor_binding_sampled_image_update_after_bind(vk::TRUE)
        .set_descriptor_binding_storage_image_update_after_bind(vk::TRUE)
        .set_descriptor_binding_update_unused_while_pending(vk::TRUE);
    if (state.draw_indirect_count) {
        features12 = features12.set_draw_indirect_count(vk::TRUE);
    }
    vk::PhysicalDeviceVulkan11Features features11 = vk::physical_device_vulkan11features()
        .set_shader_draw_parameters(vk::TRUE)
        .set_next(&features12);
    vk::PhysicalDeviceFeatures2 features2 = vk::physical_device_features2().set_next(&features11);
    features2.features.shader_int64 = vk::TRUE;
    features2.features.independent_blend = vk::TRUE;
    features2.features.depth_bias_clamp = vk::TRUE;
    if (line_polygon_mode) features2.features.fill_mode_non_solid = vk::TRUE;
    if (state.max_sampler_anisotropy > 0) features2.features.sampler_anisotropy = vk::TRUE;
    features2.features.multi_draw_indirect = vk::TRUE;
    features2.features.shader_storage_image_read_without_format  = vk::TRUE;
    features2.features.shader_storage_image_write_without_format = vk::TRUE;

    state.memory_budget_enabled = device_supports_extension(state.physical_device, "VK_EXT_memory_budget");
    ZString[7] extensions;
    uint extension_count = 0;
    extensions[extension_count++] = DYNAMIC_STATE_3_EXTENSION;
    if (state.memory_budget_enabled) extensions[extension_count++] = "VK_EXT_memory_budget";
    if (state.generated_work.enabled) {
        extensions[extension_count++] = GENERATED_WORK_EXTENSION;
        extensions[extension_count++] = GENERATED_WORK_USAGE_EXTENSION;
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

    vk::Result create_result;
    if (creation_ops == null) {
        create_result = vk::create_device(
            physical_device: state.physical_device,
            p_create_info:   &device_info,
            p_allocator:     null,
            p_device:        &state.device,
        );
    } else {
        create_result = creation_ops.create(
            user_data:       creation_ops.user_data,
            physical_device: state.physical_device,
            create_info:     &device_info,
            device:          &state.device,
        );
    }
    check_backend_create_result(create_result)!;
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

<* Report enabled capability state and configured shader heap capacities. *>
fn gpu::DeviceCaps build_caps(
    vk::PhysicalDeviceFeatures* base,
    vk::PhysicalDeviceVulkan12Features* v12,
    vk::PhysicalDevice physical_device,
    CapsInputs* inputs,
    ViewportLimits* viewport_limits,
    ulong* max_timeline_difference,
    usz* non_coherent_atom_size,
) @private {
    vk::PhysicalDeviceTimelineSemaphoreProperties timeline_props =
        vk::physical_device_timeline_semaphore_properties();
    vk::PhysicalDeviceProperties2 props2 =
        vk::physical_device_properties2().set_next(&timeline_props);
    vk::get_physical_device_properties2(physical_device, &props2);
    vk::PhysicalDeviceProperties props = props2.properties;
    *viewport_limits = {
        .max_width  = props.limits.max_viewport_dimensions[0],
        .max_height = props.limits.max_viewport_dimensions[1],
        .min_bound  = props.limits.viewport_bounds_range[0],
        .max_bound  = props.limits.viewport_bounds_range[1],
    };
    *max_timeline_difference =
        timeline_props.max_timeline_semaphore_value_difference;
    *non_coherent_atom_size = (usz)props.limits.non_coherent_atom_size;
    return (gpu::DeviceCaps){
        .presentation_enabled       = inputs.presentation_enabled,
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
