module gpu::vk;

import gpu;
import vk;
import vma;
import std::core::mem;
import std::atomic;
import std::atomic::types;
import std::thread;

// Disambiguates from vma::Allocator: std::core::mem's host-memory interface.
alias HostAllocator = std::core::mem::alloc::Allocator;

// Per-device backend state, reached through Device.backend_state.
struct VkDeviceState {
    vk::Instance               instance;
    vk::DebugUtilsMessengerEXT messenger;
    vk::PhysicalDevice         physical_device;
    vk::Device                 device;
    vma::Allocator             allocator;
    uint                       graphics_family;
    uint                       compute_family;
    uint                       transfer_family;
    vk::Queue                  graphics_queue;
    vk::Queue                  compute_queue;
    vk::Queue                  transfer_queue;
    gpu::DeviceCaps            caps;
    bool                       validation;
    bool                       memory_budget_enabled;
    bool                       draw_indirect_count;
    float                      max_sampler_anisotropy;
    gpu::DescriptorHeapMode    heap_mode;
    uint                       texture_descriptor_capacity;
    uint                       sampler_descriptor_capacity;
    bool                       presentation_enabled;
    BufferTable                buffers;
    vk::Semaphore              frame_timeline;
    // Per-queue aux timelines piggybacked onto user submits; each signals
    // from one queue only, so values are monotonic. counter/last_signaled
    // mutate under that queue's mutex, last_signaled only after success.
    vk::Semaphore              aux_compute_timeline;
    vk::Semaphore              aux_transfer_timeline;
    ulong                      aux_compute_counter;
    ulong                      aux_compute_last_signaled;
    ulong                      aux_transfer_counter;
    ulong                      aux_transfer_last_signaled;
    // Blocking-helper completions signal here (turnstiled), never frame_timeline.
    vk::Semaphore              helper_timeline;
    Atomic{ulong}              timeline_counter;
    // Next helper reservation value; mutated only under transfer_mutex.
    ulong                      helper_counter;
    // Set unconditionally by vk_submit so end_frame also waits off-frame use.
    bool                       compute_used_since_frame;
    bool                       transfer_used_since_frame;
    // Off-frame submission not yet covered by an end_frame chain or teardown.
    Atomic{bool}               off_frame_pending;
    // Lock order: helper_record_mutex -> transfer_mutex -> resource_mutex ->
    // queue mutexes. helper_record_mutex is outermost because the blocking
    // helpers it wraps take transfer_mutex internally for their allocation.
    Mutex                      helper_record_mutex;
    Mutex                      resource_mutex;
    Mutex                      transfer_mutex;
    Mutex                      graphics_queue_mutex;
    Mutex                      compute_queue_mutex;
    Mutex                      transfer_queue_mutex;
    bool                       compute_queue_distinct;
    Atomic{int}                inflight_calls;
    uint                       frames_in_flight;
    uint                       frame_index;
    FrameArenaState[]          frame_arenas;
    PersistentArenaState       persistent_arena;
    RecordingContextTable      recording_contexts;
    gpu::RecordingContextHandle default_context;
    // Reserved for the four blocking helpers; never touched by Tier-C
    // recording, so it never races the default context's pool.
    gpu::RecordingContextHandle helper_context;
    SemaphoreTable             semaphores;
    ShaderTable                shaders;
    PipelineTable              pipelines;
    PipelineCacheTable         pipeline_cache;
    vk::PipelineLayout         graphics_layout;
    ComputeLayoutEntry[COMPUTE_LAYOUT_CAP] compute_layouts;
    vk::PipelineCache          driver_cache;
    TextureTable               textures;
    SwapchainTable             swapchains;
    DescriptorHeapState        heap;
    TransferArenaState         staging_arena;
    TransferArenaState         readback_arena;
    DedicatedRetireState       dedicated_staging;
    // Thread-safe by construction (libc): deferred-queue growth may run on
    // any destroying thread, outside any allocator's tracking scope.
    HostAllocator               host_allocator;
    DeferredQueue              deferred;
    // Consumer-side monotonic lower bound of the retired timeline; written
    // only under resource_mutex, by a tier-3 query or begin_frame's post-wait
    // fold — never by a signaler, so it can never cover unfinished work.
    ulong                      signaled_shadow;
    DeferredStats              deferred_stats;
    // Advisory, best-effort across queue mutexes — mirrors get_memory_stats.
    SubmitStats                submit_stats;
    bool                       frame_active;
    // begin_frame's reset-loop accounting for the current call; written
    // only there, whitebox-exposed via pool_reset_stats.
    uint                       pools_reset_last_frame;
    uint                       pools_skipped_last_frame;
    // Helper-path pool resets outside begin_frame; whitebox-exposed.
    ulong                      helper_pool_resets;
    // Dedicated-fallback buffers created by transfer_alloc/ticket_alloc;
    // mutated only under transfer_mutex, whitebox-exposed.
    ulong                      dedicated_fallbacks;
}

<* Snapshot of begin_frame's reset-loop counters, exposed for whitebox
   dirty-skip accounting in tests (deferred_queue_count precedent). *>
struct PoolResetStats {
    uint reset;
    uint skipped;
}

struct CapsInputs {
    bool indexing_supported;
    bool buffer_supported;
    bool async_compute;
    bool line_polygon_mode;
    uint texture_capacity;
    uint sampler_capacity;
}

<* Frame-boundary quiescence accounting, active only under validation. *>
fn void enter_tracked(VkDeviceState* state) {
    if (state.validation) state.inflight_calls.add(1, AtomicOrdering.RELAXED);
}

fn void exit_tracked(VkDeviceState* state) {
    if (state.validation) state.inflight_calls.sub(1, AtomicOrdering.RELAXED);
}

fn bool boundary_blocked(VkDeviceState* state) {
    return state.validation && state.inflight_calls.load(AtomicOrdering.RELAXED) != 0;
}

fn PoolResetStats pool_reset_stats(VkDeviceState* state) {
    return { .reset = state.pools_reset_last_frame, .skipped = state.pools_skipped_last_frame };
}

// One static vtable shared by every Vulkan device.
const gpu::BackendVTable VK_VTABLE = {
    .destroy_device      = &vk_destroy_device,
    .get_memory_stats    = &vk_get_memory_stats,
    .build_memory_report = &vk_build_memory_report,
    .create_buffer       = &vk_create_buffer,
    .destroy_buffer      = &vk_destroy_buffer,
    .get_buffer_address  = &vk_get_buffer_address,
    .get_buffer_span     = &vk_get_buffer_span,
    .begin_frame         = &vk_begin_frame,
    .alloc_frame_span    = &vk_alloc_frame_span,
    .end_frame           = &vk_end_frame,
    .alloc_persistent_span = &vk_alloc_persistent_span,
    .free_persistent_span  = &vk_free_persistent_span,
    .get_persistent_stats  = &vk_get_persistent_stats,
    .begin_commands      = &vk_begin_commands,
    .create_recording_context  = &vk_create_recording_context,
    .destroy_recording_context = &vk_destroy_recording_context,
    .end_commands        = &vk_end_commands,
    .submit              = &vk_submit,
    .wait_queue_idle     = &vk_wait_queue_idle,
    .cmd_copy_buffer     = &vk_cmd_copy_buffer,
    .cmd_fill_buffer     = &vk_cmd_fill_buffer,
    .cmd_copy_buffer_to_texture = &vk_cmd_copy_buffer_to_texture,
    .cmd_copy_texture_to_buffer = &vk_cmd_copy_texture_to_buffer,
    .flush_buffer               = &vk_flush_buffer,
    .invalidate_buffer          = &vk_invalidate_buffer,
    .cmd_buffer_barrier  = &vk_cmd_buffer_barrier,
    .cmd_global_barrier  = &vk_cmd_global_barrier,
    .cmd_texture_barrier = &vk_cmd_texture_barrier,
    .create_semaphore    = &vk_create_semaphore,
    .destroy_semaphore   = &vk_destroy_semaphore,
    .wait_semaphore      = &vk_wait_semaphore,
    .create_texture          = &vk_create_texture,
    .destroy_texture         = &vk_destroy_texture,
    .create_texture_descriptor  = &vk_create_texture_descriptor,
    .destroy_texture_descriptor = &vk_destroy_texture_descriptor,
    .create_texture_descriptors = &vk_create_texture_descriptors,
    .create_sampler             = &vk_create_sampler,
    .destroy_sampler            = &vk_destroy_sampler,
    .create_shader           = &vk_create_shader,
    .destroy_shader          = &vk_destroy_shader,
    .create_compute_pipeline = &vk_create_compute_pipeline,
    .create_graphics_pipeline = &vk_create_graphics_pipeline,
    .destroy_pipeline        = &vk_destroy_pipeline,
    .get_pipeline_cache_size = &vk_get_pipeline_cache_size,
    .get_pipeline_cache_data = &vk_get_pipeline_cache_data,
    .cmd_dispatch            = &vk_cmd_dispatch,
    .cmd_begin_render_pass   = &vk_cmd_begin_render_pass,
    .cmd_end_render_pass     = &vk_cmd_end_render_pass,
    .cmd_draw                = &vk_cmd_draw,
    .cmd_draw_indexed        = &vk_cmd_draw_indexed,
    .cmd_draw_indirect       = &vk_cmd_draw_indirect,
    .cmd_draw_indexed_indirect = &vk_cmd_draw_indexed_indirect,
    .cmd_draw_indexed_indirect_count = &vk_cmd_draw_indexed_indirect_count,
    .cmd_dispatch_indirect   = &vk_cmd_dispatch_indirect,
    .cmd_begin_label         = &vk_cmd_begin_label,
    .cmd_end_label           = &vk_cmd_end_label,
    .cmd_upload_buffer       = &vk_cmd_upload_buffer,
    .cmd_upload_texture      = &vk_cmd_upload_texture,
    .upload_buffer_data      = &vk_upload_buffer_data,
    .upload_texture_data     = &vk_upload_texture_data,
    .readback_buffer_data    = &vk_readback_buffer_data,
    .readback_texture_data   = &vk_readback_texture_data,
    .cmd_readback_buffer     = &vk_cmd_readback_buffer,
    .cmd_readback_texture    = &vk_cmd_readback_texture,
    .poll_readback           = &vk_poll_readback,
    .resolve_readback        = &vk_resolve_readback,
    .create_swapchain        = &vk_create_swapchain,
    .destroy_swapchain       = &vk_destroy_swapchain,
    .resize_swapchain        = &vk_resize_swapchain,
    .acquire_next_image      = &vk_acquire_next_image,
    .present                 = &vk_present,
    .get_present_mode_support = &vk_get_present_mode_support,
};

<* Build a Vulkan device end to end. On failure the partial state is torn down
   and freed before the fault propagates, so nothing leaks. *>
fn gpu::Device? create_vk_device(gpu::DeviceDesc* desc) {
    VkDeviceState* state = mem::new(VkDeviceState);
    defer catch destroy_state(state);
    state.host_allocator = &allocators::LIBC_ALLOCATOR;
    state.helper_record_mutex.init()!;
    state.resource_mutex.init()!;
    state.transfer_mutex.init()!;
    state.graphics_queue_mutex.init()!;
    state.compute_queue_mutex.init()!;
    state.transfer_queue_mutex.init()!;

    state.texture_descriptor_capacity = gpu::resolve_descriptor_capacity(desc.texture_descriptor_capacity, gpu::DEFAULT_TEXTURE_DESCRIPTORS)!;
    state.sampler_descriptor_capacity = gpu::resolve_descriptor_capacity(desc.sampler_descriptor_capacity, gpu::DEFAULT_SAMPLER_DESCRIPTORS)!;

    state.validation = desc.enable_validation;
    state.presentation_enabled = desc.enable_presentation;
    state.instance = create_vk_instance(desc)!;
    vk::load_extensions(state.instance);
    if (desc.enable_validation) setup_debug_messenger(state);

    state.physical_device = pick_physical_device(state.instance, desc)!;

    vk::PhysicalDeviceVulkan13Features features13 = vk::physical_device_vulkan13features();
    vk::PhysicalDeviceVulkan12Features features12 = vk::physical_device_vulkan12features().set_next(&features13);
    vk::PhysicalDeviceVulkan11Features features11 = vk::physical_device_vulkan11features().set_next(&features12);
    vk::PhysicalDeviceFeatures2 features2 = vk::physical_device_features2().set_next(&features11);
    vk::get_physical_device_features2(state.physical_device, &features2);
    check_required_features(
        base: &features2.features,
        v11:  &features11,
        v12:  &features12,
        v13:  &features13,
    )!;
    state.draw_indirect_count = features12.draw_indirect_count != 0;
    bool line_polygon_mode = select_line_polygon_mode(&features2.features);

    if (features2.features.sampler_anisotropy != 0) {
        vk::PhysicalDeviceProperties anisotropy_props;
        vk::get_physical_device_properties(state.physical_device, &anisotropy_props);
        state.max_sampler_anisotropy = anisotropy_props.limits.max_sampler_anisotropy;
    }
    bool indexing_supported = heap_indexing_supported(&features2.features, &features12);
    bool buffer_supported   = heap_shared_features_supported(&features2.features, &features12) && query_descriptor_buffer_support(state.physical_device);
    state.heap_mode = resolve_heap_mode(desc.descriptor_heap_mode, indexing_supported, buffer_supported)!;

    QueueFamilies families = select_queue_families(state.physical_device)!;
    state.graphics_family = families.graphics;
    state.compute_family  = families.compute;
    state.transfer_family = families.transfer;

    create_logical_device(state, &families, line_polygon_mode)!;
    vk::get_device_queue(state.device, state.graphics_family, 0, &state.graphics_queue);
    state.compute_queue_distinct = families.async_compute;
    if (families.async_compute) {
        vk::get_device_queue(state.device, families.compute, families.compute_index, &state.compute_queue);
    } else {
        state.compute_queue = state.graphics_queue;
    }
    vk::get_device_queue(state.device, state.transfer_family, 0, &state.transfer_queue);

    create_vk_allocator(state)!;
    state.buffers.slots = mem::new_array(BufferCell, gpu::MAX_BUFFERS);
    create_frame_resources(state, desc)!;
    create_persistent_arena(state, desc)!;
    state.recording_contexts.slots = mem::new_array(RecordingContextCell, gpu::MAX_RECORDING_CONTEXTS);
    RecordingContextState default_ctx;
    create_context_pools(state, &default_ctx)!;
    state.default_context = state.recording_contexts.alloc(default_ctx)!;
    RecordingContextState helper_ctx;
    create_context_pools(state, &helper_ctx)!;
    state.helper_context = state.recording_contexts.alloc(helper_ctx)!;
    state.semaphores.slots = mem::new_array(SemaphoreCell, gpu::MAX_SEMAPHORES);
    state.shaders.slots    = mem::new_array(ShaderCell, gpu::MAX_SHADERS);
    uint pipeline_capacity = gpu::resolve_descriptor_capacity(desc.pipeline_capacity, gpu::MAX_PIPELINES)!;
    state.pipelines.slots  = mem::new_array(PipelineCell, pipeline_capacity);
    state.textures.slots   = mem::new_array(TextureCell, gpu::resolve_descriptor_capacity(desc.texture_capacity, gpu::DEFAULT_TEXTURE_CAPACITY)!);
    state.swapchains.slots = mem::new_array(SwapchainCell, gpu::MAX_SWAPCHAINS);
    state.deferred.entries = alloc::new_array(state.host_allocator, DeferredEntry, DEFERRED_QUEUE_INITIAL_CAP);
    create_transfer_arenas(state, desc)!;
    create_descriptor_heap(state)!;
    create_pipeline_shared(state, desc.pipeline_cache_data, pipeline_capacity)!;
    CapsInputs caps_inputs = {
        .indexing_supported = indexing_supported,
        .buffer_supported   = buffer_supported,
        .async_compute      = state.compute_queue_distinct,
        .line_polygon_mode  = line_polygon_mode,
        .texture_capacity   = state.texture_descriptor_capacity,
        .sampler_capacity   = state.sampler_descriptor_capacity,
    };
    state.caps = build_caps(
        base:            &features2.features,
        v12:             &features12,
        v13:             &features13,
        physical_device: state.physical_device,
        inputs:          &caps_inputs,
    );

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
        deferred_drain_all(state);
        destroy_deferred_queue(state);
        // Leak reporting first: the sweeps below free the tables it reads.
        if (state.validation) (void)report_resource_leaks(state);
        // Destroy any buffers the caller did not, so no VMA allocation leaks.
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
        if (state.swapchains.slots.ptr != null) {
            foreach (&cell : state.swapchains.slots[:state.swapchains.next_new]) {
                if (cell.used) destroy_swapchain_slot(state, &cell.value);
            }
            mem::free(state.swapchains.slots.ptr);
            state.swapchains.slots = {};
        }
        if (state.validation) report_descriptor_leaks(state);
        destroy_descriptor_heap(state);
        // Destroy any textures the caller did not, so no image or view leaks.
        if (state.textures.slots.ptr != null) {
            foreach (&cell : state.textures.slots[:state.textures.next_new]) {
                if (cell.used) {
                    destroy_cached_views(state, &cell.value);
                    vk::destroy_image_view(state.device, cell.value.default_view, null);
                    if (state.allocator != null) state.allocator.destroy_image(cell.value.image, cell.value.allocation);
                }
            }
            mem::free(state.textures.slots.ptr);
            state.textures.slots = {};
        }
        // Cache entries own the pipelines; slots only alias them.
        if (state.pipelines.slots.ptr != null) {
            mem::free(state.pipelines.slots.ptr);
            state.pipelines.slots = {};
        }
        destroy_pipeline_shared(state);
        if (state.shaders.slots.ptr != null) {
            foreach (&cell : state.shaders.slots[:state.shaders.next_new]) {
                if (cell.used) vk::destroy_shader_module(state.device, cell.value.shader_module, null);
            }
            mem::free(state.shaders.slots.ptr);
            state.shaders.slots = {};
        }
        if (state.recording_contexts.slots.ptr != null) {
            foreach (&cell : state.recording_contexts.slots[:state.recording_contexts.next_new]) {
                if (cell.used) destroy_context_pools(state, &cell.value);
            }
            mem::free(state.recording_contexts.slots.ptr);
            state.recording_contexts.slots = {};
        }
        if (state.semaphores.slots.ptr != null) {
            foreach (&cell : state.semaphores.slots[:state.semaphores.next_new]) {
                if (cell.used) vk::destroy_semaphore(state.device, cell.value.semaphore, null);
            }
            mem::free(state.semaphores.slots.ptr);
            state.semaphores.slots = {};
        }
        destroy_frame_resources(state);
        destroy_persistent_arena(state);
        if (state.allocator != null) {
            state.allocator.destroy();
            state.allocator = (vma::Allocator)null;
        }
        vk::destroy_device(state.device, null);
        state.device = (vk::Device)null;
    }
    destroy_debug_messenger(state);
    if (state.instance != null) {
        vk::destroy_instance(state.instance, null);
        state.instance = (vk::Instance)null;
    }
    if (state.helper_record_mutex.is_initialized())  (void)state.helper_record_mutex.destroy();
    if (state.resource_mutex.is_initialized())       (void)state.resource_mutex.destroy();
    if (state.transfer_mutex.is_initialized())       (void)state.transfer_mutex.destroy();
    if (state.graphics_queue_mutex.is_initialized()) (void)state.graphics_queue_mutex.destroy();
    if (state.compute_queue_mutex.is_initialized())  (void)state.compute_queue_mutex.destroy();
    if (state.transfer_queue_mutex.is_initialized()) (void)state.transfer_queue_mutex.destroy();
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

fn bool select_line_polygon_mode(vk::PhysicalDeviceFeatures* base) {
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

<* AUTO prefers indexing: lavapipe (Mesa 25.0.7) miscompiles descriptor-buffer
   image access, so descriptor buffer stays opt-in via the forced mode. Pure. *>
fn gpu::DescriptorHeapMode? resolve_heap_mode(gpu::DescriptorHeapMode requested, bool indexing_supported, bool buffer_supported) {
    switch (requested) {
        case gpu::DescriptorHeapMode.AUTO:
            if (indexing_supported) return gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING;
            if (buffer_supported)   return gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER;
            return gpu::UNSUPPORTED_FEATURE~;
        case gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER:
            if (!buffer_supported) return gpu::UNSUPPORTED_FEATURE~;
            return requested;
        case gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING:
            if (!indexing_supported) return gpu::UNSUPPORTED_FEATURE~;
            return requested;
    }
}

<* The feature struct is only chained once the extension is confirmed. *>
fn bool query_descriptor_buffer_support(vk::PhysicalDevice physical_device) @private {
    if (!device_supports_extension(physical_device, "VK_EXT_descriptor_buffer")) return false;
    vk::PhysicalDeviceDescriptorBufferFeaturesEXT db_features = vk::physical_device_descriptor_buffer_features_ext();
    vk::PhysicalDeviceFeatures2 features2 = vk::physical_device_features2().set_next(&db_features);
    vk::get_physical_device_features2(physical_device, &features2);
    return db_features.descriptor_buffer != 0;
}

fn bool candidate_satisfies(vk::PhysicalDevice candidate, gpu::DeviceDesc* desc) @private {
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

    bool indexing = heap_indexing_supported(&features2.features, &features12);
    bool buffer   = heap_shared_features_supported(&features2.features, &features12) && query_descriptor_buffer_support(candidate);
    if (catch resolve_heap_mode(desc.descriptor_heap_mode, indexing, buffer)) return false;
    return true;
}

<* Enumerate physical devices, drop candidates that cannot satisfy the desc
   (API version, required features, heap mode), return the highest-scored. *>
fn vk::PhysicalDevice? pick_physical_device(vk::Instance instance, gpu::DeviceDesc* desc) {
    uint count = 0;
    check(vk::enumerate_physical_devices(instance, &count, null))!;
    if (count == 0) return gpu::UNSUPPORTED_FEATURE~;

    @pool() {
        vk::PhysicalDevice[] devices = mem::talloc_array(vk::PhysicalDevice, (sz)count);
        check(vk::enumerate_physical_devices(instance, &count, devices.ptr))!;

        vk::PhysicalDevice best = (vk::PhysicalDevice)null;
        int best_score = -1;
        foreach (candidate : devices) {
            if (!candidate_satisfies(candidate, desc)) continue;
            vk::PhysicalDeviceProperties props;
            vk::get_physical_device_properties(candidate, &props);
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
   feature chain enabled. maintenance4 is enabled for SPIR-V 1.6 compute later;
   heap-path features follow state.heap_mode. Emits one queue create info per
   distinct family in use; the main family asks for a second queue when the
   compute tier lives there. *>
fn void? create_logical_device(VkDeviceState* state, QueueFamilies* families, bool line_polygon_mode) @private {
    float[2] priorities = { 1.0, 1.0 };
    uint main_queue_count = families.async_compute && families.compute == families.graphics ? 2 : 1;
    vk::DeviceQueueCreateInfo[MAX_QUEUE_FAMILIES] queue_infos;
    uint queue_info_count = 0;
    queue_infos[queue_info_count++] = vk::device_queue_create_info()
        .set_queue_family_index(families.graphics)
        .set_queue_priorities(priorities[:main_queue_count]);
    if (families.compute != families.graphics) {
        queue_infos[queue_info_count++] = vk::device_queue_create_info()
            .set_queue_family_index(families.compute)
            .set_queue_priorities(priorities[:1]);
    }
    if (families.transfer != families.graphics && families.transfer != families.compute) {
        queue_infos[queue_info_count++] = vk::device_queue_create_info()
            .set_queue_family_index(families.transfer)
            .set_queue_priorities(priorities[:1]);
    }

    vk::PhysicalDeviceDescriptorBufferFeaturesEXT db_features = vk::physical_device_descriptor_buffer_features_ext()
        .set_descriptor_buffer(vk::TRUE);
    vk::PhysicalDeviceVulkan13Features features13 = vk::physical_device_vulkan13features()
        .set_dynamic_rendering(vk::TRUE)
        .set_synchronization2(vk::TRUE)
        .set_maintenance4(vk::TRUE);
    if (state.heap_mode == gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER) {
        features13 = features13.set_next(&db_features);
    }
    vk::PhysicalDeviceVulkan12Features features12 = vk::physical_device_vulkan12features()
        .set_buffer_device_address(vk::TRUE)
        .set_timeline_semaphore(vk::TRUE)
        .set_runtime_descriptor_array(vk::TRUE)
        .set_shader_sampled_image_array_non_uniform_indexing(vk::TRUE)
        .set_shader_storage_image_array_non_uniform_indexing(vk::TRUE)
        .set_next(&features13);
    if (state.heap_mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
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
    features2.features.shader_storage_image_read_without_format  = vk::TRUE;
    features2.features.shader_storage_image_write_without_format = vk::TRUE;

    state.memory_budget_enabled = device_supports_extension(state.physical_device, "VK_EXT_memory_budget");
    ZString[3] extensions;
    uint extension_count = 0;
    if (state.memory_budget_enabled) extensions[extension_count++] = "VK_EXT_memory_budget";
    if (state.heap_mode == gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER) {
        extensions[extension_count++] = "VK_EXT_descriptor_buffer";
    }
    if (state.presentation_enabled) {
        if (!device_supports_extension(state.physical_device, "VK_KHR_swapchain")) return gpu::UNSUPPORTED_FEATURE~;
        extensions[extension_count++] = "VK_KHR_swapchain";
    }

    vk::DeviceCreateInfo device_info = vk::device_create_info()
        .set_next(&features2)
        .set_queue_create_infos(queue_infos[:queue_info_count]);
    if (extension_count > 0) {
        device_info = device_info.set_enabled_extension_names(extensions[:extension_count]);
    }

    check(vk::create_device(state.physical_device, &device_info, null, &state.device))!;
}

<* True if `physical_device` advertises the named device extension. *>
fn bool device_supports_extension(vk::PhysicalDevice physical_device, ZString name) @private {
    uint count = 0;
    if (catch vk::try_enumerate_device_extension_properties(physical_device, null, &count, null)) return false;
    if (count == 0) return false;

    @pool() {
        vk::ExtensionProperties[] properties = mem::talloc_array(vk::ExtensionProperties, (sz)count);
        if (catch vk::try_enumerate_device_extension_properties(physical_device, null, &count, properties.ptr)) return false;
        for (uint i = 0; i < count; i++) {
            if (((ZString)&properties[i].extension_name[0]).str_view() == name.str_view()) return true;
        }
    };
    return false;
}

<* Descriptor flags report adapter support per heap path; max counts report
   the configured capacities. *>
fn gpu::DeviceCaps build_caps(vk::PhysicalDeviceFeatures* base, vk::PhysicalDeviceVulkan12Features* v12, vk::PhysicalDeviceVulkan13Features* v13, vk::PhysicalDevice physical_device, CapsInputs* inputs) @private {
    vk::PhysicalDeviceProperties props;
    vk::get_physical_device_properties(physical_device, &props);
    return (gpu::DeviceCaps){
        .buffer_device_address      = v12.buffer_device_address != 0,
        .synchronization2           = v13.synchronization2 != 0,
        .dynamic_rendering          = v13.dynamic_rendering != 0,
        .timeline_semaphore         = v12.timeline_semaphore != 0,
        .shader_int64               = base.shader_int64 != 0,
        .draw_indirect_count        = v12.draw_indirect_count != 0,
        .descriptor_buffer          = inputs.buffer_supported,
        .descriptor_indexing        = inputs.indexing_supported,
        .async_compute              = inputs.async_compute,
        .line_polygon_mode          = inputs.line_polygon_mode,
        .max_texture_descriptors    = inputs.texture_capacity,
        .max_sampler_descriptors    = inputs.sampler_capacity,
        .min_uniform_alignment      = (usz)props.limits.min_uniform_buffer_offset_alignment,
        .min_storage_alignment      = (usz)props.limits.min_storage_buffer_offset_alignment,
        .min_texel_buffer_alignment = (usz)props.limits.min_texel_buffer_offset_alignment,
        .max_sampler_anisotropy     = base.sampler_anisotropy != 0 ? props.limits.max_sampler_anisotropy : 0.0f,
    };
}
