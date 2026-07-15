module gpu_test;

import gpu;
import gpu::vk @public;
import vk;

const gpu::BackendVTable DEVICE_LOSS_VTABLE = {
    .destroy_device = &destroy_device_loss_fixture,
};

struct BackendResultCapture {
    uint calls;
    bool fields_match;
}

fn void destroy_device_loss_fixture(void* backend_state) {
    (void)backend_state;
}

fn gpu::Device device_loss_fixture(gpu::vk::VkDeviceState* state) {
    return gpu::register_device(
        backend:       gpu::BackendKind.VULKAN,
        caps:          { .strict_enabled = true },
        vtable:        &DEVICE_LOSS_VTABLE,
        backend_state: state,
        owner:         &state.owner,
    )!!;
}

fn void capture_backend_result(gpu::DebugMessage* message, void* user_data) {
    BackendResultCapture* capture = (BackendResultCapture*)user_data;
    capture.calls++;
    capture.fields_match = message.severity == gpu::DebugMessageSeverity.ERROR
        && message.category.backend
        && message.operation != null
        && message.operation.str_view() == "wait_queue_idle"
        && message.has_fault
        && message.public_fault == gpu::BACKEND_ERROR
        && message.backend_text != null
        && message.backend_text.str_view() == "unmapped Vulkan result"
        && message.validation_id_number == -999;
}

fn void backend_result_diagnostic_preserves_unmapped_fault() @test {
    BackendResultCapture capture;
    gpu::vk::VkDeviceState configured = {
        .debug_callback  = &capture_backend_result,
        .debug_user_data = &capture,
    };
    fault configured_fault = @catch(gpu::vk::check_backend_result(
        state:     &configured,
        operation: "wait_queue_idle",
        result:    (vk::Result)-999,
    ));

    assert(configured_fault == gpu::BACKEND_ERROR, "configured backend result remains BACKEND_ERROR");
    assert(capture.calls == 1, "configured backend result dispatches exactly once");
    assert(capture.fields_match, "callback observes operation, native code, backend text, and fault during delivery");

    gpu::vk::VkDeviceState fallback = {};
    fault fallback_fault = @catch(gpu::vk::check_backend_result(
        state:     &fallback,
        operation: "wait_queue_idle",
        result:    (vk::Result)-999,
    ));
    assert(fallback_fault == configured_fault, "stderr fallback preserves exact fault parity");
    assert(capture.calls == 1, "null callback does not dispatch a structured duplicate");
}

fn void explicit_device_loss_invalidates_only_its_owner() @test {
    gpu::vk::VkDeviceState first_state;
    gpu::vk::VkDeviceState second_state;
    gpu::Device first = device_loss_fixture(&first_state);
    gpu::Device second = device_loss_fixture(&second_state);

    assert(
        @catch(gpu::vk::check_backend_result(
            state:     &first_state,
            operation: "submit",
            result:    (vk::Result)-999,
        )) == gpu::BACKEND_ERROR,
        "an unknown native result remains a backend error"
    );
    assert(gpu::get_device_caps(&first)!!.strict_enabled, "an unknown result leaves its owner usable");
    assert(gpu::get_device_caps(&second)!!.strict_enabled, "an unknown result leaves a peer usable");

    assert(
        @catch(gpu::vk::check_backend_result(
            state:     &first_state,
            operation: "submit",
            result:    vk::Result.ERROR_DEVICE_LOST,
        )) == gpu::DEVICE_LOST,
        "explicit native loss remains exact"
    );
    assert(
        @catch(gpu::get_device_caps(&first)) == gpu::DEVICE_LOST,
        "the reporting device becomes unusable"
    );
    assert(gpu::get_device_caps(&second)!!.strict_enabled, "a peer device remains usable");

    gpu::destroy_device(&first)!!;
    gpu::destroy_device(&second)!!;
}

fn void custom_result_routes_mark_the_reporting_device() @test {
    gpu::vk::VkDeviceState texture_state;
    gpu::Device texture_device = device_loss_fixture(&texture_state);
    gpu::vk::TextureSlot texture_slot;
    assert(
        @catch(gpu::vk::check_texture_view_backend_result(
            state:     &texture_state,
            operation: null,
            texture:   {},
            slot:      &texture_slot,
            result:    vk::Result.ERROR_DEVICE_LOST,
        )) == gpu::DEVICE_LOST,
        "texture-view loss remains exact"
    );
    assert(@catch(gpu::get_device_caps(&texture_device)) == gpu::DEVICE_LOST, "texture-view loss marks its owner");
    gpu::destroy_device(&texture_device)!!;

    gpu::vk::VkDeviceState helper_state;
    gpu::Device helper_device = device_loss_fixture(&helper_state);
    assert(
        @catch(gpu::vk::check_helper_backend_result(
            state:           &helper_state,
            operation:       "upload_buffer_data",
            result:          vk::Result.ERROR_DEVICE_LOST,
            fault_is_public: false,
        )) == gpu::DEVICE_LOST,
        "swallowed-helper loss remains exact"
    );
    assert(@catch(gpu::get_device_caps(&helper_device)) == gpu::DEVICE_LOST, "helper loss marks its owner");
    gpu::destroy_device(&helper_device)!!;

    gpu::vk::VkDeviceState wsi_state;
    gpu::Device wsi_device = device_loss_fixture(&wsi_state);
    gpu::vk::WsiDiagnosticContext context = {
        .state     = &wsi_state,
        .operation = "present",
    };
    assert(
        @catch(gpu::vk::check_wsi_backend_result(
            context: &context,
            kind:    gpu::vk::WsiResultKind.PRESENT,
            result:  vk::Result.ERROR_DEVICE_LOST,
        )) == gpu::DEVICE_LOST,
        "presentation loss remains exact"
    );
    assert(@catch(gpu::get_device_caps(&wsi_device)) == gpu::DEVICE_LOST, "presentation loss marks its owner");
    gpu::destroy_device(&wsi_device)!!;
}

fn void generic_result_mapping_is_truthful() @test {
    gpu::vk::check(vk::Result.SUCCESS)!!;

    fault host_oom = @catch(gpu::vk::check(vk::Result.ERROR_OUT_OF_HOST_MEMORY));
    fault device_oom = @catch(gpu::vk::check(vk::Result.ERROR_OUT_OF_DEVICE_MEMORY));
    fault device_lost = @catch(gpu::vk::check(vk::Result.ERROR_DEVICE_LOST));
    fault unknown = @catch(gpu::vk::check(vk::Result.ERROR_UNKNOWN));
    fault format = @catch(gpu::vk::check(vk::Result.ERROR_FORMAT_NOT_SUPPORTED));
    fault initialization = @catch(gpu::vk::check(vk::Result.ERROR_INITIALIZATION_FAILED));
    fault arbitrary = @catch(gpu::vk::check((vk::Result)-999));

    assert(host_oom == gpu::OUT_OF_HOST_MEMORY, "host OOM remains exact");
    assert(device_oom == gpu::OUT_OF_DEVICE_MEMORY, "device OOM remains exact");
    assert(device_lost == gpu::DEVICE_LOST, "explicit device loss remains exact");
    assert(unknown == gpu::BACKEND_ERROR, "unknown native result is not device loss");
    assert(format == gpu::BACKEND_ERROR, "format result needs operation context");
    assert(initialization == gpu::BACKEND_ERROR, "initialization failure needs operation context");
    assert(arbitrary == gpu::BACKEND_ERROR, "arbitrary native result is truthful backend error");
}

fn void backend_and_wsi_result_mapping_is_specific() @test {
    fault init = @catch(gpu::vk::check_backend_create_result(vk::Result.ERROR_INITIALIZATION_FAILED));
    fault driver = @catch(gpu::vk::check_backend_create_result(vk::Result.ERROR_INCOMPATIBLE_DRIVER));
    fault physical_init = @catch(gpu::vk::check_physical_device_enumeration_result(vk::Result.ERROR_INITIALIZATION_FAILED));
    fault surface = @catch(gpu::vk::check_surface_result(vk::Result.ERROR_SURFACE_LOST_KHR));
    fault surface_window = @catch(gpu::vk::check_surface_result(vk::Result.ERROR_NATIVE_WINDOW_IN_USE_KHR));
    fault surface_fill = @catch(gpu::vk::check_surface_enumeration_result(vk::Result.ERROR_SURFACE_LOST_KHR));
    fault surface_incomplete = @catch(gpu::vk::check_surface_enumeration_result(vk::Result.INCOMPLETE));
    fault out_of_date = @catch(gpu::vk::check_swapchain_create_result(vk::Result.ERROR_OUT_OF_DATE_KHR));
    fault window_in_use = @catch(gpu::vk::check_swapchain_create_result(vk::Result.ERROR_NATIVE_WINDOW_IN_USE_KHR));
    fault image_usage = @catch(gpu::vk::check_swapchain_create_result(vk::Result.ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR));
    fault acquire_surface = @catch(gpu::vk::check_acquire_result(vk::Result.ERROR_SURFACE_LOST_KHR));
    fault present_surface = @catch(gpu::vk::check_present_result(vk::Result.ERROR_SURFACE_LOST_KHR));

    assert(init == gpu::UNSUPPORTED_BACKEND, "backend initialization failure is bootstrap-specific");
    assert(driver == gpu::UNSUPPORTED_BACKEND, "incompatible driver is unsupported backend");
    assert(physical_init == gpu::UNSUPPORTED_BACKEND, "physical-device initialization failure remains bootstrap-specific");
    assert(surface == gpu::SURFACE_LOST, "surface loss remains recoverable and distinct");
    assert(surface_window == gpu::INVALID_RESOURCE_STATE, "native surface ownership is a resource state");
    assert(surface_fill == gpu::SURFACE_LOST, "surface loss remains exact during enumeration");
    assert(surface_incomplete == gpu::BACKEND_ERROR, "partial surface enumeration is a backend error");
    assert(out_of_date == gpu::SWAPCHAIN_OUT_OF_DATE, "swapchain obsolescence remains distinct");
    assert(window_in_use == gpu::INVALID_RESOURCE_STATE, "native window ownership is a resource state");
    assert(image_usage == gpu::UNSUPPORTED_FEATURE, "unsupported swapchain image usage is a capability result");
    assert(acquire_surface == gpu::SURFACE_LOST, "acquire preserves surface loss");
    assert(present_surface == gpu::SURFACE_LOST, "present preserves surface loss");
}

fn void resource_result_mapping_is_specific() @test {
    fault format = @catch(gpu::vk::check_texture_create_result(vk::Result.ERROR_FORMAT_NOT_SUPPORTED));
    fault usage = @catch(gpu::vk::check_texture_create_result(vk::Result.ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR));
    fault shader = @catch(gpu::vk::check_shader_create_result(vk::Result.ERROR_INVALID_SHADER_NV));
    fault shader_validation = @catch(gpu::vk::check_shader_create_result(vk::Result.ERROR_VALIDATION_FAILED_EXT));
    fault descriptor = @catch(gpu::vk::check_descriptor_allocate_result(vk::Result.ERROR_OUT_OF_POOL_MEMORY));
    fault pool_fragmentation = @catch(gpu::vk::check_descriptor_pool_create_result(vk::Result.ERROR_FRAGMENTATION));
    fault fragmented_pool = @catch(gpu::vk::check_descriptor_allocate_result(vk::Result.ERROR_FRAGMENTED_POOL));
    fault arena = @catch(gpu::vk::check_virtual_allocate_result(vk::Result.ERROR_OUT_OF_DEVICE_MEMORY));
    fault pure_arena = gpu::vk::virtual_allocate_result_fault(vk::Result.ERROR_OUT_OF_DEVICE_MEMORY);

    assert(format == gpu::UNSUPPORTED_FEATURE, "unsupported image format is a capability result");
    assert(usage == gpu::UNSUPPORTED_FEATURE, "unsupported image usage is a capability result");
    assert(shader == gpu::SHADER_INVALID, "invalid native shader result is shader invalid");
    assert(shader_validation == gpu::SHADER_INVALID, "shader validation failure is shader invalid");
    assert(descriptor == gpu::DESCRIPTOR_HEAP_FULL, "descriptor pool exhaustion is heap full");
    assert(pool_fragmentation == gpu::DESCRIPTOR_HEAP_FULL, "descriptor pool fragmentation is heap full");
    assert(fragmented_pool == gpu::DESCRIPTOR_HEAP_FULL, "fragmented descriptor allocation is heap full");
    assert(arena == gpu::ARENA_FULL, "virtual block exhaustion is arena full");
    assert(pure_arena == arena, "legacy and context-aware virtual allocation share one taxonomy");
}

fn void pipeline_and_cache_result_mapping_preserves_fatal_faults() @test {
    fault pipeline_unknown = @catch(gpu::vk::check_pipeline_create_result(vk::Result.ERROR_UNKNOWN));
    fault pipeline_shader = @catch(gpu::vk::check_pipeline_create_result(vk::Result.ERROR_INVALID_SHADER_NV));
    fault pipeline_compile = @catch(gpu::vk::check_pipeline_create_result(vk::Result.PIPELINE_COMPILE_REQUIRED));
    fault pipeline_oom = @catch(gpu::vk::check_pipeline_create_result(vk::Result.ERROR_OUT_OF_DEVICE_MEMORY));
    fault pipeline_lost = @catch(gpu::vk::check_pipeline_create_result(vk::Result.ERROR_DEVICE_LOST));
    fault cache_incomplete = @catch(gpu::vk::check_pipeline_cache_data_result(vk::Result.INCOMPLETE));
    fault enumeration_incomplete = @catch(gpu::vk::check_enumeration_result(vk::Result.INCOMPLETE));
    bool unknown_retryable = gpu::vk::pipeline_cache_blob_retryable(vk::Result.ERROR_UNKNOWN);
    bool validation_retryable = gpu::vk::pipeline_cache_blob_retryable(vk::Result.ERROR_VALIDATION_FAILED_EXT);

    assert(pipeline_unknown == gpu::PIPELINE_CREATE_FAILED, "pipeline unknown is operation-specific");
    assert(pipeline_shader == gpu::PIPELINE_CREATE_FAILED, "pipeline invalid shader is creation failure");
    assert(pipeline_compile == gpu::PIPELINE_CREATE_FAILED, "compile-required status is creation failure");
    assert(pipeline_oom == gpu::OUT_OF_DEVICE_MEMORY, "pipeline device OOM is not collapsed");
    assert(pipeline_lost == gpu::DEVICE_LOST, "pipeline device loss remains exact");
    assert(cache_incomplete == gpu::INVALID_ARGUMENT, "cache export growth reports undersized output");
    assert(enumeration_incomplete == gpu::BACKEND_ERROR, "partial enumeration is not device loss");

    assert(unknown_retryable, "unknown warm blob rejection retries empty");
    assert(validation_retryable, "validation warm blob rejection retries empty");
    assert(!gpu::vk::pipeline_cache_blob_retryable(vk::Result.SUCCESS), "success never retries");
    assert(!gpu::vk::pipeline_cache_blob_retryable(vk::Result.ERROR_OUT_OF_HOST_MEMORY), "host OOM never retries");
    assert(!gpu::vk::pipeline_cache_blob_retryable(vk::Result.ERROR_OUT_OF_DEVICE_MEMORY), "device OOM never retries");
    assert(!gpu::vk::pipeline_cache_blob_retryable(vk::Result.ERROR_DEVICE_LOST), "device loss never retries");
}

fn void wait_timeout_mapping_stays_retryable() @test {
    fault timeout = @catch(gpu::vk::check_wait(vk::Result.TIMEOUT));
    fault pure_timeout = gpu::vk::wait_result_fault(vk::Result.TIMEOUT);
    fault pure_device_lost = gpu::vk::wait_result_fault(vk::Result.ERROR_DEVICE_LOST);
    assert(timeout == gpu::WAIT_TIMEOUT, "bounded wait timeout remains retryable");
    assert(pure_timeout == timeout, "legacy and context-aware waits share one timeout taxonomy");
    assert(pure_device_lost == gpu::DEVICE_LOST, "pure wait mapping preserves generic device loss");
}
