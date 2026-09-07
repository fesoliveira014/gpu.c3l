module gpu_test;

import gpu::util;
import gpu @public;
import gpu::internal @public;
import gpu::internal::vk @public;

struct ContextCallbackCapture {
    uint surface_calls;
    uint adapter_calls;
    Runtime runtime;
    Surface surface;
    Adapter selected;
    DeviceDesc requirements;
    bool return_selected;
    bool return_surface;
    fault callback_fault;
}

struct ContextDebugCapture {
    uint errors;
    bool swapchain_failure;
    bool allocator_failure;
    bool device_failure;
    bool surface_failure;
    bool runtime_failure;
    bool valid;
}

fn void reset_context_stub() {
    vk::stub_set_adapter_count(1);
    vk::stub_set_context_creation_failures(false, false);
    vk::stub_reset_runtime_counts();
    vk::stub_reset_device_counts();
    vk::stub_reset_surface_counts();
    vk::stub_reset_context_counts();
}

fn Surface? context_surface_factory(Runtime* runtime, void* user_data) {
    ContextCallbackCapture* capture = (ContextCallbackCapture*)user_data;
    capture.surface_calls++;
    capture.runtime = *runtime;
    if (capture.callback_fault) return capture.callback_fault~;
    if (capture.return_surface || capture.surface.is_valid()) return capture.surface;
    capture.surface = create_test_surface(runtime);
    return capture.surface;
}

fn Adapter? context_adapter_selector(Runtime* runtime, DeviceDesc* requirements, void* user_data) {
    ContextCallbackCapture* capture = (ContextCallbackCapture*)user_data;
    capture.adapter_calls++;
    capture.runtime = *runtime;
    capture.requirements = *requirements;
    if (capture.callback_fault) return capture.callback_fault~;
    if (capture.return_selected) return capture.selected;
    capture.selected = gpu::enumerate_adapters(runtime)!!.get(0)!!;
    return capture.selected;
}

fn void context_debug_callback(DebugMessage* message, void* user_data) {
    ContextDebugCapture* capture = (ContextDebugCapture*)user_data;
    if (message.severity != DebugMessageSeverity.ERROR) return;
    capture.errors++;
    capture.valid = capture.valid && message.category.resource_lifetime
        && message.has_fault && message.public_fault == gpu::RESOURCE_IN_USE;
    String operation = message.operation.str_view();
    if (operation == "destroy_swapchain") capture.swapchain_failure = true;
    if (operation == "destroy_command_allocator") capture.allocator_failure = true;
    if (operation == "destroy_device") capture.device_failure = true;
    if (operation == "destroy_surface") capture.surface_failure = true;
    if (operation == "destroy_runtime") capture.runtime_failure = true;
}

fn void assert_context_empty(util::DeviceContext* context) {
    assert(!context.runtime.is_valid());
    assert(!context.surface.is_valid());
    assert(!context.adapter.is_valid());
    assert(!context.device.is_valid());
    assert(!context.queue.device.is_valid());
    assert(context.queue.id == 0 && (uint)context.queue.roles == 0);
    assert(!context.command_allocator.handle.is_valid());
    assert(!context.command_allocator.device.is_valid());
    assert(!context.swapchain.is_valid());
    assert(context.debug_callback == null && context.debug_user_data == null);
}

fn void context_defaults_enable_contract_checks_without_layers() @test {
    util::DeviceContextDesc desc = util::default_device_context_desc();
    assert(desc.runtime.contract_validation == gpu::ContractValidation.FULL);
    assert(!desc.runtime.enable_vulkan_validation);
    assert(desc.command_queue == gpu::QueueKind.GRAPHICS);
    assert(desc.command_allocator.command_buffer_capacity == gpu::DEFAULT_COMMAND_ALLOCATOR_CAPACITY);
    assert(desc.swapchain.present_mode == gpu::PresentMode.FIFO);
    assert(desc.runtime.texture_capacity == gpu::DEFAULT_TEXTURE_CAPACITY);
    assert(desc.runtime.texture_heap_capacity == gpu::DEFAULT_TEXTURE_HEAP_CAPACITY);
    assert(desc.runtime.sampler_heap_capacity == gpu::DEFAULT_SAMPLER_HEAP_CAPACITY);
    assert(desc.runtime.pipeline_capacity == gpu::MAX_PIPELINES);
    assert(desc.command_allocator.max_resource_references_per_list == gpu::DEFAULT_COMMAND_REFERENCES_PER_LIST);
    assert(desc.device.queues.required.graphics);
    assert(desc.device.queues.required.compute);
    assert(desc.device.queues.required.transfer);
    assert(!desc.device.enable_sparse_textures && !desc.device.enable_ray_queries);
    assert(!desc.device.enable_ray_tracing_pipelines && !desc.device.enable_mesh_shaders);
    assert(desc.swapchain.preferred_format == gpu::Format.BGRA8_UNORM);
    assert(desc.swapchain.width == 0 && desc.swapchain.height == 0);
    assert(desc.swapchain.image_count == 0);
    assert(desc.create_surface == null && desc.select_adapter == null);
}

fn void context_headless_owns_exact_adapter_device_queue_and_allocator() @test {
    reset_context_stub();
    util::DeviceContextDesc desc = util::default_device_context_desc();
    util::DeviceContext context = util::create_device_context(&desc)!!;
    assert(context.runtime.is_valid() && context.device.is_valid());
    assert(context.adapter == vk::stub_device_adapter(&context.device)!!);
    assert(context.queue == gpu::get_queue(&context.device, QueueKind.GRAPHICS)!!);
    assert(context.command_allocator.device == context.device);
    assert(!context.surface.is_valid() && !context.swapchain.is_valid());
    assert(vk::stub_swapchain_create_attempt_count() == 0);

    util::destroy_device_context(&context);
    assert_context_empty(&context);
    assert(vk::stub_allocator_destroy_attempt_count() == 1);
    assert(vk::stub_device_temporary_destroy_count() == 1);
    assert(vk::stub_runtime_destroy_count() == 1);
    assert(vk::stub_context_wait_count() == 0);
}

fn void context_ranks_every_supported_adapter_class() @test {
    reset_context_stub();
    defer vk::stub_set_adapter_count(1);
    AdapterClass[5] classes = {
        AdapterClass.SOFTWARE, AdapterClass.OTHER, AdapterClass.VIRTUAL,
        AdapterClass.INTEGRATED, AdapterClass.DISCRETE,
    };
    util::DeviceContextDesc desc = util::default_device_context_desc();
    for (uint count = 2; count <= classes.len; count++) {
        vk::stub_set_adapter_count(count);
        for (uint index = 0; index < count; index++) {
            vk::stub_set_adapter_class(index, classes[index]);
        }
        util::DeviceContext context = util::create_device_context(&desc)!!;
        assert(context.adapter.adapter_index == count - 1);
        util::destroy_device_context(&context);
    }
}

fn void context_skips_unsupported_discrete_and_keeps_first_tie() @test {
    reset_context_stub();
    defer vk::stub_set_adapter_count(1);
    vk::stub_set_adapter_count(3);
    vk::stub_set_adapter_class(0, AdapterClass.INTEGRATED);
    vk::stub_set_adapter_class(1, AdapterClass.DISCRETE);
    vk::stub_set_adapter_class(2, AdapterClass.INTEGRATED);
    vk::stub_set_adapter_supported(1, false);
    util::DeviceContextDesc desc = util::default_device_context_desc();
    util::DeviceContext context = util::create_device_context(&desc)!!;
    assert(context.adapter.adapter_index == 0);
    util::destroy_device_context(&context);
}

fn void context_no_supported_adapter_releases_runtime() @test {
    reset_context_stub();
    defer vk::stub_set_adapter_count(1);
    vk::stub_set_adapter_supported(0, false);
    util::DeviceContextDesc desc = util::default_device_context_desc();
    assert(@catch(util::create_device_context(&desc)) == gpu::UNSUPPORTED_FEATURE);
    assert(vk::stub_device_temporary_create_count() == 0);
    assert(vk::stub_runtime_destroy_count() == 1);
}

fn void context_does_not_retry_authoritative_device_failure() @test {
    reset_context_stub();
    defer vk::stub_set_adapter_count(1);
    vk::stub_set_adapter_count(2);
    vk::stub_fail_device_after_temporary_allocation(true);
    defer vk::stub_fail_device_after_temporary_allocation(false);
    util::DeviceContextDesc desc = util::default_device_context_desc();
    assert(@catch(util::create_device_context(&desc)) == gpu::BACKEND_ERROR);
    assert(vk::stub_device_temporary_create_count() == 1);
    assert(vk::stub_device_temporary_destroy_count() == 1);
    assert(vk::stub_runtime_destroy_count() == 1);
}

fn void context_compute_override_preserves_descriptor_and_capacities() @test {
    reset_context_stub();
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.device.queues.required = { .compute };
    desc.command_queue = QueueKind.COMPUTE;
    desc.command_allocator.command_buffer_capacity = 7;
    desc.command_allocator.max_resource_references_per_list = 13;
    util::DeviceContextDesc original = desc;
    for (uint index = 0; index < 2; index++) {
        util::DeviceContext context = util::create_device_context(&desc)!!;
        assert(context.queue.roles.compute && !context.queue.roles.graphics);
        assert(@catch(gpu::get_queue(&context.device, QueueKind.GRAPHICS)) == gpu::INVALID_ARGUMENT);
        vk::VkDeviceState* state = present_wait_state(&context.device);
        assert(state.allocator_desc.command_buffer_capacity == 7);
        assert(state.allocator_desc.max_resource_references_per_list == 13);
        assert(device_desc_equal(desc.device, original.device));
        assert(desc.command_queue == original.command_queue);
        assert(desc.command_allocator.command_buffer_capacity == 7);
        util::destroy_device_context(&context);
    }
}

fn void context_zero_descriptor_preserves_core_defaults() @test {
    reset_context_stub();
    util::DeviceContextDesc desc;
    util::DeviceContext context = util::create_device_context(&desc)!!;
    assert(desc.runtime.contract_validation == ContractValidation.TRUSTED);
    assert(desc.command_allocator.command_buffer_capacity == 0);
    assert(context.queue.roles.graphics);
    util::destroy_device_context(&context);
}

fn void context_invalid_policy_fails_before_runtime_creation() @test {
    reset_context_stub();
    util::DeviceContextDesc desc = util::default_device_context_desc();
    assert(@catch(util::create_device_context(null)) == gpu::INVALID_ARGUMENT);
    desc.surface_user_data = (void*)1;
    assert(@catch(util::create_device_context(&desc)) == gpu::INVALID_ARGUMENT);
    desc.surface_user_data = null;
    desc.adapter_user_data = (void*)1;
    assert(@catch(util::create_device_context(&desc)) == gpu::INVALID_ARGUMENT);
    desc.adapter_user_data = null;
    desc.device.surface = { .index = 1, .generation = 1 };
    assert(@catch(util::create_device_context(&desc)) == gpu::INVALID_ARGUMENT);
    desc.device.surface = {};
    desc.device.queues.required = { .compute };
    assert(@catch(util::create_device_context(&desc)) == gpu::INVALID_ARGUMENT);
    desc.command_queue = bitcast(99, QueueKind);
    assert(@catch(util::create_device_context(&desc)) == gpu::INVALID_ARGUMENT);
    assert(vk::stub_runtime_create_count() == 0);
}

fn void context_custom_selector_receives_effective_private_requirements() @test {
    reset_context_stub();
    defer vk::stub_set_adapter_count(1);
    vk::stub_set_adapter_count(2);
    vk::stub_set_adapter_class(0, AdapterClass.INTEGRATED);
    vk::stub_set_adapter_class(1, AdapterClass.DISCRETE);
    ContextCallbackCapture capture;
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.device.enable_ray_queries = true;
    desc.runtime.acceleration_structure_heap_capacity = 1;
    desc.select_adapter = &context_adapter_selector;
    desc.adapter_user_data = &capture;
    util::DeviceContext context = util::create_device_context(&desc)!!;
    assert(capture.adapter_calls == 1 && capture.runtime == context.runtime);
    assert(capture.requirements.enable_ray_queries);
    assert(desc.device.enable_ray_queries);
    assert(context.adapter == capture.selected);
    assert(context.adapter.adapter_index == 0);
    assert(gpu::get_device_caps(&context.device)!!.ray_queries.enabled);
    util::destroy_device_context(&context);
}

fn void context_custom_selector_rejects_foreign_stale_and_zero_adapters() @test {
    reset_context_stub();
    RuntimeDesc runtime_desc;
    Runtime foreign = gpu::create_runtime(&runtime_desc)!!;
    ContextCallbackCapture capture = {
        .selected = gpu::enumerate_adapters(&foreign)!!.get(0)!!,
        .return_selected = true,
    };
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.select_adapter = &context_adapter_selector;
    desc.adapter_user_data = &capture;
    assert(@catch(util::create_device_context(&desc)) == gpu::INVALID_HANDLE);
    assert(gpu::enumerate_adapters(&foreign)!!.count == 1);
    gpu::destroy_runtime(&foreign)!!;
    assert(@catch(util::create_device_context(&desc)) == gpu::INVALID_HANDLE);
    capture.selected = {};
    assert(@catch(util::create_device_context(&desc)) == gpu::INVALID_HANDLE);
    assert(vk::stub_device_temporary_create_count() == 0);
}

fn void context_custom_selector_fault_and_unsupported_choice_propagate() @test {
    reset_context_stub();
    defer vk::stub_set_adapter_count(1);
    ContextCallbackCapture capture = { .callback_fault = gpu::SLOT_TABLE_FULL };
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.select_adapter = &context_adapter_selector;
    desc.adapter_user_data = &capture;
    assert(@catch(util::create_device_context(&desc)) == gpu::SLOT_TABLE_FULL);
    capture.callback_fault = {};
    vk::stub_set_adapter_supported(0, false);
    assert(@catch(util::create_device_context(&desc)) == gpu::UNSUPPORTED_FEATURE);
    assert(vk::stub_device_temporary_create_count() == 0);
    assert(vk::stub_runtime_destroy_count() == 2);
}

fn void context_presentation_transfers_same_runtime_surface_once() @test {
    reset_context_stub();
    ContextCallbackCapture capture;
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.create_surface = &context_surface_factory;
    desc.surface_user_data = &capture;
    desc.select_adapter = &context_adapter_selector;
    desc.adapter_user_data = &capture;
    desc.swapchain.width = 640;
    desc.swapchain.height = 480;
    util::DeviceContext context = util::create_device_context(&desc)!!;
    assert(capture.surface_calls == 1 && capture.adapter_calls == 1);
    assert(capture.runtime == context.runtime && capture.surface == context.surface);
    assert(capture.requirements.surface == context.surface);
    assert(!desc.device.surface.is_valid());
    assert(context.swapchain.is_valid());
    assert_device_ownership(&context.device, context.adapter, context.surface);
    util::destroy_device_context(&context);
    assert_context_empty(&context);
    assert(vk::stub_swapchain_destroy_attempt_count() == 1);
    assert(vk::stub_allocator_destroy_attempt_count() == 1);
    assert(vk::stub_surface_destroy_count() == 1);
    assert(vk::stub_runtime_destroy_count() == 1);
    assert(vk::stub_context_wait_count() == 0);
}

fn void context_empty_adapter_list_never_calls_surface_factory() @test {
    reset_context_stub();
    defer vk::stub_set_adapter_count(1);
    vk::stub_set_adapter_count(0);
    ContextCallbackCapture capture;
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.create_surface = &context_surface_factory;
    desc.surface_user_data = &capture;
    assert(@catch(util::create_device_context(&desc)) == gpu::UNSUPPORTED_FEATURE);
    assert(capture.surface_calls == 0);
    assert(vk::stub_runtime_destroy_count() == 1);
}

fn void context_surface_factory_fault_releases_runtime() @test {
    reset_context_stub();
    ContextCallbackCapture capture = { .callback_fault = gpu::BACKEND_ERROR };
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.create_surface = &context_surface_factory;
    desc.surface_user_data = &capture;
    assert(@catch(util::create_device_context(&desc)) == gpu::BACKEND_ERROR);
    assert(capture.surface_calls == 1);
    assert(vk::stub_surface_create_count() == 0);
    assert(vk::stub_runtime_destroy_count() == 1);
}

fn void context_rejects_foreign_surface_without_destroying_it() @test {
    reset_context_stub();
    RuntimeDesc runtime_desc;
    Runtime foreign = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&foreign)!!;
    ContextCallbackCapture capture = { .surface = create_test_surface(&foreign) };
    defer gpu::destroy_surface(&capture.surface)!!;
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.create_surface = &context_surface_factory;
    desc.surface_user_data = &capture;
    assert(@catch(util::create_device_context(&desc)) == gpu::INVALID_HANDLE);
    assert(vk::stub_surface_destroy_count() == 0);
    Adapter adapter = gpu::enumerate_adapters(&foreign)!!.get(0)!!;
    assert(gpu::supports_presentation(&adapter, &capture.surface)!!);
    assert(vk::stub_runtime_destroy_count() == 1);
}

fn void context_rejects_zero_surface_and_releases_unsupported_surface() @test {
    reset_context_stub();
    ContextCallbackCapture capture = { .return_surface = true };
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.create_surface = &context_surface_factory;
    desc.surface_user_data = &capture;
    assert(@catch(util::create_device_context(&desc)) == gpu::INVALID_HANDLE);
    assert(vk::stub_surface_destroy_count() == 0);
    assert(vk::stub_runtime_destroy_count() == 1);

    capture.return_surface = false;
    vk::stub_set_presentation_support(false);
    defer vk::stub_set_presentation_support(true);
    assert(@catch(util::create_device_context(&desc)) == gpu::UNSUPPORTED_FEATURE);
    assert(vk::stub_surface_create_count() == 1);
    assert(vk::stub_surface_destroy_count() == 1);
    assert(vk::stub_runtime_destroy_count() == 2);
}

fn void context_allocator_failure_rolls_back_headless_and_presentation() @test {
    for (uint presentation = 0; presentation < 2; presentation++) {
        reset_context_stub();
        vk::stub_set_context_creation_failures(true, false);
        ContextCallbackCapture capture;
        util::DeviceContextDesc desc = util::default_device_context_desc();
        if (presentation != 0) {
            desc.create_surface = &context_surface_factory;
            desc.surface_user_data = &capture;
        }
        assert(@catch(util::create_device_context(&desc)) == gpu::DEVICE_BUSY);
        assert(vk::stub_device_temporary_destroy_count() == 1);
        assert(vk::stub_surface_destroy_count() == presentation);
        assert(vk::stub_runtime_destroy_count() == 1);
        assert(vk::stub_swapchain_create_attempt_count() == 0);
        assert(vk::stub_context_wait_count() == 0);
    }
    vk::stub_set_context_creation_failures(false, false);
}

fn void context_rejects_stale_surface_without_destroying_replacement() @test {
    reset_context_stub();
    RuntimeDesc runtime_desc;
    Runtime foreign = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&foreign)!!;
    Surface original = create_test_surface(&foreign);
    ContextCallbackCapture capture = { .surface = original };
    gpu::destroy_surface(&original)!!;
    Surface replacement = create_test_surface(&foreign);
    defer gpu::destroy_surface(&replacement)!!;
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.create_surface = &context_surface_factory;
    desc.surface_user_data = &capture;
    assert(@catch(util::create_device_context(&desc)) == gpu::INVALID_HANDLE);
    assert(vk::stub_surface_destroy_count() == 1);
    Adapter adapter = gpu::enumerate_adapters(&foreign)!!.get(0)!!;
    assert(gpu::supports_presentation(&adapter, &replacement)!!);
}

fn void context_swapchain_failure_releases_prior_owners() @test {
    reset_context_stub();
    vk::stub_set_context_creation_failures(false, true);
    defer vk::stub_set_context_creation_failures(false, false);
    ContextCallbackCapture capture;
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.create_surface = &context_surface_factory;
    desc.surface_user_data = &capture;
    assert(@catch(util::create_device_context(&desc)) == gpu::BACKEND_ERROR);
    assert(vk::stub_allocator_destroy_attempt_count() == 1);
    assert(vk::stub_device_temporary_destroy_count() == 1);
    assert(vk::stub_surface_destroy_count() == 1);
    assert(vk::stub_runtime_destroy_count() == 1);
    assert(vk::stub_context_wait_count() == 0);
}

fn void context_rollback_preserves_original_fault_when_cleanup_fails() @test {
    reset_context_stub();
    vk::stub_set_context_creation_failures(false, true, true);
    defer vk::stub_set_context_creation_failures(false, false);
    ContextCallbackCapture capture;
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.create_surface = &context_surface_factory;
    desc.surface_user_data = &capture;
    assert(@catch(util::create_device_context(&desc)) == gpu::BACKEND_ERROR);
    assert(vk::stub_allocator_destroy_attempt_count() == 1);
    assert(vk::stub_surface_destroy_count() == 1);
    assert(vk::stub_device_temporary_destroy_count() == 0);
    assert(vk::stub_runtime_destroy_count() == 0);
    assert(vk::stub_context_wait_count() == 0);

    Device device = vk::stub_last_created_device();
    vk::VkDeviceState* state = present_wait_state(&device);
    state.destroy_busy = false;
    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&capture.runtime)!!;
}

fn void context_teardown_reports_each_failure_continues_and_clears() @test {
    reset_context_stub();
    ContextDebugCapture debug = { .valid = true };
    ContextCallbackCapture capture;
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.runtime.debug_callback = &context_debug_callback;
    desc.runtime.debug_user_data = &debug;
    desc.create_surface = &context_surface_factory;
    desc.surface_user_data = &capture;
    util::DeviceContext context = util::create_device_context(&desc)!!;
    Runtime runtime = context.runtime;
    Device device = context.device;
    CommandAllocator allocator = context.command_allocator;
    vk::VkDeviceState* state = present_wait_state(&context.device);
    state.reject_allocator_destroy = true;
    util::destroy_device_context(&context);
    assert_context_empty(&context);
    assert(debug.valid && debug.errors == 3);
    assert(debug.allocator_failure && debug.device_failure);
    assert(!debug.surface_failure && debug.runtime_failure);
    assert(vk::stub_surface_destroy_count() == 1);
    assert(vk::stub_swapchain_destroy_attempt_count() == 1);
    assert(vk::stub_allocator_destroy_attempt_count() == 1);
    assert(vk::stub_context_wait_count() == 0);
    state.reject_allocator_destroy = false;
    gpu::destroy_command_allocator(&allocator)!!;
    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void context_swapchain_teardown_failure_still_releases_allocator() @test {
    reset_context_stub();
    ContextDebugCapture debug = { .valid = true };
    ContextCallbackCapture capture;
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.runtime.debug_callback = &context_debug_callback;
    desc.runtime.debug_user_data = &debug;
    desc.create_surface = &context_surface_factory;
    desc.surface_user_data = &capture;
    util::DeviceContext context = util::create_device_context(&desc)!!;
    Runtime runtime = context.runtime;
    Surface surface = context.surface;
    Device device = context.device;
    SwapchainHandle swapchain = context.swapchain;
    vk::VkDeviceState* state = present_wait_state(&context.device);
    state.reject_swapchain_destroy = true;
    util::destroy_device_context(&context);
    assert_context_empty(&context);
    assert(debug.valid && debug.errors == 4);
    assert(debug.swapchain_failure && !debug.allocator_failure);
    assert(debug.device_failure && debug.surface_failure && debug.runtime_failure);
    assert(vk::stub_allocator_destroy_attempt_count() == 1);
    assert(vk::stub_surface_destroy_count() == 0);
    assert(vk::stub_context_wait_count() == 0);

    state.reject_swapchain_destroy = false;
    gpu::destroy_swapchain(&device, swapchain)!!;
    gpu::destroy_device(&device)!!;
    gpu::destroy_surface(&surface)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void context_zero_teardown_is_harmless() @test {
    reset_context_stub();
    util::DeviceContext context;
    util::destroy_device_context(&context);
    util::destroy_device_context(&context);
    assert_context_empty(&context);
    assert(vk::stub_allocator_destroy_attempt_count() == 0);
    assert(vk::stub_runtime_destroy_count() == 0);
}
