module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::atomic::types;
import std::core::mem::alloc;

struct PolicyDiagnosticCapture {
    uint calls;
}

fn void capture_policy_diagnostic(
    gpu::DebugMessage* message,
    void* user_data,
) {
    (void)message;
    PolicyDiagnosticCapture* capture = (PolicyDiagnosticCapture*)user_data;
    capture.calls++;
}

fn gpu::RuntimeDesc policy_runtime_desc(
    gpu::ContractValidation contract,
    bool vulkan_layers,
    bool callback,
    bool debug_names,
    PolicyDiagnosticCapture* capture,
) {
    return {
        .contract_validation      = contract,
        .enable_vulkan_validation = vulkan_layers,
        .enable_debug_names       = debug_names,
        .application_name         = "gpu_validation_policy_test",
        .debug_callback           = callback ? &capture_policy_diagnostic : null,
        .debug_user_data          = callback ? capture : null,
    };
}

fn vk::VkDeviceState* policy_device_state(gpu::Device* device) {
    return gpu::internal::device_slots[device.index].data.state;
}

fn gpu::CommandList record_policy_fill(
    gpu::CommandAllocator* allocator,
    gpu::GpuSpan span,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::cmd_fill_buffer(
        commands: &commands,
        dst:      span,
        value:    0xA5A5_A5A5,
    )!!;
    return commands;
}

fn OwnedSpan policy_span(
    gpu::Device* device,
    gpu::QueueRoles access,
    ZString debug_name,
) {
    return allocate_owned_span(
        device:       device,
        size:         256,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       access,
        debug_name:   debug_name,
    )!!;
}

fn void policy_matrix_creates_exact_runtime_and_device_state() @test {
    gpu::ContractValidation[2] contracts = {
        gpu::ContractValidation.TRUSTED,
        gpu::ContractValidation.FULL,
    };
    bool[2] values = { false, true };

    foreach (contract : contracts) {
        foreach (layers : values) {
            foreach (callback : values) {
                foreach (debug_names : values) {
                    PolicyDiagnosticCapture capture;
                    gpu::RuntimeDesc desc = policy_runtime_desc(
                        contract:      contract,
                        vulkan_layers: layers,
                        callback:      callback,
                        debug_names:   debug_names,
                        capture:       &capture,
                    );
                    gpu::Device device = create_test_device(&desc)!!;
                    vk::VkDeviceState* state = policy_device_state(&device);
                    assert(state.validation_policy.contract == contract,
                        "device preserves the contract policy");
                    assert(state.validation_policy.vulkan_layers == layers,
                        "device preserves the independent layer policy");
                    assert(state.debug_names == debug_names,
                        "device preserves the debug-name policy");
                    assert((state.debug_callback != null) == callback,
                        "device preserves callback presence independently");
                    destroy_test_device(&device)!!;
                }
            }
        }
    }
}

fn void both_modes_reject_null_command_record() @test {
    gpu::ContractValidation[2] contracts = {
        gpu::ContractValidation.TRUSTED,
        gpu::ContractValidation.FULL,
    };

    foreach (contract : contracts) {
        PolicyDiagnosticCapture capture;
        gpu::RuntimeDesc desc = policy_runtime_desc(
            contract:      contract,
            vulkan_layers: false,
            callback:      false,
            debug_names:   false,
            capture:       &capture,
        );
        gpu::Device device = create_test_device(&desc)!!;
        gpu::CommandList fabricated = { .generation = 1 };
        fault rejected = @catch(gpu::end_commands(&fabricated));
        assert(rejected == gpu::INVALID_HANDLE,
            "every contract policy rejects a null command record");
        destroy_test_device(&device)!!;
    }
}

fn void trusted_keeps_barrier_stage_shape_as_caller_contract() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:      gpu::ContractValidation.TRUSTED,
        vulkan_layers: false,
        callback:      false,
        debug_names:   false,
        capture:       &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator =
        gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::Barrier barrier = {
        .before = { .all = true, .compute = true },
        .after  = { .compute = true },
    };

    gpu::cmd_barrier(&commands, &barrier)!!;

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
}

fn void instance_plans_callback_names_and_layers_independently() @test {
    vk::DebugUtilsMessengerCreateInfoEXT messenger_info;
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:      gpu::ContractValidation.FULL,
        vulkan_layers: false,
        callback:      false,
        debug_names:   false,
        capture:       &capture,
    );

    vk::RuntimeConfig config = {
        .validation = {
            .contract      = desc.contract_validation,
            .vulkan_layers = desc.enable_vulkan_validation,
        },
        .debug_names    = desc.enable_debug_names,
        .debug_callback = desc.debug_callback,
    };
    vk::VkInstanceDesc plan = vk::runtime_instance_desc(
        config:           &config,
        application_name: desc.application_name,
        messenger_info:   &messenger_info,
    );
    assert(!plan.enable_validation && !plan.enable_debug_utils,
        "contract and tracking policies request no instance instrumentation");

    config.debug_callback = &capture_policy_diagnostic;
    plan = vk::runtime_instance_desc(
        config:           &config,
        application_name: desc.application_name,
        messenger_info:   &messenger_info,
    );
    assert(!plan.enable_validation && plan.enable_debug_utils,
        "callback-only planning requests debug utils without layers");
    assert(plan.debug_messenger_info == null,
        "callback-only planning does not chain the layer messenger");

    config.debug_callback = null;
    config.debug_names = true;
    plan = vk::runtime_instance_desc(
        config:           &config,
        application_name: desc.application_name,
        messenger_info:   &messenger_info,
    );
    assert(!plan.enable_validation && plan.enable_debug_utils,
        "names-only planning requests debug utils without layers");

    config.debug_names = false;
    config.validation.vulkan_layers = true;
    plan = vk::runtime_instance_desc(
        config:           &config,
        application_name: desc.application_name,
        messenger_info:   &messenger_info,
    );
    assert(plan.enable_validation && plan.enable_debug_utils,
        "layer planning requests layers and debug utils");
    assert(plan.debug_messenger_info == &messenger_info,
        "layer planning alone chains the validation messenger");
}

fn void trusted_command_completion_retires_its_unit() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.TRUSTED,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    OwnedSpan owned = policy_span(
        &device,
        { .graphics },
        "trusted_completion",
    );
    defer free_owned_span(&device, &owned)!!;

    gpu::CommandList commands = record_policy_fill(&allocator, owned.span);

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;

    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        &submit,
    )!!;
    vk::QueueCompletionState* completion_state =
        vk::queue_completion_state(state, queue.id)!!;
    gpu::wait_completion(completion)!!;
    assert(completion_state.submitted_count == 0,
        "completion retires the trusted command batch");
    gpu::free_allocation(&device, &owned.allocation)!!;
    owned.span = {};
}

fn void contract_diagnostics_and_callback_presence_are_independent() @test {
    bool[2] callbacks = { false, true };
    foreach (callback : callbacks) {
        PolicyDiagnosticCapture capture;
        gpu::RuntimeDesc desc = policy_runtime_desc(
            contract:       gpu::ContractValidation.FULL,
            vulkan_layers:   false,
            callback:        callback,
            debug_names:     false,
            capture:         &capture,
        );
        gpu::Device device = create_test_device(&desc)!!;
        capture = {};
        gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
        gpu::TextureDesc texture_desc = {
            .access = { .graphics },
            .width  = 8,
            .height = 8,
            .format = gpu::Format.RGBA8_UNORM,
            .usage  = { .transfer_dst },
        };
        gpu::TextureHandle texture = gpu::create_texture(
            &device,
            &texture_desc,
        )!!;
        gpu::CommandList commands = gpu::begin_commands(
            &allocator,
        )!!;
        gpu::TextureBarrier barrier = gpu::texture_transition(
            texture: texture,
            before:  { .layout = gpu::TextureLayout.UNDEFINED },
            after:   {
                .layout = gpu::TextureLayout.SAMPLED,
                .stages = { .indirect },
                .access = { .read },
            },
        )!!;
        fault misuse = @catch(gpu::cmd_texture_barrier(&commands, &barrier));
        assert(misuse == gpu::INVALID_ARGUMENT,
            "full command checks are independent of callback presence");
        assert(capture.calls == (callback ? 1 : 0),
            "configured callbacks observe but never enable full checks");
        gpu::discard_commands(&commands)!!;
        gpu::destroy_texture(&device, texture)!!;
        gpu::destroy_command_allocator(&allocator)!!;
        destroy_test_device(&device)!!;
    }
}

fn void trusted_callback_and_names_enable_no_command_checks() @test {
    bool[2] callbacks = { false, true };
    foreach (callback : callbacks) {
        PolicyDiagnosticCapture capture;
        gpu::RuntimeDesc desc = policy_runtime_desc(
            contract:       gpu::ContractValidation.TRUSTED,
            vulkan_layers:   false,
            callback:        callback,
            debug_names:     true,
            capture:         &capture,
        );
        gpu::Device device = create_test_device(&desc)!!;
        capture = {};
        gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
        gpu::TextureDesc texture_desc = {
            .access = { .graphics },
            .width  = 8,
            .height = 8,
            .format = gpu::Format.RGBA8_UNORM,
            .usage  = { .transfer_dst },
        };
        gpu::TextureHandle texture = gpu::create_texture(
            &device,
            &texture_desc,
        )!!;
        gpu::CommandList commands = gpu::begin_commands(
            &allocator,
        )!!;
        gpu::TextureBarrier barrier = gpu::texture_transition(
            texture: texture,
            before:  { .layout = gpu::TextureLayout.UNDEFINED },
            after:   {
                .layout = gpu::TextureLayout.SAMPLED,
                .stages = { .indirect },
                .access = { .read },
            },
        )!!;
        gpu::cmd_texture_barrier(&commands, &barrier)!!;
        assert(capture.calls == 0,
            "callback presence alone produces no contract diagnostic");
        gpu::discard_commands(&commands)!!;
        gpu::destroy_texture(&device, texture)!!;
        gpu::destroy_command_allocator(&allocator)!!;
        destroy_test_device(&device)!!;
    }
}

fn void trusted_commands_retain_the_mandatory_safety_floor() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.TRUSTED,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(
        &allocator,
    )!!;
    defer discard_test_commands(&commands)!!;

    assert(@catch(gpu::cmd_copy_buffer(&commands, null))
        == gpu::INVALID_ARGUMENT,
        "trusted commands reject null host descriptors");
}

fn void trusted_rejects_foreign_and_stale_resources() @test {
    PolicyDiagnosticCapture first_capture;
    PolicyDiagnosticCapture second_capture;
    gpu::RuntimeDesc first_desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.TRUSTED,
        vulkan_layers:   false,
        callback:        true,
        debug_names:     false,
        capture:         &first_capture,
    );
    gpu::RuntimeDesc second_desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.TRUSTED,
        vulkan_layers:   false,
        callback:        true,
        debug_names:     false,
        capture:         &second_capture,
    );
    gpu::Device first = create_test_device(&first_desc)!!;
    defer destroy_test_device(&first)!!;
    gpu::Device second = create_test_device(&second_desc)!!;
    defer destroy_test_device(&second)!!;
    first_capture = {};
    second_capture = {};
    gpu::Queue first_queue = gpu::get_queue(&first, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator first_allocator = gpu::create_command_allocator(&first, first_queue)!!;
    defer gpu::destroy_command_allocator(&first_allocator)!!;
    OwnedSpan owned = policy_span(&first, { .graphics }, "boundary_owned");

    gpu::CommandList commands = record_policy_fill(&first_allocator, owned.span);
    gpu::discard_commands(&commands)!!;

    gpu::GpuAllocation foreign = owned.allocation;
    assert(@catch(gpu::free_allocation(&second, &foreign))
        == gpu::INVALID_HANDLE,
        "trusted mode rejects foreign ownership");
    gpu::GpuAllocation stale = owned.allocation;
    gpu::free_allocation(&first, &owned.allocation)!!;
    owned.span = {};
    assert(@catch(gpu::get_allocation_info(&first, stale))
        == gpu::INVALID_HANDLE,
        "trusted mode rejects stale identities");
}

fn void teardown_leak_scans_depend_only_on_contract() @test {
    vk::VkDeviceState state;
    state.validation_policy.contract = gpu::ContractValidation.TRUSTED;
    assert(!vk::report_teardown_leaks(&state),
        "trusted teardown without a callback skips accepted leak scans");

    state.debug_callback = &capture_policy_diagnostic;
    assert(!vk::report_teardown_leaks(&state),
        "callback presence does not enable trusted teardown leak scans");

    state.validation_policy.contract = gpu::ContractValidation.FULL;
    assert(vk::report_teardown_leaks(&state),
        "full policy enables accepted teardown leak scans");
}

fn void structured_diagnostics_follow_contract_not_callback_presence() @test {
    PolicyDiagnosticCapture capture;
    vk::VkDeviceState state = {
        .debug_callback  = &capture_policy_diagnostic,
        .debug_user_data = &capture,
    };

    state.validation_policy.contract = gpu::ContractValidation.TRUSTED;
    fault returned = vk::vk_report_root_contract_failure(
        state:          &state,
        operation:      "free_allocation",
        public_fault:   gpu::INVALID_HANDLE,
        resource_kind:  gpu::DebugResourceKind.ALLOCATION,
        rejected_field: "GpuAllocation",
        invariant:      "must belong to the device",
    );
    assert(returned == gpu::INVALID_HANDLE && capture.calls == 0,
        "trusted boundaries preserve faults without detailed diagnostics");

    state.validation_policy.contract = gpu::ContractValidation.FULL;
    returned = vk::vk_report_root_contract_failure(
        state:          &state,
        operation:      "free_allocation",
        public_fault:   gpu::INVALID_HANDLE,
        resource_kind:  gpu::DebugResourceKind.ALLOCATION,
        rejected_field: "GpuAllocation",
        invariant:      "must belong to the device",
    );
    assert(returned == gpu::INVALID_HANDLE && capture.calls == 1,
        "full mode emits the configured contract diagnostic");

    state.debug_callback = null;
    returned = vk::vk_report_root_contract_failure(
        state:          &state,
        operation:      "free_allocation",
        public_fault:   gpu::INVALID_HANDLE,
        resource_kind:  gpu::DebugResourceKind.ALLOCATION,
        rejected_field: "GpuAllocation",
        invariant:      "must belong to the device",
    );
    assert(returned == gpu::INVALID_HANDLE && capture.calls == 1,
        "callback absence never changes the boundary fault");
}

fn uint policy_live_device_count() {
    uint count;
    for (uint i = 0; i < gpu::internal::MAX_DEVICES; i++) {
        gpu::internal::DeviceSlotControl control = (gpu::internal::DeviceSlotControl)
            gpu::internal::device_slots[i].control.load(AtomicOrdering.RELAXED);
        if (control.state == gpu::internal::DEVICE_SLOT_LIVE) count++;
    }
    return count;
}

fn void partial_creation_rolls_back_across_both_contract_modes() @test {
    gpu::ContractValidation[2] contracts = {
        gpu::ContractValidation.TRUSTED,
        gpu::ContractValidation.FULL,
    };

    foreach (contract : contracts) {
        PolicyDiagnosticCapture capture;
        gpu::RuntimeDesc desc = policy_runtime_desc(
            contract:       contract,
            vulkan_layers:   false,
            callback:        false,
            debug_names:     false,
            capture:         &capture,
        );
        gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
        gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
        bool exercised;
        for (uint i = 0; i < adapters.count; i++) {
            gpu::Adapter adapter = adapters.get(i)!!;
            if (!gpu::supports_device_desc(&adapter)!!
                .supported) continue;
            exercised = true;
            gpu::Runtime retained_runtime = runtime;
            gpu::internal::runtime_retain_device(&retained_runtime)!!;
            vk::DeviceCreationProbe probe = {
                .fail_at = vk::DeviceCreationCheckpoint.BEFORE_PUBLICATION,
            };
            uint live_before = policy_live_device_count();
            assert(@catch(vk::create_vk_device_with_probe(
                adapter:          &adapter,
                device_desc:      null,
                retained_runtime: &retained_runtime,
                probe:            &probe,
            )) == gpu::BACKEND_ERROR,
                "partial creation returns its injected backend fault");
            gpu::internal::runtime_release_device(&retained_runtime)!!;
            assert(probe.reached
                    == vk::DeviceCreationCheckpoint.BEFORE_PUBLICATION
                && probe.rollback_count == 1,
                "partial creation reaches and rolls back exactly once");
            assert(policy_live_device_count() == live_before,
                "partial creation publishes no device slot");
            break;
        }
        assert(exercised,
            "the pinned Vulkan adapter supports the device baseline");
        gpu::destroy_runtime(&runtime)!!;
    }
}

fn void device_loss_discards_commands_under_each_policy() @test {
    gpu::ContractValidation[2] contracts = {
        gpu::ContractValidation.TRUSTED,
        gpu::ContractValidation.FULL,
    };
    foreach (contract : contracts) {
        PolicyDiagnosticCapture capture;
        gpu::RuntimeDesc desc = policy_runtime_desc(
            contract:       contract,
            vulkan_layers:   false,
            callback:        false,
            debug_names:     false,
            capture:         &capture,
        );
        gpu::Device device = create_test_device(&desc)!!;
        gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
        OwnedSpan owned = policy_span(&device, { .graphics }, "lost_command");
        gpu::CommandList commands = record_policy_fill(&allocator, owned.span);

        gpu::internal::mark_device_lost(&device);
        gpu::discard_commands(&commands)!!;

        gpu::destroy_command_allocator(&allocator)!!;
        destroy_test_device(&device)!!;
        assert(!device.is_valid(),
            "device-loss teardown consumes devices in either contract mode");
    }
}
