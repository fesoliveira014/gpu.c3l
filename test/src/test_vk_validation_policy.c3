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

fn void trusted_rejects_retired_descriptor_hazard_bit() @test {
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
        .before  = { .compute = true },
        .after   = { .compute = true },
        .hazards = (gpu::HazardFlags)0x2u,
    };

    fault rejected = @catch(gpu::cmd_barrier(&commands, &barrier));
    assert(rejected == gpu::INVALID_ARGUMENT,
        "trusted validation rejects the retired descriptor hazard bit");

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

    vk::VkInstanceDesc plan = vk::runtime_instance_desc(&desc, &messenger_info);
    assert(!plan.enable_validation && !plan.enable_debug_utils,
        "contract and tracking policies request no instance instrumentation");

    desc.debug_callback = &capture_policy_diagnostic;
    plan = vk::runtime_instance_desc(&desc, &messenger_info);
    assert(!plan.enable_validation && plan.enable_debug_utils,
        "callback-only planning requests debug utils without layers");
    assert(plan.debug_messenger_info == null,
        "callback-only planning does not chain the layer messenger");

    desc.debug_callback = null;
    desc.enable_debug_names = true;
    plan = vk::runtime_instance_desc(&desc, &messenger_info);
    assert(!plan.enable_validation && plan.enable_debug_utils,
        "names-only planning requests debug utils without layers");

    desc.enable_debug_names = false;
    desc.enable_vulkan_validation = true;
    plan = vk::runtime_instance_desc(&desc, &messenger_info);
    assert(plan.enable_validation && plan.enable_debug_utils,
        "layer planning requests layers and debug utils");
    assert(plan.debug_messenger_info == &messenger_info,
        "layer planning alone chains the validation messenger");
}

fn void trusted_keeps_reference_storage_empty_through_retirement() @test {
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
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    assert(record.scratch.references.ptr == null
        && record.scratch.reference_count == 0,
        "trusted recording owns no reference storage");

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    record = executable.record;
    assert(record.scratch.references.ptr == null
        && record.scratch.reference_count == 0,
        "trusted executable owns no reference storage");
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        &submit,
    )!!;
    vk::QueueCompletionState* completion_state =
        vk::queue_completion_state(state, queue.id)!!;
    if (completion_state.submitted_head != null) {
        vk::CommandRecord* submitted = completion_state.submitted_head;
        vk::CommandAllocatorSlot* allocator_slot =
            state.command_allocators.get(submitted.allocator)!!;
        vk::CommandScratch* scratch =
            &allocator_slot.scratch[submitted.allocator_buffer];
        assert(scratch.references.ptr == null
            && scratch.reference_count == 0,
            "trusted submission transfers empty reference storage");
    }
    gpu::wait_completion(completion)!!;
    assert(completion_state.submitted_count == 0,
        "completion retires the trusted command batch");
    gpu::free_allocation(&device, &owned.allocation)!!;
    owned.span = {};
}

fn void trusted_allows_early_caller_owned_destruction() @test {
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
    OwnedSpan owned = policy_span(&device, { .graphics }, "caller_owned");

    gpu::CommandList commands = record_policy_fill(&allocator, owned.span);
    gpu::free_allocation(&device, &owned.allocation)!!;
    owned.span = {};
    gpu::discard_commands(&commands)!!;
    commands = gpu::begin_commands(
        &allocator,
    )!!;
    vk::CommandRecord* reused = vk::resolve_command(&commands)!!;
    assert(reused.scratch.references.ptr == null
        && reused.scratch.reference_count == 0,
        "trusted command reuse starts from empty reference storage");
    gpu::discard_commands(&commands)!!;
}

fn void full_rejects_destruction_until_each_lifecycle_release() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.FULL,
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

    OwnedSpan recording = policy_span(&device, { .graphics }, "recording");
    Atomic{uint}* recording_counter =
        &state.allocations.get(
            recording.allocation,
        )!!.tracked_command_references;
    gpu::CommandList commands = record_policy_fill(&allocator, recording.span);
    assert(recording_counter.load(AtomicOrdering.RELAXED) == 1,
        "recording owns one retained resource reference");
    assert(@catch(gpu::free_allocation(&device, &recording.allocation))
        == gpu::RESOURCE_IN_USE,
        "full validation rejects destruction during recording");
    gpu::discard_commands(&commands)!!;
    assert(recording_counter.load(AtomicOrdering.RELAXED) == 0,
        "recording discard releases its one retained resource exactly once");
    gpu::free_allocation(&device, &recording.allocation)!!;
    recording.span = {};

    OwnedSpan executable = policy_span(&device, { .graphics }, "executable");
    Atomic{uint}* executable_counter =
        &state.allocations.get(
            executable.allocation,
        )!!.tracked_command_references;
    commands = record_policy_fill(&allocator, executable.span);
    gpu::ExecutableCommandList list = gpu::end_commands(&commands)!!;
    assert(executable_counter.load(AtomicOrdering.RELAXED) == 1,
        "executable commands own one retained resource reference");
    assert(@catch(gpu::free_allocation(&device, &executable.allocation))
        == gpu::RESOURCE_IN_USE,
        "full validation rejects destruction while executable");
    gpu::discard_executable_commands(&list)!!;
    assert(executable_counter.load(AtomicOrdering.RELAXED) == 0,
        "executable discard releases its retained resource exactly once");
    gpu::free_allocation(&device, &executable.allocation)!!;
    executable.span = {};

    OwnedSpan submitted = policy_span(&device, { .graphics }, "submitted");
    Atomic{uint}* submitted_counter =
        &state.allocations.get(
            submitted.allocation,
        )!!.tracked_command_references;
    commands = record_policy_fill(&allocator, submitted.span);
    list = gpu::end_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { list };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        &submit,
    )!!;
    assert(submitted_counter.load(AtomicOrdering.RELAXED) == 1,
        "submitted commands own one retained resource reference");
    assert(@catch(gpu::free_allocation(&device, &submitted.allocation))
        == gpu::RESOURCE_IN_USE,
        "full validation rejects destruction before submitted work is retired");
    gpu::wait_completion(completion)!!;
    assert(submitted_counter.load(AtomicOrdering.RELAXED) == 0,
        "completion retirement releases its retained resource exactly once");
    gpu::free_allocation(&device, &submitted.allocation)!!;
    submitted.span = {};
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
            after:   gpu::sampled_at({ .fragment_shader }),
        )!!;
        gpu::reset_command_resolution_stats();
        fault misuse = @catch(gpu::cmd_texture_barrier(&commands, &barrier));
        assert(misuse == gpu::INVALID_ARGUMENT,
            "full command checks are independent of callback presence");
        assert(gpu::command_resolution_stats().native_commands == 0,
            "full rejection precedes native command emission");
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
            after:   gpu::sampled_at({ .fragment_shader }),
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

    gpu::reset_command_resolution_stats();
    assert(@catch(gpu::cmd_copy_buffer(&commands, null))
        == gpu::INVALID_ARGUMENT,
        "trusted commands reject null host descriptors");
    assert(gpu::command_resolution_stats().native_commands == 0,
        "mandatory safety rejection performs no native call");
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

fn void command_reference_clear_releases_the_live_list() @test {
    Atomic{uint}[2] tracked;
    vk::CommandResourceReference[2] references;
    foreach (i, &counter : tracked) {
        counter.store(1, AtomicOrdering.RELAXED);
        references[i] = {
            .owner            = 19,
            .index            = (uint)i,
            .generation       = 1,
            .retained_counter = counter,
        };
    }
    vk::CommandScratch scratch = {
        .references      = references[..],
        .reference_count = 2,
    };

    vk::clear_command_scratch_references(&scratch);

    assert(tracked[0].load(AtomicOrdering.RELAXED) == 0
        && tracked[1].load(AtomicOrdering.RELAXED) == 0
        && scratch.reference_count == 0,
        "clear releases every live reference exactly once");
    vk::release_tracked_command_references({}, 0);
    assert(tracked[0].load(AtomicOrdering.RELAXED) == 0
        && tracked[1].load(AtomicOrdering.RELAXED) == 0,
        "an empty release preserves every counter");
}

fn void submission_transfer_accepts_empty_and_populated_reference_storage() @test {
    vk::VkDeviceState state = {
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    state.command_mutex.init()!!;
    defer (void)state.command_mutex.destroy();
    gpu::Queue queue = {
        .device = { .generation = 1 },
        .id     = 0,
        .roles  = { .graphics },
    };
    Atomic{uint} tracked;
    tracked.store(1, AtomicOrdering.RELAXED);
    vk::CommandResourceReference[1] populated_references = {{
        .retained_counter = &tracked,
    }};
    vk::CommandScratch[2] scratch = {{
        .in_use = true,
    }, {
        .references      = populated_references[..],
        .reference_count = 1,
        .in_use          = true,
    }};
    uint[2] available;
    vk::CommandAllocatorCell[1] allocator_cells;
    state.command_allocators = {
        .slots = allocator_cells[..],
        .owner = 1,
    };
    gpu::CommandAllocatorHandle allocator = state.command_allocators.alloc({
        .queue            = queue,
        .scratch          = scratch[..],
        .available        = available[..],
        .live_executables = 1,
    })!!;
    vk::CommandAllocatorSlot* allocator_slot =
        state.command_allocators.get(allocator)!!;
    allocator_slot.mutex.init()!!;
    defer (void)allocator_slot.mutex.destroy();

    vk::QueueCompletionState[1] completions = {{ .bound = true }};
    state.queue_completions = completions[..];
    completions[0].retirement_mutex.init()!!;
    defer (void)completions[0].retirement_mutex.destroy();
    vk::CommandCell[2] command_cells;
    state.commands = {
        .slots = command_cells[..],
        .owner = 2,
    };
    vk::CommandRecord empty = {
        .allocator        = allocator,
        .allocator_buffer = 0,
        .scratch          = &scratch[0],
    };
    vk::CommandRecord* empty_record = state.commands.claim(
        device:       {},
        ops:          null,
        device_state: &state,
        value:        empty,
    )!!;
    empty_record.state = gpu::internal::CommandState.SUBMITTING;
    vk::CommandRecord*[1] empty_records = { empty_record };
    completions[0].retirement_mutex.lock()!!;
    vk::publish_pending_commands_locked(
        state:      &state,
        completion: &completions[0],
        records:    empty_records[..],
        point:      gpu::internal::make_completion_point(queue, 1)!!,
    );
    completions[0].retirement_mutex.unlock();
    empty_record.state = gpu::internal::CommandState.SUBMITTED;
    assert(completions[0].submitted_count == 1
        && scratch[0].reference_count == 0,
        "submission transfers empty reference storage without allocation");
    vk::destroy_submitted_commands(&state);
    assert(scratch[0].reference_count == 0,
        "empty submitted teardown retains no references");

    allocator_slot.live_executables = 1;
    vk::CommandRecord populated = {
        .allocator        = allocator,
        .allocator_buffer = 1,
        .scratch          = &scratch[1],
    };
    vk::CommandRecord* populated_record = state.commands.claim(
        device:       {},
        ops:          null,
        device_state: &state,
        value:        populated,
    )!!;
    populated_record.state = gpu::internal::CommandState.SUBMITTING;
    vk::CommandRecord*[1] populated_records = { populated_record };
    completions[0].retirement_mutex.lock()!!;
    vk::publish_pending_commands_locked(
        state:      &state,
        completion: &completions[0],
        records:    populated_records[..],
        point:      gpu::internal::make_completion_point(queue, 2)!!,
    );
    completions[0].retirement_mutex.unlock();
    populated_record.state = gpu::internal::CommandState.SUBMITTED;
    assert(scratch[1].reference_count == 1,
        "submission transfers populated reference ownership exactly once");
    vk::destroy_submitted_commands(&state);
    assert(tracked.load(AtomicOrdering.RELAXED) == 0,
        "submitted teardown releases the transferred reference exactly once");
}

fn void device_loss_accepts_empty_or_populated_command_references() @test {
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
        vk::VkDeviceState* state = policy_device_state(&device);
        Atomic{uint}* retained_counter =
            &state.allocations.get(
                owned.allocation,
            )!!.tracked_command_references;
        vk::CommandBufferStats before = vk::command_buffer_stats(state);
        gpu::CommandList commands = record_policy_fill(&allocator, owned.span);
        vk::CommandRecord* record = vk::resolve_command(&commands)!!;
        vk::CommandBufferStats recorded = vk::command_buffer_stats(state);
        bool full = contract == gpu::ContractValidation.FULL;
        assert((record.scratch.reference_count == 1) == full,
            "device-loss setup reflects the selected contract mode");
        assert((record.scratch.references.ptr != null) == full,
            "device-loss setup owns reference storage only in full mode");
        assert(retained_counter.load(AtomicOrdering.RELAXED)
                == (full ? (uint)1 : (uint)0),
            "device-loss setup retains exactly the selected resource work");
        assert(recorded.allocated == before.allocated
            && recorded.freed == before.freed
            && recorded.pending == before.pending,
            "device-loss recording reuses its allocator's native buffer");
        gpu::internal::mark_device_lost(&device);
        gpu::discard_commands(&commands)!!;
        assert(retained_counter.load(AtomicOrdering.RELAXED) == 0,
            "device-loss cleanup releases exactly the selected reference work");
        vk::CommandBufferStats discarded = vk::command_buffer_stats(state);
        assert(discarded.allocated == recorded.allocated
            && discarded.freed == recorded.freed
            && discarded.pending == before.pending,
            "lost-command discard performs no extra native allocation or free");
        gpu::destroy_command_allocator(&allocator)!!;
        destroy_test_device(&device)!!;
        assert(!device.is_valid(),
            "device-loss teardown consumes devices in either contract mode");
    }
}

fn void command_reference_linear_lookup_uses_full_identity() @test {
    vk::CommandResourceReference first = {
        .owner      = 17,
        .index      = 23,
        .generation = 5,
    };
    vk::CommandResourceReference different_owner = first;
    different_owner.owner++;
    vk::CommandResourceReference different_index = first;
    different_index.index++;
    vk::CommandResourceReference different_generation = first;
    different_generation.generation++;
    vk::CommandResourceReference[1] live = { first };
    vk::CommandScratch scratch = {
        .references      = live[..],
        .reference_count = 1,
    };

    assert(vk::contains_command_reference(&scratch, &first),
        "linear lookup finds an exact resource identity");
    assert(!vk::contains_command_reference(&scratch, &different_owner)
        && !vk::contains_command_reference(&scratch, &different_index)
        && !vk::contains_command_reference(&scratch, &different_generation),
        "linear lookup compares owner, index, and generation");
}

fn void command_reference_maximum_capacity_is_transactional() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.FULL,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity          = 1,
        .max_resource_references_per_list =
            gpu::MAX_COMMAND_REFERENCES_PER_LIST,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        device: &device,
        queue:  queue,
        desc:   &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    Atomic{uint}[gpu::MAX_COMMAND_REFERENCES_PER_LIST] counters;

    for (uint i = 0; i < gpu::MAX_COMMAND_REFERENCES_PER_LIST; i++) {
        counters[i].store(1, AtomicOrdering.RELAXED);
        vk::CommandResourceReference candidate = {
            .owner            = 41,
            .index            = i,
            .generation       = 1,
            .retained_counter = &counters[i],
        };
        vk::CommandResourceReference[1] candidates = { candidate };
        vk::preflight_command_references(record, candidates[..])!!;
        vk::append_command_reference(record.scratch, candidate);
    }

    vk::CommandResourceReference overflow = {
        .owner      = 41,
        .index      = gpu::MAX_COMMAND_REFERENCES_PER_LIST,
        .generation = 1,
    };
    vk::CommandResourceReference[1] overflow_candidates = { overflow };
    assert(@catch(vk::preflight_command_references(
        record,
        overflow_candidates[..],
    )) == gpu::COMMAND_ALLOCATOR_CAPACITY_EXCEEDED,
        "the first distinct reference above the public maximum is rejected");
    assert(record.scratch.reference_count
            == gpu::MAX_COMMAND_REFERENCES_PER_LIST,
        "maximum-capacity rejection preserves the complete retained prefix");

    gpu::discard_commands(&commands)!!;
    foreach (&counter : counters) {
        assert(counter.load(AtomicOrdering.RELAXED) == 0,
            "discard releases every retained maximum-capacity reference");
    }
}

fn void command_reference_tracking_deduplicates_and_reuses_scratch() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.FULL,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 1,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    OwnedSpan owned = policy_span(&device, { .graphics }, "reference_list");
    defer free_owned_span(&device, &owned)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    vk::CommandScratch* scratch = record.scratch;

    vk::track_command_reference(
        state:      state,
        record:     record,
        owner:      owned.span.owner,
        index:      owned.span.index,
        generation: owned.span.generation,
    )!!;
    vk::track_command_reference(
        state:      state,
        record:     record,
        owner:      owned.span.owner,
        index:      owned.span.index,
        generation: owned.span.generation,
    )!!;
    assert(record.scratch.reference_count == 1
        && record.scratch.references[0].retained_counter.load(
            AtomicOrdering.RELAXED,
        ) == 1,
        "an exact duplicate is retained and appended once");
    Atomic{uint}* retained_counter =
        record.scratch.references[0].retained_counter;
    gpu::discard_commands(&commands)!!;
    assert(scratch.reference_count == 0,
        "recording discard clears the live reference list");

    commands = gpu::begin_commands(&allocator)!!;
    record = vk::resolve_command(&commands)!!;
    assert(record.scratch == scratch,
        "the allocator reuses its fixed scratch storage");
    vk::track_command_reference(
        state:      state,
        record:     record,
        owner:      owned.span.owner,
        index:      owned.span.index,
        generation: owned.span.generation,
    )!!;
    assert(record.scratch.reference_count == 1,
        "discarded scratch starts with an empty reference list when reused");
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;

    commands = gpu::begin_commands(&allocator)!!;
    record = vk::resolve_command(&commands)!!;
    assert(record.scratch.reference_count == 0
        && retained_counter.load(AtomicOrdering.RELAXED) == 0,
        "completion retirement clears references before scratch reuse");
    vk::track_command_reference(
        state:      state,
        record:     record,
        owner:      owned.span.owner,
        index:      owned.span.index,
        generation: owned.span.generation,
    )!!;
    assert(record.scratch.reference_count == 1,
        "retired scratch starts with an empty reference list when reused");
    gpu::discard_commands(&commands)!!;
}

fn void two_full_command_lists_retain_same_resource_independently() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.FULL,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 2,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    OwnedSpan owned = policy_span(&device, { .graphics }, "shared_reference");
    defer free_owned_span(&device, &owned)!!;

    gpu::CommandList first = record_policy_fill(&allocator, owned.span);
    gpu::CommandList second = record_policy_fill(&allocator, owned.span);
    vk::CommandRecord* first_record = vk::resolve_command(&first)!!;
    vk::CommandRecord* second_record = vk::resolve_command(&second)!!;
    Atomic{uint}* retained_counter =
        first_record.scratch.references[0].retained_counter;

    assert(first_record.scratch.reference_count == 1
        && second_record.scratch.reference_count == 1
        && second_record.scratch.references[0].retained_counter
            == retained_counter
        && retained_counter.load(AtomicOrdering.RELAXED) == 2,
        "each simultaneous list owns one retain on the shared resource");
    assert(@catch(gpu::free_allocation(&device, &owned.allocation))
        == gpu::RESOURCE_IN_USE,
        "both live command lists block premature destruction");

    gpu::discard_commands(&first)!!;
    assert(retained_counter.load(AtomicOrdering.RELAXED) == 1,
        "discarding one list releases only its independent retain");
    assert(@catch(gpu::free_allocation(&device, &owned.allocation))
        == gpu::RESOURCE_IN_USE,
        "the second live command list keeps destruction blocked");

    gpu::discard_commands(&second)!!;
    assert(retained_counter.load(AtomicOrdering.RELAXED) == 0,
        "discarding the second list releases the final retain");
    gpu::free_allocation(&device, &owned.allocation)!!;
    owned.span = {};
}

fn void texture_reference_retain_is_transactional_and_deduplicated() @test {
    vk::VkDeviceState state;
    vk::TextureCell[1] texture_cells = {{
        .generation = 1,
        .used       = true,
    }};
    state.textures = {
        .slots = texture_cells[..],
        .owner = 73,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    vk::CommandResourceReference[2] references;
    vk::CommandScratch scratch = {
        .references = references[..],
    };
    vk::CommandRecord record = { .scratch = &scratch };
    gpu::TextureHandle valid = {
        .owner      = state.textures.owner,
        .index      = 0,
        .generation = 1,
    };
    gpu::TextureHandle stale = valid;
    stale.generation++;

    fault retain_fault = @catch(vk::resolve_tracked_texture_command_reference(
        &state,
        &record,
        stale,
    ));
    assert(retain_fault == gpu::INVALID_HANDLE
        && scratch.reference_count == 0
        && texture_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0,
        "a failed texture retain leaves reference storage unchanged");

    (void)vk::resolve_tracked_texture_command_reference(
        &state,
        &record,
        valid,
    )!!;
    (void)vk::resolve_tracked_texture_command_reference(
        &state,
        &record,
        valid,
    )!!;
    assert(scratch.reference_count == 1
        && texture_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 1,
        "a successful texture retain appends exactly one reference");

    vk::clear_command_scratch_references(&scratch);
    assert(texture_cells[0].value.tracked_command_references.load(
        AtomicOrdering.RELAXED,
    ) == 0,
        "texture reference cleanup releases the successful retain");
}

fn void command_reference_rollback_releases_only_the_suffix() @test {
    Atomic{uint}[3] tracked;
    vk::CommandResourceReference[3] references;
    foreach (i, &counter : tracked) {
        counter.store(1, AtomicOrdering.RELAXED);
        references[i] = {
            .owner            = 31,
            .index            = (uint)i,
            .generation       = 1,
            .retained_counter = counter,
        };
    }
    vk::CommandScratch scratch = {
        .references      = references[..],
        .reference_count = 3,
    };
    vk::CommandRecord record = { .scratch = &scratch };

    vk::rollback_command_references(&record, 3);
    assert(scratch.reference_count == 3
        && tracked[0].load(AtomicOrdering.RELAXED) == 1
        && tracked[1].load(AtomicOrdering.RELAXED) == 1
        && tracked[2].load(AtomicOrdering.RELAXED) == 1,
        "rollback at the current checkpoint is a no-op");
    vk::rollback_command_references(&record, 1);
    assert(scratch.reference_count == 1,
        "rollback keeps the reference prefix");
    assert(tracked[0].load(AtomicOrdering.RELAXED) == 1
        && tracked[1].load(AtomicOrdering.RELAXED) == 0
        && tracked[2].load(AtomicOrdering.RELAXED) == 0,
        "rollback releases only the discarded reference suffix");
    assert(vk::contains_command_reference(&scratch, &references[0])
        && !vk::contains_command_reference(&scratch, &references[1]),
        "the live list contains only the retained prefix");
}

fn void command_reference_retain_failure_rolls_back_prior_appends() @test {
    vk::VkDeviceState state;
    vk::BufferCell[2] buffer_cells;
    buffer_cells[0] = { .generation = 1, .used = true };
    state.buffers = {
        .slots    = buffer_cells[..],
        .owner    = 71,
        .next_new = 1,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    vk::CommandResourceReference[2] references;
    vk::CommandScratch scratch = {
        .references = references[..],
    };
    vk::CommandRecord record = { .scratch = &scratch };

    vk::track_command_reference(
        state:      &state,
        record:     &record,
        owner:      state.buffers.owner,
        index:      0,
        generation: 1,
    )!!;
    fault retain_fault = @catch(vk::track_command_reference(
        state:      &state,
        record:     &record,
        owner:      state.buffers.owner,
        index:      1,
        generation: 1,
    ));
    assert(retain_fault == gpu::INVALID_HANDLE,
        "the injected second retain rejects its invalid resource identity");
    vk::rollback_command_references(&record, 0);

    assert(scratch.reference_count == 0
        && buffer_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0,
        "a failed later retain rolls back every earlier append");
}
