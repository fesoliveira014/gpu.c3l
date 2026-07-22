module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::atomic::types;
import std::core::mem::alloc;
import std::core::mem::allocators;

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
    bool track_lifetimes,
    bool vulkan_layers,
    bool callback,
    bool debug_names,
    PolicyDiagnosticCapture* capture,
) {
    return {
        .backend                  = gpu::BackendKind.VULKAN,
        .contract_validation      = contract,
        .track_resource_lifetimes = track_lifetimes,
        .enable_vulkan_validation = vulkan_layers,
        .enable_debug_names       = debug_names,
        .application_name         = "gpu_validation_policy_test",
        .debug_callback           = callback ? &capture_policy_diagnostic : null,
        .debug_user_data          = callback ? capture : null,
    };
}

fn vk::VkDeviceState* policy_device_state(gpu::Device* device) {
    return (vk::VkDeviceState*)gpu::device_backend_state(*device);
}

fn gpu::CommandList record_policy_fill(
    gpu::Device* device,
    gpu::GpuSpan span,
) {
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
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

fn void assert_zero_tracking_work(
    vk::VkDeviceState* state,
) {
    vk::CommandPolicyCounters counters = vk::command_policy_counters(state);
    assert(counters.tracking_calls == 0
        && counters.reference_allocations == 0
        && counters.reference_increments == 0
        && counters.reference_releases == 0,
        "the operation performs no lifetime-tracking work");
}

fn void policy_matrix_creates_exact_runtime_and_device_state() @test {
    gpu::ContractValidation[3] contracts = {
        gpu::ContractValidation.TRUSTED,
        gpu::ContractValidation.OBJECT_BOUNDARIES,
        gpu::ContractValidation.FULL,
    };
    bool[2] values = { false, true };

    foreach (contract : contracts) {
        foreach (tracking : values) {
            foreach (layers : values) {
                foreach (callback : values) {
                    foreach (names : values) {
                        PolicyDiagnosticCapture capture;
                        gpu::RuntimeDesc desc = policy_runtime_desc(
                            contract:       contract,
                            track_lifetimes: tracking,
                            vulkan_layers:   layers,
                            callback:        callback,
                            debug_names:     names,
                            capture:         &capture,
                        );
                        gpu::Device device = create_test_device(&desc)!!;
                        vk::VkDeviceState* state = policy_device_state(&device);
                        assert(state.validation_policy.contract == contract,
                            "device preserves the contract policy");
                        assert(state.validation_policy.track_lifetimes == tracking,
                            "device preserves the lifetime policy");
                        assert(state.validation_policy.vulkan_layers == layers,
                            "device preserves the layer policy");
                        assert(state.debug_names == names,
                            "device preserves the debug-name policy");
                        assert((state.debug_callback != null) == callback,
                            "device preserves callback presence independently");
                        destroy_test_device(&device)!!;
                    }
                }
            }
        }
    }
}

fn void instance_plans_callback_names_and_layers_independently() @test {
    vk::DebugUtilsMessengerCreateInfoEXT messenger_info;
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.FULL,
        track_lifetimes: true,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
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

fn void tracking_off_keeps_reference_storage_empty_through_retirement() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.FULL,
        track_lifetimes: false,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    OwnedSpan owned = policy_span(
        &device,
        { .graphics },
        "tracking_off_completion",
    );
    defer free_owned_span(&device, &owned)!!;

    vk::reset_command_policy_counters(state);
    gpu::CommandList commands = record_policy_fill(&device, owned.span);
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    assert(record.references.ptr == null && record.reference_count == 0,
        "tracking-off recording owns no reference storage");
    assert_zero_tracking_work(state);

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    record = state.commands.get(executable.handle)!!;
    assert(record.references.ptr == null && record.reference_count == 0,
        "tracking-off executable owns no reference storage");
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        &submit,
    )!!;
    if (state.submitted_command_count > 0) {
        vk::SubmittedCommandBatch* batch = &state.submitted_commands[0];
        assert(batch.references.ptr == null && batch.reference_count == 0,
            "tracking-off submission transfers empty reference storage");
    }
    gpu::wait_completion(completion)!!;
    assert(state.submitted_command_count == 0,
        "completion retires the tracking-off command batch");
    assert_zero_tracking_work(state);
    gpu::free_allocation(&device, &owned.allocation)!!;
    owned.span = {};
}

fn void tracking_off_allows_early_caller_owned_destruction() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.OBJECT_BOUNDARIES,
        track_lifetimes: false,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    OwnedSpan owned = policy_span(&device, { .graphics }, "caller_owned");

    vk::reset_command_policy_counters(state);
    gpu::CommandList commands = record_policy_fill(&device, owned.span);
    gpu::free_allocation(&device, &owned.allocation)!!;
    owned.span = {};
    gpu::discard_commands(&commands)!!;
    commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    vk::CommandRecord* reused = vk::resolve_command(&commands)!!;
    assert(reused.references.ptr == null && reused.reference_count == 0,
        "tracking-off command reuse starts from empty reference storage");
    gpu::discard_commands(&commands)!!;
    assert_zero_tracking_work(state);
}

fn void tracking_on_rejects_destruction_until_each_lifecycle_release() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.OBJECT_BOUNDARIES,
        track_lifetimes: true,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = policy_device_state(&device);

    OwnedSpan recording = policy_span(&device, { .graphics }, "recording");
    vk::reset_command_policy_counters(state);
    gpu::CommandList commands = record_policy_fill(&device, recording.span);
    assert(@catch(gpu::free_allocation(&device, &recording.allocation))
        == gpu::RESOURCE_IN_USE,
        "tracking rejects destruction during recording");
    gpu::discard_commands(&commands)!!;
    vk::CommandPolicyCounters counters = vk::command_policy_counters(state);
    assert(counters.tracking_calls == 1
        && counters.reference_allocations == 1
        && counters.reference_increments == 1
        && counters.reference_releases == 1,
        "recording discard releases its one retained resource exactly once");
    gpu::free_allocation(&device, &recording.allocation)!!;
    recording.span = {};

    OwnedSpan executable = policy_span(&device, { .graphics }, "executable");
    vk::reset_command_policy_counters(state);
    commands = record_policy_fill(&device, executable.span);
    gpu::ExecutableCommandList list = gpu::end_commands(&commands)!!;
    assert(@catch(gpu::free_allocation(&device, &executable.allocation))
        == gpu::RESOURCE_IN_USE,
        "tracking rejects destruction while executable");
    gpu::discard_executable_commands(&list)!!;
    counters = vk::command_policy_counters(state);
    assert(counters.reference_increments == 1
        && counters.reference_releases == 1,
        "executable discard releases its retained resource exactly once");
    gpu::free_allocation(&device, &executable.allocation)!!;
    executable.span = {};

    OwnedSpan submitted = policy_span(&device, { .graphics }, "submitted");
    vk::reset_command_policy_counters(state);
    commands = record_policy_fill(&device, submitted.span);
    list = gpu::end_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { list };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        &submit,
    )!!;
    assert(@catch(gpu::free_allocation(&device, &submitted.allocation))
        == gpu::RESOURCE_IN_USE,
        "tracking rejects destruction before submitted work is retired");
    gpu::wait_completion(completion)!!;
    counters = vk::command_policy_counters(state);
    assert(counters.reference_increments == 1
        && counters.reference_releases == 1,
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
            track_lifetimes: false,
            vulkan_layers:   false,
            callback:        callback,
            debug_names:     false,
            capture:         &capture,
        );
        gpu::Device device = create_test_device(&desc)!!;
        capture = {};
        vk::VkDeviceState* state = policy_device_state(&device);
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
            gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        )!!;
        gpu::TextureBarrier barrier = gpu::texture_transition(
            texture: texture,
            before:  { .layout = gpu::TextureLayout.UNDEFINED },
            after:   gpu::sampled_at({ .fragment_shader }),
        )!!;
        vk::reset_command_policy_counters(state);
        gpu::reset_command_resolution_stats();
        fault misuse = @catch(gpu::cmd_texture_barrier(&commands, &barrier));
        assert(misuse == gpu::INVALID_ARGUMENT,
            "full command checks are independent of callback presence");
        assert(gpu::command_resolution_stats().native_commands == 0,
            "full rejection precedes native command emission");
        vk::CommandPolicyCounters counters = vk::command_policy_counters(state);
        assert(counters.semantic_checks == 1
            && counters.tracking_calls == 0,
            "callback presence does not change policy work");
        assert(capture.calls == (callback ? 1 : 0),
            "configured callbacks observe but never enable full checks");
        gpu::discard_commands(&commands)!!;
        gpu::destroy_texture(&device, texture)!!;
        destroy_test_device(&device)!!;
    }
}

fn void trusted_callback_and_names_enable_no_command_checks() @test {
    bool[2] callbacks = { false, true };
    foreach (callback : callbacks) {
        PolicyDiagnosticCapture capture;
        gpu::RuntimeDesc desc = policy_runtime_desc(
            contract:       gpu::ContractValidation.TRUSTED,
            track_lifetimes: false,
            vulkan_layers:   false,
            callback:        callback,
            debug_names:     true,
            capture:         &capture,
        );
        gpu::Device device = create_test_device(&desc)!!;
        capture = {};
        vk::VkDeviceState* state = policy_device_state(&device);
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
            gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        )!!;
        gpu::TextureBarrier barrier = gpu::texture_transition(
            texture: texture,
            before:  { .layout = gpu::TextureLayout.UNDEFINED },
            after:   gpu::sampled_at({ .fragment_shader }),
        )!!;
        vk::reset_command_policy_counters(state);
        gpu::cmd_texture_barrier(&commands, &barrier)!!;
        vk::CommandPolicyCounters counters = vk::command_policy_counters(state);
        assert(counters.semantic_checks == 0
            && counters.tracking_calls == 0,
            "trusted callback and names settings enable no command policy work");
        assert(capture.calls == 0,
            "callback presence alone produces no contract diagnostic");
        gpu::discard_commands(&commands)!!;
        gpu::destroy_texture(&device, texture)!!;
        destroy_test_device(&device)!!;
    }
}

fn void trusted_commands_retain_the_mandatory_safety_floor() @test {
    PolicyDiagnosticCapture capture;
    gpu::RuntimeDesc desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.TRUSTED,
        track_lifetimes: false,
        vulkan_layers:   false,
        callback:        false,
        debug_names:     false,
        capture:         &capture,
    );
    gpu::Device device = create_test_device(&desc)!!;
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    defer discard_test_commands(&commands)!!;

    vk::reset_command_policy_counters(state);
    gpu::reset_command_resolution_stats();
    assert(@catch(gpu::cmd_copy_buffer(&commands, null))
        == gpu::INVALID_ARGUMENT,
        "trusted commands reject null host descriptors");
    assert(gpu::command_resolution_stats().native_commands == 0,
        "mandatory safety rejection performs no native call");
    assert_zero_tracking_work(state);
}

fn void object_boundaries_reject_foreign_and_stale_resources_without_command_checks() @test {
    PolicyDiagnosticCapture first_capture;
    PolicyDiagnosticCapture second_capture;
    gpu::RuntimeDesc first_desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.OBJECT_BOUNDARIES,
        track_lifetimes: false,
        vulkan_layers:   false,
        callback:        true,
        debug_names:     false,
        capture:         &first_capture,
    );
    gpu::RuntimeDesc second_desc = policy_runtime_desc(
        contract:       gpu::ContractValidation.OBJECT_BOUNDARIES,
        track_lifetimes: false,
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
    vk::VkDeviceState* first_state = policy_device_state(&first);
    OwnedSpan owned = policy_span(&first, { .graphics }, "boundary_owned");

    vk::reset_command_policy_counters(first_state);
    gpu::CommandList commands = record_policy_fill(&first, owned.span);
    gpu::discard_commands(&commands)!!;
    vk::CommandPolicyCounters counters =
        vk::command_policy_counters(first_state);
    assert(counters.semantic_checks == 0,
        "object-boundary policy performs no per-command semantic work");

    gpu::GpuAllocation foreign = owned.allocation;
    assert(@catch(gpu::free_allocation(&second, &foreign))
        == gpu::INVALID_HANDLE,
        "object-boundary policy rejects foreign ownership");
    gpu::GpuAllocation stale = owned.allocation;
    gpu::free_allocation(&first, &owned.allocation)!!;
    owned.span = {};
    assert(@catch(gpu::get_allocation_info(&first, stale))
        == gpu::INVALID_HANDLE,
        "object-boundary policy rejects stale identities");
}

fn void teardown_leak_scans_depend_only_on_contract_or_callback() @test {
    vk::VkDeviceState state;
    state.validation_policy.contract = gpu::ContractValidation.TRUSTED;
    assert(!vk::report_teardown_leaks(&state),
        "trusted teardown without a callback skips accepted leak scans");

    state.debug_callback = &capture_policy_diagnostic;
    assert(vk::report_teardown_leaks(&state),
        "callback presence enables accepted teardown leak scans");

    state.debug_callback = null;
    state.validation_policy.contract =
        gpu::ContractValidation.OBJECT_BOUNDARIES;
    assert(vk::report_teardown_leaks(&state),
        "object-boundary policy enables accepted teardown leak scans");

    state.validation_policy.contract = gpu::ContractValidation.FULL;
    assert(vk::report_teardown_leaks(&state),
        "full policy enables accepted teardown leak scans");
}

fn void boundary_diagnostics_follow_contract_not_callback_presence() @test {
    PolicyDiagnosticCapture capture;
    vk::VkDeviceState state = {
        .debug_callback  = &capture_policy_diagnostic,
        .debug_user_data = &capture,
    };
    vk::bind_command_diagnostic_sink(&state);

    state.validation_policy.contract = gpu::ContractValidation.TRUSTED;
    fault returned = vk::vk_report_root_contract_failure(
        backend_state:  &state,
        operation:      "free_allocation",
        public_fault:   gpu::INVALID_HANDLE,
        resource_kind:  gpu::DebugResourceKind.ALLOCATION,
        rejected_field: "GpuAllocation",
        invariant:      "must belong to the device",
    );
    assert(returned == gpu::INVALID_HANDLE && capture.calls == 0,
        "trusted boundaries preserve faults without detailed diagnostics");

    state.validation_policy.contract =
        gpu::ContractValidation.OBJECT_BOUNDARIES;
    returned = vk::vk_report_root_contract_failure(
        backend_state:  &state,
        operation:      "free_allocation",
        public_fault:   gpu::INVALID_HANDLE,
        resource_kind:  gpu::DebugResourceKind.ALLOCATION,
        rejected_field: "GpuAllocation",
        invariant:      "must belong to the device",
    );
    assert(returned == gpu::INVALID_HANDLE && capture.calls == 1,
        "object-boundary policy emits the configured boundary diagnostic");

    state.debug_callback = null;
    state.validation_policy.contract = gpu::ContractValidation.FULL;
    returned = vk::vk_report_root_contract_failure(
        backend_state:  &state,
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
    for (uint i = 0; i < gpu::MAX_DEVICES; i++) {
        gpu::DeviceSlotControl control = (gpu::DeviceSlotControl)
            gpu::device_slots[i].control.load(AtomicOrdering.RELAXED);
        if (control.state == gpu::DEVICE_SLOT_LIVE) count++;
    }
    return count;
}

fn void partial_creation_rolls_back_across_contract_and_tracking_policies() @test {
    gpu::ContractValidation[3] contracts = {
        gpu::ContractValidation.TRUSTED,
        gpu::ContractValidation.OBJECT_BOUNDARIES,
        gpu::ContractValidation.FULL,
    };
    bool[2] tracking_values = { false, true };

    foreach (contract : contracts) {
        foreach (tracking : tracking_values) {
            PolicyDiagnosticCapture capture;
            gpu::RuntimeDesc desc = policy_runtime_desc(
                contract:       contract,
                track_lifetimes: tracking,
                vulkan_layers:   false,
                callback:        false,
                debug_names:     false,
                capture:         &capture,
            );
            gpu::Runtime runtime = gpu::create_runtime(&desc)!!;
            gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
            gpu::DeviceRequest request = gpu::strict_device_request();
            bool exercised;
            for (uint i = 0; i < adapters.count; i++) {
                gpu::Adapter adapter = adapters.get(i)!!;
                if (!gpu::supports_device_request(&adapter, &request)!!
                    .supported) continue;
                exercised = true;
                gpu::Runtime retained_runtime = runtime;
                gpu::runtime_retain_device(&retained_runtime)!!;
                vk::DeviceCreationProbe probe = {
                    .fail_at = vk::DeviceCreationCheckpoint.BEFORE_PUBLICATION,
                };
                uint live_before = policy_live_device_count();
                assert(@catch(vk::create_vk_device_with_probe(
                    adapter:          &adapter,
                    request:          &request,
                    retained_runtime: &retained_runtime,
                    probe:            &probe,
                )) == gpu::BACKEND_ERROR,
                    "partial creation returns its injected backend fault");
                gpu::runtime_release_device(&retained_runtime)!!;
                assert(probe.reached
                        == vk::DeviceCreationCheckpoint.BEFORE_PUBLICATION
                    && probe.rollback_count == 1,
                    "partial creation reaches and rolls back exactly once");
                assert(policy_live_device_count() == live_before,
                    "partial creation publishes no device slot");
                break;
            }
            assert(exercised,
                "the pinned Vulkan adapter supports strict device creation");
            gpu::destroy_runtime(&runtime)!!;
        }
    }
}

fn void empty_and_populated_reference_storage_release_exactly() @test {
    vk::VkDeviceState state = {
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();

    vk::CommandRecord empty;
    vk::reset_command_policy_counters(&state);
    vk::free_command_references(&state, &empty);
    assert(empty.references.ptr == null && empty.reference_count == 0,
        "empty command reference storage remains empty during release");
    assert_zero_tracking_work(&state);

    Atomic{uint} tracked;
    tracked.store(1, AtomicOrdering.RELAXED);
    vk::CommandRecord populated;
    populated.references = alloc::new_array(
        state.host_allocator,
        vk::CommandResourceReference,
        1,
    );
    populated.references[0] = {
        .tracked_command_references = &tracked,
    };
    populated.reference_count = 1;
    vk::reset_command_policy_counters(&state);
    vk::free_command_references(&state, &populated);
    vk::CommandPolicyCounters counters = vk::command_policy_counters(&state);
    assert(populated.references.ptr == null
        && populated.reference_count == 0,
        "populated command reference storage is freed and reset");
    assert(tracked.load(AtomicOrdering.RELAXED) == 0
        && counters.reference_releases == 1,
        "populated release decrements its retained counter exactly once");
}

fn void submission_transfer_accepts_empty_and_populated_reference_storage() @test {
    vk::VkDeviceState state = {
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();

    vk::CommandRecord empty;
    vk::CommandRecord*[1] empty_records = { &empty };
    uint empty_reservation = vk::prepare_submitted_commands(
        &state,
        empty_records[..],
    );
    vk::publish_submitted_commands(
        state:       &state,
        records:     empty_records[..],
        completion:  {},
        reservation: empty_reservation,
    );
    assert(state.submitted_command_count == 1
        && state.submitted_commands[0].references.ptr == null
        && state.submitted_commands[0].reference_count == 0,
        "submission transfers empty reference storage without allocation");
    vk::reset_command_policy_counters(&state);
    vk::destroy_submitted_commands(&state);
    assert_zero_tracking_work(&state);

    Atomic{uint} tracked;
    tracked.store(1, AtomicOrdering.RELAXED);
    vk::CommandRecord populated;
    populated.references = alloc::new_array(
        state.host_allocator,
        vk::CommandResourceReference,
        1,
    );
    populated.references[0] = {
        .tracked_command_references = &tracked,
    };
    populated.reference_count = 1;
    vk::CommandRecord*[1] populated_records = { &populated };
    uint populated_reservation = vk::prepare_submitted_commands(
        &state,
        populated_records[..],
    );
    vk::publish_submitted_commands(
        state:       &state,
        records:     populated_records[..],
        completion:  {},
        reservation: populated_reservation,
    );
    assert(populated.references.ptr == null
        && populated.reference_count == 0
        && state.submitted_commands[0].reference_count == 1,
        "submission transfers populated reference ownership exactly once");
    vk::reset_command_policy_counters(&state);
    vk::destroy_submitted_commands(&state);
    vk::CommandPolicyCounters counters = vk::command_policy_counters(&state);
    assert(tracked.load(AtomicOrdering.RELAXED) == 0
        && counters.reference_releases == 1,
        "submitted teardown releases the transferred reference exactly once");
}

fn void device_loss_accepts_empty_or_populated_command_references() @test {
    bool[2] tracking_values = { false, true };
    foreach (tracking : tracking_values) {
        PolicyDiagnosticCapture capture;
        gpu::RuntimeDesc desc = policy_runtime_desc(
            contract:       gpu::ContractValidation.OBJECT_BOUNDARIES,
            track_lifetimes: tracking,
            vulkan_layers:   false,
            callback:        false,
            debug_names:     false,
            capture:         &capture,
        );
        gpu::Device device = create_test_device(&desc)!!;
        OwnedSpan owned = policy_span(&device, { .graphics }, "lost_command");
        vk::VkDeviceState* state = policy_device_state(&device);
        vk::CommandBufferStats before = vk::command_buffer_stats(state);
        vk::reset_command_policy_counters(state);
        gpu::CommandList commands = record_policy_fill(&device, owned.span);
        vk::CommandRecord* record = vk::resolve_command(&commands)!!;
        vk::CommandBufferStats recorded = vk::command_buffer_stats(state);
        assert((record.reference_count == 1) == tracking,
            "device-loss setup reflects the selected tracking policy");
        assert((record.references.ptr != null) == tracking,
            "device-loss setup owns storage only when tracking is enabled");
        assert(recorded.allocated == before.allocated + 1
            && recorded.freed == before.freed
            && recorded.pending == before.pending,
            "device-loss setup allocates one live native command buffer");
        vk::CommandPolicyCounters counters =
            vk::command_policy_counters(state);
        assert((counters.reference_allocations == 1) == tracking,
            "device-loss setup allocates host reference storage only when selected");
        gpu::mark_device_lost(&device);
        gpu::discard_commands(&commands)!!;
        vk::CommandBufferStats discarded = vk::command_buffer_stats(state);
        counters = vk::command_policy_counters(state);
        assert((counters.reference_releases == 1) == tracking,
            "lost-command discard releases exactly the selected reference work");
        assert(discarded.allocated == recorded.allocated
            && discarded.freed == recorded.freed
            && discarded.pending == before.pending,
            "lost-command discard performs no extra native allocation or free");
        destroy_test_device(&device)!!;
        assert(!device.is_valid(),
            "device-loss teardown consumes devices with either reference shape");
    }
}
