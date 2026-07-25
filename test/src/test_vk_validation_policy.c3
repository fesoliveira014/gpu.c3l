module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
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

fn void every_policy_rejects_null_command_record() @test {
    gpu::ContractValidation[3] contracts = {
        gpu::ContractValidation.TRUSTED,
        gpu::ContractValidation.OBJECT_BOUNDARIES,
        gpu::ContractValidation.FULL,
    };

    foreach (contract : contracts) {
        PolicyDiagnosticCapture capture;
        gpu::RuntimeDesc desc = policy_runtime_desc(
            contract:       contract,
            track_lifetimes: false,
            vulkan_layers:   false,
            callback:        false,
            debug_names:     false,
            capture:         &capture,
        );
        gpu::Device device = create_test_device(&desc)!!;
        gpu::CommandList fabricated = { .generation = 1 };
        fault rejected = @catch(gpu::end_commands(&fabricated));
        assert(rejected == gpu::INVALID_HANDLE,
            "every contract policy rejects a null command record");
        destroy_test_device(&device)!!;
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
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    OwnedSpan owned = policy_span(
        &device,
        { .graphics },
        "tracking_off_completion",
    );
    defer free_owned_span(&device, &owned)!!;

    vk::reset_command_policy_counters(state);
    gpu::CommandList commands = record_policy_fill(&allocator, owned.span);
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    assert(record.scratch.references.ptr == null
        && record.scratch.reference_count == 0,
        "tracking-off recording owns no reference storage");
    assert_zero_tracking_work(state);

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    record = executable.record;
    assert(record.scratch.references.ptr == null
        && record.scratch.reference_count == 0,
        "tracking-off executable owns no reference storage");
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
            && scratch.reference_count == 0
            && scratch.reference_index.epoch == 0,
            "tracking-off submission transfers empty reference storage");
    }
    gpu::wait_completion(completion)!!;
    assert(completion_state.submitted_count == 0,
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
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    OwnedSpan owned = policy_span(&device, { .graphics }, "caller_owned");

    vk::reset_command_policy_counters(state);
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
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::VkDeviceState* state = policy_device_state(&device);

    OwnedSpan recording = policy_span(&device, { .graphics }, "recording");
    vk::reset_command_policy_counters(state);
    gpu::CommandList commands = record_policy_fill(&allocator, recording.span);
    assert(@catch(gpu::free_allocation(&device, &recording.allocation))
        == gpu::RESOURCE_IN_USE,
        "tracking rejects destruction during recording");
    gpu::discard_commands(&commands)!!;
    vk::CommandPolicyCounters counters = vk::command_policy_counters(state);
    assert(counters.tracking_calls == 1
        && counters.reference_allocations == 0
        && counters.reference_increments == 1
        && counters.reference_releases == 1,
        "recording discard releases its one retained resource exactly once");
    gpu::free_allocation(&device, &recording.allocation)!!;
    recording.span = {};

    OwnedSpan executable = policy_span(&device, { .graphics }, "executable");
    vk::reset_command_policy_counters(state);
    commands = record_policy_fill(&allocator, executable.span);
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
    commands = record_policy_fill(&allocator, submitted.span);
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
        gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
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
            &allocator,
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
            track_lifetimes: false,
            vulkan_layers:   false,
            callback:        callback,
            debug_names:     true,
            capture:         &capture,
        );
        gpu::Device device = create_test_device(&desc)!!;
        capture = {};
        gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
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
            &allocator,
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
        gpu::destroy_command_allocator(&allocator)!!;
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
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    gpu::CommandList commands = gpu::begin_commands(
        &allocator,
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
    gpu::Queue first_queue = gpu::get_queue(&first, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator first_allocator = gpu::create_command_allocator(&first, first_queue)!!;
    defer gpu::destroy_command_allocator(&first_allocator)!!;
    OwnedSpan owned = policy_span(&first, { .graphics }, "boundary_owned");

    vk::reset_command_policy_counters(first_state);
    gpu::CommandList commands = record_policy_fill(&first_allocator, owned.span);
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

    state.validation_policy.contract =
        gpu::ContractValidation.OBJECT_BOUNDARIES;
    returned = vk::vk_report_root_contract_failure(
        state:          &state,
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
                gpu::internal::runtime_retain_device(&retained_runtime)!!;
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
                "the pinned Vulkan adapter supports strict device creation");
            gpu::destroy_runtime(&runtime)!!;
        }
    }
}

fn void empty_and_populated_fixed_reference_storage_release_exactly() @test {
    vk::VkDeviceState state = {
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    state.command_mutex.init()!!;
    defer (void)state.command_mutex.destroy();

    vk::reset_command_policy_counters(&state);
    vk::CommandResourceReference[] empty;
    vk::release_tracked_command_references(&state, empty, 0);
    assert(empty.len == 0,
        "empty command reference storage remains empty during release");
    assert_zero_tracking_work(&state);

    Atomic{uint} tracked;
    tracked.store(1, AtomicOrdering.RELAXED);
    vk::CommandResourceReference[1] populated = {{
        .tracked_command_references = &tracked,
    }};
    vk::reset_command_policy_counters(&state);
    vk::release_tracked_command_references(&state, populated[..], 1);
    vk::CommandPolicyCounters counters = vk::command_policy_counters(&state);
    assert(tracked.load(AtomicOrdering.RELAXED) == 0
        && counters.reference_releases == 1,
        "fixed populated storage decrements its retained counter exactly once");
}

fn void command_reference_clear_releases_list_and_resets_index() @test {
    vk::VkDeviceState state;
    Atomic{uint}[2] tracked;
    vk::CommandResourceReference[2] references;
    vk::CommandReferenceIndexCell[4] cells;
    foreach (i, &counter : tracked) {
        counter.store(1, AtomicOrdering.RELAXED);
        references[i] = {
            .owner                      = 19,
            .index                      = (uint)i,
            .generation                 = 1,
            .tracked_command_references = counter,
        };
    }
    vk::CommandScratch scratch = {
        .references      = references[..],
        .reference_index = {
            .cells = cells[..],
            .epoch = 3,
        },
        .reference_count = 2,
    };
    for (uint i = 0; i < scratch.reference_count; i++) {
        vk::CommandReferenceProbe insertion = vk::probe_command_reference(
            &state,
            &scratch,
            &scratch.references[i],
        )!!;
        vk::publish_command_reference(
            state:           &state,
            scratch:         &scratch,
            reference:       &scratch.references[i],
            probe:           &insertion,
            reference_index: i,
        );
    }
    vk::reset_command_policy_counters(&state);
    vk::reset_command_reference_work_counters(&state);

    vk::clear_command_scratch_references(&state, &scratch);

    vk::CommandPolicyCounters policy = vk::command_policy_counters(&state);
    vk::CommandReferenceWorkCounters work =
        vk::command_reference_work_counters(&state);
    bool prior_epoch_only = true;
    foreach (&cell : cells) {
        prior_epoch_only &= cell.epoch != scratch.reference_index.epoch;
    }
    assert(tracked[0].load(AtomicOrdering.RELAXED) == 0
        && tracked[1].load(AtomicOrdering.RELAXED) == 0
        && scratch.reference_count == 0
        && scratch.reference_index.epoch == 4
        && prior_epoch_only,
        "clear releases the recorded list and publishes an empty new epoch");
    assert(policy.reference_releases == 2 && work.releases == 2,
        "clear reports one release for every recorded reference");
}

fn void submission_transfer_accepts_empty_and_populated_reference_storage() @test {
    vk::VkDeviceState state = {
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
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
        .tracked_command_references = &tracked,
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
        .queue             = queue,
        .scratch           = scratch[..],
        .available         = available[..],
        .live_executables  = 1,
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
    gpu::internal::vk::CommandRecord* empty_record = state.commands.claim(
        device:        {},
        ops:           null,
        device_state:  &state,
        value:         empty,
    )!!;
    empty_record.state = gpu::internal::CommandState.SUBMITTING;
    gpu::internal::vk::CommandRecord*[1] empty_records = { empty_record };
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
    vk::reset_command_policy_counters(&state);
    vk::destroy_submitted_commands(&state);
    assert_zero_tracking_work(&state);

    allocator_slot.live_executables = 1;
    vk::CommandRecord populated = {
        .allocator        = allocator,
        .allocator_buffer = 1,
        .scratch          = &scratch[1],
    };
    gpu::internal::vk::CommandRecord* populated_record = state.commands.claim(
        device:        {},
        ops:           null,
        device_state:  &state,
        value:         populated,
    )!!;
    populated_record.state = gpu::internal::CommandState.SUBMITTING;
    gpu::internal::vk::CommandRecord*[1] populated_records = {
        populated_record,
    };
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
        gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
        OwnedSpan owned = policy_span(&device, { .graphics }, "lost_command");
        vk::VkDeviceState* state = policy_device_state(&device);
        vk::CommandBufferStats before = vk::command_buffer_stats(state);
        vk::reset_command_policy_counters(state);
        gpu::CommandList commands = record_policy_fill(&allocator, owned.span);
        vk::CommandRecord* record = vk::resolve_command(&commands)!!;
        vk::CommandBufferStats recorded = vk::command_buffer_stats(state);
        assert((record.scratch.reference_count == 1) == tracking,
            "device-loss setup reflects the selected tracking policy");
        assert((record.scratch.references.ptr != null) == tracking,
            "device-loss setup owns storage only when tracking is enabled");
        assert(recorded.allocated == before.allocated
            && recorded.freed == before.freed
            && recorded.pending == before.pending,
            "device-loss recording reuses its allocator's native buffer");
        vk::CommandPolicyCounters counters =
            vk::command_policy_counters(state);
        assert(counters.reference_allocations == 0,
            "device-loss setup uses its allocator's fixed reference storage");
        gpu::internal::mark_device_lost(&device);
        gpu::discard_commands(&commands)!!;
        vk::CommandBufferStats discarded = vk::command_buffer_stats(state);
        counters = vk::command_policy_counters(state);
        assert((counters.reference_releases == 1) == tracking,
            "lost-command discard releases exactly the selected reference work");
        assert(discarded.allocated == recorded.allocated
            && discarded.freed == recorded.freed
            && discarded.pending == before.pending,
            "lost-command discard performs no extra native allocation or free");
        gpu::destroy_command_allocator(&allocator)!!;
        destroy_test_device(&device)!!;
        assert(!device.is_valid(),
            "device-loss teardown consumes devices with either reference shape");
    }
}

fn void command_reference_index_uses_full_identity_and_exact_probes() @test {
    vk::VkDeviceState state;
    vk::CommandReferenceIndexCell[8] cells;
    vk::CommandResourceReference[4] references;
    vk::CommandScratch scratch = {
        .references      = references[..],
        .reference_index = {
            .cells = cells[..],
            .epoch = 1,
        },
    };
    vk::CommandResourceReference first = {
        .owner      = 17,
        .index      = 23,
        .generation = 5,
    };

    vk::reset_command_reference_work_counters(&state);
    vk::CommandReferenceProbe insertion = vk::probe_command_reference(
        &state,
        &scratch,
        &first,
    )!!;
    assert(!insertion.found, "a fresh index reports an empty insertion cell");
    vk::publish_command_reference(
        state:           &state,
        scratch:         &scratch,
        reference:       &first,
        probe:           &insertion,
        reference_index: 0,
    );
    vk::CommandReferenceProbe duplicate = vk::probe_command_reference(
        &state,
        &scratch,
        &first,
    )!!;
    assert(duplicate.found && duplicate.reference_index == 0,
        "an exact resource identity resolves to its reference slot");

    vk::CommandResourceReference owner_collision = first;
    do {
        owner_collision.owner++;
    } while ((vk::command_reference_hash(&owner_collision)
            ^ (vk::command_reference_hash(&owner_collision) >> 32))
        % cells.len != (vk::command_reference_hash(&first)
            ^ (vk::command_reference_hash(&first) >> 32))
        % cells.len);
    vk::CommandResourceReference index_collision = first;
    do {
        index_collision.index++;
    } while ((vk::command_reference_hash(&index_collision)
            ^ (vk::command_reference_hash(&index_collision) >> 32))
        % cells.len != (vk::command_reference_hash(&first)
            ^ (vk::command_reference_hash(&first) >> 32))
        % cells.len);
    vk::CommandResourceReference generation_collision = first;
    do {
        generation_collision.generation++;
    } while ((vk::command_reference_hash(&generation_collision)
            ^ (vk::command_reference_hash(&generation_collision) >> 32))
        % cells.len != (vk::command_reference_hash(&first)
            ^ (vk::command_reference_hash(&first) >> 32))
        % cells.len);
    vk::CommandReferenceProbe owner_insertion = vk::probe_command_reference(
        &state,
        &scratch,
        &owner_collision,
    )!!;
    vk::publish_command_reference(
        state:           &state,
        scratch:         &scratch,
        reference:       &owner_collision,
        probe:           &owner_insertion,
        reference_index: 1,
    );
    vk::CommandReferenceProbe index_insertion = vk::probe_command_reference(
        &state,
        &scratch,
        &index_collision,
    )!!;
    vk::publish_command_reference(
        state:           &state,
        scratch:         &scratch,
        reference:       &index_collision,
        probe:           &index_insertion,
        reference_index: 2,
    );
    vk::CommandReferenceProbe generation_insertion =
        vk::probe_command_reference(
            &state,
            &scratch,
            &generation_collision,
        )!!;
    vk::publish_command_reference(
        state:           &state,
        scratch:         &scratch,
        reference:       &generation_collision,
        probe:           &generation_insertion,
        reference_index: 3,
    );
    vk::CommandReferenceProbe first_again = vk::probe_command_reference(
        &state,
        &scratch,
        &first,
    )!!;
    vk::CommandReferenceProbe owner_again = vk::probe_command_reference(
        &state,
        &scratch,
        &owner_collision,
    )!!;
    vk::CommandReferenceProbe index_again = vk::probe_command_reference(
        &state,
        &scratch,
        &index_collision,
    )!!;
    vk::CommandReferenceProbe generation_again = vk::probe_command_reference(
        &state,
        &scratch,
        &generation_collision,
    )!!;
    vk::CommandReferenceWorkCounters counters =
        vk::command_reference_work_counters(&state);
    assert(!owner_insertion.found
        && !index_insertion.found
        && !generation_insertion.found
        && first_again.found && first_again.reference_index == 0
        && owner_again.found && owner_again.reference_index == 1
        && index_again.found && index_again.reference_index == 2
        && generation_again.found
            && generation_again.reference_index == 3,
        "colliding owner, index, and generation values remain distinct identities");
    assert(counters.lookups == 9
        && counters.probes == 21
        && counters.equality_checks == 17
        && counters.index_publications == 4,
        "reference-index counters expose exact lookup, probe, equality, and publication work");
}

fn void command_reference_duplicate_avoids_mutex_and_retain_work() @test {
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
    OwnedSpan owned = policy_span(&device, { .graphics }, "reference_index");
    defer free_owned_span(&device, &owned)!!;
    vk::VkDeviceState* state = policy_device_state(&device);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    vk::reset_command_reference_work_counters(state);

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
    vk::CommandReferenceWorkCounters counters =
        vk::command_reference_work_counters(state);
    assert(record.scratch.reference_count == 1
        && counters.lookups == 2
        && counters.duplicate_hits == 1
        && counters.resource_mutex_acquisitions == 1
        && counters.retains == 1
        && counters.index_publications == 1,
        "a duplicate lookup returns before mutex, retain, and publication work");
    gpu::discard_commands(&commands)!!;

    commands = gpu::begin_commands(&allocator)!!;
    record = vk::resolve_command(&commands)!!;
    vk::reset_command_reference_work_counters(state);
    vk::track_command_reference(
        state:      state,
        record:     record,
        owner:      owned.span.owner,
        index:      owned.span.index,
        generation: owned.span.generation,
    )!!;
    counters = vk::command_reference_work_counters(state);
    assert(record.scratch.reference_count == 1
        && counters.duplicate_hits == 0
        && counters.resource_mutex_acquisitions == 1
        && counters.retains == 1,
        "discard resets the index epoch before the scratch buffer is reused");
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    gpu::wait_completion(completion)!!;

    commands = gpu::begin_commands(&allocator)!!;
    record = vk::resolve_command(&commands)!!;
    vk::reset_command_reference_work_counters(state);
    vk::track_command_reference(
        state:      state,
        record:     record,
        owner:      owned.span.owner,
        index:      owned.span.index,
        generation: owned.span.generation,
    )!!;
    counters = vk::command_reference_work_counters(state);
    assert(record.scratch.reference_count == 1
        && counters.duplicate_hits == 0
        && counters.resource_mutex_acquisitions == 1
        && counters.retains == 1,
        "completion retirement resets the index epoch before scratch reuse");
    gpu::discard_commands(&commands)!!;
}

fn void texture_reference_publishes_after_retain_and_deduplicates() @test {
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
    vk::CommandReferenceIndexCell[4] cells;
    vk::CommandScratch scratch = {
        .references      = references[..],
        .reference_index = {
            .cells = cells[..],
            .epoch = 1,
        },
    };
    vk::CommandRecord record = { .scratch = &scratch };
    gpu::TextureHandle valid = {
        .owner      = state.textures.owner,
        .index      = 0,
        .generation = 1,
    };
    gpu::TextureHandle stale = valid;
    stale.generation++;
    vk::reset_command_policy_counters(&state);
    vk::reset_command_reference_work_counters(&state);

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
    vk::CommandPolicyCounters policy = vk::command_policy_counters(&state);
    vk::CommandReferenceWorkCounters work =
        vk::command_reference_work_counters(&state);
    assert(scratch.reference_count == 1
        && texture_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 1,
        "a successful texture retain publishes exactly one reference");
    assert(policy.tracking_calls == 3
        && policy.reference_increments == 1
        && work.lookups == 3
        && work.duplicate_hits == 1
        && work.resource_mutex_acquisitions == 2
        && work.retains == 1
        && work.index_publications == 1,
        "failed and duplicate texture paths cannot publish or retain extra references");

    vk::clear_command_scratch_references(&state, &scratch);
    assert(texture_cells[0].value.tracked_command_references.load(
        AtomicOrdering.RELAXED,
    ) == 0,
        "texture reference cleanup releases the successful retain");
}

fn void command_reference_rollback_releases_suffix_and_rebuilds_epoch() @test {
    vk::VkDeviceState state;
    Atomic{uint}[3] tracked;
    vk::CommandResourceReference[3] references;
    foreach (i, &counter : tracked) {
        counter.store(1, AtomicOrdering.RELAXED);
        references[i] = {
            .owner                       = 31,
            .index                       = (uint)i,
            .generation                  = 1,
            .tracked_command_references  = counter,
        };
    }
    vk::CommandReferenceIndexCell[8] cells;
    vk::CommandScratch scratch = {
        .references      = references[..],
        .reference_index = {
            .cells = cells[..],
            .epoch = 7,
        },
        .reference_count = 3,
    };
    for (uint i = 0; i < scratch.reference_count; i++) {
        vk::CommandReferenceProbe probe = vk::probe_command_reference(
            &state,
            &scratch,
            &scratch.references[i],
        )!!;
        vk::publish_command_reference(
            state:           &state,
            scratch:         &scratch,
            reference:       &scratch.references[i],
            probe:           &probe,
            reference_index: i,
        );
    }
    vk::CommandRecord record = { .scratch = &scratch };
    vk::reset_command_reference_work_counters(&state);

    vk::rollback_command_references(&state, &record, 1);
    vk::CommandReferenceWorkCounters counters =
        vk::command_reference_work_counters(&state);
    assert(scratch.reference_count == 1 && scratch.reference_index.epoch == 8,
        "rollback advances to a fresh epoch and keeps only the prefix");
    assert(tracked[0].load(AtomicOrdering.RELAXED) == 1
        && tracked[1].load(AtomicOrdering.RELAXED) == 0
        && tracked[2].load(AtomicOrdering.RELAXED) == 0,
        "rollback releases only the discarded reference suffix");
    assert(counters.releases == 2
        && counters.rollback_rebuild_publications == 1
        && counters.lookups == 1
        && counters.index_publications == 1,
        "rollback counters expose suffix releases and one prefix rebuild");
    vk::CommandReferenceProbe retained = vk::probe_command_reference(
        &state,
        &scratch,
        &scratch.references[0],
    )!!;
    assert(retained.found && retained.reference_index == 0,
        "the retained prefix is queryable after rollback rebuild");
}

fn void command_reference_noop_rollback_preserves_epoch_and_index() @test {
    vk::VkDeviceState state;
    vk::CommandResourceReference[1] references;
    references[0] = {
        .owner      = 37,
        .index      = 11,
        .generation = 3,
    };
    vk::CommandReferenceIndexCell[4] cells;
    vk::CommandScratch scratch = {
        .references      = references[..],
        .reference_index = {
            .cells = cells[..],
            .epoch = 9,
        },
        .reference_count = 1,
    };
    vk::CommandReferenceProbe insertion = vk::probe_command_reference(
        &state,
        &scratch,
        &scratch.references[0],
    )!!;
    vk::publish_command_reference(
        state:           &state,
        scratch:         &scratch,
        reference:       &scratch.references[0],
        probe:           &insertion,
        reference_index: 0,
    );
    vk::CommandRecord record = { .scratch = &scratch };
    vk::reset_command_reference_work_counters(&state);

    vk::rollback_command_references(&state, &record, 1);
    vk::CommandReferenceWorkCounters counters =
        vk::command_reference_work_counters(&state);
    assert(scratch.reference_count == 1 && scratch.reference_index.epoch == 9,
        "a no-op rollback preserves the retained prefix and current epoch");
    assert(counters.releases == 0
        && counters.lookups == 0
        && counters.index_publications == 0
        && counters.rollback_rebuild_publications == 0,
        "a no-op rollback performs no release, reset, or rebuild work");
    vk::CommandReferenceProbe retained = vk::probe_command_reference(
        &state,
        &scratch,
        &scratch.references[0],
    )!!;
    assert(retained.found && retained.reference_index == 0,
        "the retained prefix remains queryable after a no-op rollback");
}

fn void command_reference_retain_failure_rolls_back_without_publication() @test {
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
    vk::CommandReferenceIndexCell[4] cells;
    vk::CommandScratch scratch = {
        .references      = references[..],
        .reference_index = {
            .cells = cells[..],
            .epoch = 1,
        },
    };
    vk::CommandRecord record = { .scratch = &scratch };
    vk::reset_command_reference_work_counters(&state);

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
    vk::rollback_command_references(&state, &record, 0);

    vk::CommandReferenceWorkCounters counters =
        vk::command_reference_work_counters(&state);
    bool current_epoch_empty = true;
    foreach (&cell : cells) {
        current_epoch_empty &= cell.epoch != scratch.reference_index.epoch;
    }
    assert(scratch.reference_count == 0
        && scratch.reference_index.epoch == 2
        && current_epoch_empty
        && buffer_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0,
        "rollback removes the retained prefix and leaves no live index entry");
    assert(counters.lookups == 2
        && counters.resource_mutex_acquisitions == 2
        && counters.retains == 1
        && counters.index_publications == 1
        && counters.releases == 1
        && counters.rollback_rebuild_publications == 0,
        "a failed retain publishes nothing and rollback releases prior work once");
}

fn void command_reference_epoch_rollover_clears_each_index_cell() @test {
    vk::VkDeviceState state;
    vk::CommandReferenceIndexCell[4] cells;
    foreach (&cell : cells) cell.epoch = uint::max;
    vk::CommandScratch scratch = {
        .reference_index = {
            .cells = cells[..],
            .epoch = uint::max,
        },
    };
    vk::reset_command_reference_work_counters(&state);

    vk::reset_command_reference_index(&state, &scratch);
    vk::CommandReferenceWorkCounters counters =
        vk::command_reference_work_counters(&state);
    bool cells_cleared = true;
    foreach (&cell : cells) cells_cleared &= cell.epoch == 0;
    assert(scratch.reference_index.epoch == 1 && cells_cleared,
        "epoch rollover clears stale occupancy before restarting at one");
    assert(counters.rollover_cells_cleared == cells.len,
        "epoch rollover reports exactly the cells it clears");

    vk::CommandResourceReference reference = {
        .owner      = 43,
        .index      = 2,
        .generation = 9,
    };
    vk::CommandReferenceProbe insertion = vk::probe_command_reference(
        &state,
        &scratch,
        &reference,
    )!!;
    vk::publish_command_reference(
        state:           &state,
        scratch:         &scratch,
        reference:       &reference,
        probe:           &insertion,
        reference_index: 0,
    );
    vk::CommandReferenceProbe duplicate = vk::probe_command_reference(
        &state,
        &scratch,
        &reference,
    )!!;
    assert(!insertion.found && duplicate.found,
        "the restarted epoch accepts insertion and exact duplicate lookup");
}
