module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;

const uint QF_G = (uint)vk::QueueFlagBits.GRAPHICS_BIT;
const uint QF_C = (uint)vk::QueueFlagBits.COMPUTE_BIT;
const uint QF_T = (uint)vk::QueueFlagBits.TRANSFER_BIT;

fn vk::QueueFamilyProperties family(uint flags, uint count) {
    return { .queue_flags = (vk::QueueFlags)flags, .queue_count = count };
}

fn gpu::Device make_queue_test_device() {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_queue_selection_test",
    };
    return create_test_device(&desc)!!;
}

fn void default_selection_minimizes_native_identities() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G | QF_C | QF_T, 16),
        family(QF_C | QF_T, 8),
        family(QF_T, 2),
    };

    vk::QueueFamilies picked = vk::pick_queue_families(table[..])!!;

    assert(picked.graphics == 0 && picked.graphics_index == 0);
    assert(picked.compute == 0 && picked.compute_index == 0);
    assert(picked.transfer == 0 && picked.transfer_index == 0);
    assert(!picked.async_compute);
}

fn void no_graphics_family_faults_for_default_request() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_C | QF_T, 4),
    };

    fault result = @catch(vk::pick_queue_families(table[..]));

    assert(result == gpu::UNSUPPORTED_FEATURE);
}

fn void graphics_only_selection_has_no_hidden_roles() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G, 1),
        family(QF_C, 1),
        family(QF_T, 1),
    };
    gpu::QueueRequest request = {
        .required = { .graphics },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_request(table[..], request)!!;

    assert(picked.selected.graphics);
    assert(!picked.selected.compute && !picked.selected.transfer);
    assert(picked.graphics == 0);
}

fn void compute_only_selection_has_no_hidden_roles() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G, 1),
        family(QF_C, 1),
        family(QF_T, 1),
    };
    gpu::QueueRequest request = {
        .required = { .compute },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_request(table[..], request)!!;

    assert(!picked.selected.graphics && picked.selected.compute);
    assert(!picked.selected.transfer && !picked.async_compute);
    assert(picked.compute == 1);
}

fn void transfer_only_selection_has_no_hidden_roles() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G, 1),
        family(QF_C, 1),
        family(QF_T, 1),
    };
    gpu::QueueRequest request = {
        .required = { .transfer },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_request(table[..], request)!!;

    assert(!picked.selected.graphics && !picked.selected.compute);
    assert(picked.selected.transfer && !picked.async_compute);
    assert(picked.transfer == 0);
}

fn void distinct_compute_uses_a_second_native_queue() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 2),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute },
        .distinct = { .compute },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_request(table[..], request)!!;

    assert(picked.graphics == picked.compute);
    assert(picked.graphics_index == 0 && picked.compute_index == 1);
    assert(picked.async_compute);
}

fn void distinct_compute_prefers_a_compute_only_family() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_G | QF_C | QF_T, 2),
        family(QF_C | QF_T, 1),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute },
        .distinct = { .compute },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_request(table[..], request)!!;

    assert(picked.graphics == 0 && picked.graphics_index == 0);
    assert(picked.compute == 1 && picked.compute_index == 0);
}

fn void distinct_graphics_leaves_other_roles_aliased() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 2),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute, .transfer },
        .distinct = { .graphics },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_request(table[..], request)!!;

    assert(picked.graphics_index == 0);
    assert(picked.compute_index == 1 && picked.transfer_index == 1);
}

fn void distinct_transfer_prefers_a_transfer_only_family() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_G | QF_C | QF_T, 2),
        family(QF_T, 1),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute, .transfer },
        .distinct = { .transfer },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_request(table[..], request)!!;

    assert(picked.graphics == 0 && picked.compute == 0);
    assert(picked.graphics_index == 0 && picked.compute_index == 0);
    assert(picked.transfer == 1 && picked.transfer_index == 0);
}

fn void distinct_roles_use_split_capability_families() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G, 1),
        family(QF_C, 1),
        family(QF_T, 1),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute, .transfer },
        .distinct = { .graphics, .compute, .transfer },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_request(table[..], request)!!;

    assert(picked.graphics == 0 && picked.graphics_index == 0);
    assert(picked.compute == 1 && picked.compute_index == 0);
    assert(picked.transfer == 2 && picked.transfer_index == 0);
}

fn void unavailable_distinct_queue_faults() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 1),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute },
        .distinct = { .compute },
    };

    fault result =
        @catch(vk::pick_queue_families_for_request(table[..], request));

    assert(result == gpu::UNSUPPORTED_FEATURE);
}

fn void presentation_selects_a_surface_capable_graphics_queue() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G | QF_C | QF_T, 1),
        family(QF_G | QF_C | QF_T, 2),
        family(QF_T, 1),
    };
    bool[3] presentation_support = { false, true, false };

    vk::QueueFamilies picked = vk::pick_queue_families_for_surface(
        table[..],
        presentation_support[..],
    )!!;

    assert(picked.graphics == 1 && picked.graphics_index == 0);
    assert(picked.compute == 1 && picked.compute_index == 0);
    assert(picked.transfer == 1 && picked.transfer_index == 0);
    assert(picked.present == 1 && picked.present_index == 0);
}

fn void presentation_can_use_a_selected_non_graphics_queue() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_G | QF_C | QF_T, 1),
        family(QF_C | QF_T, 2),
    };
    bool[2] presentation_support = { false, true };

    vk::QueueFamilies picked = vk::pick_queue_families_for_surface(
        table[..],
        presentation_support[..],
    )!!;

    assert(picked.graphics == 0);
    assert(picked.compute == 1 && picked.transfer == 0);
    assert(picked.present == picked.compute);
}

fn void presentation_can_use_a_private_queue_family() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_G | QF_C | QF_T, 1),
        family(0, 1),
    };
    bool[2] presentation_support = { false, true };

    vk::QueueFamilies picked = vk::pick_queue_families_for_surface(
        table[..],
        presentation_support[..],
    )!!;

    assert(picked.graphics == 0 && picked.compute == 0);
    assert(picked.transfer == 0);
    assert(picked.present == 1 && picked.presentation);
}

fn void presentation_support_must_cover_every_queue_family() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 1),
    };

    fault result =
        @catch(vk::pick_queue_families_for_surface(table[..], {}));

    assert(result == gpu::INVALID_ARGUMENT);
}

fn void presentation_requires_a_capable_queue_family() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 1),
    };
    bool[1] presentation_support = { false };

    fault result = @catch(vk::pick_queue_families_for_surface(
        table[..],
        presentation_support[..],
    ));

    assert(result == gpu::UNSUPPORTED_FEATURE);
}

fn void presentation_without_graphics_is_rejected() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_C, 1),
        family(0, 1),
    };
    bool[2] presentation_support = { false, true };
    gpu::QueueRequest request = {
        .required = { .compute },
    };

    fault result = @catch(vk::pick_queue_families_for_surface_request(
        table[..],
        presentation_support[..],
        request,
    ));

    assert(result == gpu::INVALID_ARGUMENT);
}

fn void presentation_queue_selection_requires_instance_dispatch() @test {
    vk::VkRuntimeState runtime;

    fault result = @catch(vk::select_queue_families_for_surface(
        &runtime,
        (vk::PhysicalDevice)(uptr)0x10,
        (vk::SurfaceKHR)(uptr)0x20,
    ));

    assert(result == gpu::UNSUPPORTED_FEATURE);
}

fn void selected_device_families_are_unique() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G, 1),
        family(QF_C, 1),
        family(QF_T, 1),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute, .transfer },
        .distinct = { .graphics, .compute, .transfer },
    };
    vk::QueueFamilies picked =
        vk::pick_queue_families_for_request(table[..], request)!!;
    uint[3] families;

    uint count = vk::selected_device_queue_families(
        &picked,
        false,
        families[..],
    );

    assert(count == 3);
    assert(families[0] == 0 && families[1] == 1 && families[2] == 2);
}

fn void used_queue_families_keep_distinct_transfer() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute, .transfer },
        .graphics_family      = 4,
        .compute_family       = 5,
        .transfer_family      = 7,
    };
    uint[3] families;

    uint count = vk::used_queue_families(&state, families[..]);

    assert(count == 3);
    assert(families[0] == 4 && families[1] == 5 && families[2] == 7);
}

fn void requested_queue_count_covers_the_highest_selected_index() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 2),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute, .transfer },
        .distinct = { .transfer },
    };
    vk::QueueFamilies picked =
        vk::pick_queue_families_for_request(table[..], request)!!;

    assert(vk::requested_queue_count(&picked, 0, false) == 2);
}

fn void shader_heap_families_alias_without_transfer() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute },
        .graphics_family      = 4,
        .compute_family       = 4,
        .transfer_family      = 7,
    };
    uint[2] families;

    uint count = vk::shader_heap_queue_families(&state, families[..]);

    assert(count == 1);
    assert(families[0] == 4);
}

fn void shader_heap_families_are_graphics_compute_only() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute },
        .graphics_family      = 4,
        .compute_family       = 5,
        .transfer_family      = 7,
    };
    uint[2] families;

    uint count = vk::shader_heap_queue_families(&state, families[..]);

    assert(count == 2);
    assert(families[0] == 4 && families[1] == 5);
}

fn void resource_access_families_include_only_admitted_roles() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute, .transfer },
        .graphics_family      = 4,
        .compute_family       = 5,
        .transfer_family      = 7,
    };
    uint[3] families;

    uint count = vk::resource_queue_families(
        &state,
        { .graphics, .compute },
        families[..],
    );

    assert(count == 2);
    assert(families[0] == 4 && families[1] == 5);

    state.compute_family = 4;
    count = vk::resource_queue_families(
        &state,
        { .graphics, .compute },
        families[..],
    );
    assert(count == 1 && families[0] == 4);
}

fn void resource_texture_create_info_uses_exact_access_families() @test {
    gpu::TextureDesc desc = {
        .access = { .graphics, .compute },
        .width  = 4,
        .height = 4,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled, .storage },
    };
    uint[1] aliased = { 4 };
    uint[2] split = { 4, 5 };

    vk::ImageCreateInfo exclusive =
        vk::make_texture_create_info(&desc, aliased[..]);
    vk::ImageCreateInfo concurrent =
        vk::make_texture_create_info(&desc, split[..]);

    assert(exclusive.sharing_mode == vk::SharingMode.EXCLUSIVE);
    assert(exclusive.queue_family_index_count == 0);
    assert(concurrent.sharing_mode == vk::SharingMode.CONCURRENT);
    assert(concurrent.queue_family_index_count == 2);
    assert(concurrent.p_queue_family_indices[0] == 4);
    assert(concurrent.p_queue_family_indices[1] == 5);
}

fn void swapchain_sharing_aliases_the_graphics_family() @test {
    vk::Queue queue = (vk::Queue)(uptr)0x10;
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics },
        .graphics_family      = 4,
        .graphics_queue       = queue,
        .presentation         = {
            .enabled = true,
            .family  = 4,
            .queue   = queue,
        },
    };
    uint[2] families;
    vk::Queue[2] queues;

    uint family_count = vk::swapchain_queue_families(&state, families[..]);
    uint queue_count = vk::swapchain_idle_queues(&state, queues[..]);

    assert(family_count == 1 && families[0] == 4);
    assert(queue_count == 1 && queues[0] == queue);
}

fn void swapchain_sharing_includes_a_separate_present_family() @test {
    vk::Queue graphics_queue = (vk::Queue)(uptr)0x10;
    vk::Queue present_queue = (vk::Queue)(uptr)0x20;
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics },
        .graphics_family      = 4,
        .graphics_queue       = graphics_queue,
        .presentation         = {
            .enabled = true,
            .family  = 7,
            .queue   = present_queue,
        },
    };
    uint[2] families;
    vk::Queue[2] queues;

    uint family_count = vk::swapchain_queue_families(&state, families[..]);
    uint queue_count = vk::swapchain_idle_queues(&state, queues[..]);

    assert(family_count == 2);
    assert(families[0] == 4 && families[1] == 7);
    assert(queue_count == 2);
    assert(queues[0] == graphics_queue && queues[1] == present_queue);
}

fn void aliased_compute_and_transfer_share_one_queue_mutex() @test {
    vk::Queue graphics_queue = (vk::Queue)(uptr)0x10;
    vk::Queue compute_queue = (vk::Queue)(uptr)0x20;
    vk::QueueCompletionState[2] completions = {
        {
            .queue    = graphics_queue,
            .timeline = (vk::Semaphore)(uptr)0x30,
            .roles    = { .graphics },
            .bound    = true,
        },
        {
            .queue    = compute_queue,
            .timeline = (vk::Semaphore)(uptr)0x40,
            .roles    = { .compute, .transfer },
            .bound    = true,
        },
    };
    vk::VkDeviceState state = {
        .selected_queue_roles  = { .graphics, .compute, .transfer },
        .graphics_queue        = graphics_queue,
        .compute_queue         = compute_queue,
        .transfer_queue        = compute_queue,
        .queue_completions     = completions[..],
        .graphics_completion_id = 0,
        .compute_completion_id  = 1,
        .transfer_completion_id = 1,
        .compute_queue_distinct = true,
    };

    assert(
        vk::queue_mutex_for(&state, gpu::QueueKind.COMPUTE)
            == vk::queue_mutex_for(&state, gpu::QueueKind.TRANSFER)
    );
}

fn void queue_alias_matches_async_compute_cap() @test {
    gpu::Device device = make_queue_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;

    assert(state.compute_queue_distinct == caps.async_compute);
    assert((state.compute_queue == state.graphics_queue) == !caps.async_compute);
    uint expected_index = state.compute_queue_distinct
        && state.compute_family == state.graphics_family
        ? 1
        : 0;
    assert(state.compute_queue_index == expected_index);
}

fn void allocator_pool_ownership_is_exact() @test {
    gpu::Device device = make_queue_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue graphics =
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::Queue compute =
        gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::Queue transfer =
        gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;
    gpu::CommandAllocator graphics_allocator =
        gpu::create_command_allocator(&device, graphics)!!;
    defer gpu::destroy_command_allocator(&graphics_allocator)!!;
    gpu::CommandAllocator compute_allocator =
        gpu::create_command_allocator(&device, compute)!!;
    defer gpu::destroy_command_allocator(&compute_allocator)!!;
    gpu::CommandAllocator transfer_allocator =
        gpu::create_command_allocator(&device, transfer)!!;
    defer gpu::destroy_command_allocator(&transfer_allocator)!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    vk::CommandAllocatorSlot* graphics_slot =
        state.command_allocators.get(graphics_allocator.handle)!!;
    vk::CommandAllocatorSlot* compute_slot =
        state.command_allocators.get(compute_allocator.handle)!!;
    vk::CommandAllocatorSlot* transfer_slot =
        state.command_allocators.get(transfer_allocator.handle)!!;

    assert(graphics_slot.queue == graphics);
    assert(compute_slot.queue == compute);
    assert(transfer_slot.queue == transfer);
    assert(graphics_slot.pool != compute_slot.pool);
    assert(graphics_slot.pool != transfer_slot.pool);
    assert(compute_slot.pool != transfer_slot.pool);
}

fn void cross_queue_timeline_chain_validates() @test {
    gpu::Device device = make_queue_test_device();
    defer destroy_test_device(&device)!!;
    gpu::Queue compute_queue =
        gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::Queue graphics_queue =
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator compute_allocator =
        gpu::create_command_allocator(&device, compute_queue)!!;
    defer gpu::destroy_command_allocator(&compute_allocator)!!;
    gpu::CommandAllocator graphics_allocator =
        gpu::create_command_allocator(&device, graphics_queue)!!;
    defer gpu::destroy_command_allocator(&graphics_allocator)!!;
    gpu::CommandList compute_commands =
        gpu::begin_commands(&compute_allocator)!!;
    defer discard_test_commands(&compute_commands)!!;
    gpu::ExecutableCommandList compute_executable =
        gpu::end_commands(&compute_commands)!!;
    defer discard_test_executable(&compute_executable)!!;
    gpu::ExecutableCommandList[1] compute_lists = { compute_executable };
    gpu::SubmitDesc compute_submit = {
        .command_lists = compute_lists[..],
    };
    gpu::CompletionPoint compute_done = gpu::submit(
        compute_queue,
        &compute_submit,
    )!!;
    gpu::CommandList graphics_commands =
        gpu::begin_commands(&graphics_allocator)!!;
    defer discard_test_commands(&graphics_commands)!!;
    gpu::ExecutableCommandList graphics_executable =
        gpu::end_commands(&graphics_commands)!!;
    defer discard_test_executable(&graphics_executable)!!;
    gpu::ExecutableCommandList[1] graphics_lists = { graphics_executable };
    gpu::CompletionWait[1] waits = {{
        .point  = compute_done,
        .before = { .all },
    }};
    gpu::SubmitDesc graphics_submit = {
        .command_lists    = graphics_lists[..],
        .completion_waits = waits[..],
    };

    gpu::CompletionPoint graphics_done = gpu::submit(
        graphics_queue,
        &graphics_submit,
    )!!;

    gpu::wait_completion(graphics_done, 1_000_000_000)!!;
}

fn void selected_role_completions_wait_cleanly() @test {
    gpu::Device device = make_queue_test_device();
    defer destroy_test_device(&device)!!;
    gpu::QueueKind[3] kinds = {
        gpu::QueueKind.GRAPHICS,
        gpu::QueueKind.COMPUTE,
        gpu::QueueKind.TRANSFER,
    };
    gpu::CommandAllocator[3] allocators;
    for (uint i = 0; i < kinds.len; i++) {
        gpu::Queue queue = gpu::get_queue(&device, kinds[i])!!;
        allocators[i] = gpu::create_command_allocator(&device, queue)!!;
    }
    defer {
        for (uint i = 0; i < allocators.len; i++) {
            (void)gpu::destroy_command_allocator(&allocators[i]);
        }
    }

    gpu::CompletionPoint[3] completions;
    for (uint i = 0; i < kinds.len; i++) {
        gpu::CommandList commands = gpu::begin_commands(&allocators[i])!!;
        defer discard_test_commands(&commands)!!;
        gpu::ExecutableCommandList executable =
            gpu::end_commands(&commands)!!;
        gpu::ExecutableCommandList[1] lists = { executable };
        gpu::SubmitDesc submit = {
            .command_lists = lists[..],
        };
        completions[i] = gpu::submit(
            gpu::get_queue(&device, kinds[i])!!,
            &submit,
        )!!;
    }
    foreach (completion : completions) {
        gpu::wait_completion(completion)!!;
    }
}

fn void distinct_role_device_queues_are_unique() @test {
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_distinct_queue_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::QueueRequest request = {
        .required = { .graphics, .compute, .transfer },
        .distinct = { .graphics, .compute, .transfer },
    };
    gpu::DeviceDesc device_desc = {
        .queues = request,
    };

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(&adapter, &device_desc)!!.supported) {
            continue;
        }

        gpu::Device device = gpu::create_device(&adapter, &device_desc)!!;
        defer gpu::destroy_device(&device)!!;
        gpu::Queue graphics =
            gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::Queue compute =
            gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
        gpu::Queue transfer =
            gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!;

        assert(graphics.id != compute.id);
        assert(graphics.id != transfer.id);
        assert(compute.id != transfer.id);
        return;
    }
}

fn void multi_role_access_stays_exclusive_on_single_family() @test {
    gpu::Device device = make_queue_test_device();
    defer destroy_test_device(&device)!!;
    gpu::AllocationDesc allocation_desc = {
        .size         = 256,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .graphics, .compute, .transfer },
        .debug_name   = "multi_role_data",
    };
    gpu::GpuAllocation allocation =
        gpu::allocate_memory(&device, &allocation_desc)!!;
    gpu::free_allocation(&device, &allocation)!!;
    gpu::TextureDesc texture_desc = {
        .access = { .graphics, .compute, .transfer },
        .width  = 4,
        .height = 4,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle texture =
        gpu::create_texture(&device, &texture_desc)!!;
    gpu::destroy_texture(&device, texture)!!;
}
