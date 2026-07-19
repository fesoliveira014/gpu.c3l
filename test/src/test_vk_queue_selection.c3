module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::atomic;

// Pure selection-tier units over synthetic family tables (no device), plus
// lavapipe integration: alias tier, caps flag, cross-queue timeline chain,
// and semantic multi-role access on a single-family device.

const uint QF_G = (uint)vk::QueueFlagBits.GRAPHICS_BIT;
const uint QF_C = (uint)vk::QueueFlagBits.COMPUTE_BIT;
const uint QF_T = (uint)vk::QueueFlagBits.TRANSFER_BIT;

fn vk::QueueFamilyProperties family(uint flags, uint count) {
    return { .queue_flags = (vk::QueueFlags)flags, .queue_count = count };
}

fn void nvidia_like_uses_second_main_queue() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G | QF_C | QF_T, 16),
        family(QF_C | QF_T, 8),
        family(QF_T, 2),
    };
    vk::QueueFamilies picked = vk::pick_queue_families(table[..])!!;
    assert(picked.graphics == 0 && picked.compute == 0 && picked.compute_index == 1, "tier 1: second queue in the main family");
    assert(picked.async_compute, "tier 1 reports async");
    assert(picked.transfer == 2, "dedicated transfer family selected");
}

fn void amd_like_uses_compute_only_family() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G | QF_C | QF_T, 1),
        family(QF_C | QF_T, 4),
        family(QF_T, 2),
    };
    vk::QueueFamilies picked = vk::pick_queue_families(table[..])!!;
    assert(picked.compute == 1 && picked.compute_index == 0, "tier 2: compute-only family");
    assert(picked.async_compute, "tier 2 reports async");
}

fn void single_family_aliases() @test {
    vk::QueueFamilyProperties[1] table = { family(QF_G | QF_C | QF_T, 1) };
    vk::QueueFamilies picked = vk::pick_queue_families(table[..])!!;
    assert(picked.compute == 0 && picked.compute_index == 0, "tier 3: alias");
    assert(!picked.async_compute, "alias reports no async");
    assert(picked.transfer == 0, "transfer reuses the main family");
}

fn void no_graphics_family_faults() @test {
    vk::QueueFamilyProperties[1] table = { family(QF_C | QF_T, 4) };
    fault f = @catch(vk::pick_queue_families(table[..]));
    assert(f == gpu::UNSUPPORTED_FEATURE, "no graphics family faults");
}

fn void presentation_selects_a_surface_capable_main_family() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G | QF_C | QF_T, 1),
        family(QF_G | QF_C | QF_T, 2),
        family(QF_T, 1),
    };
    bool[3] present_support = { false, true, false };

    vk::QueueFamilies picked = vk::pick_queue_families_for_surface(
        table[..],
        present_support[..],
    )!!;

    assert(picked.graphics == 1, "presentation chooses the supported graphics+compute family");
    assert(picked.present == 1, "the main family also presents");
    assert(picked.compute == 1 && picked.compute_index == 1, "main-family compute tier remains available");
    assert(picked.transfer == 2, "dedicated transfer selection is preserved");
}

fn void presentation_can_use_a_separate_queue_family() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_G | QF_C | QF_T, 1),
        family(QF_C | QF_T, 2),
    };
    bool[2] present_support = { false, true };

    vk::QueueFamilies picked = vk::pick_queue_families_for_surface(
        table[..],
        present_support[..],
    )!!;

    assert(picked.graphics == 0, "graphics remains on the main family");
    assert(picked.present == 1, "presentation uses the supported separate family");
    assert(picked.compute == 1, "compute selection remains independent");
}

fn void presentation_support_must_cover_every_queue_family() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 1),
    };

    fault result = @catch(vk::pick_queue_families_for_surface(table[..], {}));

    assert(result == gpu::INVALID_ARGUMENT, "mismatched presentation support is rejected");
}

fn void presentation_requires_a_capable_queue_family() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 1),
    };
    bool[1] present_support = { false };

    fault result = @catch(vk::pick_queue_families_for_surface(
        table[..],
        present_support[..],
    ));

    assert(result == gpu::UNSUPPORTED_FEATURE, "no presentation-capable family faults");
}

fn void presentation_queue_selection_requires_instance_dispatch() @test {
    vk::VkRuntimeState runtime;
    fault result = @catch(vk::select_queue_families_for_surface(
        &runtime,
        (vk::PhysicalDevice)(uptr)0x10,
        (vk::SurfaceKHR)(uptr)0x20,
    ));
    assert(result == gpu::UNSUPPORTED_FEATURE, "surface selection requires instance dispatch");
}

fn vk::PhysicalDevice fake_physical_device(uptr value) {
    return (vk::PhysicalDevice)value;
}

fn vk::PhysicalDeviceCandidate candidate(uptr device, int score, bool suitable, vk::QueueFamilies families = {}) {
    return {
        .selection = {
            .physical_device = fake_physical_device(device),
            .families = families,
        },
        .score = score,
        .suitable = suitable,
    };
}

fn void later_unsuitable_high_score_does_not_replace_selection() @test {
    vk::PhysicalDeviceCandidate[2] candidates = {
        candidate(device: 0x92, score: 50, suitable: true),
        candidate(device: 0x93, score: 100, suitable: false),
    };
    vk::PhysicalDeviceSelection selected = vk::choose_physical_device(candidates[..])!!;
    assert(selected.physical_device == fake_physical_device(0x92),
        "later unsuitable candidate does not replace the selected adapter");
}

fn void unsuitable_high_score_falls_back_and_preserves_families() @test {
    vk::QueueFamilies fallback_families = {
        .graphics = 3,
        .compute = 5,
        .compute_index = 1,
        .transfer = 7,
        .async_compute = true,
    };
    vk::PhysicalDeviceCandidate[2] candidates = {
        candidate(0x10, 100, false),
        candidate(
            device: 0x20,
            score: 50,
            suitable: true,
            families: fallback_families,
        ),
    };

    vk::PhysicalDeviceSelection selected = vk::choose_physical_device(candidates[..])!!;
    assert(selected.physical_device == fake_physical_device(0x20), "lower-scored suitable adapter selected");
    assert(selected.families.graphics == 3, "cached graphics family preserved");
    assert(selected.families.compute == 5 && selected.families.compute_index == 1, "cached compute family preserved");
    assert(
        selected.families.transfer == 7 && selected.families.async_compute,
        "cached transfer and async fields preserved"
    );
}

fn void higher_scored_suitable_candidate_replaces_first_and_preserves_families() @test {
    vk::QueueFamilies first_families = { .graphics = 1, .compute = 1, .transfer = 1 };
    vk::QueueFamilies higher_scored_families = {
        .graphics = 3,
        .compute = 5,
        .compute_index = 1,
        .transfer = 7,
        .async_compute = true,
    };
    vk::PhysicalDeviceCandidate[2] candidates = {
        candidate(
            device: 0x30,
            score: 50,
            suitable: true,
            families: first_families,
        ),
        candidate(
            device: 0x40,
            score: 100,
            suitable: true,
            families: higher_scored_families,
        ),
    };

    vk::PhysicalDeviceSelection selected = vk::choose_physical_device(candidates[..])!!;
    assert(
        selected.physical_device == fake_physical_device(0x40),
        "higher-scored suitable adapter replaces first"
    );
    assert(selected.families.graphics == 3, "replacement graphics family preserved");
    assert(
        selected.families.compute == 5 && selected.families.compute_index == 1,
        "replacement compute family preserved"
    );
    assert(
        selected.families.transfer == 7 && selected.families.async_compute,
        "replacement transfer and async fields preserved"
    );
}

fn void device_score_wins_without_async_bonus() @test {
    vk::QueueFamilies alias_families = { .graphics = 1, .compute = 1, .transfer = 1 };
    vk::QueueFamilies async_families = {
        .graphics = 2,
        .compute = 2,
        .compute_index = 1,
        .transfer = 2,
        .async_compute = true,
    };
    vk::PhysicalDeviceCandidate[2] candidates = {
        candidate(
            device: 0x30,
            score: 100,
            suitable: true,
            families: alias_families,
        ),
        candidate(
            device: 0x40,
            score: 50,
            suitable: true,
            families: async_families,
        ),
    };

    vk::PhysicalDeviceSelection selected = vk::choose_physical_device(candidates[..])!!;
    assert(
        selected.physical_device == fake_physical_device(0x30),
        "device score remains the only cross-adapter preference"
    );
    assert(!selected.families.async_compute, "async topology is not a score bonus");
}

fn void equal_device_scores_keep_first_candidate() @test {
    vk::QueueFamilies first_families = { .graphics = 4, .compute = 4, .transfer = 4 };
    vk::PhysicalDeviceCandidate[2] candidates = {
        candidate(
            device: 0x50,
            score: 50,
            suitable: true,
            families: first_families,
        ),
        candidate(0x60, 50, true),
    };

    vk::PhysicalDeviceSelection selected = vk::choose_physical_device(candidates[..])!!;
    assert(selected.physical_device == fake_physical_device(0x50), "equal scores preserve enumeration order");
}

fn void no_suitable_physical_device_faults() @test {
    vk::PhysicalDeviceCandidate[2] unsuitable = {
        candidate(0x70, 100, false),
        candidate(0x80, 50, false),
    };
    fault empty_fault = @catch(vk::choose_physical_device({}));
    fault unsuitable_fault = @catch(vk::choose_physical_device(unsuitable[..]));
    assert(empty_fault == gpu::UNSUPPORTED_FEATURE, "empty candidate set faults unsupported");
    assert(unsuitable_fault == gpu::UNSUPPORTED_FEATURE, "all-unsuitable candidate set faults unsupported");
}

fn gpu::Device make_queue_test_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_queue_selection_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
}

fn void queue_alias_matches_async_compute_cap() @test {
    gpu::Device device = make_queue_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* s = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(s.compute_queue_distinct == gpu::get_device_caps(&device)!!.async_compute, "distinct flag mirrors the cap");
    assert((s.compute_queue == s.graphics_queue) == !gpu::get_device_caps(&device)!!.async_compute, "alias iff no async compute");
    uint expected_index = s.compute_queue_distinct && s.compute_family == s.graphics_family ? 1 : 0;
    assert(s.compute_queue_index == expected_index, "stored compute queue index matches selection");
}

fn void cross_queue_timeline_chain_validates() @test {
    gpu::Device device = make_queue_test_device();
    defer destroy_test_device(&device)!!;
    gpu::CommandList compute_cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!)!!;
    defer discard_test_commands(&compute_cmd)!!;
    gpu::ExecutableCommandList compute_cmd_executable = gpu::end_commands(&compute_cmd)!!;
    defer discard_test_executable(&compute_cmd_executable)!!;
    gpu::ExecutableCommandList[1] compute_lists = { compute_cmd_executable };
    gpu::SubmitDesc compute_submit = {
        .command_lists = compute_lists[..],
    };
    gpu::CompletionPoint compute_done = submit_test_queue(&device, gpu::QueueKind.COMPUTE,
        &compute_submit,
    )!!;

    gpu::CommandList gfx_cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&gfx_cmd)!!;
    gpu::ExecutableCommandList gfx_cmd_executable = gpu::end_commands(&gfx_cmd)!!;
    defer discard_test_executable(&gfx_cmd_executable)!!;
    gpu::ExecutableCommandList[1] gfx_lists = { gfx_cmd_executable };
    gpu::CompletionPoint[1] completion_waits = { compute_done };
    gpu::SubmitDesc gfx_submit = {
        .command_lists   = gfx_lists[..],
        .completion_waits = completion_waits[..],
    };
    gpu::CompletionPoint graphics_done = submit_test_queue(&device, gpu::QueueKind.GRAPHICS,
        &gfx_submit,
    )!!;

    gpu::wait_completion(graphics_done, 1_000_000_000)!!;
}

fn void distinct_transfer_compute_graphics_completion_chain() @test {
    gpu::RuntimeDesc runtime_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_distinct_completion_chain_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = 1,
            .compute  = 1,
            .transfer = 1,
        },
        .distinct = { .graphics, .compute, .transfer },
    };
    gpu::DeviceRequest request = gpu::request_queues(
        gpu::strict_device_request(),
        requirements,
    )!!;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_request(&adapter, &request)!!.supported) {
            continue;
        }

        gpu::Device device = gpu::create_device(&adapter, &request)!!;
        defer destroy_test_device(&device)!!;
        gpu::Queue graphics = gpu::get_queue(
            &device,
            gpu::QueueKind.GRAPHICS,
        )!!;
        gpu::Queue compute = gpu::get_queue(
            &device,
            gpu::QueueKind.COMPUTE,
        )!!;
        gpu::Queue transfer = gpu::get_queue(
            &device,
            gpu::QueueKind.TRANSFER,
        )!!;
        assert(graphics.id != compute.id
            && graphics.id != transfer.id
            && compute.id != transfer.id,
            "the integration request owns three distinct queues");

        gpu::CommandList transfer_commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.TRANSFER)!!)!!;
        defer discard_test_commands(&transfer_commands)!!;
        gpu::ExecutableCommandList transfer_commands_executable = gpu::end_commands(&transfer_commands)!!;
        defer discard_test_executable(&transfer_commands_executable)!!;
        gpu::ExecutableCommandList[1] transfer_lists = { transfer_commands_executable };
        gpu::SubmitDesc transfer_submit = {
            .command_lists = transfer_lists[..],
        };
        gpu::CompletionPoint transfer_done = submit_test_queue(&device, gpu::QueueKind.TRANSFER,
            &transfer_submit,
        )!!;

        gpu::CommandList compute_commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!)!!;
        defer discard_test_commands(&compute_commands)!!;
        gpu::ExecutableCommandList compute_commands_executable = gpu::end_commands(&compute_commands)!!;
        defer discard_test_executable(&compute_commands_executable)!!;
        gpu::ExecutableCommandList[1] compute_lists = { compute_commands_executable };
        gpu::CompletionPoint[1] compute_waits = { transfer_done };
        gpu::SubmitDesc compute_submit = {
            .command_lists    = compute_lists[..],
            .completion_waits = compute_waits[..],
        };
        gpu::CompletionPoint compute_done = submit_test_queue(&device, gpu::QueueKind.COMPUTE,
            &compute_submit,
        )!!;

        gpu::CommandList graphics_commands = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
        defer discard_test_commands(&graphics_commands)!!;
        gpu::ExecutableCommandList graphics_commands_executable = gpu::end_commands(&graphics_commands)!!;
        defer discard_test_executable(&graphics_commands_executable)!!;
        gpu::ExecutableCommandList[1] graphics_lists = { graphics_commands_executable };
        gpu::CompletionPoint[1] graphics_waits = { compute_done };
        gpu::SubmitDesc graphics_submit = {
            .command_lists    = graphics_lists[..],
            .completion_waits = graphics_waits[..],
        };
        gpu::CompletionPoint graphics_done = submit_test_queue(&device, gpu::QueueKind.GRAPHICS,
            &graphics_submit,
        )!!;

        gpu::wait_completion(graphics_done, 1_000_000_000)!!;
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
    gpu::GpuAllocation allocation = gpu::allocate_memory(&device, &allocation_desc)!!;
    gpu::free_allocation(&device, &allocation)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle tex = gpu::create_texture(&device, &tex_desc)!!;
    gpu::destroy_texture(&device, tex)!!;
}

fn void strict_heap_families_alias_without_transfer() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute },
        .graphics_family = 4,
        .compute_family  = 4,
        .transfer_family = 7,
    };
    uint[2] families;
    uint count = vk::strict_heap_queue_families(&state, families[..]);
    assert(count == 1, "aliased graphics and compute produce one family");
    assert(families[0] == 4, "graphics family is first");
}

fn void strict_heap_families_are_graphics_compute_only() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute },
        .graphics_family = 4,
        .compute_family  = 5,
        .transfer_family = 7,
    };
    uint[2] families;
    uint count = vk::strict_heap_queue_families(&state, families[..]);
    assert(count == 2, "distinct graphics and compute produce two families");
    assert(families[0] == 4, "graphics family is first");
    assert(families[1] == 5, "compute family is second");
}

fn void selected_queue_families_keep_distinct_transfer() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute, .transfer },
        .graphics_family = 4,
        .compute_family  = 5,
        .transfer_family = 7,
    };
    uint[3] families;
    uint count = vk::used_queue_families(&state, families[..]);
    assert(count == 3, "selected roles keep three distinct families");
    assert(
        families[0] == 4 && families[1] == 5 && families[2] == 7,
        "selected families keep graphics, compute, and transfer order"
    );
}

fn void resource_access_families_include_only_admitted_roles() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute, .transfer },
        .graphics_family = 4,
        .compute_family  = 5,
        .transfer_family = 7,
    };
    uint[3] families;
    uint count = vk::resource_queue_families(
        &state,
        { .graphics, .compute },
        families[..],
    );
    assert(count == 2, "two admitted families are retained");
    assert(families[0] == 4 && families[1] == 5, "unadmitted transfer is excluded");

    state.compute_family = 4;
    count = vk::resource_queue_families(
        &state,
        { .graphics, .compute },
        families[..],
    );
    assert(count == 1 && families[0] == 4, "same-family roles remain exclusive");
}

fn void resource_access_families_include_every_identity_of_one_role() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .compute },
        .compute_family = 5,
    };
    state.queues.counts.compute = 2;
    state.queues.compute_ids[0] = vk::native_queue_id(5, 0);
    state.queues.compute_ids[1] = vk::native_queue_id(7, 0);

    uint[2] families;
    uint count = vk::resource_queue_families(&state, { .compute }, families[..]);
    assert(count == 2, "all admitted compute identities contribute families");
    assert(families[0] == 5 && families[1] == 7, "compute families preserve selected order");
}

fn void resource_texture_create_info_uses_exact_access_families() @test {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };

    uint[1] aliased = { 4 };
    vk::ImageCreateInfo exclusive = vk::make_texture_create_info(&desc, aliased[..]);
    assert(exclusive.sharing_mode == vk::SharingMode.EXCLUSIVE);
    assert(exclusive.queue_family_index_count == 0);

    uint[2] split = { 4, 5 };
    vk::ImageCreateInfo concurrent = vk::make_texture_create_info(&desc, split[..]);
    assert(concurrent.sharing_mode == vk::SharingMode.CONCURRENT);
    assert(concurrent.queue_family_index_count == 2);
    assert(concurrent.p_queue_family_indices[0] == 4);
    assert(concurrent.p_queue_family_indices[1] == 5);
}

fn void pool_selection_follows_queue_family() @test {
    vk::CommandPool[1] gfx  = { (vk::CommandPool)(uptr)0x10 };
    vk::CommandPool[1] comp = { (vk::CommandPool)(uptr)0x30 };
    vk::CommandPool[1] xfer = { (vk::CommandPool)(uptr)0x20 };

    vk::RecordingContextState aliased = {
        .graphics_pools = gfx[..],
        .transfer_pools = xfer[..],
    };
    aliased.compute_pools = aliased.graphics_pools;
    assert(vk::pool_for(&aliased, { .roles = { .compute } })
        == gfx[0], "aliased compute uses graphics pool");
    assert(vk::pool_for(&aliased, { .roles = { .transfer } })
        == xfer[0], "transfer uses transfer pool");

    vk::RecordingContextState distinct = {
        .graphics_pools = gfx[..],
        .compute_pools  = comp[..],
        .transfer_pools = xfer[..],
        .compute_pools_distinct = true,
    };
    assert(vk::pool_for(&distinct, { .roles = { .compute } })
        == comp[0], "distinct compute uses compute pool");
    assert(vk::pool_for(&distinct, { .roles = { .graphics } })
        == gfx[0], "graphics remains independent");
}

fn void tri_queue_completions_wait_cleanly() @test {
    gpu::Device device = make_queue_test_device();
    defer destroy_test_device(&device)!!;

    for (uint round = 0; round < 4; round++) {
        gpu::QueueKind[3] kinds = {
            gpu::QueueKind.GRAPHICS,
            gpu::QueueKind.COMPUTE,
            gpu::QueueKind.TRANSFER,
        };
        gpu::CompletionPoint[3] completions;
        for (uint i = 0; i < kinds.len; i++) {
            gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, kinds[i])!!)!!;
            defer discard_test_commands(&cmd)!!;
            gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
            defer discard_test_executable(&cmd_executable)!!;
            gpu::ExecutableCommandList[1] one = { cmd_executable };
            gpu::SubmitDesc submit = { .command_lists = one[..] };
            completions[i] = submit_test_queue(&device, kinds[i], &submit)!!;
        }
        for (uint i = 0; i < completions.len; i++) {
            gpu::wait_completion(completions[i])!!;
        }
    }
}

fn void swapchain_sharing_aliases_the_main_family() @test {
    vk::Queue queue = (vk::Queue)(uptr)0x10;
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics },
        .graphics_family = 4,
        .graphics_queue  = queue,
        .presentation    = {
            .enabled = true,
            .family  = 4,
            .queue   = queue,
        },
    };
    uint[2] families;
    uint family_count = vk::swapchain_queue_families(&state, families[..]);
    assert(family_count == 1, "aliased presentation uses exclusive sharing");
    assert(families[0] == 4, "the graphics family is retained");

    vk::Queue[2] queues;
    uint queue_count = vk::swapchain_idle_queues(&state, queues[..]);
    assert(queue_count == 1, "an aliased queue is waited once");
    assert(queues[0] == queue, "the graphics queue is waited");
}

fn void swapchain_sharing_includes_every_selected_graphics_family() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics },
        .graphics_family      = 4,
        .presentation         = {
            .enabled = true,
            .family  = 7,
        },
    };
    state.queues.counts.graphics = 2;
    state.queues.graphics_ids[0] = vk::native_queue_id(4, 0);
    state.queues.graphics_ids[1] = vk::native_queue_id(6, 0);

    uint[4] families;
    uint family_count = vk::swapchain_queue_families(&state, families[..]);
    assert(family_count == 3,
        "swapchain sharing includes every selected graphics family");
    assert(families[0] == 4 && families[1] == 6 && families[2] == 7,
        "graphics families precede the private presentation family");
}

fn void swapchain_sharing_includes_a_separate_present_family() @test {
    vk::Queue graphics_queue = (vk::Queue)(uptr)0x10;
    vk::Queue present_queue = (vk::Queue)(uptr)0x20;
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics },
        .graphics_family = 4,
        .graphics_queue  = graphics_queue,
        .presentation    = {
            .enabled = true,
            .family  = 7,
            .queue   = present_queue,
        },
    };
    uint[2] families;
    uint family_count = vk::swapchain_queue_families(&state, families[..]);
    assert(family_count == 2, "split presentation uses concurrent sharing");
    assert(
        families[0] == 4 && families[1] == 7,
        "sharing keeps graphics then presentation"
    );

    vk::Queue[2] queues;
    uint queue_count = vk::swapchain_idle_queues(&state, queues[..]);
    assert(queue_count == 2, "split queues are both waited");
    assert(
        queues[0] == graphics_queue && queues[1] == present_queue,
        "idle waits preserve graphics then presentation order"
    );
}

fn void same_family_presentation_keeps_its_exact_queue() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_T, 4),
    };
    bool[1] presentation_support = { true };
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 2, .transfer = 2 },
        .distinct = { .transfer },
    };
    vk::QueueFamilies picked =
        vk::pick_queue_families_for_surface_requirements(
            table[..],
            presentation_support[..],
            requirements,
        )!!;
    assert(picked.graphics == picked.present);
    assert(picked.graphics_index != picked.present_index);

    vk::Queue graphics_queue = (vk::Queue)(uptr)0x20;
    vk::Queue transfer_queue = (vk::Queue)(uptr)0x30;
    vk::Queue present_queue = (vk::Queue)(uptr)0x10;
    vk::QueueCompletionState[1] completions = {{
        .queue    = graphics_queue,
        .timeline = (vk::Semaphore)(uptr)0x40,
        .roles    = { .graphics },
        .bound    = true,
    }};
    vk::VkDeviceState state = {
        .selected_queue_roles  = picked.selected,
        .graphics_family       = picked.graphics,
        .graphics_queue        = graphics_queue,
        .transfer_family       = picked.transfer,
        .transfer_queue        = transfer_queue,
        .queue_completions     = completions[..],
        .graphics_completion_id = 0,
        .presentation          = {
            .enabled = true,
            .family  = picked.present,
            .queue   = present_queue,
        },
    };
    vk::Queue[2] queues;
    uint queue_count = vk::swapchain_idle_queues(&state, queues[..]);
    assert(queue_count == 2);
    assert(queues[0] == graphics_queue && queues[1] == present_queue);
    assert(
        vk::present_queue_mutex_for(&state)
            != vk::queue_mutex_for(&state, gpu::QueueKind.GRAPHICS)
    );
    assert(
        vk::native_queue_mutex_for(&state, present_queue)
            == vk::present_queue_mutex_for(&state)
    );
}

fn void distinct_compute_requirement_uses_a_second_native_queue() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 2),
    };
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 1 },
        .distinct = { .compute },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(picked.graphics == picked.compute, "both roles use the capable family");
    assert(picked.compute_index == 1, "compute uses a distinct queue index");
}

fn void distinct_graphics_requirement_uses_a_second_native_queue() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 2),
    };
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 1, .transfer = 1 },
        .distinct = { .graphics },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(picked.graphics_index == 0, "graphics keeps the first queue");
    assert(
        picked.compute_index == 1 && picked.transfer_index == 1,
        "compute and transfer share the other queue"
    );
}

fn void distinct_transfer_requirement_uses_a_second_native_queue() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 2),
    };
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 1, .transfer = 1 },
        .distinct = { .transfer },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(
        picked.graphics_index == 0 && picked.compute_index == 0,
        "graphics and compute share the first queue"
    );
    assert(picked.transfer_index == 1, "transfer uses the other queue");
}

fn void distinct_roles_use_split_capability_families() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G, 1),
        family(QF_C, 1),
        family(QF_T, 1),
    };
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = 1,
            .compute  = 1,
            .transfer = 1,
        },
        .distinct = { .graphics, .compute, .transfer },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(picked.graphics == 0 && picked.graphics_index == 0);
    assert(picked.compute == 1 && picked.compute_index == 0);
    assert(picked.transfer == 2 && picked.transfer_index == 0);
}

fn void presentation_requirements_consider_a_later_main_family() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G | QF_C | QF_T, 1),
        family(QF_G | QF_C | QF_T, 2),
        family(QF_T, 1),
    };
    bool[3] present_support = { true, false, false };
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 1 },
        .distinct = { .compute },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_surface_requirements(
            table[..],
            present_support[..],
            requirements,
        )!!;
    assert(picked.graphics == 1, "the later family supplies two semantic queues");
    assert(picked.compute == 1 && picked.compute_index == 1);
    assert(picked.present == 0, "presentation stays on the supported family");
}

fn void supported_selection_requests_every_selected_queue_index() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 2),
    };
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 1, .transfer = 1 },
        .distinct = { .transfer },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(
        vk::requested_queue_count(&picked, picked.graphics, false) == 2,
        "device creation requests through the highest selected index"
    );
}

fn void graphics_only_selection_has_no_hidden_roles() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G, 1),
        family(QF_C, 1),
        family(QF_T, 1),
    };
    gpu::QueueRequirements requirements = {
        .counts = { .graphics = 1 },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(picked.selected.graphics);
    assert(!picked.selected.compute && !picked.selected.transfer);

    uint[4] device_families;
    uint device_count = vk::selected_device_queue_families(
        &picked,
        false,
        device_families[..],
    );
    assert(device_count == 1 && device_families[0] == 0);
    assert(vk::requested_queue_count(&picked, 1, false) == 0);
    assert(vk::requested_queue_count(&picked, 2, false) == 0);

    vk::VkDeviceState state = {
        .selected_queue_roles = picked.selected,
        .graphics_family      = picked.graphics,
        .compute_family       = picked.compute,
        .transfer_family      = picked.transfer,
    };
    uint[3] shared_families;
    uint shared_count = vk::used_queue_families(&state, shared_families[..]);
    assert(shared_count == 1 && shared_families[0] == 0);
}

fn void compute_only_selection_has_no_hidden_roles() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G, 1),
        family(QF_C, 1),
        family(QF_T, 1),
    };
    gpu::QueueRequirements requirements = {
        .counts = { .compute = 1 },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(!picked.selected.graphics && picked.selected.compute);
    assert(!picked.selected.transfer && !picked.async_compute);

    uint[4] device_families;
    uint device_count = vk::selected_device_queue_families(
        &picked,
        false,
        device_families[..],
    );
    assert(device_count == 1 && device_families[0] == 1);
    assert(vk::requested_queue_count(&picked, 0, false) == 0);
    assert(vk::requested_queue_count(&picked, 2, false) == 0);

    vk::VkDeviceState state = {
        .selected_queue_roles = picked.selected,
        .graphics_family      = picked.graphics,
        .compute_family       = picked.compute,
        .transfer_family      = picked.transfer,
    };
    uint[3] shared_families;
    uint shared_count = vk::used_queue_families(&state, shared_families[..]);
    assert(shared_count == 1 && shared_families[0] == 1);
}

fn void transfer_only_selection_has_no_hidden_roles() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G, 1),
        family(QF_C, 1),
        family(QF_T, 1),
    };
    gpu::QueueRequirements requirements = {
        .counts = { .transfer = 1 },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(!picked.selected.graphics && !picked.selected.compute);
    assert(picked.selected.transfer && !picked.async_compute);

    uint[4] device_families;
    uint device_count = vk::selected_device_queue_families(
        &picked,
        false,
        device_families[..],
    );
    assert(device_count == 1 && device_families[0] == 2);
    assert(vk::requested_queue_count(&picked, 0, false) == 0);
    assert(vk::requested_queue_count(&picked, 1, false) == 0);

    vk::VkDeviceState state = {
        .selected_queue_roles = picked.selected,
        .graphics_family      = picked.graphics,
        .compute_family       = picked.compute,
        .transfer_family      = picked.transfer,
    };
    uint[3] shared_families;
    uint shared_count = vk::used_queue_families(&state, shared_families[..]);
    assert(shared_count == 1 && shared_families[0] == 2);
}

fn void presentation_without_graphics_is_rejected() @test {
    vk::QueueFamilyProperties[4] table = {
        family(QF_G, 1),
        family(QF_C, 1),
        family(QF_T, 1),
        family(0, 1),
    };
    bool[4] present_support = { false, false, false, true };
    gpu::QueueRequirements requirements = {
        .counts = { .compute = 1 },
    };

    assert(
        @catch(vk::pick_queue_families_for_surface_requirements(
            table[..],
            present_support[..],
            requirements,
        )) == gpu::INVALID_ARGUMENT,
        "presentation requires a public graphics queue"
    );
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
        .selected_queue_roles   = { .graphics, .compute, .transfer },
        .graphics_queue          = graphics_queue,
        .compute_queue           = compute_queue,
        .transfer_queue          = compute_queue,
        .queue_completions       = completions[..],
        .graphics_completion_id  = 0,
        .compute_completion_id   = 1,
        .transfer_completion_id  = 1,
        .compute_queue_distinct  = true,
    };

    assert(
        vk::queue_mutex_for(&state, gpu::QueueKind.COMPUTE)
            == vk::queue_mutex_for(&state, gpu::QueueKind.TRANSFER),
        "aliased semantic roles serialize one native queue"
    );
}

fn void unavailable_distinct_queue_requirement_faults() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 1),
    };
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 1 },
        .distinct = { .compute },
    };

    assert(
        @catch(vk::pick_queue_families_for_requirements(table[..], requirements))
            == gpu::UNSUPPORTED_FEATURE,
        "an unavailable distinct queue is rejected"
    );
}

fn void multiple_queues_per_role_share_available_native_queues() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 2),
    };
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = 2,
            .compute  = 2,
            .transfer = 2,
        },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(
        vk::requested_queue_count(&picked, picked.graphics, false) == 2,
        "two native queues satisfy all three semantic roles"
    );
}

fn void compute_only_multi_count_keeps_exact_families() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_C, 1),
        family(QF_C, 1),
    };
    gpu::QueueRequirements requirements = {
        .counts = { .compute = 2 },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(!picked.selected.graphics && picked.selected.compute);
    assert(!picked.selected.transfer && picked.counts.compute == 2);

    uint[2] device_families;
    uint device_count = vk::selected_device_queue_families(
        &picked,
        false,
        device_families[..],
    );
    assert(device_count == 2);
    assert(device_families[0] == 0 && device_families[1] == 1);

    vk::VkDeviceState state = {
        .selected_queue_roles = picked.selected,
        .compute_family       = picked.compute,
        .queues               = picked,
    };
    uint[2] shared_families;
    uint shared_count = vk::used_queue_families(&state, shared_families[..]);
    assert(shared_count == 2);
    assert(shared_families[0] == 0 && shared_families[1] == 1);
}

fn void private_presentation_preserves_graphics_multi_count() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G, 1),
        family(QF_G, 1),
        family(0, 1),
    };
    bool[3] presentation_support = { false, false, true };
    gpu::QueueRequirements requirements = {
        .counts = { .graphics = 2 },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_surface_requirements(
            table[..],
            presentation_support[..],
            requirements,
        )!!;
    assert(picked.selected.graphics && !picked.selected.compute);
    assert(picked.presentation && picked.present == 2);

    uint[3] device_families;
    uint device_count = vk::selected_device_queue_families(
        &picked,
        true,
        device_families[..],
    );
    assert(device_count == 3);
    assert(
        device_families[0] == 0
            && device_families[1] == 1
            && device_families[2] == 2
    );
}

fn void unavailable_queue_count_faults() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 1),
    };
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = 2,
            .compute  = 1,
            .transfer = 1,
        },
    };

    assert(
        @catch(vk::pick_queue_families_for_requirements(table[..], requirements))
            == gpu::UNSUPPORTED_FEATURE
    );
}

fn void distinct_role_counts_request_disjoint_native_queues() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 3),
    };
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = 2,
            .compute  = 1,
        },
        .distinct = { .compute },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(
        vk::requested_queue_count(&picked, picked.graphics, false) == 3,
        "distinct compute and two graphics queues require three native queues"
    );
}

fn void multiple_distinct_roles_preserve_constrained_capacity() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_G | QF_C | QF_T, 3),
        family(QF_G, 1),
    };
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = 2,
            .compute  = 2,
        },
        .distinct = { .graphics, .compute },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(picked.counts.graphics == 2 && picked.counts.compute == 2);
    for (uint graphics = 0; graphics < picked.counts.graphics; graphics++) {
        for (uint compute = 0; compute < picked.counts.compute; compute++) {
            assert(
                vk::selected_native_queue(&picked, gpu::QueueKind.GRAPHICS, graphics)
                    != vk::selected_native_queue(&picked, gpu::QueueKind.COMPUTE, compute),
                "distinct role identities do not overlap"
            );
        }
    }
}

fn void one_distinct_role_preserves_other_role_capacity() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_G | QF_C | QF_T, 3),
        family(QF_G, 1),
    };
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = 2,
            .compute  = 2,
        },
        .distinct = { .graphics },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(picked.counts.graphics == 2 && picked.counts.compute == 2);
    for (uint graphics = 0; graphics < picked.counts.graphics; graphics++) {
        for (uint compute = 0; compute < picked.counts.compute; compute++) {
            assert(
                vk::selected_native_queue(&picked, gpu::QueueKind.GRAPHICS, graphics)
                    != vk::selected_native_queue(&picked, gpu::QueueKind.COMPUTE, compute),
                "the distinct graphics role leaves both compute queues available"
            );
        }
    }
}

fn void device_queue_families_include_multi_count_selections() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_G | QF_C | QF_T, 1),
        family(QF_G, 1),
    };
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = 2,
            .compute  = 1,
            .transfer = 1,
        },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    uint[4] families;
    uint count = vk::selected_device_queue_families(
        &picked,
        false,
        families[..],
    );
    assert(count == 2, "logical device creation includes both graphics families");
    assert(families[0] == 0 && families[1] == 1);
}

fn void maximum_queue_counts_share_full_native_range() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, gpu::MAX_REQUESTED_QUEUES_PER_ROLE),
    };
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = gpu::MAX_REQUESTED_QUEUES_PER_ROLE,
            .compute  = gpu::MAX_REQUESTED_QUEUES_PER_ROLE,
            .transfer = gpu::MAX_REQUESTED_QUEUES_PER_ROLE,
        },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(picked.counts.graphics == gpu::MAX_REQUESTED_QUEUES_PER_ROLE);
    assert(picked.counts.compute == gpu::MAX_REQUESTED_QUEUES_PER_ROLE);
    assert(picked.counts.transfer == gpu::MAX_REQUESTED_QUEUES_PER_ROLE);
    assert(
        vk::requested_queue_count(&picked, 0, false)
            == gpu::MAX_REQUESTED_QUEUES_PER_ROLE
    );
    assert(
        vk::selected_native_queue(&picked, gpu::QueueKind.GRAPHICS, 254)
            == vk::selected_native_queue(&picked, gpu::QueueKind.COMPUTE, 254)
    );
    assert(
        vk::selected_native_queue(&picked, gpu::QueueKind.COMPUTE, 254)
            == vk::selected_native_queue(&picked, gpu::QueueKind.TRANSFER, 254)
    );
}

fn void maximum_distinct_queue_counts_use_disjoint_native_ranges() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, gpu::MAX_SELECTED_QUEUE_IDENTITIES),
    };
    gpu::QueueRequirements requirements = {
        .counts = {
            .graphics = gpu::MAX_REQUESTED_QUEUES_PER_ROLE,
            .compute  = gpu::MAX_REQUESTED_QUEUES_PER_ROLE,
            .transfer = gpu::MAX_REQUESTED_QUEUES_PER_ROLE,
        },
        .distinct = {
            .graphics,
            .compute,
            .transfer,
        },
    };

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_requirements(table[..], requirements)!!;
    assert(picked.counts.graphics == gpu::MAX_REQUESTED_QUEUES_PER_ROLE);
    assert(picked.counts.compute == gpu::MAX_REQUESTED_QUEUES_PER_ROLE);
    assert(picked.counts.transfer == gpu::MAX_REQUESTED_QUEUES_PER_ROLE);
    assert(vk::requested_queue_count(&picked, 0, false)
        == gpu::MAX_SELECTED_QUEUE_IDENTITIES);
}
