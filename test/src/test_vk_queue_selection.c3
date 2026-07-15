module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;

// Pure selection-tier units over synthetic family tables (no device), plus
// lavapipe integration: alias tier, caps flag, cross-queue timeline chain,
// and the shared_queues no-op on a single-family device.

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
        .frames_in_flight  = 2,
        .application_name  = "gpu_queue_selection_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
}

fn void queue_alias_matches_async_compute_cap() @test {
    gpu::Device device = make_queue_test_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* s = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(s.compute_queue_distinct == gpu::get_device_caps(&device)!!.async_compute, "distinct flag mirrors the cap");
    assert((s.compute_queue == s.graphics_queue) == !gpu::get_device_caps(&device)!!.async_compute, "alias iff no async compute");
    uint expected_index = s.compute_queue_distinct && s.compute_family == s.graphics_family ? 1 : 0;
    assert(s.compute_queue_index == expected_index, "stored compute queue index matches selection");
}

fn void cross_queue_timeline_chain_validates() @test {
    gpu::Device device = make_queue_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::SemaphoreDesc sem_desc = { .debug_name = "queue_chain" };
    gpu::SemaphoreHandle sem = gpu::create_semaphore(&device, &sem_desc)!!;
    defer gpu::destroy_semaphore(&device, sem)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList compute_cmd = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::end_commands(&compute_cmd)!!;
    gpu::CommandList[1] compute_lists = { compute_cmd };
    gpu::SemaphoreSignal[1] signals = {{ .semaphore = sem, .value = (gpu::SemaphoreValue)1, .stage = gpu::Stage.COMPUTE_SHADER }};
    gpu::SubmitDesc compute_submit = { .command_lists = compute_lists[..], .signals = signals[..] };
    gpu::submit(&device, &compute_submit)!!;

    gpu::CommandList gfx_cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&gfx_cmd)!!;
    gpu::CommandList[1] gfx_lists = { gfx_cmd };
    gpu::SemaphoreWait[1] waits = {{ .semaphore = sem, .value = (gpu::SemaphoreValue)1, .stage = gpu::Stage.COMPUTE_SHADER }};
    gpu::SubmitDesc gfx_submit = { .command_lists = gfx_lists[..], .waits = waits[..] };
    gpu::submit(&device, &gfx_submit)!!;

    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::end_frame(&frame_token)!!;
}

fn void shared_queues_noop_on_single_family() @test {
    gpu::Device device = make_queue_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc buf_desc = {
        .size        = 256,
        .usage       = { .storage, .addressable, .shared_queues },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .debug_name  = "shared_buf",
    };
    gpu::BufferHandle buf = gpu::create_buffer(&device, &buf_desc)!!;
    gpu::destroy_buffer(&device, buf)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst, .shared_queues },
    };
    gpu::TextureHandle tex = gpu::create_texture(&device, &tex_desc)!!;
    gpu::destroy_texture(&device, tex)!!;
}

fn void shared_queues_alone_is_invalid() @test {
    gpu::Device device = make_queue_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc buf_desc = {
        .size        = 64,
        .usage       = { .shared_queues },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
    };
    fault f = @catch(gpu::create_buffer(&device, &buf_desc));
    assert(f == gpu::INVALID_ARGUMENT, "usage with no Vulkan bits faults INVALID_ARGUMENT");
}

fn void frame_arena_families_alias_without_transfer() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute },
        .graphics_family = 4,
        .compute_family  = 4,
        .transfer_family = 7,
    };
    uint[2] families;
    uint count = vk::graphics_compute_queue_families(&state, families[..]);
    assert(count == 1, "aliased graphics and compute produce one family");
    assert(families[0] == 4, "graphics family is first");
}

fn void frame_arena_families_are_graphics_compute_only() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute },
        .graphics_family = 4,
        .compute_family  = 5,
        .transfer_family = 7,
    };
    uint[2] families;
    uint count = vk::graphics_compute_queue_families(&state, families[..]);
    assert(count == 2, "distinct graphics and compute produce two families");
    assert(families[0] == 4, "graphics family is first");
    assert(families[1] == 5, "compute family is second");
}

fn void frame_arena_create_info_is_exclusive_when_families_alias() @test {
    gpu::BufferDesc desc = {
        .size        = 256,
        .usage       = { .storage, .addressable, .indirect, .index },
        .memory_kind = gpu::MemoryKind.FRAME_UPLOAD,
    };
    uint[1] families = { 4 };
    vk::BufferCreateInfo info = vk::make_buffer_create_info(&desc, families[..])!!;
    assert(info.sharing_mode == vk::SharingMode.EXCLUSIVE, "one family keeps exclusive sharing");
    assert(info.queue_family_index_count == 0, "exclusive creation publishes no family count");
    assert(info.p_queue_family_indices == null, "exclusive creation publishes no family pointer");
}

fn void frame_arena_create_info_uses_exact_distinct_families() @test {
    gpu::BufferDesc desc = {
        .size        = 256,
        .usage       = { .storage, .addressable, .indirect, .index },
        .memory_kind = gpu::MemoryKind.FRAME_UPLOAD,
    };
    uint[2] families = { 4, 5 };
    vk::BufferCreateInfo info = vk::make_buffer_create_info(&desc, families[..])!!;
    assert(info.sharing_mode == vk::SharingMode.CONCURRENT, "two families use concurrent sharing");
    assert(info.queue_family_index_count == 2, "concurrent creation publishes two families");
    assert(info.p_queue_family_indices[0] == 4, "concurrent creation keeps graphics first");
    assert(info.p_queue_family_indices[1] == 5, "concurrent creation keeps compute second");
}

fn void public_shared_families_keep_distinct_transfer() @test {
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute, .transfer },
        .graphics_family = 4,
        .compute_family  = 5,
        .transfer_family = 7,
    };
    uint[3] families;
    uint count = vk::used_queue_families(&state, families[..]);
    assert(count == 3, "public sharing keeps three distinct families");
    assert(
        families[0] == 4 && families[1] == 5 && families[2] == 7,
        "public sharing keeps graphics, compute, and transfer order"
    );
}

fn void pool_selection_follows_queue_family() @test {
    vk::CommandPool[2] gfx  = { (vk::CommandPool)(uptr)0x10, (vk::CommandPool)(uptr)0x11 };
    vk::CommandPool[2] comp = { (vk::CommandPool)(uptr)0x30, (vk::CommandPool)(uptr)0x31 };
    vk::CommandPool[2] xfer = { (vk::CommandPool)(uptr)0x20, (vk::CommandPool)(uptr)0x21 };

    vk::RecordingContextState aliased = {
        .graphics_pools = gfx[..],
        .transfer_pools = xfer[..],
    };
    aliased.compute_pools = aliased.graphics_pools;
    assert(vk::pool_for(&aliased, gpu::QueueKind.COMPUTE, 0) == gfx[0], "aliased compute uses graphics pool");
    assert(vk::pool_for(&aliased, gpu::QueueKind.TRANSFER, 1) == xfer[1], "transfer uses transfer pool");

    vk::RecordingContextState distinct = {
        .graphics_pools = gfx[..],
        .compute_pools  = comp[..],
        .transfer_pools = xfer[..],
        .compute_pools_distinct = true,
    };
    assert(vk::pool_for(&distinct, gpu::QueueKind.COMPUTE, 0) == comp[0], "distinct compute uses compute pool");
    assert(vk::pool_for(&distinct, gpu::QueueKind.GRAPHICS, 0) == gfx[0], "graphics unaffected");
}

fn void tri_queue_frame_retires_cleanly() @test {
    gpu::Device device = make_queue_test_device();
    defer gpu::destroy_device(&device)!!;

    for (uint frame = 0; frame < 4; frame++) {
        gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
        gpu::CommandList[3] lists;
        gpu::QueueKind[3] kinds = { gpu::QueueKind.GRAPHICS, gpu::QueueKind.COMPUTE, gpu::QueueKind.TRANSFER };
        for (uint i = 0; i < 3; i++) {
            gpu::CommandList cmd = gpu::begin_commands(&device, kinds[i])!!;
            gpu::end_commands(&cmd)!!;
            gpu::CommandList[1] one = { cmd };
            gpu::SubmitDesc submit = { .command_lists = one[..] };
            gpu::submit(&device, &submit)!!;
        }
        gpu::end_frame(&frame_token)!!;
    }
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
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

fn void presentation_without_graphics_adds_only_a_private_queue() @test {
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

    vk::QueueFamilies picked =
        vk::pick_queue_families_for_surface_requirements(
            table[..],
            present_support[..],
            requirements,
        )!!;
    assert(!picked.selected.graphics && picked.selected.compute);
    assert(!picked.selected.transfer && picked.present == 3);

    uint[4] device_families;
    uint device_count = vk::selected_device_queue_families(
        &picked,
        true,
        device_families[..],
    );
    assert(device_count == 2);
    assert(device_families[0] == 1 && device_families[1] == 3);
    assert(vk::requested_queue_count(&picked, 0, true) == 0);
    assert(vk::requested_queue_count(&picked, 2, true) == 0);

    vk::Queue compute_queue = (vk::Queue)(uptr)0x20;
    vk::Queue present_queue = (vk::Queue)(uptr)0x30;
    vk::VkDeviceState state = {
        .selected_queue_roles = picked.selected,
        .compute_family       = picked.compute,
        .compute_queue        = compute_queue,
        .presentation         = {
            .family = picked.present,
            .queue  = present_queue,
        },
    };
    uint[3] shared_families;
    uint shared_count = vk::used_queue_families(&state, shared_families[..]);
    assert(shared_count == 1 && shared_families[0] == 1);

    vk::Queue[2] idle_queues;
    uint idle_count = vk::swapchain_idle_queues(&state, idle_queues[..]);
    assert(idle_count == 1 && idle_queues[0] == present_queue);
    assert(
        vk::native_queue_mutex_for(&state, idle_queues[0])
            == vk::present_queue_mutex_for(&state),
        "presentation-only teardown uses the presentation mutex"
    );
}

fn void frame_anchor_uses_first_admitted_role() @test {
    vk::Queue graphics_queue = (vk::Queue)(uptr)0x10;
    vk::Queue compute_queue = (vk::Queue)(uptr)0x20;
    vk::Queue transfer_queue = (vk::Queue)(uptr)0x30;
    vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute, .transfer },
        .graphics_queue       = graphics_queue,
        .compute_queue        = compute_queue,
        .transfer_queue       = transfer_queue,
    };

    assert(vk::frame_queue_kind(&state) == gpu::QueueKind.GRAPHICS);
    assert(!vk::queue_uses_frame_aux(&state, gpu::QueueKind.GRAPHICS));
    assert(vk::queue_uses_frame_aux(&state, gpu::QueueKind.COMPUTE));
    assert(vk::queue_uses_frame_aux(&state, gpu::QueueKind.TRANSFER));

    state = {
        .selected_queue_roles = { .compute, .transfer },
        .compute_queue        = compute_queue,
        .transfer_queue       = transfer_queue,
    };
    assert(vk::frame_queue_kind(&state) == gpu::QueueKind.COMPUTE);
    assert(!vk::queue_uses_frame_aux(&state, gpu::QueueKind.COMPUTE));
    assert(vk::queue_uses_frame_aux(&state, gpu::QueueKind.TRANSFER));

    gpu::Device device;
    gpu::SubmitDesc desc;
    vk::SubmitInputs inputs = vk::resolve_submit_inputs(&state, &device, &desc)!!;
    assert(inputs.kind == gpu::QueueKind.COMPUTE);

    state.transfer_queue = compute_queue;
    assert(
        !vk::queue_uses_frame_aux(&state, gpu::QueueKind.TRANSFER),
        "an aliased transfer role is ordered on the compute anchor"
    );

    state = {
        .selected_queue_roles = { .transfer },
        .transfer_queue       = transfer_queue,
    };
    assert(vk::frame_queue_kind(&state) == gpu::QueueKind.TRANSFER);
    assert(!vk::queue_uses_frame_aux(&state, gpu::QueueKind.TRANSFER));
}

fn void aliased_compute_and_transfer_share_one_queue_mutex() @test {
    vk::Queue graphics_queue = (vk::Queue)(uptr)0x10;
    vk::Queue compute_queue = (vk::Queue)(uptr)0x20;
    vk::VkDeviceState state = {
        .selected_queue_roles  = { .graphics, .compute, .transfer },
        .graphics_queue         = graphics_queue,
        .compute_queue          = compute_queue,
        .transfer_queue         = compute_queue,
        .compute_queue_distinct = true,
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
