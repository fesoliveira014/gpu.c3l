module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;

const uint QF_G = (uint)vk::QueueFlagBits.GRAPHICS_BIT;
const uint QF_C = (uint)vk::QueueFlagBits.COMPUTE_BIT;
const uint QF_T = (uint)vk::QueueFlagBits.TRANSFER_BIT;
const uint QF_S = (uint)vk::QueueFlagBits.SPARSE_BINDING_BIT;

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

fn void default_selection_prefers_dedicated_compute_and_transfer() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G | QF_C | QF_T, 16),
        family(QF_C | QF_T, 8),
        family(QF_T, 2),
    };

    vk::QueueFamilies picked = vk::pick_queue_families_for_request(
        table[..],
        gpu::internal::default_queue_request(),
    )!!;

    assert(picked.graphics == 0 && picked.graphics_index == 0,
        "graphics uses the first universal queue");
    assert(picked.compute == 0 && picked.compute_index == 1,
        "compute prefers a second queue in the graphics family");
    assert(picked.transfer == 2 && picked.transfer_index == 0,
        "transfer prefers the dedicated transfer family");
    assert(picked.async_compute,
        "the default selection preserves asynchronous compute");
}

fn void no_graphics_family_faults_for_default_request() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_C | QF_T, 4),
    };

    fault result = @catch(vk::pick_queue_families_for_request(
        table[..],
        gpu::internal::default_queue_request(),
    ));

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
    uint[3] families;
    assert(vk::selected_device_queue_families(
            &picked,
            false,
            families[..],
        ) == 1
            && families[0] == 0,
        "graphics-only selection creates no hidden queue family");
    assert(vk::requested_queue_count(&picked, 1, false) == 0
            && vk::requested_queue_count(&picked, 2, false) == 0,
        "graphics-only selection requests no hidden queues");
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
    uint[3] families;
    assert(vk::selected_device_queue_families(
            &picked,
            false,
            families[..],
        ) == 1
            && families[0] == 1,
        "compute-only selection creates no hidden queue family");
    assert(vk::requested_queue_count(&picked, 0, false) == 0
            && vk::requested_queue_count(&picked, 2, false) == 0,
        "compute-only selection requests no hidden queues");
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
    assert(picked.transfer == 2,
        "a transfer-only request prefers the dedicated family");
    uint[3] families;
    assert(vk::selected_device_queue_families(
            &picked,
            false,
            families[..],
        ) == 1
            && families[0] == 2,
        "transfer-only selection creates no hidden queue family");
    assert(vk::requested_queue_count(&picked, 0, false) == 0
            && vk::requested_queue_count(&picked, 1, false) == 0,
        "transfer-only selection requests no hidden queues");
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
    assert(picked.graphics_index == 0 && picked.compute_index == 1,
        "compute retains the backend's asynchronous preference");
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

fn void sparse_selection_rejects_unselected_sparse_only_family() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_G | QF_C | QF_T, 2),
        family(QF_S, 1),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute, .transfer },
    };

    assert(@catch(vk::pick_queue_families_for_request(
        table[..],
        request,
        true,
    )) == gpu::UNSUPPORTED_FEATURE);
}

fn void sparse_selection_uses_best_eligible_topology() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_G | QF_C | QF_T, 2),
        family(QF_G | QF_C | QF_T | QF_S, 2),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute, .transfer },
    };

    vk::QueueFamilies baseline =
        vk::pick_queue_families_for_request(table[..], request)!!;
    vk::QueueFamilies sparse = vk::pick_queue_families_for_request(
        table[..],
        request,
        true,
    )!!;

    assert((uint)baseline.sparse_binding_queues == 0,
        "the unconstrained preferred topology contains no sparse queue");
    assert((uint)sparse.sparse_binding_queues != 0,
        "the sparse filter rejects the unconstrained candidate");
    assert(sparse.graphics != baseline.graphics
            || sparse.compute != baseline.compute
            || sparse.transfer != baseline.transfer,
        "sparse eligibility changes only the topology needed to gain a sparse queue");
}

fn void sparse_selection_reports_capable_aliased_roles() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T | QF_S, 2),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute, .transfer },
    };

    vk::QueueFamilies picked = vk::pick_queue_families_for_request(
        table[..],
        request,
        true,
    )!!;

    assert(picked.sparse_binding_queues.graphics);
    assert(picked.sparse_binding_queues.compute);
    assert(picked.sparse_binding_queues.transfer);
}

fn void sparse_selection_reports_only_capable_selected_roles() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G | QF_S, 1),
        family(QF_C, 1),
        family(QF_T, 1),
    };
    gpu::QueueRequest request = {
        .required = { .graphics, .compute, .transfer },
        .distinct = { .graphics, .compute, .transfer },
    };

    vk::QueueFamilies picked = vk::pick_queue_families_for_request(
        table[..],
        request,
        true,
    )!!;

    assert(picked.sparse_binding_queues.graphics);
    assert(!picked.sparse_binding_queues.compute);
    assert(!picked.sparse_binding_queues.transfer);
}

fn void distinct_roles_must_also_be_required() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 2),
    };
    gpu::QueueRequest request = {
        .distinct = { .graphics },
    };

    fault result =
        @catch(vk::pick_queue_families_for_request(table[..], request));

    assert(result == gpu::INVALID_ARGUMENT,
        "a distinct role outside the required set is invalid");
}

fn void presentation_selects_a_surface_capable_graphics_queue() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G | QF_C | QF_T, 1),
        family(QF_G | QF_C | QF_T, 2),
        family(QF_T, 1),
    };
    bool[3] presentation_support = { false, true, false };

    vk::QueueFamilies picked = vk::pick_queue_families_for_surface_request(
        table[..],
        presentation_support[..],
        gpu::internal::default_queue_request(),
    )!!;

    assert(picked.graphics == 1 && picked.graphics_index == 0);
    assert(picked.compute == 1 && picked.compute_index == 1,
        "compute uses the second queue in the presenting family");
    assert(picked.transfer == 2 && picked.transfer_index == 0,
        "transfer keeps the dedicated family");
    assert(picked.present == 1 && picked.present_index == 0);
}

fn void presentation_can_use_a_selected_non_graphics_queue() @test {
    vk::QueueFamilyProperties[2] table = {
        family(QF_G | QF_C | QF_T, 1),
        family(QF_C | QF_T, 2),
    };
    bool[2] presentation_support = { false, true };

    vk::QueueFamilies picked = vk::pick_queue_families_for_surface_request(
        table[..],
        presentation_support[..],
        gpu::internal::default_queue_request(),
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

    vk::QueueFamilies picked = vk::pick_queue_families_for_surface_request(
        table[..],
        presentation_support[..],
        gpu::internal::default_queue_request(),
    )!!;

    assert(picked.graphics == 0 && picked.compute == 0);
    assert(picked.transfer == 0);
    assert(picked.present == 1 && picked.presentation);
}

fn void presentation_support_must_cover_every_queue_family() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 1),
    };

    fault result = @catch(vk::pick_queue_families_for_surface_request(
        table[..],
        {},
        gpu::internal::default_queue_request(),
    ));

    assert(result == gpu::INVALID_ARGUMENT);
}

fn void presentation_requires_a_capable_queue_family() @test {
    vk::QueueFamilyProperties[1] table = {
        family(QF_G | QF_C | QF_T, 1),
    };
    bool[1] presentation_support = { false };

    fault result = @catch(vk::pick_queue_families_for_surface_request(
        table[..],
        presentation_support[..],
        gpu::internal::default_queue_request(),
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

fn void selected_device_families_include_private_presentation() @test {
    vk::QueueFamilies picked = {
        .selected     = { .graphics, .compute },
        .graphics     = 0,
        .compute      = 1,
        .presentation = true,
        .present      = 2,
    };
    uint[3] families;

    uint count = vk::selected_device_queue_families(
        &picked,
        true,
        families[..],
    );

    assert(count == 3
            && families[0] == 0
            && families[1] == 1
            && families[2] == 2,
        "logical-device families include the private presentation family");
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
    vk::QueueCompletionState[1] completions = {{
        .queue    = queue,
        .timeline = (vk::Semaphore)(uptr)0x40,
        .roles    = { .graphics },
        .bound    = true,
    }};
    vk::VkDeviceState state = {
        .selected_queue_roles   = { .graphics },
        .graphics_family        = 4,
        .graphics_queue         = queue,
        .queue_completions      = completions[..],
        .graphics_completion_id = 0,
        .presentation           = {
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
    assert(vk::present_queue_mutex_for(&state)
            == vk::queue_mutex_for(&state, gpu::QueueKind.GRAPHICS),
        "aliased presentation and graphics use one native queue mutex");
}

fn void swapchain_sharing_includes_a_separate_present_family() @test {
    vk::Queue graphics_queue = (vk::Queue)(uptr)0x10;
    vk::Queue present_queue = (vk::Queue)(uptr)0x20;
    vk::QueueCompletionState[1] completions = {{
        .queue    = graphics_queue,
        .timeline = (vk::Semaphore)(uptr)0x40,
        .roles    = { .graphics },
        .bound    = true,
    }};
    vk::VkDeviceState state = {
        .selected_queue_roles   = { .graphics },
        .graphics_family        = 4,
        .graphics_queue         = graphics_queue,
        .queue_completions      = completions[..],
        .graphics_completion_id = 0,
        .presentation           = {
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
    assert(vk::present_queue_mutex_for(&state)
            != vk::queue_mutex_for(&state, gpu::QueueKind.GRAPHICS),
        "a private presentation queue owns a separate mutex");
    assert(vk::native_queue_mutex_for(&state, present_queue)
            == vk::present_queue_mutex_for(&state),
        "the native queue lookup reaches the private presentation mutex");
}

fn void selected_non_graphics_present_queues_share_role_mutexes() @test {
    vk::Queue graphics_queue = (vk::Queue)(uptr)0x10;
    vk::Queue compute_queue = (vk::Queue)(uptr)0x20;
    vk::Queue transfer_queue = (vk::Queue)(uptr)0x30;
    vk::QueueCompletionState[3] completions = {
        {
            .queue    = graphics_queue,
            .timeline = (vk::Semaphore)(uptr)0x40,
            .roles    = { .graphics },
            .bound    = true,
        },
        {
            .queue    = compute_queue,
            .timeline = (vk::Semaphore)(uptr)0x50,
            .roles    = { .compute },
            .bound    = true,
        },
        {
            .queue    = transfer_queue,
            .timeline = (vk::Semaphore)(uptr)0x60,
            .roles    = { .transfer },
            .bound    = true,
        },
    };
    vk::VkDeviceState state = {
        .selected_queue_roles   = { .graphics, .compute, .transfer },
        .graphics_queue         = graphics_queue,
        .compute_queue          = compute_queue,
        .transfer_queue         = transfer_queue,
        .queue_completions      = completions[..],
        .graphics_completion_id = 0,
        .compute_completion_id  = 1,
        .transfer_completion_id = 2,
        .presentation           = {
            .enabled = true,
            .queue   = compute_queue,
        },
    };

    assert(vk::present_queue_mutex_for(&state)
            == vk::queue_mutex_for(&state, gpu::QueueKind.COMPUTE),
        "presentation on the selected compute queue shares its mutex");

    state.presentation.queue = transfer_queue;
    assert(vk::present_queue_mutex_for(&state)
            == vk::queue_mutex_for(&state, gpu::QueueKind.TRANSFER),
        "presentation on the selected transfer queue shares its mutex");
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
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name         = "gpu_cross_queue_timeline_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::QueueRequest request = {
        .required = { .graphics, .compute },
        .distinct = { .compute },
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
        gpu::Queue compute =
            gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
        gpu::Queue graphics =
            gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
        assert(compute.id != graphics.id,
            "the timeline chain crosses distinct compute and graphics queues");

        gpu::SubmitDesc empty;
        gpu::CompletionPoint compute_done =
            gpu::submit(compute, &empty)!!;
        gpu::CompletionWait[1] waits = {{
            .point  = compute_done,
            .before = { .all },
        }};
        gpu::SubmitDesc graphics_submit = {
            .completion_waits = waits[..],
        };
        gpu::CompletionPoint graphics_done =
            gpu::submit(graphics, &graphics_submit)!!;

        gpu::wait_completion(graphics_done, 1_000_000_000)!!;
        return;
    }
}

fn void distinct_transfer_compute_graphics_completion_chain() @test {
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

        assert(graphics.id != compute.id
                && graphics.id != transfer.id
                && compute.id != transfer.id,
            "the integration request publishes three distinct queues");

        gpu::SubmitDesc empty;
        gpu::CompletionPoint transfer_done =
            gpu::submit(transfer, &empty)!!;
        gpu::CompletionWait[1] compute_waits = {{
            .point  = transfer_done,
            .before = { .compute },
        }};
        gpu::SubmitDesc compute_submit = {
            .completion_waits = compute_waits[..],
        };
        gpu::CompletionPoint compute_done =
            gpu::submit(compute, &compute_submit)!!;

        gpu::CompletionWait[1] fragment_waits = {{
            .point  = compute_done,
            .before = { .fragment_shader },
        }};
        gpu::SubmitDesc fragment_submit = {
            .completion_waits = fragment_waits[..],
        };
        gpu::CompletionPoint fragment_done =
            gpu::submit(graphics, &fragment_submit)!!;

        gpu::CompletionPoint transfer_for_vertex =
            gpu::submit(transfer, &empty)!!;
        gpu::CompletionWait[1] vertex_waits = {{
            .point  = transfer_for_vertex,
            .before = { .vertex_shader },
        }};
        gpu::SubmitDesc vertex_submit = {
            .completion_waits = vertex_waits[..],
        };
        gpu::CompletionPoint vertex_done =
            gpu::submit(graphics, &vertex_submit)!!;

        gpu::CompletionWait[1] graphics_to_compute = {{
            .point  = vertex_done,
            .before = { .compute },
        }};
        gpu::SubmitDesc compute_after_graphics = {
            .completion_waits = graphics_to_compute[..],
        };
        gpu::CompletionPoint compute_after_vertex =
            gpu::submit(compute, &compute_after_graphics)!!;

        gpu::CompletionPoint transfer_for_multi =
            gpu::submit(transfer, &empty)!!;
        gpu::CompletionWait[2] multi_waits = {
            {
                .point  = compute_after_vertex,
                .before = { .all },
            },
            {
                .point  = transfer_for_multi,
                .before = { .fragment_shader },
            },
        };
        gpu::SubmitDesc multi_submit = {
            .completion_waits = multi_waits[..],
        };
        gpu::CompletionPoint multi_done =
            gpu::submit(graphics, &multi_submit)!!;

        gpu::wait_completion(fragment_done, 1_000_000_000)!!;
        gpu::wait_completion(multi_done, 1_000_000_000)!!;

        gpu::Queue[3] queues = { graphics, compute, transfer };
        for (uint round = 0; round < 4; round++) {
            gpu::CompletionPoint[3] completions;
            foreach (queue_index, queue : queues) {
                completions[queue_index] = gpu::submit(queue, &empty)!!;
            }
            foreach (completion : completions) {
                gpu::wait_completion(completion)!!;
            }
        }
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

fn vk::QueueFamilyProperties timestamp_family(
    uint flags,
    uint valid_bits,
) {
    return {
        .queue_flags          = (vk::QueueFlags)flags,
        .queue_count          = 1,
        .timestamp_valid_bits = valid_bits,
    };
}

fn void timestamp_caps_preserve_distinct_selected_widths() @test {
    vk::QueueFamilyProperties[3] families = {
        timestamp_family(QF_G, 36),
        timestamp_family(QF_C, 48),
        timestamp_family(QF_T, 64),
    };
    vk::QueueFamilies selection = {
        .selected = { .graphics, .compute, .transfer },
        .graphics = 0,
        .compute  = 1,
        .transfer = 2,
    };

    gpu::TimestampCaps caps =
        vk::resolve_timestamp_caps(&selection, families[..], 1.25f);

    assert(caps.queues.graphics && caps.queues.compute);
    assert(!caps.queues.transfer);
    assert(caps.graphics_valid_bits == 36);
    assert(caps.compute_valid_bits == 48);
    assert(caps.transfer_valid_bits == 0);
    assert(caps.period_ns == 1.25f);
}

fn void timestamp_caps_copy_width_for_aliased_roles() @test {
    vk::QueueFamilyProperties[1] families = {
        timestamp_family(QF_G | QF_C | QF_T, 40),
    };
    vk::QueueFamilies selection = {
        .selected = { .graphics, .compute, .transfer },
        .graphics = 0,
        .compute  = 0,
        .transfer = 0,
    };

    gpu::TimestampCaps caps =
        vk::resolve_timestamp_caps(&selection, families[..], 2.0f);

    assert(caps.queues.graphics
        && caps.queues.compute
        && caps.queues.transfer);
    assert(caps.graphics_valid_bits == 40);
    assert(caps.compute_valid_bits == 40);
    assert(caps.transfer_valid_bits == 40);
    assert(caps.period_ns == 2.0f);
}

fn void timestamp_caps_ignore_unselected_roles() @test {
    vk::QueueFamilyProperties[1] families = {
        timestamp_family(QF_G | QF_C | QF_T, 32),
    };
    vk::QueueFamilies selection = {
        .selected = { .graphics },
        .graphics = 0,
        .compute  = 0,
        .transfer = 0,
    };

    gpu::TimestampCaps caps =
        vk::resolve_timestamp_caps(&selection, families[..], 3.0f);

    assert(caps.queues.graphics);
    assert(!caps.queues.compute && !caps.queues.transfer);
    assert(caps.graphics_valid_bits == 32);
    assert(caps.compute_valid_bits == 0);
    assert(caps.transfer_valid_bits == 0);
    assert(caps.period_ns == 3.0f);
}

fn void timestamp_caps_require_nonzero_valid_bits() @test {
    vk::QueueFamilyProperties[2] families = {
        timestamp_family(QF_G, 0),
        timestamp_family(QF_C, 52),
    };
    vk::QueueFamilies selection = {
        .selected = { .graphics, .compute },
        .graphics = 0,
        .compute  = 1,
    };

    gpu::TimestampCaps caps =
        vk::resolve_timestamp_caps(&selection, families[..], 0.5f);

    assert(!caps.queues.graphics && caps.queues.compute);
    assert(caps.graphics_valid_bits == 0);
    assert(caps.compute_valid_bits == 52);
    assert(caps.period_ns == 0.5f);
}

fn void timestamp_caps_clear_period_without_supported_roles() @test {
    vk::QueueFamilyProperties[3] families = {
        timestamp_family(QF_G, 0),
        timestamp_family(QF_C, 0),
        timestamp_family(QF_T, 64),
    };
    vk::QueueFamilies selection = {
        .selected = { .graphics, .compute, .transfer },
        .graphics = 0,
        .compute  = 1,
        .transfer = 2,
    };

    gpu::TimestampCaps caps =
        vk::resolve_timestamp_caps(&selection, families[..], 4.0f);

    assert((uint)caps.queues == 0);
    assert(caps.graphics_valid_bits == 0);
    assert(caps.compute_valid_bits == 0);
    assert(caps.transfer_valid_bits == 0);
    assert(caps.period_ns == 0.0f);
}
