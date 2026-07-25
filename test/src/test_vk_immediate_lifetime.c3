module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::core::mem::allocators;
import std::atomic::types;
import std::thread;
import std::time;

const char[*] LIFETIME_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");
const long QUIESCENT_RELEASE_DEADLINE_NS = 5_000_000_000;
ulong lifetime_counter_value;
uint lifetime_counter_query_calls;
ulong distinct_upstream_counter_value;
uint distinct_upstream_query_calls;
vk::Semaphore distinct_upstream_queried_semaphore;
vk::Result distinct_upstream_query_failure;

struct QuiescentReleaseWorkerArgs {
    gpu::Device*       device;
    gpu::GpuAllocation allocation;
    Atomic{bool}       entered;
    Atomic{bool}       finished;
    fault              failure;
    bool               failed;
}

fn int quiescent_release_worker(void* raw_args) {
    QuiescentReleaseWorkerArgs* args =
        (QuiescentReleaseWorkerArgs*)raw_args;
    args.entered.store(true, AtomicOrdering.RELEASE);
    if (catch release_fault = gpu::free_allocation(
        args.device,
        &args.allocation,
    )) {
        args.failure = release_fault;
        args.failed = true;
    }
    args.finished.store(true, AtomicOrdering.RELEASE);
    return 0;
}

fn vk::Result lifetime_counter_query(
    vk::Device device,
    vk::Semaphore semaphore,
    ulong* value,
) {
    (void)device;
    (void)semaphore;
    lifetime_counter_query_calls++;
    *value = lifetime_counter_value;
    return vk::Result.SUCCESS;
}

fn vk::Result distinct_upstream_counter_query(
    vk::Device device,
    vk::Semaphore semaphore,
    ulong* value,
) {
    (void)device;
    distinct_upstream_query_calls++;
    distinct_upstream_queried_semaphore = semaphore;
    if (semaphore == (vk::Semaphore)(uptr)11
        && distinct_upstream_query_failure != vk::Result.SUCCESS) {
        return distinct_upstream_query_failure;
    }
    *value = distinct_upstream_counter_value;
    return vk::Result.SUCCESS;
}

fn gpu::Device make_lifetime_test_device() {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_immediate_lifetime_test",
    };
    return create_test_device(&desc)!!;
}

fn OwnedSpan make_lifetime_span(
    gpu::Device* device,
    ZString debug_name,
) {
    return allocate_owned_span(
        device,
        256,
        gpu::MemoryClass.GPU_PRIVATE,
        { .graphics, .compute, .transfer },
        debug_name,
    )!!;
}

fn gpu::CommandList record_lifetime_copy(
    gpu::CommandAllocator* allocator,
    OwnedSpan* src,
    OwnedSpan* dst,
) {
    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    gpu::BufferCopyDesc copy = {
        .src = src.span,
        .dst = dst.span,
    };
    gpu::cmd_copy_buffer(&commands, &copy)!!;
    return commands;
}

fn gpu::PipelineHandle make_lifetime_compute_pipeline(gpu::Device* device) {
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = LIFETIME_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc pipeline_desc = { .shader = shader };
    return gpu::create_compute_pipeline(device, &pipeline_desc)!!;
}

fn void unreferenced_destroy_is_immediate() @test {
    gpu::Device device = make_lifetime_test_device();
    defer destroy_test_device(&device)!!;
    OwnedSpan owned = make_lifetime_span(&device, "immediate_unreferenced");
    gpu::GpuAllocation stale = owned.allocation;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    vk::reset_completion_work_counters(state);

    gpu::free_allocation(&device, &owned.allocation)!!;
    owned.span = {};

    vk::CompletionWorkCounters completion =
        vk::completion_work_counters(state);
    assert(!owned.allocation.is_valid(), "successful release consumes the allocation token");
    assert(@catch(gpu::get_allocation_info(&device, stale)) == gpu::INVALID_HANDLE,
        "immediate release invalidates copied tokens without a lifecycle boundary");
    assert(completion.native_counter_queries == 0
        && completion.native_wait_calls == 0,
        "quiescent release performs no native completion query or wait");
}

fn void quiescent_release_ignores_unrelated_stalled_queue() @test {
    gpu::Device device = make_lifetime_test_device();
    defer destroy_test_device(&device)!!;
    OwnedSpan owned = make_lifetime_span(&device, "stalled_queue_unrelated");
    defer {
        if (owned.allocation.is_valid()) {
            free_owned_span(&device, &owned)!!;
        }
    }
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    vk::QueueCompletionState* completion =
        vk::queue_completion_state(state, queue.id)!!;
    vk::Semaphore blocker;
    vk::Fence fence;
    Thread worker;
    bool blocker_live;
    bool fence_live;
    bool submission_live;
    bool worker_created;
    bool worker_joined;
    defer {
        if (blocker_live && submission_live) {
            vk::SemaphoreSignalInfo cleanup_signal =
                vk::semaphore_signal_info()
                    .set_semaphore(blocker)
                    .set_value(1);
            (void)vk::signal_semaphore(state.device, &cleanup_signal);
        }
        if (worker_created && !worker_joined) {
            (void)worker.join();
        }
        if (fence_live && submission_live) {
            (void)vk::wait_for_fences(
                device:      state.device,
                fence_count: 1,
                p_fences:    &fence,
                wait_all:    vk::TRUE,
                timeout:     ulong::max,
            );
        }
        if (fence_live) vk::destroy_fence(state.device, fence, null);
        if (blocker_live) {
            vk::destroy_semaphore(state.device, blocker, null);
        }
    }
    blocker = vk::create_private_timeline(state)!!;
    blocker_live = true;
    vk::FenceCreateInfo fence_info = vk::fence_create_info();
    vk::Result fence_create_result = vk::create_fence(
        device:        state.device,
        p_create_info: &fence_info,
        p_allocator:   null,
        p_fence:       &fence,
    );
    if (fence_create_result == vk::Result.SUCCESS) fence_live = true;
    vk::check_backend_result(
        state:     state,
        operation: "create stalled-queue fence",
        result:    fence_create_result,
    )!!;
    vk::SemaphoreSubmitInfo wait = vk::semaphore_submit_info()
        .set_semaphore(blocker)
        .set_value(1)
        .set_stage_mask(
            vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        );
    vk::SubmitInfo2 submit = vk::submit_info2()
        .set_wait_semaphore_infos((&wait)[:1]);
    completion.submission_mutex.lock()!!;
    vk::Result submit_result = vk::queue_submit2(
        queue:        completion.queue,
        submit_count: 1,
        p_submits:    &submit,
        fence:        fence,
    );
    completion.submission_mutex.unlock();
    if (submit_result == vk::Result.SUCCESS) submission_live = true;
    vk::check_backend_result(
        state:     state,
        operation: "submit stalled queue work",
        result:    submit_result,
    )!!;
    vk::reset_completion_work_counters(state);
    QuiescentReleaseWorkerArgs args = {
        .device     = &device,
        .allocation = owned.allocation,
    };
    worker.create(&quiescent_release_worker, &args)!!;
    worker_created = true;
    while (!args.entered.load(AtomicOrdering.ACQUIRE)) thread::yield();
    std::time::Clock deadline = std::time::clock::now();
    while (!args.finished.load(AtomicOrdering.ACQUIRE)
        && (long)deadline.to_now() < QUIESCENT_RELEASE_DEADLINE_NS) {
        thread::yield();
    }
    bool finished_before_signal = args.finished.load(AtomicOrdering.ACQUIRE);

    vk::SemaphoreSignalInfo signal = vk::semaphore_signal_info()
        .set_semaphore(blocker)
        .set_value(1);
    vk::Result signal_result = vk::signal_semaphore(state.device, &signal);
    int worker_result;
    bool worker_join_failed;
    int? joined = worker.join();
    worker_joined = true;
    if (catch join_fault = joined) {
        (void)join_fault;
        worker_join_failed = true;
    } else {
        worker_result = joined;
    }
    vk::Result fence_result = vk::wait_for_fences(
        device:      state.device,
        fence_count: 1,
        p_fences:    &fence,
        wait_all:    vk::TRUE,
        timeout:     ulong::max,
    );
    vk::destroy_fence(state.device, fence, null);
    fence_live = false;
    vk::destroy_semaphore(state.device, blocker, null);
    blocker_live = false;
    submission_live = false;
    if (!args.failed) {
        owned.allocation = {};
        owned.span = {};
    }
    vk::CompletionWorkCounters work = vk::completion_work_counters(state);

    assert(signal_result == vk::Result.SUCCESS
        && fence_result == vk::Result.SUCCESS,
        "the stalled submission resumes and completes during cleanup");
    assert(!worker_join_failed && worker_result == 0 && !args.failed,
        "quiescent release succeeds on the worker: %s", args.failure);
    assert(finished_before_signal,
        "quiescent release does not wait for unrelated queue or device idle");
    assert(work.native_counter_queries == 0 && work.native_wait_calls == 0,
        "quiescent release performs no native completion query or wait");
}

fn void recording_references_reject_allocation_release() @test {
    gpu::Device device = make_lifetime_test_device();
    defer destroy_test_device(&device)!!;

    OwnedSpan src = make_lifetime_span(&device, "recording_src");
    defer free_owned_span(&device, &src)!!;
    OwnedSpan dst = make_lifetime_span(&device, "recording_dst");
    defer free_owned_span(&device, &dst)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = record_lifetime_copy(&allocator, &src, &dst);
    fault src_fault = @catch(gpu::free_allocation(&device, &src.allocation));
    fault dst_fault = @catch(gpu::free_allocation(&device, &dst.allocation));
    assert(src_fault == gpu::RESOURCE_IN_USE && dst_fault == gpu::RESOURCE_IN_USE,
        "recording references reject release");
    assert(src.allocation.is_valid() && dst.allocation.is_valid(),
        "rejected release preserves both owning tokens");

    gpu::discard_commands(&commands)!!;
    gpu::free_allocation(&device, &src.allocation)!!;
    src.span = {};
    gpu::free_allocation(&device, &dst.allocation)!!;
    dst.span = {};
}

fn void executable_references_reject_allocation_release() @test {
    gpu::Device device = make_lifetime_test_device();
    defer destroy_test_device(&device)!!;

    OwnedSpan src = make_lifetime_span(&device, "executable_src");
    defer free_owned_span(&device, &src)!!;
    OwnedSpan dst = make_lifetime_span(&device, "executable_dst");
    defer free_owned_span(&device, &dst)!!;
    gpu::PipelineHandle pipeline = make_lifetime_compute_pipeline(&device);
    defer (void)gpu::destroy_pipeline(&device, pipeline);

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = record_lifetime_copy(&allocator, &src, &dst);
    bind_test_compute_pipeline(&commands, pipeline);
    gpu::cmd_dispatch(&commands, (gpu::GpuAddress)1, { 0, 0, 0 })!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    assert(@catch(gpu::free_allocation(&device, &dst.allocation))
        == gpu::RESOURCE_IN_USE, "executable references reject release");
    assert(dst.allocation.is_valid(), "rejected executable release preserves ownership");
    assert(@catch(gpu::destroy_pipeline(&device, pipeline))
        == gpu::RESOURCE_IN_USE,
        "executable command retains its bound pipeline");

    gpu::discard_executable_commands(&executable)!!;
    gpu::free_allocation(&device, &src.allocation)!!;
    src.span = {};
    gpu::free_allocation(&device, &dst.allocation)!!;
    dst.span = {};
    gpu::destroy_pipeline(&device, pipeline)!!;
    pipeline = {};
}

fn void incomplete_submitted_commands_retain_references() @test {
    gpu::Device device = make_lifetime_test_device();
    defer destroy_test_device(&device)!!;

    OwnedSpan src = make_lifetime_span(&device, "submitted_src");
    defer free_owned_span(&device, &src)!!;
    OwnedSpan dst = make_lifetime_span(&device, "submitted_dst");
    defer free_owned_span(&device, &dst)!!;
    gpu::PipelineHandle pipeline = make_lifetime_compute_pipeline(&device);
    defer (void)gpu::destroy_pipeline(&device, pipeline);

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = record_lifetime_copy(&allocator, &src, &dst);
    bind_test_compute_pipeline(&commands, pipeline);
    gpu::cmd_dispatch(&commands, (gpu::GpuAddress)1, { 0, 0, 0 })!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;

    if (!gpu::poll_completion(completion)!!) {
        assert(@catch(gpu::free_allocation(&device, &dst.allocation))
            == gpu::RESOURCE_IN_USE,
            "incomplete submitted work rejects release");
        assert(dst.allocation.is_valid(),
            "rejected submitted release preserves ownership");
        assert(@catch(gpu::destroy_pipeline(&device, pipeline))
            == gpu::RESOURCE_IN_USE,
            "incomplete submitted command retains its bound pipeline");
    }
    gpu::wait_completion(completion)!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    vk::QueueCompletionState* queue_completion =
        vk::queue_completion_state(state, queue.id)!!;
    assert(queue_completion.submitted_count == 0,
        "waiting releases completed validation-reference batches");
    gpu::free_allocation(&device, &src.allocation)!!;
    src.span = {};
    gpu::free_allocation(&device, &dst.allocation)!!;
    dst.span = {};
    gpu::destroy_pipeline(&device, pipeline)!!;
    pipeline = {};
}

fn void borrowed_attachment_view_reference_releases_after_completion_drain() @test {
    gpu::Device device = make_lifetime_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;

    vk::AttachmentViewSlot borrowed_view = { .swapchain_owned = true };
    gpu::AttachmentViewHandle view =
        state.attachment_views.alloc(borrowed_view)!!;
    defer state.attachment_views.free(view)!!;
    vk::AttachmentViewSlot* slot = state.attachment_views.get(view)!!;

    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    vk::track_command_reference(
        state:      state,
        record:     record,
        owner:      view.owner,
        index:      view.index,
        generation: view.generation,
    )!!;
    assert(slot.tracked_command_references.load(AtomicOrdering.RELAXED) == 1,
        "recording retains the borrowed attachment view");

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    vk::QueueCompletionState* queue_completion =
        vk::queue_completion_state(state, queue.id)!!;
    assert(slot.tracked_command_references.load(AtomicOrdering.RELAXED) == 1
        && queue_completion.submitted_count == 1,
        "submission transfers the borrowed-view reference to the completion batch");

    ulong sequence = gpu::internal::completion_point_sequence(completion);
    vk::SemaphoreWaitInfo wait_info = vk::semaphore_wait_info()
        .set_semaphore_count(1)
        .set_semaphores((&queue_completion.timeline)[:1])
        .set_values((&sequence)[:1]);
    assert(vk::wait_semaphores(
        device:        state.device,
        p_wait_info:   &wait_info,
        timeout:       ulong::max,
    ) == vk::Result.SUCCESS, "the submitted command reaches completion");
    assert(slot.tracked_command_references.load(AtomicOrdering.RELAXED) == 1
        && queue_completion.submitted_count == 1,
        "completion alone preserves the borrowed-view reference until drain");

    vk::drain_completed_submitted_commands(state)!!;
    assert(slot.tracked_command_references.load(AtomicOrdering.RELAXED) == 0
        && queue_completion.submitted_count == 0,
        "drain releases the completed borrowed-view reference");
    vk::QueueCompletionState* completion_state =
        vk::queue_completion_state(
            state,
            gpu::internal::completion_point_queue_id(completion),
        )!!;
    assert(completion_state.retired_sequence.load(AtomicOrdering.ACQUIRE)
        == gpu::internal::completion_point_sequence(completion),
        "the explicit drain publishes the retired prefix");
    vk::reset_completion_work_counters(state);
    assert(gpu::poll_completion(completion)!!,
        "the drained point uses the cached poll path");
    vk::CompletionWorkCounters work = vk::completion_work_counters(state);
    assert(work.native_counter_queries == 0
        && work.retirement_lock_acquisitions == 0
        && work.cached_poll_hits == 1,
        "post-drain polling performs no native or retirement-lock work");
    vk::drain_completed_submitted_commands(state)!!;
    assert(slot.tracked_command_references.load(AtomicOrdering.RELAXED) == 0
        && queue_completion.submitted_count == 0,
        "a repeated drain cannot release the borrowed-view reference twice");
}

fn void explicit_texture_and_pipeline_references_reject_destroy() @test {
    gpu::Device device = make_lifetime_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc texture_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &texture_desc)!!;
    defer (void)gpu::destroy_texture(&device, texture);

    gpu::PipelineHandle pipeline = make_lifetime_compute_pipeline(&device);
    defer (void)gpu::destroy_pipeline(&device, pipeline);

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::TextureBarrier barrier = {
        .texture = texture,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    };
    gpu::cmd_texture_barrier(&commands, &barrier)!!;
    bind_test_compute_pipeline(&commands, pipeline);

    assert(@catch(gpu::destroy_texture(&device, texture))
        == gpu::RESOURCE_IN_USE, "recorded texture references reject destroy");
    assert(@catch(gpu::destroy_pipeline(&device, pipeline))
        == gpu::RESOURCE_IN_USE, "recorded pipeline references reject destroy");

    gpu::reset_command_resolution_stats();
    gpu::cmd_dispatch(&commands, (gpu::GpuAddress)1, { 1, 1, 1 })!!;
    assert_warm_command_resolution(gpu::command_resolution_stats(), 1);

    gpu::discard_commands(&commands)!!;
    gpu::destroy_texture(&device, texture)!!;
    texture = {};
    gpu::destroy_pipeline(&device, pipeline)!!;
    pipeline = {};
}

fn void failed_command_validation_retains_no_resource_reference() @test {
    gpu::Device device = make_lifetime_test_device();

    defer destroy_test_device(&device)!!;

    gpu::TextureDesc desc = {
        .access    = { .graphics },
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .transfer_dst },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::TextureBarrier barrier = {
        .texture = texture,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = { .layout = gpu::TextureLayout.UNDEFINED },
    };
    assert(@catch(gpu::cmd_texture_barrier(&commands, &barrier))
        == gpu::INVALID_ARGUMENT,
        "an invalid semantic use faults");
    gpu::destroy_texture(&device, texture)!!;
    gpu::discard_commands(&commands)!!;
}

fn void incomplete_reference_drain_is_deterministic() @test {
    vk::VkDeviceState state = {
        .validation_policy = {
            .contract        = gpu::ContractValidation.FULL,
            .track_lifetimes = true,
        },
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    vk::QueueCompletionState[1] completions = {{
        .timeline = (vk::Semaphore)(uptr)1,
        .bound    = true,
    }};
    completions[0].published_sequence.store(7, AtomicOrdering.RELEASE);
    state.queue_completions = completions[..];
    completions[0].retirement_mutex.init()!!;
    defer (void)completions[0].retirement_mutex.destroy();
    gpu::Queue queue = {
        .device = { .generation = 1 },
        .id     = 0,
        .roles  = { .graphics },
    };
    vk::CommandScratch[1] scratch = {{ .in_use = true }};
    uint[1] available;
    vk::CommandAllocatorCell[1] allocator_cells;
    state.command_allocators = {
        .slots = allocator_cells[..],
        .owner = 1,
    };
    gpu::CommandAllocatorHandle allocator = state.command_allocators.alloc({
        .queue             = queue,
        .scratch           = scratch[..],
        .available         = available[..],
        .in_flight_buffers = 1,
    })!!;
    vk::CommandAllocatorSlot* allocator_slot =
        state.command_allocators.get(allocator)!!;
    allocator_slot.mutex.init()!!;
    defer (void)allocator_slot.mutex.destroy();
    vk::CommandCell[1] command_cells;
    init_submitted_command_fixture(&state, command_cells[..]);
    defer (void)state.command_mutex.destroy();
    append_submitted_command_fixture(
        completion: &completions[0],
        record:     submitted_command_fixture_record(&state, allocator, 0),
        point:      gpu::internal::make_completion_point(queue, 7)!!,
    );

    lifetime_counter_value = 6;
    vk::drain_completed_submitted_commands_with_query(
        &state,
        &lifetime_counter_query,
    )!!;
    assert(completions[0].submitted_count == 1,
        "an incomplete counter preserves the submitted reference batch");

    lifetime_counter_value = 7;
    vk::drain_completed_submitted_commands_with_query(
        &state,
        &lifetime_counter_query,
    )!!;
    assert(completions[0].submitted_count == 0,
        "the reached counter releases the submitted reference batch");
}

fn void successful_observation_drains_distinct_upstream_identity() @test {
    vk::VkDeviceState state = {
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    vk::QueueCompletionState[2] completions = {
        {
            .timeline = (vk::Semaphore)(uptr)11,
            .bound    = true,
        },
        {
            .timeline = (vk::Semaphore)(uptr)22,
            .bound    = true,
        },
    };
    completions[0].published_sequence.store(3, AtomicOrdering.RELEASE);
    completions[1].published_sequence.store(3, AtomicOrdering.RELEASE);
    state.queue_completions = completions[..];
    foreach (&completion : completions) {
        completion.retirement_mutex.init()!!;
    }
    defer {
        foreach (&completion : completions) {
            (void)completion.retirement_mutex.destroy();
        }
    }
    gpu::Queue upstream_queue = {
        .device = { .generation = 1 },
        .id     = 0,
        .roles  = { .compute },
    };
    vk::CommandScratch[1] scratch = {{ .in_use = true }};
    uint[1] available;
    vk::CommandAllocatorCell[1] allocator_cells;
    state.command_allocators = {
        .slots = allocator_cells[..],
        .owner = 1,
    };
    gpu::CommandAllocatorHandle allocator = state.command_allocators.alloc({
        .queue             = upstream_queue,
        .scratch           = scratch[..],
        .available         = available[..],
        .in_flight_buffers = 1,
    })!!;
    vk::CommandAllocatorSlot* slot = state.command_allocators.get(
        allocator,
    )!!;
    slot.mutex.init()!!;
    defer (void)slot.mutex.destroy();
    vk::CommandCell[1] command_cells;
    init_submitted_command_fixture(&state, command_cells[..]);
    defer (void)state.command_mutex.destroy();
    append_submitted_command_fixture(
        completion: &completions[0],
        record:     submitted_command_fixture_record(&state, allocator, 0),
        point:      gpu::internal::make_completion_point(upstream_queue, 3)!!,
    );
    distinct_upstream_counter_value = 3;
    distinct_upstream_query_calls = 0;
    distinct_upstream_queried_semaphore = (vk::Semaphore)null;
    distinct_upstream_query_failure = vk::Result.ERROR_OUT_OF_HOST_MEMORY;

    assert(@catch(vk::poll_queue_completion_with_query(
        state:      &state,
        completion: &completions[1],
        queue_id:   1,
        sequence:   3,
        query_fn:   &distinct_upstream_counter_query,
    )) == gpu::OUT_OF_HOST_MEMORY,
        "a failing pending-identity query propagates before retirement");
    assert(distinct_upstream_query_calls == 2
            && completions[0].retired_sequence.load(
                AtomicOrdering.ACQUIRE,
            ) == 0
            && completions[1].retired_sequence.load(
                AtomicOrdering.ACQUIRE,
            ) == 0
            && completions[0].submitted_count == 1,
        "query failure publishes no retired prefix and preserves pending ownership");

    distinct_upstream_query_failure = vk::Result.SUCCESS;
    distinct_upstream_query_calls = 0;
    assert(vk::poll_queue_completion_with_query(
        state:      &state,
        completion: &completions[1],
        queue_id:   1,
        sequence:   3,
        query_fn:   &distinct_upstream_counter_query,
    )!!, "the injected downstream poll observes its exact sequence");

    assert(distinct_upstream_query_calls == 2
            && distinct_upstream_queried_semaphore == completions[0].timeline,
        "poll queries the downstream and exact pending upstream identities");
    assert(completions[0].retired_sequence.load(AtomicOrdering.ACQUIRE) == 3
            && completions[1].retired_sequence.load(AtomicOrdering.ACQUIRE)
                == 3,
        "distinct upstream and downstream retired prefixes remain independent");
    assert(completions[0].submitted_count == 0
            && completions[1].submitted_count == 0
            && slot.in_flight_buffers == 0
            && slot.available_count == 1
            && slot.available[0] == 0
            && !scratch[0].in_use,
        "downstream completion returns the exact upstream allocator unit");
}

fn void completed_submission_release_boundary_is_exact() @test {
    vk::VkDeviceState state = {
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    vk::CommandCell[3] command_cells;
    init_submitted_command_fixture(&state, command_cells[..]);
    defer (void)state.command_mutex.destroy();
    vk::QueueCompletionState[1] completions = {{ .bound = true }};
    state.queue_completions = completions[..];
    completions[0].retirement_mutex.init()!!;
    defer (void)completions[0].retirement_mutex.destroy();
    completions[0].published_sequence.store(9, AtomicOrdering.RELEASE);
    gpu::Queue queue = {
        .device = { .generation = 1 },
        .id     = 0,
        .roles  = { .graphics },
    };
    Atomic{uint} first_tracked_command_reference;
    Atomic{uint} second_tracked_command_reference;
    Atomic{uint} third_tracked_command_reference;
    first_tracked_command_reference.store(1, AtomicOrdering.RELAXED);
    second_tracked_command_reference.store(1, AtomicOrdering.RELAXED);
    third_tracked_command_reference.store(1, AtomicOrdering.RELAXED);
    vk::CommandResourceReference[1] first_references = {{
        .tracked_command_references = &first_tracked_command_reference,
    }};
    vk::CommandResourceReference[1] second_references = {{
        .tracked_command_references = &second_tracked_command_reference,
    }};
    vk::CommandResourceReference[1] third_references = {{
        .tracked_command_references = &third_tracked_command_reference,
    }};
    uint[1] first_preprocess = { 0 };
    uint[1] second_preprocess = { 0 };
    uint[1] third_preprocess = { 1 };
    vk::CommandScratch[1] first_scratch = {{
        .references                   = first_references[..],
        .reference_count              = 1,
        .generated_preprocess_indices = first_preprocess[..],
        .generated_preprocess_count   = 1,
        .in_use                      = true,
    }};
    vk::CommandScratch[2] second_scratch = {{
        .references                   = second_references[..],
        .reference_count              = 1,
        .generated_preprocess_indices = second_preprocess[..],
        .generated_preprocess_count   = 1,
        .in_use                      = true,
    }, {
        .references                   = third_references[..],
        .reference_count              = 1,
        .generated_preprocess_indices = third_preprocess[..],
        .generated_preprocess_count   = 1,
        .in_use                      = true,
    }};
    vk::GeneratedPreprocessBuffer[3] reserved_preprocess = {{
        .reservation_in_use = true,
    }, {
        .reservation_in_use = true,
    }, {
        .reservation_in_use = true,
    }};
    uint[1] first_available;
    uint[2] second_available;
    vk::CommandAllocatorCell[2] allocator_cells;
    state.command_allocators = {
        .slots = allocator_cells[..],
        .owner = 1,
    };
    gpu::CommandAllocatorHandle first_allocator = state.command_allocators.alloc({
        .queue                         = queue,
        .scratch                       = first_scratch[..],
        .available                     = first_available[..],
        .reservations                  = reserved_preprocess[:1],
        .generated_reservations_in_use = 1,
        .in_flight_buffers             = 1,
    })!!;
    gpu::CommandAllocatorHandle second_allocator = state.command_allocators.alloc({
        .queue                         = queue,
        .scratch                       = second_scratch[..],
        .available                     = second_available[..],
        .reservations                  = reserved_preprocess[1..],
        .generated_reservations_in_use = 2,
        .in_flight_buffers             = 2,
    })!!;
    vk::CommandAllocatorSlot* first_allocator_slot =
        state.command_allocators.get(first_allocator)!!;
    vk::CommandAllocatorSlot* second_allocator_slot =
        state.command_allocators.get(second_allocator)!!;
    first_allocator_slot.mutex.init()!!;
    defer (void)first_allocator_slot.mutex.destroy();
    second_allocator_slot.mutex.init()!!;
    defer (void)second_allocator_slot.mutex.destroy();
    append_submitted_command_fixture(
        completion: &completions[0],
        record:     submitted_command_fixture_record(
            &state,
            first_allocator,
            0,
        ),
        point:      gpu::internal::make_completion_point(queue, 7)!!,
    );
    append_submitted_command_fixture(
        completion: &completions[0],
        record:     submitted_command_fixture_record(
            &state,
            second_allocator,
            0,
        ),
        point:      gpu::internal::make_completion_point(queue, 8)!!,
    );
    append_submitted_command_fixture(
        completion: &completions[0],
        record:     submitted_command_fixture_record(
            &state,
            second_allocator,
            1,
        ),
        point:      gpu::internal::make_completion_point(queue, 8)!!,
    );

    vk::CommandBufferStats before = vk::command_buffer_stats(&state);
    completions[0].retirement_mutex.lock()!!;
    vk::retire_queue_through_locked(&state, &completions[0], 7);
    completions[0].retirement_mutex.unlock();
    vk::CommandBufferStats incomplete = vk::command_buffer_stats(&state);
    assert(incomplete.pending == before.pending - 1,
        "sequence N - 1 retires only its covered batch");
    assert(incomplete.freed == before.freed,
        "sequence N - 1 cannot free the submitted command buffer");
    assert(first_tracked_command_reference.load(AtomicOrdering.RELAXED) == 0
        && second_tracked_command_reference.load(AtomicOrdering.RELAXED) == 1
        && third_tracked_command_reference.load(AtomicOrdering.RELAXED) == 1,
        "sequence N - 1 releases only its stable retained counter pointer");
    assert(!reserved_preprocess[0].reservation_in_use
        && reserved_preprocess[1].reservation_in_use
        && reserved_preprocess[2].reservation_in_use,
        "sequence N - 1 recycles only its covered preprocess reservation");
    assert(first_allocator_slot.available_count == 1
            && first_allocator_slot.available[0] == 0
            && !first_scratch[0].in_use
            && first_allocator_slot.in_flight_buffers == 0,
        "sequence N - 1 returns its unit to the first allocator");
    assert(second_allocator_slot.available_count == 0
            && second_allocator_slot.in_flight_buffers == 2
            && second_scratch[0].in_use
            && second_scratch[1].in_use,
        "sequence N - 1 leaves the later allocator entirely unchanged");
    assert(completions[0].retired_sequence.load(AtomicOrdering.ACQUIRE) == 7,
        "sequence N - 1 publishes only the exact retired prefix");

    completions[0].retirement_mutex.lock()!!;
    vk::retire_queue_through_locked(&state, &completions[0], 8);
    completions[0].retirement_mutex.unlock();
    vk::CommandBufferStats completed = vk::command_buffer_stats(&state);
    assert(completed.pending == 0,
        "sequence N retires every batch covered by that sequence");
    assert(completed.freed == before.freed,
        "a null native buffer performs no backend free");
    assert(second_tracked_command_reference.load(AtomicOrdering.RELAXED) == 0
        && third_tracked_command_reference.load(AtomicOrdering.RELAXED) == 0,
        "sequence N releases every stable retained counter pointer");
    assert(!reserved_preprocess[1].reservation_in_use
        && !reserved_preprocess[2].reservation_in_use,
        "sequence N recycles every covered preprocess reservation");
    assert(first_allocator_slot.available_count == 1
            && first_allocator_slot.in_flight_buffers == 0
            && second_allocator_slot.available_count == 2
            && second_allocator_slot.in_flight_buffers == 0
            && !second_scratch[0].in_use
            && !second_scratch[1].in_use,
        "sequence N routes every later unit to the second allocator");
    assert(completions[0].retired_sequence.load(AtomicOrdering.ACQUIRE) == 8,
        "sequence N publishes only after all covered batches retire");
    completions[0].retirement_mutex.lock()!!;
    vk::retire_queue_through_locked(&state, &completions[0], 8);
    completions[0].retirement_mutex.unlock();
    assert(completions[0].submitted_count == 0
            && first_allocator_slot.available_count == 1
            && second_allocator_slot.available_count == 2
            && second_allocator_slot.in_flight_buffers == 0,
        "repeating the same retirement is idempotent across allocators");
    completions[0].retirement_mutex.lock()!!;
    vk::retire_queue_through_locked(&state, &completions[0], 9);
    completions[0].retirement_mutex.unlock();
    assert(completions[0].retired_sequence.load(AtomicOrdering.ACQUIRE) == 9,
        "an empty submission advances the retired prefix without a synthetic batch");
}

fn void device_loss_teardown_returns_each_in_flight_allocator_unit() @test {
    vk::VkDeviceState state = {
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    vk::CommandCell[2] command_cells;
    init_submitted_command_fixture(&state, command_cells[..]);
    defer (void)state.command_mutex.destroy();
    vk::QueueCompletionState[1] completions = {{ .bound = true }};
    state.queue_completions = completions[..];
    completions[0].retirement_mutex.init()!!;
    defer (void)completions[0].retirement_mutex.destroy();
    gpu::Queue queue = {
        .device = { .generation = 1 },
        .id     = 0,
        .roles  = { .graphics },
    };
    Atomic{uint} first_reference;
    Atomic{uint} second_reference;
    first_reference.store(1, AtomicOrdering.RELAXED);
    second_reference.store(1, AtomicOrdering.RELAXED);
    vk::CommandResourceReference[1] first_references = {{
        .tracked_command_references = &first_reference,
    }};
    vk::CommandResourceReference[1] second_references = {{
        .tracked_command_references = &second_reference,
    }};
    uint[1] first_preprocess = { 0 };
    uint[1] second_preprocess = { 0 };
    vk::CommandScratch[1] first_scratch = {{
        .references                   = first_references[..],
        .reference_count              = 1,
        .generated_preprocess_indices = first_preprocess[..],
        .generated_preprocess_count   = 1,
        .in_use                       = true,
    }};
    vk::CommandScratch[1] second_scratch = {{
        .references                   = second_references[..],
        .reference_count              = 1,
        .generated_preprocess_indices = second_preprocess[..],
        .generated_preprocess_count   = 1,
        .in_use                       = true,
    }};
    vk::GeneratedPreprocessBuffer[1] first_reservations = {{
        .reservation_in_use = true,
    }};
    vk::GeneratedPreprocessBuffer[1] second_reservations = {{
        .reservation_in_use = true,
    }};
    uint[1] first_available;
    uint[1] second_available;
    vk::CommandAllocatorCell[2] allocator_cells;
    state.command_allocators = {
        .slots = allocator_cells[..],
        .owner = 1,
    };
    gpu::CommandAllocatorHandle first_allocator = state.command_allocators.alloc({
        .queue                         = queue,
        .scratch                       = first_scratch[..],
        .available                     = first_available[..],
        .reservations                  = first_reservations[..],
        .generated_reservations_in_use = 1,
        .in_flight_buffers             = 1,
    })!!;
    gpu::CommandAllocatorHandle second_allocator = state.command_allocators.alloc({
        .queue                         = queue,
        .scratch                       = second_scratch[..],
        .available                     = second_available[..],
        .reservations                  = second_reservations[..],
        .generated_reservations_in_use = 1,
        .in_flight_buffers             = 1,
    })!!;
    vk::CommandAllocatorSlot* first_slot =
        state.command_allocators.get(first_allocator)!!;
    vk::CommandAllocatorSlot* second_slot =
        state.command_allocators.get(second_allocator)!!;
    first_slot.mutex.init()!!;
    defer (void)first_slot.mutex.destroy();
    second_slot.mutex.init()!!;
    defer (void)second_slot.mutex.destroy();
    append_submitted_command_fixture(
        completion: &completions[0],
        record:     submitted_command_fixture_record(
            &state,
            first_allocator,
            0,
        ),
        point:      gpu::internal::make_completion_point(queue, 1)!!,
    );
    append_submitted_command_fixture(
        completion: &completions[0],
        record:     submitted_command_fixture_record(
            &state,
            second_allocator,
            0,
        ),
        point:      gpu::internal::make_completion_point(queue, 2)!!,
    );

    vk::destroy_submitted_commands(&state);

    assert(completions[0].submitted_count == 0
            && completions[0].submitted_head == null
            && completions[0].submitted_tail == null,
        "device-loss teardown consumes every submitted batch");
    assert(first_reference.load(AtomicOrdering.RELAXED) == 0
            && second_reference.load(AtomicOrdering.RELAXED) == 0,
        "device-loss teardown releases references from every allocator");
    assert(!first_reservations[0].reservation_in_use
            && !second_reservations[0].reservation_in_use
            && first_slot.generated_reservations_in_use == 0
            && second_slot.generated_reservations_in_use == 0,
        "device-loss teardown returns generated reservations to each allocator");
    assert(first_slot.available_count == 1
            && second_slot.available_count == 1
            && first_slot.in_flight_buffers == 0
            && second_slot.in_flight_buffers == 0
            && !first_scratch[0].in_use
            && !second_scratch[0].in_use,
        "device-loss teardown returns each exact buffer and scratch block");

    vk::destroy_submitted_commands(&state);
    assert(completions[0].submitted_count == 0
            && first_slot.available_count == 1
            && second_slot.available_count == 1,
        "repeated defensive teardown is idempotent");
}

fn void threshold_drain_queries_once_and_publishes_retirement() @test {
    vk::VkDeviceState state = {
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    vk::CommandCell[vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD] command_cells;
    init_submitted_command_fixture(&state, command_cells[..]);
    defer (void)state.command_mutex.destroy();
    vk::QueueCompletionState[1] completions = {{
        .timeline = (vk::Semaphore)(uptr)1,
        .bound    = true,
    }};
    state.queue_completions = completions[..];
    completions[0].retirement_mutex.init()!!;
    defer (void)completions[0].retirement_mutex.destroy();
    completions[0].published_sequence.store(
        vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD,
        AtomicOrdering.RELEASE,
    );
    gpu::Queue queue = {
        .device = { .generation = 1 },
        .id     = 0,
        .roles  = { .graphics },
    };
    vk::CommandScratch[vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD] scratch;
    uint[vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD] available;
    vk::CommandAllocatorCell[1] allocator_cells;
    state.command_allocators = {
        .slots = allocator_cells[..],
        .owner = 1,
    };
    gpu::CommandAllocatorHandle allocator = state.command_allocators.alloc({
        .queue             = queue,
        .scratch           = scratch[..],
        .available         = available[..],
        .in_flight_buffers = vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD,
    })!!;
    vk::CommandAllocatorSlot* allocator_slot =
        state.command_allocators.get(allocator)!!;
    allocator_slot.mutex.init()!!;
    defer (void)allocator_slot.mutex.destroy();
    for (uint i = 0; i < vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD; i++) {
        scratch[i].in_use = true;
        append_submitted_command_fixture(
            completion: &completions[0],
            record:     submitted_command_fixture_record(
                &state,
                allocator,
                i,
            ),
            point:      gpu::internal::make_completion_point(
                queue,
                (ulong)i + 1,
            )!!,
        );
    }
    lifetime_counter_value = vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD;
    lifetime_counter_query_calls = 0;

    vk::drain_submitted_commands_if_needed_with_query(
        &state,
        &lifetime_counter_query,
    )!!;
    assert(lifetime_counter_query_calls == 1,
        "the threshold drain queries one represented queue once");
    assert(completions[0].submitted_count == 0,
        "the threshold drain releases every covered batch");
    assert(completions[0].retired_sequence.load(AtomicOrdering.ACQUIRE)
        == vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD,
        "the threshold drain publishes the retired prefix");
}
