module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::core::mem::alloc;
import std::core::mem::allocators;
import std::atomic::types;

const char[*] LIFETIME_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");
ulong lifetime_counter_value;
uint lifetime_counter_query_calls;

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

fn gpu::Device make_lifetime_test_device() {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
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
    gpu::Device* device,
    OwnedSpan* src,
    OwnedSpan* dst,
) {
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
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

    gpu::free_allocation(&device, &owned.allocation)!!;
    owned.span = {};

    assert(!owned.allocation.is_valid(), "successful release consumes the allocation token");
    assert(@catch(gpu::get_allocation_info(&device, stale)) == gpu::INVALID_HANDLE,
        "immediate release invalidates copied tokens without a lifecycle boundary");
}

fn void recording_references_reject_allocation_release() @test {
    gpu::Device device = make_lifetime_test_device();
    defer destroy_test_device(&device)!!;

    OwnedSpan src = make_lifetime_span(&device, "recording_src");
    defer free_owned_span(&device, &src)!!;
    OwnedSpan dst = make_lifetime_span(&device, "recording_dst");
    defer free_owned_span(&device, &dst)!!;

    gpu::CommandList commands = record_lifetime_copy(&device, &src, &dst);
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

    gpu::CommandList commands = record_lifetime_copy(&device, &src, &dst);
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

    gpu::CommandList commands = record_lifetime_copy(&device, &src, &dst);
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
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(state.submitted_command_count == 0,
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
        (vk::VkDeviceState*)gpu::device_backend_state(device);

    vk::AttachmentViewSlot borrowed_view = { .swapchain_owned = true };
    gpu::AttachmentViewHandle view =
        state.attachment_views.alloc(borrowed_view)!!;
    defer state.attachment_views.free(view)!!;
    vk::AttachmentViewSlot* slot = state.attachment_views.get(view)!!;

    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    vk::track_command_reference(
        state:      state,
        record:     record,
        owner:      view.owner,
        index:      view.index,
        generation: view.generation,
    )!!;
    assert(slot.validation_references.load(AtomicOrdering.RELAXED) == 1,
        "recording retains the borrowed attachment view");

    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!!;
    assert(slot.validation_references.load(AtomicOrdering.RELAXED) == 1
        && state.submitted_command_count == 1,
        "submission transfers the borrowed-view reference to the completion batch");

    vk::QueueCompletionState* queue_completion =
        vk::queue_completion_state(state, queue.id)!!;
    ulong sequence = gpu::completion_point_sequence(completion);
    vk::SemaphoreWaitInfo wait_info = vk::semaphore_wait_info()
        .set_semaphore_count(1)
        .set_semaphores((&queue_completion.timeline)[:1])
        .set_values((&sequence)[:1]);
    assert(vk::wait_semaphores(
        device:        state.device,
        p_wait_info:   &wait_info,
        timeout:       ulong::max,
    ) == vk::Result.SUCCESS, "the submitted command reaches completion");
    assert(slot.validation_references.load(AtomicOrdering.RELAXED) == 1
        && state.submitted_command_count == 1,
        "completion alone preserves the borrowed-view reference until drain");

    vk::drain_completed_submitted_commands(state)!!;
    assert(slot.validation_references.load(AtomicOrdering.RELAXED) == 0
        && state.submitted_command_count == 0,
        "drain releases the completed borrowed-view reference");
    vk::QueueCompletionState* completion_state =
        vk::queue_completion_state(
            state,
            gpu::completion_point_queue_id(completion),
        )!!;
    assert(completion_state.retired_sequence.load(AtomicOrdering.ACQUIRE)
        == gpu::completion_point_sequence(completion),
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
    assert(slot.validation_references.load(AtomicOrdering.RELAXED) == 0
        && state.submitted_command_count == 0,
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

    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = LIFETIME_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = shader,
    };
    gpu::PipelineHandle pipeline =
        gpu::create_compute_pipeline(&device, &pipeline_desc)!!;
    defer (void)gpu::destroy_pipeline(&device, pipeline);

    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
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
    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
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

fn void submitted_command_reservations_prevent_capacity_overcommit() @test {
    vk::VkDeviceState state = {
        .validation     = true,
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    vk::ensure_submitted_command_capacity(&state, 15);
    state.submitted_command_count = 15;
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();

    vk::CommandRecord first_record = { .reference_count = 1 };
    vk::CommandRecord second_record = { .reference_count = 1 };
    vk::CommandRecord*[1] first_records = { &first_record };
    vk::CommandRecord*[1] second_records = { &second_record };

    uint first = vk::prepare_submitted_commands(
        &state,
        first_records[..],
    );
    uint second = vk::prepare_submitted_commands(
        &state,
        second_records[..],
    );
    assert(first == 1 && second == 1,
        "each concurrent submit reserves its reference batch");
    assert(state.submitted_command_reserved == 2,
        "both reservations remain visible until publish or cancellation");
    assert(state.submitted_commands.len >= 17,
        "the second reservation grows capacity past the first reservation");

    vk::cancel_submitted_command_reservation(&state, second);
    vk::cancel_submitted_command_reservation(&state, first);
    assert(state.submitted_command_reserved == 0,
        "cancellation releases every reserved batch");
    state.submitted_command_count = 0;
    vk::destroy_submitted_commands(&state);
}

fn void incomplete_reference_drain_is_deterministic() @test {
    vk::VkDeviceState state = {
        .validation     = true,
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    vk::QueueCompletionState[1] completions = {{
        .timeline = (vk::Semaphore)(uptr)1,
        .bound    = true,
    }};
    completions[0].published_sequence.store(7, AtomicOrdering.RELEASE);
    state.queue_completions = completions[..];
    vk::ensure_submitted_command_capacity(&state, 1);
    gpu::Queue queue = {
        .device = { .generation = 1 },
        .id     = 0,
        .roles  = { .graphics },
    };
    state.submitted_commands[0] = {
        .completion = gpu::make_completion_point(queue, 7)!!,
    };
    state.submitted_command_count = 1;

    lifetime_counter_value = 6;
    vk::drain_completed_submitted_commands_with_query(
        &state,
        &lifetime_counter_query,
    )!!;
    assert(state.submitted_command_count == 1,
        "an incomplete counter preserves the submitted reference batch");

    lifetime_counter_value = 7;
    vk::drain_completed_submitted_commands_with_query(
        &state,
        &lifetime_counter_query,
    )!!;
    assert(state.submitted_command_count == 0,
        "the reached counter releases the submitted reference batch");
    vk::destroy_submitted_commands(&state);
}

fn void completed_submission_release_boundary_is_exact() @test {
    vk::VkDeviceState state = {
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    state.command_mutex.init()!!;
    defer (void)state.command_mutex.destroy();
    vk::RecordingContextCell[1] context_cells;
    state.recording_contexts.slots = context_cells[..];
    vk::RecordingContextHandle context = state.recording_contexts.alloc({})!!;
    vk::RecordingContextState* context_state =
        state.recording_contexts.get(context)!!;
    vk::GeneratedPreprocessBuffer[3] reserved_preprocess = {{
        .reservation_in_use = true,
    }, {
        .reservation_in_use = true,
    }, {
        .reservation_in_use = true,
    }};
    context_state.reserved_preprocess = reserved_preprocess[..];
    defer {
        if (context_state.retired_commands.ptr != null) {
            alloc::free(
                state.host_allocator,
                context_state.retired_commands.ptr,
            );
        }
    }
    vk::QueueCompletionState[1] completions = {{ .bound = true }};
    state.queue_completions = completions[..];
    completions[0].published_sequence.store(9, AtomicOrdering.RELEASE);
    vk::ensure_submitted_command_capacity(&state, 3);
    gpu::Queue queue = {
        .device = { .generation = 1 },
        .id     = 0,
        .roles  = { .graphics },
    };
    Atomic{uint} first_validation_reference;
    Atomic{uint} second_validation_reference;
    Atomic{uint} third_validation_reference;
    first_validation_reference.store(1, AtomicOrdering.RELAXED);
    second_validation_reference.store(1, AtomicOrdering.RELAXED);
    third_validation_reference.store(1, AtomicOrdering.RELAXED);
    vk::CommandResourceReference[1] first_references = {{
        .validation_references = &first_validation_reference,
    }};
    vk::CommandResourceReference[1] second_references = {{
        .validation_references = &second_validation_reference,
    }};
    vk::CommandResourceReference[1] third_references = {{
        .validation_references = &third_validation_reference,
    }};
    vk::GeneratedPreprocessBuffer[1] first_preprocess = {{
        .reservation_context = context,
        .reservation_index   = 0,
        .reservation_owned   = true,
    }};
    vk::GeneratedPreprocessBuffer[1] second_preprocess = {{
        .reservation_context = context,
        .reservation_index   = 1,
        .reservation_owned   = true,
    }};
    vk::GeneratedPreprocessBuffer[1] third_preprocess = {{
        .reservation_context = context,
        .reservation_index   = 2,
        .reservation_owned   = true,
    }};
    state.submitted_commands[0] = {
        .completion                 = gpu::make_completion_point(queue, 7)!!,
        .context                    = context,
        .command_buffer             = (vk::CommandBuffer)(uptr)1,
        .references                 = first_references[..],
        .reference_count            = 1,
        .generated_preprocess       = first_preprocess[..],
        .generated_preprocess_count = 1,
    };
    state.submitted_commands[1] = {
        .completion                 = gpu::make_completion_point(queue, 8)!!,
        .context                    = context,
        .command_buffer             = (vk::CommandBuffer)(uptr)2,
        .references                 = second_references[..],
        .reference_count            = 1,
        .generated_preprocess       = second_preprocess[..],
        .generated_preprocess_count = 1,
    };
    state.submitted_commands[2] = {
        .completion                 = gpu::make_completion_point(queue, 8)!!,
        .context                    = context,
        .command_buffer             = (vk::CommandBuffer)(uptr)3,
        .references                 = third_references[..],
        .reference_count            = 1,
        .generated_preprocess       = third_preprocess[..],
        .generated_preprocess_count = 1,
    };
    state.submitted_command_count = 3;

    vk::CommandBufferStats before = vk::command_buffer_stats(&state);
    state.resource_mutex.lock()!!;
    vk::retire_queue_through_locked(
        state:            &state,
        completion:       &completions[0],
        queue_id:         0,
        native_completed: 7,
    );
    state.resource_mutex.unlock();
    vk::CommandBufferStats incomplete = vk::command_buffer_stats(&state);
    assert(incomplete.pending == before.pending - 1,
        "sequence N - 1 retires only its covered batch");
    assert(incomplete.freed == before.freed,
        "sequence N - 1 cannot free the submitted command buffer");
    assert(first_validation_reference.load(AtomicOrdering.RELAXED) == 0
        && second_validation_reference.load(AtomicOrdering.RELAXED) == 1
        && third_validation_reference.load(AtomicOrdering.RELAXED) == 1,
        "sequence N - 1 releases only its stable retained counter pointer");
    assert(!reserved_preprocess[0].reservation_in_use
        && reserved_preprocess[1].reservation_in_use
        && reserved_preprocess[2].reservation_in_use,
        "sequence N - 1 recycles only its covered preprocess reservation");
    assert(context_state.retired_command_count == 1
        && context_state.retired_commands[0].buffer
            == (vk::CommandBuffer)(uptr)1,
        "sequence N - 1 retires only its covered command buffer");
    assert(completions[0].retired_sequence.load(AtomicOrdering.ACQUIRE) == 7,
        "sequence N - 1 publishes only the exact retired prefix");

    state.resource_mutex.lock()!!;
    vk::retire_queue_through_locked(
        state:            &state,
        completion:       &completions[0],
        queue_id:         0,
        native_completed: 8,
    );
    state.resource_mutex.unlock();
    vk::CommandBufferStats completed = vk::command_buffer_stats(&state);
    assert(completed.pending == 0,
        "sequence N retires every batch covered by that sequence");
    assert(completed.freed == before.freed,
        "a null native buffer performs no backend free");
    assert(second_validation_reference.load(AtomicOrdering.RELAXED) == 0
        && third_validation_reference.load(AtomicOrdering.RELAXED) == 0,
        "sequence N releases every stable retained counter pointer");
    assert(!reserved_preprocess[1].reservation_in_use
        && !reserved_preprocess[2].reservation_in_use,
        "sequence N recycles every covered preprocess reservation");
    assert(context_state.retired_command_count == 3,
        "sequence N retires every covered command buffer");
    assert(completions[0].retired_sequence.load(AtomicOrdering.ACQUIRE) == 8,
        "sequence N publishes only after all covered batches retire");
    state.resource_mutex.lock()!!;
    vk::retire_queue_through_locked(
        state:            &state,
        completion:       &completions[0],
        queue_id:         0,
        native_completed: 9,
    );
    state.resource_mutex.unlock();
    assert(completions[0].retired_sequence.load(AtomicOrdering.ACQUIRE) == 9,
        "an empty submission advances the retired prefix without a synthetic batch");
    vk::destroy_submitted_commands(&state);
}

fn void threshold_drain_queries_once_and_publishes_retirement() @test {
    vk::VkDeviceState state = {
        .host_allocator = &allocators::LIBC_ALLOCATOR,
    };
    state.resource_mutex.init()!!;
    defer (void)state.resource_mutex.destroy();
    vk::QueueCompletionState[1] completions = {{
        .timeline = (vk::Semaphore)(uptr)1,
        .bound    = true,
    }};
    state.queue_completions = completions[..];
    completions[0].published_sequence.store(
        vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD,
        AtomicOrdering.RELEASE,
    );
    vk::ensure_submitted_command_capacity(
        &state,
        vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD,
    );
    gpu::Queue queue = {
        .device = { .generation = 1 },
        .id     = 0,
        .roles  = { .graphics },
    };
    for (uint i = 0; i < vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD; i++) {
        state.submitted_commands[i] = {
            .completion = gpu::make_completion_point(queue, (ulong)i + 1)!!,
        };
    }
    state.submitted_command_count = vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD;
    lifetime_counter_value = vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD;
    lifetime_counter_query_calls = 0;

    vk::drain_submitted_commands_if_needed_with_query(
        &state,
        &lifetime_counter_query,
    )!!;
    assert(lifetime_counter_query_calls == 1,
        "the threshold drain queries one represented queue once");
    assert(state.submitted_command_count == 0,
        "the threshold drain releases every covered batch");
    assert(completions[0].retired_sequence.load(AtomicOrdering.ACQUIRE)
        == vk::SUBMITTED_COMMAND_DRAIN_THRESHOLD,
        "the threshold drain publishes the retired prefix");
    vk::destroy_submitted_commands(&state);
}
