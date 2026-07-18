module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::core::mem::allocators;

const char[*] LIFETIME_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");
ulong lifetime_counter_value;

fn vk::Result lifetime_counter_query(
    vk::Device device,
    vk::Semaphore semaphore,
    ulong* value,
) {
    (void)device;
    (void)semaphore;
    *value = lifetime_counter_value;
    return vk::Result.SUCCESS;
}

fn gpu::Device make_lifetime_test_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_immediate_lifetime_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
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

fn void unreferenced_mid_frame_destroy_is_immediate() @test {
    gpu::Device device = make_lifetime_test_device();
    defer destroy_test_device(&device)!!;

    gpu::FrameToken frame = gpu::begin_frame(&device)!!;
    OwnedSpan owned = make_lifetime_span(&device, "immediate_mid_frame");
    gpu::GpuAllocation stale = owned.allocation;

    gpu::free_allocation(&device, &owned.allocation)!!;
    owned.span = {};

    assert(!owned.allocation.is_valid(), "successful release consumes the allocation token");
    assert(@catch(gpu::get_allocation_info(&device, stale)) == gpu::INVALID_HANDLE,
        "immediate release invalidates copied tokens before the frame ends");
    gpu::end_frame(&frame)!!;
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

    gpu::CommandList commands = record_lifetime_copy(&device, &src, &dst);
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    assert(@catch(gpu::free_allocation(&device, &dst.allocation))
        == gpu::RESOURCE_IN_USE, "executable references reject release");
    assert(dst.allocation.is_valid(), "rejected executable release preserves ownership");

    gpu::discard_executable_commands(&executable)!!;
    gpu::free_allocation(&device, &src.allocation)!!;
    src.span = {};
    gpu::free_allocation(&device, &dst.allocation)!!;
    dst.span = {};
}

fn void incomplete_submitted_commands_retain_references() @test {
    gpu::Device device = make_lifetime_test_device();
    defer destroy_test_device(&device)!!;

    OwnedSpan src = make_lifetime_span(&device, "submitted_src");
    defer free_owned_span(&device, &src)!!;
    OwnedSpan dst = make_lifetime_span(&device, "submitted_dst");
    defer free_owned_span(&device, &dst)!!;

    gpu::CommandList commands = record_lifetime_copy(&device, &src, &dst);
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
}

fn void explicit_texture_and_pipeline_references_reject_destroy() @test {
    gpu::Device device = make_lifetime_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc texture_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
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
    gpu::ShaderHandle shader = gpu::create_shader(&device, &shader_desc)!!;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = shader,
        .push_constant_size = gpu::RootPush::size,
    };
    gpu::PipelineHandle pipeline =
        gpu::create_compute_pipeline(&device, &pipeline_desc)!!;
    defer (void)gpu::destroy_pipeline(&device, pipeline);

    gpu::CommandList commands = gpu::begin_commands(
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
    )!!;
    gpu::TextureBarrier barrier = {
        .texture       = texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.TRANSFER_DST,
    };
    gpu::cmd_texture_barrier(&commands, &barrier)!!;
    gpu::cmd_dispatch(&commands, pipeline, {}, { 1, 1, 1 })!!;

    assert(@catch(gpu::destroy_texture(&device, texture))
        == gpu::RESOURCE_IN_USE, "recorded texture references reject destroy");
    assert(@catch(gpu::destroy_pipeline(&device, pipeline))
        == gpu::RESOURCE_IN_USE, "recorded pipeline references reject destroy");
    gpu::destroy_shader(&device, shader)!!;

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
        .dimension = gpu::TextureDimension.TEX_2D,
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
        .texture      = texture,
        .before_stage = gpu::Stage.TRANSFER,
        .after_stage  = gpu::Stage.TRANSFER,
        .old_layout   = gpu::TextureLayout.TRANSFER_DST,
        .new_layout   = gpu::TextureLayout.TRANSFER_DST,
    };
    assert(@catch(gpu::cmd_texture_barrier(&commands, &barrier))
        == gpu::INVALID_RESOURCE_STATE,
        "the deliberately mismatched barrier faults");
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
