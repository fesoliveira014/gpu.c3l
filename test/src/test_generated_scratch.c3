module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::io;

const char[*] GENERATED_SCRATCH_SPIRV =
    $embed("shaders/root_pointer.comp.spv");

fn void generated_scratch_reservation_is_driver_compatible_and_bounded() @test {
    gpu::RuntimeDesc runtime_desc = {
        .backend          = gpu::BackendKind.VULKAN,
        .application_name = "generated_scratch_reservation_test",
    };
    gpu::Device device = create_test_device(&runtime_desc)!!;
    defer destroy_test_device(&device)!!;
    if (!gpu::get_device_caps(&device)!!.generated_work) {
        io::printfn(
            "generated-scratch-reservation: NOT EXERCISED (generated work unavailable)",
        );
        return;
    }

    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = GENERATED_SCRATCH_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader = shader,
    };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(
        &device,
        &pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.COMPUTE,
    )!!;
    gpu::GeneratedScratchDesc reservation = {
        .pipeline                = pipeline,
        .kind                    = gpu::GeneratedWorkKind.DISPATCH,
        .max_commands_per_list   = 8,
        .preprocess_buffer_count = 1,
    };
    gpu::reserve_generated_scratch(queue, &reservation)!!;
    assert(@catch(gpu::destroy_pipeline(&device, pipeline))
        == gpu::RESOURCE_IN_USE,
        "a live reservation retains its exact pipeline key");

    vk::VkDeviceState* state = (vk::VkDeviceState*)
        gpu::device_backend_state_ptr(&device);
    vk::MemoryRequirements requirements = vk::generated_scratch_requirements(
        state,
        &reservation,
    )!!;
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    defer discard_test_commands(&commands)!!;
    assert(@catch(gpu::reserve_generated_scratch(queue, &reservation))
        == gpu::RESOURCE_IN_USE,
        "reservation replacement requires a quiescent context");

    vk::CommandRecord* record = vk::resolve_command(&commands)!!;
    vk::RecordingContextState* context = state.recording_contexts.get(
        record.context,
    )!!;
    assert(context.reserved_preprocess.len == 1,
        "one requested preprocess buffer is retained");
    vk::GeneratedPreprocessBuffer* reserved =
        &context.reserved_preprocess[0];
    assert(reserved.reservation_pipeline == pipeline
            && reserved.reservation_kind
                == gpu::GeneratedWorkKind.DISPATCH
            && reserved.reservation_max_commands == 8,
        "reservation retains its exact pipeline, kind, and count key");
    assert(vk::generated_preprocess_compatible(reserved, &requirements),
        "reserved storage satisfies the driver's exact requirements");

    gpu::RecordingWorkCounters before = vk::command_recording_work_counters(
        state,
    );
    vk::GeneratedPreprocessBuffer* acquired =
        vk::acquire_generated_preprocess_buffer(
            state:        state,
            record:       record,
            pipeline:     pipeline,
            kind:         gpu::GeneratedWorkKind.DISPATCH,
            requirements: &requirements,
            max_count:    8,
        )!!;
    gpu::RecordingWorkCounters after = vk::command_recording_work_counters(
        state,
    );
    assert(acquired.address == reserved.address,
        "generated recording consumes the exact reserved buffer");
    assert(after.host_allocations == before.host_allocations
            && after.vma_allocations == before.vma_allocations,
        "reserved acquisition performs no host or VMA allocation");
    assert(@catch(vk::acquire_generated_preprocess_buffer(
        state:        state,
        record:       record,
        pipeline:     pipeline,
        kind:         gpu::GeneratedWorkKind.DISPATCH,
        requirements: &requirements,
        max_count:    9,
    )) == gpu::GENERATED_SCRATCH_EXHAUSTED,
        "work above the reserved command count faults deterministically");
    assert(@catch(vk::acquire_generated_preprocess_buffer(
        state:        state,
        record:       record,
        pipeline:     pipeline,
        kind:         gpu::GeneratedWorkKind.DISPATCH,
        requirements: &requirements,
        max_count:    1,
    )) == gpu::GENERATED_SCRATCH_EXHAUSTED,
        "an in-use reserved buffer cannot be aliased by one command list");

    gpu::discard_commands(&commands)!!;
    gpu::CommandList retry = gpu::begin_commands(queue)!!;
    vk::CommandRecord* retry_record = vk::resolve_command(&retry)!!;
    vk::GeneratedPreprocessBuffer* reacquired =
        vk::acquire_generated_preprocess_buffer(
            state:        state,
            record:       retry_record,
            pipeline:     pipeline,
            kind:         gpu::GeneratedWorkKind.DISPATCH,
            requirements: &requirements,
            max_count:    1,
        )!!;
    assert(reacquired.address == reserved.address,
        "discard returns reserved preprocess storage for retry");
    gpu::discard_commands(&retry)!!;
    gpu::release_generated_scratch(
        queue,
        pipeline,
        gpu::GeneratedWorkKind.DISPATCH,
    )!!;
    io::printfn("generated-scratch-reservation: EXERCISED");
}
