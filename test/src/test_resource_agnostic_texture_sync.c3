module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;

const uint RESOURCE_AGNOSTIC_SYNC_TEXEL_COUNT = TEX_SIZE * TEX_SIZE * 4;

enum DependencyObservation {
    ORDERED,
    MISSING,
}

struct BarrierEmitCapture {
    uint               calls;
    vk::MemoryBarrier2 native;
}

struct TextureSyncWorkloadObservation {
    DependencyObservation                  dependency;
    uint                                   memory_barrier_captures;
    vk::MemoryBarrier2                     memory_barrier;
    vk::BarrierWorkCounters                barriers;
    gpu::internal::RecordingWorkCounters recording;
}

struct TextureSyncOutput {
    uint[RESOURCE_AGNOSTIC_SYNC_TEXEL_COUNT] texels;
}

struct TextureSyncFixture {
    gpu::Device           device;
    gpu::PipelineHandle   write_pipeline;
    gpu::PipelineHandle   sample_pipeline;
    gpu::SamplerIndex     sampler;
    gpu::CommandAllocator allocator;
}

fn void capture_barrier_emit(
    void* user_data,
    vk::MemoryBarrier2* native,
) {
    BarrierEmitCapture* capture = (BarrierEmitCapture*)user_data;
    capture.calls++;
    capture.native = *native;
}

fn vk::Result create_injected_selected_native_device(
    void* user_data,
    vk::PhysicalDevice physical_device,
    vk::DeviceCreateInfo* requested_info,
    vk::Device* device,
) {
    (void)user_data;
    vk::PhysicalDeviceFeatures2* features =
        (vk::PhysicalDeviceFeatures2*)requested_info.p_next;
    vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR* unified =
        (vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR*)features.p_next;
    assert(
        unified.s_type
            == vk::StructureType.PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR
            && unified.unified_image_layouts == vk::TRUE
            && unified.unified_image_layouts_video == vk::FALSE,
        "the injected native boundary receives the exact selected feature"
    );
    features.p_next = unified.p_next;
    defer features.p_next = unified;

    ZString[8] extensions;
    uint extension_count;
    for (uint i = 0; i < requested_info.enabled_extension_count; i++) {
        ZString extension =
            (ZString)requested_info.pp_enabled_extension_names[i];
        if (extension.str_view()
            == vk::RESOURCE_AGNOSTIC_TEXTURE_SYNC_EXTENSION.str_view()) {
            continue;
        }
        extensions[extension_count++] = extension;
    }
    vk::DeviceCreateInfo native_info = *requested_info;
    native_info = native_info.set_enabled_extension_names(
        extensions[:extension_count],
    );
    return vk::create_device(
        physical_device: physical_device,
        p_create_info:   &native_info,
        p_allocator:     null,
        p_device:        device,
    );
}

fn gpu::Device? create_injected_selected_device(ZString application_name) {
    gpu::RuntimeDesc desc = {
        .backend                  = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name         = application_name,
    };
    gpu::Runtime runtime = gpu::create_runtime(&desc)!;
    defer (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::DeviceRequest request = gpu::request_resource_agnostic_texture_sync(
        gpu::strict_device_request(),
    )!;

    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!;
        gpu::internal::RuntimeData* runtime_data =
            gpu::internal::resolve_adapter(&adapter);
        vk::VkRuntimeState* runtime_state =
            (vk::VkRuntimeState*)runtime_data.backend_state;
        vk::VkAdapterState* adapter_state =
            &runtime_state.adapters[adapter.adapter_index];
        adapter_state.resource_agnostic_texture_sync = {
            .extension             = true,
            .unified_image_layouts = true,
        };
        if (!gpu::supports_device_request(&adapter, &request)!.supported) {
            continue;
        }

        vk::DeviceCreationProbe probe = {
            .resource_agnostic_texture_sync_override = true,
            .resource_agnostic_texture_sync_extension_available = true,
            .resource_agnostic_texture_sync_feature_available = true,
            .native_device_create_hook = &create_injected_selected_native_device,
        };
        gpu::Runtime retained_runtime = runtime;
        gpu::internal::runtime_retain_device(&retained_runtime)!;
        gpu::Device? created = vk::create_vk_device_with_probe(
            adapter:          &adapter,
            request:          &request,
            retained_runtime: &retained_runtime,
            probe:            &probe,
        );
        if (catch creation_fault = created) {
            gpu::internal::runtime_release_device(&retained_runtime)!;
            if (creation_fault == gpu::UNSUPPORTED_FEATURE) continue;
            return creation_fault~;
        }
        assert(
            probe.resource_agnostic_texture_sync_requested
                && probe.resource_agnostic_texture_sync_selected
                && probe.resource_agnostic_texture_sync_extension_enabled
                && probe.resource_agnostic_texture_sync_feature_struct_count
                    == 1,
            "the injected selected device traverses the exact requested path"
        );
        assert(
            gpu::get_device_caps(&created)!.resource_agnostic_texture_sync,
            "selected capability publishes only after native creation succeeds"
        );
        return created;
    }
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void init_texture_sync_fixture(
    TextureSyncFixture* fixture,
    ZString application_name,
    bool selected,
) {
    gpu::RuntimeDesc desc = {
        .backend                  = gpu::BackendKind.VULKAN,
        .contract_validation      = gpu::ContractValidation.FULL,
        .track_resource_lifetimes = true,
        .enable_vulkan_validation = true,
        .application_name         = application_name,
    };
    fixture.device = selected
        ? create_injected_selected_device(application_name)!!
        : create_test_device(&desc)!!;
    assert(
        gpu::get_device_caps(&fixture.device)!!
            .resource_agnostic_texture_sync == selected,
        "the fixture consumes the published capability state"
    );
    gpu::ShaderDesc write_shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = HEAP_WRITE_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode write_shader =
        gpu::prepare_shader_code(&write_shader_desc)!!;
    gpu::ComputePipelineDesc write_pipeline_desc = {
        .shader = write_shader,
    };
    fixture.write_pipeline = gpu::create_compute_pipeline(
        &fixture.device,
        &write_pipeline_desc,
    )!!;
    gpu::ShaderDesc sample_shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = HEAP_SAMPLE_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode sample_shader =
        gpu::prepare_shader_code(&sample_shader_desc)!!;
    gpu::ComputePipelineDesc sample_pipeline_desc = {
        .shader = sample_shader,
    };
    fixture.sample_pipeline = gpu::create_compute_pipeline(
        &fixture.device,
        &sample_pipeline_desc,
    )!!;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    fixture.sampler = gpu::intern_sampler(
        &fixture.device,
        &sampler_desc,
    )!!;
    gpu::Queue queue = gpu::get_queue(
        &fixture.device,
        gpu::QueueKind.COMPUTE,
    )!!;
    fixture.allocator = gpu::create_command_allocator(
        &fixture.device,
        queue,
    )!!;
}

fn void destroy_texture_sync_fixture(TextureSyncFixture* fixture) {
    gpu::destroy_command_allocator(&fixture.allocator)!!;
    gpu::destroy_pipeline(&fixture.device, fixture.sample_pipeline)!!;
    gpu::destroy_pipeline(&fixture.device, fixture.write_pipeline)!!;
    destroy_test_device(&fixture.device)!!;
}

fn void copy_texture_sync_output(
    gpu::Device* device,
    gpu::GpuSpan output,
    TextureSyncOutput* result,
) {
    gpu::invalidate_mapped_span(device, output)!!;
    float* texels = (float*)test_span_mapping(device, output).ptr;
    for (uint i = 0; i < result.texels.len; i++) {
        result.texels[i] = quantize_unorm8(texels[i]);
    }
}

fn void run_texture_sync_workload(
    gpu::Device* device,
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle write_pipeline,
    gpu::PipelineHandle sample_pipeline,
    gpu::SamplerIndex sampler,
    bool selected,
    bool omit_dependency,
    TextureSyncOutput* output_result,
    TextureSyncWorkloadObservation* observation,
) {
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::internal::device_backend_state_ptr(device);
    assert(
        state.caps.resource_agnostic_texture_sync == selected,
        "the workload mode matches successful device selection"
    );

    gpu::TextureDesc texture_desc = {
        .access    = { .compute },
        .width      = TEX_SIZE,
        .height     = TEX_SIZE,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .storage },
        .debug_name = selected
            ? "selected_texture_sync_target"
            : "classic_texture_sync_target",
    };
    gpu::TextureHandle texture =
        gpu::create_texture(device, &texture_desc)!!;
    defer gpu::destroy_texture(device, texture)!!;
    gpu::TextureView texture_view =
        gpu::create_texture_view(device, texture, null)!!;
    defer gpu::destroy_texture_view(device, texture_view)!!;

    OwnedSpan output = allocate_owned_span(
        device:       device,
        size:         TEX_SIZE * TEX_SIZE * 16,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .compute },
        debug_name:   "texture_sync_output",
    )!!;
    defer free_owned_span(device, &output)!!;
    OwnedSpan write_root_owned = allocate_owned_span(
        device:       device,
        size:         WriteRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "texture_sync_write_root",
    )!!;
    defer free_owned_span(device, &write_root_owned)!!;
    OwnedSpan sample_root_owned = allocate_owned_span(
        device:       device,
        size:         SampleRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .compute },
        debug_name:   "texture_sync_sample_root",
    )!!;
    defer free_owned_span(device, &sample_root_owned)!!;

    WriteRoot* write_root = (WriteRoot*)test_span_mapping(
        device,
        write_root_owned.span,
    ).ptr;
    write_root.texture_index = texture_view.index;
    write_root.width = TEX_SIZE;
    write_root.height = TEX_SIZE;
    SampleRoot* sample_root = (SampleRoot*)test_span_mapping(
        device,
        sample_root_owned.span,
    ).ptr;
    sample_root.output_gpu = test_span_address(device, output.span);
    sample_root.texture_index = texture_view.index;
    sample_root.sampler_index = sampler;
    sample_root.width = TEX_SIZE;
    sample_root.height = TEX_SIZE;
    gpu::flush_mapped_span(device, write_root_owned.span)!!;
    gpu::flush_mapped_span(device, sample_root_owned.span)!!;

    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
    defer discard_test_commands(&commands)!!;
    gpu::TextureBarrier initialize = gpu::texture_transition(
        texture: texture,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   gpu::storage_at({ .compute }, { .write }),
    )!!;
    gpu::cmd_texture_barrier(&commands, &initialize)!!;

    const uint GROUPS = (TEX_SIZE + 7) / 8;
    bind_test_compute_pipeline(&commands, write_pipeline);
    gpu::cmd_dispatch(
        commands: &commands,
        root:     test_span_address(device, write_root_owned.span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    BarrierEmitCapture capture;
    vk::set_barrier_emit_hook(state, &capture_barrier_emit, &capture);
    defer vk::set_barrier_emit_hook(state, null, null);
    vk::reset_barrier_work_counters(state);
    vk::reset_command_recording_work_counters(state);
    gpu::internal::RecordingWorkCounters recording_before =
        vk::command_recording_work_counters(state);
    if (!omit_dependency) {
        if (selected) {
            gpu::Barrier dependency = {
                .before = { .compute },
                .after  = { .compute },
            };
            gpu::cmd_barrier(&commands, &dependency)!!;
        } else {
            gpu::TextureBarrier dependency = gpu::texture_transition(
                texture: texture,
                before:  gpu::storage_at({ .compute }, { .write }),
                after:   gpu::sampled_at({ .compute }),
            )!!;
            gpu::cmd_texture_barrier(&commands, &dependency)!!;
        }
    }
    gpu::internal::RecordingWorkCounters recording_after =
        vk::command_recording_work_counters(state);
    observation.recording = vk::command_recording_work_delta(
        recording_after,
        recording_before,
    );
    observation.barriers = vk::barrier_work_counters(state);
    observation.memory_barrier_captures = capture.calls;
    observation.memory_barrier = capture.native;
    observation.dependency =
        observation.barriers.native_memory_emissions == 0
            && observation.barriers.native_image_emissions == 0
                ? DependencyObservation.MISSING
                : DependencyObservation.ORDERED;

    bind_test_compute_pipeline(&commands, sample_pipeline);
    gpu::cmd_dispatch(
        commands: &commands,
        root:     test_span_address(device, sample_root_owned.span),
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    if (omit_dependency) {
        gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
        gpu::discard_executable_commands(&executable)!!;
        return;
    }

    gpu::Barrier host_dependency = {
        .before = { .compute },
        .after  = { .host },
    };
    gpu::cmd_barrier(&commands, &host_dependency)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        device,
        gpu::QueueKind.COMPUTE,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
    copy_texture_sync_output(device, output.span, output_result);
}

fn void selected_texture_sync_workload_matches_classic_without_texture_work()
    @test {
    TextureSyncFixture classic_fixture;
    init_texture_sync_fixture(
        &classic_fixture,
        "resource_agnostic_texture_sync_test",
        false,
    );
    defer destroy_texture_sync_fixture(&classic_fixture);
    TextureSyncFixture selected_fixture;
    init_texture_sync_fixture(
        &selected_fixture,
        "resource_agnostic_texture_sync_test",
        true,
    );
    defer destroy_texture_sync_fixture(&selected_fixture);

    TextureSyncOutput classic_output;
    TextureSyncWorkloadObservation classic;
    run_texture_sync_workload(
        device:           &classic_fixture.device,
        allocator:        &classic_fixture.allocator,
        write_pipeline:   classic_fixture.write_pipeline,
        sample_pipeline:  classic_fixture.sample_pipeline,
        sampler:          classic_fixture.sampler,
        selected:         false,
        omit_dependency:  false,
        output_result:    &classic_output,
        observation:      &classic,
    );
    TextureSyncOutput selected_output;
    TextureSyncWorkloadObservation selected;
    run_texture_sync_workload(
        device:           &selected_fixture.device,
        allocator:        &selected_fixture.allocator,
        write_pipeline:   selected_fixture.write_pipeline,
        sample_pipeline:  selected_fixture.sample_pipeline,
        sampler:          selected_fixture.sampler,
        selected:         true,
        omit_dependency:  false,
        output_result:    &selected_output,
        observation:      &selected,
    );

    assert(classic.dependency == DependencyObservation.ORDERED);
    assert(classic.barriers.native_memory_emissions == 0);
    assert(classic.barriers.native_image_emissions == 1);
    assert(selected.dependency == DependencyObservation.ORDERED);
    assert(selected.barriers.native_memory_emissions == 1,
        "the selected ordinary dependency emits one memory barrier");
    assert(selected.memory_barrier_captures == 1,
        "native observation captures the selected memory barrier once");
    assert(selected.memory_barrier.src_stage_mask
            == vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT
        && selected.memory_barrier.dst_stage_mask
            == vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT,
        "the selected dependency retains the compute execution scopes");
    assert(selected.memory_barrier.src_access_mask
            == vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT
        && selected.memory_barrier.dst_access_mask
            == (vk::AccessFlagBits2.ACCESS_2_SHADER_READ_BIT
                | vk::AccessFlagBits2.ACCESS_2_SHADER_WRITE_BIT),
        "the selected dependency retains the global shader memory scopes");
    assert(selected.barriers.native_image_emissions == 0,
        "the selected ordinary dependency emits no image barrier");
    assert(selected.barriers.texture_helper_calls == 0,
        "the selected ordinary dependency performs no texture inference");
    assert(selected.barriers.texture_handle_resolutions == 0,
        "the selected ordinary dependency resolves no texture");
    assert(selected.recording.host_allocations == 0,
        "the selected ordinary dependency performs no host allocation");
    assert(selected_output.texels == classic_output.texels,
        "selected global synchronization preserves classic workload output");
}

fn void missing_selected_texture_dependency_is_observed_without_submission()
    @test {
    TextureSyncFixture fixture;
    init_texture_sync_fixture(
        &fixture,
        "missing_texture_sync_dependency_test",
        true,
    );
    defer destroy_texture_sync_fixture(&fixture);
    TextureSyncOutput ignored_output;
    TextureSyncWorkloadObservation observation;
    run_texture_sync_workload(
        device:           &fixture.device,
        allocator:        &fixture.allocator,
        write_pipeline:   fixture.write_pipeline,
        sample_pipeline:  fixture.sample_pipeline,
        sampler:          fixture.sampler,
        selected:         true,
        omit_dependency:  true,
        output_result:    &ignored_output,
        observation:      &observation,
    );

    assert(observation.dependency == DependencyObservation.MISSING,
        "deterministic native-barrier observation reports missing ordering");
    assert(observation.barriers.native_memory_emissions == 0);
    assert(observation.barriers.native_image_emissions == 0,
        "the library inserts no hidden texture barrier");
    assert(observation.barriers.texture_helper_calls == 0,
        "missing ordering triggers no hidden texture helper");
    assert(observation.barriers.texture_handle_resolutions == 0,
        "missing ordering triggers no hidden texture inference");
}
