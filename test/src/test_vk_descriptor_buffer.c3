module gpu_test;

import gpu @public;
import gpu::vk @public;
import std::os::env;
import std::io;

// Forced-DESCRIPTOR_BUFFER coverage. The shader-access end-to-end is gated
// behind GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E=1: Mesa 25.0.7 lavapipe miscompiles
// descriptor-buffer image access (JIT segfault; fixed in later Mesa).

const char[*] DB_WRITE_SPIRV  = $embed("shaders/heap_write.comp.spv");
const char[*] DB_SAMPLE_SPIRV = $embed("shaders/heap_sample.comp.spv");
const char[*] DB_VERT_SPIRV   = $embed("shaders/offscreen.vert.spv");
const char[*] DB_FRAG_SPIRV   = $embed("shaders/offscreen.frag.spv");

fn bool db_e2e_enabled() {
    if (try value = env::tget_var("GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E")) {
        return value == "1";
    }
    return false;
}

fn void descriptor_buffer_device_and_heap() @test {
    gpu::DeviceDesc desc = {
        .backend                     = gpu::BackendKind.VULKAN,
        .enable_validation           = true,
        .frames_in_flight            = 2,
        .descriptor_heap_mode        = gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        .texture_descriptor_capacity = 8,
        .sampler_descriptor_capacity = 4,
        .application_name            = "gpu_descriptor_buffer_test",
    };

    gpu::Device device;
    if (try dev = gpu::create_device_from_desc(&desc)) {
        device = dev;
    } else {
        io::eprintfn("descriptor_buffer_device_and_heap: no capable adapter, skipping");
        return;
    }
    defer destroy_test_device(&device)!!;
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(caps.descriptor_buffer, "the forced descriptor-buffer path is enabled");
    assert(!caps.descriptor_indexing, "the unselected indexing path is disabled");

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureDescriptorDesc[2] batch = {
        { .texture = texture },
        { .texture = texture },
    };
    gpu::TextureIndex[2] indices;
    gpu::create_texture_descriptors(&device, batch[..], indices[..])!!;
    foreach (index : indices) {
        assert(index.is_valid(), "batch descriptor written into the descriptor buffer");
        gpu::destroy_texture_descriptor(&device, index)!!;
    }

    gpu::SamplerDesc sampler_desc = {};
    gpu::SamplerIndex sampler = gpu::create_sampler(&device, &sampler_desc)!!;
    gpu::destroy_sampler(&device, sampler)!!;
}

fn void descriptor_buffer_compute_to_graphics() @test {
    if (!db_e2e_enabled()) {
        io::eprintfn("descriptor_buffer_compute_to_graphics: skipped (set GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E=1; known older lavapipe descriptor-buffer bug)");
        return;
    }

    const uint TEX_SIZE = 16;
    const uint TARGET_SIZE = 32;
    const usz TARGET_BYTES = TARGET_SIZE * TARGET_SIZE * 4;
    const usz VERTEX_BYTES = 3 * 16;
    gpu::DeviceDesc device_desc = {
        .backend              = gpu::BackendKind.VULKAN,
        .enable_validation    = true,
        .frames_in_flight     = 2,
        .descriptor_heap_mode = gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        .application_name     = "gpu_descriptor_buffer_compute_to_graphics",
    };
    gpu::Device? maybe_device = gpu::create_device_from_desc(&device_desc);
    if (catch err = maybe_device) {
        assert(err == gpu::UNSUPPORTED_FEATURE,
            "forced descriptor-buffer device creation failed unexpectedly: %s", err);
        io::eprintfn("descriptor_buffer_compute_to_graphics: no capable adapter, skipping");
        return;
    }
    gpu::Device device = maybe_device;
    defer destroy_test_device(&device)!!;
    gpu::vk::VkDeviceState* state = (gpu::vk::VkDeviceState*)gpu::device_backend_state(device);
    io::eprintfn(
        "descriptor_buffer_compute_to_graphics: graphics_family=%d compute_family=%d topology=%s",
        state.graphics_family,
        state.compute_family,
        state.graphics_family == state.compute_family ? "same-family" : "distinct-family",
    );

    gpu::TextureDesc shared_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = TEX_SIZE,
        .height    = TEX_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureHandle shared_texture = gpu::create_texture(&device, &shared_desc)!!;
    defer gpu::destroy_texture(&device, shared_texture)!!;
    gpu::TextureIndex texture_index = gpu::create_texture_descriptor(&device, shared_texture, null)!!;
    defer gpu::destroy_texture_descriptor(&device, texture_index)!!;

    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex sampler_index = gpu::create_sampler(&device, &sampler_desc)!!;
    defer gpu::destroy_sampler(&device, sampler_index)!!;

    gpu::BufferDesc compute_root_desc = {
        .size        = WriteRoot::size,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .access      = { .graphics, .compute, .transfer },
    };
    gpu::BufferHandle compute_root_buffer = gpu::create_buffer(&device, &compute_root_desc)!!;
    defer gpu::destroy_buffer(&device, compute_root_buffer)!!;
    gpu::GpuSpan compute_root_span = gpu::get_buffer_span(&device, compute_root_buffer)!!;
    WriteRoot* compute_root = (WriteRoot*)compute_root_span.cpu;
    *compute_root = {
        .texture_index = texture_index,
        .width         = TEX_SIZE,
        .height        = TEX_SIZE,
    };
    gpu::flush_buffer(
        device: &device,
        buffer: compute_root_buffer,
        offset: 0,
        size:   0,
    )!!;

    gpu::BufferDesc compute_output_desc = {
        .size        = TEX_SIZE * TEX_SIZE * 16,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.READBACK,
        .access      = { .graphics, .compute, .transfer },
    };
    gpu::BufferHandle compute_output = gpu::create_buffer(&device, &compute_output_desc)!!;
    defer gpu::destroy_buffer(&device, compute_output)!!;
    gpu::GpuSpan compute_output_span = gpu::get_buffer_span(&device, compute_output)!!;

    gpu::BufferDesc sample_root_desc = {
        .size        = SampleRoot::size,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .access      = { .graphics, .compute, .transfer },
    };
    gpu::BufferHandle sample_root_buffer = gpu::create_buffer(&device, &sample_root_desc)!!;
    defer gpu::destroy_buffer(&device, sample_root_buffer)!!;
    gpu::GpuSpan sample_root_span = gpu::get_buffer_span(&device, sample_root_buffer)!!;
    SampleRoot* sample_root = (SampleRoot*)sample_root_span.cpu;
    *sample_root = {
        .output_gpu    = compute_output_span.gpu,
        .texture_index = texture_index,
        .sampler_index = sampler_index,
        .width         = TEX_SIZE,
        .height        = TEX_SIZE,
    };
    gpu::flush_buffer(
        device: &device,
        buffer: sample_root_buffer,
        offset: 0,
        size:   0,
    )!!;

    gpu::BufferDesc graphics_root_desc = {
        .size        = VERTEX_BYTES + FragRoot::size,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .access      = { .graphics, .compute, .transfer },
    };
    gpu::BufferHandle graphics_root_buffer = gpu::create_buffer(&device, &graphics_root_desc)!!;
    defer gpu::destroy_buffer(&device, graphics_root_buffer)!!;
    gpu::GpuSpan graphics_root_span = gpu::get_buffer_span(&device, graphics_root_buffer)!!;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    float* vertices = (float*)graphics_root_span.cpu;
    for (uint i = 0; i < vertex_data.len; i++) vertices[i] = vertex_data[i];
    FragRoot* fragment_root = (FragRoot*)((char*)graphics_root_span.cpu + VERTEX_BYTES);
    *fragment_root = {
        .texture_index = texture_index,
        .sampler_index = sampler_index,
    };
    gpu::flush_buffer(
        device: &device,
        buffer: graphics_root_buffer,
        offset: 0,
        size:   0,
    )!!;

    gpu::TextureDesc target_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = TARGET_SIZE,
        .height    = TARGET_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &target_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;
    gpu::BufferDesc readback_desc = {
        .size        = TARGET_BYTES,
        .usage       = { .transfer_dst },
        .memory_kind = gpu::MemoryKind.READBACK,
        .access      = { .graphics, .compute, .transfer },
    };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &readback_desc)!!;
    defer gpu::destroy_buffer(&device, readback)!!;

    gpu::ShaderDesc write_shader_desc = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DB_WRITE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle write_shader = gpu::create_shader(&device, &write_shader_desc)!!;
    defer gpu::destroy_shader(&device, write_shader)!!;
    gpu::ComputePipelineDesc compute_pipeline_desc = { .shader = write_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle compute_pipeline = gpu::create_compute_pipeline(&device, &compute_pipeline_desc)!!;
    gpu::ShaderDesc sample_shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = DB_SAMPLE_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderHandle sample_shader = gpu::create_shader(&device, &sample_shader_desc)!!;
    defer gpu::destroy_shader(&device, sample_shader)!!;
    gpu::ComputePipelineDesc sample_pipeline_desc = {
        .shader             = sample_shader,
        .push_constant_size = gpu::RootPush::size,
    };
    gpu::PipelineHandle sample_pipeline = gpu::create_compute_pipeline(&device, &sample_pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, sample_pipeline)!!;
    defer gpu::destroy_pipeline(&device, compute_pipeline)!!;

    gpu::ShaderDesc vertex_shader_desc = { .stage = gpu::ShaderStage.VERTEX, .spirv = DB_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle vertex_shader = gpu::create_shader(&device, &vertex_shader_desc)!!;
    defer gpu::destroy_shader(&device, vertex_shader)!!;
    gpu::ShaderDesc fragment_shader_desc = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = DB_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle fragment_shader = gpu::create_shader(&device, &fragment_shader_desc)!!;
    defer gpu::destroy_shader(&device, fragment_shader)!!;
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc graphics_pipeline_desc = {
        .vertex_shader   = vertex_shader,
        .fragment_shader = fragment_shader,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = color_formats[..],
    };
    gpu::PipelineHandle graphics_pipeline = gpu::create_graphics_pipeline(&device, &graphics_pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, graphics_pipeline)!!;

    gpu::SemaphoreDesc semaphore_desc = {};
    gpu::SemaphoreHandle dependency = gpu::create_semaphore(&device, &semaphore_desc)!!;
    defer gpu::destroy_semaphore(&device, dependency)!!;
    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;

    gpu::CommandList compute_commands = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;
    defer discard_test_commands(&compute_commands)!!;
    gpu::TextureBarrier to_general = {
        .texture       = shared_texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&compute_commands, &to_general)!!;
    const uint GROUPS = (TEX_SIZE + 7) / 8;
    gpu::cmd_dispatch(
        commands: &compute_commands,
        pipeline: compute_pipeline,
        root:     compute_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!!;
    gpu::TextureBarrier write_to_sample = {
        .texture       = shared_texture,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
        .old_layout    = gpu::TextureLayout.GENERAL,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&compute_commands, &write_to_sample)!!;
    gpu::cmd_dispatch(
        commands: &compute_commands,
        pipeline: sample_pipeline,
        root:     sample_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!!;
    gpu::BufferBarrier compute_output_to_host = {
        .buffer        = compute_output,
        .offset        = 0,
        .size          = TEX_SIZE * TEX_SIZE * 16,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&compute_commands, &compute_output_to_host)!!;
    gpu::end_commands(&compute_commands)!!;
    gpu::CommandList[1] compute_lists = { compute_commands };
    gpu::SemaphoreSignal[1] compute_signals = {{
        .semaphore = dependency,
        .value     = (gpu::SemaphoreValue)1,
        .stage     = gpu::Stage.COMPUTE_SHADER,
    }};
    gpu::SubmitDesc compute_submit = { .command_lists = compute_lists[..], .signals = compute_signals[..] };
    gpu::submit(&device, &compute_submit)!!;

    gpu::CommandList graphics_commands = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    defer discard_test_commands(&graphics_commands)!!;
    gpu::TextureBarrier target_to_attachment = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&graphics_commands, &target_to_attachment)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .texture  = target,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear    = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = TARGET_SIZE, .height = TARGET_SIZE };
    gpu::cmd_begin_render_pass(&graphics_commands, &pass)!!;
    gpu::GpuAddress fragment_root_gpu = (gpu::GpuAddress)((ulong)graphics_root_span.gpu + VERTEX_BYTES);
    gpu::cmd_draw(
        commands:       &graphics_commands,
        pipeline:       graphics_pipeline,
        vertex_root:    graphics_root_span.gpu,
        fragment_root:  fragment_root_gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&graphics_commands)!!;
    gpu::TextureBarrier target_to_source = {
        .texture       = target,
        .before_stage  = gpu::Stage.COLOR_ATTACHMENT,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.COLOR_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(&graphics_commands, &target_to_source)!!;
    gpu::GpuSpan readback_span = gpu::get_buffer_span(&device, readback)!!;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = readback_span };
    gpu::cmd_copy_texture_to_buffer(&graphics_commands, &copy)!!;
    gpu::BufferBarrier host_read = {
        .buffer = readback, .offset = 0, .size = TARGET_BYTES,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&graphics_commands, &host_read)!!;
    gpu::end_commands(&graphics_commands)!!;
    gpu::CommandList[1] graphics_lists = { graphics_commands };
    gpu::SemaphoreWait[1] graphics_waits = {{
        .semaphore = dependency,
        .value     = (gpu::SemaphoreValue)1,
        .stage     = gpu::Stage.FRAGMENT_SHADER,
    }};
    gpu::SubmitDesc graphics_submit = { .command_lists = graphics_lists[..], .waits = graphics_waits[..] };
    gpu::submit(&device, &graphics_submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::invalidate_buffer(
        device: &device,
        buffer: readback,
        offset: 0,
        size:   0,
    )!!;
    gpu::invalidate_buffer(
        device: &device,
        buffer: compute_output,
        offset: 0,
        size:   0,
    )!!;

    float* compute_texels = (float*)compute_output_span.cpu;
    for (uint y = 0; y < TEX_SIZE; y++) {
        for (uint x = 0; x < TEX_SIZE; x++) {
            float* texel = &compute_texels[(y * TEX_SIZE + x) * 4];
            assert(quantize_unorm8(texel[0]) == x, "compute sampled red matches");
            assert(quantize_unorm8(texel[1]) == y, "compute sampled green matches");
            assert(quantize_unorm8(texel[2]) == (x ^ y), "compute sampled blue matches");
        }
    }

    char* pixels = (char*)readback_span.cpu;
    char* center = &pixels[(TARGET_SIZE / 2 * TARGET_SIZE + TARGET_SIZE / 2) * 4];
    assert(center[0] == (char)8, "center red comes from the compute-written descriptor");
    assert(center[1] == (char)8, "center green comes from the compute-written descriptor");
    assert(center[2] == (char)0, "center blue comes from the compute-written descriptor");
    assert(center[3] == (char)255, "center alpha remains opaque");
    assert(pixels[0] == (char)8 && pixels[1] == (char)16 && pixels[2] == (char)32 && pixels[3] == (char)255,
        "corner remains the clear color");
    if (state.graphics_family == state.compute_family) {
        io::eprintfn("descriptor_buffer_compute_to_graphics: PASS same-family; distinct-family=N/A (%d)", state.graphics_family);
    } else {
        io::eprintfn("descriptor_buffer_compute_to_graphics: PASS distinct-family graphics=%d compute=%d",
            state.graphics_family, state.compute_family);
    }
    gpu::end_frame(&frame_token)!!;
}

fn void descriptor_buffer_write_then_sample() @test {
    if (!db_e2e_enabled()) {
        io::eprintfn("descriptor_buffer_write_then_sample: skipped (set GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E=1; known Mesa 25.0.7 lavapipe descriptor-buffer bug)");
        return;
    }

    const uint TEX = 16;
    gpu::DeviceDesc device_desc = {
        .backend              = gpu::BackendKind.VULKAN,
        .enable_validation    = true,
        .frames_in_flight     = 2,
        .descriptor_heap_mode = gpu::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        .application_name     = "gpu_descriptor_buffer_e2e",
    };
    gpu::Device device;
    if (try dev = gpu::create_device_from_desc(&device_desc)) {
        device = dev;
    } else {
        io::eprintfn("descriptor_buffer_write_then_sample: no capable adapter, skipping");
        return;
    }
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = TEX,
        .height    = TEX,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureIndex tex_index = gpu::create_texture_descriptor(&device, texture, null)!!;
    defer gpu::destroy_texture_descriptor(&device, tex_index)!!;

    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex smp_index = gpu::create_sampler(&device, &sampler_desc)!!;
    defer gpu::destroy_sampler(&device, smp_index)!!;

    gpu::BufferDesc out_desc = {
        .size        = TEX * TEX * 16,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.READBACK,
        .access      = { .graphics, .compute, .transfer },
    };
    gpu::BufferHandle output = gpu::create_buffer(&device, &out_desc)!!;
    defer gpu::destroy_buffer(&device, output)!!;

    gpu::ShaderDesc write_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DB_WRITE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle write_shader = gpu::create_shader(&device, &write_sd)!!;
    defer gpu::destroy_shader(&device, write_shader)!!;
    gpu::ShaderDesc sample_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DB_SAMPLE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle sample_shader = gpu::create_shader(&device, &sample_sd)!!;
    defer gpu::destroy_shader(&device, sample_shader)!!;

    gpu::ComputePipelineDesc write_pd = { .shader = write_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle write_pipe = gpu::create_compute_pipeline(&device, &write_pd)!!;
    defer gpu::destroy_pipeline(&device, write_pipe)!!;
    gpu::ComputePipelineDesc sample_pd = { .shader = sample_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle sample_pipe = gpu::create_compute_pipeline(&device, &sample_pd)!!;
    defer gpu::destroy_pipeline(&device, sample_pipe)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;

    gpu::GpuSpan write_root_span = gpu::alloc_frame_span(&frame_token, WriteRoot::size, 16)!!;
    WriteRoot* write_root = (WriteRoot*)write_root_span.cpu;
    write_root.texture_index = tex_index;
    write_root.width         = TEX;
    write_root.height        = TEX;

    gpu::GpuSpan sample_root_span = gpu::alloc_frame_span(&frame_token, SampleRoot::size, 16)!!;
    SampleRoot* sample_root = (SampleRoot*)sample_root_span.cpu;
    sample_root.output_gpu    = gpu::get_buffer_address(&device, output)!!;
    sample_root.texture_index = tex_index;
    sample_root.sampler_index = smp_index;
    sample_root.width         = TEX;
    sample_root.height        = TEX;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureBarrier to_general = {
        .texture       = texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&cmd, &to_general)!!;

    const uint GROUPS = (TEX + 7) / 8;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: write_pipe,
        root:     write_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::TextureBarrier write_to_sample = {
        .texture       = texture,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.COMPUTE_SHADER,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
        .old_layout    = gpu::TextureLayout.GENERAL,
        .new_layout    = gpu::TextureLayout.GENERAL,
    };
    gpu::cmd_texture_barrier(&cmd, &write_to_sample)!!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: sample_pipe,
        root:     sample_root_span.gpu,
        groups:   { GROUPS, GROUPS, 1 },
    )!!;

    gpu::BufferBarrier out_bar = {
        .buffer = output, .offset = 0, .size = TEX * TEX * 16,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &out_bar)!!;
    gpu::end_commands(&cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;

    gpu::GpuSpan out_span = gpu::get_buffer_span(&device, output)!!;
    float* texels = (float*)out_span.cpu;
    for (uint y = 0; y < TEX; y++) {
        for (uint x = 0; x < TEX; x++) {
            float* texel = &texels[(y * TEX + x) * 4];
            assert(quantize_unorm8(texel[0]) == x, "red channel matches on the buffer path");
            assert(quantize_unorm8(texel[1]) == y, "green channel matches on the buffer path");
            assert(quantize_unorm8(texel[2]) == (x ^ y), "blue channel matches on the buffer path");
        }
    }
    gpu::end_frame(&frame_token)!!;
}
