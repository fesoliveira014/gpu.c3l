module gpu_test;

import gpu;
import gpu::vk;
import vk;
import std::core::mem;

// Headless (lavapipe): leak reporters, debug-name storage, stats counts, and
// command labels. The leak test intentionally prints leak lines to stderr.

const char[*] DBG_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

fn gpu::Device make_dbg_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_debug_test",
    };
    return gpu::create_device(&desc)!!;
}

fn void leak_reporters_count_leaks() @test {
    gpu::Device device = make_dbg_device();
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    gpu::BufferDesc buf_desc = {
        .size        = 256,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "leaky_buffer",
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &buf_desc)!!;

    gpu::TextureDesc tex_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = 8,
        .height     = 8,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled },
        .debug_name = "leaky_texture",
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &tex_desc)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = DBG_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &comp_sd)!!;
    gpu::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = 8 };
    gpu::PipelineHandle first_alias = gpu::create_compute_pipeline(&device, &pipe_desc)!!;
    gpu::PipelineHandle second_alias = gpu::create_compute_pipeline(&device, &pipe_desc)!!;

    gpu::SemaphoreDesc sem_desc = { .debug_name = "leaky_semaphore" };
    gpu::SemaphoreHandle semaphore = gpu::create_semaphore(&device, &sem_desc)!!;
    gpu::RecordingContextHandle ctx = gpu::create_recording_context(&device)!!;

    assert(vk::report_buffer_leaks(state) == 1, "one leaked buffer counted");
    assert(vk::report_texture_leaks(state) == 1, "one leaked texture counted");
    assert(vk::report_pipeline_leaks(state) == 1, "one leaked cache entry counted");
    assert(vk::report_shader_leaks(state) == 1, "one leaked shader counted");
    assert(vk::report_semaphore_leaks(state) == 1, "one leaked semaphore counted");
    assert(vk::report_context_leaks(state) == 1, "one leaked context counted (default and helper exempt)");

    // Slot data feeding the report lines.
    vk::TextureSlot* tex_slot = &state.textures.slots[texture.index].value;
    assert(tex_slot.allocation_size > 0, "texture leak line carries the allocation size");

    gpu::destroy_recording_context(&device, ctx)!!;
    gpu::destroy_semaphore(&device, semaphore)!!;
    gpu::destroy_pipeline(&device, first_alias)!!;
    gpu::destroy_pipeline(&device, second_alias)!!;
    gpu::destroy_shader(&device, shader)!!;
    gpu::destroy_texture(&device, texture)!!;
    gpu::destroy_buffer(&device, buffer)!!;

    assert(vk::report_resource_leaks(state) == 0, "clean state reports zero leaks");
    gpu::destroy_device(&device)!!;
}

fn void debug_name_copy_is_independent() @test {
    gpu::Device device = make_dbg_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    char[16] volatile_name;
    char[7] source = { 'v', 'o', 'l', 'a', 't', 'i', 'l' };
    for (uint i = 0; i < 7; i++) volatile_name[i] = source[i];
    volatile_name[7] = 0;

    gpu::BufferDesc desc = {
        .size        = 64,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = (ZString)&volatile_name[0],
    };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;
    defer gpu::destroy_buffer(&device, buffer)!!;

    foreach (&b : volatile_name) *b = 'X';

    char* stored = &state.buffers.slots[buffer.index].value.debug_name[0];
    assert(stored[0] == 'v' && stored[6] == 'l' && stored[7] == 0,
        "the slot holds its own copy, unaffected by the caller's buffer");
}

fn void stats_counts_populated() @test {
    gpu::Device device = make_dbg_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferHandle[3] buffers;
    for (uint i = 0; i < 3; i++) {
        gpu::BufferDesc desc = { .size = 128, .usage = { .storage }, .memory_kind = gpu::MemoryKind.DEVICE };
        buffers[i] = gpu::create_buffer(&device, &desc)!!;
    }
    gpu::TextureHandle[2] textures;
    for (uint i = 0; i < 2; i++) {
        gpu::TextureDesc desc = {
            .dimension = gpu::TextureDimension.TEX_2D,
            .width     = 4,
            .height    = 4,
            .format    = gpu::Format.RGBA8_UNORM,
            .usage     = { .sampled },
        };
        textures[i] = gpu::create_texture(&device, &desc)!!;
    }

    gpu::MemoryStats stats = gpu::get_memory_stats(&device)!!;
    assert(stats.buffer_count == 3, "buffer count reflects live buffers");
    assert(stats.texture_count == 2, "texture count reflects live textures");
    assert(stats.heap_count > 0 && stats.heaps[0].budget > 0, "stats include a VMA budget");

    foreach (t : textures) gpu::destroy_texture(&device, t)!!;
    foreach (b : buffers) gpu::destroy_buffer(&device, b)!!;
    stats = gpu::get_memory_stats(&device)!!;
    assert(stats.buffer_count == 0 && stats.texture_count == 0, "counts drop to zero after destroy");
}

fn void labels_record_validation_clean() @test {
    gpu::Device device = make_dbg_device();
    defer gpu::destroy_device(&device)!!;

    gpu::TextureDesc rt_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 8,
        .height    = 8,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::cmd_begin_label(&cmd, "frame_setup")!!;
    gpu::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::cmd_end_label(&cmd)!!;

    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = 8, .height = 8 };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_begin_label(&cmd, "main_pass", { 0.2f, 0.8f, 0.2f, 1.0f })!!;
    gpu::cmd_begin_label(&cmd, "nested")!!;
    gpu::cmd_end_label(&cmd)!!;
    gpu::cmd_end_label(&cmd)!!;
    gpu::cmd_end_render_pass(&cmd)!!;

    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&device)!!;

    fault ended = @catch(gpu::cmd_begin_label(&cmd, "too_late"));
    assert(ended == gpu::COMMAND_RECORDING_ERROR, "labels on an ended list fault COMMAND_RECORDING_ERROR");
}

fn void destroy_reports_leaks_before_sweeping() @test {
    gpu::Device device = make_dbg_device();
    gpu::BufferDesc desc = {
        .size        = 128,
        .usage       = { .storage },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name  = "canary_buffer",
    };
    (void)gpu::create_buffer(&device, &desc)!!;
    // Teardown must print the canary's leak line (visible in test stderr)
    // and still destroy everything without faulting.
    gpu::destroy_device(&device)!!;
}
