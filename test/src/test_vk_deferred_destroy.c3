module gpu_test;

import gpu::compat @public;
import gpu::compat::vk @public;
import vk;

// Headless (lavapipe): destroys free the handle immediately but defer the
// backend release until the referencing frame retires; plus a whitebox check
// on the drain queue's accounting.

const char[*] DEFER_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");
const uint DEFER_COUNT = 64;

fn gpu::compat::Device make_defer_test_device() {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_deferred_destroy_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

fn uint deferred_count(gpu::compat::Device* device) {
    return vk::deferred_queue_count((vk::VkDeviceState*)gpu::compat::device_backend_state(*device));
}

fn vk::DeferredStats defer_stats(gpu::compat::Device* device) {
    return vk::deferred_stats((vk::VkDeviceState*)gpu::compat::device_backend_state(*device));
}

struct DeferRig {
    gpu::compat::BufferHandle   io;
    gpu::compat::TextureHandle  texture;
    gpu::compat::ShaderHandle   shader;
    gpu::compat::PipelineHandle pipeline;
}

<* Enough surface to submit real GPU work against before destroying it. *>
fn DeferRig build_defer_rig(gpu::compat::Device* device) {
    gpu::compat::BufferDesc io_desc = {
        .size        = DEFER_COUNT * float::size,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::compat::MemoryKind.PERSISTENT_UPLOAD,
        .debug_name  = "defer_io",
    };
    gpu::compat::BufferHandle io = gpu::compat::create_buffer(device, &io_desc)!!;
    gpu::compat::GpuSpan io_span = gpu::compat::get_buffer_span(device, io)!!;
    float* data = (float*)io_span.cpu;
    for (uint i = 0; i < DEFER_COUNT; i++) data[i] = (float)i;

    gpu::compat::TextureDesc tex_desc = {
        .dimension  = gpu::compat::TextureDimension.TEX_2D,
        .width      = 8,
        .height     = 8,
        .format     = gpu::compat::Format.RGBA8_UNORM,
        .usage      = { .sampled, .transfer_dst },
        .debug_name = "defer_texture",
    };
    gpu::compat::TextureHandle texture = gpu::compat::create_texture(device, &tex_desc)!!;

    gpu::compat::ShaderDesc shader_desc = {
        .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = DEFER_COMP_SPIRV[..], .entry_point = "main",
    };
    gpu::compat::ShaderHandle shader = gpu::compat::create_shader(device, &shader_desc)!!;

    gpu::compat::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle pipeline = gpu::compat::create_compute_pipeline(device, &pipe_desc)!!;

    return { .io = io, .texture = texture, .shader = shader, .pipeline = pipeline };
}

<* Submits without waiting, so the caller destroys resources the GPU may
   still be running against. *>
fn void submit_defer_rig(gpu::compat::FrameToken* frame, gpu::compat::Device* device, DeferRig* rig) {
    gpu::compat::GpuSpan root_span = gpu::compat::alloc_frame_span(frame, ComputeRoot::size, 16)!!;
    ComputeRoot* root = (ComputeRoot*)root_span.cpu;
    root.input_gpu  = gpu::compat::get_buffer_address(device, rig.io)!!;
    root.output_gpu = root.input_gpu;
    root.count      = DEFER_COUNT;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::TextureBarrier tex_bar = {
        .texture       = rig.texture,
        .before_stage  = gpu::compat::Stage.HOST,
        .after_stage   = gpu::compat::Stage.TRANSFER,
        .before_hazard = gpu::compat::Hazard.HOST_WRITE,
        .after_hazard  = gpu::compat::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
        .new_layout    = gpu::compat::TextureLayout.TRANSFER_DST,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &tex_bar)!!;

    gpu::compat::cmd_dispatch(
        commands: &cmd,
        pipeline: rig.pipeline,
        root:     root_span.gpu,
        groups:   { (DEFER_COUNT + ROOT_POINTER_WORKGROUP - 1) / ROOT_POINTER_WORKGROUP, 1, 1 },
    )!!;

    gpu::compat::BufferBarrier buf_bar = {
        .buffer        = rig.io,
        .offset        = 0,
        .size          = DEFER_COUNT * float::size,
        .before_stage  = gpu::compat::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::compat::Stage.HOST,
        .before_hazard = gpu::compat::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::compat::Hazard.HOST_READ,
    };
    gpu::compat::cmd_buffer_barrier(&cmd, &buf_bar)!!;
    gpu::compat::end_commands(&cmd)!!;

    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(device, &submit)!!;
}

fn void destroy_after_submit_survives_frame_cycle() @test {
    gpu::compat::Device device = make_defer_test_device();

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    DeferRig rig = build_defer_rig(&device);
    submit_defer_rig(&frame_token, &device, &rig);

    gpu::compat::destroy_texture(&device, rig.texture)!!;
    gpu::compat::destroy_buffer(&device, rig.io)!!;
    gpu::compat::destroy_pipeline(&device, rig.pipeline)!!;
    gpu::compat::destroy_shader(&device, rig.shader)!!;
    assert(deferred_count(&device) > 0, "destroy-after-submit enqueues backend objects rather than freeing them");
    gpu::compat::end_frame(&frame_token)!!;

    // Two more cycles: frames_in_flight == 2, so the second begin_frame waits
    // on the value end_frame signaled right after the destroys, guaranteeing
    // the drain observes retirement.
    frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::end_frame(&frame_token)!!;
    frame_token = gpu::compat::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "every deferred entry retires within frames_in_flight cycles");
    gpu::compat::end_frame(&frame_token)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    assert(vk::report_resource_leaks(state) == 0, "destroyed handles leave no leaked slots");
    gpu::compat::destroy_device(&device)!!;
}

fn void deferred_queue_count_rises_then_drains() @test {
    gpu::compat::Device device = make_defer_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc desc = { .size = 256, .usage = { .storage }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &desc)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "queue starts empty");
    gpu::compat::destroy_buffer(&device, buffer)!!;
    assert(deferred_count(&device) == 1, "a mid-frame destroy raises the enqueued count");
    gpu::compat::end_frame(&frame_token)!!;

    frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::end_frame(&frame_token)!!;
    frame_token = gpu::compat::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "the entry drains once frames_in_flight cycles retire it");
    gpu::compat::end_frame(&frame_token)!!;
}

fn void deferred_destroy_outside_frame_is_immediate() @test {
    gpu::compat::Device device = make_defer_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc desc = { .size = 256, .usage = { .storage }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
    gpu::compat::BufferHandle buffer = gpu::compat::create_buffer(&device, &desc)!!;

    assert(deferred_count(&device) == 0, "queue starts empty");
    gpu::compat::destroy_buffer(&device, buffer)!!;
    assert(deferred_count(&device) == 0, "a destroy with no pending off-frame work retires through the opportunistic drain immediately");
}

<* An off-frame submit is sanctioned usage: destroying a buffer it
   referenced must defer backend release, not free under in-flight work. *>
fn void off_frame_destroy_defers_release() @test {
    gpu::compat::Device device = make_defer_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferDesc src_desc = {
        .size = 256, .usage = { .transfer_src }, .memory_kind = gpu::compat::MemoryKind.DEVICE, .debug_name = "off_frame_src",
    };
    gpu::compat::BufferHandle src = gpu::compat::create_buffer(&device, &src_desc)!!;

    gpu::compat::BufferDesc dst_desc = {
        .size = 256, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.DEVICE, .debug_name = "off_frame_dst",
    };
    gpu::compat::BufferHandle dst = gpu::compat::create_buffer(&device, &dst_desc)!!;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::BufferCopyDesc copy = { .src = src, .dst = dst, .src_offset = 0, .dst_offset = 0, .size = 256 };
    gpu::compat::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::compat::end_commands(&cmd)!!;

    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;

    assert(deferred_count(&device) == 0, "queue starts empty");
    gpu::compat::destroy_buffer(&device, dst)!!;
    assert(deferred_count(&device) > 0, "destroying an off-frame submission's target enqueues rather than freeing synchronously");
    gpu::compat::destroy_buffer(&device, src)!!;

    // No end_frame ever covers it; teardown must retire it validation-clean.
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    assert(vk::report_resource_leaks(state) == 0, "off-frame destroy leaves no leaked slots at teardown");
}

const uint STORM_COUNT = 128;

<* Every retire value stays in the future while the frame is open, so tier 1
   (unsubmitted ⇒ unsignalable) must catch the whole storm without asking the
   driver anything. *>
fn void mid_frame_destroy_storm_issues_no_host_queries() @test {
    gpu::compat::Device device = make_defer_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::BufferHandle[STORM_COUNT] buffers;
    for (uint i = 0; i < STORM_COUNT; i++) {
        gpu::compat::BufferDesc desc = { .size = 256, .usage = { .storage }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
        buffers[i] = gpu::compat::create_buffer(&device, &desc)!!;
    }

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    vk::DeferredStats before = defer_stats(&device);
    for (uint i = 0; i < STORM_COUNT; i++) {
        gpu::compat::destroy_buffer(&device, buffers[i])!!;
    }
    vk::DeferredStats after = defer_stats(&device);
    assert(after.host_queries == before.host_queries, "tier 1 skips every mid-frame destroy: zero host timeline queries");
    assert(after.pushed - before.pushed == STORM_COUNT, "every mid-frame destroy still enqueues a deferred entry");
    assert(deferred_count(&device) == STORM_COUNT, "the storm's entries stay queued until the covering frame retires");
    gpu::compat::end_frame(&frame_token)!!;

    frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::end_frame(&frame_token)!!;
    frame_token = gpu::compat::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "the storm drains once its covering frame retires");
    gpu::compat::end_frame(&frame_token)!!;
}

<* Once every pushed entry retires through the ordinary tiered path (no
   teardown sweep involved), the counters must reconcile exactly. *>
fn void deferred_stats_reconcile_without_teardown() @test {
    gpu::compat::Device device = make_defer_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);

    gpu::compat::BufferHandle[16] buffers;
    for (uint i = 0; i < 16; i++) {
        gpu::compat::BufferDesc desc = { .size = 256, .usage = { .storage }, .memory_kind = gpu::compat::MemoryKind.DEVICE };
        buffers[i] = gpu::compat::create_buffer(&device, &desc)!!;
    }

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    foreach (buffer : buffers) gpu::compat::destroy_buffer(&device, buffer)!!;
    gpu::compat::end_frame(&frame_token)!!;

    frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::end_frame(&frame_token)!!;
    frame_token = gpu::compat::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "every deferred entry retires within frames_in_flight cycles");
    gpu::compat::end_frame(&frame_token)!!;

    vk::DeferredStats stats = vk::deferred_stats(state);
    assert(stats.pushed == 16, "every destroy pushed exactly one deferred entry");
    assert(stats.pushed == stats.drained, "every pushed entry drains through the tiered path, with nothing left for teardown");
    assert(stats.drain_calls > 0, "the tiered drain path ran");
    assert(stats.host_queries <= stats.drain_calls, "host queries never exceed drain invocations");
}
