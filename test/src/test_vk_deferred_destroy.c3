module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;

// Headless (lavapipe): destroys free the handle immediately but defer the
// backend release until the referencing frame retires; plus a whitebox check
// on the drain queue's accounting.

const char[*] DEFER_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");
const uint DEFER_COUNT = 64;

fn gpu::Device make_defer_test_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_deferred_destroy_test",
    };
    return gpu::create_device(&desc)!!;
}

fn uint deferred_count(gpu::Device* device) {
    return vk::deferred_queue_count((vk::VkDeviceState*)gpu::device_backend_state(*device));
}

fn vk::DeferredStats defer_stats(gpu::Device* device) {
    return vk::deferred_stats((vk::VkDeviceState*)gpu::device_backend_state(*device));
}

struct DeferRig {
    gpu::BufferHandle   io;
    gpu::TextureHandle  texture;
    gpu::ShaderHandle   shader;
    gpu::PipelineHandle pipeline;
}

<* Enough surface to submit real GPU work against before destroying it. *>
fn DeferRig build_defer_rig(gpu::Device* device) {
    gpu::BufferDesc io_desc = {
        .size        = DEFER_COUNT * float::size,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .debug_name  = "defer_io",
    };
    gpu::BufferHandle io = gpu::create_buffer(device, &io_desc)!!;
    gpu::GpuSpan io_span = gpu::get_buffer_span(device, io)!!;
    float* data = (float*)io_span.cpu;
    for (uint i = 0; i < DEFER_COUNT; i++) data[i] = (float)i;

    gpu::TextureDesc tex_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = 8,
        .height     = 8,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .transfer_dst },
        .debug_name = "defer_texture",
    };
    gpu::TextureHandle texture = gpu::create_texture(device, &tex_desc)!!;

    gpu::ShaderDesc shader_desc = {
        .stage = gpu::ShaderStage.COMPUTE, .spirv = DEFER_COMP_SPIRV[..], .entry_point = "main",
    };
    gpu::ShaderHandle shader = gpu::create_shader(device, &shader_desc)!!;

    gpu::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(device, &pipe_desc)!!;

    return { .io = io, .texture = texture, .shader = shader, .pipeline = pipeline };
}

<* Submits without waiting, so the caller destroys resources the GPU may
   still be running against. *>
fn void submit_defer_rig(gpu::FrameToken* frame, gpu::Device* device, DeferRig* rig) {
    gpu::GpuSpan root_span = gpu::alloc_frame_span(frame, ComputeRoot::size, 16)!!;
    ComputeRoot* root = (ComputeRoot*)root_span.cpu;
    root.input_gpu  = gpu::get_buffer_address(device, rig.io)!!;
    root.output_gpu = root.input_gpu;
    root.count      = DEFER_COUNT;

    gpu::CommandList cmd = gpu::begin_commands(device, gpu::QueueKind.GRAPHICS)!!;

    gpu::TextureBarrier tex_bar = {
        .texture       = rig.texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.TRANSFER_DST,
    };
    gpu::cmd_texture_barrier(&cmd, &tex_bar)!!;

    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: rig.pipeline,
        root:     root_span.gpu,
        groups:   { (DEFER_COUNT + ROOT_POINTER_WORKGROUP - 1) / ROOT_POINTER_WORKGROUP, 1, 1 },
    )!!;

    gpu::BufferBarrier buf_bar = {
        .buffer        = rig.io,
        .offset        = 0,
        .size          = DEFER_COUNT * float::size,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &buf_bar)!!;
    gpu::end_commands(&cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(device, &submit)!!;
}

fn void destroy_after_submit_survives_frame_cycle() @test {
    gpu::Device device = make_defer_test_device();

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    DeferRig rig = build_defer_rig(&device);
    submit_defer_rig(&frame_token, &device, &rig);

    gpu::destroy_texture(&device, rig.texture)!!;
    gpu::destroy_buffer(&device, rig.io)!!;
    gpu::destroy_pipeline(&device, rig.pipeline)!!;
    gpu::destroy_shader(&device, rig.shader)!!;
    assert(deferred_count(&device) > 0, "destroy-after-submit enqueues backend objects rather than freeing them");
    gpu::end_frame(&frame_token)!!;

    // Two more cycles: frames_in_flight == 2, so the second begin_frame waits
    // on the value end_frame signaled right after the destroys, guaranteeing
    // the drain observes retirement.
    frame_token = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame_token)!!;
    frame_token = gpu::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "every deferred entry retires within frames_in_flight cycles");
    gpu::end_frame(&frame_token)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(vk::report_resource_leaks(state) == 0, "destroyed handles leave no leaked slots");
    gpu::destroy_device(&device)!!;
}

fn void deferred_queue_count_rises_then_drains() @test {
    gpu::Device device = make_defer_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = { .size = 256, .usage = { .storage }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "queue starts empty");
    gpu::destroy_buffer(&device, buffer)!!;
    assert(deferred_count(&device) == 1, "a mid-frame destroy raises the enqueued count");
    gpu::end_frame(&frame_token)!!;

    frame_token = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame_token)!!;
    frame_token = gpu::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "the entry drains once frames_in_flight cycles retire it");
    gpu::end_frame(&frame_token)!!;
}

fn void deferred_destroy_outside_frame_is_immediate() @test {
    gpu::Device device = make_defer_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = { .size = 256, .usage = { .storage }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;

    assert(deferred_count(&device) == 0, "queue starts empty");
    gpu::destroy_buffer(&device, buffer)!!;
    assert(deferred_count(&device) == 0, "a destroy with no pending off-frame work retires through the opportunistic drain immediately");
}

<* An off-frame submit is sanctioned usage: destroying a buffer it
   referenced must defer backend release, not free under in-flight work. *>
fn void off_frame_destroy_defers_release() @test {
    gpu::Device device = make_defer_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc src_desc = {
        .size = 256, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.DEVICE, .debug_name = "off_frame_src",
    };
    gpu::BufferHandle src = gpu::create_buffer(&device, &src_desc)!!;

    gpu::BufferDesc dst_desc = {
        .size = 256, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE, .debug_name = "off_frame_dst",
    };
    gpu::BufferHandle dst = gpu::create_buffer(&device, &dst_desc)!!;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::BufferCopyDesc copy = { .src = src, .dst = dst, .src_offset = 0, .dst_offset = 0, .size = 256 };
    gpu::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::end_commands(&cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;

    assert(deferred_count(&device) == 0, "queue starts empty");
    gpu::destroy_buffer(&device, dst)!!;
    assert(deferred_count(&device) > 0, "destroying an off-frame submission's target enqueues rather than freeing synchronously");
    gpu::destroy_buffer(&device, src)!!;

    // No end_frame ever covers it; teardown must retire it validation-clean.
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(vk::report_resource_leaks(state) == 0, "off-frame destroy leaves no leaked slots at teardown");
}

const uint STORM_COUNT = 128;

<* Every retire value stays in the future while the frame is open, so tier 1
   (unsubmitted ⇒ unsignalable) must catch the whole storm without asking the
   driver anything. *>
fn void mid_frame_destroy_storm_issues_no_host_queries() @test {
    gpu::Device device = make_defer_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferHandle[STORM_COUNT] buffers;
    for (uint i = 0; i < STORM_COUNT; i++) {
        gpu::BufferDesc desc = { .size = 256, .usage = { .storage }, .memory_kind = gpu::MemoryKind.DEVICE };
        buffers[i] = gpu::create_buffer(&device, &desc)!!;
    }

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    vk::DeferredStats before = defer_stats(&device);
    for (uint i = 0; i < STORM_COUNT; i++) {
        gpu::destroy_buffer(&device, buffers[i])!!;
    }
    vk::DeferredStats after = defer_stats(&device);
    assert(after.host_queries == before.host_queries, "tier 1 skips every mid-frame destroy: zero host timeline queries");
    assert(after.pushed - before.pushed == STORM_COUNT, "every mid-frame destroy still enqueues a deferred entry");
    assert(deferred_count(&device) == STORM_COUNT, "the storm's entries stay queued until the covering frame retires");
    gpu::end_frame(&frame_token)!!;

    frame_token = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame_token)!!;
    frame_token = gpu::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "the storm drains once its covering frame retires");
    gpu::end_frame(&frame_token)!!;
}

<* Once every pushed entry retires through the ordinary tiered path (no
   teardown sweep involved), the counters must reconcile exactly. *>
fn void deferred_stats_reconcile_without_teardown() @test {
    gpu::Device device = make_defer_test_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    gpu::BufferHandle[16] buffers;
    for (uint i = 0; i < 16; i++) {
        gpu::BufferDesc desc = { .size = 256, .usage = { .storage }, .memory_kind = gpu::MemoryKind.DEVICE };
        buffers[i] = gpu::create_buffer(&device, &desc)!!;
    }

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    foreach (buffer : buffers) gpu::destroy_buffer(&device, buffer)!!;
    gpu::end_frame(&frame_token)!!;

    frame_token = gpu::begin_frame(&device)!!;
    gpu::end_frame(&frame_token)!!;
    frame_token = gpu::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "every deferred entry retires within frames_in_flight cycles");
    gpu::end_frame(&frame_token)!!;

    vk::DeferredStats stats = vk::deferred_stats(state);
    assert(stats.pushed == 16, "every destroy pushed exactly one deferred entry");
    assert(stats.pushed == stats.drained, "every pushed entry drains through the tiered path, with nothing left for teardown");
    assert(stats.drain_calls > 0, "the tiered drain path ran");
    assert(stats.host_queries <= stats.drain_calls, "host queries never exceed drain invocations");
}
