module gpu_test;

import gpu;
import gpu::vk;
import vk;

// Headless (lavapipe): deferred destruction (openspec/changes/deferred-destruction).
// Destroy paths free the public handle immediately but keep the backend
// object alive until the frame that could reference it retires. (a) proves
// destroy-after-submit survives a frame cycle validation- and leak-clean;
// (b) is a whitebox check on the drain queue's own accounting.

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
    return vk::deferred_queue_count((vk::VkDeviceState*)device.backend_state);
}

struct DeferRig {
    gpu::BufferHandle   io;
    gpu::TextureHandle  texture;
    gpu::ShaderHandle   shader;
    gpu::PipelineHandle pipeline;
}

<* A storage buffer, a sampled texture, and a compute pipeline that reads and
   writes the buffer in place — enough surface to submit real GPU work against
   all three before destroying them. *>
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

<* Records a texture layout transition, a compute dispatch over the buffer,
   and submits both without waiting — the caller destroys the referenced
   resources immediately afterward, while the GPU may still be running them. *>
fn void submit_defer_rig(gpu::Device* device, DeferRig* rig) {
    gpu::GpuSpan root_span = gpu::alloc_frame_span(device, ComputeRoot::size, 16)!!;
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
    gpu::end_commands(device, &cmd)!!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(device, &submit)!!;
}

fn void destroy_after_submit_survives_frame_cycle() @test {
    gpu::Device device = make_defer_test_device();

    gpu::begin_frame(&device)!!;
    DeferRig rig = build_defer_rig(&device);
    submit_defer_rig(&device, &rig);

    // Destroy immediately after submit, with no idle wait: the GPU may still
    // be running the dispatch and the barrier against these handles.
    gpu::destroy_texture(&device, rig.texture)!!;
    gpu::destroy_buffer(&device, rig.io)!!;
    gpu::destroy_pipeline(&device, rig.pipeline)!!;
    gpu::destroy_shader(&device, rig.shader)!!;
    assert(deferred_count(&device) > 0, "destroy-after-submit enqueues backend objects rather than freeing them");
    gpu::end_frame(&device)!!;

    // Two more cycles: frames_in_flight == 2, so the second begin_frame waits
    // on the value end_frame signaled right after the destroys, guaranteeing
    // the drain observes retirement.
    gpu::begin_frame(&device)!!;
    gpu::end_frame(&device)!!;
    gpu::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "every deferred entry retires within frames_in_flight cycles");
    gpu::end_frame(&device)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    assert(vk::report_resource_leaks(state) == 0, "destroyed handles leave no leaked slots");
    gpu::destroy_device(&device)!!;
}

fn void deferred_queue_count_rises_then_drains() @test {
    gpu::Device device = make_defer_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = { .size = 256, .usage = { .storage }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;

    gpu::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "queue starts empty");
    gpu::destroy_buffer(&device, buffer)!!;
    assert(deferred_count(&device) == 1, "a mid-frame destroy raises the enqueued count");
    gpu::end_frame(&device)!!;

    gpu::begin_frame(&device)!!;
    gpu::end_frame(&device)!!;
    gpu::begin_frame(&device)!!;
    assert(deferred_count(&device) == 0, "the entry drains once frames_in_flight cycles retire it");
    gpu::end_frame(&device)!!;
}

fn void deferred_destroy_outside_frame_is_immediate() @test {
    gpu::Device device = make_defer_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc desc = { .size = 256, .usage = { .storage }, .memory_kind = gpu::MemoryKind.DEVICE };
    gpu::BufferHandle buffer = gpu::create_buffer(&device, &desc)!!;

    assert(deferred_count(&device) == 0, "queue starts empty");
    gpu::destroy_buffer(&device, buffer)!!;
    assert(deferred_count(&device) == 0, "a destroy outside a frame retires through the opportunistic drain immediately");
}
