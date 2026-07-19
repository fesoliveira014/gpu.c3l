module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::thread;
import std::atomic;
import std::core::mem::allocators;

// Headless (lavapipe): the tiered threading model. Storm tests maximize
// detection surface (handle uniqueness, disjoint spans, cache convergence);
// parallel recording runs one context per thread into disjoint targets.

const char[*] THR_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

const uint THR_THREADS = 4;
const uint THR_ITERS = 32;
const uint THR_ELEMENTS = 64;
const usz THR_TRANSFER_BYTES = 256;

struct TransferRecordArgs {
    gpu::Device*                device;
    uint                        worker_index;
    OwnedSpan                   upload;
    OwnedSpan                   intermediate;
    OwnedSpan                   readback;
    gpu::ExecutableCommandList  list;
}

fn gpu::Device make_thr_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_threading_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
}

struct StormArgs {
    gpu::Device* device;
    uint         seed;
    bool         failed;
}

fn int creation_storm_worker(void* arg) {
    StormArgs* args = (StormArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
    for (uint i = 0; i < THR_ITERS; i++) {
        OwnedSpan buffer = allocate_owned_span(
            device:       args.device,
            size:         (usz)(256 + (args.seed * 64 + i) % 512),
            memory_class: gpu::MemoryClass.GPU_PRIVATE,
            access:       { .graphics, .compute, .transfer },
            debug_name:   "creation_storm_buffer",
        )!!;

        gpu::TextureDesc tex_desc = {
            .dimension = gpu::TextureDimension.TEX_2D,
            .access    = { .graphics, .compute, .transfer },
            .width     = 4,
            .height    = 4,
            .format    = gpu::Format.RGBA8_UNORM,
            .usage     = { .sampled, .transfer_dst },
        };
        gpu::TextureHandle texture = gpu::create_texture(args.device, &tex_desc)!!;

        gpu::ShaderDesc shader_desc = { .stage = gpu::ShaderStage.COMPUTE, .spirv = THR_COMP_SPIRV[..], .entry_point = "main" };
        gpu::ShaderHandle shader = gpu::create_shader(args.device, &shader_desc)!!;

        gpu::SamplerDesc sampler_desc = {
            .min_filter = gpu::Filter.NEAREST,
            .mag_filter = gpu::Filter.NEAREST,
            .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
            .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
            .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
        };
        gpu::SamplerIndex sampler = gpu::create_sampler(args.device, &sampler_desc)!!;

        if (!buffer.allocation.is_valid() || !texture.is_valid() || !shader.is_valid() || !sampler.is_valid()) {
            args.failed = true;
        }
        gpu::destroy_sampler(args.device, sampler)!!;
        gpu::destroy_shader(args.device, shader)!!;
        gpu::destroy_texture(args.device, texture)!!;
        free_owned_span(args.device, &buffer)!!;
    }
    };
    return 0;
}

fn void concurrent_creation_storm() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;

    StormArgs[THR_THREADS] args;
    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i] = { .device = &device, .seed = i };
        threads[i].create(&creation_storm_worker, &args[i])!!;
    }
    foreach (&t : threads) t.join()!!;
    foreach (&a : args) assert(!a.failed, "every storm create yielded a valid handle");
}

struct PipelineRaceArgs {
    gpu::Device*        device;
    gpu::ShaderHandle   shader;
    gpu::PipelineHandle handle;
}

fn int pipeline_race_worker(void* arg) {
    PipelineRaceArgs* args = (PipelineRaceArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        gpu::ComputePipelineDesc desc = { .shader = args.shader, .push_constant_size = gpu::RootPush::size };
        args.handle = gpu::create_compute_pipeline(args.device, &desc)!!;
    };
    return 0;
}

fn void same_descriptor_pipeline_race() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = THR_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &comp_sd)!!;
    defer gpu::destroy_shader(&device, shader)!!;

    PipelineRaceArgs[THR_THREADS] args;
    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i] = { .device = &device, .shader = shader };
        threads[i].create(&pipeline_race_worker, &args[i])!!;
    }
    foreach (&t : threads) t.join()!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(vk::pipeline_cache_entry_count(state) == 1, "same-key race converges to one entry");
    uint entry = state.pipelines.slots[args[0].handle.index].value.entry;
    assert(state.pipeline_cache.entries[entry].refcount == THR_THREADS,
        "refcount equals the number of racing creates");
    foreach (&a : args) gpu::destroy_pipeline(&device, a.handle)!!;
    assert(vk::pipeline_cache_entry_count(state) == 0, "all aliases released the entry");
}

struct ShaderPinArgs {
    vk::VkDeviceState* state;
    gpu::ShaderHandle  shader;
    Atomic{uint}       entered;
    Atomic{uint}       release;
}

fn int shader_pin_worker(void* arg) {
    ShaderPinArgs* args = (ShaderPinArgs*)arg;
    vk::ShaderPipelineReference reference =
        vk::retain_shader_for_pipeline(args.state, args.shader)!!;
    args.entered.store(1, AtomicOrdering.RELEASE);
    while (args.release.load(AtomicOrdering.ACQUIRE) == 0) {
        thread::yield();
    }
    vk::release_shader_for_pipeline(args.state, reference);
    return 0;
}

fn void shader_destroy_rejects_concurrent_pipeline_build_pin() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = THR_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &desc)!!;
    defer (void)gpu::destroy_shader(&device, shader);
    ShaderPinArgs args = {
        .state  = (vk::VkDeviceState*)gpu::device_backend_state(device),
        .shader = shader,
    };
    Thread worker;
    worker.create(&shader_pin_worker, &args)!!;
    while (args.entered.load(AtomicOrdering.ACQUIRE) == 0) {
        thread::yield();
    }

    assert(@catch(gpu::destroy_shader(&device, shader))
        == gpu::RESOURCE_IN_USE,
        "an active pipeline build preserves its shader module");
    args.release.store(1, AtomicOrdering.RELEASE);
    worker.join()!!;

    gpu::destroy_shader(&device, shader)!!;
    shader = {};
}

struct SpanArgs {
    gpu::FrameToken* frame;
    uint         seed;
    usz[THR_ITERS] offsets;
    usz[THR_ITERS] sizes;
}

fn int span_storm_worker(void* arg) {
    SpanArgs* args = (SpanArgs*)arg;
    for (uint i = 0; i < THR_ITERS; i++) {
        usz size = (usz)(16 + (args.seed * 32 + i * 8) % 128);
        usz align = (usz)1 << (4 + (args.seed + i) % 3);
        gpu::GpuSpan span = gpu::alloc_frame_span(args.frame, size, align)!!;
        if (span.offset % align != 0) return 1;
        args.offsets[i] = span.offset;
        args.sizes[i] = size;
    }
    return 0;
}

fn void frame_span_storm_disjoint() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    SpanArgs[THR_THREADS] args;
    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i].frame = &frame_token;
        args[i].seed = i;
        threads[i].create(&span_storm_worker, &args[i])!!;
    }
    foreach (&t : threads) {
        int code = t.join()!!;
        assert(code == 0, "every span honored its alignment");
    }

    for (uint a = 0; a < THR_THREADS; a++) {
        for (uint i = 0; i < THR_ITERS; i++) {
            for (uint b = 0; b < THR_THREADS; b++) {
                for (uint j = 0; j < THR_ITERS; j++) {
                    if (a == b && i == j) continue;
                    usz a_start = args[a].offsets[i];
                    usz a_end = a_start + args[a].sizes[i];
                    usz b_start = args[b].offsets[j];
                    usz b_end = b_start + args[b].sizes[j];
                    assert(a_end <= b_start || b_end <= a_start, "concurrent spans never overlap");
                }
            }
        }
    }
    gpu::end_frame(&frame_token)!!;
}

struct ReuseWorkerArgs {
    gpu::Device*                  device;
    gpu::ExecutableCommandList[2] executables;
}

fn int reuse_recording_pool_worker(void* raw_args) {
    ReuseWorkerArgs* args = (ReuseWorkerArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        gpu::Queue queue = gpu::get_queue(
            args.device,
            gpu::QueueKind.GRAPHICS,
        )!!;
        for (uint i = 0; i < 2; i++) {
            gpu::CommandList commands = gpu::begin_commands(queue)!!;
            args.executables[i] = gpu::end_commands(&commands)!!;
        }
    };
    return 0;
}

fn void worker_thread_reuses_private_recording_pool() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;

    ReuseWorkerArgs args = { .device = &device };
    Thread worker;
    worker.create(&reuse_recording_pool_worker, &args)!!;
    assert(worker.join()!! == 0, "worker records both lists");

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(state.recording_contexts.next_new == 1,
        "the worker reuses one recording pool");

    gpu::ExecutableCommandList[2] lists = args.executables;
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
}

fn char transfer_pattern(uint worker_index, usz byte_index) {
    return (char)((byte_index * 17 + (usz)(worker_index + 1) * 31) % 251);
}

fn int explicit_transfer_worker(void* raw_args) {
    TransferRecordArgs* args = (TransferRecordArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        char[] upload_mapping = test_span_mapping(
            args.device,
            args.upload.span,
        );
        for (usz i = 0; i < upload_mapping.len; i++) {
            upload_mapping[i] = transfer_pattern(args.worker_index, i);
        }
        gpu::flush_mapped_span(args.device, args.upload.span)!!;

        gpu::Queue queue = gpu::get_queue(
            args.device,
            gpu::QueueKind.TRANSFER,
        )!!;
        gpu::CommandList commands = gpu::begin_commands(queue)!!;
        defer (void)discard_test_commands(&commands);
        gpu::BufferCopyDesc upload_copy = {
            .src = args.upload.span,
            .dst = args.intermediate.span,
        };
        gpu::cmd_copy_buffer(&commands, &upload_copy)!!;
        gpu::BufferBarrier before_readback = {
            .span          = args.intermediate.span,
            .before_stage  = gpu::Stage.TRANSFER,
            .after_stage   = gpu::Stage.TRANSFER,
            .before_hazard = gpu::Hazard.TRANSFER_WRITE,
            .after_hazard  = gpu::Hazard.TRANSFER_READ,
        };
        gpu::cmd_buffer_barrier(&commands, &before_readback)!!;
        gpu::BufferCopyDesc readback_copy = {
            .src = args.intermediate.span,
            .dst = args.readback.span,
        };
        gpu::cmd_copy_buffer(&commands, &readback_copy)!!;
        gpu::BufferBarrier host_read = {
            .span          = args.readback.span,
            .before_stage  = gpu::Stage.TRANSFER,
            .after_stage   = gpu::Stage.HOST,
            .before_hazard = gpu::Hazard.TRANSFER_WRITE,
            .after_hazard  = gpu::Hazard.HOST_READ,
        };
        gpu::cmd_buffer_barrier(&commands, &host_read)!!;
        args.list = gpu::end_commands(&commands)!!;
    };
    return 0;
}

fn void concurrent_caller_owned_explicit_transfers_preserve_bytes() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;

    TransferRecordArgs[THR_THREADS] args = {};
    defer {
        foreach (&worker_args : args) {
            (void)free_owned_span(&device, &worker_args.readback);
            (void)free_owned_span(&device, &worker_args.intermediate);
            (void)free_owned_span(&device, &worker_args.upload);
        }
    }
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i].device = &device;
        args[i].worker_index = i;
        args[i].upload = allocate_owned_span(
            device:       &device,
            size:         THR_TRANSFER_BYTES,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .transfer },
            debug_name:   "concurrent_transfer_upload",
        )!!;
        args[i].intermediate = allocate_owned_span(
            device:       &device,
            size:         THR_TRANSFER_BYTES,
            memory_class: gpu::MemoryClass.GPU_PRIVATE,
            access:       { .transfer },
            debug_name:   "concurrent_transfer_intermediate",
        )!!;
        args[i].readback = allocate_owned_span(
            device:       &device,
            size:         THR_TRANSFER_BYTES,
            memory_class: gpu::MemoryClass.CPU_READ,
            access:       { .transfer },
            debug_name:   "concurrent_transfer_readback",
        )!!;
    }

    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) {
        threads[i].create(&explicit_transfer_worker, &args[i])!!;
    }
    foreach (&worker : threads) {
        assert(worker.join()!! == 0, "every worker records its explicit transfer chain");
    }

    gpu::ExecutableCommandList[THR_THREADS] lists;
    for (uint i = 0; i < THR_THREADS; i++) lists[i] = args[i].list;
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.TRANSFER,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;

    for (uint i = 0; i < THR_THREADS; i++) {
        gpu::invalidate_mapped_span(&device, args[i].readback.span)!!;
        char[] readback_mapping = test_span_mapping(
            &device,
            args[i].readback.span,
        );
        for (usz byte_index = 0; byte_index < readback_mapping.len; byte_index++) {
            assert(readback_mapping[byte_index] == transfer_pattern(i, byte_index),
                "each worker's complete readback matches its distinct upload pattern");
        }
    }

    foreach (&worker_args : args) {
        free_owned_span(&device, &worker_args.readback)!!;
        free_owned_span(&device, &worker_args.intermediate)!!;
        free_owned_span(&device, &worker_args.upload)!!;
    }
}

struct RecordArgs {
    gpu::Device*                device;
    gpu::PipelineHandle         pipeline;
    gpu::GpuSpan                io_span;
    gpu::GpuSpan                root_span;
    gpu::ExecutableCommandList  list;
}

fn int record_worker(void* arg) {
    RecordArgs* args = (RecordArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {

        ComputeRoot* root = (ComputeRoot*)test_span_mapping(args.device, args.root_span).ptr;
        root.input_gpu = test_span_address(args.device, args.io_span);
        root.output_gpu = (gpu::GpuAddress)(
            (ulong)test_span_address(args.device, args.io_span) + THR_ELEMENTS * 4
        );
        root.count = THR_ELEMENTS;

        gpu::CommandList commands = gpu::begin_commands(
            gpu::get_queue(args.device, gpu::QueueKind.GRAPHICS)!!,
        )!!;
        gpu::cmd_dispatch(
            commands: &commands,
            pipeline: args.pipeline,
            root:     test_span_address(args.device, args.root_span),
            groups:   {
                (THR_ELEMENTS + 63) / 64,
                1,
                1,
            },
        )!!;
        gpu::BufferBarrier host_read = {
            .span          = args.io_span,
            .before_stage  = gpu::Stage.COMPUTE_SHADER,
            .after_stage   = gpu::Stage.HOST,
            .before_hazard = gpu::Hazard.SHADER_WRITE,
            .after_hazard  = gpu::Hazard.HOST_READ,
        };
        gpu::cmd_buffer_barrier(&commands, &host_read)!!;
        args.list = gpu::end_commands(&commands)!!;
    };
    return 0;
}

fn void parallel_recording_across_workers() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = THR_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &comp_sd)!!;
    defer gpu::destroy_shader(&device, shader)!!;
    gpu::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;


    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    RecordArgs[THR_THREADS] args;
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i].device = &device;

        args[i].pipeline = pipeline;
        args[i].io_span = gpu::alloc_frame_span(&frame_token, THR_ELEMENTS * 4 * 2, 16)!!;
        args[i].root_span = gpu::alloc_frame_span(&frame_token, ComputeRoot::size, 16)!!;
        float* values = (float*)test_span_mapping(&device, args[i].io_span).ptr;
        for (uint e = 0; e < THR_ELEMENTS; e++) values[e] = (float)(i * 1000 + e);
    }

    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) {
        threads[i].create(&record_worker, &args[i])!!;
    }
    foreach (&t : threads) t.join()!!;

    gpu::ExecutableCommandList[THR_THREADS] lists;
    for (uint i = 0; i < THR_THREADS; i++) lists[i] = args[i].list;
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;

    for (uint i = 0; i < THR_THREADS; i++) {

        float* values = (float*)test_span_mapping(&device, args[i].io_span).ptr;
        float* doubled = &values[THR_ELEMENTS];
        for (uint e = 0; e < THR_ELEMENTS; e++) {
            assert(doubled[e] == (float)(i * 1000 + e) * 2.0f,
                "each thread's recorded dispatch produced its own results");
        }
    }
    gpu::end_frame(&frame_token)!!;
}

struct LayoutRecordArgs {
    gpu::Device*               device;
    gpu::TextureHandle         texture;
    gpu::ExecutableCommandList list;
}

fn int layout_record_worker(void* arg) {
    LayoutRecordArgs* args = (LayoutRecordArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        gpu::CommandList commands = gpu::begin_commands(
            gpu::get_queue(args.device, gpu::QueueKind.GRAPHICS)!!,
        )!!;
        gpu::TextureBarrier to_dst = {
            .texture       = args.texture,
            .before_stage  = gpu::Stage.HOST,
            .after_stage   = gpu::Stage.TRANSFER,
            .before_hazard = gpu::Hazard.HOST_WRITE,
            .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
            .old_layout    = gpu::TextureLayout.UNDEFINED,
            .new_layout    = gpu::TextureLayout.TRANSFER_DST,
        };
        gpu::cmd_texture_barrier(&commands, &to_dst)!!;
        args.list = gpu::end_commands(&commands)!!;
    };
    return 0;
}
<* Concurrent layout records keep their pending state isolated. *>
fn void parallel_recording_disjoint_texture_layouts() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;


    gpu::TextureHandle[THR_THREADS] textures;
    for (uint i = 0; i < THR_THREADS; i++) {
        gpu::TextureDesc tex_desc = {
            .dimension = gpu::TextureDimension.TEX_2D,
            .access    = { .graphics, .compute, .transfer },
            .width     = 4,
            .height    = 4,
            .format    = gpu::Format.RGBA8_UNORM,
            .usage     = { .transfer_src, .transfer_dst },
        };
        textures[i] = gpu::create_texture(&device, &tex_desc)!!;
    }
    defer {
        foreach (t : textures) (void)gpu::destroy_texture(&device, t);
    }

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    LayoutRecordArgs[THR_THREADS] args;
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i] = { .device = &device, .texture = textures[i] };
    }

    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) threads[i].create(&layout_record_worker, &args[i])!!;
    foreach (&t : threads) t.join()!!;

    gpu::ExecutableCommandList[THR_THREADS] lists;
    for (uint i = 0; i < THR_THREADS; i++) lists[i] = args[i].list;
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    submit_test_queue(&device, gpu::QueueKind.GRAPHICS, &submit)!!;
    wait_test_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;
}

const uint RELEASE_THREADS = 4;
const uint RELEASES_PER_THREAD = 24;

struct ReleaseArgs {
    gpu::Device* device;
    OwnedSpan[RELEASES_PER_THREAD] buffers;
}

fn int release_worker(void* arg) {
    ReleaseArgs* args = (ReleaseArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        foreach (&buffer : args.buffers) {
            free_owned_span(args.device, buffer)!!;
        }
    };
    return 0;
}

<* Release independent allocations concurrently while a frame is active. *>
fn void concurrent_allocation_release_from_worker_threads() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;

    ReleaseArgs[RELEASE_THREADS] args;
    for (uint thread_index = 0; thread_index < RELEASE_THREADS; thread_index++) {
        args[thread_index].device = &device;
        for (uint i = 0; i < RELEASES_PER_THREAD; i++) {
            args[thread_index].buffers[i] = allocate_owned_span(
                device:       &device,
                size:         256,
                memory_class: gpu::MemoryClass.GPU_PRIVATE,
                access:       { .graphics, .compute, .transfer },
                debug_name:   "worker_release_buffer",
            )!!;
        }
    }

    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    Thread[RELEASE_THREADS] threads;
    for (uint i = 0; i < RELEASE_THREADS; i++) {
        threads[i].create(&release_worker, &args[i])!!;
    }
    foreach (&worker : threads) worker.join()!!;

    foreach (&thread_args : args) {
        foreach (&buffer : thread_args.buffers) {
            assert(!buffer.allocation.is_valid() && buffer.span.owner == 0,
                "worker release clears every owned allocation and span");
        }
    }
    gpu::end_frame(&frame_token)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(vk::report_resource_leaks(state) == 0,
        "concurrent allocation release leaves no live resource slots");
}

struct BoundaryArgs {
    gpu::Device* device;
    Atomic{int}  entered;
    Atomic{int}  release;
}

fn int boundary_worker(void* arg) {
    BoundaryArgs* args = (BoundaryArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
    // One allocation under the lock is too fast to hold reliably, so churn
    // allocations until told to stop.
    while (args.release.load(AtomicOrdering.RELAXED) == 0) {
        OwnedSpan buffer = allocate_owned_span(
            device:       args.device,
            size:         4096,
            memory_class: gpu::MemoryClass.GPU_PRIVATE,
            access:       { .graphics, .compute, .transfer },
            debug_name:   "boundary_buffer",
        )!!;
        args.entered.store(1, AtomicOrdering.RELAXED);
        free_owned_span(args.device, &buffer)!!;
    }
    };
    return 0;
}

fn void boundary_violation_detected() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;

    BoundaryArgs args;
    args.device = &device;
    Thread worker;
    worker.create(&boundary_worker, &args)!!;

    while (args.entered.load(AtomicOrdering.RELAXED) == 0) { }
    // The worker churns Tier S calls; begin_frame must eventually observe one
    // in flight and fault rather than proceed.
    bool tripped = false;
    for (uint attempt = 0; attempt < 10_000 && !tripped; attempt++) {
        gpu::FrameToken? frame_result = gpu::begin_frame(&device);
        if (catch err = frame_result) {
            assert(err == gpu::INVALID_RESOURCE_STATE, "boundary violation faults INVALID_RESOURCE_STATE");
            tripped = true;
        } else {
            gpu::FrameToken frame_token = frame_result;
            if (catch end_err = gpu::end_frame(&frame_token)) {
                assert(end_err == gpu::INVALID_RESOURCE_STATE, "boundary violation faults INVALID_RESOURCE_STATE");
                tripped = true;
            }
        }
    }
    args.release.store(1, AtomicOrdering.RELAXED);
    worker.join()!!;
    assert(tripped, "a Tier S call in flight at begin_frame was detected");
}
