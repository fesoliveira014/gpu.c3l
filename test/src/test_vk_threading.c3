module gpu_test;

import gpu;
import gpu::vk;
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
const uint HAMMER_ITERS = 32;

const uint HELPER_HAMMER_THREADS = 2;
const uint HELPER_HAMMER_ITERS = 300;
const usz HELPER_HAMMER_PAYLOAD = 128;
// Tightest arena that avoids the deferred-release queue's growth threshold
// (its cross-thread-allocator gap is gpu.c3l#98, unrelated here).
const usz HELPER_HAMMER_TINY_STAGING = (HELPER_HAMMER_THREADS + 1) * HELPER_HAMMER_PAYLOAD;

const uint HELPER_RACE_ITERS = 100;

fn gpu::Device make_thr_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_threading_test",
    };
    return gpu::create_device(&desc)!!;
}

fn gpu::Device make_thr_device_tiny_staging(usz staging_size) {
    gpu::DeviceDesc desc = {
        .backend            = gpu::BackendKind.VULKAN,
        .enable_validation  = true,
        .frames_in_flight   = 2,
        .staging_arena_size = staging_size,
        .application_name   = "gpu_threading_test",
    };
    return gpu::create_device(&desc)!!;
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
        gpu::BufferDesc buf_desc = {
            .size        = (usz)(256 + (args.seed * 64 + i) % 512),
            .usage       = { .storage, .addressable },
            .memory_kind = gpu::MemoryKind.DEVICE,
        };
        gpu::BufferHandle buffer = gpu::create_buffer(args.device, &buf_desc)!!;

        gpu::TextureDesc tex_desc = {
            .dimension = gpu::TextureDimension.TEX_2D,
            .width     = 4,
            .height    = 4,
            .format    = gpu::Format.RGBA8_UNORM,
            .usage     = { .sampled, .transfer_dst },
        };
        gpu::TextureHandle texture = gpu::create_texture(args.device, &tex_desc)!!;

        if (!buffer.is_valid() || !texture.is_valid()) args.failed = true;
        gpu::destroy_texture(args.device, texture)!!;
        gpu::destroy_buffer(args.device, buffer)!!;
    }
    };
    return 0;
}

fn void concurrent_creation_storm() @test {
    gpu::Device device = make_thr_device();
    defer gpu::destroy_device(&device)!!;

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
    defer gpu::destroy_device(&device)!!;

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

    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    assert(vk::pipeline_cache_entry_count(state) == 1, "same-key race converges to one entry");
    uint entry = state.pipelines.slots[args[0].handle.index].value.entry;
    assert(state.pipeline_cache.entries[entry].refcount == THR_THREADS,
        "refcount equals the number of racing creates");
    foreach (&a : args) gpu::destroy_pipeline(&device, a.handle)!!;
    assert(vk::pipeline_cache_entry_count(state) == 0, "all aliases released the entry");
}

struct SpanArgs {
    gpu::Device* device;
    uint         seed;
    usz[THR_ITERS] offsets;
    usz[THR_ITERS] sizes;
}

fn int span_storm_worker(void* arg) {
    SpanArgs* args = (SpanArgs*)arg;
    for (uint i = 0; i < THR_ITERS; i++) {
        usz size = (usz)(16 + (args.seed * 32 + i * 8) % 128);
        usz align = (usz)1 << (4 + (args.seed + i) % 3);
        gpu::GpuSpan span = gpu::alloc_frame_span(args.device, size, align)!!;
        if (span.offset % align != 0) return 1;
        args.offsets[i] = span.offset;
        args.sizes[i] = size;
    }
    return 0;
}

fn void frame_span_storm_disjoint() @test {
    gpu::Device device = make_thr_device();
    defer gpu::destroy_device(&device)!!;

    gpu::begin_frame(&device)!!;
    SpanArgs[THR_THREADS] args;
    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i].device = &device;
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
    gpu::end_frame(&device)!!;
}

struct RecordArgs {
    gpu::Device*                device;
    gpu::RecordingContextHandle ctx;
    gpu::PipelineHandle         pipeline;
    gpu::GpuSpan                io_span;
    gpu::GpuSpan                root_span;
    gpu::CommandList            list;
}

fn int record_worker(void* arg) {
    RecordArgs* args = (RecordArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
    ComputeRoot* root = (ComputeRoot*)args.root_span.cpu;
    root.input_gpu  = args.io_span.gpu;
    root.output_gpu = (gpu::GpuAddress)((ulong)args.io_span.gpu + THR_ELEMENTS * 4);
    root.count      = THR_ELEMENTS;

    args.list = gpu::begin_commands(args.device, gpu::QueueKind.GRAPHICS, args.ctx)!!;
    gpu::cmd_dispatch(
        commands: &args.list,
        pipeline: args.pipeline,
        root:     args.root_span.gpu,
        groups:   { (THR_ELEMENTS + 63) / 64, 1, 1 },
    )!!;
    gpu::BufferBarrier host_read = {
        .buffer = args.io_span.buffer, .offset = args.io_span.offset, .size = args.io_span.size,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&args.list, &host_read)!!;
    gpu::end_commands(args.device, &args.list)!!;
    };
    return 0;
}

fn void parallel_recording_across_contexts() @test {
    gpu::Device device = make_thr_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = THR_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &comp_sd)!!;
    defer gpu::destroy_shader(&device, shader)!!;
    gpu::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::RecordingContextHandle[THR_THREADS] contexts;
    foreach (&ctx : contexts) *ctx = gpu::create_recording_context(&device)!!;
    defer {
        foreach (ctx : contexts) (void)gpu::destroy_recording_context(&device, ctx);
    }

    gpu::begin_frame(&device)!!;
    RecordArgs[THR_THREADS] args;
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i].device = &device;
        args[i].ctx = contexts[i];
        args[i].pipeline = pipeline;
        args[i].io_span = gpu::alloc_frame_span(&device, THR_ELEMENTS * 4 * 2, 16)!!;
        args[i].root_span = gpu::alloc_frame_span(&device, ComputeRoot::size, 16)!!;
        float* values = (float*)args[i].io_span.cpu;
        for (uint e = 0; e < THR_ELEMENTS; e++) values[e] = (float)(i * 1000 + e);
    }

    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) {
        threads[i].create(&record_worker, &args[i])!!;
    }
    foreach (&t : threads) t.join()!!;

    gpu::CommandList[THR_THREADS] lists;
    for (uint i = 0; i < THR_THREADS; i++) lists[i] = args[i].list;
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    for (uint i = 0; i < THR_THREADS; i++) {
        gpu::invalidate_buffer(&device, args[i].io_span.buffer, 0, 0)!!;
        float* values = (float*)args[i].io_span.cpu;
        float* doubled = &values[THR_ELEMENTS];
        for (uint e = 0; e < THR_ELEMENTS; e++) {
            assert(doubled[e] == (float)(i * 1000 + e) * 2.0f,
                "each thread's recorded dispatch produced its own results");
        }
    }
    gpu::end_frame(&device)!!;
}

struct LayoutRecordArgs {
    gpu::Device*                device;
    gpu::RecordingContextHandle ctx;
    gpu::TextureHandle          texture;
    gpu::CommandList            list;
}

fn int layout_record_worker(void* arg) {
    LayoutRecordArgs* args = (LayoutRecordArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
    args.list = gpu::begin_commands(args.device, gpu::QueueKind.GRAPHICS, args.ctx)!!;
    gpu::TextureBarrier to_dst = {
        .texture       = args.texture,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.TRANSFER_DST,
    };
    gpu::cmd_texture_barrier(&args.list, &to_dst)!!;
    gpu::end_commands(args.device, &args.list)!!;
    };
    return 0;
}

<* Each thread records a layout transition on its own recording context and
   its own texture, then all lists submit together; per-list pending maps
   must stay isolated across concurrent recorders. *>
fn void parallel_recording_disjoint_texture_layouts() @test {
    gpu::Device device = make_thr_device();
    defer gpu::destroy_device(&device)!!;

    gpu::RecordingContextHandle[THR_THREADS] contexts;
    foreach (&ctx : contexts) *ctx = gpu::create_recording_context(&device)!!;
    defer {
        foreach (ctx : contexts) (void)gpu::destroy_recording_context(&device, ctx);
    }

    gpu::TextureHandle[THR_THREADS] textures;
    for (uint i = 0; i < THR_THREADS; i++) {
        gpu::TextureDesc tex_desc = {
            .dimension = gpu::TextureDimension.TEX_2D,
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

    gpu::begin_frame(&device)!!;
    LayoutRecordArgs[THR_THREADS] args;
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i] = { .device = &device, .ctx = contexts[i], .texture = textures[i] };
    }

    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) threads[i].create(&layout_record_worker, &args[i])!!;
    foreach (&t : threads) t.join()!!;

    gpu::CommandList[THR_THREADS] lists;
    for (uint i = 0; i < THR_THREADS; i++) lists[i] = args[i].list;
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&device)!!;
}

struct UploadArgs {
    gpu::Device*     device;
    gpu::BufferHandle buffer;
    char             fill;
}

fn int upload_worker(void* arg) {
    UploadArgs* args = (UploadArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
    char[256] payload;
    foreach (&b : payload) *b = args.fill;
    gpu::upload_buffer_data(
        device:      args.device,
        dst:         args.buffer,
        offset:      0,
        data:        payload[..],
        next_stage:  gpu::Stage.TRANSFER,
        next_hazard: gpu::Hazard.TRANSFER_READ,
    )!!;
    };
    return 0;
}

fn void concurrent_blocking_uploads() @test {
    gpu::Device device = make_thr_device();
    defer gpu::destroy_device(&device)!!;

    UploadArgs[THR_THREADS] args;
    gpu::BufferHandle[THR_THREADS] buffers;
    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) {
        gpu::BufferDesc desc = {
            .size = 256, .usage = { .transfer_src, .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE,
        };
        buffers[i] = gpu::create_buffer(&device, &desc)!!;
        args[i] = { .device = &device, .buffer = buffers[i], .fill = (char)(i + 1) };
    }
    for (uint i = 0; i < THR_THREADS; i++) {
        threads[i].create(&upload_worker, &args[i])!!;
    }
    foreach (&t : threads) t.join()!!;

    char[256] readback;
    for (uint i = 0; i < THR_THREADS; i++) {
        gpu::readback_buffer_data(
            device:      &device,
            src:         buffers[i],
            offset:      0,
            out_data:    readback[..],
            from_stage:  gpu::Stage.TRANSFER,
            from_hazard: gpu::Hazard.TRANSFER_READ,
        )!!;
        foreach (b : readback) assert(b == (char)(i + 1), "every concurrent upload landed intact");
        gpu::destroy_buffer(&device, buffers[i])!!;
    }
}

struct HammerArgs {
    gpu::Device*      device;
    gpu::BufferHandle buffer;
    char              fill;
}

fn int upload_hammer_worker(void* arg) {
    HammerArgs* args = (HammerArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
    char[256] payload;
    foreach (&b : payload) *b = args.fill;
    for (uint i = 0; i < HAMMER_ITERS; i++) {
        gpu::upload_buffer_data(
            device:      args.device,
            dst:         args.buffer,
            offset:      0,
            data:        payload[..],
            next_stage:  gpu::Stage.TRANSFER,
            next_hazard: gpu::Hazard.TRANSFER_READ,
        )!!;
    }
    };
    return 0;
}

<* Repeated blocking uploads per thread widen the race window between the
   timeline increment and the host signal far beyond one call each. *>
fn void concurrent_upload_hammer() @test {
    gpu::Device device = make_thr_device();
    defer gpu::destroy_device(&device)!!;

    HammerArgs[THR_THREADS] args;
    gpu::BufferHandle[THR_THREADS] buffers;
    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) {
        gpu::BufferDesc desc = {
            .size = 256, .usage = { .transfer_src, .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE,
        };
        buffers[i] = gpu::create_buffer(&device, &desc)!!;
        args[i] = { .device = &device, .buffer = buffers[i], .fill = (char)(i + 1) };
    }
    for (uint i = 0; i < THR_THREADS; i++) {
        threads[i].create(&upload_hammer_worker, &args[i])!!;
    }
    foreach (&t : threads) t.join()!!;

    char[256] readback;
    for (uint i = 0; i < THR_THREADS; i++) {
        gpu::readback_buffer_data(
            device:      &device,
            src:         buffers[i],
            offset:      0,
            out_data:    readback[..],
            from_stage:  gpu::Stage.TRANSFER,
            from_hazard: gpu::Hazard.TRANSFER_READ,
        )!!;
        foreach (b : readback) assert(b == (char)(i + 1), "the device survived and buffers stayed isolated");
        gpu::destroy_buffer(&device, buffers[i])!!;
    }
}

struct HelperHammerArgs {
    gpu::Device*      device;
    gpu::BufferHandle buffer;
    uint              seed;
    bool              mismatch;
}

fn int helper_hammer_worker(void* arg) {
    HelperHammerArgs* args = (HelperHammerArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
    char[HELPER_HAMMER_PAYLOAD] payload;
    char[HELPER_HAMMER_PAYLOAD] readback;
    for (uint iter = 0; iter < HELPER_HAMMER_ITERS; iter++) {
        for (uint i = 0; i < HELPER_HAMMER_PAYLOAD; i++) {
            payload[i] = (char)(args.seed * 37 + iter * 3 + i);
        }
        gpu::upload_buffer_data(
            device:      args.device,
            dst:         args.buffer,
            offset:      0,
            data:        payload[..],
            next_stage:  gpu::Stage.TRANSFER,
            next_hazard: gpu::Hazard.TRANSFER_READ,
        )!!;
        gpu::readback_buffer_data(
            device:      args.device,
            src:         args.buffer,
            offset:      0,
            out_data:    readback[..],
            from_stage:  gpu::Stage.TRANSFER,
            from_hazard: gpu::Hazard.TRANSFER_READ,
        )!!;
        for (uint i = 0; i < HELPER_HAMMER_PAYLOAD; i++) {
            if (readback[i] != payload[i]) args.mismatch = true;
        }
    }
    };
    return 0;
}

<* Reuse-pressure coverage, not a guaranteed repro: per-thread varying
   patterns surface staging aliasing as byte mismatches. Full pressure now
   that helpers record in isolation (gpu.c3l#94): a zero-slack arena and 300
   iterations, the parameters that used to reproduce the shared-command-pool
   race before helper recording was isolated onto its own context and mutex. *>
fn void two_helper_tiny_arena_hammer() @test {
    gpu::Device device = make_thr_device_tiny_staging(HELPER_HAMMER_TINY_STAGING);
    defer gpu::destroy_device(&device)!!;

    HelperHammerArgs[HELPER_HAMMER_THREADS] args;
    gpu::BufferHandle[HELPER_HAMMER_THREADS] buffers;
    Thread[HELPER_HAMMER_THREADS] threads;
    for (uint i = 0; i < HELPER_HAMMER_THREADS; i++) {
        gpu::BufferDesc desc = {
            .size = HELPER_HAMMER_PAYLOAD, .usage = { .transfer_src, .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE,
        };
        buffers[i] = gpu::create_buffer(&device, &desc)!!;
        args[i] = { .device = &device, .buffer = buffers[i], .seed = i + 1 };
    }
    for (uint i = 0; i < HELPER_HAMMER_THREADS; i++) {
        threads[i].create(&helper_hammer_worker, &args[i])!!;
    }
    foreach (&t : threads) t.join()!!;

    foreach (&a : args) {
        assert(!a.mismatch, "each helper's readback pattern matched its own upload despite tiny-arena reuse pressure");
    }
    for (uint i = 0; i < HELPER_HAMMER_THREADS; i++) gpu::destroy_buffer(&device, buffers[i])!!;
}

struct HelperRaceArgs {
    gpu::Device*      device;
    gpu::BufferHandle buffer;
}

fn int helper_race_upload_worker(void* arg) {
    HelperRaceArgs* args = (HelperRaceArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
    char[HELPER_HAMMER_PAYLOAD] payload;
    foreach (&b : payload) *b = (char)0xAB;
    for (uint i = 0; i < HELPER_RACE_ITERS; i++) {
        gpu::upload_buffer_data(
            device:      args.device,
            dst:         args.buffer,
            offset:      0,
            data:        payload[..],
            next_stage:  gpu::Stage.TRANSFER,
            next_hazard: gpu::Hazard.TRANSFER_READ,
        )!!;
    }
    };
    return 0;
}

<* Proves helper recording no longer shares a pool with the default context:
   one thread hammers the blocking upload helper while the main thread
   records and submits its own lists on the default context, both
   validation-clean under lavapipe (gpu.c3l#94). *>
fn void blocking_upload_races_default_context_recording() @test {
    gpu::Device device = make_thr_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc upload_desc = {
        .size = HELPER_HAMMER_PAYLOAD, .usage = { .transfer_src, .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle upload_buffer = gpu::create_buffer(&device, &upload_desc)!!;
    defer gpu::destroy_buffer(&device, upload_buffer)!!;

    gpu::BufferDesc copy_desc = {
        .size = 256, .usage = { .transfer_src, .transfer_dst }, .memory_kind = gpu::MemoryKind.DEVICE,
    };
    gpu::BufferHandle copy_src = gpu::create_buffer(&device, &copy_desc)!!;
    defer gpu::destroy_buffer(&device, copy_src)!!;
    gpu::BufferHandle copy_dst = gpu::create_buffer(&device, &copy_desc)!!;
    defer gpu::destroy_buffer(&device, copy_dst)!!;

    HelperRaceArgs race_args = { .device = &device, .buffer = upload_buffer };
    Thread uploader;
    uploader.create(&helper_race_upload_worker, &race_args)!!;

    gpu::BufferCopyDesc copy = { .src = copy_src, .dst = copy_dst, .src_offset = 0, .dst_offset = 0, .size = 256 };
    for (uint i = 0; i < HELPER_RACE_ITERS; i++) {
        gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
        gpu::cmd_copy_buffer(&cmd, &copy)!!;
        gpu::end_commands(&device, &cmd)!!;
        gpu::CommandList[1] lists = { cmd };
        gpu::SubmitDesc submit = { .command_lists = lists[..] };
        gpu::submit(&device, &submit)!!;
        gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    }

    uploader.join()!!;
}

struct BoundaryArgs {
    gpu::Device* device;
    Atomic{int}  entered;
    Atomic{int}  release;
}

fn int boundary_worker(void* arg) {
    BoundaryArgs* args = (BoundaryArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
    // Park inside a Tier S call: a large persistent alloc under the lock is
    // too fast to hold reliably, so hold the phase counter open by looping
    // creates until told to stop.
    while (args.release.load(AtomicOrdering.RELAXED) == 0) {
        gpu::BufferDesc desc = {
            .size = 4096, .usage = { .storage }, .memory_kind = gpu::MemoryKind.DEVICE,
        };
        gpu::BufferHandle buffer = gpu::create_buffer(args.device, &desc)!!;
        args.entered.store(1, AtomicOrdering.RELAXED);
        gpu::destroy_buffer(args.device, buffer)!!;
    }
    };
    return 0;
}

fn void boundary_violation_detected() @test {
    gpu::Device device = make_thr_device();
    defer gpu::destroy_device(&device)!!;

    BoundaryArgs args;
    args.device = &device;
    Thread worker;
    worker.create(&boundary_worker, &args)!!;

    while (args.entered.load(AtomicOrdering.RELAXED) == 0) { }
    // The worker churns Tier S calls; begin_frame must eventually observe one
    // in flight and fault rather than proceed.
    bool tripped = false;
    for (uint attempt = 0; attempt < 10_000 && !tripped; attempt++) {
        if (catch err = gpu::begin_frame(&device)) {
            assert(err == gpu::INVALID_RESOURCE_STATE, "boundary violation faults INVALID_RESOURCE_STATE");
            tripped = true;
        } else if (catch end_err = gpu::end_frame(&device)) {
            assert(end_err == gpu::INVALID_RESOURCE_STATE, "boundary violation faults INVALID_RESOURCE_STATE");
            tripped = true;
        }
    }
    args.release.store(1, AtomicOrdering.RELAXED);
    worker.join()!!;
    assert(tripped, "a Tier S call in flight at begin_frame was detected");
}
