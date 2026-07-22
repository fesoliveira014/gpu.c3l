module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::thread;
import std::atomic;
import std::core::mem;
import std::core::mem::allocators;

// Headless (lavapipe): the tiered threading model. Storm tests maximize
// detection surface (handle uniqueness, allocation release, cache convergence);
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

struct ReuseWorkerArgs {
    gpu::Device*                  device;
    gpu::ExecutableCommandList[2] executables;
}

struct ScratchWorkerArgs {
    gpu::Device*        device;
    gpu::PipelineHandle pipeline;
    Atomic{bool}*       start;
    bool                failed;
}

fn gpu::Device make_thr_device() {
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_threading_test",
    };
    return create_test_device(&desc)!!;
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
        gpu::TextureView view =
            gpu::create_texture_view(args.device, texture, null)!!;

        gpu::ShaderDesc shader_desc = { .stage = gpu::ShaderStage.COMPUTE, .spirv = THR_COMP_SPIRV[..], .entry_point = "main" };
        gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;

        gpu::SamplerDesc sampler_desc = {
            .min_filter = gpu::Filter.NEAREST,
            .mag_filter = gpu::Filter.NEAREST,
            .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
            .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
            .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
        };
        gpu::Sampler sampler_identity = gpu::intern_sampler(args.device, &sampler_desc)!!;
        gpu::SamplerIndex sampler = gpu::publish_sampler(args.device, sampler_identity)!!;

        if (!buffer.allocation.is_valid() || !texture.is_valid()
            || !view.is_valid() || !view.index.is_valid()
            || !shader.is_valid() || !sampler.is_valid()) {
            args.failed = true;
        }
        gpu::destroy_texture_view(args.device, view)!!;
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
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(state.strict_heap.descriptors.texture_free_count
        == state.strict_heap.descriptors.texture_next_new,
        "concurrent texture-view publication and release returns every slot");
    for (uint i = 0; i < state.strict_heap.descriptors.texture_next_new; i++) {
        assert(!state.strict_heap.descriptors.texture_cells[i].used,
            "no texture-view cell remains live after concurrent release");
    }
}

struct TextureViewAllocatorArgs {
    gpu::Device*       device;
    gpu::TextureHandle texture;
    gpu::TextureView   published;
}

fn int texture_view_allocator_worker(void* raw_args) {
    TextureViewAllocatorArgs* args = (TextureViewAllocatorArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        mem::@scoped(&allocators::NULL_ALLOCATOR) {
            gpu::TextureViewDesc view = {
                .mip_count   = 1,
                .base_layer  = 1,
                .layer_count = 1,
            };
            args.published = gpu::create_texture_view(
                args.device,
                args.texture,
                &view,
            )!!;
        };
    };
    return 0;
}

fn void texture_view_cache_uses_device_allocator_across_threads() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .dimension    = gpu::TextureDimension.TEX_2D,
        .access       = { .graphics },
        .width        = 8,
        .height       = 8,
        .mip_levels   = 2,
        .array_layers = 2,
        .format       = gpu::Format.RGBA8_UNORM,
        .usage        = { .sampled },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    TextureViewAllocatorArgs args = {
        .device  = &device,
        .texture = texture,
    };
    Thread worker;
    worker.create(&texture_view_allocator_worker, &args)!!;
    assert(worker.join()!! == 0,
        "worker publishes through a thread-local allocator that rejects allocations");
    assert(args.published.is_valid(),
        "texture-view cache storage comes from the device allocator");
    gpu::destroy_texture_view(&device, args.published)!!;
}

struct RenderViewGateArgs {
    gpu::CommandList           commands;
    gpu::AttachmentViewHandle  view;
    gpu::ExecutableCommandList executable;
    Atomic{bool}               entered;
    Atomic{bool}               completed;
}

fn int render_view_gate_worker(void* raw_args) {
    RenderViewGateArgs* args = (RenderViewGateArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        gpu::ColorTargetDesc[1] colors = { {
            .view     = args.view,
            .load_op  = gpu::LoadOp.DONT_CARE,
            .store_op = gpu::StoreOp.DONT_CARE,
        } };
        gpu::RenderPassDesc pass = {
            .colors = colors[..],
            .width  = 4,
            .height = 4,
        };
        args.entered.store(true, AtomicOrdering.RELEASE);
        gpu::cmd_begin_render_pass(&args.commands, &pass)!!;
        gpu::cmd_end_render_pass(&args.commands)!!;
        args.executable = gpu::end_commands(&args.commands)!!;
        args.completed.store(true, AtomicOrdering.RELEASE);
    };
    return 0;
}

fn void render_view_recording_avoids_texture_view_cache_lock() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;
    gpu::TextureDesc desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .access     = { .graphics },
        .width      = 8,
        .height     = 8,
        .mip_levels = 2,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .color_attach },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &desc)!!;
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::AttachmentViewHandle view = create_test_attachment_view(
        &device,
        texture,
        1,
    )!!;
    defer destroy_test_attachment_view(&device, &view)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    RenderViewGateArgs args = {
        .commands = gpu::begin_commands(
            gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!,
        )!!,
        .view = view,
    };
    state.texture_view_cache_mutex.lock()!!;
    Thread worker;
    worker.create(&render_view_gate_worker, &args)!!;
    while (!args.entered.load(AtomicOrdering.ACQUIRE)) thread::yield();
    thread::sleep_ms(25)!!;
    assert(args.completed.load(AtomicOrdering.ACQUIRE),
        "precreated attachment recording avoids the texture-view cache lock");
    state.texture_view_cache_mutex.unlock();
    assert(worker.join()!! == 0,
        "render recording completes independently of texture-view publication");
    assert(args.completed.load(AtomicOrdering.ACQUIRE),
        "render recording publishes the executable without cache resolution");
    discard_test_executable(&args.executable)!!;
}

struct SamplerRaceArgs {
    gpu::Device*      device;
    gpu::SamplerDesc* desc;
    Atomic{uint}*     ready;
    Atomic{bool}*     release;
    gpu::Sampler      sampler;
    gpu::SamplerIndex index;
}

fn int sampler_race_worker(void* raw_args) {
    SamplerRaceArgs* args = (SamplerRaceArgs*)raw_args;
    (void)args.ready.add(1, AtomicOrdering.RELEASE);
    while (!args.release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    args.sampler = gpu::intern_sampler(args.device, args.desc)!!;
    args.index = gpu::publish_sampler(args.device, args.sampler)!!;
    return 0;
}

fn void concurrent_sampler_interning_and_publication_converges() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;
    gpu::SamplerDesc desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.LINEAR,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    Atomic{uint} ready;
    Atomic{bool} release;
    SamplerRaceArgs[THR_THREADS] args;
    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i] = {
            .device  = &device,
            .desc    = &desc,
            .ready   = &ready,
            .release = &release,
        };
        threads[i].create(&sampler_race_worker, &args[i])!!;
    }
    while (ready.load(AtomicOrdering.ACQUIRE) < THR_THREADS) {
        thread::yield();
    }
    release.store(true, AtomicOrdering.RELEASE);
    foreach (&worker : threads) assert(worker.join()!! == 0);
    for (uint i = 1; i < THR_THREADS; i++) {
        assert(args[i].sampler == args[0].sampler,
            "concurrent interning returns one sampler identity");
        assert(args[i].index == args[0].index,
            "concurrent publication returns one stable index");
    }
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(state.samplers.count == 1,
        "concurrent interning creates one native sampler cell");
    assert(state.strict_heap.descriptors.sampler_next_new == 1,
        "concurrent publication writes one strict heap entry");
}

struct PipelineRaceArgs {
    gpu::Device*        device;
    gpu::ShaderCode     shader;
    gpu::PipelineHandle handle;
}

fn int pipeline_race_worker(void* arg) {
    PipelineRaceArgs* args = (PipelineRaceArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        gpu::ComputePipelineDesc desc = { .shader = args.shader };
        args.handle = gpu::create_compute_pipeline(args.device, &desc)!!;
    };
    return 0;
}

fn void same_descriptor_pipeline_race() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = THR_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&comp_sd)!!;

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

fn int generated_scratch_worker(void* raw_args) {
    ScratchWorkerArgs* args = (ScratchWorkerArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        while (!args.start.load(AtomicOrdering.ACQUIRE)) thread::yield();
        gpu::Queue queue = gpu::get_queue(
            args.device,
            gpu::QueueKind.COMPUTE,
        )!!;
        gpu::GeneratedScratchDesc reservation = {
            .pipeline                = args.pipeline,
            .kind                    = gpu::GeneratedWorkKind.DISPATCH,
            .max_commands_per_list   = 1,
            .preprocess_buffer_count = 1,
        };
        if (catch reserve_fault = gpu::reserve_generated_scratch(
            queue,
            &reservation,
        )) {
            (void)reserve_fault;
            args.failed = true;
            return 1;
        }
        gpu::CommandList commands = gpu::begin_commands(queue)!!;
        vk::VkDeviceState* state =
            (vk::VkDeviceState*)gpu::device_backend_state(*args.device);
        vk::CommandRecord* record = vk::resolve_command(&commands)!!;
        vk::RecordingContextState* context = state.recording_contexts.get(
            record.context,
        )!!;
        vk::GeneratedPreprocessBuffer* reserved =
            &context.reserved_preprocess[0];
        vk::MemoryRequirements requirements = {
            .size             = reserved.size,
            .alignment        = 1,
            .memory_type_bits = 1 << reserved.memory_type,
        };
        if (catch acquire_fault = vk::acquire_generated_preprocess_buffer(
            state:        state,
            record:       record,
            pipeline:     args.pipeline,
            kind:         gpu::GeneratedWorkKind.DISPATCH,
            requirements: &requirements,
            max_count:    1,
        )) {
            (void)acquire_fault;
            args.failed = true;
        }
        gpu::discard_commands(&commands)!!;
        gpu::release_generated_scratch(
            queue,
            args.pipeline,
            gpu::GeneratedWorkKind.DISPATCH,
        )!!;
    };
    return args.failed ? 1 : 0;
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

fn void generated_scratch_reservations_are_thread_context_local() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;
    if (!gpu::get_device_caps(&device)!!.generated_work) return;
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = THR_COMP_SPIRV[..],
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

    Atomic{bool} start;
    ScratchWorkerArgs[2] args;
    Thread[2] workers;
    for (uint i = 0; i < 2; i++) {
        args[i] = {
            .device   = &device,
            .pipeline = pipeline,
            .start    = &start,
        };
        workers[i].create(&generated_scratch_worker, &args[i])!!;
    }
    start.store(true, AtomicOrdering.RELEASE);
    foreach (i, &worker : workers) {
        assert(worker.join()!! == 0 && !args[i].failed,
            "each worker reserves and consumes its own generated scratch");
    }

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(state.recording_contexts.next_new == 2,
        "two worker threads own two independent recording contexts");
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
        gpu::Barrier before_readback = {
            .before = { .transfer = true },
            .after = { .transfer = true },
        };
        gpu::cmd_barrier(&commands, &before_readback)!!;
        gpu::BufferCopyDesc readback_copy = {
            .src = args.intermediate.span,
            .dst = args.readback.span,
        };
        gpu::cmd_copy_buffer(&commands, &readback_copy)!!;
        gpu::Barrier host_read = {
            .before = { .transfer = true },
            .after = { .host = true },
        };
        gpu::cmd_barrier(&commands, &host_read)!!;
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
        gpu::flush_mapped_span(args.device, args.root_span)!!;

        gpu::CommandList commands = gpu::begin_commands(
            gpu::get_queue(args.device, gpu::QueueKind.GRAPHICS)!!,
        )!!;
        bind_test_compute_pipeline(&commands, args.pipeline);
        gpu::cmd_dispatch(
            commands: &commands,
            root:     test_span_address(args.device, args.root_span),
            groups:   {
                (THR_ELEMENTS + 63) / 64,
                1,
                1,
            },
        )!!;
        gpu::Barrier host_read = {
            .before = { .compute = true },
            .after = { .host = true },
        };
        gpu::cmd_barrier(&commands, &host_read)!!;
        args.list = gpu::end_commands(&commands)!!;
    };
    return 0;
}

fn void parallel_recording_across_workers() @test {
    gpu::Device device = make_thr_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = THR_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&comp_sd)!!;
    gpu::ComputePipelineDesc pipe_desc = { .shader = shader };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    RecordArgs[THR_THREADS] args;
    OwnedSpan[THR_THREADS] io_owned;
    OwnedSpan[THR_THREADS] root_owned;
    defer for (uint i = 0; i < THR_THREADS; i++) {
        free_owned_span(&device, &root_owned[i])!!;
        free_owned_span(&device, &io_owned[i])!!;
    }
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i].device = &device;
        args[i].pipeline = pipeline;
        io_owned[i] = allocate_owned_span(
            device:       &device,
            size:         THR_ELEMENTS * 4 * 2,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .graphics },
            debug_name:   "thread_io",
        )!!;
        root_owned[i] = allocate_owned_span(
            device:       &device,
            size:         ComputeRoot::size,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .graphics },
            debug_name:   "thread_root",
        )!!;
        args[i].io_span = io_owned[i].span;
        args[i].root_span = root_owned[i].span;
        float* values = (float*)test_span_mapping(&device, args[i].io_span).ptr;
        for (uint e = 0; e < THR_ELEMENTS; e++) values[e] = (float)(i * 1000 + e);
        gpu::flush_mapped_span(&device, args[i].io_span)!!;
    }

    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) {
        threads[i].create(&record_worker, &args[i])!!;
    }
    foreach (&t : threads) t.join()!!;

    gpu::ExecutableCommandList[THR_THREADS] lists;
    for (uint i = 0; i < THR_THREADS; i++) lists[i] = args[i].list;
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;

    for (uint i = 0; i < THR_THREADS; i++) {
        gpu::invalidate_mapped_span(&device, args[i].io_span)!!;
        float* values = (float*)test_span_mapping(&device, args[i].io_span).ptr;
        float* doubled = &values[THR_ELEMENTS];
        for (uint e = 0; e < THR_ELEMENTS; e++) {
            assert(doubled[e] == (float)(i * 1000 + e) * 2.0f,
                "each thread's recorded dispatch produced its own results");
        }
    }
}

struct TransitionRecordArgs {
    gpu::Device*               device;
    gpu::TextureHandle         texture;
    gpu::ExecutableCommandList list;
}

fn int transition_record_worker(void* arg) {
    TransitionRecordArgs* args = (TransitionRecordArgs*)arg;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        gpu::CommandList commands = gpu::begin_commands(
            gpu::get_queue(args.device, gpu::QueueKind.GRAPHICS)!!,
        )!!;
        gpu::TextureBarrier to_dst = {
            .texture = args.texture,
            .before  = { .layout = gpu::TextureLayout.UNDEFINED },
            .after   = {
                .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
                .stages = { .transfer },
                .access = { .write },
            },
        };
        gpu::cmd_texture_barrier(&commands, &to_dst)!!;
        args.list = gpu::end_commands(&commands)!!;
    };
    return 0;
}

fn void parallel_recording_disjoint_texture_transitions() @test {
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

    TransitionRecordArgs[THR_THREADS] args;
    for (uint i = 0; i < THR_THREADS; i++) {
        args[i] = { .device = &device, .texture = textures[i] };
    }

    Thread[THR_THREADS] threads;
    for (uint i = 0; i < THR_THREADS; i++) threads[i].create(&transition_record_worker, &args[i])!!;
    foreach (&t : threads) t.join()!!;

    gpu::ExecutableCommandList[THR_THREADS] lists;
    for (uint i = 0; i < THR_THREADS; i++) lists[i] = args[i].list;
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
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

<* Release independent allocations concurrently. *>
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
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(vk::report_resource_leaks(state) == 0,
        "concurrent allocation release leaves no live resource slots");
}
