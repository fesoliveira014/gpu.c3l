module gpu_bench;

import gpu;
import gpu::vk;
import vk;
import std::io;
import std::time;
import std::os::env;

// Advisory, not gating: observes whether a saturating compute dispatch and a
// heavy fullscreen-fragment pass actually overlap across the graphics and
// (when distinct) compute queues. Phase A chains the two workloads through
// one timeline semaphore; phase B submits them independently through two.
// Overlap materializes iff wall(B) approaches max(parts) rather than
// sum(parts). Per-queue vkCmdWriteTimestamp2 query pools give GPU-side
// durations when the adapter supports them; wall clocks are the fallback and
// remain the primary evidence printed alongside any GPU-duration figures.
// Never asserts — this is an observation harness, not a correctness test.
// Needs an ICD — pin lavapipe via VK_DRIVER_FILES.

const char[*] OVERLAP_COMP_SPIRV   = $embed("shaders/overlap.comp.spv");
const char[*] OVERLAP_FRAG_SPIRV   = $embed("shaders/overlap.frag.spv");
const char[*] OFFSCREEN_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");

const uint DISPATCH_GROUPS = 4096;
const uint OVERLAP_RT_SIZE = 1024;
const double TARGET_MS = 4.0;
const uint REPS = 5;
const uint CALIBRATION_ROUNDS = 2;
const uint SEED_ITERS = 64;
const uint MIN_ITERS = 1;
const uint MAX_ITERS = 2_000_000;
const ulong PHASE_WAIT_TIMEOUT_NS = 10_000_000_000;
const uint OVERLAP_DATA_BYTES = DISPATCH_GROUPS * OVERLAP_WORKGROUP * 4;

<* Env-gated validation opt-in, same idiom as descriptor_churn_bench's
   GPU_C3L_RUN_DESCRIPTOR_BUFFER_E2E: off by default so steady-state numbers
   aren't skewed by layer overhead. *>
fn bool validation_enabled() {
    if (try value = env::tget_var("GPU_C3L_BENCH_VALIDATION")) {
        return value == "1";
    }
    return false;
}

fn double mask_and_scale(ulong raw, uint valid_bits, float period) {
    ulong mask = valid_bits < 64 ? ((1ul << valid_bits) - 1) : ~0ul;
    return (double)(raw & mask) * (double)period;
}

<* Blocking (WAIT_BIT) read of a two-query TIMESTAMP pool, converted to ns and
   masked to the family's valid bits. *>
fn void read_pool(vk::Device device, vk::QueryPool pool, uint valid_bits, float period, double* begin_ns, double* end_ns) {
    ulong[2] ticks;
    vk::QueryResultFlags flags = (vk::QueryResultFlags)((uint)vk::QueryResultFlagBits.RESULT_64_BIT | (uint)vk::QueryResultFlagBits.WAIT_BIT);
    vk::check(vk::get_query_pool_results(device, pool, 0, 2, 16, &ticks, 8, flags))!!;
    *begin_ns = mask_and_scale(ticks[0], valid_bits, period);
    *end_ns   = mask_and_scale(ticks[1], valid_bits, period);
}

<* Record a compute list: reset + top-of-pipe timestamp, the saturating
   dispatch, all-commands timestamp. Query calls are raw vk:: (hostQueryReset
   isn't enabled, so resets are cmd-recorded only) against the whitebox
   backend command buffer. *>
fn gpu::CommandList record_compute(gpu::Device* device, gpu::PipelineHandle pipeline, gpu::GpuAddress root_gpu, vk::QueryPool pool, bool timestamps_ok) {
    gpu::CommandList cmd = gpu::begin_commands(device, gpu::QueueKind.COMPUTE)!!;
    vk::CommandBuffer raw_cmd = (vk::CommandBuffer)cmd.backend;
    if (timestamps_ok) {
        vk::cmd_reset_query_pool(raw_cmd, pool, 0, 2);
        vk::cmd_write_timestamp2(raw_cmd, vk::PipelineStageFlagBits2.STAGE_2_TOP_OF_PIPE_BIT, pool, 0);
    }
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: pipeline,
        root:     root_gpu,
        groups:   { DISPATCH_GROUPS, 1, 1 },
    )!!;
    if (timestamps_ok) {
        vk::cmd_write_timestamp2(raw_cmd, vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT, pool, 1);
    }
    gpu::end_commands(&cmd)!!;
    return cmd;
}

<* Record a graphics list: reset + timestamps outside the pass (a query
   write mid-pass is avoidable and this keeps the pattern identical to the
   compute list), the heavy-fragment pass over the whole RT, end. *>
fn gpu::CommandList record_graphics(gpu::Device* device, gpu::PipelineHandle pipeline, gpu::TextureHandle rt, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, vk::QueryPool pool, bool timestamps_ok) {
    gpu::CommandList cmd = gpu::begin_commands(device, gpu::QueueKind.GRAPHICS)!!;
    vk::CommandBuffer raw_cmd = (vk::CommandBuffer)cmd.backend;
    if (timestamps_ok) {
        vk::cmd_reset_query_pool(raw_cmd, pool, 0, 2);
        vk::cmd_write_timestamp2(raw_cmd, vk::PipelineStageFlagBits2.STAGE_2_TOP_OF_PIPE_BIT, pool, 0);
    }
    gpu::ColorTargetDesc[1] colors = {{
        .texture = rt,
        .load_op = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = OVERLAP_RT_SIZE, .height = OVERLAP_RT_SIZE };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        pipeline:       pipeline,
        vertex_root:    vertex_root,
        fragment_root:  fragment_root,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    if (timestamps_ok) {
        vk::cmd_write_timestamp2(raw_cmd, vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT, pool, 1);
    }
    gpu::end_commands(&cmd)!!;
    return cmd;
}

fn double median_of(double[REPS] values) {
    double[REPS] sorted = values;
    for (uint i = 1; i < REPS; i++) {
        double key = sorted[i];
        int j = (int)i - 1;
        while (j >= 0 && sorted[j] > key) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }
    return sorted[REPS / 2];
}

fn ZString verdict_text(bool async_compute, bool overlap_detected) {
    if (!async_compute) return "not applicable (single queue)";
    return overlap_detected ? "OVERLAP" : "NO OVERLAP";
}

fn int main() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = validation_enabled(),
        .frames_in_flight  = 2,
        .application_name  = "gpu_async_overlap_bench",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;

    bool distinct = state.graphics_family != state.compute_family;
    io::printfn(
        "topology: async_compute=%s graphics_family=%d compute_family=%d distinct=%s",
        device.caps.async_compute ? "true" : "false", state.graphics_family, state.compute_family,
        distinct ? "true" : "false",
    );
    if (!device.caps.async_compute) {
        io::printfn("single queue: compute aliases graphics on this adapter -- phase A and phase B submit to the same queue, so wall_a ~= wall_b is expected");
    }

    vk::PhysicalDeviceProperties props;
    vk::get_physical_device_properties(state.physical_device, &props);
    float timestamp_period = props.limits.timestamp_period;
    bool timestamp_compute_and_graphics = props.limits.timestamp_compute_and_graphics != 0;

    uint family_count = 8;
    vk::QueueFamilyProperties[8] families;
    vk::get_physical_device_queue_family_properties(state.physical_device, &family_count, &families[0]);
    uint gfx_valid_bits = state.graphics_family < family_count ? families[state.graphics_family].timestamp_valid_bits : 0;
    uint comp_valid_bits = state.compute_family < family_count ? families[state.compute_family].timestamp_valid_bits : 0;

    bool timestamps_ok = timestamp_compute_and_graphics && timestamp_period > 0.0f
        && gfx_valid_bits > 0 && comp_valid_bits > 0;
    io::printfn(
        "timestamps: compute_and_graphics=%s period_ns=%.4f valid_bits(graphics)=%d valid_bits(compute)=%d",
        timestamp_compute_and_graphics ? "true" : "false", timestamp_period, gfx_valid_bits, comp_valid_bits,
    );
    if (!timestamps_ok) {
        io::printfn("timestamp queries unsupported or degenerate on this adapter: gpu-duration columns fall back to the calibrated wall-clock baseline");
    }

    vk::QueryPool comp_pool;
    vk::QueryPool gfx_pool;
    if (timestamps_ok) {
        vk::QueryPoolCreateInfo pool_info = vk::query_pool_create_info().set_query_type(vk::QueryType.TIMESTAMP).set_query_count(2);
        vk::check(vk::create_query_pool(state.device, &pool_info, null, &comp_pool))!!;
        vk::check(vk::create_query_pool(state.device, &pool_info, null, &gfx_pool))!!;
    }
    defer {
        if (timestamps_ok) {
            vk::destroy_query_pool(state.device, comp_pool, null);
            vk::destroy_query_pool(state.device, gfx_pool, null);
        }
    }

    gpu::BufferDesc data_desc = {
        .size = OVERLAP_DATA_BYTES, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name = "overlap_compute_data",
    };
    gpu::BufferHandle data_buffer = gpu::create_buffer(&device, &data_desc)!!;
    defer gpu::destroy_buffer(&device, data_buffer)!!;

    gpu::BufferDesc comp_root_desc = {
        .size = OverlapComputeRoot::size, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .debug_name = "overlap_compute_root",
    };
    gpu::BufferHandle comp_root_buffer = gpu::create_buffer(&device, &comp_root_desc)!!;
    defer gpu::destroy_buffer(&device, comp_root_buffer)!!;
    gpu::GpuSpan comp_root_span = gpu::get_buffer_span(&device, comp_root_buffer)!!;
    OverlapComputeRoot* comp_root = (OverlapComputeRoot*)comp_root_span.cpu;
    comp_root.data_gpu = gpu::get_buffer_address(&device, data_buffer)!!;
    comp_root.elem_count = DISPATCH_GROUPS * OVERLAP_WORKGROUP;

    gpu::TextureDesc rt_desc = {
        .dimension = gpu::TextureDimension.TEX_2D, .width = OVERLAP_RT_SIZE, .height = OVERLAP_RT_SIZE,
        .format = gpu::Format.RGBA8_UNORM, .usage = { .color_attach }, .debug_name = "overlap_rt",
    };
    gpu::TextureHandle rt = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, rt)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = OVERLAP_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle comp_shader = gpu::create_shader(&device, &comp_sd)!!;
    defer gpu::destroy_shader(&device, comp_shader)!!;
    gpu::ComputePipelineDesc comp_pipe_desc = { .shader = comp_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle comp_pipeline = gpu::create_compute_pipeline(&device, &comp_pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, comp_pipeline)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = OFFSCREEN_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle vert_shader = gpu::create_shader(&device, &vert_sd)!!;
    defer gpu::destroy_shader(&device, vert_shader)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = OVERLAP_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle frag_shader = gpu::create_shader(&device, &frag_sd)!!;
    defer gpu::destroy_shader(&device, frag_shader)!!;
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc gfx_pipe_desc = {
        .vertex_shader = vert_shader, .fragment_shader = frag_shader,
        .topology = gpu::PrimitiveTopology.TRIANGLES, .color_formats = color_formats[..],
        .debug_name = "overlap_graphics",
    };
    gpu::PipelineHandle gfx_pipeline = gpu::create_graphics_pipeline(&device, &gfx_pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, gfx_pipeline)!!;

    gpu::SemaphoreDesc sem_a_desc = { .debug_name = "overlap_phase_a" };
    gpu::SemaphoreHandle sem_a = gpu::create_semaphore(&device, &sem_a_desc)!!;
    defer gpu::destroy_semaphore(&device, sem_a)!!;
    gpu::SemaphoreDesc sem_c_desc = { .debug_name = "overlap_phase_b_compute" };
    gpu::SemaphoreHandle sem_c = gpu::create_semaphore(&device, &sem_c_desc)!!;
    defer gpu::destroy_semaphore(&device, sem_c)!!;
    gpu::SemaphoreDesc sem_g_desc = { .debug_name = "overlap_phase_b_graphics" };
    gpu::SemaphoreHandle sem_g = gpu::create_semaphore(&device, &sem_g_desc)!!;
    defer gpu::destroy_semaphore(&device, sem_g)!!;

    // One startup transition: re-recording UNDEFINED->COLOR_ATTACHMENT per rep
    // would fault INVALID_RESOURCE_STATE under the pending-layout model.
    gpu::FrameToken frame_token = gpu::begin_frame(&device)!!;
    gpu::CommandList transition_cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::TextureBarrier to_attach = {
        .texture = rt, .before_stage = gpu::Stage.HOST, .after_stage = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE, .after_hazard = gpu::Hazard.COLOR_WRITE,
        .old_layout = gpu::TextureLayout.UNDEFINED, .new_layout = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&transition_cmd, &to_attach)!!;
    gpu::end_commands(&transition_cmd)!!;
    gpu::CommandList[1] transition_lists = { transition_cmd };
    gpu::SubmitDesc transition_submit = { .command_lists = transition_lists[..] };
    gpu::submit(&device, &transition_submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&frame_token)!!;

    // Solo baselines double as calibration: two proportional rounds nudge
    // iters toward TARGET_MS per workload.
    uint comp_iters = SEED_ITERS;
    uint frag_iters = SEED_ITERS;
    double comp_ns = 0;
    double frag_ns = 0;
    for (uint round = 0; round < CALIBRATION_ROUNDS; round++) {
        comp_root.iters = comp_iters;
        gpu::flush_buffer(&device, comp_root_buffer, 0, 0)!!;
        frame_token = gpu::begin_frame(&device)!!;
        gpu::CommandList comp_cmd = record_compute(
            device:        &device,
            pipeline:      comp_pipeline,
            root_gpu:      comp_root_span.gpu,
            pool:          comp_pool,
            timestamps_ok: timestamps_ok,
        );
        std::time::Clock comp_start = std::time::clock::now();
        gpu::CommandList[1] comp_lists = { comp_cmd };
        gpu::SubmitDesc comp_submit = { .command_lists = comp_lists[..] };
        gpu::submit(&device, &comp_submit)!!;
        gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;
        double comp_wall_ns = (double)(long)comp_start.to_now();
        if (timestamps_ok) {
            double begin_ns; double end_ns;
            read_pool(state.device, comp_pool, comp_valid_bits, timestamp_period, &begin_ns, &end_ns);
            comp_ns = end_ns - begin_ns;
        } else {
            comp_ns = comp_wall_ns;
        }
        gpu::end_frame(&frame_token)!!;
        double comp_scale = (TARGET_MS * 1.0e6) / comp_ns;
        double comp_next = (double)comp_iters * comp_scale;
        comp_next = comp_next < (double)MIN_ITERS ? (double)MIN_ITERS : comp_next;
        comp_next = comp_next > (double)MAX_ITERS ? (double)MAX_ITERS : comp_next;
        comp_iters = (uint)comp_next;

        frame_token = gpu::begin_frame(&device)!!;
        gpu::GpuSpan frag_span = gpu::alloc_frame_span(&frame_token, OverlapFragRoot::size, 16)!!;
        OverlapFragRoot* frag_root = (OverlapFragRoot*)frag_span.cpu;
        frag_root.iters = frag_iters;
        gpu::GpuSpan vertex_span = gpu::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
        float* verts = (float*)vertex_span.cpu;
        float[12] fullscreen_triangle = {
            -1.0f, -1.0f, 0.0f, 0.0f,
             3.0f, -1.0f, 2.0f, 0.0f,
            -1.0f,  3.0f, 0.0f, 2.0f,
        };
        for (uint i = 0; i < 12; i++) verts[i] = fullscreen_triangle[i];
        gpu::CommandList frag_cmd = record_graphics(

            device:        &device,

            pipeline:      gfx_pipeline,

            rt:            rt,

            vertex_root:   vertex_span.gpu,

            fragment_root: frag_span.gpu,

            pool:          gfx_pool,

            timestamps_ok: timestamps_ok,

        );
        std::time::Clock frag_start = std::time::clock::now();
        gpu::CommandList[1] frag_lists = { frag_cmd };
        gpu::SubmitDesc frag_submit = { .command_lists = frag_lists[..] };
        gpu::submit(&device, &frag_submit)!!;
        gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
        double frag_wall_ns = (double)(long)frag_start.to_now();
        if (timestamps_ok) {
            double begin_ns; double end_ns;
            read_pool(state.device, gfx_pool, gfx_valid_bits, timestamp_period, &begin_ns, &end_ns);
            frag_ns = end_ns - begin_ns;
        } else {
            frag_ns = frag_wall_ns;
        }
        gpu::end_frame(&frame_token)!!;
        double frag_scale = (TARGET_MS * 1.0e6) / frag_ns;
        double frag_next = (double)frag_iters * frag_scale;
        frag_next = frag_next < (double)MIN_ITERS ? (double)MIN_ITERS : frag_next;
        frag_next = frag_next > (double)MAX_ITERS ? (double)MAX_ITERS : frag_next;
        frag_iters = (uint)frag_next;
    }
    comp_root.iters = comp_iters;
    gpu::flush_buffer(&device, comp_root_buffer, 0, 0)!!;
    io::printfn(
        "calibration: compute_iters=%d (%.3f ms) fragment_iters=%d (%.3f ms) target_ms=%.1f",
        comp_iters, comp_ns / 1.0e6, frag_iters, frag_ns / 1.0e6, TARGET_MS,
    );

    double[REPS] wall_a;
    double[REPS] wall_b;
    double[REPS] gfx_parts;
    double[REPS] comp_parts;
    double[REPS] intersections;

    ulong sem_a_counter = 0;
    ulong sem_c_counter = 0;
    ulong sem_g_counter = 0;

    for (uint rep = 0; rep < REPS; rep++) {
        frame_token = gpu::begin_frame(&device)!!;

        gpu::GpuSpan frag_span = gpu::alloc_frame_span(&frame_token, OverlapFragRoot::size, 16)!!;
        OverlapFragRoot* frag_root = (OverlapFragRoot*)frag_span.cpu;
        frag_root.iters = frag_iters;
        gpu::GpuSpan vertex_span = gpu::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
        float* verts = (float*)vertex_span.cpu;
        float[12] fullscreen_triangle = {
            -1.0f, -1.0f, 0.0f, 0.0f,
             3.0f, -1.0f, 2.0f, 0.0f,
            -1.0f,  3.0f, 0.0f, 2.0f,
        };
        for (uint i = 0; i < 12; i++) verts[i] = fullscreen_triangle[i];

        // Phase A: dependency-chained through one timeline semaphore.
        ulong av = sem_a_counter + 1;
        std::time::Clock a_start = std::time::clock::now();
        gpu::CommandList a_compute = record_compute(

            device:        &device,

            pipeline:      comp_pipeline,

            root_gpu:      comp_root_span.gpu,

            pool:          comp_pool,

            timestamps_ok: timestamps_ok,

        );
        gpu::CommandList[1] a_compute_lists = { a_compute };
        gpu::SemaphoreSignal[1] a_compute_signals = {{ .semaphore = sem_a, .value = (gpu::SemaphoreValue)av, .stage = gpu::Stage.COMPUTE_SHADER }};
        gpu::SubmitDesc a_compute_submit = { .command_lists = a_compute_lists[..], .signals = a_compute_signals[..] };
        gpu::submit(&device, &a_compute_submit)!!;

        gpu::CommandList a_graphics = record_graphics(


            device:        &device,


            pipeline:      gfx_pipeline,


            rt:            rt,


            vertex_root:   vertex_span.gpu,


            fragment_root: frag_span.gpu,


            pool:          gfx_pool,


            timestamps_ok: timestamps_ok,


        );
        gpu::CommandList[1] a_graphics_lists = { a_graphics };
        // Stage.PRESENT is the only public all-commands wait spelling
        // (STAGE_2_ALL_COMMANDS_BIT, gpu/vk/helpers.c3:60).
        gpu::SemaphoreWait[1] a_graphics_waits = {{ .semaphore = sem_a, .value = (gpu::SemaphoreValue)av, .stage = gpu::Stage.PRESENT }};
        gpu::SemaphoreSignal[1] a_graphics_signals = {{ .semaphore = sem_a, .value = (gpu::SemaphoreValue)(av + 1), .stage = gpu::Stage.COLOR_ATTACHMENT }};
        gpu::SubmitDesc a_graphics_submit = { .command_lists = a_graphics_lists[..], .waits = a_graphics_waits[..], .signals = a_graphics_signals[..] };
        gpu::submit(&device, &a_graphics_submit)!!;

        gpu::wait_semaphore(&device, sem_a, (gpu::SemaphoreValue)(av + 1), PHASE_WAIT_TIMEOUT_NS)!!;
        wall_a[rep] = (double)(long)a_start.to_now();
        sem_a_counter = av + 1;

        // Phase B: independent submission through two timelines (a single
        // shared timeline would race the strictly-increasing rule across
        // two signaling queues).
        ulong cv = sem_c_counter + 1;
        ulong gv = sem_g_counter + 1;
        std::time::Clock b_start = std::time::clock::now();
        gpu::CommandList b_compute = record_compute(

            device:        &device,

            pipeline:      comp_pipeline,

            root_gpu:      comp_root_span.gpu,

            pool:          comp_pool,

            timestamps_ok: timestamps_ok,

        );
        gpu::CommandList[1] b_compute_lists = { b_compute };
        gpu::SemaphoreSignal[1] b_compute_signals = {{ .semaphore = sem_c, .value = (gpu::SemaphoreValue)cv, .stage = gpu::Stage.COMPUTE_SHADER }};
        gpu::SubmitDesc b_compute_submit = { .command_lists = b_compute_lists[..], .signals = b_compute_signals[..] };
        gpu::submit(&device, &b_compute_submit)!!;

        gpu::CommandList b_graphics = record_graphics(


            device:        &device,


            pipeline:      gfx_pipeline,


            rt:            rt,


            vertex_root:   vertex_span.gpu,


            fragment_root: frag_span.gpu,


            pool:          gfx_pool,


            timestamps_ok: timestamps_ok,


        );
        gpu::CommandList[1] b_graphics_lists = { b_graphics };
        gpu::SemaphoreSignal[1] b_graphics_signals = {{ .semaphore = sem_g, .value = (gpu::SemaphoreValue)gv, .stage = gpu::Stage.COLOR_ATTACHMENT }};
        gpu::SubmitDesc b_graphics_submit = { .command_lists = b_graphics_lists[..], .signals = b_graphics_signals[..] };
        gpu::submit(&device, &b_graphics_submit)!!;

        gpu::wait_semaphore(&device, sem_c, (gpu::SemaphoreValue)cv, PHASE_WAIT_TIMEOUT_NS)!!;
        gpu::wait_semaphore(&device, sem_g, (gpu::SemaphoreValue)gv, PHASE_WAIT_TIMEOUT_NS)!!;
        wall_b[rep] = (double)(long)b_start.to_now();
        sem_c_counter = cv;
        sem_g_counter = gv;

        double intersection_ns = 0.0;
        if (timestamps_ok) {
            double gfx_begin; double gfx_end;
            double comp_begin; double comp_end;
            read_pool(state.device, gfx_pool, gfx_valid_bits, timestamp_period, &gfx_begin, &gfx_end);
            read_pool(state.device, comp_pool, comp_valid_bits, timestamp_period, &comp_begin, &comp_end);
            gfx_parts[rep] = gfx_end - gfx_begin;
            comp_parts[rep] = comp_end - comp_begin;
            double overlap_lo = gfx_begin > comp_begin ? gfx_begin : comp_begin;
            double overlap_hi = gfx_end < comp_end ? gfx_end : comp_end;
            intersection_ns = overlap_hi > overlap_lo ? overlap_hi - overlap_lo : 0.0;
        } else {
            // No per-rep GPU duration without timestamps: the calibrated
            // solo baseline is the best available per-workload estimate.
            gfx_parts[rep] = frag_ns;
            comp_parts[rep] = comp_ns;
        }
        intersections[rep] = intersection_ns;

        io::printfn(
            "rep %d: wall_a=%.3fms wall_b=%.3fms gpu_gfx=%.3fms gpu_comp=%.3fms intersection=%.3fms",
            rep, wall_a[rep] / 1.0e6, wall_b[rep] / 1.0e6, gfx_parts[rep] / 1.0e6, comp_parts[rep] / 1.0e6, intersection_ns / 1.0e6,
        );

        gpu::end_frame(&frame_token)!!;
    }

    double[REPS] sum_parts;
    double[REPS] max_parts;
    for (uint rep = 0; rep < REPS; rep++) {
        sum_parts[rep] = gfx_parts[rep] + comp_parts[rep];
        max_parts[rep] = gfx_parts[rep] > comp_parts[rep] ? gfx_parts[rep] : comp_parts[rep];
    }
    double median_a = median_of(wall_a);
    double median_b = median_of(wall_b);
    double median_sum = median_of(sum_parts);
    double median_max = median_of(max_parts);
    double median_intersection = median_of(intersections);

    io::printfn(
        "median: wall_a=%.3fms wall_b=%.3fms sum_parts=%.3fms max_parts=%.3fms",
        median_a / 1.0e6, median_b / 1.0e6, median_sum / 1.0e6, median_max / 1.0e6,
    );

    bool overlap_detected = median_b <= 1.25 * median_max && median_b <= 0.75 * median_sum;
    double intersection_ratio = median_max > 0.0 ? median_intersection / median_max : 0.0;
    io::printfn(
        "verdict: %s (intersection_ratio=%.3f) -- cross-queue timestamp comparison assumes one clock domain; per-queue GPU durations and wall clocks above are the primary evidence",
        verdict_text(device.caps.async_compute, overlap_detected), intersection_ratio,
    );

    return 0;
}
