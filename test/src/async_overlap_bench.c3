module gpu_bench;

import gpu @public;
import gpu::vk @public;
import gpu_test;
import vk;
import std::io;
import std::time;
import std::os::env;

// Compares serialized and independent compute/graphics submissions.
// GPU timestamps are advisory; wall time is the fallback.

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

<* Validation is opt-in so layer overhead does not affect timings. *>
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

<* Read and normalize one timestamp pair. *>
fn void read_pool(vk::VkDeviceState* state, vk::QueryPool pool, uint valid_bits, float period, double* begin_ns, double* end_ns) {
    ulong[2] ticks;
    vk::QueryResultFlags flags = (vk::QueryResultFlags)((uint)vk::QueryResultFlagBits.RESULT_64_BIT | (uint)vk::QueryResultFlagBits.WAIT_BIT);
    gpu::vk::check_backend_result(
        state:     state,
        operation: "async_overlap_bench",
        result:    vk::get_query_pool_results(
            device:      state.device,
            query_pool:  pool,
            first_query: 0,
            query_count: 2,
            data_size:   16,
            p_data:      &ticks,
            stride:      8,
            flags:       flags,
        ),
    )!!;
    *begin_ns = mask_and_scale(ticks[0], valid_bits, period);
    *end_ns   = mask_and_scale(ticks[1], valid_bits, period);
}

<* Record the compute workload and optional timestamps. *>
fn gpu::ExecutableCommandList record_compute(gpu::Device* device, gpu::PipelineHandle pipeline, gpu::GpuAddress root_gpu, vk::QueryPool pool, bool timestamps_ok) {
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(device, gpu::QueueKind.COMPUTE)!!)!!;
    vk::CommandBuffer raw_cmd = gpu::vk::resolve_command(&cmd)!!.command_buffer;
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
    return gpu::end_commands(&cmd)!!;
}

<* Record the graphics workload and optional timestamps. *>
fn gpu::ExecutableCommandList record_graphics(gpu::Device* device, gpu::PipelineHandle pipeline, gpu::TextureHandle rt, gpu::GpuAddress vertex_root, gpu::GpuAddress fragment_root, vk::QueryPool pool, bool timestamps_ok) {
    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!!)!!;
    vk::CommandBuffer raw_cmd = gpu::vk::resolve_command(&cmd)!!.command_buffer;
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
    return gpu::end_commands(&cmd)!!;
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
    io::printfn(
        "iterations=calibration=%d;measured=%d units=ms",
        CALIBRATION_ROUNDS,
        REPS,
    );
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = validation_enabled(),
        .application_name  = "gpu_async_overlap_bench",
    };
    gpu::Device device = gpu::create_device_from_desc(&desc)!!;
    defer destroy_bench_device(&device)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    bool families_distinct = state.graphics_family != state.compute_family;
    io::printfn(
        "topology: async_compute=%s graphics=%d:0 compute=%d:%d queues_distinct=%s families_distinct=%s",
        gpu::get_device_caps(&device)!!.async_compute ? "true" : "false",
        state.graphics_family,
        state.compute_family,
        state.compute_queue_index,
        state.compute_queue_distinct ? "true" : "false",
        families_distinct ? "true" : "false",
    );
    if (!gpu::get_device_caps(&device)!!.async_compute) {
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
        gpu::vk::check_backend_result(
            state:     state,
            operation: "async_overlap_bench",
            result:    vk::create_query_pool(
                device:        state.device,
                p_create_info: &pool_info,
                p_allocator:   null,
                p_query_pool:  &comp_pool,
            ),
        )!!;
        gpu::vk::check_backend_result(
            state:     state,
            operation: "async_overlap_bench",
            result:    vk::create_query_pool(
                device:        state.device,
                p_create_info: &pool_info,
                p_allocator:   null,
                p_query_pool:  &gfx_pool,
            ),
        )!!;
    }
    defer {
        if (timestamps_ok) {
            vk::destroy_query_pool(state.device, comp_pool, null);
            vk::destroy_query_pool(state.device, gfx_pool, null);
        }
    }

    gpu::AllocationDesc data_desc = {
        .size         = OVERLAP_DATA_BYTES,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .graphics, .compute, .transfer },
        .debug_name   = "overlap_compute_data",
    };
    gpu::GpuAllocation data = gpu::allocate_memory(&device, &data_desc)!!;
    defer gpu::free_allocation(&device, &data)!!;
    gpu::GpuSpan data_span = gpu::get_allocation_span(&device, data)!!;

    gpu::AllocationDesc comp_root_desc = {
        .size         = OverlapComputeRoot::size,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .graphics, .compute, .transfer },
        .debug_name   = "overlap_compute_root",
    };
    gpu::GpuAllocation comp_root_allocation = gpu::allocate_memory(
        &device,
        &comp_root_desc,
    )!!;
    defer gpu::free_allocation(&device, &comp_root_allocation)!!;
    gpu::GpuSpan comp_root_span = gpu::get_allocation_span(
        &device,
        comp_root_allocation,
    )!!;
    OverlapComputeRoot* comp_root = (OverlapComputeRoot*)gpu::get_span_mapping(
        &device,
        comp_root_span,
    )!!.ptr;
    comp_root.data_gpu = gpu::get_span_address(&device, data_span)!!;
    comp_root.elem_count = DISPATCH_GROUPS * OVERLAP_WORKGROUP;

    gpu::TextureDesc rt_desc = {
        .dimension = gpu::TextureDimension.TEX_2D, .width = OVERLAP_RT_SIZE, .height = OVERLAP_RT_SIZE,
        .access    = { .graphics, .compute, .transfer },
        .format = gpu::Format.RGBA8_UNORM, .usage = { .color_attach }, .debug_name = "overlap_rt",
    };
    gpu::TextureHandle rt = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, rt)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = OVERLAP_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode comp_shader = gpu::prepare_shader_code(&comp_sd)!!;
    gpu::ComputePipelineDesc comp_pipe_desc = { .shader = comp_shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle comp_pipeline = gpu::create_compute_pipeline(&device, &comp_pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, comp_pipeline)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = OFFSCREEN_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode vert_shader = gpu::prepare_shader_code(&vert_sd)!!;
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = OVERLAP_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode frag_shader = gpu::prepare_shader_code(&frag_sd)!!;
    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc gfx_pipe_desc = {
        .vertex_shader = vert_shader, .fragment_shader = frag_shader,
        .topology = gpu::PrimitiveTopology.TRIANGLES, .color_formats = color_formats[..],
        .debug_name = "overlap_graphics",
    };
    gpu::PipelineHandle gfx_pipeline = gpu::create_graphics_pipeline(&device, &gfx_pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, gfx_pipeline)!!;

    // One startup transition: re-recording UNDEFINED->COLOR_ATTACHMENT per rep
    // would fault INVALID_RESOURCE_STATE under the pending-layout model.
    gpu::CommandList transition_cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    gpu::TextureBarrier to_attach = {
        .texture = rt, .before_stage = gpu::Stage.HOST, .after_stage = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE, .after_hazard = gpu::Hazard.COLOR_WRITE,
        .old_layout = gpu::TextureLayout.UNDEFINED, .new_layout = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&transition_cmd, &to_attach)!!;
    gpu::ExecutableCommandList transition_cmd_executable = gpu::end_commands(&transition_cmd)!!;
    defer discard_bench_executable(&transition_cmd_executable)!!;
    gpu::ExecutableCommandList[1] transition_lists = { transition_cmd_executable };
    gpu::SubmitDesc transition_submit = { .command_lists = transition_lists[..] };
    gpu::CompletionPoint transition_completion = submit_bench_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &transition_submit,
    )!!;
    gpu::wait_completion(transition_completion)!!;
    // Solo baselines double as calibration: two proportional rounds nudge
    // iters toward TARGET_MS per workload.
    uint comp_iters = SEED_ITERS;
    uint frag_iters = SEED_ITERS;
    double comp_ns = 0;
    double frag_ns = 0;
    for (uint round = 0; round < CALIBRATION_ROUNDS; round++) {
        comp_root.iters = comp_iters;
        gpu::flush_mapped_span(&device, comp_root_span)!!;
        gpu::ExecutableCommandList comp_cmd = record_compute(
            device:        &device,
            pipeline:      comp_pipeline,
            root_gpu:      gpu::get_span_address(&device, comp_root_span)!!,
            pool:          comp_pool,
            timestamps_ok: timestamps_ok,
        );
        std::time::Clock comp_start = std::time::clock::now();
        gpu::ExecutableCommandList[1] comp_lists = { comp_cmd };
        gpu::SubmitDesc comp_submit = { .command_lists = comp_lists[..] };
        gpu::CompletionPoint comp_completion = submit_bench_queue(
            &device,
            gpu::QueueKind.COMPUTE,
            &comp_submit,
        )!!;
        gpu::wait_completion(comp_completion)!!;
        double comp_wall_ns = (double)(long)comp_start.to_now();
        if (timestamps_ok) {
            double begin_ns; double end_ns;
            read_pool(state, comp_pool, comp_valid_bits, timestamp_period, &begin_ns, &end_ns);
            comp_ns = end_ns - begin_ns;
        } else {
            comp_ns = comp_wall_ns;
        }
        double comp_scale = (TARGET_MS * 1.0e6) / comp_ns;
        double comp_next = (double)comp_iters * comp_scale;
        comp_next = comp_next < (double)MIN_ITERS ? (double)MIN_ITERS : comp_next;
        comp_next = comp_next > (double)MAX_ITERS ? (double)MAX_ITERS : comp_next;
        comp_iters = (uint)comp_next;
        OwnedSpan frag_owned = gpu_test::allocate_owned_span(
            device:       &device,
            size:         OverlapFragRoot::size,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .graphics },
            debug_name:   "calibration_fragment_root",
        )!!;
        defer gpu_test::free_owned_span(&device, &frag_owned)!!;
        gpu::GpuSpan frag_span = frag_owned.span;
        OverlapFragRoot* frag_root = (OverlapFragRoot*)gpu::get_span_mapping(&device, frag_span)!!.ptr;
        frag_root.iters = frag_iters;
        OwnedSpan vertex_owned = gpu_test::allocate_owned_span(
            device:       &device,
            size:         3 * 16,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .graphics },
            debug_name:   "calibration_vertices",
        )!!;
        defer gpu_test::free_owned_span(&device, &vertex_owned)!!;
        gpu::GpuSpan vertex_span = vertex_owned.span;
        float* verts = (float*)gpu::get_span_mapping(&device, vertex_span)!!.ptr;
        float[12] fullscreen_triangle = {
            -1.0f, -1.0f, 0.0f, 0.0f,
             3.0f, -1.0f, 2.0f, 0.0f,
            -1.0f,  3.0f, 0.0f, 2.0f,
        };
        for (uint i = 0; i < 12; i++) verts[i] = fullscreen_triangle[i];
        gpu::flush_mapped_span(&device, frag_span)!!;
        gpu::flush_mapped_span(&device, vertex_span)!!;
        gpu::ExecutableCommandList frag_cmd = record_graphics(
            device:        &device,
            pipeline:      gfx_pipeline,
            rt:            rt,
            vertex_root:   gpu::get_span_address(&device, vertex_span)!!,
            fragment_root: gpu::get_span_address(&device, frag_span)!!,
            pool:          gfx_pool,
            timestamps_ok: timestamps_ok,
        );
        std::time::Clock frag_start = std::time::clock::now();
        gpu::ExecutableCommandList[1] frag_lists = { frag_cmd };
        gpu::SubmitDesc frag_submit = { .command_lists = frag_lists[..] };
        gpu::CompletionPoint frag_completion = submit_bench_queue(
            &device,
            gpu::QueueKind.GRAPHICS,
            &frag_submit,
        )!!;
        gpu::wait_completion(frag_completion)!!;
        double frag_wall_ns = (double)(long)frag_start.to_now();
        if (timestamps_ok) {
            double begin_ns; double end_ns;
            read_pool(state, gfx_pool, gfx_valid_bits, timestamp_period, &begin_ns, &end_ns);
            frag_ns = end_ns - begin_ns;
        } else {
            frag_ns = frag_wall_ns;
        }
        double frag_scale = (TARGET_MS * 1.0e6) / frag_ns;
        double frag_next = (double)frag_iters * frag_scale;
        frag_next = frag_next < (double)MIN_ITERS ? (double)MIN_ITERS : frag_next;
        frag_next = frag_next > (double)MAX_ITERS ? (double)MAX_ITERS : frag_next;
        frag_iters = (uint)frag_next;
    }
    comp_root.iters = comp_iters;
    gpu::flush_mapped_span(&device, comp_root_span)!!;
    io::printfn(
        "calibration: compute_iters=%d (%.3f ms) fragment_iters=%d (%.3f ms) target_ms=%.1f",
        comp_iters, comp_ns / 1.0e6, frag_iters, frag_ns / 1.0e6, TARGET_MS,
    );

    double[REPS] wall_a;
    double[REPS] wall_b;
    double[REPS] gfx_parts;
    double[REPS] comp_parts;
    double[REPS] intersections;

    for (uint rep = 0; rep < REPS; rep++) {
        OwnedSpan frag_owned = gpu_test::allocate_owned_span(
            device:       &device,
            size:         OverlapFragRoot::size,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .graphics },
            debug_name:   "overlap_fragment_root",
        )!!;
        defer gpu_test::free_owned_span(&device, &frag_owned)!!;
        gpu::GpuSpan frag_span = frag_owned.span;
        OverlapFragRoot* frag_root = (OverlapFragRoot*)gpu::get_span_mapping(&device, frag_span)!!.ptr;
        frag_root.iters = frag_iters;
        OwnedSpan vertex_owned = gpu_test::allocate_owned_span(
            device:       &device,
            size:         3 * 16,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            access:       { .graphics },
            debug_name:   "overlap_vertices",
        )!!;
        defer gpu_test::free_owned_span(&device, &vertex_owned)!!;
        gpu::GpuSpan vertex_span = vertex_owned.span;
        float* verts = (float*)gpu::get_span_mapping(&device, vertex_span)!!.ptr;
        float[12] fullscreen_triangle = {
            -1.0f, -1.0f, 0.0f, 0.0f,
             3.0f, -1.0f, 2.0f, 0.0f,
            -1.0f,  3.0f, 0.0f, 2.0f,
        };
        for (uint i = 0; i < 12; i++) verts[i] = fullscreen_triangle[i];
        gpu::flush_mapped_span(&device, frag_span)!!;
        gpu::flush_mapped_span(&device, vertex_span)!!;

        // Phase A chains graphics after compute.
        std::time::Clock a_start = std::time::clock::now();
        gpu::ExecutableCommandList a_compute = record_compute(
            device:        &device,
            pipeline:      comp_pipeline,
            root_gpu:      gpu::get_span_address(&device, comp_root_span)!!,
            pool:          comp_pool,
            timestamps_ok: timestamps_ok,
        );
        gpu::ExecutableCommandList[1] a_compute_lists = { a_compute };
        gpu::SubmitDesc a_compute_submit = {
            .command_lists = a_compute_lists[..],
        };
        gpu::CompletionPoint a_compute_completion = submit_bench_queue(
            &device,
            gpu::QueueKind.COMPUTE,
            &a_compute_submit,
        )!!;

        gpu::ExecutableCommandList a_graphics = record_graphics(
            device:        &device,
            pipeline:      gfx_pipeline,
            rt:            rt,
            vertex_root:   gpu::get_span_address(&device, vertex_span)!!,
            fragment_root: gpu::get_span_address(&device, frag_span)!!,
            pool:          gfx_pool,
            timestamps_ok: timestamps_ok,
        );
        gpu::ExecutableCommandList[1] a_graphics_lists = { a_graphics };
        gpu::CompletionPoint[1] a_graphics_waits = { a_compute_completion };
        gpu::SubmitDesc a_graphics_submit = {
            .command_lists    = a_graphics_lists[..],
            .completion_waits = a_graphics_waits[..],
        };
        gpu::CompletionPoint a_graphics_completion = submit_bench_queue(
            &device,
            gpu::QueueKind.GRAPHICS,
            &a_graphics_submit,
        )!!;

        gpu::wait_completion(
            a_graphics_completion,
            PHASE_WAIT_TIMEOUT_NS,
        )!!;
        wall_a[rep] = (double)(long)a_start.to_now();

        // Phase B submits the workloads independently.
        std::time::Clock b_start = std::time::clock::now();
        gpu::ExecutableCommandList b_compute = record_compute(
            device:        &device,
            pipeline:      comp_pipeline,
            root_gpu:      gpu::get_span_address(&device, comp_root_span)!!,
            pool:          comp_pool,
            timestamps_ok: timestamps_ok,
        );
        gpu::ExecutableCommandList[1] b_compute_lists = { b_compute };
        gpu::SubmitDesc b_compute_submit = {
            .command_lists = b_compute_lists[..],
        };
        gpu::CompletionPoint b_compute_completion = submit_bench_queue(
            &device,
            gpu::QueueKind.COMPUTE,
            &b_compute_submit,
        )!!;

        gpu::ExecutableCommandList b_graphics = record_graphics(
            device:        &device,
            pipeline:      gfx_pipeline,
            rt:            rt,
            vertex_root:   gpu::get_span_address(&device, vertex_span)!!,
            fragment_root: gpu::get_span_address(&device, frag_span)!!,
            pool:          gfx_pool,
            timestamps_ok: timestamps_ok,
        );
        gpu::ExecutableCommandList[1] b_graphics_lists = { b_graphics };
        gpu::SubmitDesc b_graphics_submit = {
            .command_lists = b_graphics_lists[..],
        };
        gpu::CompletionPoint b_graphics_completion = submit_bench_queue(
            &device,
            gpu::QueueKind.GRAPHICS,
            &b_graphics_submit,
        )!!;

        gpu::wait_completion(
            b_compute_completion,
            PHASE_WAIT_TIMEOUT_NS,
        )!!;
        gpu::wait_completion(
            b_graphics_completion,
            PHASE_WAIT_TIMEOUT_NS,
        )!!;
        wall_b[rep] = (double)(long)b_start.to_now();

        double intersection_ns = 0.0;
        if (timestamps_ok) {
            double gfx_begin; double gfx_end;
            double comp_begin; double comp_end;
            read_pool(state, gfx_pool, gfx_valid_bits, timestamp_period, &gfx_begin, &gfx_end);
            read_pool(state, comp_pool, comp_valid_bits, timestamp_period, &comp_begin, &comp_end);
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
        verdict_text(gpu::get_device_caps(&device)!!.async_compute, overlap_detected), intersection_ratio,
    );

    return 0;
}
