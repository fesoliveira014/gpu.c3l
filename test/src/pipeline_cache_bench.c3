module gpu_bench;

import gpu;
import gpu::vk @public;
import std::io;
import std::time;

// Advisory, not gating: request one alias for each command-time raster-state
// permutation, then measure duplicate single and batch cache lookup. The raster
// matrix must remain one immutable graphics pipeline identity.

const char[*] BENCH_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] BENCH_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");

const uint BENCH_RASTER_PERMUTATION_COUNT = 200;
const uint BENCH_DUPLICATE_COUNT          = 200_000;
const uint BENCH_BATCH_SIZE               = 64;
const uint BENCH_BATCH_ROUNDS             = 2_000;

fn gpu::DynamicRasterState raster_permutation(uint index) {
    gpu::DynamicRasterState raster = {
        .topology          = (gpu::PrimitiveTopology)(index % 3),
        .cull_mode         = (gpu::CullMode)((index / 3) % 3),
        .front_face        = (gpu::FrontFace)((index / 9) % 2),
        .depth_bias_enable = (index / 18) % 2 != 0,
    };
    if (raster.depth_bias_enable) {
        raster.depth_bias_constant = (float)(index + 1) * 0.25f;
        raster.depth_bias_slope = (float)((index % 7) + 1) * 0.125f;
        raster.depth_bias_clamp = (float)(index % 5) * 0.25f;
    }
    return raster;
}

fn double record_raster_matrix(
    gpu::Device* device,
    gpu::PipelineHandle pipeline,
    gpu::DynamicRasterState[] rasters,
) {
    gpu::TextureDesc target_desc = {
        .access     = { .graphics },
        .width      = 1,
        .height     = 1,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .color_attach },
        .debug_name = "pipeline_cache_bench_target",
    };
    gpu::TextureHandle target = gpu::create_texture(device, &target_desc)!!;
    defer gpu::destroy_texture(device, target)!!;
    gpu::AttachmentViewDesc view_desc = { .texture = target };
    gpu::AttachmentViewHandle view = gpu::create_attachment_view(
        device,
        &view_desc,
    )!!;
    defer gpu::destroy_attachment_view(device, view)!!;

    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandList commands = gpu::begin_commands(queue)!!;
    gpu::TextureBarrier to_attachment = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(&commands, &to_attachment)!!;
    gpu::ColorTargetDesc[1] pass_colors = {{ .view = view }};
    gpu::RenderPassDesc pass = {
        .colors = pass_colors[..],
        .width  = 1,
        .height = 1,
    };
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    std::time::Clock start = std::time::clock::now();
    foreach (&raster : rasters) gpu::cmd_set_raster_state(&commands, raster)!!;
    long elapsed_ns = (long)start.to_now();
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    discard_bench_executable(&executable)!!;
    return (double)elapsed_ns / (double)rasters.len;
}

fn int main() {
    io::printfn(
        "iterations=raster=%d;duplicate=%d;batch=%dx%d units=ns/create,ns/state",
        BENCH_RASTER_PERMUTATION_COUNT,
        BENCH_DUPLICATE_COUNT,
        BENCH_BATCH_SIZE,
        BENCH_BATCH_ROUNDS,
    );
    gpu::RuntimeDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .pipeline_capacity = BENCH_RASTER_PERMUTATION_COUNT + BENCH_BATCH_SIZE,
        .application_name  = "gpu_pipeline_cache_bench",
    };
    gpu::Device device = create_bench_device(&desc)!!;
    defer destroy_bench_device(&device)!!;

    gpu::ShaderDesc vert_sd = {
        .stage       = gpu::ShaderStage.VERTEX,
        .spirv       = BENCH_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc frag_sd = {
        .stage       = gpu::ShaderStage.FRAGMENT,
        .spirv       = BENCH_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode vert = gpu::prepare_shader_code(&vert_sd)!!;
    gpu::ShaderCode frag = gpu::prepare_shader_code(&frag_sd)!!;

    gpu::ColorTargetState[1] colors = {{
        .format     = gpu::Format.RGBA8_UNORM,
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    gpu::GraphicsPipelineDesc immutable = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .colors          = colors[..],
    };
    gpu::DynamicRasterState[BENCH_RASTER_PERMUTATION_COUNT] rasters;
    foreach (i, &raster : rasters) *raster = raster_permutation((uint)i);

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state_ptr(&device);
    ulong native_before = vk::pipeline_native_create_count(state);
    gpu::PipelineHandle[BENCH_RASTER_PERMUTATION_COUNT] aliases;
    std::time::Clock cold_start = std::time::clock::now();
    foreach (&pipeline : aliases) {
        *pipeline = gpu::create_graphics_pipeline(&device, &immutable)!!;
    }
    long cold_elapsed_ns = (long)cold_start.to_now();
    double cold_ns_per_create = (double)cold_elapsed_ns
        / (double)BENCH_RASTER_PERMUTATION_COUNT;
    ulong native_creates = vk::pipeline_native_create_count(state) - native_before;
    uint cache_entries = vk::pipeline_cache_entry_count(state);
    assert(native_creates == 1, "raster permutations compile one native pipeline");
    assert(cache_entries == 1, "raster permutations share one cache entry");
    io::printfn(
        "phase 1 (raster matrix, requested=%d native=%d cache_entries=%d aliases=%d): %.1f ns/create",
        BENCH_RASTER_PERMUTATION_COUNT,
        native_creates,
        cache_entries,
        BENCH_RASTER_PERMUTATION_COUNT,
        cold_ns_per_create,
    );

    double raster_ns_per_state = record_raster_matrix(
        device:   &device,
        pipeline: aliases[0],
        rasters:  rasters[..],
    );
    io::printfn(
        "raster recording (requested=%d native=%d): %.1f ns/state",
        BENCH_RASTER_PERMUTATION_COUNT,
        native_creates,
        raster_ns_per_state,
    );

    long dup_elapsed_ns = 0;
    for (uint i = 0; i < BENCH_DUPLICATE_COUNT; i++) {
        std::time::Clock mark = std::time::clock::now();
        gpu::PipelineHandle handle = gpu::create_graphics_pipeline(
            &device,
            &immutable,
        )!!;
        dup_elapsed_ns += (long)mark.to_now();
        gpu::destroy_pipeline(&device, handle)!!;
    }
    double dup_ns_per_create = (double)dup_elapsed_ns
        / (double)BENCH_DUPLICATE_COUNT;
    io::printfn(
        "phase 2 (duplicate, %d at full alias set): %.1f ns/create",
        BENCH_DUPLICATE_COUNT,
        dup_ns_per_create,
    );

    gpu::GraphicsPipelineDesc[BENCH_BATCH_SIZE] batch_descs;
    foreach (&batch_desc : batch_descs) *batch_desc = immutable;
    gpu::PipelineHandle[BENCH_BATCH_SIZE] batch_pipelines;
    long batch_elapsed_ns = 0;
    for (uint round = 0; round < BENCH_BATCH_ROUNDS; round++) {
        std::time::Clock mark = std::time::clock::now();
        gpu::create_graphics_pipelines(
            device:        &device,
            descs:         batch_descs[..],
            out_pipelines: batch_pipelines[..],
        )!!;
        batch_elapsed_ns += (long)mark.to_now();
        foreach (pipeline : batch_pipelines) {
            gpu::destroy_pipeline(&device, pipeline)!!;
        }
    }
    double batch_ns_per_create = (double)batch_elapsed_ns
        / (double)(BENCH_BATCH_SIZE * BENCH_BATCH_ROUNDS);
    io::printfn(
        "phase 3 (cached batch, %dx%d): %.1f ns/create",
        BENCH_BATCH_SIZE,
        BENCH_BATCH_ROUNDS,
        batch_ns_per_create,
    );

    foreach (handle : aliases) gpu::destroy_pipeline(&device, handle)!!;

    return 0;
}
