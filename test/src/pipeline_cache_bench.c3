module gpu_bench;

import gpu;
import std::io;
import std::time;

// Advisory, not gating: record one complete packet for each raster-state
// permutation, then measure duplicate singular cache lookup. The raster matrix
// must remain one format-only graphics pipeline identity.

const char[*] BENCH_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] BENCH_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");

const uint BENCH_RASTER_PERMUTATION_COUNT = 200;
const uint BENCH_DUPLICATE_COUNT          = 200_000;

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

fn double record_complete_packet_matrix(
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::DynamicRasterState[] rasters,
) {
    gpu::Device* device = &allocator.device;
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

    gpu::CommandList commands = gpu::begin_commands(allocator)!!;
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
    gpu::GraphicsState graphics =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {{
        .write_mask = gpu::COLOR_WRITE_ALL,
    }};
    graphics.color = { .targets = color_targets[..] };
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!!;
    gpu::cmd_set_graphics_state(&commands, &graphics)!!;
    std::time::Clock start = std::time::clock::now();
    foreach (&raster : rasters) {
        graphics.raster = *raster;
        gpu::cmd_set_graphics_state(&commands, &graphics)!!;
    }
    long elapsed_ns = (long)start.to_now();
    gpu::cmd_end_render_pass(&commands)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    discard_bench_executable(&executable)!!;
    return (double)elapsed_ns / (double)rasters.len;
}

fn int main() {
    io::printfn(
        "iterations=complete_packet=%d;duplicate=%d units=ns/create,ns/packet",
        BENCH_RASTER_PERMUTATION_COUNT,
        BENCH_DUPLICATE_COUNT,
    );
    gpu::RuntimeDesc desc = {
        .pipeline_capacity = BENCH_RASTER_PERMUTATION_COUNT + 1,
        .application_name  = "gpu_pipeline_cache_bench",
    };
    gpu::Device device = create_bench_device(&desc)!!;
    defer destroy_bench_device(&device)!!;
    gpu::Queue graphics_queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        graphics_queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;

    gpu::ShaderDesc vert_sd = {
        .spirv       = BENCH_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc frag_sd = {
        .spirv       = BENCH_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc vert = vert_sd;
    gpu::ShaderDesc frag = frag_sd;

    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc format_only = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .color_formats   = color_formats[..],
    };
    gpu::DynamicRasterState[BENCH_RASTER_PERMUTATION_COUNT] rasters;
    foreach (i, &raster : rasters) *raster = raster_permutation((uint)i);

    gpu::PipelineHandle[BENCH_RASTER_PERMUTATION_COUNT] aliases;
    std::time::Clock cold_start = std::time::clock::now();
    foreach (&pipeline : aliases) {
        *pipeline = gpu::create_graphics_pipeline(
            &device,
            &format_only,
        )!!;
    }
    long cold_elapsed_ns = (long)cold_start.to_now();
    double cold_ns_per_create = (double)cold_elapsed_ns
        / (double)BENCH_RASTER_PERMUTATION_COUNT;
    io::printfn(
        "raster matrix: %d aliases, %.1f ns/create",
        BENCH_RASTER_PERMUTATION_COUNT,
        cold_ns_per_create,
    );

    double packet_ns = record_complete_packet_matrix(
        allocator: &allocator,
        pipeline: aliases[0],
        rasters:  rasters[..],
    );
    io::printfn(
        "complete packet recording: %d packets, %.1f ns/packet",
        BENCH_RASTER_PERMUTATION_COUNT,
        packet_ns,
    );

    long dup_elapsed_ns = 0;
    for (uint i = 0; i < BENCH_DUPLICATE_COUNT; i++) {
        std::time::Clock mark = std::time::clock::now();
        gpu::PipelineHandle handle = gpu::create_graphics_pipeline(
            &device,
            &format_only,
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

    foreach (handle : aliases) gpu::destroy_pipeline(&device, handle)!!;

    return 0;
}
