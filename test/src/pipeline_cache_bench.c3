module gpu_bench;

import gpu::compat;
import std::io;
import std::time;

// Advisory, not gating: cold-create N distinct graphics pipelines (unbounded
// distinct keys via a swept raster depth-bias-constant, one shared shader
// pair), then M duplicate creates at the full table to isolate index lookup
// from driver compilation. Numbers are indicative on lavapipe.

const char[*] BENCH_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] BENCH_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");

const uint BENCH_DISTINCT_COUNT  = 200;
const uint BENCH_DUPLICATE_COUNT = 200_000;

fn int main() {
    io::printfn(
        "iterations=cold=%d;duplicate=%d units=ns/create",
        BENCH_DISTINCT_COUNT,
        BENCH_DUPLICATE_COUNT,
    );
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .pipeline_capacity = BENCH_DISTINCT_COUNT + 1,
        .application_name  = "gpu_pipeline_cache_bench",
    };
    gpu::compat::Device device = gpu::compat::create_device(&desc)!!;
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::ShaderDesc vert_sd = { .stage = gpu::compat::ShaderStage.VERTEX, .spirv = BENCH_VERT_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderDesc frag_sd = { .stage = gpu::compat::ShaderStage.FRAGMENT, .spirv = BENCH_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle vert = gpu::compat::create_shader(&device, &vert_sd)!!;
    defer gpu::compat::destroy_shader(&device, vert)!!;
    gpu::compat::ShaderHandle frag = gpu::compat::create_shader(&device, &frag_sd)!!;
    defer gpu::compat::destroy_shader(&device, frag)!!;

    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc base = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .topology        = gpu::compat::PrimitiveTopology.TRIANGLES,
        .color_formats   = formats[..],
    };

    gpu::compat::PipelineHandle[BENCH_DISTINCT_COUNT] distinct;
    std::time::Clock cold_start = std::time::clock::now();
    for (uint i = 0; i < BENCH_DISTINCT_COUNT; i++) {
        gpu::compat::GraphicsPipelineDesc variant = base;
        variant.raster.depth_bias_constant = (float)(i + 1);
        distinct[i] = gpu::compat::create_graphics_pipeline(&device, &variant)!!;
    }
    long cold_elapsed_ns = (long)cold_start.to_now();
    double cold_ns_per_create = (double)cold_elapsed_ns / (double)BENCH_DISTINCT_COUNT;
    io::printfn("phase 1 (cold, %d distinct): %.1f ns/create", BENCH_DISTINCT_COUNT, cold_ns_per_create);

    gpu::compat::GraphicsPipelineDesc duplicate = base;
    duplicate.raster.depth_bias_constant = 1.0f;

    long dup_elapsed_ns = 0;
    for (uint i = 0; i < BENCH_DUPLICATE_COUNT; i++) {
        std::time::Clock mark = std::time::clock::now();
        gpu::compat::PipelineHandle handle = gpu::compat::create_graphics_pipeline(&device, &duplicate)!!;
        dup_elapsed_ns += (long)mark.to_now();
        gpu::compat::destroy_pipeline(&device, handle)!!;
    }
    double dup_ns_per_create = (double)dup_elapsed_ns / (double)BENCH_DUPLICATE_COUNT;
    io::printfn("phase 2 (duplicate, %d at full table): %.1f ns/create", BENCH_DUPLICATE_COUNT, dup_ns_per_create);

    foreach (handle : distinct) gpu::compat::destroy_pipeline(&device, handle)!!;

    return 0;
}
