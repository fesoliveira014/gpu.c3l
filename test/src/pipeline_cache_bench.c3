module gpu_bench;

import gpu;
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
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .pipeline_capacity = BENCH_DISTINCT_COUNT + 1,
        .application_name  = "gpu_pipeline_cache_bench",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = BENCH_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = BENCH_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle vert = gpu::create_shader(&device, &vert_sd)!!;
    defer gpu::destroy_shader(&device, vert)!!;
    gpu::ShaderHandle frag = gpu::create_shader(&device, &frag_sd)!!;
    defer gpu::destroy_shader(&device, frag)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc base = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = formats[..],
    };

    gpu::PipelineHandle[BENCH_DISTINCT_COUNT] distinct;
    std::time::Clock cold_start = std::time::clock::now();
    for (uint i = 0; i < BENCH_DISTINCT_COUNT; i++) {
        gpu::GraphicsPipelineDesc variant = base;
        variant.raster.depth_bias_constant = (float)(i + 1);
        distinct[i] = gpu::create_graphics_pipeline(&device, &variant)!!;
    }
    long cold_elapsed_ns = (long)cold_start.to_now();
    double cold_ns_per_create = (double)cold_elapsed_ns / (double)BENCH_DISTINCT_COUNT;
    io::printfn("phase 1 (cold, %d distinct): %.1f ns/create", BENCH_DISTINCT_COUNT, cold_ns_per_create);

    gpu::GraphicsPipelineDesc duplicate = base;
    duplicate.raster.depth_bias_constant = 1.0f;

    long dup_elapsed_ns = 0;
    for (uint i = 0; i < BENCH_DUPLICATE_COUNT; i++) {
        std::time::Clock mark = std::time::clock::now();
        gpu::PipelineHandle handle = gpu::create_graphics_pipeline(&device, &duplicate)!!;
        dup_elapsed_ns += (long)mark.to_now();
        gpu::destroy_pipeline(&device, handle)!!;
    }
    double dup_ns_per_create = (double)dup_elapsed_ns / (double)BENCH_DUPLICATE_COUNT;
    io::printfn("phase 2 (duplicate, %d at full table): %.1f ns/create", BENCH_DUPLICATE_COUNT, dup_ns_per_create);

    foreach (handle : distinct) gpu::destroy_pipeline(&device, handle)!!;

    return 0;
}
