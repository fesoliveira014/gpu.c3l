module gpu_import_only;

import gpu;

fn void public_module_imports() @test {}

fn void check_value_lookups(gpu::Runtime* runtime, gpu::Device* device) {
    gpu::AdapterList adapters = gpu::enumerate_adapters(runtime);
    gpu::Adapter adapter = adapters.get(0);
    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS);
}

fn gpu::RayTracingPipelineInfo ray_tracing_pipeline_info(gpu::Device* device, gpu::PipelineHandle pipeline) {
    return gpu::get_ray_tracing_pipeline_info(device, pipeline);
}

module gpu_surface_win32_import_only;

import gpu::surface::win32;

module gpu_surface_wayland_import_only;

import gpu::surface::wayland;

module gpu_surface_x11_import_only;

import gpu::surface::x11;
