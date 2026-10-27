module gpu_import_only;

import gpu;

fn void public_module_imports() @test {}

$assert !$defined(gpu::INVALID_ARGUMENT);

fn void check_value_lookups(gpu::Runtime* runtime, gpu::Device* device) {
    gpu::AdapterList adapters = gpu::enumerate_adapters(runtime);
    gpu::Adapter adapter = adapters.get(0);
    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS);
}

fn gpu::RayTracingPipelineInfo ray_tracing_pipeline_info(gpu::Device* device, gpu::PipelineHandle pipeline) {
    return gpu::get_ray_tracing_pipeline_info(device, pipeline);
}

fn gpu::SparseTextureRequirements sparse_texture_requirements(gpu::Device* device, gpu::TextureHandle texture) {
    return gpu::get_sparse_texture_requirements(device, texture);
}

fn gpu::GraphicsState geometry_state() {
    return gpu::render_geometry_state(320, 240);
}

module gpu_surface_win32_import_only;

import gpu::surface::win32;

module gpu_surface_wayland_import_only;

import gpu::surface::wayland;

module gpu_surface_x11_import_only;

import gpu::surface::x11;
