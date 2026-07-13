module profile_boundary_test;

import gpu;

fn void accept_strict_device(gpu::Device value) => (void)value;
fn void accept_strict_handle(gpu::BufferHandle value) => (void)value;
fn void accept_strict_command(gpu::CommandList value) => (void)value;
fn void accept_strict_barrier(gpu::GlobalBarrier value) => (void)value;
fn void accept_strict_descriptor(gpu::TextureIndex value) => (void)value;
fn void accept_strict_pipeline(gpu::PipelineHandle value) => (void)value;

fn void accept_compat_device(gpu::compat::Device value) => (void)value;
fn void accept_compat_handle(gpu::compat::BufferHandle value) => (void)value;
fn void accept_compat_command(gpu::compat::CommandList value) => (void)value;
fn void accept_compat_barrier(gpu::compat::GlobalBarrier value) => (void)value;
fn void accept_compat_descriptor(gpu::compat::TextureIndex value) => (void)value;
fn void accept_compat_pipeline(gpu::compat::PipelineHandle value) => (void)value;

fn int main() {
    gpu::Device strict_device = {};
    gpu::BufferHandle strict_handle = {};
    gpu::CommandList strict_command = {};
    gpu::GlobalBarrier strict_barrier = {};
    gpu::TextureIndex strict_descriptor = {};
    gpu::PipelineHandle strict_pipeline = {};

    gpu::compat::Device compat_device = {};
    gpu::compat::BufferHandle compat_handle = {};
    gpu::compat::CommandList compat_command = {};
    gpu::compat::GlobalBarrier compat_barrier = {};
    gpu::compat::TextureIndex compat_descriptor = {};
    gpu::compat::PipelineHandle compat_pipeline = {};

    accept_strict_device(strict_device);
    accept_strict_handle(strict_handle);
    accept_strict_command(strict_command);
    accept_strict_barrier(strict_barrier);
    accept_strict_descriptor(strict_descriptor);
    accept_strict_pipeline(strict_pipeline);

    accept_compat_device(compat_device);
    accept_compat_handle(compat_handle);
    accept_compat_command(compat_command);
    accept_compat_barrier(compat_barrier);
    accept_compat_descriptor(compat_descriptor);
    accept_compat_pipeline(compat_pipeline);

    return 0;
}
