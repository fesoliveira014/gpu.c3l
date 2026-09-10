module gpu_import_only;

import gpu;

fn void public_module_imports() @test {}

module gpu_surface_win32_import_only;

import gpu::surface::win32;

module gpu_surface_wayland_import_only;

import gpu::surface::wayland;

module gpu_surface_x11_import_only;

import gpu::surface::x11;
