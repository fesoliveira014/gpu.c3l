module context_headless;

import gpu;
import gpu::util;

fn void? run_headless() {
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.device.queues.required = { .compute };
    desc.command_queue = gpu::QueueKind.COMPUTE;
    util::DeviceContext context = util::create_device_context(&desc)!;

    gpu::CommandList commands = gpu::begin_commands(&context.command_allocator)!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit_desc = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(context.queue, &submit_desc)!;
    gpu::wait_completion(completion)!;

    util::destroy_device_context(&context);
}

module context_window;

import gpu;
import gpu::util;
import gpu::surface::win32;

struct NativeWindow {
    win32::InstanceHandle instance;
    win32::WindowHandle window;
}

fn gpu::Surface? create_window_surface(gpu::Runtime* runtime, void* user_data) {
    NativeWindow* native = (NativeWindow*)user_data;
    return win32::create_surface(runtime, native.instance, native.window);
}

fn util::DeviceContext? create_window_context(NativeWindow* window, uint width, uint height) {
    util::DeviceContextDesc desc = util::default_device_context_desc();
    desc.create_surface = &create_window_surface;
    desc.surface_user_data = window;
    desc.swapchain.width = width;
    desc.swapchain.height = height;
    return util::create_device_context(&desc);
}
