module gpu_canonical_strict_surface;

import gpu;

fn void pin_runtime_adapter_device() {
    gpu::RuntimeDesc runtime_desc = {
        .backend = gpu::BackendKind.VULKAN,
    };
    gpu::Runtime runtime = {};
    gpu::AdapterList adapters = {};
    gpu::Adapter adapter = {};
    gpu::DeviceRequest request = gpu::strict_device_request();
    gpu::QueueRequirements queues = {
        .counts = {
            .graphics = 1,
            .compute  = 1,
            .transfer = 1,
        },
    };
    gpu::Device device = {};

    (void)gpu::create_runtime(&runtime_desc);
    (void)gpu::enumerate_adapters(&runtime);
    (void)adapters.get(0);
    (void)gpu::get_adapter_info(&adapter);
    (void)gpu::get_adapter_diagnostics(&adapter);
    (void)gpu::request_queues(request, queues);
    (void)gpu::supports_device_request(&adapter, &request);
    (void)gpu::create_device(&adapter, &request);
    (void)gpu::get_device_backend(&device);
    (void)gpu::get_device_caps(&device);
    (void)gpu::get_queue_counts(&device);
    (void)gpu::get_queue(&device, gpu::QueueKind.GRAPHICS);
    (void)gpu::destroy_device(&device);
    (void)gpu::destroy_runtime(&runtime);
}

fn void pin_memory_and_resources() {
    gpu::Device device = {};
    gpu::AllocationDesc allocation_desc = {
        .size         = 256,
        .alignment    = 16,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .graphics, .compute, .transfer },
    };
    gpu::GpuAllocation allocation = {};
    gpu::GpuSpan span = {};
    gpu::TextureDesc texture_desc = {
        .width  = 1,
        .height = 1,
        .depth  = 1,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled },
        .access = { .graphics, .compute },
    };
    gpu::TextureHandle texture = {};
    gpu::TextureViewDesc view_desc = {};
    gpu::TextureView view = {};
    gpu::SamplerDesc sampler_desc = {};
    gpu::Sampler sampler = {};

    (void)gpu::allocate_memory(&device, &allocation_desc);
    (void)gpu::get_allocation_info(&device, allocation);
    (void)gpu::get_allocation_span(&device, allocation);
    (void)gpu::get_span_mapping(&device, span);
    (void)gpu::get_span_address(&device, span);
    (void)gpu::flush_mapped_span(&device, span);
    (void)gpu::invalidate_mapped_span(&device, span);
    (void)gpu::get_texture_requirements(&device, &texture_desc);
    (void)gpu::create_placed_texture(
        device:     &device,
        desc:       &texture_desc,
        allocation: allocation,
        offset:     0,
    );
    (void)gpu::create_dedicated_texture(
        device:          &device,
        desc:            &texture_desc,
        allocation_desc: &allocation_desc,
    );
    (void)gpu::create_texture_view(&device, texture, &view_desc);
    (void)gpu::intern_sampler(&device, &sampler_desc);
    (void)gpu::publish_sampler(&device, sampler);
    (void)gpu::destroy_texture_view(&device, view);
    (void)gpu::destroy_texture(&device, texture);
    (void)gpu::free_allocation(&device, &allocation);
}

fn void pin_commands_and_completion() {
    gpu::Queue queue = {};
    gpu::CommandList commands = {};
    gpu::ExecutableCommandList executable = {};
    gpu::GpuSpan span = {};
    gpu::BufferCopyDesc copy = {
        .src = span,
        .dst = span,
    };
    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
    };
    gpu::SubmitDesc submit = {};
    gpu::CompletionPoint completion = {};

    (void)gpu::begin_commands(queue);
    (void)gpu::cmd_copy_buffer(&commands, &copy);
    (void)gpu::cmd_fill_buffer(&commands, span, 0);
    (void)gpu::cmd_barrier(&commands, &barrier);
    (void)gpu::cmd_dispatch(
        commands: &commands,
        root:     {},
        groups:   { 1, 1, 1 },
    );
    (void)gpu::end_commands(&commands);
    (void)gpu::discard_commands(&commands);
    (void)gpu::submit(queue, &submit);
    (void)gpu::poll_completion(completion);
    (void)gpu::wait_completion(completion);
    (void)gpu::discard_executable_commands(&executable);
}

fn void pin_presentation() {
    gpu::DeviceRequest request = gpu::strict_device_request();
    gpu::Surface surface = {};
    gpu::Device device = {};
    gpu::SwapchainDesc desc = {
        .width  = 1,
        .height = 1,
    };
    gpu::SwapchainHandle swapchain = {};
    gpu::AcquiredImage image = {};
    gpu::CompletionPoint completion = {};

    (void)gpu::request_presentation(request, &surface);
    (void)gpu::create_swapchain(&device, &surface, &desc);
    (void)gpu::get_swapchain_info(&device, swapchain);
    (void)gpu::get_present_mode_support(&device, swapchain);
    (void)gpu::acquire_next_image(&device, swapchain);
    (void)gpu::present(&device, &image, completion);
    (void)gpu::resize_swapchain(&device, swapchain, 1, 1);
    (void)gpu::destroy_swapchain(&device, swapchain);
}
