module gpu_driven_draw_sdl;

import gpu;
import sample_window;
import std::io;

const char[*] BUILD_SPIRV = $embed("shaders/build_draws.comp.spv");
const char[*] VERT_SPIRV  = $embed("shaders/gpu_driven.vert.spv");
const char[*] FRAG_SPIRV  = $embed("shaders/gpu_driven.frag.spv");

const uint GRID = 16;
const uint INSTANCES = GRID * GRID;
const uint MAX_TRACKED_IMAGES = 8;

struct Options {
    long frames;
    bool vsync;
}

fn void? recover_swapchain(gpu::Device* device, gpu::SwapchainHandle swapchain, sample_window::SampleWindow* window, uint* width, uint* height, bool[] image_seen) {
    sample_window::pixel_size(window, width, height);
    gpu::resize_swapchain(device, swapchain, *width, *height)!;
    foreach (&seen : image_seen) *seen = false;
}

fn Options parse_options(String[] args) {
    Options options = { .frames = -1, .vsync = true };
    foreach (i, arg : args) {
        if (arg == "--no-vsync") options.vsync = false;
        if (arg == "--frames" && i + 1 < args.len) {
            if (try value = args[i + 1].to_long()) options.frames = value;
        }
    }
    return options;
}

fn int main(String[] args) {
    if (catch err = run(parse_options(args))) {
        io::printfn("gpu_driven_draw_sdl: FAIL (%s)", err);
        return 1;
    }
    io::printn("gpu_driven_draw_sdl: PASS");
    return 0;
}

fn void? run(Options options) {
    sample_window::SampleWindow window = sample_window::create_sample_window("gpu.c3l gpu-driven draws", 800, 600)!;
    defer sample_window::destroy_sample_window(&window);

    gpu::DeviceDesc device_desc = {
        .backend             = gpu::BackendKind.VULKAN,
        .enable_validation   = true,
        .enable_presentation = true,
        .frames_in_flight    = 2,
        .application_name    = "gpu_driven_draw_sdl",
    };
    gpu::Device device = gpu::create_device(&device_desc)!;
    defer (void)gpu::destroy_device(&device);
    bool use_count = device.caps.draw_indirect_count;
    io::printfn("gpu_driven_draw_sdl: count-buffer path %s", use_count ? "on" : "off");

    gpu::SurfaceDesc surface = sample_window::surface_desc_for(&window)!;
    uint width;
    uint height;
    sample_window::pixel_size(&window, &width, &height);
    gpu::SwapchainDesc swapchain_desc = {
        .width        = width,
        .height       = height,
        .present_mode = options.vsync ? gpu::PresentMode.FIFO : gpu::PresentMode.MAILBOX,
        .srgb         = false,
        .debug_name   = "gpu_driven_swapchain",
    };
    gpu::SwapchainHandle swapchain = gpu::create_swapchain(&device, &surface, &swapchain_desc)!;
    defer (void)gpu::destroy_swapchain(&device, swapchain);

    gpu::ShaderDesc build_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = BUILD_SPIRV[..], .entry_point = "main", .debug_name = "build_draws" };
    gpu::ShaderHandle build_shader = gpu::create_shader(&device, &build_sd)!;
    defer (void)gpu::destroy_shader(&device, build_shader);
    gpu::ComputePipelineDesc build_desc = { .shader = build_shader, .push_constant_size = gpu::RootPush::size, .debug_name = "build_draws" };
    gpu::PipelineHandle build_pipeline = gpu::create_compute_pipeline(&device, &build_desc)!;
    defer (void)gpu::destroy_pipeline(&device, build_pipeline);

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = VERT_SPIRV[..], .entry_point = "main", .debug_name = "gpu_driven_vert" };
    gpu::ShaderHandle vert = gpu::create_shader(&device, &vert_sd)!;
    defer (void)gpu::destroy_shader(&device, vert);
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = FRAG_SPIRV[..], .entry_point = "main", .debug_name = "gpu_driven_frag" };
    gpu::ShaderHandle frag = gpu::create_shader(&device, &frag_sd)!;
    defer (void)gpu::destroy_shader(&device, frag);

    gpu::Format[1] color_formats = { gpu::Format.BGRA8_UNORM };
    gpu::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = color_formats[..],
        .debug_name      = "gpu_driven_draw",
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(&device, &pipe_desc)!;
    defer (void)gpu::destroy_pipeline(&device, pipeline);

    bool[MAX_TRACKED_IMAGES] image_seen;
    long frame = 0;
    bool running = true;
    while (running) {
        sample_window::WindowEvents events = sample_window::pump_events(&window);
        if (events.quit) break;
        if (events.resized) {
            recover_swapchain(
                device:     &device,
                swapchain:  swapchain,
                window:     &window,
                width:      &width,
                height:     &height,
                image_seen: image_seen[..],
            )!;
        }

        gpu::begin_frame(&device)!;

        gpu::AcquiredImage acquired;
        if (try image = gpu::acquire_next_image(&device, swapchain)) {
            acquired = image;
        } else {
            recover_swapchain(
                device:     &device,
                swapchain:  swapchain,
                window:     &window,
                width:      &width,
                height:     &height,
                image_seen: image_seen[..],
            )!;
            gpu::end_frame(&device)!;
            continue;
        }

        float time = (float)frame / 60.0f;

        gpu::GpuSpan instances_span = gpu::alloc_frame_span(&device, INSTANCES * Instance::size, 16)!;
        Instance* instances = (Instance*)instances_span.cpu;
        for (uint y = 0; y < GRID; y++) {
            for (uint x = 0; x < GRID; x++) {
                uint i = y * GRID + x;
                float fx = ((float)x + 0.5f) / GRID * 2.0f - 1.0f;
                float fy = ((float)y + 0.5f) / GRID * 2.0f - 1.0f;
                instances[i] = {
                    .pos   = { fx, fy },
                    .scale = 0.04f,
                    .color = { (float)x / GRID, (float)y / GRID, 1.0f - (float)x / GRID, 1.0f },
                };
            }
        }

        gpu::GpuSpan corners_span = gpu::alloc_frame_span(&device, 4 * 8, 16)!;
        float* corners = (float*)corners_span.cpu;
        float[8] corner_data = { -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f };
        for (uint i = 0; i < 8; i++) corners[i] = corner_data[i];

        gpu::GpuSpan index_span = gpu::alloc_frame_span(&device, 6 * 4, 4)!;
        uint* indices = (uint*)index_span.cpu;
        uint[6] index_data = { 0, 1, 2, 2, 1, 3 };
        for (uint i = 0; i < 6; i++) indices[i] = index_data[i];

        gpu::GpuSpan args_span = gpu::alloc_frame_span(&device, INSTANCES * gpu::DrawIndexedIndirectCommand::size, 16)!;
        gpu::GpuSpan count_span = gpu::alloc_frame_span(&device, 4, 4)!;

        gpu::GpuSpan build_root_span = gpu::alloc_frame_span(&device, BuildRoot::size, 16)!;
        BuildRoot* build_root = (BuildRoot*)build_root_span.cpu;
        build_root.instances_gpu  = instances_span.gpu;
        build_root.args_gpu       = args_span.gpu;
        build_root.count_gpu      = count_span.gpu;
        build_root.instance_count = INSTANCES;
        build_root.time           = time;

        gpu::GpuSpan draw_root_span = gpu::alloc_frame_span(&device, DrawRoot::size, 16)!;
        DrawRoot* draw_root = (DrawRoot*)draw_root_span.cpu;
        draw_root.corners_gpu   = corners_span.gpu;
        draw_root.instances_gpu = instances_span.gpu;

        gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!;

        gpu::cmd_dispatch(
            commands: &cmd,
            pipeline: build_pipeline,
            root:     build_root_span.gpu,
            groups:   { (INSTANCES + BUILD_WORKGROUP - 1) / BUILD_WORKGROUP, 1, 1 },
        )!;

        gpu::BufferBarrier args_ready = {
            .buffer        = args_span.buffer,
            .offset        = args_span.offset,
            .size          = args_span.size,
            .before_stage  = gpu::Stage.COMPUTE_SHADER,
            .after_stage   = gpu::Stage.INDIRECT_COMMAND,
            .before_hazard = gpu::Hazard.SHADER_WRITE,
            .after_hazard  = gpu::Hazard.INDIRECT_READ,
        };
        gpu::cmd_buffer_barrier(&cmd, &args_ready)!;
        gpu::BufferBarrier count_ready = {
            .buffer        = count_span.buffer,
            .offset        = count_span.offset,
            .size          = count_span.size,
            .before_stage  = gpu::Stage.COMPUTE_SHADER,
            .after_stage   = gpu::Stage.INDIRECT_COMMAND,
            .before_hazard = gpu::Hazard.SHADER_WRITE,
            .after_hazard  = gpu::Hazard.INDIRECT_READ,
        };
        gpu::cmd_buffer_barrier(&cmd, &count_ready)!;

        gpu::TextureBarrier to_attach = {
            .texture       = acquired.texture,
            .before_stage  = image_seen[acquired.index] ? gpu::Stage.PRESENT : gpu::Stage.HOST,
            .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
            .before_hazard = image_seen[acquired.index] ? gpu::Hazard.PRESENT_READ : gpu::Hazard.HOST_WRITE,
            .after_hazard  = gpu::Hazard.COLOR_WRITE,
            .old_layout    = image_seen[acquired.index] ? gpu::TextureLayout.PRESENT : gpu::TextureLayout.UNDEFINED,
            .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
        };
        gpu::cmd_texture_barrier(&cmd, &to_attach)!;
        image_seen[acquired.index] = true;

        gpu::ColorTargetDesc[1] colors = {{
            .texture = acquired.texture,
            .load_op = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear = { .rgba = { 0.03f, 0.03f, 0.08f, 1.0f } },
        }};
        gpu::RenderPassDesc pass = { .colors = colors[..], .width = width, .height = height };
        gpu::cmd_begin_render_pass(&cmd, &pass)!;
        if (use_count) {
            gpu::cmd_draw_indexed_indirect_count(
                commands:       &cmd,
                pipeline:       pipeline,
                vertex_root:    draw_root_span.gpu,
                fragment_root:  (gpu::GpuAddress)0,
                args:           args_span,
                count_span:     count_span,
                max_draw_count: INSTANCES,
                index_span:     index_span,
            )!;
        } else {
            gpu::cmd_draw_indexed_indirect(
                commands:      &cmd,
                pipeline:      pipeline,
                vertex_root:   draw_root_span.gpu,
                fragment_root: (gpu::GpuAddress)0,
                args:          args_span,
                draw_count:    INSTANCES,
                index_span:    index_span,
            )!;
        }
        gpu::cmd_end_render_pass(&cmd)!;

        gpu::TextureBarrier to_present = {
            .texture       = acquired.texture,
            .before_stage  = gpu::Stage.COLOR_ATTACHMENT,
            .after_stage   = gpu::Stage.PRESENT,
            .before_hazard = gpu::Hazard.COLOR_WRITE,
            .after_hazard  = gpu::Hazard.PRESENT_READ,
            .old_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
            .new_layout    = gpu::TextureLayout.PRESENT,
        };
        gpu::cmd_texture_barrier(&cmd, &to_present)!;

        gpu::end_commands(&device, &cmd)!;
        gpu::CommandList[1] lists = { cmd };
        gpu::SubmitDesc submit = { .command_lists = lists[..], .swapchain = swapchain };
        gpu::submit(&device, &submit)!;

        gpu::PresentDesc present_desc = { .swapchain = swapchain };
        if (catch err = gpu::present(&device, &present_desc)) {
            if (err != gpu::SWAPCHAIN_OUT_OF_DATE) return err~;
            recover_swapchain(
                device:     &device,
                swapchain:  swapchain,
                window:     &window,
                width:      &width,
                height:     &height,
                image_seen: image_seen[..],
            )!;
        }

        gpu::end_frame(&device)!;

        frame++;
        if (options.frames > 0 && frame >= options.frames) running = false;
    }

    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!;
}
