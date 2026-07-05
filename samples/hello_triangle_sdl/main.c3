module hello_triangle_sdl;

import gpu;
import sample_window;
import std::io;

const char[*] TRIANGLE_VERT_SPIRV = $embed("../offscreen_triangle/shaders/offscreen.vert.spv");
const char[*] TRIANGLE_FRAG_SPIRV = $embed("../offscreen_triangle/shaders/offscreen.frag.spv");

const uint TEX_SIZE = 16;
const uint MAX_TRACKED_IMAGES = 8;

struct Options {
    long frames;
    bool vsync;
}

<* Re-query the window size, recreate the swapchain, and forget per-image
   layout tracking (fresh images start UNDEFINED). *>
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
        io::printfn("hello_triangle_sdl: FAIL (%s)", err);
        return 1;
    }
    io::printn("hello_triangle_sdl: PASS");
    return 0;
}

fn void? run(Options options) {
    sample_window::SampleWindow window = sample_window::create_sample_window("gpu.c3l hello triangle", 800, 600)!;
    defer sample_window::destroy_sample_window(&window);

    gpu::DeviceDesc device_desc = {
        .backend             = gpu::BackendKind.VULKAN,
        .enable_validation   = true,
        .enable_presentation = true,
        .frames_in_flight    = 2,
        .application_name    = "hello_triangle_sdl",
    };
    gpu::Device device = gpu::create_device(&device_desc)!;
    defer (void)gpu::destroy_device(&device);

    gpu::SurfaceDesc surface = sample_window::surface_desc_for(&window)!;
    uint width;
    uint height;
    sample_window::pixel_size(&window, &width, &height);
    gpu::SwapchainDesc swapchain_desc = {
        .width        = width,
        .height       = height,
        .present_mode = options.vsync ? gpu::PresentMode.FIFO : gpu::PresentMode.MAILBOX,
        .srgb         = false,
        .debug_name   = "hello_triangle_swapchain",
    };
    gpu::SwapchainHandle swapchain = gpu::create_swapchain(&device, &surface, &swapchain_desc)!;
    defer (void)gpu::destroy_swapchain(&device, swapchain);

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = TEX_SIZE,
        .height    = TEX_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle checker = gpu::create_texture(&device, &tex_desc)!;
    defer (void)gpu::destroy_texture(&device, checker);
    gpu::TextureIndex tex_index = gpu::create_texture_descriptor(&device, checker, null)!;
    defer (void)gpu::destroy_texture_descriptor(&device, tex_index);

    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex smp_index = gpu::create_sampler(&device, &sampler_desc)!;
    defer (void)gpu::destroy_sampler(&device, smp_index);

    gpu::BufferDesc staging_desc = {
        .size = TEX_SIZE * TEX_SIZE * 4, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.STAGING,
    };
    gpu::BufferHandle staging = gpu::create_buffer(&device, &staging_desc)!;
    defer (void)gpu::destroy_buffer(&device, staging);
    gpu::GpuSpan staging_span = gpu::get_buffer_span(&device, staging)!;
    char* texels = (char*)staging_span.cpu;
    for (uint y = 0; y < TEX_SIZE; y++) {
        for (uint x = 0; x < TEX_SIZE; x++) {
            bool checkered = ((x / 4) + (y / 4)) % 2 == 0;
            char* texel = &texels[(y * TEX_SIZE + x) * 4];
            texel[0] = checkered ? (char)230 : (char)40;
            texel[1] = checkered ? (char)180 : (char)90;
            texel[2] = checkered ? (char)60 : (char)200;
            texel[3] = (char)255;
        }
    }
    gpu::flush_buffer(&device, staging, 0, 0)!;

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = TRIANGLE_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle vert = gpu::create_shader(&device, &vert_sd)!;
    defer (void)gpu::destroy_shader(&device, vert);
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = TRIANGLE_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle frag = gpu::create_shader(&device, &frag_sd)!;
    defer (void)gpu::destroy_shader(&device, frag);

    // Match the swapchain's chosen format; BGRA8_UNORM is what most surfaces
    // pick for a non-srgb request.
    gpu::Format[1] color_formats = { gpu::Format.BGRA8_UNORM };
    gpu::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = color_formats[..],
        .debug_name      = "hello_triangle",
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(&device, &pipe_desc)!;
    defer (void)gpu::destroy_pipeline(&device, pipeline);

    bool checker_uploaded = false;
    // Tracks which swapchain images have been presented, so the pre-render
    // barrier uses PRESENT (not UNDEFINED) as the tracked old layout.
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

        if (frame == 0) {
            // Swapchain-owned images must reject destroy_texture.
            fault owned_fault = @catch(gpu::destroy_texture(&device, acquired.texture));
            if (owned_fault != gpu::INVALID_ARGUMENT) return gpu::INVALID_RESOURCE_STATE~;
        }

        gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!;

        if (!checker_uploaded) {
            gpu::TextureBarrier to_dst = {
                .texture       = checker,
                .before_stage  = gpu::Stage.HOST,
                .after_stage   = gpu::Stage.TRANSFER,
                .before_hazard = gpu::Hazard.HOST_WRITE,
                .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
                .old_layout    = gpu::TextureLayout.UNDEFINED,
                .new_layout    = gpu::TextureLayout.TRANSFER_DST,
            };
            gpu::cmd_texture_barrier(&cmd, &to_dst)!;
            gpu::BufferTextureCopyDesc upload = { .src = staging_span, .texture = checker };
            gpu::cmd_copy_buffer_to_texture(&cmd, &upload)!;
            gpu::TextureBarrier to_read = {
                .texture       = checker,
                .before_stage  = gpu::Stage.TRANSFER,
                .after_stage   = gpu::Stage.FRAGMENT_SHADER,
                .before_hazard = gpu::Hazard.TRANSFER_WRITE,
                .after_hazard  = gpu::Hazard.SHADER_READ,
                .old_layout    = gpu::TextureLayout.TRANSFER_DST,
                .new_layout    = gpu::TextureLayout.SHADER_READ,
            };
            gpu::cmd_texture_barrier(&cmd, &to_read)!;
            checker_uploaded = true;
        }

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

        float pulse = (float)(frame % 120) / 120.0f * 0.15f;
        gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!;
        float* verts = (float*)verts_span.cpu;
        float[12] vertex_data = {
            -0.8f, -0.8f, 0.0f, 0.0f,
             0.8f, -0.8f, 1.0f, 0.0f,
             0.0f,  0.8f, 0.5f, 1.0f,
        };
        for (uint i = 0; i < 12; i++) verts[i] = vertex_data[i];

        gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, FragRoot::size, 16)!;
        FragRoot* frag_root = (FragRoot*)frag_span.cpu;
        frag_root.texture_index = tex_index;
        frag_root.sampler_index = smp_index;

        gpu::ColorTargetDesc[1] colors = {{
            .texture = acquired.texture,
            .load_op = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear = { .rgba = { 0.05f + pulse, 0.05f, 0.12f, 1.0f } },
        }};
        gpu::RenderPassDesc pass = { .colors = colors[..], .width = width, .height = height };
        gpu::cmd_begin_render_pass(&cmd, &pass)!;
        gpu::cmd_draw(
            commands:       &cmd,
            pipeline:       pipeline,
            vertex_root:    verts_span.gpu,
            fragment_root:  frag_span.gpu,
            vertex_count:   3,
            instance_count: 1,
        )!;
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
