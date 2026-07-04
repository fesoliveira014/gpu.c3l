module offscreen_triangle;

import gpu;
import std::io;
import std::io::file;
import std::core::mem;

const char[*] OFFSCREEN_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] OFFSCREEN_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");

const uint RT_SIZE = 256;
const usz RT_BYTES = RT_SIZE * RT_SIZE * 4;
const uint TEX_SIZE = 16;
const usz PPM_HEADER_MAX = 32;

fn int main() {
    if (catch err = run()) {
        io::printfn("offscreen_triangle: FAIL (%s)", err);
        return 1;
    }
    io::printn("offscreen_triangle: PASS (wrote offscreen_triangle.ppm)");
    return 0;
}

fn void? run() {
    gpu::DeviceDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "offscreen_triangle",
    };
    gpu::Device device = gpu::create_device(&device_desc)!;
    defer (void)gpu::destroy_device(&device);

    gpu::TextureDesc rt_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = RT_SIZE,
        .height     = RT_SIZE,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .color_attach, .transfer_src },
        .debug_name = "render_target",
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &rt_desc)!;
    defer (void)gpu::destroy_texture(&device, target);

    gpu::TextureDesc tex_desc = {
        .dimension  = gpu::TextureDimension.TEX_2D,
        .width      = TEX_SIZE,
        .height     = TEX_SIZE,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .sampled, .transfer_dst },
        .debug_name = "checker",
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

    gpu::BufferDesc rb_desc = {
        .size = RT_BYTES, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK,
    };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &rb_desc)!;
    defer (void)gpu::destroy_buffer(&device, readback);

    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = OFFSCREEN_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle vert = gpu::create_shader(&device, &vert_sd)!;
    defer (void)gpu::destroy_shader(&device, vert);
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = OFFSCREEN_FRAG_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle frag = gpu::create_shader(&device, &frag_sd)!;
    defer (void)gpu::destroy_shader(&device, frag);

    gpu::Format[1] color_formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc pipe_desc = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = color_formats[..],
        .debug_name      = "offscreen_triangle",
    };
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(&device, &pipe_desc)!;
    defer (void)gpu::destroy_pipeline(&device, pipeline);

    gpu::begin_frame(&device)!;

    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.0f, 0.0f,
         0.8f, -0.8f, 1.0f, 0.0f,
         0.0f,  0.8f, 0.5f, 1.0f,
    };
    mem::copy(verts_span.cpu, &vertex_data, vertex_data.len * float::size);

    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, FragRoot::size, 16)!;
    FragRoot* frag_root = (FragRoot*)frag_span.cpu;
    frag_root.texture_index = tex_index;
    frag_root.sampler_index = smp_index;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!;

    gpu::TextureBarrier checker_to_dst = {
        .texture       = checker,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.TRANSFER_DST,
    };
    gpu::cmd_texture_barrier(&cmd, &checker_to_dst)!;
    gpu::BufferTextureCopyDesc upload = { .src = staging_span, .texture = checker };
    gpu::cmd_copy_buffer_to_texture(&cmd, &upload)!;
    gpu::TextureBarrier checker_to_read = {
        .texture       = checker,
        .before_stage  = gpu::Stage.TRANSFER,
        .after_stage   = gpu::Stage.FRAGMENT_SHADER,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE,
        .after_hazard  = gpu::Hazard.SHADER_READ,
        .old_layout    = gpu::TextureLayout.TRANSFER_DST,
        .new_layout    = gpu::TextureLayout.SHADER_READ,
    };
    gpu::cmd_texture_barrier(&cmd, &checker_to_read)!;

    gpu::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!;

    gpu::ColorTargetDesc[1] colors = {{
        .texture = target,
        .load_op = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear = { .rgba = { 0.05f, 0.05f, 0.1f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = RT_SIZE, .height = RT_SIZE };
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

    gpu::TextureBarrier to_src = {
        .texture       = target,
        .before_stage  = gpu::Stage.COLOR_ATTACHMENT,
        .after_stage   = gpu::Stage.TRANSFER,
        .before_hazard = gpu::Hazard.COLOR_WRITE,
        .after_hazard  = gpu::Hazard.TRANSFER_READ,
        .old_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
        .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
    };
    gpu::cmd_texture_barrier(&cmd, &to_src)!;

    gpu::GpuSpan rb_span = gpu::get_buffer_span(&device, readback)!;
    gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = rb_span };
    gpu::cmd_copy_texture_to_buffer(&cmd, &copy)!;
    gpu::BufferBarrier host_read = {
        .buffer = readback, .offset = 0, .size = RT_BYTES,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!;

    gpu::end_commands(&device, &cmd)!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!;
    gpu::invalidate_buffer(&device, readback, 0, 0)!;

    write_ppm((char*)rb_span.cpu)!;
    gpu::end_frame(&device)!;
}

fn void? write_ppm(char* rgba) {
    @pool() {
        char[] out = mem::talloc_array(char, (sz)(PPM_HEADER_MAX + RT_SIZE * RT_SIZE * 3));
        String header = string::tformat("P6\n%d %d\n255\n", RT_SIZE, RT_SIZE);
        mem::copy(out.ptr, header.ptr, header.len);
        usz cursor = header.len;
        for (uint i = 0; i < RT_SIZE * RT_SIZE; i++) {
            out[cursor++] = rgba[i * 4];
            out[cursor++] = rgba[i * 4 + 1];
            out[cursor++] = rgba[i * 4 + 2];
        }
        file::save("offscreen_triangle.ppm", out[:cursor])!;
    };
}
