module gpu_test;

import gpu::compat @public;
import gpu::compat::vk @public;
import vk;

// Headless (lavapipe): pipeline dedup cache. Entry accounting across aliases,
// refcounted destroy, shared layouts, dynamic-state replay proven by drawing
// through two aliases of one cache entry with different cull state.

const char[*] CACHE_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] CACHE_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");
const char[*] CACHE_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

const uint CACHE_RT_SIZE = 16;
const usz CACHE_RT_BYTES = CACHE_RT_SIZE * CACHE_RT_SIZE * 4;

fn gpu::compat::Device make_cache_test_device() {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_pipeline_cache_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

fn gpu::compat::Device make_cache_test_device_with_capacity(uint pipeline_capacity) {
    gpu::compat::DeviceDesc desc = {
        .backend           = gpu::compat::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .pipeline_capacity = pipeline_capacity,
        .application_name  = "gpu_pipeline_cache_capacity_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

fn uint cache_entry_count(gpu::compat::Device* device) {
    return vk::pipeline_cache_entry_count((vk::VkDeviceState*)gpu::compat::device_backend_state(*device));
}

fn bool cache_index_consistent(gpu::compat::Device* device) {
    return vk::pipeline_cache_index_consistent((vk::VkDeviceState*)gpu::compat::device_backend_state(*device));
}

struct CacheShaders {
    gpu::compat::ShaderHandle vert;
    gpu::compat::ShaderHandle frag;
}

fn CacheShaders make_cache_shaders(gpu::compat::Device* device) {
    gpu::compat::ShaderDesc vert_sd = { .stage = gpu::compat::ShaderStage.VERTEX, .spirv = CACHE_VERT_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderDesc frag_sd = { .stage = gpu::compat::ShaderStage.FRAGMENT, .spirv = CACHE_FRAG_SPIRV[..], .entry_point = "main" };
    return {
        .vert = gpu::compat::create_shader(device, &vert_sd)!!,
        .frag = gpu::compat::create_shader(device, &frag_sd)!!,
    };
}

fn gpu::compat::GraphicsPipelineDesc base_graphics_desc(CacheShaders* shaders, gpu::compat::Format[] color_formats) {
    return {
        .vertex_shader   = shaders.vert,
        .fragment_shader = shaders.frag,
        .topology        = gpu::compat::PrimitiveTopology.TRIANGLES,
        .color_formats   = color_formats,
    };
}

fn void identical_descs_alias_one_entry() @test {
    gpu::compat::Device device = make_cache_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::compat::destroy_shader(&device, shaders.vert)!!;
    defer gpu::compat::destroy_shader(&device, shaders.frag)!!;

    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, formats[..]);

    gpu::compat::PipelineHandle first = gpu::compat::create_graphics_pipeline(&device, &desc)!!;
    gpu::compat::PipelineHandle second = gpu::compat::create_graphics_pipeline(&device, &desc)!!;
    assert((ulong)first != (ulong)second, "aliases are distinct handles");
    assert(cache_entry_count(&device) == 1, "identical descriptors share one cache entry");

    gpu::compat::destroy_pipeline(&device, first)!!;
    assert(cache_entry_count(&device) == 1, "entry survives while an alias lives");

    assert(@catch(gpu::compat::destroy_pipeline(&device, first)) == gpu::compat::INVALID_HANDLE,
        "double-destroy of the same handle faults INVALID_HANDLE");
    assert(cache_entry_count(&device) == 1, "double-destroy does not disturb the entry");

    gpu::compat::destroy_pipeline(&device, second)!!;
    assert(cache_entry_count(&device) == 0, "last alias releases the entry");
}

fn void dynamic_state_and_blend_normalization_share_entries() @test {
    gpu::compat::Device device = make_cache_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::compat::destroy_shader(&device, shaders.vert)!!;
    defer gpu::compat::destroy_shader(&device, shaders.frag)!!;

    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, formats[..]);
    gpu::compat::PipelineHandle base = gpu::compat::create_graphics_pipeline(&device, &desc)!!;
    defer gpu::compat::destroy_pipeline(&device, base)!!;

    gpu::compat::GraphicsPipelineDesc culled = desc;
    culled.raster.cull_mode = gpu::compat::CullMode.BACK;
    culled.depth = { .test_enable = true, .write_enable = true, .compare = gpu::compat::CompareOp.LESS };
    gpu::compat::PipelineHandle variant = gpu::compat::create_graphics_pipeline(&device, &culled)!!;
    defer gpu::compat::destroy_pipeline(&device, variant)!!;
    assert(cache_entry_count(&device) == 1, "raster/depth-only variants share one entry");

    gpu::compat::GraphicsPipelineDesc garbage_blend = desc;
    garbage_blend.blend = {
        .enable    = false,
        .src_color = gpu::compat::BlendFactor.DST_ALPHA,
        .dst_color = gpu::compat::BlendFactor.SRC_ALPHA,
        .color_op  = gpu::compat::BlendOp.MAX,
    };
    gpu::compat::PipelineHandle normalized = gpu::compat::create_graphics_pipeline(&device, &garbage_blend)!!;
    defer gpu::compat::destroy_pipeline(&device, normalized)!!;
    assert(cache_entry_count(&device) == 1, "disabled blend ignores factor fields in the key");

    gpu::compat::GraphicsPipelineDesc blended = desc;
    blended.blend = {
        .enable    = true,
        .src_color = gpu::compat::BlendFactor.SRC_ALPHA,
        .dst_color = gpu::compat::BlendFactor.ONE_MINUS_SRC_ALPHA,
        .color_op  = gpu::compat::BlendOp.ADD,
        .src_alpha = gpu::compat::BlendFactor.ONE,
        .dst_alpha = gpu::compat::BlendFactor.ZERO,
        .alpha_op  = gpu::compat::BlendOp.ADD,
    };
    gpu::compat::PipelineHandle distinct = gpu::compat::create_graphics_pipeline(&device, &blended)!!;
    defer gpu::compat::destroy_pipeline(&device, distinct)!!;
    assert(cache_entry_count(&device) == 2, "enabled blend is immutable state and splits the key");
}

fn void formats_split_entries_and_limits_enforced() @test {
    gpu::compat::Device device = make_cache_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::compat::destroy_shader(&device, shaders.vert)!!;
    defer gpu::compat::destroy_shader(&device, shaders.frag)!!;

    gpu::compat::Format[1] rgba = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc desc_rgba = base_graphics_desc(&shaders, rgba[..]);
    gpu::compat::PipelineHandle first = gpu::compat::create_graphics_pipeline(&device, &desc_rgba)!!;
    defer gpu::compat::destroy_pipeline(&device, first)!!;

    gpu::compat::Format[1] bgra = { gpu::compat::Format.BGRA8_UNORM };
    gpu::compat::GraphicsPipelineDesc desc_bgra = base_graphics_desc(&shaders, bgra[..]);
    gpu::compat::PipelineHandle second = gpu::compat::create_graphics_pipeline(&device, &desc_bgra)!!;
    defer gpu::compat::destroy_pipeline(&device, second)!!;
    assert(cache_entry_count(&device) == 2, "different formats produce distinct pipelines");

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    vk::PipelineLayout first_layout = state.pipelines.slots[first.index].value.layout;
    vk::PipelineLayout second_layout = state.pipelines.slots[second.index].value.layout;
    assert(first_layout == second_layout, "graphics pipelines share one layout");
    assert(first_layout == state.graphics_layout, "the shared layout is the device graphics layout");

    gpu::compat::Format[gpu::compat::MAX_COLOR_ATTACHMENTS + 1] too_many;
    foreach (&format : too_many) *format = gpu::compat::Format.RGBA8_UNORM;
    gpu::compat::GraphicsPipelineDesc over_limit = base_graphics_desc(&shaders, too_many[..]);
    assert(@catch(gpu::compat::create_graphics_pipeline(&device, &over_limit)) == gpu::compat::INVALID_ARGUMENT,
        "more than MAX_COLOR_ATTACHMENTS color formats faults INVALID_ARGUMENT");
}

fn void compute_pipelines_dedup_and_share_layouts() @test {
    gpu::compat::Device device = make_cache_test_device();
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::ShaderDesc comp_sd = { .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = CACHE_COMP_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle comp = gpu::compat::create_shader(&device, &comp_sd)!!;
    defer gpu::compat::destroy_shader(&device, comp)!!;

    gpu::compat::ComputePipelineDesc desc = { .shader = comp, .push_constant_size = gpu::compat::RootPush::size };
    gpu::compat::PipelineHandle first = gpu::compat::create_compute_pipeline(&device, &desc)!!;
    defer gpu::compat::destroy_pipeline(&device, first)!!;
    gpu::compat::PipelineHandle second = gpu::compat::create_compute_pipeline(&device, &desc)!!;
    defer gpu::compat::destroy_pipeline(&device, second)!!;
    assert((ulong)first != (ulong)second, "compute aliases are distinct handles");
    assert(cache_entry_count(&device) == 1, "identical compute descriptors share one entry");

    gpu::compat::ComputePipelineDesc wider = { .shader = comp, .push_constant_size = 16 };
    gpu::compat::PipelineHandle third = gpu::compat::create_compute_pipeline(&device, &wider)!!;
    defer gpu::compat::destroy_pipeline(&device, third)!!;
    assert(cache_entry_count(&device) == 2, "push size is part of the compute key");

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    assert(state.compute_layout_cache.count == 2, "duplicate push sizes reuse one layout entry");
    vk::PipelineLayout first_layout = state.pipelines.slots[first.index].value.layout;
    vk::PipelineLayout second_layout = state.pipelines.slots[second.index].value.layout;
    vk::PipelineLayout third_layout = state.pipelines.slots[third.index].value.layout;
    assert(first_layout == second_layout, "equal push sizes share a compute layout");
    assert(first_layout != third_layout, "different push sizes use different layouts");
}

fn void cull_replay_renders_per_alias() @test {
    gpu::compat::Device device = make_cache_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::compat::destroy_shader(&device, shaders.vert)!!;
    defer gpu::compat::destroy_shader(&device, shaders.frag)!!;

    // The offscreen fragment shader samples the heap, so a 1x1 white pattern
    // texture and a sampler feed both draws.
    gpu::compat::TextureDesc tex_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = 1,
        .height    = 1,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::compat::TextureHandle pattern = gpu::compat::create_texture(&device, &tex_desc)!!;
    defer gpu::compat::destroy_texture(&device, pattern)!!;
    gpu::compat::TextureIndex tex_index = gpu::compat::create_texture_descriptor(&device, pattern, null)!!;
    defer gpu::compat::destroy_texture_descriptor(&device, tex_index)!!;
    gpu::compat::SamplerDesc sampler_desc = {
        .min_filter = gpu::compat::Filter.NEAREST,
        .mag_filter = gpu::compat::Filter.NEAREST,
        .address_u  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::compat::SamplerIndex smp_index = gpu::compat::create_sampler(&device, &sampler_desc)!!;
    defer gpu::compat::destroy_sampler(&device, smp_index)!!;

    char[4] white = { (char)255, (char)255, (char)255, (char)255 };
    gpu::compat::upload_texture_data(
        device:      &device,
        dst:         pattern,
        mip:         0,
        data:        white[..],
        next_stage:  gpu::compat::Stage.FRAGMENT_SHADER,
        next_hazard: gpu::compat::Hazard.SHADER_READ,
        next_layout: gpu::compat::TextureLayout.SHADER_READ,
    )!!;

    gpu::compat::TextureDesc rt_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = CACHE_RT_SIZE,
        .height    = CACHE_RT_SIZE,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    gpu::compat::TextureHandle target_culled = gpu::compat::create_texture(&device, &rt_desc)!!;
    defer gpu::compat::destroy_texture(&device, target_culled)!!;
    gpu::compat::TextureHandle target_drawn = gpu::compat::create_texture(&device, &rt_desc)!!;
    defer gpu::compat::destroy_texture(&device, target_drawn)!!;
    gpu::compat::TextureHandle target_reset = gpu::compat::create_texture(&device, &rt_desc)!!;
    defer gpu::compat::destroy_texture(&device, target_reset)!!;

    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc no_cull_desc = base_graphics_desc(&shaders, formats[..]);
    gpu::compat::PipelineHandle no_cull = gpu::compat::create_graphics_pipeline(&device, &no_cull_desc)!!;
    defer gpu::compat::destroy_pipeline(&device, no_cull)!!;

    gpu::compat::GraphicsPipelineDesc cull_desc = no_cull_desc;
    cull_desc.raster.cull_mode = gpu::compat::CullMode.BACK;
    gpu::compat::PipelineHandle culled = gpu::compat::create_graphics_pipeline(&device, &cull_desc)!!;
    defer gpu::compat::destroy_pipeline(&device, culled)!!;
    assert(cache_entry_count(&device) == 1, "both draw pipelines alias one cache entry");

    gpu::compat::BufferDesc rb_desc = {
        .size = CACHE_RT_BYTES, .usage = { .transfer_dst }, .memory_kind = gpu::compat::MemoryKind.READBACK,
    };
    gpu::compat::BufferHandle rb_culled = gpu::compat::create_buffer(&device, &rb_desc)!!;
    defer gpu::compat::destroy_buffer(&device, rb_culled)!!;
    gpu::compat::BufferHandle rb_drawn = gpu::compat::create_buffer(&device, &rb_desc)!!;
    defer gpu::compat::destroy_buffer(&device, rb_drawn)!!;
    gpu::compat::BufferHandle rb_reset = gpu::compat::create_buffer(&device, &rb_desc)!!;
    defer gpu::compat::destroy_buffer(&device, rb_reset)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;

    gpu::compat::GpuSpan verts_span = gpu::compat::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
    float* verts = (float*)verts_span.cpu;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = vertex_data[i];

    gpu::compat::GpuSpan frag_span = gpu::compat::alloc_frame_span(&frame_token, FragRoot::size, 16)!!;
    FragRoot* frag_root = (FragRoot*)frag_span.cpu;
    frag_root.texture_index = tex_index;
    frag_root.sampler_index = smp_index;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::TextureHandle[3] targets = { target_culled, target_drawn, target_reset };
    gpu::compat::PipelineHandle[3] pipelines = { culled, no_cull, no_cull };
    gpu::compat::BufferHandle[3] readbacks = { rb_culled, rb_drawn, rb_reset };
    for (uint i = 0; i < targets.len; i++) {
        gpu::compat::TextureBarrier to_attach = {
            .texture       = targets[i],
            .before_stage  = gpu::compat::Stage.HOST,
            .after_stage   = gpu::compat::Stage.COLOR_ATTACHMENT,
            .before_hazard = gpu::compat::Hazard.HOST_WRITE,
            .after_hazard  = gpu::compat::Hazard.COLOR_WRITE,
            .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
            .new_layout    = gpu::compat::TextureLayout.COLOR_ATTACHMENT,
        };
        gpu::compat::cmd_texture_barrier(&cmd, &to_attach)!!;

        gpu::compat::ColorTargetDesc[1] colors = {{
            .texture = targets[i],
            .load_op = gpu::compat::LoadOp.CLEAR,
            .store_op = gpu::compat::StoreOp.STORE,
            .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
        }};
        gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = CACHE_RT_SIZE, .height = CACHE_RT_SIZE };
        gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
        if (i == 1) {
            gpu::compat::Viewport viewport = {
                .width = (float)CACHE_RT_SIZE / 2.0f, .height = (float)CACHE_RT_SIZE, .max_depth = 1.0f,
            };
            gpu::compat::ScissorRect scissor = {
                .width = (int)CACHE_RT_SIZE / 2, .height = (int)CACHE_RT_SIZE,
            };
            gpu::compat::cmd_set_viewport(&cmd, &viewport)!!;
            gpu::compat::cmd_set_scissor(&cmd, &scissor)!!;
        }
        gpu::compat::cmd_draw(
            commands:       &cmd,
            pipeline:       pipelines[i],
            vertex_root:    verts_span.gpu,
            fragment_root:  frag_span.gpu,
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::compat::cmd_end_render_pass(&cmd)!!;

        gpu::compat::TextureBarrier to_src = {
            .texture       = targets[i],
            .before_stage  = gpu::compat::Stage.COLOR_ATTACHMENT,
            .after_stage   = gpu::compat::Stage.TRANSFER,
            .before_hazard = gpu::compat::Hazard.COLOR_WRITE,
            .after_hazard  = gpu::compat::Hazard.TRANSFER_READ,
            .old_layout    = gpu::compat::TextureLayout.COLOR_ATTACHMENT,
            .new_layout    = gpu::compat::TextureLayout.TRANSFER_SRC,
        };
        gpu::compat::cmd_texture_barrier(&cmd, &to_src)!!;

        gpu::compat::GpuSpan rb_span = gpu::compat::get_buffer_span(&device, readbacks[i])!!;
        gpu::compat::TextureBufferCopyDesc copy = { .texture = targets[i], .dst = rb_span };
        gpu::compat::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
        gpu::compat::BufferBarrier host_read = {
            .buffer = readbacks[i], .offset = 0, .size = CACHE_RT_BYTES,
            .before_stage = gpu::compat::Stage.TRANSFER, .after_stage = gpu::compat::Stage.HOST,
            .before_hazard = gpu::compat::Hazard.TRANSFER_WRITE, .after_hazard = gpu::compat::Hazard.HOST_READ,
        };
        gpu::compat::cmd_buffer_barrier(&cmd, &host_read)!!;
    }

    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::invalidate_buffer(&device, rb_culled, 0, 0)!!;
    gpu::compat::GpuSpan culled_span = gpu::compat::get_buffer_span(&device, rb_culled)!!;
    char* culled_pixels = (char*)culled_span.cpu;
    char* culled_center = &culled_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE / 2) * 4];
    assert(culled_center[0] == (char)8 && culled_center[1] == (char)16 && culled_center[2] == (char)32,
        "back-face culled draw leaves the clear color");

    gpu::compat::invalidate_buffer(&device, rb_drawn, 0, 0)!!;
    gpu::compat::GpuSpan drawn_span = gpu::compat::get_buffer_span(&device, rb_drawn)!!;
    char* drawn_pixels = (char*)drawn_span.cpu;
    char* drawn_left = &drawn_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE / 4) * 4];
    char* drawn_right = &drawn_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE * 3 / 4) * 4];
    assert(drawn_left[0] == (char)255 && drawn_left[1] == (char)255 && drawn_left[2] == (char)255,
        "pipeline alias switch preserves the explicit viewport and scissor on the left");
    assert(drawn_right[0] == (char)8 && drawn_right[1] == (char)16 && drawn_right[2] == (char)32,
        "pipeline alias switch preserves clipping on the right");

    gpu::compat::invalidate_buffer(&device, rb_reset, 0, 0)!!;
    gpu::compat::GpuSpan reset_span = gpu::compat::get_buffer_span(&device, rb_reset)!!;
    char* reset_pixels = (char*)reset_span.cpu;
    char* reset_center = &reset_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE / 2) * 4];
    assert(reset_center[0] == (char)255 && reset_center[1] == (char)255 && reset_center[2] == (char)255,
        "the next pass restores full viewport and scissor defaults");

    gpu::compat::end_frame(&frame_token)!!;
}

fn void alias_survives_peer_destroy() @test {
    gpu::compat::Device device = make_cache_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::compat::destroy_shader(&device, shaders.vert)!!;
    defer gpu::compat::destroy_shader(&device, shaders.frag)!!;

    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, formats[..]);
    gpu::compat::PipelineHandle doomed = gpu::compat::create_graphics_pipeline(&device, &desc)!!;
    gpu::compat::PipelineHandle survivor = gpu::compat::create_graphics_pipeline(&device, &desc)!!;
    gpu::compat::destroy_pipeline(&device, doomed)!!;

    gpu::compat::TextureDesc rt_desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = CACHE_RT_SIZE,
        .height    = CACHE_RT_SIZE,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::compat::TextureHandle target = gpu::compat::create_texture(&device, &rt_desc)!!;
    defer gpu::compat::destroy_texture(&device, target)!!;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::GpuSpan verts_span = gpu::compat::alloc_frame_span(&frame_token, 3 * 16, 16)!!;
    gpu::compat::GpuSpan frag_span = gpu::compat::alloc_frame_span(&frame_token, 16, 16)!!;

    gpu::compat::CommandList cmd = gpu::compat::begin_commands(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::compat::Stage.HOST,
        .after_stage   = gpu::compat::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::compat::Hazard.HOST_WRITE,
        .after_hazard  = gpu::compat::Hazard.COLOR_WRITE,
        .old_layout    = gpu::compat::TextureLayout.UNDEFINED,
        .new_layout    = gpu::compat::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::compat::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::compat::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::compat::LoadOp.CLEAR, .store_op = gpu::compat::StoreOp.STORE,
    }};
    gpu::compat::RenderPassDesc pass = { .colors = colors[..], .width = CACHE_RT_SIZE, .height = CACHE_RT_SIZE };
    gpu::compat::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::compat::cmd_draw(
        commands:       &cmd,
        pipeline:       survivor,
        vertex_root:    verts_span.gpu,
        fragment_root:  frag_span.gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::compat::cmd_end_render_pass(&cmd)!!;
    gpu::compat::end_commands(&cmd)!!;
    gpu::compat::CommandList[1] lists = { cmd };
    gpu::compat::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::compat::submit(&device, &submit)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;
    gpu::compat::end_frame(&frame_token)!!;

    gpu::compat::destroy_pipeline(&device, survivor)!!;
    assert(cache_entry_count(&device) == 0, "entry released after the surviving alias is destroyed");
}

const uint CHURN_CAPACITY = 8;
// Headroom above CHURN_CAPACITY: pipeline_capacity also sizes the handle slot
// table, and an aliasing create needs a free slot even on a cache hit.
const uint CHURN_DEVICE_CAPACITY = CHURN_CAPACITY + 2;
const uint CHURN_ROUNDS = 4;

fn void index_survives_create_alias_destroy_churn() @test {
    gpu::compat::Device device = make_cache_test_device_with_capacity(CHURN_DEVICE_CAPACITY);
    defer gpu::compat::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::compat::destroy_shader(&device, shaders.vert)!!;
    defer gpu::compat::destroy_shader(&device, shaders.frag)!!;

    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    uint[CHURN_CAPACITY] teardown_order = { 3, 0, 7, 1, 6, 2, 5, 4 };

    for (uint round = 0; round < CHURN_ROUNDS; round++) {
        gpu::compat::PipelineHandle[CHURN_CAPACITY] handles;
        for (uint i = 0; i < CHURN_CAPACITY; i++) {
            gpu::compat::GraphicsPipelineDesc variant = base_graphics_desc(&shaders, formats[..]);
            variant.raster.depth_bias_constant = (float)(round * CHURN_CAPACITY + i + 1);
            handles[i] = gpu::compat::create_graphics_pipeline(&device, &variant)!!;
        }
        assert(cache_index_consistent(&device), "index consistent once all distinct entries are inserted");
        assert(cache_entry_count(&device) == CHURN_CAPACITY, "each round inserts exactly CHURN_CAPACITY distinct entries");

        // Alias two live entries; the refcount>0 path must not touch the index.
        gpu::compat::GraphicsPipelineDesc alias_desc = base_graphics_desc(&shaders, formats[..]);
        alias_desc.raster.depth_bias_constant = (float)(round * CHURN_CAPACITY + 0 + 1);
        gpu::compat::PipelineHandle alias_a = gpu::compat::create_graphics_pipeline(&device, &alias_desc)!!;
        alias_desc.raster.depth_bias_constant = (float)(round * CHURN_CAPACITY + 3 + 1);
        gpu::compat::PipelineHandle alias_b = gpu::compat::create_graphics_pipeline(&device, &alias_desc)!!;
        assert(cache_index_consistent(&device), "index consistent after aliasing existing entries");
        assert(cache_entry_count(&device) == CHURN_CAPACITY, "aliasing does not grow the entry count");

        gpu::compat::destroy_pipeline(&device, alias_a)!!;
        gpu::compat::destroy_pipeline(&device, alias_b)!!;
        assert(cache_index_consistent(&device), "index consistent after alias release");
        assert(cache_entry_count(&device) == CHURN_CAPACITY, "peer entries survive alias release");

        // Destroy the base handles out of order to recycle free-list slots
        // through the index in a different pattern each round.
        foreach (i : teardown_order) {
            gpu::compat::destroy_pipeline(&device, handles[i])!!;
            assert(cache_index_consistent(&device), "index consistent mid-teardown");
        }
        assert(cache_entry_count(&device) == 0, "table empty after full teardown");
    }
}

fn void small_pipeline_capacity_faults_at_limit() @test {
    const uint CAPACITY = 3;
    gpu::compat::Device device = make_cache_test_device_with_capacity(CAPACITY);
    defer gpu::compat::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::compat::destroy_shader(&device, shaders.vert)!!;
    defer gpu::compat::destroy_shader(&device, shaders.frag)!!;

    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    gpu::compat::PipelineHandle[CAPACITY] handles;
    for (uint i = 0; i < CAPACITY; i++) {
        gpu::compat::GraphicsPipelineDesc variant = base_graphics_desc(&shaders, formats[..]);
        variant.raster.depth_bias_constant = (float)(i + 1);
        handles[i] = gpu::compat::create_graphics_pipeline(&device, &variant)!!;
    }
    assert(cache_entry_count(&device) == CAPACITY, "the small capacity fills exactly");
    assert(cache_index_consistent(&device), "index consistent at capacity");

    gpu::compat::GraphicsPipelineDesc over_capacity = base_graphics_desc(&shaders, formats[..]);
    over_capacity.raster.depth_bias_constant = (float)(CAPACITY + 1);
    assert(@catch(gpu::compat::create_graphics_pipeline(&device, &over_capacity)) == gpu::compat::SLOT_TABLE_FULL,
        "the distinct pipeline past the configured capacity faults SLOT_TABLE_FULL");

    foreach (handle : handles) gpu::compat::destroy_pipeline(&device, handle)!!;
}

fn void line_polygon_mode_respects_optional_feature() @test {
    gpu::compat::Device device = make_cache_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::compat::destroy_shader(&device, shaders.vert)!!;
    defer gpu::compat::destroy_shader(&device, shaders.frag)!!;
    gpu::compat::Format[1] formats = { gpu::compat::Format.RGBA8_UNORM };
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    bool supported = state.caps.line_polygon_mode;

    state.caps.line_polygon_mode = false;
    gpu::compat::GraphicsPipelineDesc unavailable = {
        .raster        = { .polygon_mode = gpu::compat::PolygonMode.LINE },
        .color_formats = formats[..],
    };
    uint entries_before = cache_entry_count(&device);
    assert(@catch(gpu::compat::create_graphics_pipeline(&device, &unavailable)) == gpu::compat::UNSUPPORTED_FEATURE,
        "unsupported LINE faults before invalid shader lookup");
    assert(cache_entry_count(&device) == entries_before, "unsupported LINE publishes no cache entry");
    state.caps.line_polygon_mode = supported;

    gpu::compat::GraphicsPipelineDesc line_topology = base_graphics_desc(&shaders, formats[..]);
    line_topology.topology = gpu::compat::PrimitiveTopology.LINES;
    gpu::compat::PipelineHandle topology_pipeline = gpu::compat::create_graphics_pipeline(&device, &line_topology)!!;
    gpu::compat::destroy_pipeline(&device, topology_pipeline)!!;

    if (!supported) return;
    gpu::compat::GraphicsPipelineDesc line_polygon = base_graphics_desc(&shaders, formats[..]);
    line_polygon.raster.polygon_mode = gpu::compat::PolygonMode.LINE;
    gpu::compat::PipelineHandle line_pipeline = gpu::compat::create_graphics_pipeline(&device, &line_polygon)!!;
    gpu::compat::destroy_pipeline(&device, line_pipeline)!!;
}

fn void compute_push_constant_sizes_validate_before_vulkan() @test {
    gpu::compat::Device device = make_cache_test_device();
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::ShaderDesc comp_sd = { .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = CACHE_COMP_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle comp = gpu::compat::create_shader(&device, &comp_sd)!!;
    defer gpu::compat::destroy_shader(&device, comp)!!;

    gpu::compat::ComputePipelineDesc below_minimum = { .push_constant_size = 4 };
    assert(@catch(gpu::compat::create_compute_pipeline(&device, &below_minimum)) == gpu::compat::INVALID_ARGUMENT,
        "below-minimum size faults before invalid shader lookup");

    gpu::compat::ComputePipelineDesc unaligned = { .shader = comp, .push_constant_size = gpu::compat::RootPush::size + 1 };
    assert(@catch(gpu::compat::create_compute_pipeline(&device, &unaligned)) == gpu::compat::INVALID_ARGUMENT,
        "unaligned push-constant size faults INVALID_ARGUMENT");

    uint limit = gpu::compat::get_device_caps(&device)!!.max_push_constant_size;
    uint aligned_limit = limit - limit % 4;
    gpu::compat::ComputePipelineDesc over_maximum = { .shader = comp, .push_constant_size = aligned_limit + 4 };
    assert(@catch(gpu::compat::create_compute_pipeline(&device, &over_maximum)) == gpu::compat::INVALID_ARGUMENT,
        "size above the selected-device maximum faults INVALID_ARGUMENT");

    gpu::compat::ComputePipelineDesc maximum = { .shader = comp, .push_constant_size = aligned_limit };
    gpu::compat::PipelineHandle pipeline = gpu::compat::create_compute_pipeline(&device, &maximum)!!;
    gpu::compat::destroy_pipeline(&device, pipeline)!!;
}

fn void compute_layout_cache_grows_beyond_pipeline_capacity_hint() @test {
    gpu::compat::Device device = make_cache_test_device_with_capacity(1);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::ShaderDesc comp_sd = { .stage = gpu::compat::ShaderStage.COMPUTE, .spirv = CACHE_COMP_SPIRV[..], .entry_point = "main" };
    gpu::compat::ShaderHandle comp = gpu::compat::create_shader(&device, &comp_sd)!!;
    defer gpu::compat::destroy_shader(&device, comp)!!;

    uint[3] sizes = { 8, 12, 16 };
    foreach (size : sizes) {
        gpu::compat::ComputePipelineDesc desc = { .shader = comp, .push_constant_size = size };
        gpu::compat::PipelineHandle pipeline = gpu::compat::create_compute_pipeline(&device, &desc)!!;
        gpu::compat::destroy_pipeline(&device, pipeline)!!;
    }

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    assert(state.compute_layout_cache.count == sizes.len, "each distinct size has one cached layout");
    assert(state.compute_layout_cache.entries.len >= sizes.len, "layout cache grows beyond the one-pipeline hint");
}
