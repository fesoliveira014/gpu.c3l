module gpu_test;

import gpu;
import gpu::vk;
import vk;

// Headless (lavapipe): pipeline dedup cache. Entry accounting across aliases,
// refcounted destroy, shared layouts, dynamic-state replay proven by drawing
// through two aliases of one cache entry with different cull state.

const char[*] CACHE_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] CACHE_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");
const char[*] CACHE_COMP_SPIRV = $embed("shaders/root_pointer.comp.spv");

const uint CACHE_RT_SIZE = 16;
const usz CACHE_RT_BYTES = CACHE_RT_SIZE * CACHE_RT_SIZE * 4;

fn gpu::Device make_cache_test_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_pipeline_cache_test",
    };
    return gpu::create_device(&desc)!!;
}

fn gpu::Device make_cache_test_device_with_capacity(uint pipeline_capacity) {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .pipeline_capacity = pipeline_capacity,
        .application_name  = "gpu_pipeline_cache_capacity_test",
    };
    return gpu::create_device(&desc)!!;
}

fn uint cache_entry_count(gpu::Device* device) {
    return vk::pipeline_cache_entry_count((vk::VkDeviceState*)device.backend_state);
}

fn bool cache_index_consistent(gpu::Device* device) {
    return vk::pipeline_cache_index_consistent((vk::VkDeviceState*)device.backend_state);
}

struct CacheShaders {
    gpu::ShaderHandle vert;
    gpu::ShaderHandle frag;
}

fn CacheShaders make_cache_shaders(gpu::Device* device) {
    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = CACHE_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = CACHE_FRAG_SPIRV[..], .entry_point = "main" };
    return {
        .vert = gpu::create_shader(device, &vert_sd)!!,
        .frag = gpu::create_shader(device, &frag_sd)!!,
    };
}

fn gpu::GraphicsPipelineDesc base_graphics_desc(CacheShaders* shaders, gpu::Format[] color_formats) {
    return {
        .vertex_shader   = shaders.vert,
        .fragment_shader = shaders.frag,
        .topology        = gpu::PrimitiveTopology.TRIANGLES,
        .color_formats   = color_formats,
    };
}

fn void identical_descs_alias_one_entry() @test {
    gpu::Device device = make_cache_test_device();
    defer gpu::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::destroy_shader(&device, shaders.vert)!!;
    defer gpu::destroy_shader(&device, shaders.frag)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, formats[..]);

    gpu::PipelineHandle first = gpu::create_graphics_pipeline(&device, &desc)!!;
    gpu::PipelineHandle second = gpu::create_graphics_pipeline(&device, &desc)!!;
    assert((ulong)first != (ulong)second, "aliases are distinct handles");
    assert(cache_entry_count(&device) == 1, "identical descriptors share one cache entry");

    gpu::destroy_pipeline(&device, first)!!;
    assert(cache_entry_count(&device) == 1, "entry survives while an alias lives");

    assert(@catch(gpu::destroy_pipeline(&device, first)) == gpu::INVALID_HANDLE,
        "double-destroy of the same handle faults INVALID_HANDLE");
    assert(cache_entry_count(&device) == 1, "double-destroy does not disturb the entry");

    gpu::destroy_pipeline(&device, second)!!;
    assert(cache_entry_count(&device) == 0, "last alias releases the entry");
}

fn void dynamic_state_and_blend_normalization_share_entries() @test {
    gpu::Device device = make_cache_test_device();
    defer gpu::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::destroy_shader(&device, shaders.vert)!!;
    defer gpu::destroy_shader(&device, shaders.frag)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, formats[..]);
    gpu::PipelineHandle base = gpu::create_graphics_pipeline(&device, &desc)!!;
    defer gpu::destroy_pipeline(&device, base)!!;

    gpu::GraphicsPipelineDesc culled = desc;
    culled.raster.cull_mode = gpu::CullMode.BACK;
    culled.depth = { .test_enable = true, .write_enable = true, .compare = gpu::CompareOp.LESS };
    gpu::PipelineHandle variant = gpu::create_graphics_pipeline(&device, &culled)!!;
    defer gpu::destroy_pipeline(&device, variant)!!;
    assert(cache_entry_count(&device) == 1, "raster/depth-only variants share one entry");

    gpu::GraphicsPipelineDesc garbage_blend = desc;
    garbage_blend.blend = {
        .enable    = false,
        .src_color = gpu::BlendFactor.DST_ALPHA,
        .dst_color = gpu::BlendFactor.SRC_ALPHA,
        .color_op  = gpu::BlendOp.MAX,
    };
    gpu::PipelineHandle normalized = gpu::create_graphics_pipeline(&device, &garbage_blend)!!;
    defer gpu::destroy_pipeline(&device, normalized)!!;
    assert(cache_entry_count(&device) == 1, "disabled blend ignores factor fields in the key");

    gpu::GraphicsPipelineDesc blended = desc;
    blended.blend = {
        .enable    = true,
        .src_color = gpu::BlendFactor.SRC_ALPHA,
        .dst_color = gpu::BlendFactor.ONE_MINUS_SRC_ALPHA,
        .color_op  = gpu::BlendOp.ADD,
        .src_alpha = gpu::BlendFactor.ONE,
        .dst_alpha = gpu::BlendFactor.ZERO,
        .alpha_op  = gpu::BlendOp.ADD,
    };
    gpu::PipelineHandle distinct = gpu::create_graphics_pipeline(&device, &blended)!!;
    defer gpu::destroy_pipeline(&device, distinct)!!;
    assert(cache_entry_count(&device) == 2, "enabled blend is immutable state and splits the key");
}

fn void formats_split_entries_and_limits_enforced() @test {
    gpu::Device device = make_cache_test_device();
    defer gpu::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::destroy_shader(&device, shaders.vert)!!;
    defer gpu::destroy_shader(&device, shaders.frag)!!;

    gpu::Format[1] rgba = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc desc_rgba = base_graphics_desc(&shaders, rgba[..]);
    gpu::PipelineHandle first = gpu::create_graphics_pipeline(&device, &desc_rgba)!!;
    defer gpu::destroy_pipeline(&device, first)!!;

    gpu::Format[1] bgra = { gpu::Format.BGRA8_UNORM };
    gpu::GraphicsPipelineDesc desc_bgra = base_graphics_desc(&shaders, bgra[..]);
    gpu::PipelineHandle second = gpu::create_graphics_pipeline(&device, &desc_bgra)!!;
    defer gpu::destroy_pipeline(&device, second)!!;
    assert(cache_entry_count(&device) == 2, "different formats produce distinct pipelines");

    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    vk::PipelineLayout first_layout = state.pipelines.slots[first.index].value.layout;
    vk::PipelineLayout second_layout = state.pipelines.slots[second.index].value.layout;
    assert(first_layout == second_layout, "graphics pipelines share one layout");
    assert(first_layout == state.graphics_layout, "the shared layout is the device graphics layout");

    gpu::Format[gpu::MAX_COLOR_ATTACHMENTS + 1] too_many;
    foreach (&format : too_many) *format = gpu::Format.RGBA8_UNORM;
    gpu::GraphicsPipelineDesc over_limit = base_graphics_desc(&shaders, too_many[..]);
    assert(@catch(gpu::create_graphics_pipeline(&device, &over_limit)) == gpu::INVALID_ARGUMENT,
        "more than MAX_COLOR_ATTACHMENTS color formats faults INVALID_ARGUMENT");
}

fn void compute_pipelines_dedup_and_share_layouts() @test {
    gpu::Device device = make_cache_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = CACHE_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle comp = gpu::create_shader(&device, &comp_sd)!!;
    defer gpu::destroy_shader(&device, comp)!!;

    gpu::ComputePipelineDesc desc = { .shader = comp, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle first = gpu::create_compute_pipeline(&device, &desc)!!;
    defer gpu::destroy_pipeline(&device, first)!!;
    gpu::PipelineHandle second = gpu::create_compute_pipeline(&device, &desc)!!;
    defer gpu::destroy_pipeline(&device, second)!!;
    assert((ulong)first != (ulong)second, "compute aliases are distinct handles");
    assert(cache_entry_count(&device) == 1, "identical compute descriptors share one entry");

    gpu::ComputePipelineDesc wider = { .shader = comp, .push_constant_size = 16 };
    gpu::PipelineHandle third = gpu::create_compute_pipeline(&device, &wider)!!;
    defer gpu::destroy_pipeline(&device, third)!!;
    assert(cache_entry_count(&device) == 2, "push size is part of the compute key");

    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    vk::PipelineLayout first_layout = state.pipelines.slots[first.index].value.layout;
    vk::PipelineLayout second_layout = state.pipelines.slots[second.index].value.layout;
    vk::PipelineLayout third_layout = state.pipelines.slots[third.index].value.layout;
    assert(first_layout == second_layout, "equal push sizes share a compute layout");
    assert(first_layout != third_layout, "different push sizes use different layouts");
}

fn void cull_replay_renders_per_alias() @test {
    gpu::Device device = make_cache_test_device();
    defer gpu::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::destroy_shader(&device, shaders.vert)!!;
    defer gpu::destroy_shader(&device, shaders.frag)!!;

    // The offscreen fragment shader samples the heap, so a 1x1 white pattern
    // texture and a sampler feed both draws.
    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle pattern = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, pattern)!!;
    gpu::TextureIndex tex_index = gpu::create_texture_descriptor(&device, pattern, null)!!;
    defer gpu::destroy_texture_descriptor(&device, tex_index)!!;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::SamplerIndex smp_index = gpu::create_sampler(&device, &sampler_desc)!!;
    defer gpu::destroy_sampler(&device, smp_index)!!;

    char[4] white = { (char)255, (char)255, (char)255, (char)255 };
    gpu::upload_texture_data(
        device:      &device,
        dst:         pattern,
        mip:         0,
        data:        white[..],
        next_stage:  gpu::Stage.FRAGMENT_SHADER,
        next_hazard: gpu::Hazard.SHADER_READ,
        next_layout: gpu::TextureLayout.SHADER_READ,
    )!!;

    gpu::TextureDesc rt_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = CACHE_RT_SIZE,
        .height    = CACHE_RT_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    gpu::TextureHandle target_culled = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target_culled)!!;
    gpu::TextureHandle target_drawn = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target_drawn)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc no_cull_desc = base_graphics_desc(&shaders, formats[..]);
    gpu::PipelineHandle no_cull = gpu::create_graphics_pipeline(&device, &no_cull_desc)!!;
    defer gpu::destroy_pipeline(&device, no_cull)!!;

    gpu::GraphicsPipelineDesc cull_desc = no_cull_desc;
    cull_desc.raster.cull_mode = gpu::CullMode.BACK;
    gpu::PipelineHandle culled = gpu::create_graphics_pipeline(&device, &cull_desc)!!;
    defer gpu::destroy_pipeline(&device, culled)!!;
    assert(cache_entry_count(&device) == 1, "both draw pipelines alias one cache entry");

    gpu::BufferDesc rb_desc = {
        .size = CACHE_RT_BYTES, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK,
    };
    gpu::BufferHandle rb_culled = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, rb_culled)!!;
    gpu::BufferHandle rb_drawn = gpu::create_buffer(&device, &rb_desc)!!;
    defer gpu::destroy_buffer(&device, rb_drawn)!!;

    gpu::begin_frame(&device)!!;

    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!!;
    float* verts = (float*)verts_span.cpu;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = vertex_data[i];

    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, FragRoot::size, 16)!!;
    FragRoot* frag_root = (FragRoot*)frag_span.cpu;
    frag_root.texture_index = tex_index;
    frag_root.sampler_index = smp_index;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::TextureHandle[2] targets = { target_culled, target_drawn };
    gpu::PipelineHandle[2] pipelines = { culled, no_cull };
    gpu::BufferHandle[2] readbacks = { rb_culled, rb_drawn };
    for (uint i = 0; i < 2; i++) {
        gpu::TextureBarrier to_attach = {
            .texture       = targets[i],
            .before_stage  = gpu::Stage.HOST,
            .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
            .before_hazard = gpu::Hazard.HOST_WRITE,
            .after_hazard  = gpu::Hazard.COLOR_WRITE,
            .old_layout    = gpu::TextureLayout.UNDEFINED,
            .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
        };
        gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

        gpu::ColorTargetDesc[1] colors = {{
            .texture = targets[i],
            .load_op = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
        }};
        gpu::RenderPassDesc pass = { .colors = colors[..], .width = CACHE_RT_SIZE, .height = CACHE_RT_SIZE };
        gpu::cmd_begin_render_pass(&cmd, &pass)!!;
        gpu::cmd_draw(
            commands:       &cmd,
            pipeline:       pipelines[i],
            vertex_root:    verts_span.gpu,
            fragment_root:  frag_span.gpu,
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::cmd_end_render_pass(&cmd)!!;

        gpu::TextureBarrier to_src = {
            .texture       = targets[i],
            .before_stage  = gpu::Stage.COLOR_ATTACHMENT,
            .after_stage   = gpu::Stage.TRANSFER,
            .before_hazard = gpu::Hazard.COLOR_WRITE,
            .after_hazard  = gpu::Hazard.TRANSFER_READ,
            .old_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
            .new_layout    = gpu::TextureLayout.TRANSFER_SRC,
        };
        gpu::cmd_texture_barrier(&cmd, &to_src)!!;

        gpu::GpuSpan rb_span = gpu::get_buffer_span(&device, readbacks[i])!!;
        gpu::TextureBufferCopyDesc copy = { .texture = targets[i], .dst = rb_span };
        gpu::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
        gpu::BufferBarrier host_read = {
            .buffer = readbacks[i], .offset = 0, .size = CACHE_RT_BYTES,
            .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
            .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
        };
        gpu::cmd_buffer_barrier(&cmd, &host_read)!!;
    }

    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;

    gpu::invalidate_buffer(&device, rb_culled, 0, 0)!!;
    gpu::GpuSpan culled_span = gpu::get_buffer_span(&device, rb_culled)!!;
    char* culled_pixels = (char*)culled_span.cpu;
    char* culled_center = &culled_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE / 2) * 4];
    assert(culled_center[0] == (char)8 && culled_center[1] == (char)16 && culled_center[2] == (char)32,
        "back-face culled draw leaves the clear color");

    gpu::invalidate_buffer(&device, rb_drawn, 0, 0)!!;
    gpu::GpuSpan drawn_span = gpu::get_buffer_span(&device, rb_drawn)!!;
    char* drawn_pixels = (char*)drawn_span.cpu;
    char* drawn_center = &drawn_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE / 2) * 4];
    assert(drawn_center[0] == (char)255 && drawn_center[1] == (char)255 && drawn_center[2] == (char)255,
        "cull-disabled alias of the same entry renders the triangle");

    gpu::end_frame(&device)!!;
}

fn void alias_survives_peer_destroy() @test {
    gpu::Device device = make_cache_test_device();
    defer gpu::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::destroy_shader(&device, shaders.vert)!!;
    defer gpu::destroy_shader(&device, shaders.frag)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, formats[..]);
    gpu::PipelineHandle doomed = gpu::create_graphics_pipeline(&device, &desc)!!;
    gpu::PipelineHandle survivor = gpu::create_graphics_pipeline(&device, &desc)!!;
    gpu::destroy_pipeline(&device, doomed)!!;

    gpu::TextureDesc rt_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = CACHE_RT_SIZE,
        .height    = CACHE_RT_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;

    gpu::begin_frame(&device)!!;
    gpu::GpuSpan verts_span = gpu::alloc_frame_span(&device, 3 * 16, 16)!!;
    gpu::GpuSpan frag_span = gpu::alloc_frame_span(&device, 16, 16)!!;

    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::TextureBarrier to_attach = {
        .texture       = target,
        .before_stage  = gpu::Stage.HOST,
        .after_stage   = gpu::Stage.COLOR_ATTACHMENT,
        .before_hazard = gpu::Hazard.HOST_WRITE,
        .after_hazard  = gpu::Hazard.COLOR_WRITE,
        .old_layout    = gpu::TextureLayout.UNDEFINED,
        .new_layout    = gpu::TextureLayout.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = CACHE_RT_SIZE, .height = CACHE_RT_SIZE };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    gpu::cmd_draw(
        commands:       &cmd,
        pipeline:       survivor,
        vertex_root:    verts_span.gpu,
        fragment_root:  frag_span.gpu,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&device)!!;

    gpu::destroy_pipeline(&device, survivor)!!;
    assert(cache_entry_count(&device) == 0, "entry released after the surviving alias is destroyed");
}

const uint CHURN_CAPACITY = 8;
// Headroom above CHURN_CAPACITY: pipeline_capacity also sizes the handle slot
// table, and an aliasing create needs a free slot even on a cache hit.
const uint CHURN_DEVICE_CAPACITY = CHURN_CAPACITY + 2;
const uint CHURN_ROUNDS = 4;

fn void index_survives_create_alias_destroy_churn() @test {
    gpu::Device device = make_cache_test_device_with_capacity(CHURN_DEVICE_CAPACITY);
    defer gpu::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::destroy_shader(&device, shaders.vert)!!;
    defer gpu::destroy_shader(&device, shaders.frag)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    uint[CHURN_CAPACITY] teardown_order = { 3, 0, 7, 1, 6, 2, 5, 4 };

    for (uint round = 0; round < CHURN_ROUNDS; round++) {
        gpu::PipelineHandle[CHURN_CAPACITY] handles;
        for (uint i = 0; i < CHURN_CAPACITY; i++) {
            gpu::GraphicsPipelineDesc variant = base_graphics_desc(&shaders, formats[..]);
            variant.raster.depth_bias_constant = (float)(round * CHURN_CAPACITY + i + 1);
            handles[i] = gpu::create_graphics_pipeline(&device, &variant)!!;
        }
        assert(cache_index_consistent(&device), "index consistent once all distinct entries are inserted");
        assert(cache_entry_count(&device) == CHURN_CAPACITY, "each round inserts exactly CHURN_CAPACITY distinct entries");

        // Alias two live entries; the refcount>0 path must not touch the index.
        gpu::GraphicsPipelineDesc alias_desc = base_graphics_desc(&shaders, formats[..]);
        alias_desc.raster.depth_bias_constant = (float)(round * CHURN_CAPACITY + 0 + 1);
        gpu::PipelineHandle alias_a = gpu::create_graphics_pipeline(&device, &alias_desc)!!;
        alias_desc.raster.depth_bias_constant = (float)(round * CHURN_CAPACITY + 3 + 1);
        gpu::PipelineHandle alias_b = gpu::create_graphics_pipeline(&device, &alias_desc)!!;
        assert(cache_index_consistent(&device), "index consistent after aliasing existing entries");
        assert(cache_entry_count(&device) == CHURN_CAPACITY, "aliasing does not grow the entry count");

        gpu::destroy_pipeline(&device, alias_a)!!;
        gpu::destroy_pipeline(&device, alias_b)!!;
        assert(cache_index_consistent(&device), "index consistent after alias release");
        assert(cache_entry_count(&device) == CHURN_CAPACITY, "peer entries survive alias release");

        // Destroy the base handles out of order to recycle free-list slots
        // through the index in a different pattern each round.
        foreach (i : teardown_order) {
            gpu::destroy_pipeline(&device, handles[i])!!;
            assert(cache_index_consistent(&device), "index consistent mid-teardown");
        }
        assert(cache_entry_count(&device) == 0, "table empty after full teardown");
    }
}

fn void small_pipeline_capacity_faults_at_limit() @test {
    const uint CAPACITY = 3;
    gpu::Device device = make_cache_test_device_with_capacity(CAPACITY);
    defer gpu::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::destroy_shader(&device, shaders.vert)!!;
    defer gpu::destroy_shader(&device, shaders.frag)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::PipelineHandle[CAPACITY] handles;
    for (uint i = 0; i < CAPACITY; i++) {
        gpu::GraphicsPipelineDesc variant = base_graphics_desc(&shaders, formats[..]);
        variant.raster.depth_bias_constant = (float)(i + 1);
        handles[i] = gpu::create_graphics_pipeline(&device, &variant)!!;
    }
    assert(cache_entry_count(&device) == CAPACITY, "the small capacity fills exactly");
    assert(cache_index_consistent(&device), "index consistent at capacity");

    gpu::GraphicsPipelineDesc over_capacity = base_graphics_desc(&shaders, formats[..]);
    over_capacity.raster.depth_bias_constant = (float)(CAPACITY + 1);
    assert(@catch(gpu::create_graphics_pipeline(&device, &over_capacity)) == gpu::SLOT_TABLE_FULL,
        "the distinct pipeline past the configured capacity faults SLOT_TABLE_FULL");

    foreach (handle : handles) gpu::destroy_pipeline(&device, handle)!!;
}

fn void line_polygon_mode_respects_optional_feature() @test {
    gpu::Device device = make_cache_test_device();
    defer gpu::destroy_device(&device)!!;
    CacheShaders shaders = make_cache_shaders(&device);
    defer gpu::destroy_shader(&device, shaders.vert)!!;
    defer gpu::destroy_shader(&device, shaders.frag)!!;
    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    vk::VkDeviceState* state = (vk::VkDeviceState*)device.backend_state;
    bool supported = state.caps.line_polygon_mode;

    state.caps.line_polygon_mode = false;
    gpu::GraphicsPipelineDesc unavailable = {
        .raster        = { .polygon_mode = gpu::PolygonMode.LINE },
        .color_formats = formats[..],
    };
    uint entries_before = cache_entry_count(&device);
    assert(@catch(gpu::create_graphics_pipeline(&device, &unavailable)) == gpu::UNSUPPORTED_FEATURE,
        "unsupported LINE faults before invalid shader lookup");
    assert(cache_entry_count(&device) == entries_before, "unsupported LINE publishes no cache entry");
    state.caps.line_polygon_mode = supported;

    gpu::GraphicsPipelineDesc line_topology = base_graphics_desc(&shaders, formats[..]);
    line_topology.topology = gpu::PrimitiveTopology.LINES;
    gpu::PipelineHandle topology_pipeline = gpu::create_graphics_pipeline(&device, &line_topology)!!;
    gpu::destroy_pipeline(&device, topology_pipeline)!!;

    if (!supported) return;
    gpu::GraphicsPipelineDesc line_polygon = base_graphics_desc(&shaders, formats[..]);
    line_polygon.raster.polygon_mode = gpu::PolygonMode.LINE;
    gpu::PipelineHandle line_pipeline = gpu::create_graphics_pipeline(&device, &line_polygon)!!;
    gpu::destroy_pipeline(&device, line_pipeline)!!;
}
