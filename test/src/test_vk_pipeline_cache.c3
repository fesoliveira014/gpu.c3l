module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::core::mem;

// Headless (lavapipe): deterministic pipeline identity, transactional batches,
// refcounted cache entries, shared layouts, and dynamic viewport/scissor state.

const char[*] CACHE_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] CACHE_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");
const char[*] CACHE_COMP_SPIRV     = $embed("shaders/root_pointer.comp.spv");
const char[*] CACHE_ALT_COMP_SPIRV = $embed("shaders/heap_write.comp.spv");

const uint CACHE_RT_SIZE = 16;
const usz CACHE_RT_BYTES = CACHE_RT_SIZE * CACHE_RT_SIZE * 4;

fn gpu::Device make_cache_test_device() {
    gpu::RuntimeDesc desc = {
        .backend = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_pipeline_cache_test",
    };
    return create_test_device(&desc)!!;
}

fn gpu::Device make_cache_test_device_with_capacity(uint pipeline_capacity) {
    gpu::RuntimeDesc desc = {
        .backend = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .pipeline_capacity = pipeline_capacity,
        .application_name  = "gpu_pipeline_cache_capacity_test",
    };
    return create_test_device(&desc)!!;
}

fn uint cache_entry_count(gpu::Device* device) {
    return vk::pipeline_cache_entry_count((vk::VkDeviceState*)gpu::device_backend_state(*device));
}

fn bool cache_index_consistent(gpu::Device* device) {
    return vk::pipeline_cache_index_consistent((vk::VkDeviceState*)gpu::device_backend_state(*device));
}

struct CacheShaders {
    gpu::ShaderCode vert;
    gpu::ShaderCode frag;
}

fn CacheShaders make_cache_shaders() {
    gpu::ShaderDesc vert_sd = { .stage = gpu::ShaderStage.VERTEX, .spirv = CACHE_VERT_SPIRV[..], .entry_point = "main" };
    gpu::ShaderDesc frag_sd = { .stage = gpu::ShaderStage.FRAGMENT, .spirv = CACHE_FRAG_SPIRV[..], .entry_point = "main" };
    return {
        .vert = gpu::prepare_shader_code(&vert_sd)!!,
        .frag = gpu::prepare_shader_code(&frag_sd)!!,
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
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, formats[..]);

    gpu::PipelineHandle first = gpu::create_graphics_pipeline(&device, &desc)!!;
    gpu::PipelineHandle second = gpu::create_graphics_pipeline(&device, &desc)!!;
    assert(first != second, "aliases are distinct handles");
    assert(cache_entry_count(&device) == 1, "identical descriptors share one cache entry");

    gpu::destroy_pipeline(&device, first)!!;
    assert(cache_entry_count(&device) == 1, "entry survives while an alias lives");

    assert(@catch(gpu::destroy_pipeline(&device, first)) == gpu::INVALID_HANDLE,
        "double-destroy of the same handle faults INVALID_HANDLE");
    assert(cache_entry_count(&device) == 1, "double-destroy does not disturb the entry");

    gpu::destroy_pipeline(&device, second)!!;
    assert(cache_entry_count(&device) == 0, "last alias releases the entry");
}

fn void immutable_state_and_blend_normalization_split_entries() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, formats[..]);
    gpu::PipelineHandle base = gpu::create_graphics_pipeline(&device, &desc)!!;
    defer gpu::destroy_pipeline(&device, base)!!;


    gpu::GraphicsPipelineDesc culled = desc;
    culled.raster.cull_mode = gpu::CullMode.BACK;
    gpu::PipelineHandle cull_variant =
        gpu::create_graphics_pipeline(&device, &culled)!!;
    defer gpu::destroy_pipeline(&device, cull_variant)!!;
    assert(cache_entry_count(&device) == 2,
        "culling is immutable graphics pipeline identity");

    gpu::GraphicsPipelineDesc garbage_blend = desc;
    garbage_blend.blend = {
        .enable    = false,
        .src_color = gpu::BlendFactor.DST_ALPHA,
        .dst_color = gpu::BlendFactor.SRC_ALPHA,
        .color_op  = gpu::BlendOp.MAX,
    };
    gpu::PipelineHandle normalized = gpu::create_graphics_pipeline(&device, &garbage_blend)!!;
    defer gpu::destroy_pipeline(&device, normalized)!!;
    assert(cache_entry_count(&device) == 2, "disabled blend ignores factor fields in the key");

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
    assert(cache_entry_count(&device) == 3, "enabled blend is immutable state and splits the key");
}

fn void formats_split_entries_and_limits_enforced() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

    gpu::Format[1] rgba = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc desc_rgba = base_graphics_desc(&shaders, rgba[..]);
    gpu::PipelineHandle first = gpu::create_graphics_pipeline(&device, &desc_rgba)!!;
    defer gpu::destroy_pipeline(&device, first)!!;

    gpu::Format[1] bgra = { gpu::Format.BGRA8_UNORM };
    gpu::GraphicsPipelineDesc desc_bgra = base_graphics_desc(&shaders, bgra[..]);
    gpu::PipelineHandle second = gpu::create_graphics_pipeline(&device, &desc_bgra)!!;
    defer gpu::destroy_pipeline(&device, second)!!;
    assert(cache_entry_count(&device) == 2, "different formats produce distinct pipelines");

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
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
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = CACHE_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode comp = gpu::prepare_shader_code(&comp_sd)!!;

    gpu::ComputePipelineDesc desc = { .shader = comp, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle first = gpu::create_compute_pipeline(&device, &desc)!!;
    defer gpu::destroy_pipeline(&device, first)!!;
    gpu::PipelineHandle second = gpu::create_compute_pipeline(&device, &desc)!!;
    defer gpu::destroy_pipeline(&device, second)!!;
    assert(first != second, "compute aliases are distinct handles");
    assert(cache_entry_count(&device) == 1, "identical compute descriptors share one entry");

    gpu::ComputePipelineDesc wider = { .shader = comp, .push_constant_size = 16 };
    gpu::PipelineHandle third = gpu::create_compute_pipeline(&device, &wider)!!;
    defer gpu::destroy_pipeline(&device, third)!!;
    assert(cache_entry_count(&device) == 2, "push size is part of the compute key");

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(state.compute_layout_cache.count == 2, "duplicate push sizes reuse one layout entry");
    vk::PipelineLayout first_layout = state.pipelines.slots[first.index].value.layout;
    vk::PipelineLayout second_layout = state.pipelines.slots[second.index].value.layout;
    vk::PipelineLayout third_layout = state.pipelines.slots[third.index].value.layout;
    assert(first_layout == second_layout, "equal push sizes share a compute layout");
    assert(first_layout != third_layout, "different push sizes use different layouts");
}

fn void identical_shader_bytes_from_distinct_storage_deduplicate() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;

    char[] source = CACHE_COMP_SPIRV[..];
    char[] first_storage = mem::new_array(char, source.len);
    defer mem::free(first_storage.ptr);
    char[] second_storage = mem::new_array(char, source.len);
    defer mem::free(second_storage.ptr);
    mem::copy(first_storage.ptr, source.ptr, source.len);
    mem::copy(second_storage.ptr, source.ptr, source.len);

    gpu::ShaderDesc first_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = first_storage,
        .entry_point = "main",
    };
    gpu::ShaderDesc second_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = second_storage,
        .entry_point = "main",
    };
    gpu::ShaderCode first_code = gpu::prepare_shader_code(&first_desc)!!;
    gpu::ShaderCode second_code = gpu::prepare_shader_code(&second_desc)!!;
    gpu::ComputePipelineDesc first_pipeline_desc = {
        .shader             = first_code,
        .push_constant_size = gpu::RootPush::size,
    };
    gpu::ComputePipelineDesc second_pipeline_desc = {
        .shader             = second_code,
        .push_constant_size = gpu::RootPush::size,
    };

    gpu::PipelineHandle first =
        gpu::create_compute_pipeline(&device, &first_pipeline_desc)!!;
    gpu::PipelineHandle second =
        gpu::create_compute_pipeline(&device, &second_pipeline_desc)!!;
    assert(cache_entry_count(&device) == 1,
        "identical IR bytes from distinct storage share one cache entry");

    gpu::destroy_pipeline(&device, first)!!;
    gpu::destroy_pipeline(&device, second)!!;
}

fn void shader_digest_collisions_compare_ir_bytes() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc first_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc distinct_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = CACHE_ALT_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode first_code = gpu::prepare_shader_code(&first_desc)!!;
    gpu::ShaderCode distinct_code = gpu::prepare_shader_code(&distinct_desc)!!;
    distinct_code.digest = first_code.digest;
    assert(first_code != distinct_code,
        "forced digest collisions remain distinct shader code");

    gpu::ComputePipelineDesc first_pipeline_desc = {
        .shader             = first_code,
        .push_constant_size = gpu::RootPush::size,
    };
    gpu::ComputePipelineDesc distinct_pipeline_desc = {
        .shader             = distinct_code,
        .push_constant_size = gpu::RootPush::size,
    };
    gpu::PipelineHandle first =
        gpu::create_compute_pipeline(&device, &first_pipeline_desc)!!;
    gpu::PipelineHandle distinct =
        gpu::create_compute_pipeline(&device, &distinct_pipeline_desc)!!;
    assert(cache_entry_count(&device) == 2,
        "equal digests with different IR bytes create distinct cache entries");

    gpu::destroy_pipeline(&device, first)!!;
    gpu::destroy_pipeline(&device, distinct)!!;
}

fn void prepared_shader_code_is_reusable_across_devices() @test {
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;

    gpu::Device first_device = make_cache_test_device();
    defer destroy_test_device(&first_device)!!;
    gpu::Device second_device = make_cache_test_device();
    defer destroy_test_device(&second_device)!!;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader             = shader,
        .push_constant_size = gpu::RootPush::size,
    };

    gpu::PipelineHandle first =
        gpu::create_compute_pipeline(&first_device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&first_device, first)!!;
    gpu::PipelineHandle second =
        gpu::create_compute_pipeline(&second_device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&second_device, second)!!;

    assert(cache_entry_count(&first_device) == 1,
        "the first device owns one pipeline cache entry");
    assert(cache_entry_count(&second_device) == 1,
        "the second device owns an independent entry for shared shader code");
}

fn void immutable_cull_state_renders_per_pipeline() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

    // The offscreen fragment shader samples the heap, so a 1x1 white pattern
    // texture and a sampler feed both draws.
    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 1,
        .height    = 1,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
    };
    gpu::TextureHandle pattern = gpu::create_texture(&device, &tex_desc)!!;
    defer gpu::destroy_texture(&device, pattern)!!;
    gpu::TextureView tex_index_view = gpu::create_texture_view(&device, pattern, null)!!;
    gpu::TextureIndex tex_index = tex_index_view.index;
    defer gpu::destroy_texture_view(&device, tex_index_view)!!;
    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::Sampler sampler = gpu::intern_sampler(&device, &sampler_desc)!!;
    gpu::SamplerIndex smp_index = gpu::publish_sampler(&device, sampler)!!;
    char[4] white = { (char)255, (char)255, (char)255, (char)255 };
    upload_test_texture_data(
        device:   &device,
        dst:      pattern,
        mip:      0,
        data:     white[..],
        next_use: gpu::TextureUse.SAMPLED_FRAGMENT,
    )!!;

    gpu::TextureDesc rt_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = CACHE_RT_SIZE,
        .height    = CACHE_RT_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    gpu::TextureHandle target_culled = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target_culled)!!;
    gpu::TextureHandle target_drawn = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target_drawn)!!;
    gpu::TextureHandle target_reset = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target_reset)!!;

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc no_cull_desc = base_graphics_desc(&shaders, formats[..]);
    gpu::PipelineHandle no_cull = gpu::create_graphics_pipeline(&device, &no_cull_desc)!!;
    defer gpu::destroy_pipeline(&device, no_cull)!!;

    gpu::GraphicsPipelineDesc cull_desc = no_cull_desc;
    cull_desc.raster.cull_mode = gpu::CullMode.BACK;
    gpu::PipelineHandle culled = gpu::create_graphics_pipeline(&device, &cull_desc)!!;
    defer gpu::destroy_pipeline(&device, culled)!!;
    assert(cache_entry_count(&device) == 2, "distinct cull state has distinct graphics identity");

    OwnedSpan rb_culled = allocate_owned_span(
        device:       &device,
        size:         CACHE_RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "cache_culled_readback",
    )!!;
    defer free_owned_span(&device, &rb_culled)!!;
    OwnedSpan rb_drawn = allocate_owned_span(
        device:       &device,
        size:         CACHE_RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "cache_drawn_readback",
    )!!;
    defer free_owned_span(&device, &rb_drawn)!!;
    OwnedSpan rb_reset = allocate_owned_span(
        device:       &device,
        size:         CACHE_RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "cache_reset_readback",
    )!!;
    defer free_owned_span(&device, &rb_reset)!!;
    OwnedSpan verts_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "cache_vertices",
    )!!;
    defer free_owned_span(&device, &verts_owned)!!;
    gpu::GpuSpan verts_span = verts_owned.span;
    float* verts = (float*)test_span_mapping(&device, verts_span).ptr;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.1f, 0.1f,
         0.8f, -0.8f, 0.9f, 0.1f,
         0.0f,  0.8f, 0.5f, 0.9f,
    };
    for (uint i = 0; i < 12; i++) verts[i] = vertex_data[i];

    OwnedSpan frag_owned = allocate_owned_span(
        device:       &device,
        size:         FragRoot::size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "cache_fragment_root",
    )!!;
    defer free_owned_span(&device, &frag_owned)!!;
    gpu::GpuSpan frag_span = frag_owned.span;
    FragRoot* frag_root = (FragRoot*)test_span_mapping(&device, frag_span).ptr;
    frag_root.texture_index = tex_index;
    frag_root.sampler_index = smp_index;
    gpu::flush_mapped_span(&device, verts_span)!!;
    gpu::flush_mapped_span(&device, frag_span)!!;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureHandle[3] targets = { target_culled, target_drawn, target_reset };
    gpu::PipelineHandle[3] pipelines = { culled, no_cull, no_cull };
    gpu::GpuSpan[3] readbacks = { rb_culled.span, rb_drawn.span, rb_reset.span };
    for (uint i = 0; i < targets.len; i++) {
        gpu::TextureBarrier to_attach = {
            .texture = targets[i],
            .before  = gpu::TextureUse.UNDEFINED,
            .after   = gpu::TextureUse.COLOR_ATTACHMENT,
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
        if (i == 1) {
            gpu::Viewport viewport = {
                .width = (float)CACHE_RT_SIZE / 2.0f, .height = (float)CACHE_RT_SIZE, .max_depth = 1.0f,
            };
            gpu::ScissorRect scissor = {
                .width = (int)CACHE_RT_SIZE / 2, .height = (int)CACHE_RT_SIZE,
            };
            gpu::cmd_set_viewport(&cmd, &viewport)!!;
            gpu::cmd_set_scissor(&cmd, &scissor)!!;
        }
        bind_test_graphics_pipeline(&cmd, pipelines[i]);
        gpu::cmd_draw(
            commands:       &cmd,
            vertex_root:    test_span_address(&device, verts_span),
            fragment_root:  test_span_address(&device, frag_span),
            vertex_count:   3,
            instance_count: 1,
        )!!;
        gpu::cmd_end_render_pass(&cmd)!!;

        gpu::TextureBarrier to_src = {
            .texture = targets[i],
            .before  = gpu::TextureUse.COLOR_ATTACHMENT,
            .after   = gpu::TextureUse.TRANSFER_SOURCE,
        };
        gpu::cmd_texture_barrier(&cmd, &to_src)!!;

        gpu::GpuSpan rb_span = readbacks[i];
        gpu::TextureBufferCopyDesc copy = { .texture = targets[i], .dst = rb_span };
        gpu::cmd_copy_texture_to_buffer(&cmd, &copy)!!;
        gpu::Barrier host_read = {
            .before = { .transfer = true },
            .after = { .host = true },
        };
        gpu::cmd_barrier(&cmd, &host_read)!!;
    }

    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;

    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;

    gpu::invalidate_mapped_span(&device, rb_culled.span)!!;
    gpu::GpuSpan culled_span = rb_culled.span;
    char* culled_pixels = (char*)test_span_mapping(&device, culled_span).ptr;
    char* culled_center = &culled_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE / 2) * 4];
    assert(culled_center[0] == (char)8 && culled_center[1] == (char)16 && culled_center[2] == (char)32,
        "back-face culled draw leaves the clear color");

    gpu::invalidate_mapped_span(&device, rb_drawn.span)!!;
    gpu::GpuSpan drawn_span = rb_drawn.span;
    char* drawn_pixels = (char*)test_span_mapping(&device, drawn_span).ptr;
    char* drawn_left = &drawn_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE / 4) * 4];
    char* drawn_right = &drawn_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE * 3 / 4) * 4];
    assert(drawn_left[0] == (char)255 && drawn_left[1] == (char)255 && drawn_left[2] == (char)255,
        "pipeline alias switch preserves the explicit viewport and scissor on the left");
    assert(drawn_right[0] == (char)8 && drawn_right[1] == (char)16 && drawn_right[2] == (char)32,
        "pipeline alias switch preserves clipping on the right");

    gpu::invalidate_mapped_span(&device, rb_reset.span)!!;
    gpu::GpuSpan reset_span = rb_reset.span;
    char* reset_pixels = (char*)test_span_mapping(&device, reset_span).ptr;
    char* reset_center = &reset_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE / 2) * 4];
    assert(reset_center[0] == (char)255 && reset_center[1] == (char)255 && reset_center[2] == (char)255,
        "the next pass restores full viewport and scissor defaults");
}

fn void alias_survives_peer_destroy() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, formats[..]);
    gpu::PipelineHandle doomed = gpu::create_graphics_pipeline(&device, &desc)!!;
    gpu::PipelineHandle survivor = gpu::create_graphics_pipeline(&device, &desc)!!;
    gpu::destroy_pipeline(&device, doomed)!!;

    gpu::TextureDesc rt_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = CACHE_RT_SIZE,
        .height    = CACHE_RT_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;
    OwnedSpan verts_owned = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "cache_alias_vertices",
    )!!;
    defer free_owned_span(&device, &verts_owned)!!;
    gpu::GpuSpan verts_span = verts_owned.span;
    OwnedSpan frag_owned = allocate_owned_span(
        device:       &device,
        size:         16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "cache_alias_fragment_root",
    )!!;
    defer free_owned_span(&device, &frag_owned)!!;
    gpu::GpuSpan frag_span = frag_owned.span;

    gpu::CommandList cmd = gpu::begin_commands(gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = gpu::TextureUse.UNDEFINED,
        .after   = gpu::TextureUse.COLOR_ATTACHMENT,
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .texture = target, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = CACHE_RT_SIZE, .height = CACHE_RT_SIZE };
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    bind_test_graphics_pipeline(&cmd, survivor);
    gpu::cmd_draw(
        commands:       &cmd,
        vertex_root:    test_span_address(&device, verts_span),
        fragment_root:  test_span_address(&device, frag_span),
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&cmd)!!;
    gpu::ExecutableCommandList cmd_executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&cmd_executable)!!;
    gpu::ExecutableCommandList[1] lists = { cmd_executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
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
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

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
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

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
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();
    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    bool supported = state.caps.line_polygon_mode;

    state.caps.line_polygon_mode = false;
    gpu::GraphicsPipelineDesc unavailable = {
        .raster        = { .polygon_mode = gpu::PolygonMode.LINE },
        .color_formats = formats[..],
    };
    uint entries_before = cache_entry_count(&device);
    assert(@catch(gpu::create_graphics_pipeline(&device, &unavailable)) == gpu::UNSUPPORTED_FEATURE,
        "unsupported LINE faults before invalid shader-code validation");
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

fn void compute_push_constant_sizes_validate_before_vulkan() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = CACHE_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode comp = gpu::prepare_shader_code(&comp_sd)!!;

    gpu::ComputePipelineDesc below_minimum = { .push_constant_size = 4 };
    assert(@catch(gpu::create_compute_pipeline(&device, &below_minimum)) == gpu::INVALID_ARGUMENT,
        "below-minimum size faults before invalid shader-code validation");

    gpu::ComputePipelineDesc unaligned = { .shader = comp, .push_constant_size = gpu::RootPush::size + 1 };
    assert(@catch(gpu::create_compute_pipeline(&device, &unaligned)) == gpu::INVALID_ARGUMENT,
        "unaligned push-constant size faults INVALID_ARGUMENT");

    uint limit = gpu::get_device_caps(&device)!!.max_push_constant_size;
    uint aligned_limit = limit - limit % 4;
    gpu::ComputePipelineDesc over_maximum = { .shader = comp, .push_constant_size = aligned_limit + 4 };
    assert(@catch(gpu::create_compute_pipeline(&device, &over_maximum)) == gpu::INVALID_ARGUMENT,
        "size above the selected-device maximum faults INVALID_ARGUMENT");

    gpu::ComputePipelineDesc maximum = { .shader = comp, .push_constant_size = aligned_limit };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &maximum)!!;
    gpu::destroy_pipeline(&device, pipeline)!!;
}

fn void compute_layout_cache_grows_beyond_pipeline_capacity_hint() @test {
    gpu::Device device = make_cache_test_device_with_capacity(1);
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc comp_sd = { .stage = gpu::ShaderStage.COMPUTE, .spirv = CACHE_COMP_SPIRV[..], .entry_point = "main" };
    gpu::ShaderCode comp = gpu::prepare_shader_code(&comp_sd)!!;

    uint[3] sizes = { 8, 12, 16 };
    foreach (size : sizes) {
        gpu::ComputePipelineDesc desc = { .shader = comp, .push_constant_size = size };
        gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &desc)!!;
        gpu::destroy_pipeline(&device, pipeline)!!;
    }

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(state.compute_layout_cache.count == sizes.len, "each distinct size has one cached layout");
    assert(state.compute_layout_cache.entries.len >= sizes.len, "layout cache grows beyond the one-pipeline hint");
}

fn void compute_pipeline_batch_deduplicates_identity_and_shader_modules() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    char[] equivalent_storage = mem::new_array(char, CACHE_COMP_SPIRV.len);
    defer mem::free(equivalent_storage.ptr);
    mem::copy(
        equivalent_storage.ptr,
        CACHE_COMP_SPIRV[..].ptr,
        CACHE_COMP_SPIRV.len,
    );
    gpu::ShaderDesc equivalent_desc = shader_desc;
    equivalent_desc.spirv = equivalent_storage;
    gpu::ShaderCode equivalent_shader = gpu::prepare_shader_code(&equivalent_desc)!!;
    gpu::ComputePipelineDesc[3] descs = {
        { .shader = shader, .push_constant_size = gpu::RootPush::size },
        { .shader = equivalent_shader, .push_constant_size = gpu::RootPush::size },
        { .shader = equivalent_shader, .push_constant_size = 16 },
    };
    gpu::PipelineHandle[3] pipelines;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong shader_modules_before = vk::pipeline_shader_module_create_count(state);
    ulong native_pipelines_before = vk::pipeline_native_create_count(state);

    gpu::create_compute_pipelines(&device, descs[..], pipelines[..])!!;

    assert(pipelines[0].is_valid() && pipelines[1].is_valid() && pipelines[2].is_valid(),
        "a successful batch publishes every pipeline handle");
    assert(pipelines[0] != pipelines[1], "duplicate descriptions publish distinct owning handles");
    assert(cache_entry_count(&device) == 2, "duplicate descriptions share deterministic pipeline identity");
    assert(vk::pipeline_shader_module_create_count(state) - shader_modules_before == 1,
        "shared shader IR creates one temporary native module per batch");
    assert(vk::pipeline_native_create_count(state) - native_pipelines_before == 2,
        "only distinct compute pipeline identities reach native creation");

    foreach (pipeline : pipelines) gpu::destroy_pipeline(&device, pipeline)!!;
}

fn void graphics_pipeline_batch_reuses_shared_shader_modules() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();
    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc[2] descs;
    descs[0] = base_graphics_desc(&shaders, formats[..]);
    descs[1] = descs[0];
    descs[1].blend = {
        .enable    = true,
        .src_color = gpu::BlendFactor.SRC_ALPHA,
        .dst_color = gpu::BlendFactor.ONE_MINUS_SRC_ALPHA,
        .color_op  = gpu::BlendOp.ADD,
        .src_alpha = gpu::BlendFactor.ONE,
        .dst_alpha = gpu::BlendFactor.ZERO,
        .alpha_op  = gpu::BlendOp.ADD,
    };
    gpu::PipelineHandle[2] pipelines;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong shader_modules_before = vk::pipeline_shader_module_create_count(state);
    ulong native_pipelines_before = vk::pipeline_native_create_count(state);

    gpu::create_graphics_pipelines(&device, descs[..], pipelines[..])!!;

    assert(cache_entry_count(&device) == 2, "distinct blend identities publish distinct graphics pipelines");
    assert(vk::pipeline_shader_module_create_count(state) - shader_modules_before == 2,
        "shared vertex and fragment IR each create one temporary native module per batch");
    assert(vk::pipeline_native_create_count(state) - native_pipelines_before == 2,
        "each distinct graphics identity reaches native creation once");

    foreach (pipeline : pipelines) gpu::destroy_pipeline(&device, pipeline)!!;
}

fn void graphics_pipeline_batch_revalidates_shared_shader_roles() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();
    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc[1] descs;
    descs[0] = base_graphics_desc(&shaders, formats[..]);
    descs[0].fragment_shader = descs[0].vertex_shader;
    gpu::PipelineHandle[1] outputs = {
        { .owner = 1, .index = 2, .generation = 3 },
    };
    gpu::PipelineHandle before = outputs[0];
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong native_pipelines_before = vk::pipeline_native_create_count(state);

    fault stage_fault = @catch(gpu::create_graphics_pipelines(
        &device,
        descs[..],
        outputs[..],
    ));

    assert(stage_fault == gpu::INVALID_ARGUMENT,
        "batch shader reuse still validates each graphics shader role");
    assert(outputs[0] == before, "shader-role failure leaves caller outputs unchanged");
    assert(cache_entry_count(&device) == 0, "shader-role failure publishes no cache entry");
    assert(vk::pipeline_native_create_count(state) == native_pipelines_before,
        "shader-role failure performs no native pipeline creation");
}

fn void pipeline_batch_shape_failure_preserves_outputs() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc[2] descs;
    foreach (&desc : descs) {
        *desc = { .shader = shader, .push_constant_size = gpu::RootPush::size };
    }
    gpu::PipelineHandle sentinel = gpu::create_compute_pipeline(&device, &descs[0])!!;
    defer gpu::destroy_pipeline(&device, sentinel)!!;
    gpu::PipelineHandle[1] outputs = { sentinel };
    uint entries_before = cache_entry_count(&device);

    fault shape_fault = @catch(gpu::create_compute_pipelines(&device, descs[..], outputs[..]));

    assert(shape_fault == gpu::INVALID_ARGUMENT, "mismatched batch slices fault INVALID_ARGUMENT");
    assert(outputs[0] == sentinel, "shape validation leaves caller outputs unchanged");
    assert(cache_entry_count(&device) == entries_before, "shape validation leaves the cache unchanged");
}

fn void pipeline_batch_capacity_failure_is_transactional() @test {
    gpu::Device device = make_cache_test_device_with_capacity(1);
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc[2] descs = {
        { .shader = shader, .push_constant_size = gpu::RootPush::size },
        { .shader = shader, .push_constant_size = 16 },
    };
    gpu::PipelineHandle[2] outputs = {
        { .owner = 7, .index = 8, .generation = 9 },
        { .owner = 10, .index = 11, .generation = 12 },
    };
    gpu::PipelineHandle[2] before = outputs;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong shader_modules_before = vk::pipeline_shader_module_create_count(state);
    ulong native_pipelines_before = vk::pipeline_native_create_count(state);

    fault capacity_fault = @catch(gpu::create_compute_pipelines(&device, descs[..], outputs[..]));

    assert(capacity_fault == gpu::SLOT_TABLE_FULL, "an oversized batch faults before partial publication");
    assert(outputs[0] == before[0] && outputs[1] == before[1],
        "capacity failure leaves every caller output unchanged");
    assert(cache_entry_count(&device) == 0, "capacity failure publishes no cache entry");
    assert(vk::pipeline_shader_module_create_count(state) == shader_modules_before,
        "capacity failure creates no temporary shader module");
    assert(vk::pipeline_native_create_count(state) == native_pipelines_before,
        "capacity failure performs no native pipeline creation");
}

fn void pipeline_batch_invalid_state_prevalidates_transactionally() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();
    gpu::Format[1] formats = { gpu::Format.RGBA8_UNORM };
    gpu::GraphicsPipelineDesc[2] descs;
    descs[0] = base_graphics_desc(&shaders, formats[..]);
    descs[1] = descs[0];
    int invalid_topology = 99;
    mem::copy(&descs[1].topology, &invalid_topology, int::size);
    gpu::PipelineHandle[2] outputs = {
        { .owner = 1, .index = 2, .generation = 3 },
        { .owner = 4, .index = 5, .generation = 6 },
    };
    gpu::PipelineHandle[2] before = outputs;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong shader_modules_before = vk::pipeline_shader_module_create_count(state);
    ulong native_pipelines_before = vk::pipeline_native_create_count(state);

    fault invalid_fault = @catch(gpu::create_graphics_pipelines(
        &device,
        descs[..],
        outputs[..],
    ));

    assert(invalid_fault == gpu::INVALID_ARGUMENT, "unsupported immutable state faults INVALID_ARGUMENT");
    assert(outputs[0] == before[0] && outputs[1] == before[1],
        "state validation leaves caller outputs unchanged");
    assert(cache_entry_count(&device) == 0, "state validation publishes no cache entry");
    assert(vk::pipeline_shader_module_create_count(state) == shader_modules_before,
        "state validation creates no temporary shader module");
    assert(vk::pipeline_native_create_count(state) == native_pipelines_before,
        "state validation performs no native pipeline creation");
}

fn void pipeline_batch_partial_failure_rolls_back_publication() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderCode shader = gpu::prepare_shader_code(&shader_desc)!!;
    gpu::ComputePipelineDesc[2] descs = {
        { .shader = shader, .push_constant_size = gpu::RootPush::size },
        { .shader = shader, .push_constant_size = 16 },
    };
    gpu::PipelineHandle[2] outputs = {
        { .owner = 13, .index = 14, .generation = 15 },
        { .owner = 16, .index = 17, .generation = 18 },
    };
    gpu::PipelineHandle[2] before = outputs;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong native_pipelines_before = vk::pipeline_native_create_count(state);
    vk::set_pipeline_creation_failure(state, 1);
    defer vk::clear_pipeline_creation_failure(state);

    fault creation_fault = @catch(gpu::create_compute_pipelines(
        &device,
        descs[..],
        outputs[..],
    ));

    assert(creation_fault == gpu::PIPELINE_CREATE_FAILED,
        "a native failure after one success preserves its exact public fault");
    assert(outputs[0] == before[0] && outputs[1] == before[1],
        "partial failure leaves caller outputs unchanged");
    assert(cache_entry_count(&device) == 0,
        "partial failure removes every cache entry created by the batch");
    assert(vk::pipeline_native_create_count(state) - native_pipelines_before == 1,
        "the rollback probe fails after exactly one native creation");
}

fn void empty_pipeline_batches_succeed_without_native_work() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    gpu::ComputePipelineDesc[] compute_descs = {};
    gpu::GraphicsPipelineDesc[] graphics_descs = {};
    gpu::PipelineHandle[] outputs = {};
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    ulong shader_modules_before = vk::pipeline_shader_module_create_count(state);
    ulong native_pipelines_before = vk::pipeline_native_create_count(state);

    gpu::create_compute_pipelines(&device, compute_descs, outputs)!!;
    gpu::create_graphics_pipelines(&device, graphics_descs, outputs)!!;

    assert(vk::pipeline_shader_module_create_count(state) == shader_modules_before,
        "empty batches create no shader modules");
    assert(vk::pipeline_native_create_count(state) == native_pipelines_before,
        "empty batches create no native pipelines");
}
