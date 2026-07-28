module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::core::mem;
import std::core::mem::allocators;
import std::atomic;
import std::thread;

// Headless (lavapipe): deterministic pipeline identity, singular rollback,
// refcounted cache entries, per-target color state, shared layouts, and dynamic
// raster state.

const char[*] CACHE_VERT_SPIRV = $embed("shaders/offscreen.vert.spv");
const char[*] CACHE_FRAG_SPIRV = $embed("shaders/offscreen.frag.spv");
const char[*] CACHE_TARGETS_FRAG_SPIRV =
    $embed("shaders/pipeline_targets.frag.spv");
const char[*] CACHE_COMP_SPIRV     = $embed("shaders/root_pointer.comp.spv");
const char[*] CACHE_ALT_COMP_SPIRV = $embed("shaders/heap_write.comp.spv");

const uint CACHE_RT_SIZE = 16;
const usz CACHE_RT_BYTES = CACHE_RT_SIZE * CACHE_RT_SIZE * 4;

fn gpu::Device make_cache_test_device() {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .application_name  = "gpu_pipeline_cache_test",
    };
    return create_test_device(&desc)!!;
}

fn gpu::Device make_cache_test_device_with_capacity(uint pipeline_capacity) {
    gpu::RuntimeDesc desc = {
        .contract_validation      = gpu::ContractValidation.FULL,
        .enable_vulkan_validation = true,
        .pipeline_capacity = pipeline_capacity,
        .application_name  = "gpu_pipeline_cache_capacity_test",
    };
    return create_test_device(&desc)!!;
}

fn gpu::Device make_generated_work_cache_test_device() {
    gpu::RuntimeDesc desc = {
        .pipeline_capacity = 2,
        .application_name  = "gpu_generated_work_cache_test",
    };
    return create_test_device(&desc)!!;
}

fn uint cache_entry_count(gpu::Device* device) {
    return vk::pipeline_cache_entry_count(gpu::internal::device_slots[device.index].data.state);
}

fn bool cache_index_consistent(gpu::Device* device) {
    return vk::pipeline_cache_index_consistent(gpu::internal::device_slots[device.index].data.state);
}

fn vk::VkDeviceState* cache_test_state(gpu::Device* device) {
    return gpu::internal::device_slots[device.index].data.state;
}

struct CacheShaders {
    gpu::ShaderDesc vert;
    gpu::ShaderDesc frag;
}

struct GeneratedLayoutGate {
    Atomic{bool} entered;
    Atomic{bool} release;
    vk::IndirectCommandsLayoutEXT captured;
}

struct GeneratedDispatchRecordArgs {
    gpu::CommandList commands;
    gpu::GpuSpan     records;
    gpu::GpuSpan     count;
    GeneratedLayoutGate* gate;
    vk::IndirectCommandsLayoutEXT prepared_layout;
    fault            failure;
}

struct PipelineShaderDiagnosticCapture {
    uint    calls;
    ZString expected_operation;
    ZString expected_debug_name;
    ZString expected_field;
    ZString expected_invariant;
    bool    matched;
}

struct ShaderModuleCodeCapture {
    uint*  expected_words;
    usz    expected_byte_size;
    char[] expected_bytes;
    bool   matched;
}

fn void capture_pipeline_shader_diagnostic(
    gpu::DebugMessage* message,
    void* user_data,
) {
    PipelineShaderDiagnosticCapture* capture =
        (PipelineShaderDiagnosticCapture*)user_data;
    capture.calls++;
    capture.matched = capture.matched
        && message.severity == gpu::DebugMessageSeverity.ERROR
        && message.category.public_contract
        && message.has_fault
        && message.public_fault == gpu::SHADER_INVALID
        && message.operation != null
        && message.operation.str_view()
            == capture.expected_operation.str_view()
        && message.resource.kind == gpu::DebugResourceKind.SHADER
        && !message.resource.has_public_identity
        && message.resource.debug_name != null
        && message.resource.debug_name.str_view()
            == capture.expected_debug_name.str_view()
        && message.rejected_field != null
        && message.rejected_field.str_view() == capture.expected_field.str_view()
        && message.invariant != null
        && message.invariant.str_view()
            == capture.expected_invariant.str_view();
}

fn vk::Result return_shader_module_create_result(
    void* user_data,
    vk::Device device,
    vk::ShaderModuleCreateInfo* create_info,
    vk::ShaderModule* out_shader,
) {
    (void)device;
    (void)create_info;
    (void)out_shader;
    return *(vk::Result*)user_data;
}

fn vk::Result capture_shader_module_code(
    void* user_data,
    vk::Device device,
    vk::ShaderModuleCreateInfo* create_info,
    vk::ShaderModule* out_shader,
) {
    ShaderModuleCodeCapture* capture =
        (ShaderModuleCodeCapture*)user_data;
    capture.matched = create_info != null
        && create_info.p_code == capture.expected_words
        && (usz)create_info.code_size == capture.expected_byte_size
        && mem::equals(
            ((char*)create_info.p_code)[:(usz)create_info.code_size],
            capture.expected_bytes,
        );
    return vk::create_shader_module(
        device:          device,
        p_create_info:   create_info,
        p_allocator:     null,
        p_shader_module: out_shader,
    );
}

fn vk::Result return_compute_pipeline_create_result(
    void* user_data,
    vk::Device device,
    vk::PipelineCache pipeline_cache,
    vk::ComputePipelineCreateInfo* create_info,
    vk::Pipeline* out_pipeline,
) {
    (void)device;
    (void)pipeline_cache;
    (void)create_info;
    (void)out_pipeline;
    return *(vk::Result*)user_data;
}

fn vk::Result return_graphics_pipeline_create_result(
    void* user_data,
    vk::Device device,
    vk::PipelineCache pipeline_cache,
    vk::GraphicsPipelineCreateInfo* create_info,
    vk::Pipeline* out_pipeline,
) {
    (void)device;
    (void)pipeline_cache;
    (void)create_info;
    (void)out_pipeline;
    return *(vk::Result*)user_data;
}

fn uint? fail_pipeline_cache_insert(
    void* user_data,
    vk::VkDeviceState* state,
    vk::PipelineCacheEntry value,
) {
    (void)user_data;
    (void)state;
    (void)value;
    return gpu::SLOT_TABLE_FULL~;
}

fn vk::IndirectCommandsLayoutEXT capture_generated_layout_read(
    void* raw_gate,
    vk::BoundPipeline* pipeline,
) {
    GeneratedLayoutGate* gate = (GeneratedLayoutGate*)raw_gate;
    gate.captured = pipeline.generated_dispatch_layout;
    gate.entered.store(true, AtomicOrdering.RELEASE);
    while (!gate.release.load(AtomicOrdering.ACQUIRE)) thread::yield();
    return gate.captured;
}

fn int record_generated_dispatch(void* raw_args) {
    GeneratedDispatchRecordArgs* args =
        (GeneratedDispatchRecordArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, 64 * 1024) {
        gpu::internal::vk::CommandRecord* record =
            gpu::internal::recording_command(&args.commands)!!;
        vk::GeneratedDispatchLayoutReadOps layout_read = {
            .user_data = args.gate,
            .read      = &capture_generated_layout_read,
        };
        if (catch dispatch_fault =
            vk::trusted_dispatch_generated_with_layout_read_ops(
                commands:           record,
                records:            args.records,
                count_span:         args.count,
                max_dispatch_count: 1,
                layout_read:        &layout_read,
                prepared_layout:    &args.prepared_layout,
            )) {
            args.failure = dispatch_fault;
            return 1;
        }
    };
    return 0;
}

fn CacheShaders make_cache_shaders() {
    gpu::ShaderDesc vert_sd = {
        .spirv       = CACHE_VERT_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc frag_sd = {
        .spirv       = CACHE_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    return {
        .vert = vert_sd,
        .frag = frag_sd,
    };
}

fn gpu::GraphicsPipelineDesc base_graphics_desc(
    CacheShaders* shaders,
    gpu::Format[] color_formats,
) {
    return {
        .vertex_shader   = shaders.vert,
        .fragment_shader = shaders.frag,
        .color_formats   = color_formats,
    };
}

fn void identical_descs_alias_one_entry() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc desc =
        base_graphics_desc(&shaders, color_formats[..]);

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

fn void two_color_targets_lower_independent_blend_and_write_masks() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();
    gpu::ShaderDesc target_frag_desc = {
        .spirv       = CACHE_TARGETS_FRAG_SPIRV[..],
        .entry_point = "main",
    };
    shaders.frag = target_frag_desc;

    gpu::Format[2] color_formats = {
        gpu::Format.RGBA8_UNORM,
        gpu::Format.RGBA8_UNORM,
    };
    gpu::ColorTargetState[2] color_targets = {
        {
            .write_mask = { .red = true },
        },
        {
            .blend = {
                .enable    = true,
                .src_color = gpu::BlendFactor.SRC_ALPHA,
                .dst_color = gpu::BlendFactor.ONE_MINUS_SRC_ALPHA,
                .color_op  = gpu::BlendOp.ADD,
                .src_alpha = gpu::BlendFactor.ONE,
                .dst_alpha = gpu::BlendFactor.ZERO,
                .alpha_op  = gpu::BlendOp.ADD,
            },
            .write_mask = gpu::COLOR_WRITE_ALL,
        },
    };
    gpu::GraphicsPipelineDesc pipeline_desc =
        base_graphics_desc(&shaders, color_formats[..]);
    gpu::PipelineHandle pipeline =
        gpu::create_graphics_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    gpu::TextureDesc texture_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = CACHE_RT_SIZE,
        .height    = CACHE_RT_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    gpu::TextureHandle[2] targets = {
        gpu::create_texture(&device, &texture_desc)!!,
        gpu::create_texture(&device, &texture_desc)!!,
    };
    defer {
        foreach (target : targets) {
            (void)gpu::destroy_texture(&device, target);
        }
    }
    gpu::AttachmentViewHandle[2] target_views = {
        create_test_attachment_view(&device, targets[0])!!,
        create_test_attachment_view(&device, targets[1])!!,
    };
    defer {
        foreach (&view : target_views) {
            (void)destroy_test_attachment_view(&device, view);
        }
    }
    OwnedSpan[2] readbacks = {
        allocate_owned_span(
            device:       &device,
            size:         CACHE_RT_BYTES,
            memory_class: gpu::MemoryClass.CPU_READ,
            access:       { .graphics, .compute, .transfer },
            debug_name:   "cache_first_target_readback",
        )!!,
        allocate_owned_span(
            device:       &device,
            size:         CACHE_RT_BYTES,
            memory_class: gpu::MemoryClass.CPU_READ,
            access:       { .graphics, .compute, .transfer },
            debug_name:   "cache_second_target_readback",
        )!!,
    };
    defer {
        foreach (&readback : readbacks) {
            (void)free_owned_span(&device, readback);
        }
    }
    OwnedSpan vertices = allocate_owned_span(
        device:       &device,
        size:         3 * 16,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics },
        debug_name:   "cache_target_vertices",
    )!!;
    defer free_owned_span(&device, &vertices)!!;
    float* mapped_vertices =
        (float*)test_span_mapping(&device, vertices.span).ptr;
    float[12] vertex_data = {
        -0.8f, -0.8f, 0.0f, 0.0f,
         0.8f, -0.8f, 0.0f, 0.0f,
         0.0f,  0.8f, 0.0f, 0.0f,
    };
    for (uint i = 0; i < vertex_data.len; i++) {
        mapped_vertices[i] = vertex_data[i];
    }
    gpu::flush_mapped_span(&device, vertices.span)!!;

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&commands)!!;
    foreach (target : targets) {
        gpu::TextureBarrier to_attach = {
            .texture = target,
            .before  = { .layout = gpu::TextureLayout.UNDEFINED },
            .after   = {
                .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
                .stages = { .color_output },
                .access = { .read, .write },
            },
        };
        gpu::cmd_texture_barrier(&commands, &to_attach)!!;
    }
    gpu::ColorTargetDesc[2] colors = {
        {
            .view     = target_views[0],
            .load_op  = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear    = { .rgba = { 0.0f, 0.0f, 1.0f, 1.0f } },
        },
        {
            .view     = target_views[1],
            .load_op  = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear    = { .rgba = { 0.0f, 0.0f, 1.0f, 1.0f } },
        },
    };
    gpu::RenderPassDesc pass = {
        .colors = colors[..],
        .width  = CACHE_RT_SIZE,
        .height = CACHE_RT_SIZE,
    };
    gpu::GraphicsState graphics =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    graphics.color.targets = color_targets[..];
    gpu::cmd_begin_render_pass(&commands, &pass)!!;
    bind_test_graphics_pipeline(&commands, pipeline);
    gpu::cmd_set_graphics_state(&commands, &graphics)!!;
    gpu::GpuAddress root = test_span_address(&device, vertices.span);
    gpu::cmd_draw(
        commands:       &commands,
        vertex_root:    root,
        fragment_root:  root,
        vertex_count:   3,
        instance_count: 1,
    )!!;
    gpu::cmd_end_render_pass(&commands)!!;

    for (uint i = 0; i < targets.len; i++) {
        gpu::TextureBarrier to_source = {
            .texture = targets[i],
            .before  = {
                .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
                .stages = { .color_output },
                .access = { .read, .write },
            },
            .after   = {
                .layout = gpu::TextureLayout.TRANSFER_SOURCE,
                .stages = { .transfer },
                .access = { .read },
            },
        };
        gpu::cmd_texture_barrier(&commands, &to_source)!!;
        gpu::TextureBufferCopyDesc copy = {
            .texture = targets[i],
            .dst     = readbacks[i].span,
        };
        gpu::cmd_copy_texture_to_buffer(&commands, &copy)!!;
    }
    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after  = { .host = true },
    };
    gpu::cmd_barrier(&commands, &host_read)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;

    foreach (&readback : readbacks) {
        gpu::invalidate_mapped_span(&device, readback.span)!!;
    }
    uint center_offset =
        (CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE / 2) * 4;
    char* first_pixels =
        (char*)test_span_mapping(&device, readbacks[0].span).ptr;
    char* first_center = &first_pixels[center_offset];
    assert(first_center[0] == (char)255
            && first_center[1] == (char)0
            && first_center[2] == (char)255
            && first_center[3] == (char)255,
        "first target writes only red and preserves cleared blue and alpha");
    char* second_pixels =
        (char*)test_span_mapping(&device, readbacks[1].span).ptr;
    char* second_center = &second_pixels[center_offset];
    assert(second_center[0] == (char)0
            && (second_center[1] == (char)127
                || second_center[1] == (char)128)
            && (second_center[2] == (char)127
                || second_center[2] == (char)128),
        "second target applies its own alpha blend equation");
}

fn void formats_split_entries_and_limits_enforced() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

    gpu::Format[1] rgba_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc desc_rgba =
        base_graphics_desc(&shaders, rgba_formats[..]);
    gpu::PipelineHandle first = gpu::create_graphics_pipeline(&device, &desc_rgba)!!;
    defer gpu::destroy_pipeline(&device, first)!!;

    gpu::Format[1] bgra_formats = {
        gpu::Format.BGRA8_UNORM,
    };
    gpu::GraphicsPipelineDesc desc_bgra =
        base_graphics_desc(&shaders, bgra_formats[..]);
    gpu::PipelineHandle second = gpu::create_graphics_pipeline(&device, &desc_bgra)!!;
    defer gpu::destroy_pipeline(&device, second)!!;
    assert(cache_entry_count(&device) == 2, "different formats produce distinct pipelines");

    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    vk::PipelineLayout first_layout = state.pipelines.slots[first.index].value.layout;
    vk::PipelineLayout second_layout = state.pipelines.slots[second.index].value.layout;
    assert(first_layout == second_layout, "graphics pipelines share one layout");
    assert(first_layout == state.graphics_layout, "the shared layout is the device graphics layout");

    gpu::Format[gpu::MAX_COLOR_ATTACHMENTS + 1] too_many;
    foreach (&format : too_many) *format = gpu::Format.RGBA8_UNORM;
    gpu::GraphicsPipelineDesc over_limit = base_graphics_desc(&shaders, too_many[..]);
    assert(@catch(gpu::create_graphics_pipeline(&device, &over_limit)) == gpu::INVALID_ARGUMENT,
        "more than MAX_COLOR_ATTACHMENTS color targets faults INVALID_ARGUMENT");
}

fn void compute_pipelines_dedup_and_share_layouts() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;

    gpu::ShaderDesc comp_sd = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc comp = comp_sd;

    gpu::ComputePipelineDesc desc = { .shader = comp };
    gpu::PipelineHandle first = gpu::create_compute_pipeline(&device, &desc)!!;
    defer gpu::destroy_pipeline(&device, first)!!;
    gpu::PipelineHandle second = gpu::create_compute_pipeline(&device, &desc)!!;
    defer gpu::destroy_pipeline(&device, second)!!;
    assert(first != second, "compute aliases are distinct handles");
    assert(cache_entry_count(&device) == 1, "identical compute descriptors share one entry");

    gpu::ShaderDesc alt_sd = {
        .spirv       = CACHE_ALT_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc alt = alt_sd;
    gpu::ComputePipelineDesc distinct_desc = { .shader = alt };
    gpu::PipelineHandle third = gpu::create_compute_pipeline(
        &device,
        &distinct_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, third)!!;
    assert(cache_entry_count(&device) == 2,
        "distinct compute shaders create distinct pipeline entries");

    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    vk::PipelineLayout first_layout = state.pipelines.slots[first.index].value.layout;
    vk::PipelineLayout second_layout = state.pipelines.slots[second.index].value.layout;
    vk::PipelineLayout third_layout = state.pipelines.slots[third.index].value.layout;
    assert(first_layout == second_layout && first_layout == third_layout,
        "all compute pipelines share one layout");
    assert(first_layout == state.compute_layout,
        "compute pipelines use the device singleton layout");
}

fn void caller_shader_storage_is_call_scoped_and_deduplicates() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;

    char[] source = CACHE_COMP_SPIRV[..];
    char[] first_storage = mem::new_array(char, source.len);
    mem::copy(first_storage.ptr, source.ptr, source.len);
    gpu::ShaderDesc first_desc = {
        .spirv       = first_storage,
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc first_pipeline_desc = {
        .shader = first_desc,
    };
    vk::VkDeviceState* state = cache_test_state(&device);
    char* first_caller_ptr = first_storage.ptr;

    gpu::PipelineHandle first =
        gpu::create_compute_pipeline(&device, &first_pipeline_desc)!!;
    uint first_entry_index = state.pipelines.slots[first.index].value.entry;
    vk::PipelineCacheEntry* cache_entry =
        &state.pipeline_cache.entries[first_entry_index];
    vk::ShaderId shader_id = cache_entry.key.vertex_shader;
    vk::ShaderStoreEntry* owned =
        &state.shader_store.entries[(uint)shader_id];
    char[] owned_spirv = vk::shader_entry_bytes(owned);
    assert(owned.used && owned.refcount == 1,
        "the cache entry owns one live shader identity");
    assert((char*)owned.spirv_words.ptr != first_caller_ptr
            && owned.spirv_byte_size == source.len
            && owned.spirv_words.len * uint::size
                == owned.spirv_byte_size
            && mem::ptr_is_aligned(
                owned.spirv_words.ptr,
                uint::alignment,
            )
            && mem::equals(owned_spirv, source),
        "the shader store owns one exact aligned copy of caller-provided SPIR-V");
    foreach (&byte : first_storage) *byte = 0;
    mem::free(first_storage.ptr);

    char[] second_storage = mem::new_array(char, source.len);
    mem::copy(second_storage.ptr, source.ptr, source.len);
    gpu::ShaderDesc second_desc = {
        .spirv       = second_storage,
        .entry_point = "main",
    };
    gpu::ComputePipelineDesc second_pipeline_desc = {
        .shader = second_desc,
    };
    gpu::PipelineHandle second =
        gpu::create_compute_pipeline(&device, &second_pipeline_desc)!!;
    uint second_entry_index = state.pipelines.slots[second.index].value.entry;
    assert(second_entry_index == first_entry_index,
        "equivalent caller storage resolves to the same cache entry");
    assert(state.pipeline_cache.entries[second_entry_index]
            .key.vertex_shader == shader_id,
        "equivalent caller storage preserves the canonical shader ID");
    assert(owned.used && owned.refcount == 1
            && mem::equals(vk::shader_entry_bytes(owned), source),
        "a cache hit keeps one cache-owned exact shader identity");
    foreach (&byte : second_storage) *byte = 0;
    mem::free(second_storage.ptr);

    assert(cache_entry_count(&device) == 1,
        "released caller storage does not change exact cache identity");

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator =
        gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::cmd_bind_pipeline(&commands, first)!!;
    gpu::cmd_bind_pipeline(&commands, second)!!;
    gpu::discard_commands(&commands)!!;

    gpu::destroy_pipeline(&device, first)!!;
    assert(owned.used && owned.refcount == 1,
        "one surviving pipeline alias keeps the canonical shader identity");
    gpu::destroy_pipeline(&device, second)!!;
    assert(vk::shader_store_live_count(state) == 0,
        "the final pipeline alias releases the canonical shader identity");
    assert(vk::shader_store_index_consistent(state),
        "caller-storage aliases leave the shader index consistent");
}

fn void native_shader_module_borrows_aligned_store_words() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = cache_test_state(&device);
    gpu::ShaderDesc desc = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    vk::ShaderIdentityInput borrowed_input;
    vk::normalize_shader_input(
        state:          state,
        desc:           &desc,
        role:           gpu::internal::ShaderRole.COMPUTE,
        operation:      "test_aligned_shader_store",
        rejected_field: "ShaderDesc",
        out_input:      &borrowed_input,
    )!!;

    state.resource_mutex.lock()!!;
    vk::ShaderId shader_id =
        vk::retain_shader_identity_locked(state, &borrowed_input)!!;
    state.resource_mutex.unlock();
    bool retained = true;
    defer {
        if (retained) {
            state.resource_mutex.lock()!!;
            vk::release_shader_identity_locked(state, shader_id)!!;
            state.resource_mutex.unlock();
        }
    }

    vk::ShaderStoreEntry* entry =
        &state.shader_store.entries[(uint)shader_id];
    vk::ShaderIdentityInput stored_input;
    vk::shader_store_input_for_retained_id(
        state:      state,
        id:         shader_id,
        debug_name: null,
        out_input:  &stored_input,
    );
    ShaderModuleCodeCapture capture = {
        .expected_words     = entry.spirv_words.ptr,
        .expected_byte_size = entry.spirv_byte_size,
        .expected_bytes     = CACHE_COMP_SPIRV[..],
    };
    vk::ShaderModuleCreateOps create_ops = {
        .user_data = &capture,
        .create    = &capture_shader_module_code,
    };
    vk::PipelineShaderModule shader =
        vk::create_pipeline_shader_module_with_ops(
            state:          state,
            code:           &stored_input,
            operation:      "test_aligned_shader_store",
            rejected_field: "ShaderDesc",
            create_ops:      &create_ops,
        )!!;
    assert(capture.matched,
        "native creation borrows the retained entry's exact aligned words");
    vk::destroy_pipeline_shader_module(state, &shader);

    state.resource_mutex.lock()!!;
    vk::release_shader_identity_locked(state, shader_id)!!;
    retained = false;
    state.resource_mutex.unlock();
    assert(vk::shader_store_live_count(state) == 0
            && entry.spirv_words.ptr == null,
        "final release frees aligned storage and recycles the entry");
}

fn void shader_store_cache_hit_reference_is_stable() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc shader_desc = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;
    gpu::ComputePipelineDesc desc = { .shader = shader };
    vk::VkDeviceState* state = cache_test_state(&device);

    gpu::PipelineHandle first = gpu::create_compute_pipeline(&device, &desc)!!;
    uint entry_index = state.pipelines.slots[first.index].value.entry;
    vk::ShaderId shader_id = state.pipeline_cache.entries[entry_index]
        .key.vertex_shader;
    assert(vk::shader_store_live_count(state) == 1,
        "one cache entry owns one compute shader identity");
    assert(state.shader_store.entries[(uint)shader_id].refcount == 1,
        "the cache entry owns exactly one shader reference");

    gpu::PipelineHandle second = gpu::create_compute_pipeline(&device, &desc)!!;
    uint second_entry_index = state.pipelines.slots[second.index].value.entry;
    assert(second_entry_index == entry_index,
        "a repeated description aliases the existing cache entry");
    assert(state.pipeline_cache.entries[second_entry_index]
            .key.vertex_shader == shader_id,
        "a cache hit preserves the canonical shader ID");
    assert(state.shader_store.entries[(uint)shader_id].refcount == 1,
        "a cache hit releases its temporary shader reference");

    gpu::destroy_pipeline(&device, first)!!;
    assert(vk::shader_store_live_count(state) == 1,
        "one surviving alias keeps its cache-owned shader identity");
    assert(state.pipeline_cache.entries[entry_index].refcount == 1,
        "one surviving alias keeps one cache-entry reference");
    gpu::destroy_pipeline(&device, second)!!;
    assert(vk::shader_store_live_count(state) == 0,
        "the last alias releases its shader identity");
    assert(vk::shader_store_index_consistent(state),
        "shader buckets remain consistent after alias release");
}

fn void shader_store_collision_verification_is_exact() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    char[] distinct_storage = mem::new_array(char, CACHE_COMP_SPIRV.len);
    defer mem::free(distinct_storage.ptr);
    mem::copy(
        distinct_storage.ptr,
        CACHE_COMP_SPIRV[..].ptr,
        CACHE_COMP_SPIRV.len,
    );
    distinct_storage[distinct_storage.len - 1] ^= 1;
    gpu::ShaderDesc first_desc = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc distinct_desc = first_desc;
    distinct_desc.spirv = distinct_storage;
    vk::VkDeviceState* state = cache_test_state(&device);
    vk::ShaderIdentityInput first;
    vk::ShaderIdentityInput distinct;
    vk::ShaderIdentityInput alternate_entry;
    vk::ShaderIdentityInput different_length;
    vk::normalize_shader_input(
        state:          state,
        desc:           &first_desc,
        role:           gpu::internal::ShaderRole.COMPUTE,
        operation:      "test_shader_store_collision",
        rejected_field: "ShaderDesc",
        out_input:      &first,
    )!!;
    vk::normalize_shader_input(
        state:          state,
        desc:           &distinct_desc,
        role:           gpu::internal::ShaderRole.COMPUTE,
        operation:      "test_shader_store_collision",
        rejected_field: "ShaderDesc",
        out_input:      &distinct,
    )!!;
    distinct.hash = first.hash;
    gpu::ShaderDesc alternate_entry_desc = first_desc;
    alternate_entry_desc.entry_point = "alternate";
    vk::normalize_shader_input(
        state:          state,
        desc:           &alternate_entry_desc,
        role:           gpu::internal::ShaderRole.COMPUTE,
        operation:      "test_shader_store_collision",
        rejected_field: "ShaderDesc",
        out_input:      &alternate_entry,
    )!!;
    alternate_entry.hash = first.hash;
    vk::ShaderIdentityInput alternate_role = first;
    alternate_role.role = gpu::internal::ShaderRole.VERTEX;
    different_length = first;
    different_length.spirv = first.spirv[0:first.spirv.len - 4];

    state.resource_mutex.lock()!!;
    vk::ShaderId first_id = vk::retain_shader_identity_locked(state, &first)!!;
    vk::ShaderId distinct_id = vk::retain_shader_identity_locked(
        state,
        &distinct,
    )!!;
    vk::ShaderId alternate_entry_id =
        vk::retain_shader_identity_locked(state, &alternate_entry)!!;
    vk::ShaderId alternate_role_id =
        vk::retain_shader_identity_locked(state, &alternate_role)!!;
    vk::ShaderId different_length_id =
        vk::retain_shader_identity_locked(state, &different_length)!!;
    vk::ShaderId first_alias_id =
        vk::retain_shader_identity_locked(state, &first)!!;
    state.resource_mutex.unlock();
    assert(first_id != distinct_id
            && first_id != alternate_entry_id
            && first_id != alternate_role_id
            && first_id != different_length_id
            && distinct_id != alternate_entry_id
            && distinct_id != alternate_role_id
            && distinct_id != different_length_id
            && alternate_entry_id != alternate_role_id
            && alternate_entry_id != different_length_id
            && alternate_role_id != different_length_id,
        "bytes, byte length, entry point, and role distinguish hash collisions");
    assert(first_alias_id == first_id
            && state.shader_store.entries[(uint)first_id].refcount == 2,
        "an exact identity deduplicates within the collision bucket");
    vk::ShaderStoreEntry* first_entry =
        &state.shader_store.entries[(uint)first_id];
    vk::ShaderStoreEntry* distinct_entry =
        &state.shader_store.entries[(uint)distinct_id];
    vk::ShaderStoreEntry* alternate_entry_value =
        &state.shader_store.entries[(uint)alternate_entry_id];
    vk::ShaderStoreEntry* alternate_role_entry =
        &state.shader_store.entries[(uint)alternate_role_id];
    vk::ShaderStoreEntry* different_length_entry =
        &state.shader_store.entries[(uint)different_length_id];
    assert(first_entry.used
            && distinct_entry.used
            && alternate_entry_value.used
            && alternate_role_entry.used
            && different_length_entry.used,
        "every collision-separated identity occupies a live store entry");
    assert(first_entry.hash == first.hash
            && distinct_entry.hash == first.hash
            && alternate_entry_value.hash == first.hash
            && alternate_role_entry.hash == first.hash
            && different_length_entry.hash == first.hash,
        "the test exercises one shared hash bucket");
    assert(mem::equals(vk::shader_entry_bytes(first_entry), first.spirv)
            && mem::equals(
                vk::shader_entry_bytes(distinct_entry),
                distinct.spirv,
            )
            && mem::equals(
                vk::shader_entry_bytes(alternate_entry_value),
                alternate_entry.spirv,
            )
            && mem::equals(
                vk::shader_entry_bytes(alternate_role_entry),
                alternate_role.spirv,
            )
            && mem::equals(
                vk::shader_entry_bytes(different_length_entry),
                different_length.spirv,
            ),
        "each stored identity preserves its exact owned SPIR-V bytes");
    assert(first_entry.role == first.role
            && alternate_role_entry.role == alternate_role.role
            && ((ZString)&alternate_entry_value.entry_point[0]).str_view()
                == "alternate",
        "stored identities preserve role and entry-point distinctions");
    assert(vk::shader_store_live_count(state) == 5,
        "all collision-separated identities remain live");
    assert(vk::shader_store_index_consistent(state),
        "collision-separated entries remain bucket-indexed");

    state.resource_mutex.lock()!!;
    vk::release_shader_identity_locked(state, first_alias_id)!!;
    assert(state.shader_store.entries[(uint)first_id].refcount == 1,
        "releasing an exact alias preserves the original identity");
    vk::release_shader_identity_locked(state, first_id)!!;
    vk::release_shader_identity_locked(state, distinct_id)!!;
    vk::release_shader_identity_locked(state, alternate_entry_id)!!;
    vk::release_shader_identity_locked(state, alternate_role_id)!!;
    vk::release_shader_identity_locked(state, different_length_id)!!;
    state.resource_mutex.unlock();
    assert(vk::shader_store_live_count(state) == 0,
        "releasing every collision-separated identity empties the store");
    assert(vk::shader_store_index_consistent(state),
        "collision release preserves bucket and free-list consistency");
}

fn void shader_store_reuses_entries_across_unique_churn() @test {
    const uint CHURN_COUNT = 8;
    gpu::Device device = make_cache_test_device_with_capacity(1);
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc shader_desc = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;
    gpu::ComputePipelineDesc desc = { .shader = shader };
    vk::VkDeviceState* state = cache_test_state(&device);

    for (uint i = 0; i < CHURN_COUNT; i++) {
        vk::ShaderIdentityInput input;
        vk::normalize_shader_input(
            state:          state,
            desc:           &desc.shader,
            role:           gpu::internal::ShaderRole.COMPUTE,
            operation:      "test_shader_store_churn",
            rejected_field: "ShaderDesc",
            out_input:      &input,
        )!!;
        input.hash += (ulong)i + 1;
        state.resource_mutex.lock()!!;
        vk::ShaderId id = vk::retain_shader_identity_locked(state, &input)!!;
        assert(id == (vk::ShaderId)0,
            "unique churn reuses the first shader-store slot");
        vk::release_shader_identity_locked(state, id)!!;
        state.resource_mutex.unlock();
        assert(vk::shader_store_live_count(state) == 0,
            "each churn iteration releases its unique shader identity");
        assert(state.shader_store.free_head == 1,
            "each released identity returns the first slot to the free list");
        assert(vk::shader_store_index_consistent(state),
            "shader free-list reuse preserves bucket consistency");
    }
    assert(state.shader_store.next_new == 1,
        "unique historical churn reuses one store slot");
}

fn void full_pipeline_capacity_faults_before_shader_store_mutation() @test {
    gpu::Device device = make_cache_test_device_with_capacity(1);
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();
    gpu::Format[1] colors = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc graphics =
        base_graphics_desc(&shaders, colors[..]);
    gpu::PipelineHandle live =
        gpu::create_graphics_pipeline(&device, &graphics)!!;
    vk::VkDeviceState* state = cache_test_state(&device);
    assert(vk::shader_store_live_count(state) == 2,
        "one live graphics pipeline fills the two-identity store");

    gpu::ShaderDesc compute_shader_desc = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc compute_shader = compute_shader_desc;
    gpu::ComputePipelineDesc compute = { .shader = compute_shader };
    uint live_before = vk::shader_store_live_count(state);
    uint next_new_before = state.shader_store.next_new;
    uint free_head_before = state.shader_store.free_head;
    uint cache_entries_before = cache_entry_count(&device);
    uint graphics_entry_index =
        state.pipelines.slots[live.index].value.entry;
    vk::ShaderId vertex_id =
        state.pipeline_cache.entries[graphics_entry_index].key.vertex_shader;
    vk::ShaderId fragment_id =
        state.pipeline_cache.entries[graphics_entry_index].key.fragment_shader;
    uint vertex_refcount =
        state.shader_store.entries[(uint)vertex_id].refcount;
    uint fragment_refcount =
        state.shader_store.entries[(uint)fragment_id].refcount;
    fault capacity_fault =
        @catch(gpu::create_compute_pipeline(&device, &compute));
    assert(capacity_fault == gpu::SLOT_TABLE_FULL,
        "a full pipeline handle table faults before interning");
    assert(vk::shader_store_live_count(state) == live_before
            && state.shader_store.next_new == next_new_before
            && state.shader_store.free_head == free_head_before,
        "capacity failure leaves existing shader identities unchanged");
    assert(cache_entry_count(&device) == cache_entries_before
            && state.shader_store.entries[(uint)vertex_id].refcount
                == vertex_refcount
            && state.shader_store.entries[(uint)fragment_id].refcount
                == fragment_refcount,
        "capacity failure preserves cache membership and shader references");
    assert(vk::shader_store_index_consistent(state),
        "capacity failure preserves the shader index");

    gpu::destroy_pipeline(&device, live)!!;
    assert(vk::shader_store_live_count(state) == 0,
        "destroying the live pipeline releases both identities");
}

fn void pipeline_creation_faults_release_owned_shader_identity() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = cache_test_state(&device);

    char[] truncated_spirv = CACHE_COMP_SPIRV[0:20];
    gpu::ShaderDesc truncated_desc = {
        .spirv       = truncated_spirv,
        .entry_point = "main",
    };
    gpu::ShaderDesc truncated = truncated_desc;
    gpu::ComputePipelineDesc reflection_desc = { .shader = truncated };
    assert(@catch(gpu::create_compute_pipeline(&device, &reflection_desc))
            == gpu::SHADER_INVALID,
        "reflection failure preserves SHADER_INVALID");
    assert(vk::shader_store_live_count(state) == 0
            && cache_entry_count(&device) == 0,
        "reflection failure retains no store or cache entry");
    assert(state.shader_store.next_new == 1
            && state.shader_store.free_head == 1
            && !state.shader_store.entries[0].used
            && state.shader_store.entries[0].spirv_words.ptr == null,
        "reflection failure returns its shader slot to the free list");
    assert(vk::shader_store_index_consistent(state)
            && cache_index_consistent(&device),
        "reflection failure preserves both indexes");

    gpu::ShaderDesc shader_desc = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;
    gpu::ComputePipelineDesc desc = { .shader = shader };

    vk::Result shader_create_result = vk::Result.ERROR_OUT_OF_HOST_MEMORY;
    vk::ComputePipelineCreateOps shader_create_ops =
        vk::compute_pipeline_create_ops_real();
    shader_create_ops.shader_module = {
        .user_data = &shader_create_result,
        .create    = &return_shader_module_create_result,
    };
    assert(@catch(vk::vk_create_compute_pipeline_with_ops(
        state:      state,
        desc:       &desc,
        create_ops: &shader_create_ops,
    ))
            == gpu::OUT_OF_HOST_MEMORY,
        "native shader creation preserves host OOM");
    assert(vk::shader_store_live_count(state) == 0
            && cache_entry_count(&device) == 0,
        "native shader failure retains no store or cache entry");
    assert(state.shader_store.next_new == 1
            && state.shader_store.free_head == 1
            && !state.shader_store.entries[0].used,
        "native shader failure returns its shader slot to the free list");

    vk::Result pipeline_create_result = vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
    vk::ComputePipelineCreateOps pipeline_create_ops =
        vk::compute_pipeline_create_ops_real();
    pipeline_create_ops.user_data = &pipeline_create_result;
    pipeline_create_ops.create = &return_compute_pipeline_create_result;
    assert(@catch(vk::vk_create_compute_pipeline_with_ops(
        state:      state,
        desc:       &desc,
        create_ops: &pipeline_create_ops,
    ))
            == gpu::OUT_OF_DEVICE_MEMORY,
        "native pipeline creation preserves device OOM");
    assert(vk::shader_store_live_count(state) == 0
            && cache_entry_count(&device) == 0,
        "native pipeline failure retains no store or cache entry");
    assert(state.shader_store.next_new == 1
            && state.shader_store.free_head == 1
            && !state.shader_store.entries[0].used,
        "native pipeline failure reuses and releases the same shader slot");

    vk::ComputePipelineCreateOps cache_insert_ops =
        vk::compute_pipeline_create_ops_real();
    cache_insert_ops.insert_cache_entry = &fail_pipeline_cache_insert;
    assert(@catch(vk::vk_create_compute_pipeline_with_ops(
        state:      state,
        desc:       &desc,
        create_ops: &cache_insert_ops,
    ))
            == gpu::SLOT_TABLE_FULL,
        "cache insertion failure preserves SLOT_TABLE_FULL");
    assert(vk::shader_store_live_count(state) == 0
            && cache_entry_count(&device) == 0,
        "cache insertion failure retains no store or cache entry");
    assert(state.shader_store.next_new == 1
            && state.shader_store.free_head == 1
            && !state.shader_store.entries[0].used,
        "cache insertion failure reuses and releases the same shader slot");
    assert(vk::shader_store_index_consistent(state)
            && cache_index_consistent(&device),
        "every injected fault preserves both indexes");
}

fn void shader_desc_is_reusable_across_devices() @test {
    gpu::ShaderDesc shader_desc = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;

    gpu::Device first_device = make_cache_test_device();
    defer destroy_test_device(&first_device)!!;
    gpu::Device second_device = make_cache_test_device();
    defer destroy_test_device(&second_device)!!;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader             = shader,
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
        "the second device owns an independent entry for shared shader input");
}

fn void dynamic_raster_state_renders_with_one_pipeline() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

    // The offscreen fragment shader samples the heap, so a 1x1 white pattern
    // texture and a sampler feed both draws.
    gpu::TextureDesc tex_desc = {
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
    gpu::SamplerIndex smp_index = gpu::intern_sampler(&device, &sampler_desc)!!;
    char[4] white = { (char)255, (char)255, (char)255, (char)255 };
    upload_test_texture_data(
        device:   &device,
        dst:      pattern,
        mip:      0,
        data:     white[..],
        next_use: gpu::sampled_at({ .fragment_shader }),
    )!!;

    gpu::TextureDesc rt_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = CACHE_RT_SIZE,
        .height    = CACHE_RT_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach, .transfer_src },
    };
    gpu::TextureHandle target_culled = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target_culled)!!;
    gpu::AttachmentViewHandle culled_view = create_test_attachment_view(
        &device,
        target_culled,
    )!!;
    defer destroy_test_attachment_view(&device, &culled_view)!!;
    gpu::TextureHandle target_drawn = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target_drawn)!!;
    gpu::AttachmentViewHandle drawn_view = create_test_attachment_view(
        &device,
        target_drawn,
    )!!;
    defer destroy_test_attachment_view(&device, &drawn_view)!!;
    gpu::TextureHandle target_reset = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target_reset)!!;
    gpu::AttachmentViewHandle reset_view = create_test_attachment_view(
        &device,
        target_reset,
    )!!;
    defer destroy_test_attachment_view(&device, &reset_view)!!;
    gpu::TextureHandle target_lines = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target_lines)!!;
    gpu::AttachmentViewHandle lines_view = create_test_attachment_view(
        &device,
        target_lines,
    )!!;
    defer destroy_test_attachment_view(&device, &lines_view)!!;

    gpu::Format[1] pipeline_colors = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc pipeline_desc = base_graphics_desc(&shaders, pipeline_colors[..]);
    gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(&device, &pipeline_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    assert(cache_entry_count(&device) == 1, "dynamic cull state shares one graphics identity");

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
    OwnedSpan rb_lines = allocate_owned_span(
        device:       &device,
        size:         CACHE_RT_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "cache_lines_readback",
    )!!;
    defer free_owned_span(&device, &rb_lines)!!;
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

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;

    gpu::TextureHandle[4] targets = {
        target_culled,
        target_drawn,
        target_reset,
        target_lines,
    };
    gpu::AttachmentViewHandle[4] target_views = {
        culled_view,
        drawn_view,
        reset_view,
        lines_view,
    };
    gpu::GpuSpan[4] readbacks = {
        rb_culled.span,
        rb_drawn.span,
        rb_reset.span,
        rb_lines.span,
    };
    for (uint i = 0; i < targets.len; i++) {
        gpu::TextureBarrier to_attach = {
            .texture = targets[i],
            .before  = { .layout = gpu::TextureLayout.UNDEFINED },
            .after   = {
                .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
                .stages = { .color_output },
                .access = { .read, .write },
            },
        };
        gpu::cmd_texture_barrier(&cmd, &to_attach)!!;

        gpu::ColorTargetDesc[1] colors = {{
            .view = target_views[i],
            .load_op = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear = { .rgba = { 8.0f / 255.0f, 16.0f / 255.0f, 32.0f / 255.0f, 1.0f } },
        }};
        gpu::RenderPassDesc pass = { .colors = colors[..], .width = CACHE_RT_SIZE, .height = CACHE_RT_SIZE };
        gpu::GraphicsState graphics =
            gpu::render_geometry_state(pass.width, pass.height)!!;
        gpu::ColorTargetState[1] color_targets = {
            gpu::color_blend_disabled(),
        };
        graphics.color.targets = color_targets[..];
        if (i == 0) {
            graphics.raster.cull_mode = gpu::CullMode.BACK;
        }
        if (i == 1) {
            graphics.raster = {
                .cull_mode  = gpu::CullMode.BACK,
                .front_face = gpu::FrontFace.CLOCKWISE,
            };
            graphics.viewport = {
                .width = (float)CACHE_RT_SIZE / 2.0f, .height = (float)CACHE_RT_SIZE, .max_depth = 1.0f,
            };
            graphics.scissor = {
                .width = (int)CACHE_RT_SIZE / 2, .height = (int)CACHE_RT_SIZE,
            };
        }
        if (i == 3) {
            graphics.raster = {
                .topology = gpu::PrimitiveTopology.LINES,
            };
        }
        gpu::cmd_begin_render_pass(&cmd, &pass)!!;
        bind_test_graphics_pipeline(&cmd, pipeline);
        gpu::cmd_set_graphics_state(&cmd, &graphics)!!;
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
            .before  = {
                .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
                .stages = { .color_output },
                .access = { .read, .write },
            },
            .after   = {
                .layout = gpu::TextureLayout.TRANSFER_SOURCE,
                .stages = { .transfer },
                .access = { .read },
            },
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
        "clockwise front-face state prevents back culling on the left");
    assert(drawn_right[0] == (char)8 && drawn_right[1] == (char)16 && drawn_right[2] == (char)32,
        "dynamic raster changes preserve clipping on the right");

    gpu::invalidate_mapped_span(&device, rb_reset.span)!!;
    gpu::GpuSpan reset_span = rb_reset.span;
    char* reset_pixels = (char*)test_span_mapping(&device, reset_span).ptr;
    char* reset_center = &reset_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE / 2) * 4];
    assert(reset_center[0] == (char)255 && reset_center[1] == (char)255 && reset_center[2] == (char)255,
        "the next pass restores raster, viewport, and scissor defaults");

    gpu::invalidate_mapped_span(&device, rb_lines.span)!!;
    gpu::GpuSpan lines_span = rb_lines.span;
    char* lines_pixels = (char*)test_span_mapping(&device, lines_span).ptr;
    char* lines_center = &lines_pixels[(CACHE_RT_SIZE / 2 * CACHE_RT_SIZE + CACHE_RT_SIZE / 2) * 4];
    assert(lines_center[0] == (char)8 && lines_center[1] == (char)16 && lines_center[2] == (char)32,
        "line topology changes triangle interior coverage without creating a pipeline");
}

fn void alias_survives_peer_destroy() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

    gpu::Format[1] pipeline_colors = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, pipeline_colors[..]);
    gpu::PipelineHandle doomed = gpu::create_graphics_pipeline(&device, &desc)!!;
    gpu::PipelineHandle survivor = gpu::create_graphics_pipeline(&device, &desc)!!;
    gpu::destroy_pipeline(&device, doomed)!!;

    gpu::TextureDesc rt_desc = {
        .access    = { .graphics, .compute, .transfer },
        .width     = CACHE_RT_SIZE,
        .height    = CACHE_RT_SIZE,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .color_attach },
    };
    gpu::TextureHandle target = gpu::create_texture(&device, &rt_desc)!!;
    defer gpu::destroy_texture(&device, target)!!;
    gpu::AttachmentViewHandle target_view = create_test_attachment_view(
        &device,
        target,
    )!!;
    defer destroy_test_attachment_view(&device, &target_view)!!;
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

    gpu::Queue command_queue = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(&device, command_queue)!!;
    defer gpu::destroy_command_allocator(&command_allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&command_allocator)!!;
    defer discard_test_commands(&cmd)!!;
    gpu::TextureBarrier to_attach = {
        .texture = target,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    };
    gpu::cmd_texture_barrier(&cmd, &to_attach)!!;
    gpu::ColorTargetDesc[1] colors = {{
        .view = target_view, .load_op = gpu::LoadOp.CLEAR, .store_op = gpu::StoreOp.STORE,
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..], .width = CACHE_RT_SIZE, .height = CACHE_RT_SIZE };
    gpu::GraphicsState graphics =
        gpu::render_geometry_state(pass.width, pass.height)!!;
    gpu::ColorTargetState[1] color_targets = {
        gpu::color_blend_disabled(),
    };
    graphics.color.targets = color_targets[..];
    gpu::cmd_begin_render_pass(&cmd, &pass)!!;
    bind_test_graphics_pipeline(&cmd, survivor);
    gpu::cmd_set_graphics_state(&cmd, &graphics)!!;
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

    gpu::Format[1] pipeline_colors = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::Format[CHURN_CAPACITY] churn_formats = {
        gpu::Format.R8_UNORM,
        gpu::Format.R8_UINT,
        gpu::Format.RG8_UNORM,
        gpu::Format.RGBA8_UNORM,
        gpu::Format.RGBA8_SRGB,
        gpu::Format.BGRA8_UNORM,
        gpu::Format.BGRA8_SRGB,
        gpu::Format.R16_UINT,
    };
    uint[CHURN_CAPACITY] teardown_order = { 3, 0, 7, 1, 6, 2, 5, 4 };

    for (uint round = 0; round < CHURN_ROUNDS; round++) {
        gpu::PipelineHandle[CHURN_CAPACITY] handles;
        for (uint i = 0; i < CHURN_CAPACITY; i++) {
            pipeline_colors[0] = churn_formats[i];
            gpu::GraphicsPipelineDesc variant = base_graphics_desc(&shaders, pipeline_colors[..]);
            handles[i] = gpu::create_graphics_pipeline(&device, &variant)!!;
        }
        assert(cache_index_consistent(&device), "index consistent once all distinct entries are inserted");
        assert(cache_entry_count(&device) == CHURN_CAPACITY,
            "each round inserts exactly CHURN_CAPACITY distinct entries");

        // Alias two live entries; the refcount>0 path must not touch the index.
        pipeline_colors[0] = churn_formats[0];
        gpu::GraphicsPipelineDesc alias_desc =
            base_graphics_desc(&shaders, pipeline_colors[..]);
        gpu::PipelineHandle alias_a = gpu::create_graphics_pipeline(&device, &alias_desc)!!;
        pipeline_colors[0] = churn_formats[1];
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

    gpu::Format[1] pipeline_colors;
    gpu::Format[CAPACITY + 1] formats = {
        gpu::Format.R8_UNORM,
        gpu::Format.RG8_UNORM,
        gpu::Format.RGBA8_UNORM,
        gpu::Format.BGRA8_UNORM,
    };
    gpu::PipelineHandle[CAPACITY] handles;
    for (uint i = 0; i < CAPACITY; i++) {
        pipeline_colors[0] = formats[i];
        gpu::GraphicsPipelineDesc variant = base_graphics_desc(&shaders, pipeline_colors[..]);
        handles[i] = gpu::create_graphics_pipeline(&device, &variant)!!;
    }
    assert(cache_entry_count(&device) == CAPACITY, "the small capacity fills exactly");
    assert(cache_index_consistent(&device), "index consistent at capacity");

    pipeline_colors[0] = formats[CAPACITY];
    gpu::GraphicsPipelineDesc over_capacity = base_graphics_desc(&shaders, pipeline_colors[..]);
    assert(@catch(gpu::create_graphics_pipeline(&device, &over_capacity)) == gpu::SLOT_TABLE_FULL,
        "the distinct pipeline past the configured capacity faults SLOT_TABLE_FULL");

    foreach (handle : handles) gpu::destroy_pipeline(&device, handle)!!;
}

fn void line_polygon_mode_respects_optional_feature() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();
    gpu::Format[1] pipeline_colors = {
        gpu::Format.RGBA8_UNORM,
    };
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    bool supported = state.caps.line_polygon_mode;

    state.caps.line_polygon_mode = false;
    gpu::GraphicsPipelineDesc unavailable = {
        .color_formats = pipeline_colors[..],
        .polygon_mode = gpu::PolygonMode.LINE,
    };
    uint entries_before = cache_entry_count(&device);
    assert(@catch(gpu::create_graphics_pipeline(&device, &unavailable)) == gpu::UNSUPPORTED_FEATURE,
        "unsupported LINE faults before invalid shader input validation");
    assert(cache_entry_count(&device) == entries_before, "unsupported LINE publishes no cache entry");
    state.caps.line_polygon_mode = supported;

    if (!supported) return;
    gpu::GraphicsPipelineDesc line_polygon = base_graphics_desc(&shaders, pipeline_colors[..]);
    line_polygon.polygon_mode = gpu::PolygonMode.LINE;
    gpu::PipelineHandle line_pipeline = gpu::create_graphics_pipeline(&device, &line_polygon)!!;
    gpu::destroy_pipeline(&device, line_pipeline)!!;
}

fn void compute_layout_exists_before_pipeline_creation() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    assert(state.compute_layout != null,
        "device creation publishes the singleton compute layout");
    assert(state.generated_work.enabled
            ? state.generated_dispatch_layout != null
            : state.generated_dispatch_layout == null,
        "generated dispatch singleton follows device capability");
    vk::PipelineLayout compute_layout = state.compute_layout;
    vk::IndirectCommandsLayoutEXT generated_layout =
        state.generated_dispatch_layout;

    gpu::ShaderDesc comp_sd = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc comp = comp_sd;
    gpu::ComputePipelineDesc desc = { .shader = comp };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &desc)!!;
    assert(state.pipelines.slots[pipeline.index].value.layout
        == state.compute_layout,
        "compute creation copies the singleton layout into the pipeline slot");
    assert(state.pipelines.slots[pipeline.index]
            .value.generated_dispatch_layout == generated_layout,
        "compute creation copies the generated singleton into the pipeline slot");
    gpu::destroy_pipeline(&device, pipeline)!!;
    assert(state.compute_layout == compute_layout
            && state.generated_dispatch_layout == generated_layout,
        "pipeline destruction leaves device-owned singleton layouts live");
}

fn void generated_dispatch_layout_survives_concurrent_pipeline_creation() @test {
    gpu::Device device = make_generated_work_cache_test_device();
    defer destroy_test_device(&device)!!;
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    if (!caps.generated_work) return;

    gpu::ShaderDesc shader_desc = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;
    gpu::ComputePipelineDesc pipeline_desc = {
        .shader             = shader,
    };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(
        &device,
        &pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;
    OwnedSpan records = allocate_owned_span(
        device:       &device,
        size:         gpu::GeneratedDispatchRecord::size,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics },
        debug_name:   "generated_dispatch_relocation_records",
    )!!;
    defer free_owned_span(&device, &records)!!;
    OwnedSpan count = allocate_owned_span(
        device:       &device,
        size:         uint::size,
        memory_class: gpu::MemoryClass.GPU_PRIVATE,
        access:       { .graphics },
        debug_name:   "generated_dispatch_relocation_count",
    )!!;
    defer free_owned_span(&device, &count)!!;

    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::GeneratedScratchDesc scratch = {
        .pipeline                = pipeline,
        .kind                    = gpu::GeneratedWorkKind.DISPATCH,
        .max_commands_per_list   = 1,
        .preprocess_buffer_count = 1,
    };
    vk::VkDeviceState* state =
        gpu::internal::device_slots[device.index].data.state;
    vk::MemoryRequirements requirements = vk::generated_scratch_requirements(
        state,
        &scratch,
    )!!;
    gpu::CommandAllocatorDesc allocator_desc = {
        .command_buffer_capacity = 1,
        .max_generated_preprocess_buffers_per_list = 1,
        .generated_preprocess_bytes = (usz)requirements.size,
    };
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
        &allocator_desc,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::reserve_generated_scratch(&allocator, &scratch)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&commands)!!;
    bind_test_compute_pipeline(&commands, pipeline);

    gpu::internal::vk::CommandRecord* record =
        gpu::internal::recording_command(&commands)!!;
    vk::IndirectCommandsLayoutEXT expected_layout =
        record.logical_pipeline.generated_dispatch_layout;
    GeneratedLayoutGate gate;
    GeneratedDispatchRecordArgs args = {
        .commands = commands,
        .records  = records.span,
        .count    = count.span,
        .gate     = &gate,
    };
    Thread worker;
    worker.create(&record_generated_dispatch, &args)!!;
    while (!gate.entered.load(AtomicOrdering.ACQUIRE)) thread::yield();
    assert(gate.captured == expected_layout,
        "generated dispatch captures the bind-time pipeline layout");

    gpu::ShaderDesc alternate_desc = {
        .spirv       = CACHE_ALT_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc alternate_shader = alternate_desc;
    gpu::ComputePipelineDesc alternate_pipeline_desc = {
        .shader = alternate_shader,
    };
    gpu::PipelineHandle alternate_pipeline = gpu::create_compute_pipeline(
        &device,
        &alternate_pipeline_desc,
    )!!;
    defer gpu::destroy_pipeline(&device, alternate_pipeline)!!;
    assert(state.pipelines.slots[alternate_pipeline.index]
        .value.generated_dispatch_layout == expected_layout,
        "concurrent compute creation copies the same generated layout");
    gate.release.store(true, AtomicOrdering.RELEASE);
    int worker_result = worker.join()!!;

    assert(expected_layout != null,
        "generated-work compute pipelines publish a generated layout");
    assert(worker_result == 0,
        "generated dispatch recording resumes after concurrent creation: %s",
        args.failure);
    assert(args.prepared_layout == gate.captured,
        "prepared dispatch retains the layout captured before concurrent creation");
    gpu::discard_commands(&commands)!!;
    gpu::release_generated_scratch(
        &allocator,
        pipeline,
        gpu::GeneratedWorkKind.DISPATCH,
    )!!;
}

fn void repeated_compute_creation_deduplicates_identity() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    gpu::ShaderDesc shader_desc = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc shader = shader_desc;
    char[] equivalent_storage = mem::new_array(char, CACHE_COMP_SPIRV.len);
    defer mem::free(equivalent_storage.ptr);
    mem::copy(
        equivalent_storage.ptr,
        CACHE_COMP_SPIRV[..].ptr,
        CACHE_COMP_SPIRV.len,
    );
    gpu::ShaderDesc equivalent_desc = shader_desc;
    equivalent_desc.spirv = equivalent_storage;
    gpu::ShaderDesc equivalent_shader = equivalent_desc;
    gpu::ShaderDesc distinct_desc = {
        .spirv       = CACHE_ALT_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderDesc distinct_shader = distinct_desc;
    gpu::ComputePipelineDesc[3] descs = {
        { .shader = shader },
        { .shader = equivalent_shader },
        { .shader = distinct_shader },
    };
    gpu::PipelineHandle[3] pipelines;
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;

    foreach (i, &desc : descs) {
        pipelines[i] = gpu::create_compute_pipeline(&device, desc)!!;
    }

    assert(pipelines[0].is_valid() && pipelines[1].is_valid() && pipelines[2].is_valid(),
        "each singular call publishes one pipeline handle");
    assert(pipelines[0] != pipelines[1], "duplicate descriptions publish distinct owning handles");
    assert(cache_entry_count(&device) == 2, "duplicate descriptions share deterministic pipeline identity");
    uint first_entry_index = state.pipelines.slots[pipelines[0].index]
        .value.entry;
    uint equivalent_entry_index = state.pipelines.slots[pipelines[1].index]
        .value.entry;
    uint distinct_entry_index = state.pipelines.slots[pipelines[2].index]
        .value.entry;
    vk::ShaderId shared_shader_id =
        state.pipeline_cache.entries[first_entry_index].key.vertex_shader;
    vk::ShaderId distinct_shader_id =
        state.pipeline_cache.entries[distinct_entry_index].key.vertex_shader;
    assert(first_entry_index == equivalent_entry_index
            && first_entry_index != distinct_entry_index,
        "equivalent SPIR-V aliases one cache entry while distinct SPIR-V does not");
    assert(shared_shader_id
            == state.pipeline_cache.entries[equivalent_entry_index]
                .key.vertex_shader
            && shared_shader_id != distinct_shader_id,
        "equivalent SPIR-V preserves one stable shader ID");
    assert(vk::shader_store_live_count(state) == 2
            && state.shader_store.entries[(uint)shared_shader_id].refcount == 1
            && state.shader_store.entries[(uint)distinct_shader_id].refcount == 1,
        "two cache entries own two exact shader identities");
    assert(state.pipeline_cache.entries[first_entry_index].pipeline
            != state.pipeline_cache.entries[distinct_entry_index].pipeline,
        "distinct identities retain distinct native pipelines");

    foreach (pipeline : pipelines) gpu::destroy_pipeline(&device, pipeline)!!;
    assert(vk::shader_store_live_count(state) == 0,
        "destroying all aliases releases both shader identities");
    assert(vk::shader_store_index_consistent(state),
        "repeated compute creation leaves the shader index consistent");
}

fn void malformed_compute_shader_rejected_before_store_mutation() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    uint[5] invalid_words = { 0, 0x00010000, 0, 1, 0 };
    gpu::ShaderDesc invalid = {
        .spirv = ((char*)&invalid_words[0])[:uint[5]::size],
        .debug_name = "invalid_compute_shader",
    };
    gpu::ComputePipelineDesc desc = { .shader = invalid };
    vk::VkDeviceState* state = cache_test_state(&device);
    PipelineShaderDiagnosticCapture capture = {
        .expected_operation  = "create_compute_pipeline",
        .expected_debug_name = "invalid_compute_shader",
        .expected_field      = "ComputePipelineDesc.shader",
        .expected_invariant  = "SPIR-V magic must be valid",
        .matched             = true,
    };
    state.debug_callback = &capture_pipeline_shader_diagnostic;
    state.debug_user_data = &capture;

    fault shader_fault = @catch(gpu::create_compute_pipeline(&device, &desc));

    state.debug_callback = null;
    state.debug_user_data = null;
    assert(shader_fault == gpu::SHADER_INVALID,
        "invalid compute shader preserves its validation fault");
    assert(capture.calls == 1 && capture.matched,
        "invalid compute shader emits one exact contract diagnostic");
    assert(cache_entry_count(&device) == 0,
        "compute shader validation publishes no cache entry");
    assert(vk::shader_store_live_count(state) == 0,
        "compute shader validation performs no interning");
}

fn void graphics_pipeline_shader_validation_reports_contract_failure() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();
    gpu::Format[1] pipeline_colors = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc desc =
        base_graphics_desc(&shaders, pipeline_colors[..]);
    desc.fragment_shader = {
        .debug_name = "invalid_graphics_fragment",
    };
    vk::VkDeviceState* state = cache_test_state(&device);
    PipelineShaderDiagnosticCapture capture = {
        .expected_operation  = "create_graphics_pipeline",
        .expected_debug_name = "invalid_graphics_fragment",
        .expected_field      = "GraphicsPipelineDesc.fragment_shader",
        .expected_invariant  = "SPIR-V must contain a word-aligned header",
        .matched             = true,
    };
    state.debug_callback = &capture_pipeline_shader_diagnostic;
    state.debug_user_data = &capture;
    uint next_new_before = state.shader_store.next_new;
    uint free_head_before = state.shader_store.free_head;

    fault shader_fault = @catch(gpu::create_graphics_pipeline(&device, &desc));

    state.debug_callback = null;
    state.debug_user_data = null;

    assert(shader_fault == gpu::SHADER_INVALID,
        "an invalid fragment shader preserves its validation fault");
    assert(capture.calls == 1 && capture.matched,
        "invalid fragment emits one exact contract diagnostic");
    assert(cache_entry_count(&device) == 0,
        "single graphics shader validation publishes no cache entry");
    assert(vk::shader_store_live_count(state) == 0,
        "single graphics shader validation publishes no shader identity");
    assert(state.shader_store.next_new == next_new_before
            && state.shader_store.free_head == free_head_before,
        "single graphics shader validation leaves shader-store allocation unchanged");
    assert(vk::shader_store_index_consistent(state),
        "single graphics shader validation preserves shader-store structure");
}

fn void graphics_fragment_failure_releases_vertex_preparation() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();
    gpu::ShaderDesc valid_fragment = shaders.frag;
    shaders.frag = {
        .spirv       = CACHE_COMP_SPIRV[..],
        .entry_point = "main",
    };
    gpu::Format[1] colors = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc desc =
        base_graphics_desc(&shaders, colors[..]);
    vk::VkDeviceState* state = cache_test_state(&device);

    fault fragment_fault =
        @catch(gpu::create_graphics_pipeline(&device, &desc));

    assert(fragment_fault == gpu::SHADER_INVALID,
        "a compute fragment faults SHADER_INVALID");
    assert(vk::shader_store_live_count(state) == 0,
        "fragment failure releases both prepared shader identities");
    assert(cache_entry_count(&device) == 0,
        "fragment failure publishes no pipeline cache entry");

    desc.fragment_shader = valid_fragment;
    gpu::PipelineHandle pipeline =
        gpu::create_graphics_pipeline(&device, &desc)!!;
    gpu::destroy_pipeline(&device, pipeline)!!;
    assert(vk::shader_store_live_count(state) == 0,
        "the corrected retry releases both identities on destroy");
}

fn void graphics_native_creation_failure_releases_prepared_identities() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = cache_test_state(&device);
    CacheShaders shaders = make_cache_shaders();
    gpu::Format[2] color_formats = {
        gpu::Format.RGBA8_UNORM,
        gpu::Format.BGRA8_UNORM,
    };
    gpu::GraphicsPipelineDesc desc = {
        .vertex_shader   = shaders.vert,
        .fragment_shader = shaders.frag,
        .color_formats   = color_formats[..],
    };
    vk::Result create_result = vk::Result.ERROR_OUT_OF_DEVICE_MEMORY;
    vk::GraphicsPipelineCreateOps create_ops =
        vk::graphics_pipeline_create_ops_real();
    create_ops.user_data = &create_result;
    create_ops.create = &return_graphics_pipeline_create_result;
    fault create_fault = @catch(vk::vk_create_graphics_pipeline_with_ops(
        state:      state,
        desc:       &desc,
        create_ops: &create_ops,
    ));

    assert(create_fault == gpu::OUT_OF_DEVICE_MEMORY,
        "the injected native failure reaches graphics pipeline creation");
    assert(cache_entry_count(&device) == 0,
        "failed native creation publishes no pipeline cache entry");
    assert(vk::shader_store_live_count(state) == 0,
        "failed native dynamic creation releases prepared shader identities");
}

fn void out_of_range_polygon_mode_rejects_before_backend_work() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

    gpu::Format[1] pipeline_colors = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, pipeline_colors[..]);
    int out_of_range = 99;
    mem::copy(&desc.polygon_mode, &out_of_range, int::size);

    fault rejected = @catch(gpu::create_graphics_pipeline(&device, &desc));
    assert(rejected == gpu::INVALID_ARGUMENT,
        "an out-of-range polygon mode faults INVALID_ARGUMENT: %s",
        rejected);
    assert(cache_entry_count(&device) == 0,
        "immutable state rejection publishes no cache entry");
}

fn void depth_format_in_color_formats_rejects_before_backend_work() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();

    gpu::Format[1] depth_colors = {
        gpu::Format.D32_FLOAT,
    };
    gpu::GraphicsPipelineDesc desc = base_graphics_desc(&shaders, depth_colors[..]);

    fault rejected = @catch(gpu::create_graphics_pipeline(&device, &desc));
    assert(rejected == gpu::INVALID_ARGUMENT,
        "a depth format in color_formats faults INVALID_ARGUMENT: %s",
        rejected);
    assert(cache_entry_count(&device) == 0,
        "color format rejection publishes no cache entry");
}

fn void graphics_color_formats_require_host_safe_storage() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state = cache_test_state(&device);
    gpu::Format* missing = null;
    gpu::Format[] forged = missing[:1];
    gpu::GraphicsPipelineDesc desc = { .color_formats = forged };

    fault single_fault = @catch(vk::vk_create_graphics_pipeline(
        state,
        &desc,
    ));
    assert(single_fault == gpu::INVALID_ARGUMENT,
        "a forged nonempty null color slice faults before iteration: %s",
        single_fault);
    assert(cache_entry_count(&device) == 0,
        "single rejection publishes no cache entry");
}

fn void repeated_format_only_graphics_calls_alias_one_pipeline_identity() @test {
    gpu::Device device = make_cache_test_device();
    defer destroy_test_device(&device)!!;
    CacheShaders shaders = make_cache_shaders();
    gpu::Format[1] color_formats = {
        gpu::Format.RGBA8_UNORM,
    };
    gpu::GraphicsPipelineDesc desc =
        base_graphics_desc(&shaders, color_formats[..]);
    gpu::GraphicsPipelineDesc[5] descs;
    foreach (&item : descs) *item = desc;
    gpu::PipelineHandle[5] pipelines;

    foreach (i, &item : descs) {
        pipelines[i] = gpu::create_graphics_pipeline(&device, item)!!;
    }

    foreach (i, pipeline : pipelines) {
        assert(pipeline.is_valid(), "every singular graphics output is valid");
        if (i > 0) {
            assert(pipeline != pipelines[0],
                "singular graphics aliases publish distinct owning handles");
        }
    }
    assert(cache_entry_count(&device) == 1,
        "all aliases retain one compact format-only cache entry");

    gpu::Queue queue = gpu::get_queue(
        &device,
        gpu::QueueKind.GRAPHICS,
    )!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        &device,
        queue,
    )!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::ColorWriteMask red_green = { .red, .green };
    gpu::ColorTargetState[5] packets = {
        gpu::color_blend_disabled(),
        gpu::alpha_blend(),
        gpu::premultiplied_alpha_blend(),
        gpu::additive_blend(),
        gpu::color_blend_disabled(red_green),
    };
    gpu::GraphicsState graphics =
        gpu::render_geometry_state(16, 16)!!;
    foreach (i, pipeline : pipelines) {
        gpu::cmd_bind_pipeline(&commands, pipeline)!!;
        gpu::ColorTargetState[1] targets = { packets[i] };
        graphics.color.targets = targets[..];
        gpu::cmd_set_graphics_state(&commands, &graphics)!!;
    }
    gpu::discard_commands(&commands)!!;

    foreach (pipeline : pipelines) {
        gpu::destroy_pipeline(&device, pipeline)!!;
    }
}
