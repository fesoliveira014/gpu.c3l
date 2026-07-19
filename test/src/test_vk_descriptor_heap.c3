module gpu_test;

import gpu @public;
import gpu::vk @public;
import vk;
import std::core::mem::allocators;

// Headless (lavapipe): descriptor heap slot bookkeeping through the public API.

const uint DESCRIPTOR_BUFFER_USAGE = (uint)vk::BufferUsageFlagBits.RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT
    | (uint)vk::BufferUsageFlagBits.SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT
    | (uint)vk::BufferUsageFlagBits.SHADER_DEVICE_ADDRESS_BIT;

struct DescriptorDiagnosticCapture {
    uint calls;
    fault expected_fault;
    gpu::DebugResourceKind expected_kind;
    bool expected_identity;
    uint expected_index;
    uint expected_generation;
    ZString expected_operation;
    ZString expected_field;
    ZString expected_invariant;
    bool matched;
}

struct DescriptorBackendCapture {
    uint calls;
    fault expected_fault;
    gpu::DebugResourceKind expected_kind;
    bool expected_identity;
    uint expected_index;
    uint expected_generation;
    ZString expected_operation;
    ZString expected_text;
    int expected_result;
    bool matched;
}

fn void capture_descriptor_backend_diagnostic(gpu::DebugMessage* message, void* user_data) {
    DescriptorBackendCapture* capture = (DescriptorBackendCapture*)user_data;
    capture.calls++;
    capture.matched = message.severity == gpu::DebugMessageSeverity.ERROR
        && message.category.backend
        && message.has_fault
        && message.public_fault == capture.expected_fault
        && message.resource.kind == capture.expected_kind
        && message.resource.has_public_identity == capture.expected_identity
        && (!capture.expected_identity || (message.resource.index == capture.expected_index
            && message.resource.generation == capture.expected_generation))
        && message.operation != null
        && message.operation.str_view() == capture.expected_operation.str_view()
        && message.backend_text != null
        && message.backend_text.str_view() == capture.expected_text.str_view()
        && message.validation_id_number == capture.expected_result;
}

fn void capture_descriptor_diagnostic(gpu::DebugMessage* message, void* user_data) {
    DescriptorDiagnosticCapture* capture = (DescriptorDiagnosticCapture*)user_data;
    capture.calls++;
    capture.matched = message.severity == gpu::DebugMessageSeverity.ERROR
        && message.category.public_contract
        && message.has_fault
        && message.public_fault == capture.expected_fault
        && message.resource.kind == capture.expected_kind
        && message.resource.has_public_identity == capture.expected_identity
        && (!capture.expected_identity || (message.resource.index == capture.expected_index
            && message.resource.generation == capture.expected_generation))
        && message.operation != null
        && message.operation.str_view() == capture.expected_operation.str_view()
        && message.rejected_field != null
        && message.rejected_field.str_view() == capture.expected_field.str_view()
        && message.invariant != null
        && message.invariant.str_view() == capture.expected_invariant.str_view();
}

fn void expect_descriptor_diagnostic(
    DescriptorDiagnosticCapture* capture,
    fault expected_fault,
    gpu::DebugResourceKind expected_kind,
    bool expected_identity,
    uint expected_index,
    uint expected_generation,
    ZString expected_operation,
    ZString expected_field,
    ZString expected_invariant,
) {
    *capture = {
        .calls = capture.calls,
        .expected_fault = expected_fault,
        .expected_kind = expected_kind,
        .expected_identity = expected_identity,
        .expected_index = expected_index,
        .expected_generation = expected_generation,
        .expected_operation = expected_operation,
        .expected_field = expected_field,
        .expected_invariant = expected_invariant,
    };
}

fn void descriptor_buffer_create_info_uses_exclusive_for_one_family() @test {
    uint[1] family_storage = { 3 };

    vk::BufferCreateInfo info = gpu::vk::make_descriptor_buffer_create_info(
        4096,
        family_storage[..],
    );

    assert(info.size == 4096, "descriptor-buffer layout size is preserved");
    assert((uint)info.usage == DESCRIPTOR_BUFFER_USAGE, "descriptor-buffer usage remains exact");
    assert(info.sharing_mode == vk::SharingMode.EXCLUSIVE, "one consumer family remains exclusive");
    assert(info.queue_family_index_count == 0, "exclusive creation publishes no family count");
    assert(info.p_queue_family_indices == null, "exclusive creation publishes no family pointer");
}

fn void descriptor_buffer_create_info_uses_only_graphics_and_compute_families() @test {
    uint[2] family_storage = { 5, 2 };

    vk::BufferCreateInfo info = gpu::vk::make_descriptor_buffer_create_info(
        8192,
        family_storage[..],
    );

    assert(info.size == 8192, "descriptor-buffer layout size is preserved");
    assert((uint)info.usage == DESCRIPTOR_BUFFER_USAGE, "descriptor-buffer usage remains exact");
    assert(info.sharing_mode == vk::SharingMode.CONCURRENT, "distinct consumers use concurrent sharing");
    assert(info.queue_family_index_count == 2, "concurrent creation publishes exactly two families");
    assert(info.p_queue_family_indices == &family_storage[0], "create info points into caller-owned storage");
    assert(info.p_queue_family_indices[0] == 5, "graphics family remains first");
    assert(info.p_queue_family_indices[1] == 2, "compute family remains second");
}

fn void disabled_descriptor_heap_bind_returns_unsupported_feature() @test {
    gpu::vk::VkDeviceState state = {};
    fault bind_fault = @catch(gpu::vk::bind_descriptor_heap(
        state:      &state,
        cb:         (vk::CommandBuffer)null,
        layout:     (vk::PipelineLayout)null,
        bind_point: vk::PipelineBindPoint.COMPUTE,
    ));

    assert(bind_fault == gpu::UNSUPPORTED_FEATURE, "disabled descriptor heaps cannot be bound");
}

fn void disabled_heap_preflights_pipeline_binds() @test {
    gpu::vk::VkDeviceState state = {};
    gpu::PipelineHandle cached = { .index = 7, .generation = 3 };
    gpu::PipelineHandle requested = { .index = 11, .generation = 5 };
    gpu::vk::CommandRecord record = {
        .command_buffer = (vk::CommandBuffer)null,
        .last_pipeline = cached,
        .last_entry = 19,
    };

    fault compute_fault = @catch(gpu::vk::bind_compute_and_push(
        state:  &state,
        record: &record,
        handle: requested,
        root:   (gpu::GpuAddress)0,
    ));
    assert(compute_fault == gpu::UNSUPPORTED_FEATURE, "compute binding preflights a disabled heap");
    assert(record.last_pipeline == cached, "rejected compute binding preserves the cached pipeline");
    assert(record.last_entry == 19, "rejected compute binding preserves the cached pipeline entry");

    fault graphics_fault = @catch(gpu::vk::bind_graphics_and_push(
        state:         &state,
        record:        &record,
        handle:        requested,
        vertex_root:   (gpu::GpuAddress)0,
        fragment_root: (gpu::GpuAddress)0,
    ));
    assert(graphics_fault == gpu::UNSUPPORTED_FEATURE, "graphics binding preflights a disabled heap");
    assert(record.last_pipeline == cached, "rejected graphics binding preserves the cached pipeline");
    assert(record.last_entry == 19, "rejected graphics binding preserves the cached pipeline entry");
}

fn gpu::Device make_heap_test_device(uint tex_capacity, uint smp_capacity) {
    gpu::DeviceDesc desc = {
        .backend               = gpu::BackendKind.VULKAN,
        .enable_validation     = true,
        .texture_heap_capacity = tex_capacity,
        .sampler_heap_capacity = smp_capacity,
        .application_name      = "gpu_descriptor_heap_test",
    };
    return gpu::create_device_from_desc(&desc)!!;
}

fn gpu::TextureHandle make_heap_texture(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn gpu::TextureHandle make_heap_view_texture(gpu::Device* device) {
    gpu::TextureDesc desc = {
        .dimension    = gpu::TextureDimension.TEX_2D,
        .access    = { .graphics, .compute, .transfer },
        .width        = 16,
        .height       = 16,
        .mip_levels   = 5,
        .array_layers = 4,
        .format       = gpu::Format.RGBA8_UNORM,
        .usage        = { .sampled, .storage },
    };
    return gpu::create_texture(device, &desc)!!;
}

fn void descriptor_sampler_failures_report_context() @test {
    gpu::Device device = make_heap_test_device(1, 1);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureHandle transfer = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, transfer)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    state.textures.slots[transfer.index].value.usage = { .transfer_dst };
    DescriptorDiagnosticCapture capture;
    state.debug_callback = &capture_descriptor_diagnostic;
    state.debug_user_data = &capture;

    gpu::TextureView descriptor = gpu::create_texture_view(&device, texture, null)!!;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::DESCRIPTOR_HEAP_FULL,
        expected_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_texture_view",
        expected_field:      "DeviceDesc.texture_heap_capacity",
        expected_invariant:  "must exceed live texture descriptors",
    );
    fault full = @catch(gpu::create_texture_view(&device, texture, null));
    assert(full == gpu::DESCRIPTOR_HEAP_FULL && capture.calls == 1 && capture.matched,
        "descriptor capacity fault and diagnostic are exact");

    gpu::destroy_texture_view(&device, descriptor)!!;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_HANDLE,
        expected_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "destroy_texture_view",
        expected_field:      "view",
        expected_invariant:  "view must identify a live generation",
    );
    fault stale = @catch(gpu::destroy_texture_view(&device, descriptor));
    assert(stale == gpu::INVALID_HANDLE && capture.calls == 2 && capture.matched,
        "stale descriptor fault has kind without unvalidated identity");

    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        expected_kind:       gpu::DebugResourceKind.TEXTURE,
        expected_identity:   true,
        expected_index:      transfer.index,
        expected_generation: transfer.generation,
        expected_operation:  "create_texture_view",
        expected_field:      "TextureDesc.usage",
        expected_invariant:  "texture must have sampled or storage usage",
    );
    fault usage = @catch(gpu::create_texture_view(&device, transfer, null));
    assert(usage == gpu::INVALID_ARGUMENT && capture.calls == 3 && capture.matched,
        "descriptor usage fault and texture identity are exact");

    gpu::SamplerDesc sampler_desc = {};
    gpu::Sampler sampler = gpu::intern_sampler(&device, &sampler_desc)!!;
    gpu::SamplerIndex sampler_index = gpu::publish_sampler(&device, sampler)!!;
    gpu::SamplerDesc other_desc = { .mag_filter = gpu::Filter.LINEAR };
    gpu::Sampler other = gpu::intern_sampler(&device, &other_desc)!!;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::DESCRIPTOR_HEAP_FULL,
        expected_kind:       gpu::DebugResourceKind.SAMPLER,
        expected_identity:   true,
        expected_index:      other.index,
        expected_generation: other.generation,
        expected_operation:  "publish_sampler",
        expected_field:      "DeviceDesc.sampler_heap_capacity",
        expected_invariant:  "must exceed published samplers",
    );
    fault sampler_full = @catch(gpu::publish_sampler(&device, other));
    assert(sampler_full == gpu::DESCRIPTOR_HEAP_FULL
            && capture.calls == 4 && capture.matched,
        "sampler publication capacity fault and diagnostic are exact");

    assert(gpu::publish_sampler(&device, sampler)!! == sampler_index,
        "an existing sampler publication remains available when the heap is full");
    state.debug_callback = null;
    fault fallback = @catch(gpu::publish_sampler(&device, other));
    assert(fallback == sampler_full && capture.calls == 4,
        "null callback preserves fault parity without a structured duplicate");}

fn uint cached_texture_view_count(vk::VkDeviceState* state, gpu::TextureHandle texture) {
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    uint count = 0;
    foreach (&cached : slot.views) if (cached.used) count++;
    return count;
}

fn void texture_view_round_trip() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureView view = gpu::create_texture_view(&device, texture, null)!!;
    assert(view.is_valid() && view.index.is_valid(),
        "created view and its raw shader index are valid");
    gpu::destroy_texture_view(&device, view)!!;

    fault stale_fault = @catch(gpu::destroy_texture_view(&device, view));
    assert(stale_fault == gpu::INVALID_HANDLE,
        "destroying a stale view faults INVALID_HANDLE");
}

fn void texture_view_rejects_foreign_device_before_heap_mutation() @test {
    gpu::Device first_device = make_heap_test_device(2, 1);
    gpu::Device second_device = make_heap_test_device(2, 1);
    gpu::TextureHandle first_texture = make_heap_texture(&first_device);
    gpu::TextureHandle second_texture = make_heap_texture(&second_device);
    gpu::TextureView first_view =
        gpu::create_texture_view(&first_device, first_texture, null)!!;
    gpu::TextureView second_view =
        gpu::create_texture_view(&second_device, second_texture, null)!!;

    assert(first_view.owner != second_view.owner,
        "texture-view ownership differs across devices");
    assert(first_view.index == second_view.index,
        "coincident devices may publish the same raw shader index");
    vk::VkDeviceState* first_state =
        (vk::VkDeviceState*)gpu::device_backend_state(first_device);
    vk::VkDeviceState* second_state =
        (vk::VkDeviceState*)gpu::device_backend_state(second_device);
    uint first_free_before = first_state.strict_heap.descriptors.texture_free_count;
    uint second_free_before = second_state.strict_heap.descriptors.texture_free_count;

    fault foreign = @catch(gpu::destroy_texture_view(&second_device, first_view));
    assert(foreign == gpu::INVALID_HANDLE,
        "a foreign texture view faults INVALID_HANDLE");
    assert(first_state.strict_heap.descriptors.texture_free_count == first_free_before
        && second_state.strict_heap.descriptors.texture_free_count == second_free_before,
        "foreign release stops before either heap mutates");
    assert(first_state.strict_heap.descriptors.texture_cells[first_view.index.value - 1].used
        && second_state.strict_heap.descriptors.texture_cells[second_view.index.value - 1].used,
        "both coincident live cells survive the rejected release");

    gpu::destroy_texture_view(&first_device, first_view)!!;
    gpu::destroy_texture_view(&second_device, second_view)!!;
    gpu::destroy_texture(&first_device, first_texture)!!;
    gpu::destroy_texture(&second_device, second_texture)!!;
    destroy_test_device(&first_device)!!;
    destroy_test_device(&second_device)!!;
}

fn void texture_view_exhaustion() @test {
    gpu::Device device = make_heap_test_device(2, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureView a = gpu::create_texture_view(&device, texture, null)!!;
    gpu::TextureView b = gpu::create_texture_view(&device, texture, null)!!;

    fault full_fault = @catch(gpu::create_texture_view(&device, texture, null));
    assert(full_fault == gpu::DESCRIPTOR_HEAP_FULL, "allocating past capacity faults DESCRIPTOR_HEAP_FULL");
    assert(a.is_valid() && b.is_valid(), "existing indices stay valid after exhaustion");

    gpu::destroy_texture_view(&device, a)!!;
    gpu::destroy_texture_view(&device, b)!!;
}

fn void texture_view_exhaustion_does_not_cache_failed_views() @test {
    gpu::Device device = make_heap_test_device(1, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureView occupied = gpu::create_texture_view(&device, texture, null)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(cached_texture_view_count(state, texture) == 0, "the fixture starts with an empty view cache");

    for (uint i = 0; i < vk::INITIAL_CACHED_VIEW_CAPACITY; i++) {
        gpu::TextureViewDesc view = {
            .base_mip    = i % 4,
            .mip_count   = 1,
            .base_layer  = i / 4,
            .layer_count = 1,
        };
        fault full = @catch(gpu::create_texture_view(&device, texture, &view));
        assert(full == gpu::DESCRIPTOR_HEAP_FULL, "an exhausted heap rejects the descriptor");
        assert(cached_texture_view_count(state, texture) == 0,
            "a rejected descriptor leaves the texture view cache unchanged");
    }

    gpu::destroy_texture_view(&device, occupied)!!;
}

fn void texture_view_reuses_destroyed_slot_immediately() @test {
    gpu::Device device = make_heap_test_device(1, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureView first =
        gpu::create_texture_view(&device, texture, null)!!;
    gpu::destroy_texture_view(&device, first)!!;

    gpu::TextureView recycled =
        gpu::create_texture_view(&device, texture, null)!!;
    assert(recycled.index.value == first.index.value,
        "destroyed descriptor slots are immediately reusable");
    assert(recycled.generation != first.generation,
        "immediate reuse advances the generation");
    assert(@catch(gpu::destroy_texture_view(&device, first))
        == gpu::INVALID_HANDLE, "the released index is stale");

    gpu::destroy_texture_view(&device, recycled)!!;
}

fn void texture_view_generation_exhaustion_retires_slot() @test {
    gpu::Device device = make_heap_test_device(1, 1);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureView view = gpu::create_texture_view(&device, texture, null)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::TextureDescriptorCell* cell =
        &state.strict_heap.descriptors.texture_cells[view.index.value - 1];
    cell.generation = vk::HANDLE_GENERATION_MAX - 1;
    view.generation = cell.generation;

    gpu::destroy_texture_view(&device, view)!!;

    assert(cell.generation == vk::HANDLE_GENERATION_MAX,
        "an exhausted texture-view generation retires at the sentinel");
    assert(state.strict_heap.descriptors.texture_free_count == 0,
        "an exhausted texture-view slot never returns to the free list");
    assert(@catch(gpu::create_texture_view(&device, texture, null))
        == gpu::DESCRIPTOR_HEAP_FULL,
        "a retired only slot cannot recycle to a stale generation");
}

fn void texture_view_stale_texture_rejected() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, texture)!!;

    fault texture_fault = @catch(gpu::create_texture_view(&device, texture, null));
    assert(texture_fault == gpu::INVALID_HANDLE, "registering a destroyed texture faults INVALID_HANDLE");
}

fn void texture_view_bad_view_rejected() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureViewDesc reinterpret = { .format = gpu::Format.R32_UINT };
    fault view_fault = @catch(gpu::create_texture_view(&device, texture, &reinterpret));
    assert(view_fault == gpu::INVALID_ARGUMENT, "format-reinterpreting views fault INVALID_ARGUMENT");
}

fn void texture_view_overflowing_view_range_rejected() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureViewDesc mip_overflow = { .base_mip = 1, .mip_count = uint::max };
    fault mip_fault = @catch(gpu::create_texture_view(&device, texture, &mip_overflow));
    assert(mip_fault == gpu::INVALID_ARGUMENT, "overflowing mip range faults INVALID_ARGUMENT");

    gpu::TextureViewDesc layer_overflow = { .base_layer = 1, .layer_count = uint::max };
    fault layer_fault = @catch(gpu::create_texture_view(&device, texture, &layer_overflow));
    assert(layer_fault == gpu::INVALID_ARGUMENT, "overflowing layer range faults INVALID_ARGUMENT");
}

fn void destroy_texture_with_live_descriptor_faults_and_recovers() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);

    gpu::TextureView index = gpu::create_texture_view(&device, texture, null)!!;

    fault in_use_fault = @catch(gpu::destroy_texture(&device, texture));
    assert(in_use_fault == gpu::RESOURCE_IN_USE, "destroying a texture with a live descriptor faults RESOURCE_IN_USE");

    gpu::TextureView second = gpu::create_texture_view(&device, texture, null)!!;
    assert(second.is_valid(), "handle stays valid and usable after the faulted destroy");
    gpu::destroy_texture_view(&device, second)!!;

    gpu::destroy_texture_view(&device, index)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_after_descriptor_release_succeeds() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);

    gpu::TextureView index = gpu::create_texture_view(&device, texture, null)!!;
    gpu::destroy_texture_view(&device, index)!!;

    gpu::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_blocks_while_any_descriptor_live() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);

    gpu::TextureView a = gpu::create_texture_view(&device, texture, null)!!;
    gpu::TextureView b = gpu::create_texture_view(&device, texture, null)!!;

    gpu::destroy_texture_view(&device, a)!!;
    fault still_in_use = @catch(gpu::destroy_texture(&device, texture));
    assert(still_in_use == gpu::RESOURCE_IN_USE, "one remaining live descriptor still blocks destroy_texture");

    gpu::destroy_texture_view(&device, b)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_unaffected_by_unrelated_descriptor() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture_a = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture_a)!!;
    gpu::TextureHandle texture_b = make_heap_texture(&device);

    gpu::TextureView index = gpu::create_texture_view(&device, texture_a, null)!!;
    defer gpu::destroy_texture_view(&device, index)!!;

    gpu::destroy_texture(&device, texture_b)!!;
}

fn void texture_views_batch_happy_path() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureViewCreateDesc[3] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureView[3] indices;
    gpu::create_texture_views(&device, descs[..], indices[..])!!;

    foreach (index : indices) assert(index.is_valid(), "every batch-created index is valid");
    assert(indices[0].index.value != indices[1].index.value && indices[1].index.value != indices[2].index.value && indices[0].index.value != indices[2].index.value,
        "batch-created indices occupy distinct slots");

    foreach (index : indices) gpu::destroy_texture_view(&device, index)!!;
}

fn void texture_views_batch_fills_capacity_exactly() @test {
    gpu::Device device = make_heap_test_device(3, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureViewCreateDesc[3] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureView[3] indices;
    gpu::create_texture_views(&device, descs[..], indices[..])!!;

    fault full_fault = @catch(gpu::create_texture_view(&device, texture, null));
    assert(full_fault == gpu::DESCRIPTOR_HEAP_FULL, "a batch that exactly fills capacity leaves none spare — the heap count rose by exactly N");

    foreach (index : indices) gpu::destroy_texture_view(&device, index)!!;
}

fn void texture_views_batch_zeroed_view_is_default() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureViewCreateDesc[1] descs = { { .texture = texture } };
    gpu::TextureView[1] indices;
    gpu::create_texture_views(&device, descs[..], indices[..])!!;
    assert(indices[0].is_valid(), "a zero-initialized view collapses to the default view and creates successfully");

    gpu::destroy_texture_view(&device, indices[0])!!;
}

fn void texture_views_mismatched_out_len_faults() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureViewCreateDesc[2] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureView[1] indices;
    fault mismatch_fault = @catch(gpu::create_texture_views(&device, descs[..], indices[..]));
    assert(mismatch_fault == gpu::INVALID_ARGUMENT, "a mismatched out-indices length faults INVALID_ARGUMENT");
}

fn void texture_views_empty_batch_is_no_op() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;

    gpu::TextureViewCreateDesc[] descs;
    gpu::TextureView[] indices;
    gpu::create_texture_views(&device, descs, indices)!!;
}

fn void texture_views_batch_preflight_faults_untouched() @test {
    gpu::Device device = make_heap_test_device(4, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureView existing = gpu::create_texture_view(&device, texture, null)!!;

    // Capacity 4, one slot already used: 3 remain; the batch asks for 4.
    gpu::TextureViewCreateDesc[4] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureView[4] indices;
    fault full_fault = @catch(gpu::create_texture_views(&device, descs[..], indices[..]));
    assert(full_fault == gpu::DESCRIPTOR_HEAP_FULL, "a batch exceeding remaining capacity faults DESCRIPTOR_HEAP_FULL");
    assert(existing.is_valid(), "the pre-existing index survives the failed pre-flight");

    // Untouched: the failed pre-flight consumed nothing, so the remaining
    // headroom (3 slots) is still there to fill exactly.
    gpu::TextureView a = gpu::create_texture_view(&device, texture, null)!!;
    gpu::TextureView b = gpu::create_texture_view(&device, texture, null)!!;
    gpu::TextureView c = gpu::create_texture_view(&device, texture, null)!!;
    assert(a.is_valid() && b.is_valid() && c.is_valid(), "the failed batch left the full remaining headroom available");

    gpu::destroy_texture_view(&device, existing)!!;
    gpu::destroy_texture_view(&device, a)!!;
    gpu::destroy_texture_view(&device, b)!!;
    gpu::destroy_texture_view(&device, c)!!;
}

fn void texture_views_failed_batch_restores_full_view_capacity() @test {
    gpu::Device device = make_heap_test_device(32, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);

    for (uint i = 0; i < vk::INITIAL_CACHED_VIEW_CAPACITY; i++) {
        gpu::TextureViewDesc failed_view = {
            .base_mip    = i % 4,
            .mip_count   = 1,
            .base_layer  = i / 4,
            .layer_count = 1,
        };
        gpu::TextureViewCreateDesc[2] descs = {
            { .texture = texture, .view = failed_view },
            { .texture = stale_texture },
        };
        gpu::TextureView[2] indices;
        fault batch_fault = @catch(gpu::create_texture_views(&device, descs[..], indices[..]));
        assert(batch_fault == gpu::INVALID_HANDLE, "a later stale texture faults the batch");
        assert(cached_texture_view_count(state, texture) == 0,
            "a failed batch restores the empty view cache");
    }

    gpu::TextureView[vk::INITIAL_CACHED_VIEW_CAPACITY] live;
    for (uint i = 0; i < vk::INITIAL_CACHED_VIEW_CAPACITY; i++) {
        gpu::TextureViewDesc valid_view = {
            .base_mip    = i % 4,
            .mip_count   = 1,
            .base_layer  = 2 + i / 4,
            .layer_count = 1,
        };
        live[i] = gpu::create_texture_view(&device, texture, &valid_view)!!;
    }
    assert(cached_texture_view_count(state, texture) == vk::INITIAL_CACHED_VIEW_CAPACITY,
        "all cache entries remain available after repeated failed batches");

    foreach (index : live) gpu::destroy_texture_view(&device, index)!!;
}

fn void texture_views_failed_batch_restores_multiple_created_views() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    DescriptorDiagnosticCapture capture;
    state.debug_callback = &capture_descriptor_diagnostic;
    state.debug_user_data = &capture;

    gpu::TextureViewDesc first_view = {
        .base_mip    = 0,
        .mip_count   = 1,
        .base_layer  = 0,
        .layer_count = 1,
    };
    gpu::TextureViewDesc second_view = {
        .base_mip    = 1,
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };
    gpu::TextureViewCreateDesc[3] descs = {
        { .texture = texture, .view = first_view },
        { .texture = texture, .view = second_view },
        { .texture = stale_texture },
    };
    gpu::TextureView[3] indices;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_HANDLE,
        expected_kind:       gpu::DebugResourceKind.TEXTURE,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_texture_views",
        expected_field:      "descs[].texture",
        expected_invariant:  "texture must identify a live generation",
    );
    fault batch_fault = @catch(gpu::create_texture_views(&device, descs[..], indices[..]));
    assert(batch_fault == gpu::INVALID_HANDLE && capture.calls == 1 && capture.matched,
        "a later stale texture emits once after two new views");
    assert(cached_texture_view_count(state, texture) == 0,
        "rollback removes every view created by the failed transaction");
    assert(slot.views.len == 0,
        "rollback restores the empty pre-transaction cache");
    state.debug_callback = null;
}

fn void texture_views_failed_batch_preserves_existing_and_duplicate_views() @test {
    gpu::Device device = make_heap_test_device(16, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;

    gpu::TextureViewDesc existing_view = {
        .base_mip    = 0,
        .mip_count   = 1,
        .base_layer  = 0,
        .layer_count = 1,
    };
    gpu::TextureView existing_index = gpu::create_texture_view(&device, texture, &existing_view)!!;
    vk::CachedView existing_cache = slot.views[0];

    gpu::TextureViewDesc new_view = {
        .base_mip    = 1,
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };
    gpu::TextureViewCreateDesc[4] descs = {
        { .texture = texture, .view = existing_view },
        { .texture = texture, .view = new_view },
        { .texture = texture, .view = new_view },
        { .texture = stale_texture },
    };
    gpu::TextureView[4] indices;
    fault batch_fault = @catch(gpu::create_texture_views(&device, descs[..], indices[..]));
    assert(batch_fault == gpu::INVALID_HANDLE, "a fault after a duplicate new view rejects the batch");
    assert(cached_texture_view_count(state, texture) == 1,
        "rollback removes the one transaction-created view and preserves the existing view");
    assert(slot.views[0].used && slot.views[0].view == existing_cache.view,
        "the pre-existing cached Vulkan view remains owned by the texture");
    assert(slot.views[0].base_mip == existing_cache.base_mip
        && slot.views[0].mip_count == existing_cache.mip_count
        && slot.views[0].base_layer == existing_cache.base_layer
        && slot.views[0].layer_count == existing_cache.layer_count,
        "the pre-existing cache entry remains unchanged");

    gpu::destroy_texture_view(&device, existing_index)!!;
}

fn void texture_views_mid_batch_fault_preserves_descriptor_state() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureView[7] occupied;
    for (uint i = 0; i < 7; i++) {
        occupied[i] = gpu::create_texture_view(&device, texture, null)!!;
    }
    gpu::TextureView released = occupied[0];
    gpu::destroy_texture_view(&device, released)!!;

    gpu::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::DescriptorHeapState* heap = &state.strict_heap.descriptors;

    vk::TextureDescriptorCell[8] cells_before;
    uint[8] free_before;
    for (uint i = 0; i < 8; i++) {
        cells_before[i] = heap.texture_cells[i];
        free_before[i] = heap.texture_free[i];
    }
    uint free_count_before = heap.texture_free_count;
    uint next_new_before = heap.texture_next_new;

    gpu::TextureViewDesc view = {
        .base_mip    = 1,
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };
    gpu::TextureViewCreateDesc[2] descs = {
        { .texture = texture, .view = view },
        { .texture = stale_texture },
    };
    gpu::TextureView[2] indices = {
        { .owner = 101, .index.value = 6, .generation = 111 },
        { .owner = 202, .index.value = 7, .generation = 222 },
    };
    gpu::TextureView[2] outputs_before = indices;
    fault batch_fault = @catch(gpu::create_texture_views(&device, descs[..], indices[..]));
    assert(batch_fault == gpu::INVALID_HANDLE, "a mid-batch destroyed-texture reference faults INVALID_HANDLE");

    assert(heap.texture_free_count == free_count_before, "failure preserves free-list count");
    assert(heap.texture_next_new == next_new_before, "failure preserves the never-used slot cursor");
    for (uint i = 0; i < 8; i++) {
        vk::TextureDescriptorCell before = cells_before[i];
        vk::TextureDescriptorCell after = heap.texture_cells[i];
        assert(after.used == before.used
            && after.generation == before.generation
            && after.owner.index == before.owner.index
            && after.owner.generation == before.owner.generation,
            "failure preserves every descriptor cell and generation");
        assert(heap.texture_free[i] == free_before[i], "failure preserves free-list ordering and contents");
    }
    assert(indices[0].owner == outputs_before[0].owner
        && indices[0].index == outputs_before[0].index
        && indices[0].generation == outputs_before[0].generation
        && indices[1].owner == outputs_before[1].owner
        && indices[1].index == outputs_before[1].index
        && indices[1].generation == outputs_before[1].generation,
        "failure leaves every output-view field untouched");

    gpu::TextureViewCreateDesc[2] valid = {
        { .texture = texture },
        { .texture = texture },
    };
    gpu::TextureView[2] committed;
    gpu::create_texture_views(&device, valid[..], committed[..])!!;
    assert(committed[0].is_valid() && committed[1].is_valid(),
        "a two-item batch commits using one free and one never-used slot");
    foreach (index : committed) gpu::destroy_texture_view(&device, index)!!;
    for (uint i = 1; i < 7; i++) {
        gpu::destroy_texture_view(&device, occupied[i])!!;
    }
}

fn void texture_views_batch_created_descriptor_blocks_texture_destroy() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);

    gpu::TextureViewCreateDesc[2] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::TextureView[2] indices;
    gpu::create_texture_views(&device, descs[..], indices[..])!!;

    fault in_use_fault = @catch(gpu::destroy_texture(&device, texture));
    assert(in_use_fault == gpu::RESOURCE_IN_USE, "a batch-created descriptor blocks destroy_texture the same as a single-created one");

    foreach (index : indices) gpu::destroy_texture_view(&device, index)!!;
    gpu::destroy_texture(&device, texture)!!;
}

fn void sampler_interning_and_publication_are_stable() @test {
    gpu::Device device = make_heap_test_device(8, 2);
    defer destroy_test_device(&device)!!;

    gpu::SamplerDesc desc = {
        .min_filter = gpu::Filter.NEAREST,
        .mag_filter = gpu::Filter.NEAREST,
        .address_u  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::Sampler first = gpu::intern_sampler(&device, &desc)!!;
    gpu::Sampler duplicate = gpu::intern_sampler(&device, &desc)!!;
    assert(first == duplicate, "identical sampler descriptions share identity");

    gpu::SamplerIndex first_index = gpu::publish_sampler(&device, first)!!;
    gpu::SamplerIndex duplicate_index = gpu::publish_sampler(&device, duplicate)!!;
    assert(first_index == duplicate_index,
        "repeated publication returns the stable sampler index");

    gpu::SamplerDesc second_desc = desc;
    second_desc.mag_filter = gpu::Filter.LINEAR;
    gpu::Sampler second = gpu::intern_sampler(&device, &second_desc)!!;
    gpu::SamplerIndex second_index = gpu::publish_sampler(&device, second)!!;
    assert(second_index.is_valid() && second_index != first_index,
        "a distinct sampler consumes the next publication slot");

    gpu::SamplerDesc third_desc = desc;
    third_desc.address_u = gpu::AddressMode.REPEAT;
    gpu::Sampler third = gpu::intern_sampler(&device, &third_desc)!!;
    fault full_fault = @catch(gpu::publish_sampler(&device, third));
    assert(full_fault == gpu::DESCRIPTOR_HEAP_FULL,
        "publishing past sampler capacity faults DESCRIPTOR_HEAP_FULL");
    assert(third.is_valid(), "publication exhaustion preserves sampler identity");
    assert(gpu::publish_sampler(&device, first)!! == first_index,
        "an existing publication succeeds after heap exhaustion");
    gpu::Sampler forged = first;
    forged.generation++;
    fault forged_fault = @catch(gpu::publish_sampler(&device, forged));
    assert(forged_fault == gpu::INVALID_HANDLE,
        "a manufactured sampler generation is rejected");
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(state.samplers.count == 3
            && state.strict_heap.descriptors.sampler_next_new == 2,
        "failed publication preserves identity and heap state");
}

fn void sampler_non_finite_and_reversed_ranges_rejected() @test {
    gpu::Device device = make_heap_test_device(8, 8);
    defer destroy_test_device(&device)!!;

    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    vk::PhysicalDeviceProperties properties;
    vk::get_physical_device_properties(state.physical_device, &properties);
    gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(caps.max_sampler_lod_bias == properties.limits.max_sampler_lod_bias,
        "caps expose the selected device mip bias limit");
    float bias_limit = caps.max_sampler_lod_bias;
    gpu::SamplerDesc positive_limit = { .mip_lod_bias = bias_limit };
    gpu::SamplerDesc negative_limit = { .mip_lod_bias = -bias_limit };
    assert(gpu::intern_sampler(&device, &positive_limit)!!.is_valid(),
        "the positive device mip bias limit is accepted");
    assert(gpu::intern_sampler(&device, &negative_limit)!!.is_valid(),
        "the negative device mip bias limit is accepted");

    gpu::SamplerDesc above_limit = { .mip_lod_bias = bias_limit + 1.0f };
    fault above_limit_fault = @catch(gpu::intern_sampler(&device, &above_limit));
    assert(above_limit_fault == gpu::INVALID_ARGUMENT,
        "mip bias above the device limit faults INVALID_ARGUMENT");

    gpu::SamplerDesc below_limit = { .mip_lod_bias = -bias_limit - 1.0f };
    fault below_limit_fault = @catch(gpu::intern_sampler(&device, &below_limit));
    assert(below_limit_fault == gpu::INVALID_ARGUMENT,
        "mip bias below the device limit faults INVALID_ARGUMENT");

    gpu::SamplerDesc nan_bias = { .mip_lod_bias = float::nan };
    fault bias_fault = @catch(gpu::intern_sampler(&device, &nan_bias));
    assert(bias_fault == gpu::INVALID_ARGUMENT, "non-finite mip bias faults INVALID_ARGUMENT");

    gpu::SamplerDesc nan_lod = { .min_lod = float::nan };
    fault lod_fault = @catch(gpu::intern_sampler(&device, &nan_lod));
    assert(lod_fault == gpu::INVALID_ARGUMENT, "non-finite LOD range faults INVALID_ARGUMENT");

    gpu::SamplerDesc nan_max_lod = { .max_lod = float::nan };
    fault max_lod_fault = @catch(gpu::intern_sampler(&device, &nan_max_lod));
    assert(max_lod_fault == gpu::INVALID_ARGUMENT, "non-finite maximum LOD faults INVALID_ARGUMENT");

    gpu::SamplerDesc reversed_lod = { .min_lod = 2.0f, .max_lod = 1.0f };
    fault reversed_fault = @catch(gpu::intern_sampler(&device, &reversed_lod));
    assert(reversed_fault == gpu::INVALID_ARGUMENT, "reversed LOD range faults INVALID_ARGUMENT");

    gpu::SamplerDesc nan_anisotropy = { .anisotropy_enable = true, .max_anisotropy = float::nan };
    fault anisotropy_fault = @catch(gpu::intern_sampler(&device, &nan_anisotropy));
    assert(anisotropy_fault == gpu::INVALID_ARGUMENT, "non-finite anisotropy faults INVALID_ARGUMENT");
}

fn void sampler_invalid_enum_values_are_rejected() @test {
    gpu::Device device = make_heap_test_device(8, 8);
    defer destroy_test_device(&device)!!;
    vk::VkDeviceState* state =
        (vk::VkDeviceState*)gpu::device_backend_state(device);
    uint count_before = state.samplers.count;

    gpu::SamplerDesc invalid_filter = {};
    *((int*)&invalid_filter.min_filter) = 99;
    assert(@catch(gpu::intern_sampler(&device, &invalid_filter))
        == gpu::INVALID_ARGUMENT, "invalid filter ordinal is rejected");

    gpu::SamplerDesc invalid_address = {};
    *((int*)&invalid_address.address_w) = -1;
    assert(@catch(gpu::intern_sampler(&device, &invalid_address))
        == gpu::INVALID_ARGUMENT, "invalid address-mode ordinal is rejected");

    gpu::SamplerDesc invalid_compare = { .compare_enable = true };
    *((int*)&invalid_compare.compare) = 99;
    assert(@catch(gpu::intern_sampler(&device, &invalid_compare))
        == gpu::INVALID_ARGUMENT, "invalid enabled compare ordinal is rejected");

    assert(state.samplers.count == count_before,
        "invalid enum values do not mutate the sampler table");
}

fn void live_descriptors_block_device_destruction() @test {
    gpu::Device device = make_heap_test_device(8, 4);
    gpu::TextureHandle texture = make_heap_texture(&device);
    gpu::TextureView descriptor = gpu::create_texture_view(&device, texture, null)!!;

    gpu::SamplerDesc desc = {};
    gpu::Sampler sampler = gpu::intern_sampler(&device, &desc)!!;
    gpu::SamplerIndex sampler_index = gpu::publish_sampler(&device, sampler)!!;

    assert(@catch(gpu::destroy_device(&device)) == gpu::RESOURCE_IN_USE,
        "live descriptors block device destruction");
    gpu::destroy_texture_view(&device, descriptor)!!;
    gpu::destroy_texture(&device, texture)!!;
    gpu::destroy_device(&device)!!;
}

// Pure: heap_fits_limits takes no device, so these synthesize HeapLimits
// values directly rather than depending on what the driver reports.

fn void heap_fits_limits_indexing_at_limit_fits() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 200,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        tex_capacity:   100,
        smp_capacity:   50,
        limits:         limits,
    );
    assert(fits, "texture and sampler capacities at the limit fit");
}

fn void heap_fits_limits_indexing_exceeds_per_stage_aggregate() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 199,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        tex_capacity:   100,
        smp_capacity:   50,
        limits:         limits,
    );
    assert(!fits, "aggregate per-stage limit binds when all per-type limits fit");
}

fn void heap_fits_limits_indexing_excludes_samplers_from_per_stage_aggregate() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 200,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        tex_capacity:   100,
        smp_capacity:   50,
        limits:         limits,
    );
    assert(fits, "plain samplers do not count against the per-stage resource aggregate");
}

fn void heap_fits_limits_indexing_exceeds_all_pools_aggregate() @test {
    gpu::vk::HeapLimits limits = {

        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 200,
        .max_all_pools_descriptors    = 249,
    };
    bool fits = gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        tex_capacity:   100,
        smp_capacity:   50,
        limits:         limits,
    );
    assert(!fits, "aggregate all-pools limit binds when all per-type limits fit");
}

fn void heap_fits_limits_indexing_widens_aggregate_arithmetic() @test {

    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = uint::max,
        .max_storage_images           = uint::max,
        .max_samplers                 = uint::max,
        .max_per_stage_sampled_images = uint::max,
        .max_per_stage_storage_images = uint::max,
        .max_per_stage_samplers       = uint::max,
        .max_per_stage_resources      = uint::max,
        .max_all_pools_descriptors    = uint::max,
    };
    bool fits = gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        tex_capacity:   0x8000_0000,
        smp_capacity:   0,
        limits:         limits,
    );
    assert(!fits, "aggregate arithmetic does not wrap at uint width");
}

fn void heap_fits_limits_indexing_exceeds_sampled_images() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images = 99,
        .max_storage_images = 100,
        .max_samplers       = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    assert(!gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        tex_capacity:   100,
        smp_capacity:   50,
        limits:         limits,
    ),
        "texture capacity over the sampled-image limit is rejected");
}

fn void heap_fits_limits_indexing_exceeds_storage_images() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images = 100,
        .max_storage_images = 99,
        .max_samplers       = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    assert(!gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        tex_capacity:   100,
        smp_capacity:   50,
        limits:         limits,
    ),
        "texture capacity over the storage-image limit is rejected");
}

fn void heap_fits_limits_indexing_exceeds_samplers() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images = 100,
        .max_storage_images = 100,
        .max_samplers       = 49,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 49,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    assert(!gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        tex_capacity:   100,
        smp_capacity:   50,
        limits:         limits,
    ),
        "sampler capacity over the sampler limit is rejected");
}

fn void heap_fits_limits_buffer_at_limit_fits() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        tex_capacity:   0,
        smp_capacity:   0,
        limits:         limits,
    ),
        "layout size at the limit fits");
}

fn void heap_fits_limits_buffer_exceeds_resource_range() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 2048,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(!gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        tex_capacity:   0,
        smp_capacity:   0,
        limits:         limits,
    ),
        "layout size over the resource descriptor buffer range is rejected");
}

fn void heap_fits_limits_buffer_exceeds_sampler_range() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 2048,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(!gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        tex_capacity:   0,
        smp_capacity:   0,
        limits:         limits,
    ),
        "layout size over the sampler descriptor buffer range is rejected");
}

fn void heap_fits_limits_buffer_exceeds_address_space() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 2048,
        .sampler_descriptor_address_space_size  = 2048,
.descriptor_buffer_address_space_size = 2048,
        .layout_size                          = 4096,
    };
    assert(!gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        tex_capacity:   0,
        smp_capacity:   0,
        limits:         limits,
    ),
        "layout size over the descriptor buffer address space is rejected");
}

fn void heap_fits_limits_indexing_exceeds_per_stage() @test {
    gpu::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 99,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_INDEXING,
        tex_capacity:   100,
        smp_capacity:   50,
        limits:         limits,
    );
    assert(!fits, "per-stage limit binds even when the set-level limit fits");
}

fn void heap_fits_limits_buffer_exceeds_per_type_address_space() @test {
    gpu::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range    = 4096,
        .max_sampler_descriptor_buffer_range     = 4096,
        .resource_descriptor_address_space_size  = 4096,
        .sampler_descriptor_address_space_size   = 2048,
        .descriptor_buffer_address_space_size    = 4096,
        .layout_size                             = 4096,
    };
    bool fits = gpu::vk::heap_fits_limits(
        implementation: gpu::vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        tex_capacity:   0,
        smp_capacity:   0,
        limits:         limits,
    );
    assert(!fits, "per-type address space binds even when the combined budget fits");
}

// Integration: a capacity above the library default, but still well within
// lavapipe's reported device limits, creates successfully.
fn void descriptor_heap_raised_capacity_creates() @test {
    gpu::Device device = make_heap_test_device(8192, 512);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_texture(&device);
    assert(gpu::get_device_caps(&device)!!.texture_heap_capacity == 8192, "caps report the created texture capacity");
    assert(gpu::get_device_caps(&device)!!.sampler_heap_capacity == 512, "caps report the created sampler capacity");
    defer gpu::destroy_texture(&device, texture)!!;

    gpu::TextureView index = gpu::create_texture_view(&device, texture, null)!!;
    assert(index.is_valid(), "raised-capacity heap creates and serves a descriptor");
    gpu::destroy_texture_view(&device, index)!!;
}

fn void descriptor_batch_and_sampler_remaining_failures_report_context() @test {
    gpu::Device device = make_heap_test_device(1, 1);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    DescriptorDiagnosticCapture capture;
    state.debug_callback = &capture_descriptor_diagnostic;
    state.debug_user_data = &capture;

    gpu::TextureViewCreateDesc[1] one = { { .texture = texture } };
    gpu::TextureView[2] too_many_outputs;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        expected_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_texture_views",
        expected_field:      "out_views",
        expected_invariant:  "output count must match descriptor count",
    );
    fault lengths = @catch(gpu::create_texture_views(&device, one[..], too_many_outputs[..]));
    assert(lengths == gpu::INVALID_ARGUMENT && capture.calls == 1 && capture.matched,
        "batch length rejection reports exact context");

    gpu::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, stale_texture)!!;
    gpu::TextureViewCreateDesc[1] stale_descs = { { .texture = stale_texture } };
    gpu::TextureView[1] stale_outputs = { { .owner = 17, .index.value = 19, .generation = 7 } };
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_HANDLE,
        expected_kind:       gpu::DebugResourceKind.TEXTURE,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_texture_views",
        expected_field:      "descs[].texture",
        expected_invariant:  "texture must identify a live generation",
    );
    fault stale = @catch(gpu::create_texture_views(&device, stale_descs[..], stale_outputs[..]));
    assert(stale == gpu::INVALID_HANDLE && capture.calls == 2 && capture.matched,
        "batch stale texture rejection omits unvalidated identity");
    assert(stale_outputs[0].owner == 17 && stale_outputs[0].index.value == 19
        && stale_outputs[0].generation == 7,
        "batch stale texture rejection preserves outputs");

    gpu::TextureViewDesc bad_view = { .format = gpu::Format.R32_UINT };
    gpu::TextureViewCreateDesc[1] bad_view_descs = { { .texture = texture, .view = bad_view } };
    gpu::TextureView[1] bad_view_outputs = { { .owner = 29, .index.value = 23, .generation = 9 } };
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        expected_kind:       gpu::DebugResourceKind.TEXTURE,
        expected_identity:   true,
        expected_index:      texture.index,
        expected_generation: texture.generation,
        expected_operation:  "create_texture_views",
        expected_field:      "descs[].view",
        expected_invariant:  "view must select a valid texture subresource range and format",
    );
    fault view = @catch(gpu::create_texture_views(&device, bad_view_descs[..], bad_view_outputs[..]));
    assert(view == gpu::INVALID_ARGUMENT && capture.calls == 3 && capture.matched,
        "batch view rejection reports resolved texture identity");
    assert(bad_view_outputs[0].owner == 29 && bad_view_outputs[0].index.value == 23
        && bad_view_outputs[0].generation == 9,
        "batch view rejection preserves outputs");

    gpu::TextureViewCreateDesc[2] over_capacity;
    foreach (&desc : over_capacity) *desc = { .texture = texture };
    gpu::TextureView[2] capacity_outputs;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::DESCRIPTOR_HEAP_FULL,
        expected_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_texture_views",
        expected_field:      "DeviceDesc.texture_heap_capacity",
        expected_invariant:  "must fit the complete descriptor batch",
    );
    fault capacity = @catch(gpu::create_texture_views(&device, over_capacity[..], capacity_outputs[..]));
    assert(capacity == gpu::DESCRIPTOR_HEAP_FULL && capture.calls == 4 && capture.matched,
        "batch capacity rejection reports exact context");

    gpu::SamplerDesc invalid_bias = {
        .mip_lod_bias = state.caps.max_sampler_lod_bias + 1.0f,
    };
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        expected_kind:       gpu::DebugResourceKind.SAMPLER,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "intern_sampler",
        expected_field:      "SamplerDesc.mip_lod_bias",
        expected_invariant:  "absolute value must not exceed DeviceCaps.max_sampler_lod_bias",
    );
    fault bias = @catch(gpu::intern_sampler(&device, &invalid_bias));
    assert(bias == gpu::INVALID_ARGUMENT && capture.calls == 5 && capture.matched,
        "over-limit sampler LOD bias reports exact context");

    gpu::SamplerDesc invalid_filter = {};
    *((int*)&invalid_filter.min_filter) = 99;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        expected_kind:       gpu::DebugResourceKind.SAMPLER,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "intern_sampler",
        expected_field:      "SamplerDesc.min_filter",
        expected_invariant:  "must name a defined enum value",
    );
    fault filter = @catch(gpu::intern_sampler(&device, &invalid_filter));
    assert(filter == gpu::INVALID_ARGUMENT && capture.calls == 6 && capture.matched,
        "invalid sampler enum reports exact context");

    gpu::SamplerDesc invalid_anisotropy = { .anisotropy_enable = true, .max_anisotropy = 0.5f };
    float supported_anisotropy = state.max_sampler_anisotropy;
    if (supported_anisotropy <= 0) state.max_sampler_anisotropy = 16.0f;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::INVALID_ARGUMENT,
        expected_kind:       gpu::DebugResourceKind.SAMPLER,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "intern_sampler",
        expected_field:      "SamplerDesc.max_anisotropy",
        expected_invariant:  "must be at least 1.0 when anisotropy is enabled",
    );
    fault anisotropy = @catch(gpu::intern_sampler(&device, &invalid_anisotropy));
    assert(anisotropy == gpu::INVALID_ARGUMENT && capture.calls == 7 && capture.matched,
        "invalid sampler anisotropy reports exact context");

    state.max_sampler_anisotropy = 0;
    gpu::SamplerDesc unsupported_anisotropy = { .anisotropy_enable = true, .max_anisotropy = 1.0f };
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::UNSUPPORTED_FEATURE,
        expected_kind:       gpu::DebugResourceKind.SAMPLER,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "intern_sampler",
        expected_field:      "SamplerDesc.anisotropy_enable",
        expected_invariant:  "device must support sampler anisotropy",
    );
    fault unsupported = @catch(gpu::intern_sampler(&device, &unsupported_anisotropy));
    assert(unsupported == gpu::UNSUPPORTED_FEATURE && capture.calls == 8 && capture.matched,
        "unsupported sampler anisotropy reports exact context");
    state.max_sampler_anisotropy = supported_anisotropy;
    state.debug_callback = null;
}

fn void texture_view_cache_growth_reports_host_oom_transactionally() @test {
    gpu::Device device = make_heap_test_device(16, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    state.host_allocator = &allocators::NULL_ALLOCATOR;
    gpu::TextureViewDesc view = {
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };

    fault allocation_fault = @catch(gpu::create_texture_view(&device, texture, &view));
    state.host_allocator = &allocators::LIBC_ALLOCATOR;
    assert(allocation_fault == gpu::OUT_OF_HOST_MEMORY,
        "host cache allocation failure remains recoverable");
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    assert(slot.views.len == 0,
        "failed host cache growth preserves the prior cache");

    gpu::TextureView published = gpu::create_texture_view(&device, texture, &view)!!;
    gpu::destroy_texture_view(&device, published)!!;
    assert(slot.views.len == vk::INITIAL_CACHED_VIEW_CAPACITY,
        "publication succeeds after the allocator recovers");
}

fn void texture_view_cache_grows_past_initial_capacity() @test {
    gpu::Device device = make_heap_test_device(16, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureView[vk::INITIAL_CACHED_VIEW_CAPACITY + 1] live;
    for (uint i = 0; i < vk::INITIAL_CACHED_VIEW_CAPACITY; i++) {
        gpu::TextureViewDesc view = {
            .base_mip    = i % 4,
            .mip_count   = 1,
            .base_layer  = i / 4,
            .layer_count = 1,
        };
        live[i] = gpu::create_texture_view(&device, texture, &view)!!;
    }

    gpu::TextureViewDesc beyond_initial_capacity = {
        .base_mip    = 4,
        .mip_count   = 1,
        .base_layer  = 3,
        .layer_count = 1,
    };
    live[vk::INITIAL_CACHED_VIEW_CAPACITY] = gpu::create_texture_view(
        &device,
        texture,
        &beyond_initial_capacity,
    )!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(cached_texture_view_count(state, texture) == vk::INITIAL_CACHED_VIEW_CAPACITY + 1,
        "distinct texture views grow past the initial native-view cache capacity");

    foreach (index : live) gpu::destroy_texture_view(&device, index)!!;
}

fn void texture_view_batch_rollback_survives_cache_growth() @test {
    gpu::Device device = make_heap_test_device(16, 4);
    defer destroy_test_device(&device)!!;
    gpu::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::destroy_texture(&device, texture)!!;
    gpu::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::destroy_texture(&device, stale_texture)!!;

    gpu::TextureViewCreateDesc[vk::INITIAL_CACHED_VIEW_CAPACITY + 2] descs;
    for (uint i = 0; i <= vk::INITIAL_CACHED_VIEW_CAPACITY; i++) {
        descs[i] = {
            .texture = texture,
            .view = {
                .base_mip    = i % 5,
                .mip_count   = 1,
                .base_layer  = i / 5,
                .layer_count = 1,
            },
        };
    }
    descs[vk::INITIAL_CACHED_VIEW_CAPACITY + 1] = { .texture = stale_texture };
    gpu::TextureView[vk::INITIAL_CACHED_VIEW_CAPACITY + 2] views;

    assert(@catch(gpu::create_texture_views(&device, descs[..], views[..]))
        == gpu::INVALID_HANDLE,
        "a stale final item rejects a batch after native-view cache growth");
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::device_backend_state(device);
    assert(cached_texture_view_count(state, texture) == 0,
        "batch rollback removes creations made before and after cache growth");
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    assert(slot.views.len == 0,
        "batch rollback restores the pre-growth native-view cache capacity");
    foreach (view : views) assert(!view.is_valid(),
        "cache-growth rollback leaves batch outputs unchanged");
}

fn void descriptor_bootstrap_result_mappings_emit_once_and_preserve_faults() @test {
    vk::VkDeviceState state;
    DescriptorBackendCapture capture;
    state.debug_callback = &capture_descriptor_backend_diagnostic;
    state.debug_user_data = &capture;

    capture = {
        .expected_fault     = gpu::DESCRIPTOR_HEAP_FULL,
        .expected_operation = "create_device",
        .expected_text      = "Vulkan descriptor pool is fragmented",
        .expected_result    = (int)vk::Result.ERROR_FRAGMENTATION,
    };
    fault pool = @catch(gpu::vk::check_descriptor_pool_backend_result(
        state:     &state,
        operation: "create_device",
        result:    vk::Result.ERROR_FRAGMENTATION,
    ));
    assert(pool == gpu::DESCRIPTOR_HEAP_FULL && capture.calls == 1 && capture.matched,
        "descriptor-pool specialization is reported without changing its fault");

    capture = {
        .calls              = capture.calls,
        .expected_fault     = gpu::DESCRIPTOR_HEAP_FULL,
        .expected_operation = "create_device",
        .expected_text      = "Vulkan descriptor allocation exhausted the pool",
        .expected_result    = (int)vk::Result.ERROR_OUT_OF_POOL_MEMORY,
    };
    fault allocate = @catch(gpu::vk::check_descriptor_allocate_backend_result(
        state:     &state,
        operation: "create_device",
        result:    vk::Result.ERROR_OUT_OF_POOL_MEMORY,
    ));
    assert(allocate == gpu::DESCRIPTOR_HEAP_FULL && capture.calls == 2 && capture.matched,
        "descriptor allocation specialization is reported without changing its fault");

    capture = {
        .calls              = capture.calls,
        .expected_fault     = gpu::BACKEND_ERROR,
        .expected_operation = "create_device",
        .expected_text      = "descriptor buffer device address is zero",
        .expected_result    = 0,
    };
    fault address = @catch(gpu::vk::report_descriptor_backend_failure(
        state:         &state,
        operation:     "create_device",
        public_fault:  gpu::BACKEND_ERROR,
        backend_text:  "descriptor buffer device address is zero",
        backend_result: 0,
    ));
    assert(address == gpu::BACKEND_ERROR && capture.calls == 3 && capture.matched,
        "descriptor bootstrap failure without VkResult is observable");

    gpu::TextureHandle texture = { .owner = 1, .index = 7, .generation = 11 };
    vk::TextureSlot texture_slot;
    capture = {
        .calls               = capture.calls,
        .expected_fault      = gpu::OUT_OF_DEVICE_MEMORY,
        .expected_kind       = gpu::DebugResourceKind.TEXTURE,
        .expected_identity   = true,
        .expected_index      = texture.index,
        .expected_generation = texture.generation,
        .expected_operation  = "create_texture_view",
        .expected_text       = "Vulkan device memory exhausted",
        .expected_result     = (int)vk::Result.ERROR_OUT_OF_DEVICE_MEMORY,
    };
    fault view_backend = @catch(gpu::vk::check_texture_view_backend_result(
        state:     &state,
        operation: "create_texture_view",
        texture:   texture,
        slot:      &texture_slot,
        result:    vk::Result.ERROR_OUT_OF_DEVICE_MEMORY,
    ));
    assert(view_backend == gpu::OUT_OF_DEVICE_MEMORY && capture.calls == 4 && capture.matched,
        "image-view creation reports operation and resolved texture identity");

    state.debug_callback = null;
    fault fallback = @catch(gpu::vk::check_descriptor_pool_backend_result(
        state:     &state,
        operation: "create_device",
        result:    vk::Result.ERROR_FRAGMENTATION,
    ));
    assert(fallback == pool && capture.calls == 4,
        "null callback preserves descriptor bootstrap fault without a duplicate");
}
