module gpu_test;

import gpu::compat @public;
import gpu::compat::vk @public;
import vk;

// Headless (lavapipe): descriptor heap slot bookkeeping through the public API.

const uint DESCRIPTOR_BUFFER_USAGE = (uint)vk::BufferUsageFlagBits.RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT
    | (uint)vk::BufferUsageFlagBits.SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT
    | (uint)vk::BufferUsageFlagBits.SHADER_DEVICE_ADDRESS_BIT;

struct DescriptorDiagnosticCapture {
    uint calls;
    fault expected_fault;
    gpu::compat::DebugResourceKind expected_kind;
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
    gpu::compat::DebugResourceKind expected_kind;
    bool expected_identity;
    uint expected_index;
    uint expected_generation;
    ZString expected_operation;
    ZString expected_text;
    int expected_result;
    bool matched;
}

fn void capture_descriptor_backend_diagnostic(gpu::compat::DebugMessage* message, void* user_data) {
    DescriptorBackendCapture* capture = (DescriptorBackendCapture*)user_data;
    capture.calls++;
    capture.matched = message.severity == gpu::compat::DebugMessageSeverity.ERROR
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

fn void capture_descriptor_diagnostic(gpu::compat::DebugMessage* message, void* user_data) {
    DescriptorDiagnosticCapture* capture = (DescriptorDiagnosticCapture*)user_data;
    capture.calls++;
    capture.matched = message.severity == gpu::compat::DebugMessageSeverity.ERROR
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
    gpu::compat::DebugResourceKind expected_kind,
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

    vk::BufferCreateInfo info = gpu::compat::vk::make_descriptor_buffer_create_info(
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

    vk::BufferCreateInfo info = gpu::compat::vk::make_descriptor_buffer_create_info(
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

fn gpu::compat::Device make_heap_test_device(uint tex_capacity, uint smp_capacity) {
    gpu::compat::DeviceDesc desc = {
        .backend                     = gpu::compat::BackendKind.VULKAN,
        .enable_validation           = true,
        .frames_in_flight            = 2,
        .descriptor_heap_mode        = gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        .texture_descriptor_capacity = tex_capacity,
        .sampler_descriptor_capacity = smp_capacity,
        .application_name            = "gpu_descriptor_heap_test",
    };
    return gpu::compat::create_device(&desc)!!;
}

fn gpu::compat::TextureHandle make_heap_texture(gpu::compat::Device* device) {
    gpu::compat::TextureDesc desc = {
        .dimension = gpu::compat::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::compat::Format.RGBA8_UNORM,
        .usage     = { .sampled, .storage },
    };
    return gpu::compat::create_texture(device, &desc)!!;
}

fn gpu::compat::TextureHandle make_heap_view_texture(gpu::compat::Device* device) {
    gpu::compat::TextureDesc desc = {
        .dimension    = gpu::compat::TextureDimension.TEX_2D,
        .width        = 16,
        .height       = 16,
        .mip_levels   = 5,
        .array_layers = 4,
        .format       = gpu::compat::Format.RGBA8_UNORM,
        .usage        = { .sampled, .storage },
    };
    return gpu::compat::create_texture(device, &desc)!!;
}

fn void descriptor_sampler_failures_report_context() @test {
    gpu::compat::Device device = make_heap_test_device(1, 1);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;
    gpu::compat::TextureHandle transfer = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, transfer)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    state.textures.slots[transfer.index].value.usage = { .transfer_dst };
    DescriptorDiagnosticCapture capture;
    state.debug_callback = &capture_descriptor_diagnostic;
    state.debug_user_data = &capture;

    gpu::compat::TextureIndex descriptor = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::DESCRIPTOR_HEAP_FULL,
        expected_kind:       gpu::compat::DebugResourceKind.TEXTURE_DESCRIPTOR,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_texture_descriptor",
        expected_field:      "DeviceDesc.texture_descriptor_capacity",
        expected_invariant:  "must exceed live and retiring texture descriptors",
    );
    fault full = @catch(gpu::compat::create_texture_descriptor(&device, texture, null));
    assert(full == gpu::compat::DESCRIPTOR_HEAP_FULL && capture.calls == 1 && capture.matched,
        "descriptor capacity fault and diagnostic are exact");

    gpu::compat::destroy_texture_descriptor(&device, descriptor)!!;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::INVALID_HANDLE,
        expected_kind:       gpu::compat::DebugResourceKind.TEXTURE_DESCRIPTOR,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "destroy_texture_descriptor",
        expected_field:      "index",
        expected_invariant:  "descriptor index must identify a live generation",
    );
    fault stale = @catch(gpu::compat::destroy_texture_descriptor(&device, descriptor));
    assert(stale == gpu::compat::INVALID_HANDLE && capture.calls == 2 && capture.matched,
        "stale descriptor fault has kind without unvalidated identity");

    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::INVALID_ARGUMENT,
        expected_kind:       gpu::compat::DebugResourceKind.TEXTURE,
        expected_identity:   true,
        expected_index:      transfer.index,
        expected_generation: transfer.generation,
        expected_operation:  "create_texture_descriptor",
        expected_field:      "TextureDesc.usage",
        expected_invariant:  "texture must have sampled or storage usage",
    );
    fault usage = @catch(gpu::compat::create_texture_descriptor(&device, transfer, null));
    assert(usage == gpu::compat::INVALID_ARGUMENT && capture.calls == 3 && capture.matched,
        "descriptor usage fault and texture identity are exact");

    gpu::compat::SamplerDesc sampler_desc = {};
    gpu::compat::SamplerIndex sampler = gpu::compat::create_sampler(&device, &sampler_desc)!!;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::DESCRIPTOR_HEAP_FULL,
        expected_kind:       gpu::compat::DebugResourceKind.SAMPLER,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_sampler",
        expected_field:      "DeviceDesc.sampler_descriptor_capacity",
        expected_invariant:  "must exceed live and retiring samplers",
    );
    fault sampler_full = @catch(gpu::compat::create_sampler(&device, &sampler_desc));
    assert(sampler_full == gpu::compat::DESCRIPTOR_HEAP_FULL && capture.calls == 4 && capture.matched,
        "sampler capacity fault and diagnostic are exact");

    gpu::compat::destroy_sampler(&device, sampler)!!;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::INVALID_HANDLE,
        expected_kind:       gpu::compat::DebugResourceKind.SAMPLER,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "destroy_sampler",
        expected_field:      "index",
        expected_invariant:  "sampler index must identify a live generation",
    );
    fault sampler_stale = @catch(gpu::compat::destroy_sampler(&device, sampler));
    assert(sampler_stale == gpu::compat::INVALID_HANDLE && capture.calls == 5 && capture.matched,
        "stale sampler fault has kind without unvalidated identity");

    state.debug_callback = null;
    fault fallback = @catch(gpu::compat::destroy_sampler(&device, sampler));
    assert(fallback == sampler_stale && capture.calls == 5,
        "null callback preserves fault parity without a structured duplicate");
}

fn uint cached_texture_view_count(vk::VkDeviceState* state, gpu::compat::TextureHandle texture) {
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    uint count = 0;
    foreach (&cached : slot.views) if (cached.used) count++;
    return count;
}

fn void texture_descriptor_round_trip() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureIndex index = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    assert(index.is_valid(), "created index is valid");
    gpu::compat::destroy_texture_descriptor(&device, index)!!;

    fault stale_fault = @catch(gpu::compat::destroy_texture_descriptor(&device, index));
    assert(stale_fault == gpu::compat::INVALID_HANDLE, "destroying a stale index faults INVALID_HANDLE");
}

fn void texture_descriptor_exhaustion() @test {
    gpu::compat::Device device = make_heap_test_device(2, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureIndex a = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    gpu::compat::TextureIndex b = gpu::compat::create_texture_descriptor(&device, texture, null)!!;

    fault full_fault = @catch(gpu::compat::create_texture_descriptor(&device, texture, null));
    assert(full_fault == gpu::compat::DESCRIPTOR_HEAP_FULL, "allocating past capacity faults DESCRIPTOR_HEAP_FULL");
    assert(a.is_valid() && b.is_valid(), "existing indices stay valid after exhaustion");

    gpu::compat::destroy_texture_descriptor(&device, a)!!;
    gpu::compat::destroy_texture_descriptor(&device, b)!!;
}

fn void texture_descriptor_exhaustion_does_not_cache_failed_views() @test {
    gpu::compat::Device device = make_heap_test_device(1, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;
    gpu::compat::TextureIndex occupied = gpu::compat::create_texture_descriptor(&device, texture, null)!!;

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    assert(cached_texture_view_count(state, texture) == 0, "the fixture starts with an empty view cache");

    for (uint i = 0; i < vk::MAX_CACHED_VIEWS; i++) {
        gpu::compat::TextureViewDesc view = {
            .base_mip    = i % 4,
            .mip_count   = 1,
            .base_layer  = i / 4,
            .layer_count = 1,
        };
        fault full = @catch(gpu::compat::create_texture_descriptor(&device, texture, &view));
        assert(full == gpu::compat::DESCRIPTOR_HEAP_FULL, "an exhausted heap rejects the descriptor");
        assert(cached_texture_view_count(state, texture) == 0,
            "a rejected descriptor leaves the texture view cache unchanged");
    }

    gpu::compat::destroy_texture_descriptor(&device, occupied)!!;
}

fn void texture_descriptor_deferred_reuse() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureIndex first = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    ushort first_slot = first.slot;

    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::destroy_texture_descriptor(&device, first)!!;

    gpu::compat::TextureIndex during = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    assert(during.slot != first_slot, "slot is not reused while its frame is in flight");

    gpu::compat::end_frame(&frame_token)!!;
    // The frame fence signals GPU-side; idle so the timeline has passed.
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::TextureIndex recycled = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    assert(recycled.slot == first_slot, "retired slot is recycled once the frame timeline passes");
    assert(recycled.generation != first.generation, "recycled slot carries a new generation");

    fault recycled_fault = @catch(gpu::compat::destroy_texture_descriptor(&device, first));
    assert(recycled_fault == gpu::compat::INVALID_HANDLE, "the pre-recycle index is stale after reuse");

    gpu::compat::destroy_texture_descriptor(&device, during)!!;
    gpu::compat::destroy_texture_descriptor(&device, recycled)!!;
}

fn void texture_descriptor_stale_texture_rejected() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    gpu::compat::destroy_texture(&device, texture)!!;

    fault texture_fault = @catch(gpu::compat::create_texture_descriptor(&device, texture, null));
    assert(texture_fault == gpu::compat::INVALID_HANDLE, "registering a destroyed texture faults INVALID_HANDLE");
}

fn void texture_descriptor_bad_view_rejected() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureViewDesc reinterpret = { .format = gpu::compat::Format.R32_UINT };
    fault view_fault = @catch(gpu::compat::create_texture_descriptor(&device, texture, &reinterpret));
    assert(view_fault == gpu::compat::INVALID_ARGUMENT, "format-reinterpreting views fault INVALID_ARGUMENT");
}

fn void texture_descriptor_overflowing_view_range_rejected() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureViewDesc mip_overflow = { .base_mip = 1, .mip_count = uint::max };
    fault mip_fault = @catch(gpu::compat::create_texture_descriptor(&device, texture, &mip_overflow));
    assert(mip_fault == gpu::compat::INVALID_ARGUMENT, "overflowing mip range faults INVALID_ARGUMENT");

    gpu::compat::TextureViewDesc layer_overflow = { .base_layer = 1, .layer_count = uint::max };
    fault layer_fault = @catch(gpu::compat::create_texture_descriptor(&device, texture, &layer_overflow));
    assert(layer_fault == gpu::compat::INVALID_ARGUMENT, "overflowing layer range faults INVALID_ARGUMENT");
}

fn void destroy_texture_with_live_descriptor_faults_and_recovers() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);

    gpu::compat::TextureIndex index = gpu::compat::create_texture_descriptor(&device, texture, null)!!;

    fault in_use_fault = @catch(gpu::compat::destroy_texture(&device, texture));
    assert(in_use_fault == gpu::compat::RESOURCE_IN_USE, "destroying a texture with a live descriptor faults RESOURCE_IN_USE");

    gpu::compat::TextureIndex second = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    assert(second.is_valid(), "handle stays valid and usable after the faulted destroy");
    gpu::compat::destroy_texture_descriptor(&device, second)!!;

    gpu::compat::destroy_texture_descriptor(&device, index)!!;
    gpu::compat::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_retired_undrained_descriptor_succeeds() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);

    gpu::compat::TextureIndex index = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    gpu::compat::destroy_texture_descriptor(&device, index)!!;

    gpu::compat::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_blocks_while_any_descriptor_live() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);

    gpu::compat::TextureIndex a = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    gpu::compat::TextureIndex b = gpu::compat::create_texture_descriptor(&device, texture, null)!!;

    gpu::compat::destroy_texture_descriptor(&device, a)!!;
    fault still_in_use = @catch(gpu::compat::destroy_texture(&device, texture));
    assert(still_in_use == gpu::compat::RESOURCE_IN_USE, "one remaining live descriptor still blocks destroy_texture");

    gpu::compat::destroy_texture_descriptor(&device, b)!!;
    gpu::compat::destroy_texture(&device, texture)!!;
}

fn void destroy_texture_unaffected_by_unrelated_descriptor() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture_a = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture_a)!!;
    gpu::compat::TextureHandle texture_b = make_heap_texture(&device);

    gpu::compat::TextureIndex index = gpu::compat::create_texture_descriptor(&device, texture_a, null)!!;
    defer gpu::compat::destroy_texture_descriptor(&device, index)!!;

    gpu::compat::destroy_texture(&device, texture_b)!!;
}

fn void texture_descriptors_batch_happy_path() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureDescriptorDesc[3] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::compat::TextureIndex[3] indices;
    gpu::compat::create_texture_descriptors(&device, descs[..], indices[..])!!;

    foreach (index : indices) assert(index.is_valid(), "every batch-created index is valid");
    assert(indices[0].slot != indices[1].slot && indices[1].slot != indices[2].slot && indices[0].slot != indices[2].slot,
        "batch-created indices occupy distinct slots");

    foreach (index : indices) gpu::compat::destroy_texture_descriptor(&device, index)!!;
}

fn void texture_descriptors_batch_fills_capacity_exactly() @test {
    gpu::compat::Device device = make_heap_test_device(3, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureDescriptorDesc[3] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::compat::TextureIndex[3] indices;
    gpu::compat::create_texture_descriptors(&device, descs[..], indices[..])!!;

    fault full_fault = @catch(gpu::compat::create_texture_descriptor(&device, texture, null));
    assert(full_fault == gpu::compat::DESCRIPTOR_HEAP_FULL, "a batch that exactly fills capacity leaves none spare — the heap count rose by exactly N");

    foreach (index : indices) gpu::compat::destroy_texture_descriptor(&device, index)!!;
}

fn void texture_descriptors_batch_zeroed_view_is_default() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureDescriptorDesc[1] descs = { { .texture = texture } };
    gpu::compat::TextureIndex[1] indices;
    gpu::compat::create_texture_descriptors(&device, descs[..], indices[..])!!;
    assert(indices[0].is_valid(), "a zero-initialized view collapses to the default view and creates successfully");

    gpu::compat::destroy_texture_descriptor(&device, indices[0])!!;
}

fn void texture_descriptors_mismatched_out_len_faults() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureDescriptorDesc[2] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::compat::TextureIndex[1] indices;
    fault mismatch_fault = @catch(gpu::compat::create_texture_descriptors(&device, descs[..], indices[..]));
    assert(mismatch_fault == gpu::compat::INVALID_ARGUMENT, "a mismatched out-indices length faults INVALID_ARGUMENT");
}

fn void texture_descriptors_empty_batch_is_no_op() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::TextureDescriptorDesc[] descs;
    gpu::compat::TextureIndex[] indices;
    gpu::compat::create_texture_descriptors(&device, descs, indices)!!;
}

fn void texture_descriptors_batch_preflight_faults_untouched() @test {
    gpu::compat::Device device = make_heap_test_device(4, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureIndex existing = gpu::compat::create_texture_descriptor(&device, texture, null)!!;

    // Capacity 4, one slot already used: 3 remain; the batch asks for 4.
    gpu::compat::TextureDescriptorDesc[4] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::compat::TextureIndex[4] indices;
    fault full_fault = @catch(gpu::compat::create_texture_descriptors(&device, descs[..], indices[..]));
    assert(full_fault == gpu::compat::DESCRIPTOR_HEAP_FULL, "a batch exceeding remaining capacity faults DESCRIPTOR_HEAP_FULL");
    assert(existing.is_valid(), "the pre-existing index survives the failed pre-flight");

    // Untouched: the failed pre-flight consumed nothing, so the remaining
    // headroom (3 slots) is still there to fill exactly.
    gpu::compat::TextureIndex a = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    gpu::compat::TextureIndex b = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    gpu::compat::TextureIndex c = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    assert(a.is_valid() && b.is_valid() && c.is_valid(), "the failed batch left the full remaining headroom available");

    gpu::compat::destroy_texture_descriptor(&device, existing)!!;
    gpu::compat::destroy_texture_descriptor(&device, a)!!;
    gpu::compat::destroy_texture_descriptor(&device, b)!!;
    gpu::compat::destroy_texture_descriptor(&device, c)!!;
}

fn void texture_descriptors_failed_batch_restores_full_view_capacity() @test {
    gpu::compat::Device device = make_heap_test_device(32, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;
    gpu::compat::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::compat::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);

    for (uint i = 0; i < vk::MAX_CACHED_VIEWS; i++) {
        gpu::compat::TextureViewDesc failed_view = {
            .base_mip    = i % 4,
            .mip_count   = 1,
            .base_layer  = i / 4,
            .layer_count = 1,
        };
        gpu::compat::TextureDescriptorDesc[2] descs = {
            { .texture = texture, .view = failed_view },
            { .texture = stale_texture },
        };
        gpu::compat::TextureIndex[2] indices;
        fault batch_fault = @catch(gpu::compat::create_texture_descriptors(&device, descs[..], indices[..]));
        assert(batch_fault == gpu::compat::INVALID_HANDLE, "a later stale texture faults the batch");
        assert(cached_texture_view_count(state, texture) == 0,
            "a failed batch restores the empty view cache");
    }

    gpu::compat::TextureIndex[vk::MAX_CACHED_VIEWS] live;
    for (uint i = 0; i < vk::MAX_CACHED_VIEWS; i++) {
        gpu::compat::TextureViewDesc valid_view = {
            .base_mip    = i % 4,
            .mip_count   = 1,
            .base_layer  = 2 + i / 4,
            .layer_count = 1,
        };
        live[i] = gpu::compat::create_texture_descriptor(&device, texture, &valid_view)!!;
    }
    assert(cached_texture_view_count(state, texture) == vk::MAX_CACHED_VIEWS,
        "all cache entries remain available after repeated failed batches");

    foreach (index : live) gpu::compat::destroy_texture_descriptor(&device, index)!!;
}

fn void texture_descriptors_failed_batch_restores_multiple_created_views() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;
    gpu::compat::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::compat::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;
    DescriptorDiagnosticCapture capture;
    state.debug_callback = &capture_descriptor_diagnostic;
    state.debug_user_data = &capture;

    gpu::compat::TextureViewDesc first_view = {
        .base_mip    = 0,
        .mip_count   = 1,
        .base_layer  = 0,
        .layer_count = 1,
    };
    gpu::compat::TextureViewDesc second_view = {
        .base_mip    = 1,
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };
    gpu::compat::TextureDescriptorDesc[3] descs = {
        { .texture = texture, .view = first_view },
        { .texture = texture, .view = second_view },
        { .texture = stale_texture },
    };
    gpu::compat::TextureIndex[3] indices;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::INVALID_HANDLE,
        expected_kind:       gpu::compat::DebugResourceKind.TEXTURE,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_texture_descriptors",
        expected_field:      "descs[].texture",
        expected_invariant:  "texture must identify a live generation",
    );
    fault batch_fault = @catch(gpu::compat::create_texture_descriptors(&device, descs[..], indices[..]));
    assert(batch_fault == gpu::compat::INVALID_HANDLE && capture.calls == 1 && capture.matched,
        "a later stale texture emits once after two new views");
    assert(cached_texture_view_count(state, texture) == 0,
        "rollback removes every view created by the failed transaction");
    assert(!slot.views[0].used && !slot.views[1].used,
        "rollback restores both cache cells to their prior unused state");
    state.debug_callback = null;
}

fn void texture_descriptors_failed_batch_preserves_existing_and_duplicate_views() @test {
    gpu::compat::Device device = make_heap_test_device(16, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;
    gpu::compat::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::compat::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    vk::TextureSlot* slot = &state.textures.slots[texture.index].value;

    gpu::compat::TextureViewDesc existing_view = {
        .base_mip    = 0,
        .mip_count   = 1,
        .base_layer  = 0,
        .layer_count = 1,
    };
    gpu::compat::TextureIndex existing_index = gpu::compat::create_texture_descriptor(&device, texture, &existing_view)!!;
    vk::CachedView existing_cache = slot.views[0];

    gpu::compat::TextureViewDesc new_view = {
        .base_mip    = 1,
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };
    gpu::compat::TextureDescriptorDesc[4] descs = {
        { .texture = texture, .view = existing_view },
        { .texture = texture, .view = new_view },
        { .texture = texture, .view = new_view },
        { .texture = stale_texture },
    };
    gpu::compat::TextureIndex[4] indices;
    fault batch_fault = @catch(gpu::compat::create_texture_descriptors(&device, descs[..], indices[..]));
    assert(batch_fault == gpu::compat::INVALID_HANDLE, "a fault after a duplicate new view rejects the batch");
    assert(cached_texture_view_count(state, texture) == 1,
        "rollback removes the one transaction-created view and preserves the existing view");
    assert(slot.views[0].used && slot.views[0].view == existing_cache.view,
        "the pre-existing cached Vulkan view remains owned by the texture");
    assert(slot.views[0].base_mip == existing_cache.base_mip
        && slot.views[0].mip_count == existing_cache.mip_count
        && slot.views[0].base_layer == existing_cache.base_layer
        && slot.views[0].layer_count == existing_cache.layer_count,
        "the pre-existing cache entry remains unchanged");

    gpu::compat::destroy_texture_descriptor(&device, existing_index)!!;
}

fn void texture_descriptors_mid_batch_fault_preserves_descriptor_state() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureIndex[7] occupied;
    for (uint i = 0; i < 7; i++) {
        occupied[i] = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    }
    gpu::compat::TextureIndex retired = occupied[0];
    gpu::compat::FrameToken frame_token = gpu::compat::begin_frame(&device)!!;
    gpu::compat::destroy_texture_descriptor(&device, retired)!!;
    gpu::compat::end_frame(&frame_token)!!;
    gpu::compat::wait_queue_idle(&device, gpu::compat::QueueKind.GRAPHICS)!!;

    gpu::compat::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::compat::destroy_texture(&device, stale_texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    vk::DescriptorHeapState* heap = &state.heap;
    assert(heap.texture_retire.count > 0, "the fixture has a ready undrained descriptor retire");

    vk::TextureDescriptorCell[8] cells_before;
    uint[8] free_before;
    for (uint i = 0; i < 8; i++) {
        cells_before[i] = heap.texture_cells[i];
        free_before[i] = heap.texture_free[i];
    }
    uint free_count_before = heap.texture_free_count;
    uint next_new_before = heap.texture_next_new;
    uint retire_head_before = heap.texture_retire.head;
    uint retire_count_before = heap.texture_retire.count;

    gpu::compat::TextureViewDesc view = {
        .base_mip    = 1,
        .mip_count   = 1,
        .base_layer  = 1,
        .layer_count = 1,
    };
    gpu::compat::TextureDescriptorDesc[2] descs = {
        { .texture = texture, .view = view },
        { .texture = stale_texture },
    };
    gpu::compat::TextureIndex[2] indices = {
        { .slot = 6, .generation = 111 },
        { .slot = 7, .generation = 222 },
    };
    gpu::compat::TextureIndex[2] outputs_before = indices;
    fault batch_fault = @catch(gpu::compat::create_texture_descriptors(&device, descs[..], indices[..]));
    assert(batch_fault == gpu::compat::INVALID_HANDLE, "a mid-batch destroyed-texture reference faults INVALID_HANDLE");

    assert(heap.texture_free_count == free_count_before, "failure preserves free-list count");
    assert(heap.texture_next_new == next_new_before, "failure preserves the never-used slot cursor");
    assert(heap.texture_retire.head == retire_head_before
        && heap.texture_retire.count == retire_count_before,
        "failure leaves ready retires undrained");
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
    assert(indices[0].slot == outputs_before[0].slot
        && indices[0].generation == outputs_before[0].generation
        && indices[1].slot == outputs_before[1].slot
        && indices[1].generation == outputs_before[1].generation,
        "failure leaves output indices untouched");

    gpu::compat::TextureDescriptorDesc[2] valid = {
        { .texture = texture },
        { .texture = texture },
    };
    gpu::compat::TextureIndex[2] committed;
    gpu::compat::create_texture_descriptors(&device, valid[..], committed[..])!!;
    assert(committed[0].is_valid() && committed[1].is_valid(),
        "a two-item batch commits using one never-used slot and one ready retire");
    assert(heap.texture_retire.count == 0, "successful commit drains the ready retire");
    foreach (index : committed) gpu::compat::destroy_texture_descriptor(&device, index)!!;
    for (uint i = 1; i < 7; i++) {
        gpu::compat::destroy_texture_descriptor(&device, occupied[i])!!;
    }
}

fn void texture_descriptors_batch_created_descriptor_blocks_texture_destroy() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);

    gpu::compat::TextureDescriptorDesc[2] descs;
    foreach (&desc : descs) *desc = { .texture = texture };
    gpu::compat::TextureIndex[2] indices;
    gpu::compat::create_texture_descriptors(&device, descs[..], indices[..])!!;

    fault in_use_fault = @catch(gpu::compat::destroy_texture(&device, texture));
    assert(in_use_fault == gpu::compat::RESOURCE_IN_USE, "a batch-created descriptor blocks destroy_texture the same as a single-created one");

    foreach (index : indices) gpu::compat::destroy_texture_descriptor(&device, index)!!;
    gpu::compat::destroy_texture(&device, texture)!!;
}

fn void sampler_round_trip_and_exhaustion() @test {
    gpu::compat::Device device = make_heap_test_device(8, 2);
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::SamplerDesc desc = {
        .min_filter = gpu::compat::Filter.NEAREST,
        .mag_filter = gpu::compat::Filter.NEAREST,
        .address_u  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_v  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
        .address_w  = gpu::compat::AddressMode.CLAMP_TO_EDGE,
    };
    gpu::compat::SamplerIndex a = gpu::compat::create_sampler(&device, &desc)!!;
    gpu::compat::SamplerIndex b = gpu::compat::create_sampler(&device, &desc)!!;
    assert(a.is_valid() && b.is_valid(), "sampler indices valid");

    fault full_fault = @catch(gpu::compat::create_sampler(&device, &desc));
    assert(full_fault == gpu::compat::DESCRIPTOR_HEAP_FULL, "allocating past sampler capacity faults DESCRIPTOR_HEAP_FULL");

    gpu::compat::destroy_sampler(&device, a)!!;
    fault stale_fault = @catch(gpu::compat::destroy_sampler(&device, a));
    assert(stale_fault == gpu::compat::INVALID_HANDLE, "destroying a stale sampler index faults INVALID_HANDLE");

    gpu::compat::destroy_sampler(&device, b)!!;
}

fn void sampler_non_finite_and_reversed_ranges_rejected() @test {
    gpu::compat::Device device = make_heap_test_device(8, 8);
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::SamplerDesc nan_bias = { .mip_lod_bias = float::nan };
    fault bias_fault = @catch(gpu::compat::create_sampler(&device, &nan_bias));
    assert(bias_fault == gpu::compat::INVALID_ARGUMENT, "non-finite mip bias faults INVALID_ARGUMENT");

    gpu::compat::SamplerDesc nan_lod = { .min_lod = float::nan };
    fault lod_fault = @catch(gpu::compat::create_sampler(&device, &nan_lod));
    assert(lod_fault == gpu::compat::INVALID_ARGUMENT, "non-finite LOD range faults INVALID_ARGUMENT");

    gpu::compat::SamplerDesc nan_max_lod = { .max_lod = float::nan };
    fault max_lod_fault = @catch(gpu::compat::create_sampler(&device, &nan_max_lod));
    assert(max_lod_fault == gpu::compat::INVALID_ARGUMENT, "non-finite maximum LOD faults INVALID_ARGUMENT");

    gpu::compat::SamplerDesc reversed_lod = { .min_lod = 2.0f, .max_lod = 1.0f };
    fault reversed_fault = @catch(gpu::compat::create_sampler(&device, &reversed_lod));
    assert(reversed_fault == gpu::compat::INVALID_ARGUMENT, "reversed LOD range faults INVALID_ARGUMENT");

    gpu::compat::SamplerDesc nan_anisotropy = { .anisotropy_enable = true, .max_anisotropy = float::nan };
    fault anisotropy_fault = @catch(gpu::compat::create_sampler(&device, &nan_anisotropy));
    assert(anisotropy_fault == gpu::compat::INVALID_ARGUMENT, "non-finite anisotropy faults INVALID_ARGUMENT");
}

fn void leaked_descriptors_survive_teardown() @test {
    gpu::compat::Device device = make_heap_test_device(8, 4);
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    gpu::compat::create_texture_descriptor(&device, texture, null)!!;

    gpu::compat::SamplerDesc desc = {};
    gpu::compat::create_sampler(&device, &desc)!!;

    // Leaked on purpose: teardown must report and stay clean.
    gpu::compat::destroy_device(&device)!!;
}

// Pure: heap_fits_limits takes no device, so these synthesize HeapLimits
// values directly rather than depending on what the driver reports.

fn void heap_fits_limits_indexing_at_limit_fits() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 200,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(fits, "texture and sampler capacities at the limit fit");
}

fn void heap_fits_limits_indexing_exceeds_per_stage_aggregate() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 199,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(!fits, "aggregate per-stage limit binds when all per-type limits fit");
}

fn void heap_fits_limits_indexing_excludes_samplers_from_per_stage_aggregate() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 200,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(fits, "plain samplers do not count against the per-stage resource aggregate");
}
fn void heap_fits_limits_indexing_exceeds_all_pools_aggregate() @test {
    gpu::compat::vk::HeapLimits limits = {

        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 200,
        .max_all_pools_descriptors    = 249,
    };
    bool fits = gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(!fits, "aggregate all-pools limit binds when all per-type limits fit");
}
fn void heap_fits_limits_indexing_widens_aggregate_arithmetic() @test {

    gpu::compat::vk::HeapLimits limits = {
        .max_sampled_images           = uint::max,
        .max_storage_images           = uint::max,
        .max_samplers                 = uint::max,
        .max_per_stage_sampled_images = uint::max,
        .max_per_stage_storage_images = uint::max,
        .max_per_stage_samplers       = uint::max,
        .max_per_stage_resources      = uint::max,
        .max_all_pools_descriptors    = uint::max,
    };
    bool fits = gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 0x8000_0000,
        smp_capacity: 0,
        limits:       limits,
    );
    assert(!fits, "aggregate arithmetic does not wrap at uint width");
}

fn void heap_fits_limits_indexing_exceeds_sampled_images() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_sampled_images = 99,
        .max_storage_images = 100,
        .max_samplers       = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    assert(!gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    ),
        "texture capacity over the sampled-image limit is rejected");
}

fn void heap_fits_limits_indexing_exceeds_storage_images() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_sampled_images = 100,
        .max_storage_images = 99,
        .max_samplers       = 50,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    assert(!gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    ),
        "texture capacity over the storage-image limit is rejected");
}

fn void heap_fits_limits_indexing_exceeds_samplers() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_sampled_images = 100,
        .max_storage_images = 100,
        .max_samplers       = 49,
        .max_per_stage_sampled_images = 100,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 49,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    assert(!gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    ),
        "sampler capacity over the sampler limit is rejected");
}

fn void heap_fits_limits_buffer_at_limit_fits() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size at the limit fits");
}

fn void heap_fits_limits_buffer_exceeds_resource_range() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 2048,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(!gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size over the resource descriptor buffer range is rejected");
}

fn void heap_fits_limits_buffer_exceeds_sampler_range() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 2048,
                .resource_descriptor_address_space_size = 4096,
        .sampler_descriptor_address_space_size  = 4096,
.descriptor_buffer_address_space_size = 4096,
        .layout_size                          = 4096,
    };
    assert(!gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size over the sampler descriptor buffer range is rejected");
}

fn void heap_fits_limits_buffer_exceeds_address_space() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range = 4096,
        .max_sampler_descriptor_buffer_range  = 4096,
                .resource_descriptor_address_space_size = 2048,
        .sampler_descriptor_address_space_size  = 2048,
.descriptor_buffer_address_space_size = 2048,
        .layout_size                          = 4096,
    };
    assert(!gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    ),
        "layout size over the descriptor buffer address space is rejected");
}

fn void heap_fits_limits_indexing_exceeds_per_stage() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_sampled_images           = 100,
        .max_storage_images           = 100,
        .max_samplers                 = 50,
        .max_per_stage_sampled_images = 99,
        .max_per_stage_storage_images = 100,
        .max_per_stage_samplers       = 50,
        .max_per_stage_resources      = 250,
        .max_all_pools_descriptors    = 250,
    };
    bool fits = gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_INDEXING,
        tex_capacity: 100,
        smp_capacity: 50,
        limits:       limits,
    );
    assert(!fits, "per-stage limit binds even when the set-level limit fits");
}

fn void heap_fits_limits_buffer_exceeds_per_type_address_space() @test {
    gpu::compat::vk::HeapLimits limits = {
        .max_resource_descriptor_buffer_range    = 4096,
        .max_sampler_descriptor_buffer_range     = 4096,
        .resource_descriptor_address_space_size  = 4096,
        .sampler_descriptor_address_space_size   = 2048,
        .descriptor_buffer_address_space_size    = 4096,
        .layout_size                             = 4096,
    };
    bool fits = gpu::compat::vk::heap_fits_limits(
        mode:         gpu::compat::DescriptorHeapMode.DESCRIPTOR_BUFFER,
        tex_capacity: 0,
        smp_capacity: 0,
        limits:       limits,
    );
    assert(!fits, "per-type address space binds even when the combined budget fits");
}

// Integration: a capacity above the library default, but still well within
// lavapipe's reported device limits, creates successfully.
fn void descriptor_heap_raised_capacity_creates() @test {
    gpu::compat::Device device = make_heap_test_device(8192, 512);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_texture(&device);
    assert(gpu::compat::get_device_caps(&device)!!.max_texture_descriptors == 8192, "caps report the created texture capacity");
    assert(gpu::compat::get_device_caps(&device)!!.max_sampler_descriptors == 512, "caps report the created sampler capacity");
    defer gpu::compat::destroy_texture(&device, texture)!!;

    gpu::compat::TextureIndex index = gpu::compat::create_texture_descriptor(&device, texture, null)!!;
    assert(index.is_valid(), "raised-capacity heap creates and serves a descriptor");
    gpu::compat::destroy_texture_descriptor(&device, index)!!;
}

fn void descriptor_batch_and_sampler_remaining_failures_report_context() @test {
    gpu::compat::Device device = make_heap_test_device(1, 1);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;
    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    DescriptorDiagnosticCapture capture;
    state.debug_callback = &capture_descriptor_diagnostic;
    state.debug_user_data = &capture;

    gpu::compat::TextureDescriptorDesc[1] one = { { .texture = texture } };
    gpu::compat::TextureIndex[2] too_many_outputs;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::INVALID_ARGUMENT,
        expected_kind:       gpu::compat::DebugResourceKind.TEXTURE_DESCRIPTOR,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_texture_descriptors",
        expected_field:      "out_indices",
        expected_invariant:  "output count must match descriptor count",
    );
    fault lengths = @catch(gpu::compat::create_texture_descriptors(&device, one[..], too_many_outputs[..]));
    assert(lengths == gpu::compat::INVALID_ARGUMENT && capture.calls == 1 && capture.matched,
        "batch length rejection reports exact context");

    gpu::compat::TextureHandle stale_texture = make_heap_texture(&device);
    gpu::compat::destroy_texture(&device, stale_texture)!!;
    gpu::compat::TextureDescriptorDesc[1] stale_descs = { { .texture = stale_texture } };
    gpu::compat::TextureIndex[1] stale_outputs = { { .slot = 19, .generation = 7 } };
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::INVALID_HANDLE,
        expected_kind:       gpu::compat::DebugResourceKind.TEXTURE,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_texture_descriptors",
        expected_field:      "descs[].texture",
        expected_invariant:  "texture must identify a live generation",
    );
    fault stale = @catch(gpu::compat::create_texture_descriptors(&device, stale_descs[..], stale_outputs[..]));
    assert(stale == gpu::compat::INVALID_HANDLE && capture.calls == 2 && capture.matched,
        "batch stale texture rejection omits unvalidated identity");
    assert(stale_outputs[0].slot == 19 && stale_outputs[0].generation == 7,
        "batch stale texture rejection preserves outputs");

    gpu::compat::TextureViewDesc bad_view = { .format = gpu::compat::Format.R32_UINT };
    gpu::compat::TextureDescriptorDesc[1] bad_view_descs = { { .texture = texture, .view = bad_view } };
    gpu::compat::TextureIndex[1] bad_view_outputs = { { .slot = 23, .generation = 9 } };
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::INVALID_ARGUMENT,
        expected_kind:       gpu::compat::DebugResourceKind.TEXTURE,
        expected_identity:   true,
        expected_index:      texture.index,
        expected_generation: texture.generation,
        expected_operation:  "create_texture_descriptors",
        expected_field:      "descs[].view",
        expected_invariant:  "view must select a valid texture subresource range and format",
    );
    fault view = @catch(gpu::compat::create_texture_descriptors(&device, bad_view_descs[..], bad_view_outputs[..]));
    assert(view == gpu::compat::INVALID_ARGUMENT && capture.calls == 3 && capture.matched,
        "batch view rejection reports resolved texture identity");
    assert(bad_view_outputs[0].slot == 23 && bad_view_outputs[0].generation == 9,
        "batch view rejection preserves outputs");

    gpu::compat::TextureDescriptorDesc[2] over_capacity;
    foreach (&desc : over_capacity) *desc = { .texture = texture };
    gpu::compat::TextureIndex[2] capacity_outputs;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::DESCRIPTOR_HEAP_FULL,
        expected_kind:       gpu::compat::DebugResourceKind.TEXTURE_DESCRIPTOR,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_texture_descriptors",
        expected_field:      "DeviceDesc.texture_descriptor_capacity",
        expected_invariant:  "must fit the complete descriptor batch after live and retiring descriptors",
    );
    fault capacity = @catch(gpu::compat::create_texture_descriptors(&device, over_capacity[..], capacity_outputs[..]));
    assert(capacity == gpu::compat::DESCRIPTOR_HEAP_FULL && capture.calls == 4 && capture.matched,
        "batch capacity rejection reports exact context");

    gpu::compat::SamplerDesc invalid_anisotropy = { .anisotropy_enable = true, .max_anisotropy = 0.5f };
    float supported_anisotropy = state.max_sampler_anisotropy;
    if (supported_anisotropy <= 0) state.max_sampler_anisotropy = 16.0f;
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::INVALID_ARGUMENT,
        expected_kind:       gpu::compat::DebugResourceKind.SAMPLER,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_sampler",
        expected_field:      "SamplerDesc.max_anisotropy",
        expected_invariant:  "must be at least 1.0 when anisotropy is enabled",
    );
    fault anisotropy = @catch(gpu::compat::create_sampler(&device, &invalid_anisotropy));
    assert(anisotropy == gpu::compat::INVALID_ARGUMENT && capture.calls == 5 && capture.matched,
        "invalid sampler anisotropy reports exact context");

    state.max_sampler_anisotropy = 0;
    gpu::compat::SamplerDesc unsupported_anisotropy = { .anisotropy_enable = true, .max_anisotropy = 1.0f };
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::UNSUPPORTED_FEATURE,
        expected_kind:       gpu::compat::DebugResourceKind.SAMPLER,
        expected_identity:   false,
        expected_index:      0,
        expected_generation: 0,
        expected_operation:  "create_sampler",
        expected_field:      "SamplerDesc.anisotropy_enable",
        expected_invariant:  "device must support sampler anisotropy",
    );
    fault unsupported = @catch(gpu::compat::create_sampler(&device, &unsupported_anisotropy));
    assert(unsupported == gpu::compat::UNSUPPORTED_FEATURE && capture.calls == 6 && capture.matched,
        "unsupported sampler anisotropy reports exact context");
    state.max_sampler_anisotropy = supported_anisotropy;
    state.debug_callback = null;
}

fn void texture_descriptor_view_cache_exhaustion_reports_distinct_context() @test {
    gpu::compat::Device device = make_heap_test_device(16, 4);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::TextureHandle texture = make_heap_view_texture(&device);
    defer gpu::compat::destroy_texture(&device, texture)!!;
    gpu::compat::TextureIndex[vk::MAX_CACHED_VIEWS] live;
    for (uint i = 0; i < vk::MAX_CACHED_VIEWS; i++) {
        gpu::compat::TextureViewDesc view = {
            .base_mip    = i % 4,
            .mip_count   = 1,
            .base_layer  = i / 4,
            .layer_count = 1,
        };
        live[i] = gpu::compat::create_texture_descriptor(&device, texture, &view)!!;
    }

    vk::VkDeviceState* state = (vk::VkDeviceState*)gpu::compat::device_backend_state(device);
    DescriptorDiagnosticCapture capture;
    state.debug_callback = &capture_descriptor_diagnostic;
    state.debug_user_data = &capture;
    gpu::compat::TextureViewDesc overflow = {
        .base_mip    = 4,
        .mip_count   = 1,
        .base_layer  = 3,
        .layer_count = 1,
    };
    expect_descriptor_diagnostic(
        capture:             &capture,
        expected_fault:      gpu::compat::SLOT_TABLE_FULL,
        expected_kind:       gpu::compat::DebugResourceKind.TEXTURE,
        expected_identity:   true,
        expected_index:      texture.index,
        expected_generation: texture.generation,
        expected_operation:  "create_texture_descriptor",
        expected_field:      "texture view cache",
        expected_invariant:  "texture must have free cached-view capacity",
    );
    fault full = @catch(gpu::compat::create_texture_descriptor(&device, texture, &overflow));
    assert(full == gpu::compat::SLOT_TABLE_FULL && capture.calls == 1 && capture.matched,
        "view-cache exhaustion has distinct structured context");

    state.debug_callback = null;
    foreach (index : live) gpu::compat::destroy_texture_descriptor(&device, index)!!;
}

fn void descriptor_bootstrap_result_mappings_emit_once_and_preserve_faults() @test {
    vk::VkDeviceState state;
    DescriptorBackendCapture capture;
    state.debug_callback = &capture_descriptor_backend_diagnostic;
    state.debug_user_data = &capture;

    capture = {
        .expected_fault     = gpu::compat::DESCRIPTOR_HEAP_FULL,
        .expected_operation = "create_device",
        .expected_text      = "Vulkan descriptor pool is fragmented",
        .expected_result    = (int)vk::Result.ERROR_FRAGMENTATION,
    };
    fault pool = @catch(gpu::compat::vk::check_descriptor_pool_backend_result(
        state:     &state,
        operation: "create_device",
        result:    vk::Result.ERROR_FRAGMENTATION,
    ));
    assert(pool == gpu::compat::DESCRIPTOR_HEAP_FULL && capture.calls == 1 && capture.matched,
        "descriptor-pool specialization is reported without changing its fault");

    capture = {
        .calls              = capture.calls,
        .expected_fault     = gpu::compat::DESCRIPTOR_HEAP_FULL,
        .expected_operation = "create_device",
        .expected_text      = "Vulkan descriptor allocation exhausted the pool",
        .expected_result    = (int)vk::Result.ERROR_OUT_OF_POOL_MEMORY,
    };
    fault allocate = @catch(gpu::compat::vk::check_descriptor_allocate_backend_result(
        state:     &state,
        operation: "create_device",
        result:    vk::Result.ERROR_OUT_OF_POOL_MEMORY,
    ));
    assert(allocate == gpu::compat::DESCRIPTOR_HEAP_FULL && capture.calls == 2 && capture.matched,
        "descriptor allocation specialization is reported without changing its fault");

    capture = {
        .calls              = capture.calls,
        .expected_fault     = gpu::compat::BACKEND_ERROR,
        .expected_operation = "create_device",
        .expected_text      = "descriptor buffer device address is zero",
        .expected_result    = 0,
    };
    fault address = @catch(gpu::compat::vk::report_descriptor_backend_failure(
        state:         &state,
        operation:     "create_device",
        public_fault:  gpu::compat::BACKEND_ERROR,
        backend_text:  "descriptor buffer device address is zero",
        backend_result: 0,
    ));
    assert(address == gpu::compat::BACKEND_ERROR && capture.calls == 3 && capture.matched,
        "descriptor bootstrap failure without VkResult is observable");

    fault silent_retire = @catch(gpu::compat::vk::check_descriptor_retire_backend_result(
        state:     &state,
        operation: null,
        result:    vk::Result.ERROR_DEVICE_LOST,
    ));
    assert(silent_retire == gpu::compat::DEVICE_LOST && capture.calls == 3,
        "silent single-descriptor retire fallback does not emit a paired fault");

    capture = {
        .calls              = capture.calls,
        .expected_fault     = gpu::compat::DEVICE_LOST,
        .expected_operation = "create_texture_descriptors",
        .expected_text      = "Vulkan device lost",
        .expected_result    = (int)vk::Result.ERROR_DEVICE_LOST,
    };
    fault batch_retire = @catch(gpu::compat::vk::check_descriptor_retire_backend_result(
        state:     &state,
        operation: "create_texture_descriptors",
        result:    vk::Result.ERROR_DEVICE_LOST,
    ));
    assert(batch_retire == gpu::compat::DEVICE_LOST && capture.calls == 4 && capture.matched,
        "batch texture-retire query reports its backend fault once");

    gpu::compat::TextureHandle texture = { .index = 7, .generation = 11 };
    vk::TextureSlot texture_slot;
    capture = {
        .calls               = capture.calls,
        .expected_fault      = gpu::compat::OUT_OF_DEVICE_MEMORY,
        .expected_kind       = gpu::compat::DebugResourceKind.TEXTURE,
        .expected_identity   = true,
        .expected_index      = texture.index,
        .expected_generation = texture.generation,
        .expected_operation  = "create_texture_descriptor",
        .expected_text       = "Vulkan device memory exhausted",
        .expected_result     = (int)vk::Result.ERROR_OUT_OF_DEVICE_MEMORY,
    };
    fault view_backend = @catch(gpu::compat::vk::check_texture_view_backend_result(
        state:     &state,
        operation: "create_texture_descriptor",
        texture:   texture,
        slot:      &texture_slot,
        result:    vk::Result.ERROR_OUT_OF_DEVICE_MEMORY,
    ));
    assert(view_backend == gpu::compat::OUT_OF_DEVICE_MEMORY && capture.calls == 5 && capture.matched,
        "image-view creation reports operation and resolved texture identity");

    state.debug_callback = null;
    fault fallback = @catch(gpu::compat::vk::check_descriptor_pool_backend_result(
        state:     &state,
        operation: "create_device",
        result:    vk::Result.ERROR_FRAGMENTATION,
    ));
    assert(fallback == pool && capture.calls == 5,
        "null callback preserves descriptor bootstrap fault without a duplicate");
}
