module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import std::core::mem;
import std::core::mem::alloc;

alias ConfinedThreadAllocatorFn @if(env::TESTING) = fn void? (void*);

struct OwnedSpan {
    gpu::GpuAllocation allocation;
    gpu::GpuSpan       span;
}

fn void? observe_confined_thread_allocator(
    vk::RecordingHostAllocator* allocator,
    ConfinedThreadAllocatorFn callback,
    void* user_data,
) @if(env::TESTING) {
    Allocator previous = mem;
    mem = allocator;
    defer mem = previous;
    callback(user_data)!;
}

fn gpu::Device? create_test_device(
    gpu::RuntimeDesc* desc,
) {
    if (desc == null) return gpu::INVALID_ARGUMENT~;
    gpu::Runtime runtime = gpu::create_runtime(desc)!;
    defer catch (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::DeviceRequest request = gpu::strict_device_request();
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!;
        gpu::DeviceRequestSupport support =
            gpu::supports_device_request(&adapter, &request)!;
        if (!support.supported) continue;
        gpu::Device? device = gpu::create_device(&adapter, &request);
        if (catch err = device) {
            if (err == gpu::UNSUPPORTED_FEATURE) continue;
            return err~;
        }
        return device;
    }
    return gpu::UNSUPPORTED_FEATURE~;
}

fn gpu::Device? create_dynamic_color_test_device(
    gpu::RuntimeDesc* desc,
) {
    if (desc == null) return gpu::INVALID_ARGUMENT~;
    gpu::Runtime runtime = gpu::create_runtime(desc)!;
    defer catch (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::DeviceRequest request = gpu::request_dynamic_color_state(
        gpu::strict_device_request(),
    )!;
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!;
        gpu::DeviceRequestSupport support =
            gpu::supports_device_request(&adapter, &request)!;
        if (!support.supported) continue;
        gpu::Device? device = gpu::create_device(&adapter, &request);
        if (catch err = device) {
            if (err == gpu::UNSUPPORTED_FEATURE) continue;
            return err~;
        }
        return device;
    }
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void bind_test_compute_pipeline(
    gpu::CommandList* commands,
    gpu::PipelineHandle pipeline,
) {
    gpu::cmd_bind_pipeline(commands, pipeline)!!;
}

fn void bind_test_graphics_pipeline(
    gpu::CommandList* commands,
    gpu::PipelineHandle pipeline,
) {
    gpu::cmd_bind_pipeline(commands, pipeline)!!;
}

fn OwnedSpan? allocate_owned_span(
    gpu::Device* device,
    usz size,
    gpu::MemoryClass memory_class,
    gpu::QueueRoles access,
    ZString debug_name,
) {
    gpu::AllocationDesc desc = {
        .size         = size,
        .alignment    = 16,
        .memory_class = memory_class,
        .access       = access,
        .debug_name   = debug_name,
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(device, &desc)!;
    defer catch (void)gpu::free_allocation(device, &allocation);
    gpu::GpuSpan span = gpu::get_allocation_span(device, allocation)!;
    return { .allocation = allocation, .span = span };
}

fn void? free_owned_span(gpu::Device* device, OwnedSpan* owned) {
    if (owned == null || !owned.allocation.is_valid()) return;
    gpu::free_allocation(device, &owned.allocation)!;
    owned.span = {};
}

fn char[] test_span_mapping(gpu::Device* device, gpu::GpuSpan span) {
    return gpu::get_span_mapping(device, span)!!;
}

fn gpu::GpuAddress test_span_address(gpu::Device* device, gpu::GpuSpan span) {
    return gpu::get_span_address(device, span)!!;
}

fn void? discard_test_commands(gpu::CommandList* commands) {
    if (commands == null || !commands.is_valid()) return;
    if (catch err = gpu::discard_commands(commands)) {
        if (err != gpu::INVALID_HANDLE) return err~;
    }
}

fn void? discard_test_executable(gpu::ExecutableCommandList* commands) {
    if (commands == null || !commands.is_valid()) return;
    if (catch err = gpu::discard_executable_commands(commands)) {
        if (err != gpu::INVALID_HANDLE) return err~;
    }
}

fn void release_submitted_executables(
    gpu::internal::DeviceOperation* operation,
    gpu::ExecutableCommandList[] commands,
) {
    (void)operation;
    foreach (&command : commands) {
        *command = {};
    }
}

fn gpu::AttachmentViewHandle? create_test_attachment_view(
    gpu::Device* device,
    gpu::TextureHandle texture,
    uint mip_level = 0,
    uint array_layer = 0,
) {
    gpu::AttachmentViewDesc desc = {
        .texture     = texture,
        .mip_level   = mip_level,
        .array_layer = array_layer,
    };
    return gpu::create_attachment_view(device, &desc);
}

fn void? destroy_test_attachment_view(
    gpu::Device* device,
    gpu::AttachmentViewHandle* view,
) {
    if (view == null || !view.is_valid()) return;
    gpu::destroy_attachment_view(device, *view)!;
    *view = {};
}

fn gpu::CompletionPoint? submit_test_queue(
    gpu::Device* device,
    gpu::QueueKind kind,
    gpu::SubmitDesc* desc,
) {
    return gpu::submit(gpu::get_queue(device, kind)!, desc);
}

fn void? wait_test_queue(gpu::Device* device, gpu::QueueKind queue) {
    gpu::Queue target = gpu::get_queue(device, queue)!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        device,
        target,
    )!;
    defer (void)gpu::destroy_command_allocator(&allocator);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!;
    defer (void)gpu::discard_commands(&commands);
    gpu::ExecutableCommandList[1] lists = { gpu::end_commands(&commands)! };
    defer (void)gpu::discard_executable_commands(&lists[0]);
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(target, &submit)!;
    gpu::wait_completion(completion)!;
}

fn void? destroy_test_device(gpu::Device* device) {
    if (device == null || !device.is_valid()) return;
    gpu::Runtime runtime;
    gpu::internal::DeviceOperation operation = gpu::internal::begin_device_operation(device, true)!;
    if (operation.data.runtime_retained) runtime = operation.data.runtime;
    gpu::internal::end_device_operation(&operation);
    gpu::QueueCounts? counts_result = gpu::get_queue_counts(device);
    if (catch err = counts_result) {
        if (err == gpu::DEVICE_LOST) {
            gpu::destroy_device(device)!;
            if (runtime.is_valid()) gpu::destroy_runtime(&runtime)!;
            return;
        }
        return err~;
    }
    gpu::QueueCounts counts = counts_result;
    if (counts.graphics > 0) {
        wait_test_queue(device, gpu::QueueKind.GRAPHICS)!;
    }
    if (counts.compute > 0) {
        wait_test_queue(device, gpu::QueueKind.COMPUTE)!;
    }
    if (counts.transfer > 0) {
        wait_test_queue(device, gpu::QueueKind.TRANSFER)!;
    }
    gpu::destroy_device(device)!;
    if (runtime.is_valid()) gpu::destroy_runtime(&runtime)!;
}

fn void? upload_test_texture_data(
    gpu::Device* device,
    gpu::TextureHandle dst,
    uint mip,
    char[] data,
    gpu::TextureState next_use,
) {
    OwnedSpan upload = allocate_owned_span(
        device:       device,
        size:         data.len,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "test_texture_upload",
    )!;
    defer (void)free_owned_span(device, &upload);
    char[] mapping = gpu::get_span_mapping(device, upload.span)!;
    mapping[..] = data[..];
    gpu::flush_mapped_span(device, upload.span)!;

    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        device,
        queue,
    )!;
    defer (void)gpu::destroy_command_allocator(&allocator);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!;
    defer (void)discard_test_commands(&commands);
    gpu::TextureViewDesc view = {
        .base_mip  = mip,
        .mip_count = 1,
    };
    gpu::TextureBarrier to_transfer = gpu::texture_view_transition(
        texture: dst,
        view:    view,
        before:  { .layout = gpu::TextureLayout.UNDEFINED },
        after:   {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    )!;
    gpu::cmd_texture_barrier(&commands, &to_transfer)!;
    gpu::BufferTextureCopyDesc copy = {
        .src     = upload.span,
        .texture = dst,
        .mip     = mip,
    };
    gpu::cmd_copy_buffer_to_texture(&commands, &copy)!;
    gpu::TextureBarrier transition = gpu::texture_view_transition(
        texture: dst,
        view:    view,
        before:  {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
        after:   next_use,
    )!;
    gpu::cmd_texture_barrier(&commands, &transition)!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!;
    defer (void)discard_test_executable(&executable);
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!;
    gpu::wait_completion(completion)!;
}

fn void? readback_test_texture_data(
    gpu::Device* device,
    gpu::TextureHandle src,
    uint mip,
    char[] out_data,
    gpu::TextureState from_use,
) {
    OwnedSpan readback = allocate_owned_span(
        device:       device,
        size:         out_data.len,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "test_texture_readback",
    )!;
    defer (void)free_owned_span(device, &readback);

    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.GRAPHICS)!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(
        device,
        queue,
    )!;
    defer (void)gpu::destroy_command_allocator(&allocator);
    gpu::CommandList commands = gpu::begin_commands(&allocator)!;
    defer (void)discard_test_commands(&commands);
    gpu::TextureViewDesc view = {
        .base_mip  = mip,
        .mip_count = 1,
    };
    gpu::TextureBarrier to_transfer = gpu::texture_view_transition(
        texture: src,
        view:    view,
        before:  from_use,
        after:   {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
    )!;
    gpu::cmd_texture_barrier(&commands, &to_transfer)!;
    gpu::TextureBufferCopyDesc copy = {
        .texture = src,
        .dst     = readback.span,
        .mip     = mip,
    };
    gpu::cmd_copy_texture_to_buffer(&commands, &copy)!;
    gpu::Barrier host_read = {
        .before = { .transfer = true },
        .after  = { .host = true },
    };
    gpu::cmd_barrier(&commands, &host_read)!;
    gpu::TextureBarrier restore = gpu::texture_view_transition(
        texture: src,
        view:    view,
        before:  {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
        after:   from_use,
    )!;
    gpu::cmd_texture_barrier(&commands, &restore)!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!;
    defer (void)discard_test_executable(&executable);
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!;
    gpu::wait_completion(completion)!;

    gpu::invalidate_mapped_span(device, readback.span)!;
    char[] mapping = gpu::get_span_mapping(device, readback.span)!;
    out_data[..] = mapping[..];
}

fn void assert_warm_command_resolution(
    gpu::CommandResolutionStats stats,
    ulong expected_commands,
    ulong expected_pipeline_lookups = 0,
) @if(env::TESTING) {
    assert(stats.device_registry_resolutions == 0,
        "warm commands do not resolve the device registry");
    assert(stats.retained_pin_operations == 0,
        "warm commands do not borrow retained pins");
    assert(stats.command_table_lookups == 0,
        "warm commands use the stable record directly");
    assert(stats.pipeline_table_lookups == expected_pipeline_lookups,
        "warm commands perform the expected pipeline table lookups");
    assert(stats.pipeline_cache_lookups == expected_pipeline_lookups,
        "warm commands perform the expected pipeline cache lookups");
    assert(stats.command_policy_selections == 0,
        "warm commands do not select a command policy");
    assert(stats.encoder_cell_computations == 0,
        "warm commands perform no frontend encoder-cell computation");
    assert(stats.encoder_lease_comparisons == 0,
        "warm commands perform no packed-lease comparison");
    assert(stats.recording_commands == expected_commands,
        "the command counter observes each warm recording call");
}

fn void assert_strict_heap_bind_work(
    vk::VkDeviceState* state,
    gpu::CommandResolutionStats stats,
    ulong expected_bind_points,
    ulong expected_buffer_binds = 1,
) @if(env::TESTING) {
    if (state.strict_heap.implementation
        == vk::VkHeapImplementation.DESCRIPTOR_INDEXING) {
        assert(stats.strict_heap_set_bind_commands == expected_bind_points,
            "indexing binds the strict set once per used bind point");
        assert(stats.strict_heap_buffer_bind_commands == 0
            && stats.strict_heap_offset_commands == 0,
            "indexing emits no descriptor-buffer commands");
        return;
    }
    assert(state.strict_heap.implementation
        == vk::VkHeapImplementation.DESCRIPTOR_BUFFER,
        "a strict test device selects a descriptor heap implementation");
    assert(stats.strict_heap_set_bind_commands == 0,
        "descriptor-buffer mode emits no descriptor-set binds");
    assert(stats.strict_heap_buffer_bind_commands == expected_buffer_binds,
        "descriptor-buffer mode emits the expected global buffer binds");
    assert(stats.strict_heap_offset_commands == expected_bind_points,
        "descriptor-buffer mode sets one offset per used bind point");
}
