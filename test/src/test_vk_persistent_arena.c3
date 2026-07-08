module gpu_test;

import gpu;

// Headless (lavapipe): persistent arena alloc/free/reuse, alignment, stable
// addresses, stats, and usage honoring (index/indirect/transfer spans usable,
// vertex-bit requests fault).

const char[*] PA_DOUBLE_SPIRV = $embed("shaders/root_pointer.comp.spv");

fn void persistent_alloc_free_stats() @test {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "gpu_persistent_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::PersistentArenaStats empty = gpu::get_persistent_stats(&device)!!;
    assert(empty.used_bytes == 0, "arena starts empty");
    assert(empty.capacity_bytes == gpu::DEFAULT_PERSISTENT_ARENA_SIZE, "capacity is the arena size");

    gpu::PersistentAllocDesc desc_a = {
        .size = 256, .align = 64, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "a",
    };
    gpu::GpuSpan a = gpu::alloc_persistent_span(&device, &desc_a)!!;
    assert(a.cpu != null, "persistent span is mapped");
    assert((ulong)a.gpu != 0, "non-zero GPU address");
    assert(a.offset % 64 == 0, "offset honors alignment");
    ulong a_gpu = (ulong)a.gpu;

    gpu::PersistentArenaStats after_a = gpu::get_persistent_stats(&device)!!;
    assert(after_a.used_bytes >= 256, "used bytes rose");
    assert(after_a.allocation_count == 1, "one live allocation");

    gpu::PersistentAllocDesc desc_b = { .size = 512, .align = 256, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD };
    gpu::GpuSpan b = gpu::alloc_persistent_span(&device, &desc_b)!!;
    assert((ulong)b.gpu != a_gpu, "B gets a distinct address");
    assert(gpu::get_persistent_stats(&device)!!.allocation_count == 2, "two live allocations");

    // Free B; A's address is unchanged, count drops.
    gpu::free_persistent_span(&device, b)!!;
    gpu::PersistentArenaStats after_free_b = gpu::get_persistent_stats(&device)!!;
    assert(after_free_b.allocation_count == 1, "back to one after free");
    assert(after_free_b.used_bytes < after_a.used_bytes + 512, "used dropped after free");
    assert((ulong)a.gpu == a_gpu, "A's GPU address is stable");

    // Reuse: C allocates into freed space, A still distinct/stable.
    gpu::PersistentAllocDesc desc_c = { .size = 128, .align = 16, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD };
    gpu::GpuSpan c = gpu::alloc_persistent_span(&device, &desc_c)!!;
    assert((ulong)c.gpu != a_gpu, "C distinct from live A");

    gpu::free_persistent_span(&device, a)!!;
    gpu::free_persistent_span(&device, c)!!;
    gpu::PersistentArenaStats drained = gpu::get_persistent_stats(&device)!!;
    assert(drained.allocation_count == 0, "arena empty after freeing all");
    assert(drained.used_bytes == 0, "used back to 0");

    // Wrong memory kind faults.
    gpu::PersistentAllocDesc bad_kind = { .size = 64, .align = 16, .memory_kind = gpu::MemoryKind.DEVICE };
    fault kind_fault = @catch(gpu::alloc_persistent_span(&device, &bad_kind));
    assert(kind_fault == gpu::INVALID_ARGUMENT, "non-persistent memory kind faults INVALID_ARGUMENT");

    // Bogus free faults.
    gpu::GpuSpan bogus = { .buffer = a.buffer, .offset = 999_999 };
    fault free_fault = @catch(gpu::free_persistent_span(&device, bogus));
    assert(free_fault == gpu::INVALID_HANDLE, "freeing a bogus span faults INVALID_HANDLE");
}

fn void custom_persistent_arena_size_reported() @test {
    gpu::DeviceDesc desc = {
        .backend               = gpu::BackendKind.VULKAN,
        .enable_validation     = true,
        .frames_in_flight      = 2,
        .persistent_arena_size = 128 << 20,
        .application_name      = "gpu_persistent_size_test",
    };
    gpu::Device device = gpu::create_device(&desc)!!;
    defer gpu::destroy_device(&device)!!;

    gpu::PersistentArenaStats stats = gpu::get_persistent_stats(&device)!!;
    assert(stats.capacity_bytes == 128 << 20, "custom persistent capacity reported in stats");
}

fn gpu::Device make_persistent_usage_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_persistent_usage_test",
    };
    return gpu::create_device(&desc)!!;
}

<* A transfer-usage persistent span is a real cmd_copy_buffer endpoint: two
   spans backed by the same arena buffer, copied disjointly. *>
fn void transfer_usage_persistent_span_copies() @test {
    gpu::Device device = make_persistent_usage_device();
    defer gpu::destroy_device(&device)!!;

    gpu::PersistentAllocDesc src_desc = {
        .size = 64, .align = 16, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "src",
    };
    gpu::GpuSpan src = gpu::alloc_persistent_span(&device, &src_desc)!!;
    defer gpu::free_persistent_span(&device, src)!!;
    gpu::PersistentAllocDesc dst_desc = {
        .size = 64, .align = 16, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "dst",
    };
    gpu::GpuSpan dst = gpu::alloc_persistent_span(&device, &dst_desc)!!;
    defer gpu::free_persistent_span(&device, dst)!!;

    char* src_bytes = (char*)src.cpu;
    for (usz i = 0; i < 64; i++) src_bytes[i] = (char)(i % 251);
    gpu::flush_buffer(&device, src.buffer, src.offset, src.size)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::BufferCopyDesc copy = { .src = src.buffer, .dst = dst.buffer, .src_offset = src.offset, .dst_offset = dst.offset, .size = 64 };
    gpu::cmd_copy_buffer(&cmd, &copy)!!;
    gpu::BufferBarrier host_read = {
        .buffer = dst.buffer, .offset = dst.offset, .size = dst.size,
        .before_stage = gpu::Stage.TRANSFER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.TRANSFER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&device)!!;

    gpu::invalidate_buffer(&device, dst.buffer, dst.offset, dst.size)!!;
    char* dst_bytes = (char*)dst.cpu;
    for (usz i = 0; i < 64; i++) {
        assert(dst_bytes[i] == (char)(i % 251), "transfer-usage persistent span copies intact bytes");
    }
}

<* resolve_indirect_span rejects buffers without the indirect bit, so this
   fails pre-fix and passes only once the arena carries the superset. *>
fn void indirect_usage_persistent_span_dispatches() @test {
    gpu::Device device = make_persistent_usage_device();
    defer gpu::destroy_device(&device)!!;

    gpu::ShaderDesc shader_desc = { .stage = gpu::ShaderStage.COMPUTE, .spirv = PA_DOUBLE_SPIRV[..], .entry_point = "main" };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &shader_desc)!!;
    defer gpu::destroy_shader(&device, shader)!!;
    gpu::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pipe_desc)!!;
    defer gpu::destroy_pipeline(&device, pipeline)!!;

    const uint COUNT = 4;
    gpu::PersistentAllocDesc io_desc = {
        .size = COUNT * 4 * 2, .align = 16, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "io",
    };
    gpu::GpuSpan io = gpu::alloc_persistent_span(&device, &io_desc)!!;
    defer gpu::free_persistent_span(&device, io)!!;
    float* values = (float*)io.cpu;
    for (uint i = 0; i < COUNT; i++) values[i] = (float)i;

    gpu::PersistentAllocDesc root_desc = {
        .size = ComputeRoot::size, .align = 16, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "root",
    };
    gpu::GpuSpan root_span = gpu::alloc_persistent_span(&device, &root_desc)!!;
    defer gpu::free_persistent_span(&device, root_span)!!;
    ComputeRoot* root = (ComputeRoot*)root_span.cpu;
    root.input_gpu  = io.gpu;
    root.output_gpu = (gpu::GpuAddress)((ulong)io.gpu + COUNT * 4);
    root.count      = COUNT;

    gpu::PersistentAllocDesc args_desc = {
        .size = gpu::DispatchIndirectCommand::size, .align = 4, .usage = { .indirect }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD, .debug_name = "dispatch_args",
    };
    gpu::GpuSpan args_span = gpu::alloc_persistent_span(&device, &args_desc)!!;
    defer gpu::free_persistent_span(&device, args_span)!!;
    gpu::DispatchIndirectCommand* args = (gpu::DispatchIndirectCommand*)args_span.cpu;
    *args = { .x = 1, .y = 1, .z = 1 };
    gpu::flush_buffer(&device, args_span.buffer, args_span.offset, args_span.size)!!;
    gpu::flush_buffer(&device, root_span.buffer, root_span.offset, root_span.size)!!;
    gpu::flush_buffer(&device, io.buffer, io.offset, io.size)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::cmd_dispatch_indirect(
        commands: &cmd,
        pipeline: pipeline,
        root:     root_span.gpu,
        args:     args_span,
    )!!;
    gpu::BufferBarrier host_read = {
        .buffer = io.buffer, .offset = io.offset, .size = io.size,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &host_read)!!;
    gpu::end_commands(&device, &cmd)!!;
    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(&device, &submit)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_frame(&device)!!;

    gpu::invalidate_buffer(&device, io.buffer, io.offset, io.size)!!;
    float* doubled = &values[COUNT];
    for (uint i = 0; i < COUNT; i++) {
        assert(doubled[i] == (float)i * 2.0f, "indirect-usage persistent span drove cmd_dispatch_indirect");
    }
}

<* Usage bits outside the arena's superset fault at allocation time instead
   of surfacing as a Vulkan validation error at bind time. *>
fn void unsupported_usage_persistent_span_faults() @test {
    gpu::Device device = make_persistent_usage_device();
    defer gpu::destroy_device(&device)!!;

    gpu::PersistentAllocDesc vertex_desc = {
        .size = 64, .align = 16, .usage = { .vertex }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
    };
    fault vertex_fault = @catch(gpu::alloc_persistent_span(&device, &vertex_desc));
    assert(vertex_fault == gpu::INVALID_ARGUMENT, "vertex-bit usage request faults INVALID_ARGUMENT");

    gpu::PersistentAllocDesc shared_desc = {
        .size = 64, .align = 16, .usage = { .shared_queues }, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
    };
    fault shared_fault = @catch(gpu::alloc_persistent_span(&device, &shared_desc));
    assert(shared_fault == gpu::INVALID_ARGUMENT, "shared_queues usage request faults INVALID_ARGUMENT");

    // Empty usage stays valid (storage-style default).
    gpu::PersistentAllocDesc empty_desc = { .size = 64, .align = 16, .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD };
    gpu::GpuSpan empty_span = gpu::alloc_persistent_span(&device, &empty_desc)!!;
    gpu::free_persistent_span(&device, empty_span)!!;
}
