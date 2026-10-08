module gpu_test;

import gpu;

const uint COPY_RGBA_SIZE = 4;
const usz COPY_RGBA_TEXEL_BYTES = 4;
const usz COPY_RGBA_MIP0_BYTES = COPY_RGBA_SIZE * COPY_RGBA_SIZE * COPY_RGBA_TEXEL_BYTES;
const usz COPY_RGBA_MIP1_BYTES = (COPY_RGBA_SIZE / 2) * (COPY_RGBA_SIZE / 2) * COPY_RGBA_TEXEL_BYTES;

const uint COPY_BC_SOURCE_SIZE = 64;
const uint COPY_BC_SOURCE_MIPS = 7;
const uint COPY_BC_DEST_SIZE = 8;
const uint COPY_BC_DEST_MIPS = 4;
const uint COPY_BC_LEVEL_SHIFT = 3;
const usz COPY_BC_BLOCK_BYTES = 8;
// Source mip footprints in blocks: 16x16, 8x8, 4x4, 2x2, 1x1, 1x1, 1x1.
const usz[COPY_BC_SOURCE_MIPS] COPY_BC_SOURCE_BLOCKS = { 256, 64, 16, 4, 1, 1, 1 };
const usz COPY_BC_SOURCE_TOTAL_BLOCKS = 343;

fn gpu::Device make_copy_test_device() {
    gpu::RuntimeDesc desc = {
        .enable_vulkan_validation = true,
        .application_name  = "gpu_texture_copy_test",
    };
    return create_test_device(&desc)!!;
}

fn gpu::TextureState copy_write_state() {
    return {
        .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
        .stages = { .transfer },
        .access = { .write },
    };
}

fn gpu::TextureState copy_read_state() {
    return {
        .layout = gpu::TextureLayout.TRANSFER_SOURCE,
        .stages = { .transfer },
        .access = { .read },
    };
}

fn void copy_transition(
    gpu::CommandList* commands,
    gpu::TextureHandle texture,
    gpu::TextureState before,
    gpu::TextureState after,
) {
    gpu::TextureBarrier barrier = gpu::texture_transition(texture, before, after)!!;
    gpu::cmd_texture_barrier(commands, &barrier)!!;
}

fn void submit_copy_commands(gpu::Device* device, gpu::CommandList* commands) {
    gpu::ExecutableCommandList executable = gpu::end_commands(commands)!!;
    defer discard_test_executable(&executable)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = submit_test_queue(
        device,
        gpu::QueueKind.COMPUTE,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
}

fn usz copy_bc_source_offset(uint mip) {
    usz blocks;
    for (uint level = 0; level < mip; level++) blocks += COPY_BC_SOURCE_BLOCKS[level];
    return blocks * COPY_BC_BLOCK_BYTES;
}

fn usz copy_bc_dest_offset(uint mip) {
    return copy_bc_source_offset(mip + COPY_BC_LEVEL_SHIFT) - copy_bc_source_offset(COPY_BC_LEVEL_SHIFT);
}

fn char copy_bc_pattern(usz index) {
    return (char)((index * 37 + 11) & 0xFF);
}

fn void copy_texture_moves_mip_between_textures() @test {
    gpu::Device device = make_copy_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc source_desc = {
        .access     = { .graphics, .compute, .transfer },
        .width      = COPY_RGBA_SIZE,
        .height     = COPY_RGBA_SIZE,
        .mip_levels = 2,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .transfer_src, .transfer_dst },
        .debug_name = "copy_source",
    };
    gpu::TextureHandle source = gpu::create_texture(&device, &source_desc)!!;
    defer gpu::destroy_texture(&device, source)!!;
    gpu::TextureDesc dest_desc = source_desc;
    dest_desc.width = COPY_RGBA_SIZE / 2;
    dest_desc.height = COPY_RGBA_SIZE / 2;
    dest_desc.mip_levels = 1;
    dest_desc.debug_name = "copy_destination";
    gpu::TextureHandle dest = gpu::create_texture(&device, &dest_desc)!!;
    defer gpu::destroy_texture(&device, dest)!!;

    OwnedSpan staging = allocate_owned_span(
        device:       &device,
        size:         COPY_RGBA_MIP0_BYTES + COPY_RGBA_MIP1_BYTES,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "copy_staging",
    )!!;
    defer free_owned_span(&device, &staging)!!;
    char* bytes = (char*)test_span_mapping(&device, staging.span).ptr;
    for (usz i = 0; i < COPY_RGBA_MIP0_BYTES; i++) bytes[i] = (char)(i + 1);
    for (usz i = 0; i < COPY_RGBA_MIP1_BYTES; i++) {
        bytes[COPY_RGBA_MIP0_BYTES + i] = (char)(200 - i);
    }
    gpu::flush_mapped_span(&device, staging.span)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         COPY_RGBA_MIP1_BYTES,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "copy_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;

    copy_transition(&cmd, source, { .layout = gpu::TextureLayout.UNDEFINED }, copy_write_state());
    copy_transition(&cmd, dest, { .layout = gpu::TextureLayout.UNDEFINED }, copy_write_state());
    gpu::BufferTextureCopyDesc upload_mip0 = {
        .src     = staging.span.checked_subspan(0, COPY_RGBA_MIP0_BYTES)!!,
        .texture = source,
        .mip     = 0,
    };
    gpu::cmd_copy_buffer_to_texture(&cmd, &upload_mip0)!!;
    gpu::BufferTextureCopyDesc upload_mip1 = {
        .src     = staging.span.checked_subspan(COPY_RGBA_MIP0_BYTES, COPY_RGBA_MIP1_BYTES)!!,
        .texture = source,
        .mip     = 1,
    };
    gpu::cmd_copy_buffer_to_texture(&cmd, &upload_mip1)!!;
    copy_transition(&cmd, source, copy_write_state(), copy_read_state());

    gpu::TextureCopyDesc copy = {
        .src     = source,
        .src_mip = 1,
        .dst     = dest,
        .dst_mip = 0,
    };
    gpu::cmd_copy_texture(&cmd, &copy)!!;

    copy_transition(&cmd, dest, copy_write_state(), copy_read_state());
    gpu::TextureBufferCopyDesc download = {
        .texture = dest,
        .dst     = readback.span,
    };
    gpu::cmd_copy_texture_to_buffer(&cmd, &download)!!;
    gpu::Barrier host_read = {
        .before = { .transfer },
        .after  = { .host },
    };
    gpu::cmd_barrier(&cmd, &host_read)!!;
    submit_copy_commands(&device, &cmd);

    gpu::invalidate_mapped_span(&device, readback.span)!!;
    char* read = (char*)test_span_mapping(&device, readback.span).ptr;
    for (usz i = 0; i < COPY_RGBA_MIP1_BYTES; i++) {
        assert(read[i] == bytes[COPY_RGBA_MIP0_BYTES + i],
            "copied mip bytes equal the uploaded source mip");
    }
}

fn void copy_texture_moves_block_compressed_tail_mips() @test {
    gpu::Device device = make_copy_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc source_desc = {
        .access     = { .graphics, .compute, .transfer },
        .width      = COPY_BC_SOURCE_SIZE,
        .height     = COPY_BC_SOURCE_SIZE,
        .mip_levels = COPY_BC_SOURCE_MIPS,
        .format     = gpu::Format.BC1_RGBA_UNORM,
        .usage      = { .transfer_src, .transfer_dst },
        .debug_name = "copy_bc_source",
    };
    if (!gpu::supports_texture_desc(&device, &source_desc)!!) return;
    gpu::TextureHandle source = gpu::create_texture(&device, &source_desc)!!;
    defer gpu::destroy_texture(&device, source)!!;
    gpu::TextureDesc dest_desc = source_desc;
    dest_desc.width = COPY_BC_DEST_SIZE;
    dest_desc.height = COPY_BC_DEST_SIZE;
    dest_desc.mip_levels = COPY_BC_DEST_MIPS;
    dest_desc.debug_name = "copy_bc_destination";
    gpu::TextureHandle dest = gpu::create_texture(&device, &dest_desc)!!;
    defer gpu::destroy_texture(&device, dest)!!;

    usz source_bytes = COPY_BC_SOURCE_TOTAL_BLOCKS * COPY_BC_BLOCK_BYTES;
    usz dest_bytes = copy_bc_source_offset(COPY_BC_SOURCE_MIPS) - copy_bc_source_offset(COPY_BC_LEVEL_SHIFT);
    OwnedSpan staging = allocate_owned_span(
        device:       &device,
        size:         source_bytes,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "copy_bc_staging",
    )!!;
    defer free_owned_span(&device, &staging)!!;
    char* bytes = (char*)test_span_mapping(&device, staging.span).ptr;
    for (usz i = 0; i < source_bytes; i++) bytes[i] = copy_bc_pattern(i);
    gpu::flush_mapped_span(&device, staging.span)!!;

    OwnedSpan readback = allocate_owned_span(
        device:       &device,
        size:         dest_bytes,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       { .graphics, .compute, .transfer },
        debug_name:   "copy_bc_readback",
    )!!;
    defer free_owned_span(&device, &readback)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;

    copy_transition(&cmd, source, { .layout = gpu::TextureLayout.UNDEFINED }, copy_write_state());
    copy_transition(&cmd, dest, { .layout = gpu::TextureLayout.UNDEFINED }, copy_write_state());
    for (uint mip = 0; mip < COPY_BC_SOURCE_MIPS; mip++) {
        gpu::BufferTextureCopyDesc upload = {
            .src     = staging.span.checked_subspan(
                copy_bc_source_offset(mip),
                COPY_BC_SOURCE_BLOCKS[mip] * COPY_BC_BLOCK_BYTES,
            )!!,
            .texture = source,
            .mip     = mip,
        };
        gpu::cmd_copy_buffer_to_texture(&cmd, &upload)!!;
    }
    copy_transition(&cmd, source, copy_write_state(), copy_read_state());

    for (uint mip = 0; mip < COPY_BC_DEST_MIPS; mip++) {
        gpu::TextureCopyDesc copy = {
            .src     = source,
            .src_mip = mip + COPY_BC_LEVEL_SHIFT,
            .dst     = dest,
            .dst_mip = mip,
        };
        gpu::cmd_copy_texture(&cmd, &copy)!!;
    }

    copy_transition(&cmd, dest, copy_write_state(), copy_read_state());
    for (uint mip = 0; mip < COPY_BC_DEST_MIPS; mip++) {
        gpu::TextureBufferCopyDesc download = {
            .texture = dest,
            .dst     = readback.span.checked_subspan(
                copy_bc_dest_offset(mip),
                COPY_BC_SOURCE_BLOCKS[mip + COPY_BC_LEVEL_SHIFT] * COPY_BC_BLOCK_BYTES,
            )!!,
            .mip     = mip,
        };
        gpu::cmd_copy_texture_to_buffer(&cmd, &download)!!;
    }
    gpu::Barrier host_read = {
        .before = { .transfer },
        .after  = { .host },
    };
    gpu::cmd_barrier(&cmd, &host_read)!!;
    submit_copy_commands(&device, &cmd);

    gpu::invalidate_mapped_span(&device, readback.span)!!;
    char* read = (char*)test_span_mapping(&device, readback.span).ptr;
    usz base = copy_bc_source_offset(COPY_BC_LEVEL_SHIFT);
    for (usz i = 0; i < dest_bytes; i++) {
        assert(read[i] == bytes[base + i],
            "every destination block byte equals the copied source level, down to the 1x1 level");
    }
}

fn void copy_texture_validation_faults_at_the_command() @test {
    gpu::Device device = make_copy_test_device();
    defer destroy_test_device(&device)!!;

    gpu::TextureDesc source_desc = {
        .access     = { .graphics, .compute, .transfer },
        .width      = COPY_RGBA_SIZE,
        .height     = COPY_RGBA_SIZE,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .transfer_src, .transfer_dst },
        .debug_name = "copy_fault_source",
    };
    gpu::TextureHandle source = gpu::create_texture(&device, &source_desc)!!;
    defer gpu::destroy_texture(&device, source)!!;
    gpu::TextureDesc other_desc = source_desc;
    other_desc.format = gpu::Format.BGRA8_UNORM;
    other_desc.debug_name = "copy_fault_other";
    gpu::TextureHandle other = gpu::create_texture(&device, &other_desc)!!;
    defer gpu::destroy_texture(&device, other)!!;

    gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList cmd = gpu::begin_commands(&allocator)!!;
    defer discard_test_commands(&cmd)!!;

    assert(@catch(gpu::cmd_copy_texture(&cmd, null)) == gpu::INVALID_ARGUMENT,
        "a null descriptor faults INVALID_ARGUMENT");
    gpu::TextureCopyDesc unequal = { .src = source, .dst = other };
    assert(@catch(gpu::cmd_copy_texture(&cmd, &unequal)) == gpu::INVALID_ARGUMENT,
        "unequal formats fault INVALID_ARGUMENT");
    gpu::TextureCopyDesc stale = { .src = source, .dst = {} };
    assert(@catch(gpu::cmd_copy_texture(&cmd, &stale)) == gpu::INVALID_HANDLE,
        "an unset destination faults INVALID_HANDLE");
    gpu::TextureCopyDesc overlapping = { .src = source, .dst = source, .dst_x = 1, .width = 2, .height = 2 };
    assert(@catch(gpu::cmd_copy_texture(&cmd, &overlapping)) == gpu::INVALID_ARGUMENT,
        "an overlapping same-texture region faults INVALID_ARGUMENT");

    gpu::ExecutableCommandList executable = gpu::end_commands(&cmd)!!;
    defer discard_test_executable(&executable)!!;
}
