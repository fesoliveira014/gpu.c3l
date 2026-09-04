module gpu_test;

import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::core::mem;
import std::core::mem::allocators;
import std::atomic::types;
import std::thread;
import std::io;

struct SparseBindCapture {
    uint             calls;
    vk::Queue        queue;
    uint             wait_count;
    vk::Semaphore[2] wait_semaphores;
    ulong[2]         wait_values;
    vk::Semaphore    signal_semaphore;
    ulong            signal_value;
    uint             image_bind_count;
    uint             opaque_bind_count;
    vk::SparseImageMemoryBind image_bind;
    vk::SparseMemoryBind      opaque_bind;
}

struct SparseBindFixture {
    vk::VkDeviceState         state;
    vk::QueueCompletionState[2] completions;
    vk::TextureCell[2]        texture_cells;
    vk::AllocationCell[3]     allocation_cells;
    gpu::Device               device;
    gpu::Queue                queue;
    gpu::TextureHandle        texture;
    gpu::GpuAllocation        allocation;
}

struct SparseFailOnNAllocator (Allocator) {
    uint fail_on;
    uint attempts;
    uint acquires;
    uint releases;
    uint outstanding;
}

struct SparseBindWorkerArgs {
    SparseBindFixture*          fixture;
    gpu::SparseTextureBindDesc* desc;
    gpu::CompletionPoint        point;
}

struct MutexProbeArgs {
    Mutex*        mutex;
    Atomic{bool}* attempted;
    Atomic{bool}* acquired;
}

struct SparseSubmitCapture {
    uint                    calls;
    uint                    wait_count;
    vk::SemaphoreSubmitInfo wait;
}

SparseBindCapture sparse_bind_capture;
SparseSubmitCapture sparse_submit_capture;
vk::Result sparse_bind_result;
vk::Result sparse_submit_result;
Atomic{bool} sparse_bind_native_entered;
Atomic{bool} sparse_bind_native_release;
ulong sparse_completed_value;

fn void*? SparseFailOnNAllocator.acquire(
    &self,
    sz size,
    AllocInitType init_type,
    sz alignment,
) @dynamic {
    self.attempts++;
    if (self.attempts == self.fail_on) return mem::OUT_OF_MEMORY~;
    void* pointer = allocators::LIBC_ALLOCATOR.acquire(
        size,
        init_type,
        alignment,
    )!;
    self.acquires++;
    self.outstanding++;
    return pointer;
}

fn void*? SparseFailOnNAllocator.resize(
    &self,
    void* pointer,
    sz size,
    sz alignment,
) @dynamic {
    return allocators::LIBC_ALLOCATOR.resize(pointer, size, alignment);
}

fn void SparseFailOnNAllocator.release(
    &self,
    void* pointer,
    bool aligned,
) @dynamic {
    self.releases++;
    self.outstanding--;
    allocators::LIBC_ALLOCATOR.release(pointer, aligned);
}

fn vk::Result capture_sparse_bind(
    vk::Queue queue,
    uint bind_info_count,
    vk::BindSparseInfo* bind_info,
    vk::Fence fence,
) {
    (void)fence;
    assert(bind_info_count == 1,
        "sparse binding emits exactly one native transaction");
    vk::TimelineSemaphoreSubmitInfo* timeline =
        (vk::TimelineSemaphoreSubmitInfo*)bind_info.p_next;
    sparse_bind_capture.calls++;
    sparse_bind_capture.queue = queue;
    sparse_bind_capture.wait_count = bind_info.wait_semaphore_count;
    sparse_bind_capture.signal_semaphore = bind_info.p_signal_semaphores[0];
    sparse_bind_capture.signal_value =
        timeline.p_signal_semaphore_values[0];
    sparse_bind_capture.image_bind_count = bind_info.image_bind_count;
    sparse_bind_capture.opaque_bind_count =
        bind_info.image_opaque_bind_count;
    assert(bind_info.wait_semaphore_count <= 2,
        "the capture fixture has storage for every expected wait");
    for (uint i = 0; i < bind_info.wait_semaphore_count; i++) {
        sparse_bind_capture.wait_semaphores[i] =
            bind_info.p_wait_semaphores[i];
        sparse_bind_capture.wait_values[i] =
            timeline.p_wait_semaphore_values[i];
    }
    if (bind_info.image_bind_count > 0) {
        sparse_bind_capture.image_bind =
            bind_info.p_image_binds[0].p_binds[0];
    }
    if (bind_info.image_opaque_bind_count > 0) {
        sparse_bind_capture.opaque_bind =
            bind_info.p_image_opaque_binds[0].p_binds[0];
    }
    return sparse_bind_result;
}

fn vk::Result block_sparse_bind(
    vk::Queue queue,
    uint bind_info_count,
    vk::BindSparseInfo* bind_info,
    vk::Fence fence,
) {
    (void)queue;
    (void)bind_info_count;
    (void)bind_info;
    (void)fence;
    sparse_bind_native_entered.store(true, AtomicOrdering.RELEASE);
    while (!sparse_bind_native_release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    return vk::Result.SUCCESS;
}

fn vk::Result sparse_counter_value(
    vk::Device device,
    vk::Semaphore semaphore,
    ulong* value,
) {
    (void)device;
    (void)semaphore;
    *value = 0;
    return vk::Result.SUCCESS;
}

fn vk::Result completed_sparse_counter_value(
    vk::Device device,
    vk::Semaphore semaphore,
    ulong* value,
) {
    (void)device;
    (void)semaphore;
    *value = sparse_completed_value;
    return vk::Result.SUCCESS;
}

fn vk::Result capture_sparse_followup_submit(
    vk::Queue queue,
    uint submit_count,
    vk::SubmitInfo2* submits,
    vk::Fence fence,
) {
    (void)queue;
    (void)fence;
    assert(submit_count == 1);
    sparse_submit_capture.calls++;
    sparse_submit_capture.wait_count =
        submits[0].wait_semaphore_info_count;
    if (sparse_submit_capture.wait_count > 0) {
        sparse_submit_capture.wait =
            submits[0].p_wait_semaphore_infos[0];
    }
    return sparse_submit_result;
}

fn vk::SparseBindNativeOps sparse_bind_ops() {
    return {
        .queue_bind_sparse            = &capture_sparse_bind,
        .get_semaphore_counter_value = &sparse_counter_value,
    };
}

fn int sparse_bind_worker(void* raw_args) {
    SparseBindWorkerArgs* args = (SparseBindWorkerArgs*)raw_args;
    vk::SparseBindNativeOps ops = {
        .queue_bind_sparse            = &block_sparse_bind,
        .get_semaphore_counter_value = &sparse_counter_value,
    };
    gpu::CompletionPoint? point =
        vk::vk_bind_sparse_texture_memory_with_ops(
            state: &args.fixture.state,
            queue: args.fixture.queue,
            desc:  args.desc,
            ops:   &ops,
        );
    if (catch point) return 1;
    args.point = point;
    return 0;
}

fn int mutex_probe_worker(void* raw_args) {
    MutexProbeArgs* args = (MutexProbeArgs*)raw_args;
    args.attempted.store(true, AtomicOrdering.RELEASE);
    args.mutex.lock()!!;
    args.acquired.store(true, AtomicOrdering.RELEASE);
    args.mutex.unlock();
    return 0;
}

fn void init_sparse_bind_fixture(SparseBindFixture* fixture) {
    *fixture = {};
    fixture.device = { .generation = 1 };
    fixture.queue = {
        .device = fixture.device,
        .id     = 0,
        .roles  = { .graphics },
    };
    fixture.completions[0] = {
        .queue    = (vk::Queue)(uptr)0x100,
        .timeline = (vk::Semaphore)(uptr)0x200,
        .roles    = { .graphics },
        .bound    = true,
    };
    fixture.completions[1] = {
        .queue              = (vk::Queue)(uptr)0x101,
        .timeline           = (vk::Semaphore)(uptr)0x201,
        .roles              = { .compute },
        .next_sequence      = 1,
        .bound              = true,
    };
    fixture.state = {
        .owner = fixture.device,
        .device = (vk::Device)(uptr)0x300,
        .host_allocator = &allocators::LIBC_ALLOCATOR,
        .max_timeline_semaphore_value_difference = ulong::max,
        .caps = {
            .sparse_textures = {
                .image_2d       = true,
                .binding_queues = { .graphics },
            },
        },
        .queue_completions = fixture.completions[..],
        .textures = {
            .slots    = fixture.texture_cells[..],
            .owner    = 71,
            .next_new = 1,
        },
        .allocations = {
            .slots    = fixture.allocation_cells[..],
            .owner    = 81,
            .next_new = 1,
        },
    };
    fixture.completions[1].published_sequence.store(
        1,
        AtomicOrdering.RELAXED,
    );
    fixture.texture = {
        .owner      = fixture.state.textures.owner,
        .index      = 0,
        .generation = 1,
    };
    fixture.texture_cells[0] = {
        .generation = 1,
        .used       = true,
        .value = {
            .image      = (vk::Image)(uptr)0x400,
            .format     = gpu::Format.RGBA8_UNORM,
            .usage      = { .sampled },
            .access     = { .graphics },
            .width      = 128,
            .height     = 128,
            .mip_levels = 8,
            .array_layers = 1,
            .sparse = true,
            .sparse_requirements = {
                .virtual_size = 65536,
                .page_allocation = {
                    .size = 4096,
                    .alignment = 4096,
                    .compatibility = vk::texture_compatibility(
                        fixture.state.allocations.owner,
                        0b10,
                    ),
                },
                .aspect_count = 2,
                .aspects = {
                    {
                        .aspect             = gpu::SparseTextureAspect.COLOR,
                        .tile_extent        = { 64, 64, 1 },
                        .mip_tail_first_lod = 2,
                        .mip_tail_size      = 16384,
                        .mip_tail_offset    = 32768,
                        .single_mip_tail    = true,
                    },
                    {
                        .aspect             = gpu::SparseTextureAspect.METADATA,
                        .mip_tail_first_lod = 0,
                        .mip_tail_size      = 4096,
                        .mip_tail_offset    = 49152,
                        .single_mip_tail    = true,
                    },
                },
            },
        },
    };
    fixture.allocation = {
        .owner      = fixture.state.allocations.owner,
        .index      = 0,
        .generation = 1,
    };
    fixture.allocation_cells[0] = {
        .generation = 1,
        .used       = true,
        .value = {
            .raw_info = {
                .memory_type  = 1,
                .device_memory = (vk::DeviceMemory)(uptr)0x500,
                .offset       = 4096,
                .size         = 65536,
            },
            .size             = 65536,
            .access           = { .graphics },
            .memory_type_bits = 0b10,
            .texture_memory   = true,
            .alignment        = 4096,
            .memory_class     = gpu::MemoryClass.TEXTURE,
        },
    };
    fixture.state.resource_mutex.init()!!;
    foreach (&completion : fixture.completions) {
        completion.submission_mutex.init()!!;
        completion.retirement_mutex.init()!!;
    }
}

fn void destroy_sparse_bind_fixture(SparseBindFixture* fixture) {
    vk::destroy_submitted_commands(&fixture.state);
    foreach (&completion : fixture.completions) {
        if (completion.submit_scratch != null) {
            mem::free(completion.submit_scratch);
            completion.submit_scratch = null;
        }
        completion.retirement_mutex.destroy();
        completion.submission_mutex.destroy();
    }
    fixture.state.resource_mutex.destroy();
}

fn void sparse_tile_backing_size_charges_partial_edge_pages() @test {
    usz size = vk::sparse_tile_backing_size(
        extent:      { 65, 64, 1 },
        tile_extent: { 64, 64, 1 },
        page_size:   4096,
    )!!;

    assert(size == 8192,
        "a partial edge tile consumes one complete sparse page");
}

fn void sparse_mip_extents_cover_2d_and_3d() @test {
    gpu::Vec3u mip_2d = vk::sparse_mip_extent(
        false,
        { 9, 5, 1 },
        3,
    )!!;
    gpu::Vec3u mip_3d = vk::sparse_mip_extent(
        true,
        { 9, 5, 7 },
        2,
    )!!;
    assert(mip_2d.x == 1 && mip_2d.y == 1 && mip_2d.z == 1,
        "2D mip extents clamp every axis to one");
    assert(mip_3d.x == 2 && mip_3d.y == 1 && mip_3d.z == 1,
        "3D mip extents reduce real depth");
}

fn void sparse_backing_math_rejects_zero_and_overflow() @test {
    assert(@catch(vk::sparse_tile_backing_size(
        extent:      { 0, 1, 1 },
        tile_extent: { 1, 1, 1 },
        page_size:   1,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::sparse_tile_backing_size(
        extent:      { 1, 1, 1 },
        tile_extent: { 0, 1, 1 },
        page_size:   1,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::sparse_tile_backing_size(
        extent:      { 1, 1, 1 },
        tile_extent: { 1, 1, 1 },
        page_size:   0,
    )) == gpu::INVALID_ARGUMENT);
    assert(@catch(vk::sparse_tile_backing_size(
        extent:      { uint::max, uint::max, uint::max },
        tile_extent: { 1, 1, 1 },
        page_size:   usz::max,
    )) == gpu::INVALID_ARGUMENT);
}

fn void sparse_interval_and_tile_overlap_are_half_open() @test {
    assert(vk::intervals_overlap(0, 8, 7, 2));
    assert(!vk::intervals_overlap(0, 8, 8, 2));
    assert(vk::intervals_overlap(usz::max - 1, 2, usz::max, 1),
        "overlap checks do not add overflowing endpoints");

    gpu::SparseTextureTileBind first = {
        .aspect = gpu::SparseTextureAspect.COLOR,
        .offset = { 0, 0, 0 },
        .extent = { 64, 64, 1 },
    };
    gpu::SparseTextureTileBind second = first;
    assert(vk::sparse_tile_boxes_overlap(&first, &second));
    second.offset.x = 64;
    assert(!vk::sparse_tile_boxes_overlap(&first, &second));
}

fn void sparse_geometry_accepts_edge_tiles_and_tail_ranges() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    gpu::SparseTextureTileBind[2] tiles = {
        {
            .aspect    = gpu::SparseTextureAspect.COLOR,
            .mip_level = 0,
            .offset    = { 0, 0, 0 },
            .extent    = { 64, 64, 1 },
        },
        {
            .aspect    = gpu::SparseTextureAspect.COLOR,
            .mip_level = 1,
            .offset    = { 0, 0, 0 },
            .extent    = { 64, 64, 1 },
        },
    };
    vk::validate_sparse_tile_geometry(
        &fixture.texture_cells[0].value,
        tiles[..],
    )!!;

    gpu::SparseTextureOpaqueBind[2] opaque = {
        {
            .aspect          = gpu::SparseTextureAspect.COLOR,
            .resource_offset = 32768,
            .size            = 4096,
        },
        {
            .aspect          = gpu::SparseTextureAspect.METADATA,
            .resource_offset = 49152,
            .size            = 4096,
        },
    };
    vk::validate_sparse_opaque_geometry(
        &fixture.texture_cells[0].value,
        opaque[..],
    )!!;
}

fn void sparse_geometry_rejects_misalignment_tail_mips_and_overlap() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    gpu::SparseTextureTileBind[2] tiles = {
        {
            .aspect    = gpu::SparseTextureAspect.COLOR,
            .mip_level = 0,
            .offset    = { 1, 0, 0 },
            .extent    = { 64, 64, 1 },
        },
        {
            .aspect    = gpu::SparseTextureAspect.COLOR,
            .mip_level = 0,
            .offset    = { 0, 0, 0 },
            .extent    = { 64, 64, 1 },
        },
    };
    assert(@catch(vk::validate_sparse_tile_geometry(
        &fixture.texture_cells[0].value,
        tiles[..],
    )) == gpu::INVALID_ARGUMENT);
    tiles[0].offset = {};
    assert(@catch(vk::validate_sparse_tile_geometry(
        &fixture.texture_cells[0].value,
        tiles[..],
    )) == gpu::INVALID_ARGUMENT,
        "duplicate tile boxes are rejected");
    tiles[1].mip_level = 2;
    assert(@catch(vk::validate_sparse_tile_geometry(
        &fixture.texture_cells[0].value,
        tiles[1:1],
    )) == gpu::INVALID_ARGUMENT,
        "mip-tail levels are unavailable to tile binds");

    gpu::SparseTextureOpaqueBind[2] opaque = {
        {
            .aspect          = gpu::SparseTextureAspect.COLOR,
            .resource_offset = 32768,
            .size            = 4096,
        },
        {
            .aspect          = gpu::SparseTextureAspect.COLOR,
            .resource_offset = 32768,
            .size            = 1,
        },
    };
    assert(@catch(vk::validate_sparse_opaque_geometry(
        &fixture.texture_cells[0].value,
        opaque[..],
    )) == gpu::INVALID_ARGUMENT,
        "same-aspect tail intervals cannot overlap");
}

fn void sparse_geometry_checks_every_dimension_and_tail_boundary() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    vk::TextureSlot* texture = &fixture.texture_cells[0].value;
    texture.width = 100;
    texture.height = 90;
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect    = gpu::SparseTextureAspect.COLOR,
        .mip_level = 0,
        .offset    = { 64, 64, 0 },
        .extent    = { 36, 26, 1 },
    }};
    vk::validate_sparse_tile_geometry(texture, tiles[..])!!;

    tiles[0].extent.x = 35;
    assert(@catch(vk::validate_sparse_tile_geometry(
        texture,
        tiles[..],
    )) == gpu::INVALID_ARGUMENT,
        "a partial x tile is valid only at the mip edge");
    tiles[0].extent = { 36, 26, 1 };
    tiles[0].offset.y = 63;
    assert(@catch(vk::validate_sparse_tile_geometry(
        texture,
        tiles[..],
    )) == gpu::INVALID_ARGUMENT,
        "y offsets use the reported tile alignment");
    tiles[0].offset = { 64, 64, 1 };
    assert(@catch(vk::validate_sparse_tile_geometry(
        texture,
        tiles[..],
    )) == gpu::INVALID_ARGUMENT,
        "2D tile binds require z offset zero");
    tiles[0].offset = { 64, 64, 0 };
    tiles[0].extent.z = 2;
    assert(@catch(vk::validate_sparse_tile_geometry(
        texture,
        tiles[..],
    )) == gpu::INVALID_ARGUMENT,
        "2D tile binds require depth one");
    tiles[0].aspect = gpu::SparseTextureAspect.METADATA;
    assert(@catch(vk::validate_sparse_tile_geometry(
        texture,
        tiles[..],
    )) == gpu::INVALID_ARGUMENT,
        "metadata is available only through opaque binds");

    tiles[0] = {
        .aspect    = gpu::SparseTextureAspect.COLOR,
        .mip_level = 0,
        .offset    = { 0, 0, 64 },
        .extent    = { 64, 64, 36 },
    };
    texture.depth = 100;
    texture.sparse_requirements.aspects[0].tile_extent.z = 64;
    vk::validate_sparse_tile_geometry(texture, tiles[..])!!;
    tiles[0].offset.z = 63;
    assert(@catch(vk::validate_sparse_tile_geometry(
        texture,
        tiles[..],
    )) == gpu::INVALID_ARGUMENT,
        "3D z offsets use the reported tile alignment");
    tiles[0].offset.z = 64;
    tiles[0].extent.z = 35;
    assert(@catch(vk::validate_sparse_tile_geometry(
        texture,
        tiles[..],
    )) == gpu::INVALID_ARGUMENT,
        "a partial z tile is valid only at the mip edge");

    gpu::SparseTextureOpaqueBind[1] opaque = {{
        .aspect          = gpu::SparseTextureAspect.COLOR,
        .resource_offset = 28672,
        .size            = 4096,
    }};
    assert(@catch(vk::validate_sparse_opaque_geometry(
        texture,
        opaque[..],
    )) == gpu::INVALID_ARGUMENT,
        "opaque color ranges cannot begin before the mip tail");
    opaque[0].resource_offset = 45056;
    opaque[0].size = 8192;
    assert(@catch(vk::validate_sparse_opaque_geometry(
        texture,
        opaque[..],
    )) == gpu::INVALID_ARGUMENT,
        "opaque color ranges cannot extend beyond the mip tail");
    opaque[0].resource_offset = 32769;
    opaque[0].size = 4096;
    assert(@catch(vk::validate_sparse_opaque_geometry(
        texture,
        opaque[..],
    )) == gpu::INVALID_ARGUMENT,
        "opaque resource offsets use page alignment");
    assert(texture.sparse_requirements.aspects[0].single_mip_tail
            && texture.sparse_requirements.aspects[0].mip_tail_stride == 0,
        "the validator consumes the normalized single-tail snapshot");
}

fn void sparse_backing_resolution_validates_unbind_and_allocation_contract() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    vk::TextureSlot* texture = &fixture.texture_cells[0].value;

    vk::ResolvedSparseBacking unbind = vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        {},
        allocation_offset: 0,
        size:              4096,
    )!!;
    assert(unbind.memory == null && unbind.absolute_offset == 0);
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        {},
        allocation_offset: 4096,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);

    vk::ResolvedSparseBacking resolved = vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 4096,
        size:              4096,
    )!!;
    assert(resolved.memory
            == fixture.allocation_cells[0].value.raw_info.device_memory
        && resolved.absolute_offset == 8192);

    gpu::GpuAllocation stale = fixture.allocation;
    stale.generation++;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        stale,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_HANDLE);
    fixture.allocation_cells[0].value.memory_class =
        gpu::MemoryClass.GPU_PRIVATE;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);
}

fn void sparse_bind_lowers_one_transaction_and_publishes_completion() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    fixture.completions[0].next_sequence = 1;
    fixture.completions[0].published_sequence.store(
        1,
        AtomicOrdering.RELAXED,
    );
    fixture.texture_cells[0].value.width = 512;
    fixture.texture_cells[0].value.height = 256;
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 1,
        .offset     = { 64, 64, 0 },
        .extent     = { 128, 64, 1 },
        .allocation = fixture.allocation,
    }};
    gpu::SparseTextureOpaqueBind[1] opaque = {{
        .aspect            = gpu::SparseTextureAspect.METADATA,
        .resource_offset   = 49152,
        .size              = 4096,
        .allocation        = fixture.allocation,
        .allocation_offset = 8192,
    }};
    gpu::CompletionPoint[2] waits = {
        gpu::internal::make_completion_point(fixture.queue, 1)!!,
        gpu::internal::make_completion_point(
            {
                .device = fixture.device,
                .id     = 1,
                .roles  = { .compute },
            },
            1,
        )!!,
    };
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
        .opaque  = opaque[..],
        .waits   = waits[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_capture = {};
    sparse_bind_result = vk::Result.SUCCESS;

    gpu::CompletionPoint point =
        vk::vk_bind_sparse_texture_memory_with_ops(
            state: &fixture.state,
            queue: fixture.queue,
            desc:  &desc,
            ops:   &ops,
        )!!;

    assert(sparse_bind_capture.calls == 1
            && sparse_bind_capture.queue == fixture.completions[0].queue);
    assert(sparse_bind_capture.wait_count == 2
            && sparse_bind_capture.wait_semaphores[0]
                == fixture.completions[0].timeline
            && sparse_bind_capture.wait_values[0] == 1
            && sparse_bind_capture.wait_semaphores[1]
                == fixture.completions[1].timeline
            && sparse_bind_capture.wait_values[1] == 1);
    assert(sparse_bind_capture.signal_semaphore
            == fixture.completions[0].timeline
        && sparse_bind_capture.signal_value == 2);
    assert(sparse_bind_capture.image_bind_count == 1
            && sparse_bind_capture.opaque_bind_count == 1);
    assert(sparse_bind_capture.image_bind.subresource.aspect_mask
            == vk::ImageAspectFlagBits.COLOR_BIT
        && sparse_bind_capture.image_bind.subresource.mip_level == 1
        && sparse_bind_capture.image_bind.subresource.array_layer == 0
        && sparse_bind_capture.image_bind.offset.x == 64
        && sparse_bind_capture.image_bind.offset.y == 64
        && sparse_bind_capture.image_bind.offset.z == 0
        && sparse_bind_capture.image_bind.extent.width == 128
        && sparse_bind_capture.image_bind.extent.height == 64
        && sparse_bind_capture.image_bind.extent.depth == 1
        && sparse_bind_capture.image_bind.memory
            == fixture.allocation_cells[0].value.raw_info.device_memory
        && sparse_bind_capture.image_bind.memory_offset == 4096
        && (uint)sparse_bind_capture.image_bind.flags == 0);
    assert(sparse_bind_capture.opaque_bind.resource_offset == 49152
        && sparse_bind_capture.opaque_bind.size == 4096
        && sparse_bind_capture.opaque_bind.memory
            == fixture.allocation_cells[0].value.raw_info.device_memory
        && sparse_bind_capture.opaque_bind.memory_offset == 12288
        && sparse_bind_capture.opaque_bind.flags
            == vk::SparseMemoryBindFlagBits.METADATA_BIT);
    assert(gpu::internal::completion_point_sequence(point) == 2
        && fixture.completions[0].published_sequence.load(
            AtomicOrdering.ACQUIRE,
        ) == 2);
    assert(fixture.texture_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 1,
        "a successful sparse bind retains its image through completion");
}

fn void sparse_bind_native_failure_cancels_without_publication() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 0,
        .offset     = { 0, 0, 0 },
        .extent     = { 64, 64, 1 },
        .allocation = fixture.allocation,
    }};
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_capture = {};
    sparse_bind_result = vk::Result.ERROR_OUT_OF_HOST_MEMORY;

    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::OUT_OF_HOST_MEMORY);
    assert(sparse_bind_capture.calls == 1);
    assert(fixture.completions[0].next_sequence == 0
        && fixture.completions[0].published_sequence.load(
            AtomicOrdering.ACQUIRE,
        ) == 0);
    assert(fixture.texture_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0,
        "native failure rolls back sparse image retention");
}

fn void sparse_bind_orders_after_the_prior_target_signal() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    fixture.completions[0].next_sequence = 1;
    fixture.completions[0].published_sequence.store(
        1,
        AtomicOrdering.RELAXED,
    );
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 0,
        .extent     = { 64, 64, 1 },
        .allocation = fixture.allocation,
    }};
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_capture = {};
    sparse_bind_result = vk::Result.SUCCESS;

    vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )!!;

    assert(sparse_bind_capture.wait_count == 1
        && sparse_bind_capture.wait_semaphores[0]
            == fixture.completions[0].timeline
        && sparse_bind_capture.wait_values[0] == 1,
        "sparse binding waits on prior target work before signalling");
}

fn void sparse_bind_orders_the_next_submit_and_retains_texture() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 0,
        .extent     = { 64, 64, 1 },
        .allocation = fixture.allocation,
    }};
    gpu::SparseTextureBindDesc bind_desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
    };
    vk::SparseBindNativeOps bind_ops = sparse_bind_ops();
    sparse_bind_result = vk::Result.SUCCESS;
    gpu::CompletionPoint bound =
        vk::vk_bind_sparse_texture_memory_with_ops(
            state: &fixture.state,
            queue: fixture.queue,
            desc:  &bind_desc,
            ops:   &bind_ops,
        )!!;

    fixture.completions[0].submit_scratch =
        mem::new(vk::QueueSubmitScratch);
    gpu::SubmitDesc submit_desc = {};
    sparse_submit_capture = {};
    sparse_submit_result = vk::Result.ERROR_UNKNOWN;
    assert(@catch(vk::vk_submit_with_queries(
        state:            &fixture.state,
        queue:            fixture.queue,
        desc:             &submit_desc,
        submit_fn:        &capture_sparse_followup_submit,
        counter_query_fn: &sparse_counter_value,
    )) == gpu::BACKEND_ERROR);
    assert(sparse_submit_capture.wait_count == 1
        && sparse_submit_capture.wait.semaphore
            == fixture.completions[0].timeline
        && sparse_submit_capture.wait.value
            == gpu::internal::completion_point_sequence(bound)
        && sparse_submit_capture.wait.stage_mask
            == vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT,
        "ordinary submission waits for the preceding sparse bind");

    sparse_submit_capture = {};
    sparse_submit_result = vk::Result.SUCCESS;
    vk::vk_submit_with_queries(
        state:            &fixture.state,
        queue:            fixture.queue,
        desc:             &submit_desc,
        submit_fn:        &capture_sparse_followup_submit,
        counter_query_fn: &sparse_counter_value,
    )!!;
    assert(sparse_submit_capture.wait_count == 1
        && sparse_submit_capture.wait.value
            == gpu::internal::completion_point_sequence(bound),
        "a failed ordinary submit preserves the sparse ordering edge");
    assert(fixture.texture_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 1);

    sparse_completed_value =
        fixture.completions[0].published_sequence.load(
            AtomicOrdering.ACQUIRE,
        );
    vk::drain_completed_submitted_commands_with_query(
        &fixture.state,
        &completed_sparse_counter_value,
    )!!;
    assert(fixture.texture_cells[0].value.tracked_command_references.load(
            AtomicOrdering.RELAXED,
        ) == 0,
        "queue retirement releases the retained sparse image");
}

fn void sparse_bind_exact_invalid_allocation_lowers_an_unbind() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect    = gpu::SparseTextureAspect.COLOR,
        .mip_level = 0,
        .offset    = { 0, 0, 0 },
        .extent    = { 64, 64, 1 },
    }};
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_capture = {};
    sparse_bind_result = vk::Result.SUCCESS;

    vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )!!;

    assert(sparse_bind_capture.image_bind.memory == null
            && sparse_bind_capture.image_bind.memory_offset == 0,
        "the exact invalid allocation lowers to null-memory unbind");
}

fn void sparse_bind_lowers_color_opaque_ranges_without_flags() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    gpu::SparseTextureOpaqueBind[1] opaque = {{
        .aspect          = gpu::SparseTextureAspect.COLOR,
        .resource_offset = 32768,
        .size            = 4096,
        .allocation      = fixture.allocation,
    }};
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .opaque  = opaque[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_capture = {};
    sparse_bind_result = vk::Result.SUCCESS;

    vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )!!;

    assert(sparse_bind_capture.opaque_bind.resource_offset == 32768
        && sparse_bind_capture.opaque_bind.size == 4096
        && (uint)sparse_bind_capture.opaque_bind.flags == 0,
        "color mip-tail binds lower without metadata flags");
}

fn void sparse_bind_preserves_a_same_queue_wait() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    fixture.completions[0].next_sequence = 1;
    fixture.completions[0].published_sequence.store(
        1,
        AtomicOrdering.RELAXED,
    );
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 0,
        .offset     = { 0, 0, 0 },
        .extent     = { 64, 64, 1 },
        .allocation = fixture.allocation,
    }};
    gpu::CompletionPoint[1] waits = {
        gpu::internal::make_completion_point(fixture.queue, 1)!!,
    };
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
        .waits   = waits[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_capture = {};
    sparse_bind_result = vk::Result.SUCCESS;

    vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )!!;

    assert(sparse_bind_capture.wait_semaphores[0]
            == fixture.completions[0].timeline
        && sparse_bind_capture.wait_values[0] == 1
        && sparse_bind_capture.signal_value == 2,
        "same-queue sparse waits are emitted before a greater signal");
}

fn void sparse_bind_points_order_same_and_cross_queue_sparse_calls() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    fixture.completions[1].next_sequence = 0;
    fixture.completions[1].published_sequence.store(
        0,
        AtomicOrdering.RELAXED,
    );
    fixture.state.caps.sparse_textures.binding_queues.compute = true;
    gpu::Queue compute_queue = {
        .device = fixture.device,
        .id     = 1,
        .roles  = { .compute },
    };
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 0,
        .offset     = { 0, 0, 0 },
        .extent     = { 64, 64, 1 },
        .allocation = fixture.allocation,
    }};
    gpu::CompletionPoint[1] waits;
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_result = vk::Result.SUCCESS;

    gpu::CompletionPoint first =
        vk::vk_bind_sparse_texture_memory_with_ops(
            state: &fixture.state,
            queue: fixture.queue,
            desc:  &desc,
            ops:   &ops,
        )!!;
    waits[0] = first;
    desc.waits = waits[..];
    gpu::CompletionPoint second =
        vk::vk_bind_sparse_texture_memory_with_ops(
            state: &fixture.state,
            queue: fixture.queue,
            desc:  &desc,
            ops:   &ops,
        )!!;
    waits[0] = second;
    sparse_bind_capture = {};
    gpu::CompletionPoint cross =
        vk::vk_bind_sparse_texture_memory_with_ops(
            state: &fixture.state,
            queue: compute_queue,
            desc:  &desc,
            ops:   &ops,
        )!!;

    assert(gpu::internal::completion_point_sequence(first) == 1
            && gpu::internal::completion_point_sequence(second) == 2,
        "same-queue sparse dependencies preserve monotonic publication");
    assert(gpu::internal::completion_point_queue_id(cross) == 1
            && gpu::internal::completion_point_sequence(cross) == 1,
        "a sparse point orders a sparse bind on a distinct queue");
    assert(sparse_bind_capture.wait_semaphores[0]
            == fixture.completions[0].timeline
        && sparse_bind_capture.wait_values[0] == 2
        && sparse_bind_capture.signal_semaphore
            == fixture.completions[1].timeline,
        "cross-queue sparse ordering lowers exact source and target timelines");
}

fn void sparse_bind_rejects_empty_mutation_and_incapable_queue() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
    };
    sparse_bind_capture = {};
    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_ARGUMENT);

    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 0,
        .offset     = { 0, 0, 0 },
        .extent     = { 64, 64, 1 },
        .allocation = fixture.allocation,
    }};
    desc.tiles = tiles[..];
    fixture.state.caps.sparse_textures.binding_queues = {};
    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(sparse_bind_capture.calls == 0
        && fixture.completions[0].next_sequence == 0);
}

fn void sparse_bind_detects_same_memory_and_placed_texture_overlap() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    fixture.state.allocations.next_new = 2;
    fixture.allocation_cells[1] = fixture.allocation_cells[0];
    fixture.allocation_cells[1].value.raw_info.offset = 0;
    gpu::GpuAllocation second = {
        .owner      = fixture.state.allocations.owner,
        .index      = 1,
        .generation = 1,
    };
    gpu::SparseTextureTileBind[2] tiles = {
        {
            .aspect     = gpu::SparseTextureAspect.COLOR,
            .mip_level  = 0,
            .offset     = { 0, 0, 0 },
            .extent     = { 64, 64, 1 },
            .allocation = fixture.allocation,
        },
        {
            .aspect            = gpu::SparseTextureAspect.COLOR,
            .mip_level         = 0,
            .offset            = { 64, 0, 0 },
            .extent            = { 64, 64, 1 },
            .allocation        = second,
            .allocation_offset = 4096,
        },
    };
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_capture = {};

    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_ARGUMENT,
        "distinct public tokens cannot overlap the same native memory bytes");
    assert(sparse_bind_capture.calls == 0);

    desc.tiles = tiles[:1];
    fixture.state.textures.next_new = 2;
    fixture.texture_cells[1] = {
        .generation = 1,
        .used       = true,
        .value = {
            .placed           = true,
            .placement        = fixture.allocation,
            .placement_offset = 0,
            .allocation_size  = 4096,
        },
    };
    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_ARGUMENT,
        "sparse backing cannot overlap a known placed texture interval");
    assert(sparse_bind_capture.calls == 0);
}

fn void sparse_backing_resolution_checks_every_live_compatibility_rule() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    vk::TextureSlot* texture = &fixture.texture_cells[0].value;
    vk::AllocationSlot original = fixture.allocation_cells[0].value;

    fixture.allocation_cells[0].value.dedicated_texture = true;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);
    fixture.allocation_cells[0].value = original;
    fixture.allocation_cells[0].value.access = { .compute };
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);
    fixture.allocation_cells[0].value = original;
    fixture.allocation_cells[0].value.raw_info.memory_type = 0;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);
    fixture.allocation_cells[0].value = original;
    fixture.allocation_cells[0].value.raw_info.offset = 1;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);
    fixture.allocation_cells[0].value = original;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 65536,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);
    fixture.allocation_cells[0].value = original;
    fixture.allocation_cells[0].value.raw_info.offset =
        (vk::DeviceSize)usz::max;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 4096,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);

    fixture.allocation_cells[0].value = original;
    fixture.allocation_cells[0].value.texture_memory = false;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);
    fixture.allocation_cells[0].value = original;
    fixture.allocation_cells[0].value.raw_info.device_memory =
        (vk::DeviceMemory)null;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);
    fixture.allocation_cells[0].value = original;
    fixture.allocation_cells[0].value.alignment = 1;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);
    fixture.allocation_cells[0].value = original;
    gpu::TextureCompatibility compatibility =
        texture.sparse_requirements.page_allocation.compatibility;
    texture.sparse_requirements.page_allocation.compatibility =
        vk::texture_compatibility(
            fixture.state.allocations.owner + 1,
            0b10,
        );
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT);
    texture.sparse_requirements.page_allocation.compatibility =
        compatibility;

    gpu::GpuAllocation foreign = fixture.allocation;
    foreign.owner++;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        foreign,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_HANDLE);
    fixture.state.validation_policy.contract =
        gpu::ContractValidation.TRUSTED;
    fixture.allocation_cells[0].value = original;
    fixture.allocation_cells[0].value.memory_class =
        gpu::MemoryClass.GPU_PRIVATE;
    assert(@catch(vk::resolve_sparse_backing_locked(
        state:             &fixture.state,
        texture:           texture,
        allocation:        fixture.allocation,
        allocation_offset: 0,
        size:              4096,
    )) == gpu::INVALID_ARGUMENT,
        "TRUSTED mode keeps sparse allocation safety validation");
}

fn void sparse_bind_host_allocation_failure_sweep_is_transactional() @test {
    for (uint fail_on = 1; fail_on <= 6; fail_on++) {
        SparseBindFixture fixture;
        init_sparse_bind_fixture(&fixture);
        SparseFailOnNAllocator allocator = { .fail_on = fail_on };
        fixture.state.host_allocator = &allocator;
        gpu::SparseTextureTileBind[1] tiles = {{
            .aspect     = gpu::SparseTextureAspect.COLOR,
            .mip_level  = 0,
            .offset     = { 0, 0, 0 },
            .extent     = { 64, 64, 1 },
            .allocation = fixture.allocation,
        }};
        gpu::SparseTextureOpaqueBind[1] opaque = {{
            .aspect            = gpu::SparseTextureAspect.METADATA,
            .resource_offset   = 49152,
            .size              = 4096,
            .allocation        = fixture.allocation,
            .allocation_offset = 8192,
        }};
        gpu::CompletionPoint[1] waits = {
            gpu::internal::make_completion_point(
                {
                    .device = fixture.device,
                    .id     = 1,
                    .roles  = { .compute },
                },
                1,
            )!!,
        };
        gpu::SparseTextureBindDesc desc = {
            .texture = fixture.texture,
            .tiles   = tiles[..],
            .opaque  = opaque[..],
            .waits   = waits[..],
        };
        vk::SparseBindNativeOps ops = sparse_bind_ops();
        sparse_bind_capture = {};

        assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
            state: &fixture.state,
            queue: fixture.queue,
            desc:  &desc,
            ops:   &ops,
        )) == gpu::OUT_OF_HOST_MEMORY);
        assert(allocator.attempts == fail_on
            && allocator.acquires == fail_on - 1
            && allocator.releases == allocator.acquires
            && allocator.outstanding == 0,
            "every partial sparse lowering allocation is released once");
        assert(sparse_bind_capture.calls == 0
            && fixture.completions[0].next_sequence == 0);
        destroy_sparse_bind_fixture(&fixture);
    }
}

fn void sparse_bind_validation_precedes_host_allocation() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    SparseFailOnNAllocator allocator = { .fail_on = 1 };
    fixture.state.host_allocator = &allocator;
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 0,
        .offset     = { 0, 0, 0 },
        .extent     = { 64, 64, 1 },
        .allocation = fixture.allocation,
    }};
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_capture = {};

    desc.texture.generation++;
    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_HANDLE);
    desc.texture = fixture.texture;

    tiles[0].offset.x = 1;
    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_ARGUMENT);
    tiles[0].offset.x = 0;

    tiles[0].allocation.generation++;
    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_HANDLE);
    tiles[0].allocation = fixture.allocation;

    gpu::CompletionPoint[1] waits = {
        gpu::internal::make_completion_point(fixture.queue, 1)!!,
    };
    desc.waits = waits[..];
    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_HANDLE);

    assert(allocator.attempts == 0
            && allocator.acquires == 0
            && sparse_bind_capture.calls == 0,
        "deterministic validation faults cannot be masked by host OOM");
}

fn void sparse_only_completion_retires_without_a_command_record() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 0,
        .offset     = { 0, 0, 0 },
        .extent     = { 64, 64, 1 },
        .allocation = fixture.allocation,
    }};
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_result = vk::Result.SUCCESS;

    gpu::CompletionPoint point =
        vk::vk_bind_sparse_texture_memory_with_ops(
            state: &fixture.state,
            queue: fixture.queue,
            desc:  &desc,
            ops:   &ops,
        )!!;
    vk::retire_observed_queue(
        &fixture.state,
        &fixture.completions[0],
        gpu::internal::completion_point_sequence(point),
    );

    assert(fixture.completions[0].retired_sequence.load(
            AtomicOrdering.ACQUIRE,
        ) == 1);
    assert(fixture.completions[0].submitted_count == 0
        && fixture.completions[0].submitted_head == null);
}

fn void sparse_bind_headroom_and_sequence_failures_are_pre_native() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 0,
        .offset     = { 0, 0, 0 },
        .extent     = { 64, 64, 1 },
        .allocation = fixture.allocation,
    }};
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_capture = {};
    fixture.state.max_timeline_semaphore_value_difference = 0;

    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::DEVICE_BUSY);
    assert(sparse_bind_capture.calls == 0
        && fixture.completions[0].next_sequence == 0);

    fixture.state.max_timeline_semaphore_value_difference = ulong::max;
    fixture.completions[0].next_sequence =
        gpu::internal::MAX_COMPLETION_SEQUENCE;
    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::SLOT_TABLE_FULL);
    assert(sparse_bind_capture.calls == 0);
}

fn void sparse_bind_unpublished_wait_and_malformed_allocation_are_pre_native() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 0,
        .offset     = { 0, 0, 0 },
        .extent     = { 64, 64, 1 },
        .allocation = fixture.allocation,
    }};
    gpu::CompletionPoint[1] waits = {
        gpu::internal::make_completion_point(fixture.queue, 1)!!,
    };
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
        .waits   = waits[..],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_capture = {};

    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_HANDLE);
    assert(sparse_bind_capture.calls == 0);

    desc.waits = {};
    tiles[0].allocation = { .generation = 1 };
    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_HANDLE);
    assert(sparse_bind_capture.calls == 0);
}

fn void sparse_bind_fault_mapping_and_inputs_are_transactional() @test {
    vk::Result[3] results = {
        vk::Result.ERROR_OUT_OF_DEVICE_MEMORY,
        vk::Result.ERROR_DEVICE_LOST,
        vk::Result.ERROR_INITIALIZATION_FAILED,
    };
    fault[3] expected = {
        gpu::OUT_OF_DEVICE_MEMORY,
        gpu::DEVICE_LOST,
        gpu::BACKEND_ERROR,
    };
    foreach (i, result : results) {
        SparseBindFixture fixture;
        init_sparse_bind_fixture(&fixture);
        gpu::SparseTextureTileBind[1] tiles = {{
            .aspect     = gpu::SparseTextureAspect.COLOR,
            .mip_level  = 0,
            .offset     = { 0, 0, 0 },
            .extent     = { 64, 64, 1 },
            .allocation = fixture.allocation,
        }};
        gpu::SparseTextureTileBind original = tiles[0];
        gpu::SparseTextureBindDesc desc = {
            .texture = fixture.texture,
            .tiles   = tiles[..],
        };
        vk::SparseBindNativeOps ops = sparse_bind_ops();
        sparse_bind_capture = {};
        sparse_bind_result = result;

        assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
            state: &fixture.state,
            queue: fixture.queue,
            desc:  &desc,
            ops:   &ops,
        )) == expected[i]);
        assert(tiles[0].allocation == original.allocation
            && tiles[0].offset.x == original.offset.x
            && desc.tiles.ptr == tiles[..].ptr,
            "native failure leaves caller descriptor storage unchanged");
        assert(fixture.completions[0].next_sequence == 0
            && fixture.completions[0].published_sequence.load(
                AtomicOrdering.ACQUIRE,
            ) == 0);
        destroy_sparse_bind_fixture(&fixture);
    }
}

fn void sparse_bind_shape_limits_precede_allocation_and_native_work() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    gpu::SparseTextureTileBind tile;
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = (&tile)[:vk::MAX_SPARSE_TILE_BINDS + 1],
    };
    vk::SparseBindNativeOps ops = sparse_bind_ops();
    sparse_bind_capture = {};

    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_ARGUMENT);
    desc.tiles = ((gpu::SparseTextureTileBind*)null)[:1];
    assert(@catch(vk::vk_bind_sparse_texture_memory_with_ops(
        state: &fixture.state,
        queue: fixture.queue,
        desc:  &desc,
        ops:   &ops,
    )) == gpu::INVALID_ARGUMENT);
    assert(sparse_bind_capture.calls == 0);
}

fn void sparse_bind_holds_resource_and_target_queue_locks_only_through_native_call() @test {
    SparseBindFixture fixture;
    init_sparse_bind_fixture(&fixture);
    defer destroy_sparse_bind_fixture(&fixture);
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect     = gpu::SparseTextureAspect.COLOR,
        .mip_level  = 0,
        .offset     = { 0, 0, 0 },
        .extent     = { 64, 64, 1 },
        .allocation = fixture.allocation,
    }};
    gpu::SparseTextureBindDesc desc = {
        .texture = fixture.texture,
        .tiles   = tiles[..],
    };
    SparseBindWorkerArgs args = {
        .fixture = &fixture,
        .desc    = &desc,
    };
    Thread worker;
    sparse_bind_native_entered.store(false, AtomicOrdering.RELAXED);
    sparse_bind_native_release.store(false, AtomicOrdering.RELAXED);
    worker.create(&sparse_bind_worker, &args)!!;
    while (!sparse_bind_native_entered.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }

    Atomic{bool}[3] attempted;
    Atomic{bool}[3] acquired;
    MutexProbeArgs[3] probes = {
        {
            .mutex     = &fixture.state.resource_mutex,
            .attempted = &attempted[0],
            .acquired  = &acquired[0],
        },
        {
            .mutex     = &fixture.completions[0].submission_mutex,
            .attempted = &attempted[1],
            .acquired  = &acquired[1],
        },
        {
            .mutex     = &fixture.completions[1].submission_mutex,
            .attempted = &attempted[2],
            .acquired  = &acquired[2],
        },
    };
    Thread[3] lock_workers;
    foreach (i, &lock_worker : lock_workers) {
        attempted[i].store(false, AtomicOrdering.RELAXED);
        acquired[i].store(false, AtomicOrdering.RELAXED);
        lock_worker.create(&mutex_probe_worker, &probes[i])!!;
    }
    foreach (&started : attempted) {
        while (!started.load(AtomicOrdering.ACQUIRE)) thread::yield();
    }
    while (!acquired[2].load(AtomicOrdering.ACQUIRE)) thread::yield();
    assert(!acquired[0].load(AtomicOrdering.ACQUIRE),
        "resource users remain blocked during the native call");
    assert(!acquired[1].load(AtomicOrdering.ACQUIRE),
        "target-queue operations remain blocked during the native call");

    sparse_bind_native_release.store(true, AtomicOrdering.RELEASE);
    assert(worker.join()!! == 0 && args.point.is_valid());
    foreach (&lock_worker : lock_workers) {
        assert(lock_worker.join()!! == 0);
    }
    assert(acquired[0].load(AtomicOrdering.ACQUIRE)
        && acquired[1].load(AtomicOrdering.ACQUIRE),
        "resource and target-queue locks release after host return");
}

fn bool run_real_sparse_bind_dimension(
    gpu::Device* device,
    gpu::DeviceCaps caps,
    bool image_3d,
) {
    if ((image_3d && !caps.sparse_textures.image_3d)
        || (!image_3d && !caps.sparse_textures.image_2d)) {
        return false;
    }
    gpu::QueueKind queue_kind;
    gpu::QueueRoles access;
    if (caps.sparse_textures.binding_queues.graphics) {
        queue_kind = gpu::QueueKind.GRAPHICS;
        access = { .graphics };
    } else if (caps.sparse_textures.binding_queues.compute) {
        queue_kind = gpu::QueueKind.COMPUTE;
        access = { .compute };
    } else if (caps.sparse_textures.binding_queues.transfer) {
        queue_kind = gpu::QueueKind.TRANSFER;
        access = { .transfer };
    } else {
        return false;
    }
    gpu::Format[5] formats = {
        gpu::Format.RGBA8_UNORM,
        gpu::Format.R32_FLOAT,
        gpu::Format.R8_UINT,
        gpu::Format.RG16_FLOAT,
        gpu::Format.RGBA16_FLOAT,
    };
    gpu::TextureDesc texture_desc = {
        .width      = 256,
        .height     = 256,
        .depth      = image_3d ? 64 : 0,
        .mip_levels = 9,
        .format     = gpu::Format.RGBA8_UNORM,
        .usage      = { .transfer_src, .transfer_dst },
        .access     = access,
    };
    gpu::TextureHandle texture;
    bool created;
    foreach (format : formats) {
        texture_desc.format = format;
        gpu::TextureHandle? candidate =
            gpu::create_sparse_texture(device, &texture_desc);
        if (try live = candidate) {
            texture = live;
            created = true;
            break;
        }
        assert(@catch(candidate) == gpu::UNSUPPORTED_FEATURE);
    }
    if (!created) return false;
    defer gpu::destroy_texture(device, texture)!!;

    gpu::SparseTextureRequirements requirements =
        gpu::get_sparse_texture_requirements(device, texture)!!;
    usz alignment = requirements.page_allocation.alignment;
    gpu::SparseTextureAspectRequirements color;
    bool found_color;
    for (uint i = 0; i < requirements.aspect_count; i++) {
        if (requirements.aspects[i].aspect
                == gpu::SparseTextureAspect.COLOR) {
            color = requirements.aspects[i];
            found_color = true;
            break;
        }
    }
    if (!found_color
        || color.mip_tail_first_lod == 0
        || color.tile_extent.x == 0
        || color.tile_extent.y == 0
        || color.tile_extent.z == 0) {
        return false;
    }
    gpu::SparseTextureTileBind[1] tiles = {{
        .aspect    = gpu::SparseTextureAspect.COLOR,
        .mip_level = 0,
        .extent = {
            min(texture_desc.width, color.tile_extent.x),
            min(texture_desc.height, color.tile_extent.y),
            image_3d
                ? min(texture_desc.depth, color.tile_extent.z)
                : 1,
        },
    }};
    gpu::SparseTextureOpaqueBind[2] opaque;
    usz allocation_size = requirements.page_allocation.size;
    uint opaque_count;
    for (uint i = 0; i < requirements.aspect_count; i++) {
        gpu::SparseTextureAspectRequirements aspect =
            requirements.aspects[i];
        if (aspect.mip_tail_size == 0) continue;
        if (aspect.mip_tail_offset % alignment != 0) {
            return false;
        }
        allocation_size = (allocation_size + alignment - 1)
            & ~(alignment - 1);
        opaque[opaque_count++] = {
            .aspect            = aspect.aspect,
            .resource_offset   = aspect.mip_tail_offset,
            .size              = aspect.mip_tail_size,
            .allocation_offset = allocation_size,
        };
        allocation_size += aspect.mip_tail_size;
    }

    gpu::TextureRequirements[1] page_requirements = {
        requirements.page_allocation,
    };
    gpu::AllocationDesc allocation_desc = {
        .size                 = allocation_size,
        .alignment            = alignment,
        .memory_class         = gpu::MemoryClass.TEXTURE,
        .access               = access,
        .texture_requirements = page_requirements[..],
    };
    gpu::GpuAllocation allocation =
        gpu::allocate_memory(device, &allocation_desc)!!;
    defer gpu::free_allocation(device, &allocation)!!;
    tiles[0].allocation = allocation;
    foreach (&bind : opaque[:opaque_count]) {
        bind.allocation = allocation;
    }
    gpu::Queue queue = gpu::get_queue(device, queue_kind)!!;
    gpu::SparseTextureBindDesc bind_desc = {
        .texture = texture,
        .tiles   = tiles[..],
        .opaque  = opaque[:opaque_count],
    };
    gpu::CompletionPoint bound =
        gpu::bind_sparse_texture_memory(queue, &bind_desc)!!;
    gpu::wait_completion(bound, ulong::max)!!;

    OwnedSpan upload;
    OwnedSpan readback;
    bool owns_upload;
    bool owns_readback;
    defer if (owns_upload) free_owned_span(device, &upload)!!;
    defer if (owns_readback) free_owned_span(device, &readback)!!;
    gpu::CommandAllocator allocator =
        gpu::create_command_allocator(device, queue)!!;
    defer gpu::destroy_command_allocator(&allocator)!!;
    gpu::CommandList commands = gpu::begin_commands(&allocator)!!;
    gpu::TextureBarrier barrier = {
        .texture = texture,
        .before  = { .layout = gpu::TextureLayout.UNDEFINED },
        .after   = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
    };
    gpu::cmd_texture_barrier(&commands, &barrier)!!;
    uint copy_depth = image_3d ? tiles[0].extent.z : 1;
    usz upload_size = (usz)tiles[0].extent.x
        * (usz)tiles[0].extent.y
        * (usz)copy_depth
        * gpu::internal::format_block(texture_desc.format).bytes;
    upload = allocate_owned_span(
        device:       device,
        size:         upload_size,
        memory_class: gpu::MemoryClass.CPU_WRITE,
        access:       access,
        debug_name:   "sparse_bind_real_upload",
    )!!;
    owns_upload = true;
    char[] upload_bytes =
        gpu::get_span_mapping(device, upload.span)!!;
    foreach (i, &byte : upload_bytes) {
        *byte = (char)(((usz)i * 17 + 3) & 0x7f);
    }
    gpu::flush_mapped_span(device, upload.span)!!;
    readback = allocate_owned_span(
        device:       device,
        size:         upload_size,
        memory_class: gpu::MemoryClass.CPU_READ,
        access:       access,
        debug_name:   "sparse_bind_real_readback",
    )!!;
    owns_readback = true;
    gpu::BufferTextureCopyDesc copy = {
        .src     = upload.span,
        .texture = texture,
        .width   = tiles[0].extent.x,
        .height  = tiles[0].extent.y,
        .depth   = image_3d ? copy_depth : 0,
    };
    gpu::cmd_copy_buffer_to_texture(&commands, &copy)!!;
    gpu::TextureBarrier to_source = {
        .texture = texture,
        .before  = {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
        .after = {
            .layout = gpu::TextureLayout.TRANSFER_SOURCE,
            .stages = { .transfer },
            .access = { .read },
        },
    };
    gpu::cmd_texture_barrier(&commands, &to_source)!!;
    gpu::TextureBufferCopyDesc copy_back = {
        .texture = texture,
        .dst     = readback.span,
        .width   = tiles[0].extent.x,
        .height  = tiles[0].extent.y,
        .depth   = image_3d ? copy_depth : 0,
    };
    gpu::cmd_copy_texture_to_buffer(&commands, &copy_back)!!;
    gpu::Barrier host_read = {
        .before = { .transfer },
        .after  = { .host },
    };
    gpu::cmd_barrier(&commands, &host_read)!!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!!;
    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::CompletionWait[1] waits = {{
        .point  = bound,
        .before = { .transfer },
    }};
    gpu::SubmitDesc submit_desc = {
        .command_lists    = lists[..],
        .completion_waits = waits[..],
    };
    gpu::CompletionPoint used = gpu::submit(queue, &submit_desc)!!;
    gpu::wait_completion(used, ulong::max)!!;
    gpu::invalidate_mapped_span(device, readback.span)!!;
    char[] readback_bytes =
        gpu::get_span_mapping(device, readback.span)!!;
    foreach (i, byte : readback_bytes) {
        assert(byte == (char)(((usz)i * 17 + 3) & 0x7f),
            "bound sparse tile preserves uploaded bytes through readback");
    }

    tiles[0].allocation = {};
    foreach (&bind : opaque[:opaque_count]) {
        bind.allocation = {};
        bind.allocation_offset = 0;
    }
    gpu::CompletionPoint[1] unbind_waits = { used };
    bind_desc.waits = unbind_waits[..];
    gpu::CompletionPoint unbound =
        gpu::bind_sparse_texture_memory(queue, &bind_desc)!!;
    gpu::wait_completion(unbound, ulong::max)!!;
    return true;
}

fn void real_sparse_bind_use_and_unbind_when_supported() @test {
    gpu::RuntimeDesc runtime_desc = {
        .contract_validation = gpu::ContractValidation.TRUSTED,
        .application_name    = "gpu_sparse_bind_test",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    gpu::DeviceDesc device_desc = {
        .enable_sparse_textures = true,
    };
    bool ran_2d;
    bool ran_3d;
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!!;
        if (!gpu::supports_device_desc(
            &adapter,
            &device_desc,
        )!!.supported) continue;
        gpu::Device device = gpu::create_device(&adapter, &device_desc)!!;
        gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
        if (!ran_2d) {
            ran_2d = run_real_sparse_bind_dimension(
                &device,
                caps,
                false,
            );
        }
        if (!ran_3d) {
            ran_3d = run_real_sparse_bind_dimension(
                &device,
                caps,
                true,
            );
        }
        gpu::destroy_device(&device)!!;
        if (ran_2d && ran_3d) return;
    }
    if (!ran_2d) {
        io::eprintfn(
            "real_sparse_bind_use_and_unbind_when_supported: "
            "no compatible sparse 2D capability, skipping 2D proof",
        );
    }
    if (!ran_3d) {
        io::eprintfn(
            "real_sparse_bind_use_and_unbind_when_supported: "
            "no compatible sparse 3D capability, skipping 3D proof",
        );
    }
}
