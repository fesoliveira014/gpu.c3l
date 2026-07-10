module gpu_test;

import gpu;
import gpu::vk;
import vk;

// Pure selection-tier units over synthetic family tables (no device), plus
// lavapipe integration: alias tier, caps flag, cross-queue timeline chain,
// and the shared_queues no-op on a single-family device.

const uint QF_G = (uint)vk::QueueFlagBits.GRAPHICS_BIT;
const uint QF_C = (uint)vk::QueueFlagBits.COMPUTE_BIT;
const uint QF_T = (uint)vk::QueueFlagBits.TRANSFER_BIT;

fn vk::QueueFamilyProperties family(uint flags, uint count) {
    return { .queue_flags = (vk::QueueFlags)flags, .queue_count = count };
}

fn void nvidia_like_uses_second_main_queue() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G | QF_C | QF_T, 16),
        family(QF_C | QF_T, 8),
        family(QF_T, 2),
    };
    vk::QueueFamilies picked = vk::pick_queue_families(table[..])!!;
    assert(picked.graphics == 0 && picked.compute == 0 && picked.compute_index == 1, "tier 1: second queue in the main family");
    assert(picked.async_compute, "tier 1 reports async");
    assert(picked.transfer == 2, "dedicated transfer family selected");
}

fn void amd_like_uses_compute_only_family() @test {
    vk::QueueFamilyProperties[3] table = {
        family(QF_G | QF_C | QF_T, 1),
        family(QF_C | QF_T, 4),
        family(QF_T, 2),
    };
    vk::QueueFamilies picked = vk::pick_queue_families(table[..])!!;
    assert(picked.compute == 1 && picked.compute_index == 0, "tier 2: compute-only family");
    assert(picked.async_compute, "tier 2 reports async");
}

fn void single_family_aliases() @test {
    vk::QueueFamilyProperties[1] table = { family(QF_G | QF_C | QF_T, 1) };
    vk::QueueFamilies picked = vk::pick_queue_families(table[..])!!;
    assert(picked.compute == 0 && picked.compute_index == 0, "tier 3: alias");
    assert(!picked.async_compute, "alias reports no async");
    assert(picked.transfer == 0, "transfer reuses the main family");
}

fn void no_graphics_family_faults() @test {
    vk::QueueFamilyProperties[1] table = { family(QF_C | QF_T, 4) };
    fault f = @catch(vk::pick_queue_families(table[..]));
    assert(f == gpu::UNSUPPORTED_FEATURE, "no graphics+compute family faults");
}

fn gpu::Device make_queue_test_device() {
    gpu::DeviceDesc desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "gpu_queue_selection_test",
    };
    return gpu::create_device(&desc)!!;
}

fn void queue_alias_matches_async_compute_cap() @test {
    gpu::Device device = make_queue_test_device();
    defer gpu::destroy_device(&device)!!;
    vk::VkDeviceState* s = (vk::VkDeviceState*)device.backend_state;
    assert(s.compute_queue_distinct == device.caps.async_compute, "distinct flag mirrors the cap");
    assert((s.compute_queue == s.graphics_queue) == !device.caps.async_compute, "alias iff no async compute");
}

fn void cross_queue_timeline_chain_validates() @test {
    gpu::Device device = make_queue_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::SemaphoreDesc sem_desc = { .debug_name = "queue_chain" };
    gpu::SemaphoreHandle sem = gpu::create_semaphore(&device, &sem_desc)!!;
    defer gpu::destroy_semaphore(&device, sem)!!;

    gpu::begin_frame(&device)!!;
    gpu::CommandList compute_cmd = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::end_commands(&device, &compute_cmd)!!;
    gpu::CommandList[1] compute_lists = { compute_cmd };
    gpu::SemaphoreSignal[1] signals = {{ .semaphore = sem, .value = (gpu::SemaphoreValue)1, .stage = gpu::Stage.COMPUTE_SHADER }};
    gpu::SubmitDesc compute_submit = { .command_lists = compute_lists[..], .signals = signals[..] };
    gpu::submit(&device, &compute_submit)!!;

    gpu::CommandList gfx_cmd = gpu::begin_commands(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::end_commands(&device, &gfx_cmd)!!;
    gpu::CommandList[1] gfx_lists = { gfx_cmd };
    gpu::SemaphoreWait[1] waits = {{ .semaphore = sem, .value = (gpu::SemaphoreValue)1, .stage = gpu::Stage.COMPUTE_SHADER }};
    gpu::SubmitDesc gfx_submit = { .command_lists = gfx_lists[..], .waits = waits[..] };
    gpu::submit(&device, &gfx_submit)!!;

    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
    gpu::wait_queue_idle(&device, gpu::QueueKind.COMPUTE)!!;
    gpu::end_frame(&device)!!;
}

fn void shared_queues_noop_on_single_family() @test {
    gpu::Device device = make_queue_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc buf_desc = {
        .size        = 256,
        .usage       = { .storage, .addressable, .shared_queues },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .debug_name  = "shared_buf",
    };
    gpu::BufferHandle buf = gpu::create_buffer(&device, &buf_desc)!!;
    gpu::destroy_buffer(&device, buf)!!;

    gpu::TextureDesc tex_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 4,
        .height    = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst, .shared_queues },
    };
    gpu::TextureHandle tex = gpu::create_texture(&device, &tex_desc)!!;
    gpu::destroy_texture(&device, tex)!!;
}

fn void shared_queues_alone_is_invalid() @test {
    gpu::Device device = make_queue_test_device();
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc buf_desc = {
        .size        = 64,
        .usage       = { .shared_queues },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
    };
    fault f = @catch(gpu::create_buffer(&device, &buf_desc));
    assert(f == gpu::INVALID_ARGUMENT, "usage with no Vulkan bits faults INVALID_ARGUMENT");
}

fn void pool_selection_follows_queue_family() @test {
    vk::CommandPool[2] gfx  = { (vk::CommandPool)(uptr)0x10, (vk::CommandPool)(uptr)0x11 };
    vk::CommandPool[2] comp = { (vk::CommandPool)(uptr)0x30, (vk::CommandPool)(uptr)0x31 };
    vk::CommandPool[2] xfer = { (vk::CommandPool)(uptr)0x20, (vk::CommandPool)(uptr)0x21 };

    vk::RecordingContextState aliased = {
        .graphics_pools = gfx[..],
        .transfer_pools = xfer[..],
    };
    aliased.compute_pools = aliased.graphics_pools;
    assert(vk::pool_for(&aliased, gpu::QueueKind.COMPUTE, 0) == gfx[0], "aliased compute uses graphics pool");
    assert(vk::pool_for(&aliased, gpu::QueueKind.TRANSFER, 1) == xfer[1], "transfer uses transfer pool");

    vk::RecordingContextState distinct = {
        .graphics_pools = gfx[..],
        .compute_pools  = comp[..],
        .transfer_pools = xfer[..],
        .compute_pools_distinct = true,
    };
    assert(vk::pool_for(&distinct, gpu::QueueKind.COMPUTE, 0) == comp[0], "distinct compute uses compute pool");
    assert(vk::pool_for(&distinct, gpu::QueueKind.GRAPHICS, 0) == gfx[0], "graphics unaffected");
}

fn void tri_queue_frame_retires_cleanly() @test {
    gpu::Device device = make_queue_test_device();
    defer gpu::destroy_device(&device)!!;

    for (uint frame = 0; frame < 4; frame++) {
        gpu::begin_frame(&device)!!;
        gpu::CommandList[3] lists;
        gpu::QueueKind[3] kinds = { gpu::QueueKind.GRAPHICS, gpu::QueueKind.COMPUTE, gpu::QueueKind.TRANSFER };
        for (uint i = 0; i < 3; i++) {
            gpu::CommandList cmd = gpu::begin_commands(&device, kinds[i])!!;
            gpu::end_commands(&device, &cmd)!!;
            gpu::CommandList[1] one = { cmd };
            gpu::SubmitDesc submit = { .command_lists = one[..] };
            gpu::submit(&device, &submit)!!;
        }
        gpu::end_frame(&device)!!;
    }
    gpu::wait_queue_idle(&device, gpu::QueueKind.GRAPHICS)!!;
}
