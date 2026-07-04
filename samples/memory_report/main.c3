module memory_report;

import gpu;
import std::io;

// Headless: creates a resource spread, prints memory stats and the VMA
// report, and tears down cleanly — zero leak lines under validation.

fn int main(String[] args) {
    bool detailed = false;
    foreach (arg : args) {
        if (arg == "--detailed") detailed = true;
    }
    if (catch err = run(detailed)) {
        io::printfn("memory_report: FAIL (%s)", err);
        return 1;
    }
    io::printn("memory_report: PASS");
    return 0;
}

fn void? run(bool detailed) {
    gpu::DeviceDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "memory_report",
    };
    gpu::Device device = gpu::create_device(&device_desc)!;
    defer (void)gpu::destroy_device(&device);

    gpu::BufferDesc device_buf = {
        .size = 1 << 20, .usage = { .storage, .addressable }, .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name = "report_device_buffer",
    };
    gpu::BufferHandle storage = gpu::create_buffer(&device, &device_buf)!;
    defer (void)gpu::destroy_buffer(&device, storage);

    gpu::BufferDesc staging_buf = {
        .size = 256 << 10, .usage = { .transfer_src }, .memory_kind = gpu::MemoryKind.STAGING,
        .debug_name = "report_staging_buffer",
    };
    gpu::BufferHandle staging = gpu::create_buffer(&device, &staging_buf)!;
    defer (void)gpu::destroy_buffer(&device, staging);

    gpu::BufferDesc readback_buf = {
        .size = 64 << 10, .usage = { .transfer_dst }, .memory_kind = gpu::MemoryKind.READBACK,
        .debug_name = "report_readback_buffer",
    };
    gpu::BufferHandle readback = gpu::create_buffer(&device, &readback_buf)!;
    defer (void)gpu::destroy_buffer(&device, readback);

    gpu::TextureDesc sampled_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 256,
        .height    = 256,
        .mip_levels = 4,
        .format    = gpu::Format.RGBA8_UNORM,
        .usage     = { .sampled, .transfer_dst },
        .debug_name = "report_sampled_texture",
    };
    gpu::TextureHandle sampled = gpu::create_texture(&device, &sampled_desc)!;
    defer (void)gpu::destroy_texture(&device, sampled);

    gpu::TextureDesc depth_desc = {
        .dimension = gpu::TextureDimension.TEX_2D,
        .width     = 512,
        .height    = 512,
        .format    = gpu::Format.D32_FLOAT,
        .usage     = { .depth_attach },
        .debug_name = "report_depth_texture",
    };
    gpu::TextureHandle depth = gpu::create_texture(&device, &depth_desc)!;
    defer (void)gpu::destroy_texture(&device, depth);

    gpu::MemoryStats stats = gpu::get_memory_stats(&device)!;
    io::printfn("live buffers:  %d", stats.buffer_count);
    io::printfn("live textures: %d", stats.texture_count);
    io::printfn("allocations:   %d", stats.live_allocation_count);
    for (uint i = 0; i < stats.heap_count; i++) {
        io::printfn("heap %d: usage %d / budget %d bytes (blocks %d)",
            i, stats.heaps[i].usage, stats.heaps[i].budget, stats.heaps[i].block_bytes);
    }

    gpu::PersistentArenaStats arena = gpu::get_persistent_stats(&device)!;
    io::printfn("persistent arena: %d / %d bytes in %d spans", arena.used_bytes, arena.capacity_bytes, arena.allocation_count);

    String report = gpu::build_memory_report(&device, detailed)!;
    defer free(report);
    io::printfn("--- VMA report (%s) ---", detailed ? "detailed" : "summary");
    io::printn(report);
}
