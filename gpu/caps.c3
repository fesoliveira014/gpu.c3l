module gpu;

<* Graphics API a device is created against. *>
enum BackendKind : int {
    VULKAN,
}

<* Queue a command list records for and submits on. *>
enum QueueKind : int {
    GRAPHICS,
    // Its own queue only when caps.async_compute; otherwise aliases GRAPHICS.
    COMPUTE,
    // May map to a dedicated transfer family.
    TRANSFER,
}

<* Capabilities of the created device. Query with get_device_caps instead of
   hardcoding limits. *>
struct DeviceCaps {
    bool strict_enabled;
    bool presentation_enabled;
    bool buffer_device_address;
    bool synchronization2;
    bool dynamic_rendering;
    bool shader_int64;
    bool draw_indirect_count;
    // GPU-written root records and work arguments can execute without host
    // readback or CPU-loop emulation.
    bool generated_work;
    bool async_compute;
    QueueCounts queues;
    // Optional non-solid triangle rasterization through PolygonMode.LINE.
    bool line_polygon_mode;
    // Configured semantic heap capacity, not the hardware maximum.
    uint texture_heap_capacity;
    // Configured semantic heap capacity, not the hardware maximum.
    uint sampler_heap_capacity;
    // Effective MRT limit: the lesser of the library and selected device caps.
    uint max_color_attachments;
    // Raw selected-device limit for one Vulkan push-constant range.
    uint max_push_constant_size;
    // Selected-device direct-dispatch work-group count limit per axis.
    Vec3u max_compute_work_group_count;
    // Selected-device limit for one multi-draw indirect command.
    uint max_draw_indirect_count;
    // Selected-device limit for one generated-work execution.
    uint max_generated_work_count;
    usz  min_uniform_alignment;
    usz  min_storage_alignment;
    usz  min_texel_buffer_alignment;
    // Maximum absolute SamplerDesc.mip_lod_bias.
    float max_sampler_lod_bias;
    // 0 means anisotropic filtering is unsupported.
    float max_sampler_anisotropy;
}
