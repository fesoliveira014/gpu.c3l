module gpu;

enum BackendKind : int {
    VULKAN,
}

enum QueueKind : int {
    GRAPHICS,
    COMPUTE,
    TRANSFER,
}

// AUTO is the zero value so a zero-initialized DeviceDesc works everywhere.
enum DescriptorHeapMode : int {
    AUTO,
    DESCRIPTOR_BUFFER,
    DESCRIPTOR_INDEXING,
}

struct DeviceDesc {
    BackendKind        backend;
    bool               enable_validation;
    bool               enable_debug_names;
    bool               enable_presentation;
    DescriptorHeapMode descriptor_heap_mode;
    uint               texture_descriptor_capacity;
    uint               sampler_descriptor_capacity;
    uint               texture_capacity;
    usz                staging_arena_size;
    usz                readback_arena_size;
    uint               frames_in_flight;
    // Warm-start blob from get_pipeline_cache_data; empty = cold. Only read
    // during create_device (the driver copies it).
    char[]             pipeline_cache_data;
    ZString            application_name;
}

struct DeviceCaps {
    bool buffer_device_address;
    bool synchronization2;
    bool dynamic_rendering;
    bool timeline_semaphore;
    bool shader_int64;
    bool draw_indirect_count;
    bool descriptor_buffer;
    bool descriptor_indexing;
    bool async_compute;
    uint max_texture_descriptors;
    uint max_sampler_descriptors;
    usz  min_uniform_alignment;
    usz  min_storage_alignment;
    usz  min_texel_buffer_alignment;
    float max_sampler_anisotropy;
}
