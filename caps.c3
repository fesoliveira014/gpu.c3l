module gpu;

enum BackendKind : int {
    VULKAN,
}

enum QueueKind : int {
    GRAPHICS,
    COMPUTE,
    TRANSFER,
}

struct DeviceDesc {
    BackendKind backend;
    bool        enable_validation;
    bool        enable_debug_names;
    bool        require_descriptor_buffer;
    bool        allow_descriptor_indexing_fallback;
    uint        frames_in_flight;
    ZString     application_name;
}

struct DeviceCaps {
    bool buffer_device_address;
    bool synchronization2;
    bool dynamic_rendering;
    bool timeline_semaphore;
    bool shader_int64;
    bool descriptor_buffer;
    bool descriptor_indexing;
    uint max_texture_descriptors;
    uint max_sampler_descriptors;
    usz  min_uniform_alignment;
    usz  min_storage_alignment;
    usz  min_texel_buffer_alignment;
}
