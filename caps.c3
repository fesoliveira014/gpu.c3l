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
    DescriptorHeapMode descriptor_heap_mode;
    uint               texture_descriptor_capacity;
    uint               sampler_descriptor_capacity;
    uint               frames_in_flight;
    ZString            application_name;
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
