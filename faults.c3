module gpu;

// Public fault set. `faultdef` declares a flat list of globally-unique values
// in module gpu, referenced as `gpu::<NAME>` and raised with the `~` suffix.
faultdef
    UNSUPPORTED_BACKEND,
    UNSUPPORTED_FEATURE,
    INVALID_ARGUMENT,
    INVALID_HANDLE,
    INVALID_RESOURCE_STATE,
    OUT_OF_HOST_MEMORY,
    OUT_OF_DEVICE_MEMORY,
    DEVICE_LOST,
    RESOURCE_IN_USE,
    ARENA_FULL,
    SLOT_TABLE_FULL,
    DESCRIPTOR_HEAP_FULL,
    PIPELINE_CREATE_FAILED,
    SHADER_INVALID,
    SURFACE_LOST,
    SWAPCHAIN_OUT_OF_DATE,
    COMMAND_RECORDING_ERROR;
