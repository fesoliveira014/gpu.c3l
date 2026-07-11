module gpu;

// Public fault set. `faultdef` declares a flat list of globally-unique values
// in module gpu, referenced as `gpu::<NAME>` and raised with the `~` suffix.
// Each entry notes when it fires; docs/api.md carries the full fault table.
faultdef
    // No backend for the requested BackendKind, no Vulkan 1.3 driver, or
    // instance/device creation failed outright.
    UNSUPPORTED_BACKEND,
    // A requested optional capability is absent: validation layers not
    // installed, presentation not enabled/supported, missing device feature.
    UNSUPPORTED_FEATURE,
    // A descriptor or parameter is malformed: zero sizes, out-of-range
    // values, undersized output buffers, mismatched spans.
    INVALID_ARGUMENT,
    // A handle is stale (generation mismatch after destroy) or was never
    // live; also raised through a zeroed Device vtable.
    INVALID_HANDLE,
    // An operation found a resource in the wrong tracked state — most
    // commonly a texture barrier whose old_layout disagrees with the
    // tracked layout chain.
    INVALID_RESOURCE_STATE,
    // Host-side allocation failure reported by the driver.
    OUT_OF_HOST_MEMORY,
    // Device memory exhausted (VMA/driver allocation failure).
    OUT_OF_DEVICE_MEMORY,
    // The driver reported VK_ERROR_DEVICE_LOST; the device is unrecoverable.
    DEVICE_LOST,
    // destroy_texture while a live TextureIndex still owns the texture;
    // destroy the descriptor first. In-flight destroys defer instead.
    RESOURCE_IN_USE,
    // A frame/staging/readback arena ran out for this frame; raise the
    // arena size in DeviceDesc or split the work.
    ARENA_FULL,
    // A handle table hit capacity (buffers, textures, pipelines, ...);
    // for textures the capacity is DeviceDesc.texture_capacity.
    SLOT_TABLE_FULL,
    // No free descriptor slot: capacity too small for live descriptors
    // plus same-frame retires (they only recycle on a later frame).
    DESCRIPTOR_HEAP_FULL,
    // The driver rejected pipeline creation (invalid state combination or
    // internal compile failure).
    PIPELINE_CREATE_FAILED,
    // SPIR-V rejected at shader-module creation.
    SHADER_INVALID,
    // The presentation surface disappeared (window destroyed mid-frame).
    SURFACE_LOST,
    // The swapchain no longer matches the surface (resize, occlusion);
    // recreate via resize_swapchain and retry.
    SWAPCHAIN_OUT_OF_DATE,
    // A cmd_* call outside its required recording state (not recording,
    // wrong pass phase, or list already ended).
    COMMAND_RECORDING_ERROR,
    // resolve_readback called before the ticket's timeline value was
    // reached; poll_readback first.
    READBACK_NOT_READY,
    // A bounded host wait elapsed before its target, or image acquisition
    // reported transient unavailability. The operation is safe to retry
    // without changing the resource being waited on.
    WAIT_TIMEOUT;
