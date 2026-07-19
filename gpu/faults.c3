module gpu;

// Public fault set. `faultdef` declares a flat list of globally-unique values
// in module gpu, referenced as `gpu::<NAME>` and raised with the `~` suffix.
// Each entry notes when it fires; docs/api.md carries the full fault table.
faultdef
    // No backend for the requested BackendKind, no Vulkan 1.3 driver, or
    // instance/device creation failed outright.
    UNSUPPORTED_BACKEND,
    // A valid requested capability is unavailable: validation layers not
    // installed, presentation unsupported, missing feature, or no adapter
    // can provide an exact semantic capacity.
    UNSUPPORTED_FEATURE,
    // A descriptor or parameter is malformed: zero sizes, out-of-range
    // values, undersized output buffers, mismatched spans.
    INVALID_ARGUMENT,
    // A handle is stale (generation mismatch after destroy) or was never
    // live; also raised for zero, consumed, or stale device tokens.
    INVALID_HANDLE,
    // An operation found a resource in the wrong tracked state — most
    // commonly a texture barrier whose old_layout disagrees with the
    // tracked layout chain.
    INVALID_RESOURCE_STATE,
    // Host-side allocation failure reported by the driver.
    OUT_OF_HOST_MEMORY,
    // Device memory exhausted (VMA/driver allocation failure).
    OUT_OF_DEVICE_MEMORY,
    // The driver reported VK_ERROR_DEVICE_LOST. The affected device rejects
    // later operations; peer devices remain usable.
    DEVICE_LOST,
    // A device operation encountered closing state or sustained pin
    // contention. Destruction may also return this fault. Safe to retry.
    DEVICE_BUSY,
    // Destruction found a live owner: a device child, runtime dependent,
    // surface swapchain, or texture descriptor. Release the owner first.
    RESOURCE_IN_USE,
    // A handle table hit capacity (buffers, textures, pipelines, ...);
    // for textures the capacity is DeviceDesc.texture_capacity.
    SLOT_TABLE_FULL,
    // No free descriptor slot: capacity is too small for live descriptors.
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
    // A bounded host wait elapsed before its target, or image acquisition
    // reported transient unavailability. The operation is safe to retry
    // without changing the resource being waited on.
    WAIT_TIMEOUT,
    // The backend returned an unclassified or internal failure. Inspect
    // native diagnostics; this does not imply that the device was lost.
    BACKEND_ERROR;
