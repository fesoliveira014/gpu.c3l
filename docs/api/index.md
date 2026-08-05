# Public API

The public facade is `gpu`. Platform surface creation lives in
`gpu::surface::wayland`, `gpu::surface::x11`, and `gpu::surface::win32`.
Everything under `gpu::internal` is private.

Source declarations and docstrings in `gpu/gpu.c3i`, `gpu/gpu.c3`, and the
platform surface modules are authoritative for exact signatures. This
reference explains ownership, lifetime, ordering, concurrency, and fault
behavior by domain.

## Common conventions

Handles are strongly typed, device-scoped generational values. Their zero
constants are invalid sentinels. `is_valid` checks only value shape unless a
domain page says otherwise; it does not prove that the owner is still live.
Handle equality compares identity, not resource contents.

Functions returning `T?` or `void?` use C3 optionals and the public fault set.
Creation publishes no object on failure. Calls preserve input owners on
failure unless their contract explicitly says that successful submission or
state transition consumes a one-shot value.

Zero-initialized descriptors select documented defaults. Required fields and
unsupported combinations return `INVALID_ARGUMENT` or
`UNSUPPORTED_FEATURE`. `ContractValidation.FULL` adds diagnostics and retained
references; it does not change valid program behavior or turn raw shader
values into owners.

Concurrency categories are:

- **externally synchronized** for runtime/surface registry mutation, one
  swapchain, and operations targeting one native queue;
- **thread-safe** for documented immutable queries and internally synchronized
  resource operations; and
- **thread-confined** for a command allocator with live recordings and every
  recording/executable token alias.

See [Architecture](../architecture.md#threading-model) for the full model.

## Faults

The public fault set is:

`UNSUPPORTED_BACKEND`, `UNSUPPORTED_FEATURE`, `INVALID_ARGUMENT`,
`INVALID_HANDLE`, `INVALID_RESOURCE_STATE`, `OUT_OF_HOST_MEMORY`,
`OUT_OF_DEVICE_MEMORY`, `DEVICE_LOST`, `DEVICE_BUSY`, `RESOURCE_IN_USE`,
`SLOT_TABLE_FULL`, `GENERATED_SCRATCH_EXHAUSTED`,
`COMMAND_ALLOCATOR_CAPACITY_EXCEEDED`, `DESCRIPTOR_HEAP_FULL`,
`PIPELINE_CREATE_FAILED`, `SHADER_INVALID`, `SURFACE_LOST`,
`SWAPCHAIN_OUT_OF_DATE`, `COMMAND_RECORDING_ERROR`, `WAIT_TIMEOUT`, and
`BACKEND_ERROR`.

Always-checked faults protect host safety, valid lowering, handle identity,
authoritative command phase, and lifecycle integrity. Full validation adds
semantic misuse detail. Runtime faults remain possible under every validation
policy.

## Complete symbol map

Each exported symbol belongs to exactly one domain page.

### [Runtime and devices](runtime_and_devices.md)

Types: `Vec2f`, `Vec4f`, `Vec4u`, `Device`, `Runtime`, `RuntimeDesc`,
`ContractValidation`, `Adapter`, `AdapterList`, `AdapterClass`,
`AdapterMemoryInfo`, `AdapterQueueInfo`, `AdapterLimits`, `AdapterInfo`,
`BackendVersion`, `AdapterDiagnostics`, `Surface`, `QueueKind`, `QueueRoles`,
`Queue`, `QueueInfo`, `QueueRequest`, `DeviceDesc`, `DeviceSupport`,
`DeviceCaps`, `TimestampCaps`, `SparseTextureCaps`,
`AccelerationStructureCaps`, `RayQueryCaps`, and `RayTracingPipelineCaps`.

Fields: `RayTracingPipelineCaps.indirect2_dispatch`.

Constants: `DEVICE_INVALID`, `RUNTIME_INVALID`, `ADAPTER_INVALID`,
`SURFACE_INVALID`, and `QUEUE_INVALID`.

Methods: `Device.is_valid`, `Runtime.is_valid`, `Adapter.is_valid`,
`Surface.is_valid`, `Queue.is_valid`, and `Queue.equals`.

Callables: `full_validation_runtime_desc`,
`create_runtime`, `destroy_runtime`, `AdapterList.get`,
`enumerate_adapters`, `get_adapter_info`, `get_adapter_diagnostics`,
`supports_presentation`, `supports_device_desc`, `create_device`,
`destroy_device`, `get_device_caps`, `get_queue`, and `get_queue_info`.

### [Memory and resources](memory_and_resources.md)

Types: `GpuAllocation`, `GpuSpan`, `MappedGpuSpan`, `GpuAddress`,
`MemoryClass`, `AllocationDesc`, `AllocationInfo`, `MemoryHeapBudget`,
`MemoryStats`, `TextureCompatibility`, `Format`, `SampleCount`, `Filter`,
`AddressMode`, `TextureUsage`, `TextureFormatFeatures`,
`TextureSampleCountSupport`, `TextureFormatSupport`, `TextureRequirements`,
`SparseTextureAspect`, `SparseTextureAspectRequirements`,
`SparseTextureRequirements`, `SparseTextureTileBind`,
`SparseTextureOpaqueBind`, `SparseTextureBindDesc`, `DedicatedTexture`,
`TextureHandle`, `TextureDesc`, `TextureViewDesc`, `TextureView`,
`TextureViewCreateDesc`, `TextureIndex`, `SamplerIndex`, `SamplerDesc`,
`AccelerationStructureHandle`, `AccelerationStructureView`,
`AccelerationStructureIndex`, `AccelerationStructureKind`,
`AccelerationStructureGeometryKind`, `AccelerationStructureIndexType`,
`AccelerationStructureBuildFlags`, `AccelerationStructureGeometryFlags`,
`AccelerationStructureInstanceFlags`,
`AccelerationStructureTriangleGeometryDesc`,
`AccelerationStructureAabbGeometryDesc`, `AccelerationStructureGeometryDesc`,
`AccelerationStructureDesc`, `AccelerationStructureRequirements`,
`AccelerationStructureInstance`, `AccelerationStructureInstanceDesc`, and
`DedicatedAccelerationStructure`.

Constants: `GPU_ALLOCATION_INVALID`, `TEXTURE_HANDLE_INVALID`,
`TEXTURE_VIEW_INVALID`, `TEXTURE_INDEX_INVALID`, `SAMPLER_INDEX_INVALID`,
`ACCELERATION_STRUCTURE_HANDLE_INVALID`,
`ACCELERATION_STRUCTURE_VIEW_INVALID`,
`ACCELERATION_STRUCTURE_INDEX_INVALID`,
`MAX_MEMORY_HEAPS`, `DEFAULT_TEXTURE_CAPACITY`,
`MAX_SPARSE_TEXTURE_ASPECTS`, `DEFAULT_TEXTURE_HEAP_CAPACITY`,
`DEFAULT_SAMPLER_HEAP_CAPACITY`, and `MAX_SHADER_HEAP_CAPACITY`.

Methods: `GpuAllocation.is_valid`, `GpuAllocation.equals`,
`TextureHandle.is_valid`, `TextureHandle.equals`, `TextureView.is_valid`,
`TextureView.equals`, `TextureIndex.is_valid`, `SamplerIndex.is_valid`,
`AccelerationStructureHandle.is_valid`,
`AccelerationStructureHandle.equals`, `AccelerationStructureView.is_valid`,
`AccelerationStructureView.equals`, `AccelerationStructureIndex.is_valid`,
`GpuSpan.unchecked_subspan`, `GpuSpan.checked_subspan`,
`MappedGpuSpan.checked_subspan`, and `TextureCompatibility.is_valid`.

Callables: `allocate_memory`,
`free_allocation`, `get_allocation_info`, `get_allocation_span`,
`get_span_mapping`, `get_span_address`, `flush_mapped_span`,
`invalidate_mapped_span`, `mapped_gpu_span`,
`get_texture_format_support`, `supports_texture_desc`,
`get_texture_requirements`, `create_texture`, `create_placed_texture`,
`create_dedicated_texture`, `create_sparse_texture`,
`get_sparse_texture_requirements`, `bind_sparse_texture_memory`,
`destroy_texture`, `create_texture_view`, `create_texture_views`,
`destroy_texture_view`, `intern_sampler`,
`get_acceleration_structure_requirements`, `create_acceleration_structure`,
`create_placed_acceleration_structure`,
`create_dedicated_acceleration_structure`, `destroy_acceleration_structure`,
`get_acceleration_structure_address`, `make_acceleration_structure_instance`,
`create_acceleration_structure_view`, and
`destroy_acceleration_structure_view`.

### [Shaders and pipelines](shaders_and_pipelines.md)

Types: `RootPush`, `GraphicsRootPush`, `GeneratedDrawRecord`,
`GeneratedDrawIndexedRecord`, `GeneratedDispatchRecord`, `PipelineHandle`,
`PrimitiveTopology`, `CompareOp`, `CullMode`, `FrontFace`, `PolygonMode`,
`BlendFactor`, `BlendOp`, `ColorWriteMask`, `ShaderDesc`, `DepthState`,
`BlendState`, `ColorTargetState`, `ColorState`, `DynamicRasterState`,
`ComputePipelineDesc`, `GraphicsPipelineDesc`, `RayTracingHitGroupKind`,
`RayTracingGroupShader`, `RayTracingHitGroupDesc`, `RayTracingPipelineDesc`,
`RayTracingShaderGroupRange`, `RayTracingPipelineInfo`,
`RayTracingShaderBindingTableRegion`, and
`RayTracingShaderBindingTable`.

Constants: `PIPELINE_HANDLE_INVALID`, `COLOR_WRITE_ALL`, `MAX_PIPELINES`, and
`MAX_COLOR_ATTACHMENTS`.

Methods: `PipelineHandle.is_valid` and `PipelineHandle.equals`.

Callables: `color_blend_disabled`,
`alpha_blend`, `premultiplied_alpha_blend`, `additive_blend`,
`uniform_color_state`, `create_compute_pipeline`,
`create_graphics_pipeline`, `create_ray_tracing_pipeline`,
`get_ray_tracing_pipeline_info`, `get_ray_tracing_shader_group_handles`,
`get_ray_tracing_shader_group_stack_size`,
`destroy_pipeline`, `get_pipeline_cache_size`, and `get_pipeline_cache_data`.

### [Commands and rendering](commands_and_rendering.md)

Types: `CommandAllocatorHandle`, `CommandAllocator`, `CommandAllocatorDesc`,
`CommandList`, `ExecutableCommandList`, `GeneratedWorkKind`,
`GeneratedScratchDesc`, `Vec3u`, `Viewport`, `ScissorRect`, `GraphicsState`,
`DrawIndirectCommand`, `DrawIndexedIndirectCommand`,
`DispatchIndirectCommand`, `TraceRaysIndirectCommand`,
`TraceRaysIndirectCommand2`, `BufferCopyDesc`,
`BufferTextureCopyDesc`,
`TextureBufferCopyDesc`, `AccelerationStructureTriangleBuildDesc`,
`AccelerationStructureAabbBuildDesc`,
`AccelerationStructureGeometryBuildDesc`, `AccelerationStructureBuildDesc`,
`IndexType`, `AttachmentViewHandle`,
`AttachmentViewDesc`, `LoadOp`, `StoreOp`, `ClearColor`, `ClearDepth`,
`ColorTargetDesc`, `DepthTargetDesc`, and `RenderPassDesc`.

Constants: `COMMAND_ALLOCATOR_HANDLE_INVALID`, `ATTACHMENT_VIEW_HANDLE_INVALID`,
`DEFAULT_COMMAND_ALLOCATOR_CAPACITY`, `DEFAULT_COMMAND_REFERENCES_PER_LIST`,
`DEFAULT_COMMAND_PREPROCESS_PER_LIST`, `MAX_COMMAND_ALLOCATOR_CAPACITY`,
`MAX_COMMAND_REFERENCES_PER_LIST`, `MAX_COMMAND_PREPROCESS_PER_LIST`, and
`DEFAULT_ATTACHMENT_VIEW_CAPACITY`.

Methods: `CommandAllocatorHandle.is_valid`, `CommandAllocatorHandle.equals`,
`CommandList.is_valid`, `ExecutableCommandList.is_valid`,
`AttachmentViewHandle.is_valid`, and `AttachmentViewHandle.equals`.

Callables:
`create_command_allocator`, `destroy_command_allocator`,
`reserve_generated_scratch`, `release_generated_scratch`, `begin_commands`,
`end_commands`, `discard_commands`, `discard_executable_commands`,
`cmd_copy_buffer`, `cmd_fill_buffer`, `cmd_copy_buffer_to_texture`,
`cmd_copy_texture_to_buffer`, `cmd_build_acceleration_structure`,
`cmd_update_acceleration_structure`, `cmd_clone_acceleration_structure`,
`cmd_bind_pipeline`, `cmd_dispatch`,
`cmd_dispatch_indirect`, `cmd_dispatch_generated`, `cmd_trace_rays`,
`cmd_trace_rays_indirect`, `cmd_trace_rays_indirect2`,
`cmd_set_ray_tracing_pipeline_stack_size`,
`cmd_set_viewport`,
`cmd_set_scissor`, `cmd_draw`, `cmd_draw_indexed`, `cmd_draw_indirect`,
`cmd_draw_indexed_indirect`, `cmd_draw_indexed_indirect_count`,
`cmd_draw_generated`, `cmd_draw_indexed_generated`,
`create_attachment_view`, `destroy_attachment_view`,
`render_geometry_state`, `cmd_begin_render_pass`,
`cmd_set_graphics_state`, and `cmd_end_render_pass`.

### [Synchronization and submission](synchronization_and_submission.md)

Types: `StageMask`, `TextureLayout`, `TextureAccess`, `TextureState`,
`Barrier`, `TextureBarrier`, `CompletionPoint`, `CompletionWait`,
`SubmitDesc`, `TimestampPoolHandle`, and `TimestampPoolDesc`.

Constants: `COMPLETION_POINT_INVALID`, `TIMEOUT_INFINITE`, and
`TIMESTAMP_POOL_HANDLE_INVALID`.

Methods: `CompletionPoint.is_valid`, `CompletionPoint.equals`,
`TimestampPoolHandle.is_valid`, and `TimestampPoolHandle.equals`.

Callables: `sampled_at`,
`storage_at`, `texture_view_transition`, `texture_transition`,
`cmd_barrier`, `cmd_texture_barrier`, `submit`, `poll_completion`,
`wait_completion`, `create_timestamp_pool`, `destroy_timestamp_pool`,
`cmd_reset_timestamps`, `cmd_write_timestamp`, `cmd_resolve_timestamps`,
`read_timestamps`, and `timestamp_delta_ns`.

### [Presentation and diagnostics](presentation_and_diagnostics.md)

Types: `SwapchainHandle`, `SwapchainDesc`, `SwapchainInfo`,
`SwapchainReadiness`, `AcquiredImage`, `PresentMode`, `PresentModeSupport`,
`DebugMessageCallback`, `DebugMessageSeverity`, `DebugResourceKind`,
`DebugMessageCategory`, `DebugResourceRef`, and `DebugMessage`.

Constants: `SWAPCHAIN_HANDLE_INVALID`, `SWAPCHAIN_READINESS_INVALID`, and
`MAX_SWAPCHAINS`.

Surface-module types:
`gpu::surface::wayland::{DisplayHandle, SurfaceHandle}`,
`gpu::surface::x11::{DisplayHandle, WindowHandle}`, and
`gpu::surface::win32::{InstanceHandle, WindowHandle}`.

Methods: `SwapchainHandle.is_valid`, `SwapchainHandle.equals`,
`SwapchainReadiness.is_valid`, and `SwapchainReadiness.equals`.

Callables: each platform
`create_surface`, `destroy_surface`, `create_swapchain`, `destroy_swapchain`,
`resize_swapchain`, `get_swapchain_info`, `get_present_mode_support`,
`acquire_next_image`, `present`, `get_memory_stats`, `build_memory_report`,
`cmd_begin_label`, and `cmd_end_label`.

The internal constants `RESOURCE_OWNER_KIND_COUNT` and
`MAX_DEVICE_RESOURCE_OWNER` are present in the imported interface for handle
encoding but are not application configuration knobs.
