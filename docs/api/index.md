# Public API

The public module is `gpu`. Surface creation is in `gpu::surface::wayland`,
`gpu::surface::x11`, and `gpu::surface::win32`. Everything under
`gpu::internal` is private.

Each domain page walks through setup and use of its objects, then lists
contracts. The [symbol map](#symbol-map) at the bottom of this page assigns
every exported name to one domain.

## Conventions

**Handles.** Strongly typed values carrying a device identity and a
generation. Zero is the invalid sentinel (`*_INVALID` constants).
`is_valid` checks shape only; it does not prove the owner is live.
Equality compares identity.

**Optionals.** Every fallible call returns `T?` or `void?` with a fault
from the set below. Use `!` to propagate, `if (catch err = ...)` to handle.

```c3
gpu::Device device = gpu::create_device(&adapter)!;

gpu::AcquiredImage? image = gpu::acquire_next_image(&device, swapchain);
if (catch err = image) {
    if (err == gpu::WAIT_TIMEOUT) return;
    return err~;
}
```

**Descriptors.** Passed by pointer and borrowed for the call. Zero fields
select documented defaults. Some calls accept `null` for an all-default
descriptor.

**Failure.** Creation publishes nothing on failure. Destruction leaves the
handle valid on `RESOURCE_IN_USE` or `DEVICE_BUSY`. One-shot tokens
(`CommandList`, `ExecutableCommandList`, `SwapchainReadiness`,
`AcquiredImage`) are consumed only by the documented successful call.

**Concurrency.** Three categories. See
[architecture](../architecture.md#threading).

| Category | Applies to |
|---|---|
| externally synchronized | runtime and surface registry; one swapchain; submit, present, sparse bind on one native queue |
| thread-safe | adapter queries; resource, pipeline, and view operations; completion poll and wait |
| thread-confined | a recording token and its copies; the allocator while a recording is live |

**Validation.** `ContractValidation.FULL` adds diagnostics and retained
references. It never changes the behavior of a valid program.

## Faults

| Fault | Meaning | Retry |
|---|---|---|
| `UNSUPPORTED_BACKEND` | No Vulkan 1.3 driver or backend creation failed | no |
| `UNSUPPORTED_FEATURE` | Capability, layer, format, or capacity unavailable | no |
| `INVALID_ARGUMENT` | Descriptor or parameter rejected | fix input |
| `INVALID_HANDLE` | Zero, stale, consumed, or foreign handle | fix input |
| `INVALID_RESOURCE_STATE` | Transition or command state contradicts prior state | fix order |
| `OUT_OF_HOST_MEMORY`, `OUT_OF_DEVICE_MEMORY` | Allocation failed | release memory |
| `DEVICE_LOST` | Driver reported loss; the device rejects further work | recreate device |
| `DEVICE_BUSY` | Work still running or contention | wait, retry |
| `RESOURCE_IN_USE` | A live child or reference blocks destruction | release child, retry |
| `SLOT_TABLE_FULL` | Fixed public table full | destroy something |
| `DESCRIPTOR_HEAP_FULL` | No free heap slot | destroy a view |
| `COMMAND_ALLOCATOR_CAPACITY_EXCEEDED` | Fixed per-allocator bookkeeping exhausted | larger allocator |
| `GENERATED_SCRATCH_EXHAUSTED` | Reservation too small for the generated call | larger reservation |
| `PIPELINE_CREATE_FAILED` | Driver rejected the pipeline | no |
| `SHADER_INVALID` | SPIR-V or ABI mismatch | fix shader |
| `SURFACE_LOST` | Native surface gone | recreate surface |
| `SWAPCHAIN_OUT_OF_DATE` | Surface changed | resize |
| `COMMAND_RECORDING_ERROR` | Token phase or command misuse | fix order |
| `WAIT_TIMEOUT` | Bounded wait elapsed; nothing consumed | retry |
| `BACKEND_ERROR` | Unclassified backend failure; not device loss | no |

Faults that guard host safety, handle identity, command phase, and
lifecycle are always checked. `FULL` validation adds semantic diagnostics.
Runtime faults can occur under any policy.

## Symbol map

### [Runtime and devices](runtime_and_devices.md)

Types: `Vec2f`, `Vec4f`, `Vec4u`, `Device`, `Runtime`, `RuntimeDesc`,
`ContractValidation`, `Adapter`, `AdapterList`, `AdapterClass`,
`AdapterMemoryInfo`, `AdapterQueueInfo`, `AdapterLimits`, `AdapterInfo`,
`BackendVersion`, `AdapterDiagnostics`, `Surface`, `QueueKind`, `QueueRoles`,
`Queue`, `QueueInfo`, `QueueRequest`, `DeviceDesc`, `DeviceSupport`,
`DeviceCaps`, `TimestampCaps`, `SparseTextureCaps`,
`AccelerationStructureCaps`, `RayQueryCaps`, `RayTracingPipelineCaps`.

Constants: `DEVICE_INVALID`, `RUNTIME_INVALID`, `ADAPTER_INVALID`,
`SURFACE_INVALID`, `QUEUE_INVALID`.

Methods: `Device.is_valid`, `Runtime.is_valid`, `Adapter.is_valid`,
`Surface.is_valid`, `Queue.is_valid`, `Queue.equals`, `AdapterList.get`.

Functions: `full_validation_runtime_desc`, `create_runtime`,
`destroy_runtime`, `enumerate_adapters`, `get_adapter_info`,
`get_adapter_diagnostics`, `supports_presentation`, `supports_device_desc`,
`create_device`, `destroy_device`, `get_device_caps`, `get_queue`,
`get_queue_info`.

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
`AccelerationStructureInstance`, `AccelerationStructureInstanceDesc`,
`DedicatedAccelerationStructure`.

Constants: `GPU_ALLOCATION_INVALID`, `TEXTURE_HANDLE_INVALID`,
`TEXTURE_VIEW_INVALID`, `TEXTURE_INDEX_INVALID`, `SAMPLER_INDEX_INVALID`,
`ACCELERATION_STRUCTURE_HANDLE_INVALID`,
`ACCELERATION_STRUCTURE_VIEW_INVALID`,
`ACCELERATION_STRUCTURE_INDEX_INVALID`, `MAX_MEMORY_HEAPS`,
`DEFAULT_TEXTURE_CAPACITY`, `MAX_SPARSE_TEXTURE_ASPECTS`,
`DEFAULT_TEXTURE_HEAP_CAPACITY`, `DEFAULT_SAMPLER_HEAP_CAPACITY`,
`MAX_SHADER_HEAP_CAPACITY`.

Methods: `GpuAllocation.is_valid`, `GpuAllocation.equals`,
`TextureHandle.is_valid`, `TextureHandle.equals`, `TextureView.is_valid`,
`TextureView.equals`, `TextureIndex.is_valid`, `SamplerIndex.is_valid`,
`AccelerationStructureHandle.is_valid`,
`AccelerationStructureHandle.equals`, `AccelerationStructureView.is_valid`,
`AccelerationStructureView.equals`, `AccelerationStructureIndex.is_valid`,
`GpuSpan.unchecked_subspan`, `GpuSpan.checked_subspan`,
`MappedGpuSpan.checked_subspan`, `TextureCompatibility.is_valid`.

Functions: `allocate_memory`, `free_allocation`, `get_allocation_info`,
`get_allocation_span`, `get_span_mapping`, `get_span_address`,
`flush_mapped_span`, `invalidate_mapped_span`, `mapped_gpu_span`,
`get_memory_stats`, `build_memory_report`, `get_texture_format_support`,
`supports_texture_desc`, `get_texture_requirements`, `create_texture`,
`create_placed_texture`, `create_dedicated_texture`, `create_sparse_texture`,
`get_sparse_texture_requirements`, `bind_sparse_texture_memory`,
`destroy_texture`, `create_texture_view`, `create_texture_views`,
`destroy_texture_view`, `intern_sampler`,
`get_acceleration_structure_requirements`, `create_acceleration_structure`,
`create_placed_acceleration_structure`,
`create_dedicated_acceleration_structure`, `destroy_acceleration_structure`,
`get_acceleration_structure_address`, `make_acceleration_structure_instance`,
`create_acceleration_structure_view`, `destroy_acceleration_structure_view`.

### [Shaders and pipelines](shaders_and_pipelines.md)

Types: `RootPush`, `GraphicsRootPush`, `GeneratedDrawRecord`,
`GeneratedDrawIndexedRecord`, `GeneratedDispatchRecord`, `PipelineHandle`,
`PrimitiveTopology`, `CompareOp`, `CullMode`, `FrontFace`, `PolygonMode`,
`BlendFactor`, `BlendOp`, `ColorWriteMask`, `ShaderDesc`, `DepthState`,
`BlendState`, `ColorTargetState`, `ColorState`, `DynamicRasterState`,
`ComputePipelineDesc`, `GraphicsPipelineDesc`, `RayTracingHitGroupKind`,
`RayTracingGroupShader`, `RayTracingHitGroupDesc`, `RayTracingPipelineDesc`,
`RayTracingShaderGroupRange`, `RayTracingPipelineInfo`,
`RayTracingShaderBindingTableRegion`, `RayTracingShaderBindingTable`.

Constants: `PIPELINE_HANDLE_INVALID`, `COLOR_WRITE_ALL`, `MAX_PIPELINES`,
`MAX_COLOR_ATTACHMENTS`.

Methods: `PipelineHandle.is_valid`, `PipelineHandle.equals`.

Functions: `color_blend_disabled`, `alpha_blend`,
`premultiplied_alpha_blend`, `additive_blend`, `uniform_color_state`,
`create_compute_pipeline`, `create_graphics_pipeline`,
`create_ray_tracing_pipeline`, `get_ray_tracing_pipeline_info`,
`get_ray_tracing_shader_group_handles`,
`get_ray_tracing_shader_group_stack_size`, `destroy_pipeline`,
`get_pipeline_cache_size`, `get_pipeline_cache_data`.

### [Commands and rendering](commands_and_rendering.md)

Types: `CommandAllocatorHandle`, `CommandAllocator`, `CommandAllocatorDesc`,
`CommandList`, `ExecutableCommandList`, `GeneratedWorkKind`,
`GeneratedWorkReservationDesc`, `Vec3u`, `Viewport`, `ScissorRect`,
`GraphicsState`, `DrawIndirectCommand`, `DrawIndexedIndirectCommand`,
`DispatchIndirectCommand`, `TraceRaysIndirectCommand`,
`TraceRaysIndirectCommand2`, `AccelerationStructureIndirectBuildRange`,
`BufferCopyDesc`, `BufferTextureCopyDesc`, `TextureBufferCopyDesc`,
`AccelerationStructureTriangleBuildDesc`,
`AccelerationStructureAabbBuildDesc`,
`AccelerationStructureGeometryBuildDesc`, `AccelerationStructureBuildDesc`,
`IndexType`, `AttachmentViewHandle`, `AttachmentViewDesc`, `LoadOp`,
`StoreOp`, `ClearColor`, `ClearDepth`, `ColorTargetDesc`, `DepthTargetDesc`,
`RenderPassDesc`.

Constants: `COMMAND_ALLOCATOR_HANDLE_INVALID`,
`ATTACHMENT_VIEW_HANDLE_INVALID`, `DEFAULT_COMMAND_ALLOCATOR_CAPACITY`,
`DEFAULT_COMMAND_REFERENCES_PER_LIST`, `MAX_COMMAND_ALLOCATOR_CAPACITY`,
`MAX_COMMAND_REFERENCES_PER_LIST`, `DEFAULT_ATTACHMENT_VIEW_CAPACITY`.

Methods: `CommandAllocatorHandle.is_valid`, `CommandAllocatorHandle.equals`,
`CommandList.is_valid`, `ExecutableCommandList.is_valid`,
`AttachmentViewHandle.is_valid`, `AttachmentViewHandle.equals`.

Functions: `create_command_allocator`, `destroy_command_allocator`,
`reserve_generated_work`, `release_generated_work`, `begin_commands`,
`end_commands`, `discard_commands`, `discard_executable_commands`,
`cmd_begin_label`, `cmd_end_label`, `cmd_copy_buffer`, `cmd_fill_buffer`,
`cmd_copy_buffer_to_texture`, `cmd_copy_texture_to_buffer`,
`cmd_build_acceleration_structure`, `cmd_update_acceleration_structure`,
`cmd_build_acceleration_structure_indirect`,
`cmd_update_acceleration_structure_indirect`,
`cmd_clone_acceleration_structure`, `cmd_bind_pipeline`, `cmd_dispatch`,
`cmd_dispatch_indirect`, `cmd_dispatch_generated`, `cmd_trace_rays`,
`cmd_trace_rays_indirect`, `cmd_trace_rays_indirect2`,
`cmd_set_ray_tracing_pipeline_stack_size`, `cmd_set_viewport`,
`cmd_set_scissor`, `cmd_draw`, `cmd_draw_indexed`, `cmd_draw_indirect`,
`cmd_draw_indexed_indirect`, `cmd_draw_indexed_indirect_count`,
`cmd_draw_generated`, `cmd_draw_indexed_generated`,
`create_attachment_view`, `destroy_attachment_view`,
`render_geometry_state`, `cmd_begin_render_pass`, `cmd_set_graphics_state`,
`cmd_end_render_pass`.

### [Synchronization and submission](synchronization_and_submission.md)

Types: `StageMask`, `TextureLayout`, `TextureAccess`, `TextureState`,
`Barrier`, `TextureBarrier`, `CompletionPoint`, `CompletionWait`,
`SubmitDesc`, `TimestampPoolHandle`, `TimestampPoolDesc`.

Constants: `COMPLETION_POINT_INVALID`, `TIMEOUT_INFINITE`,
`TIMESTAMP_POOL_HANDLE_INVALID`.

Methods: `CompletionPoint.is_valid`, `CompletionPoint.equals`,
`TimestampPoolHandle.is_valid`, `TimestampPoolHandle.equals`.

Functions: `sampled_at`, `storage_at`, `texture_view_transition`,
`texture_transition`, `cmd_barrier`, `cmd_texture_barrier`, `submit`,
`poll_completion`, `wait_completion`, `create_timestamp_pool`,
`destroy_timestamp_pool`, `cmd_reset_timestamps`, `cmd_write_timestamp`,
`cmd_resolve_timestamps`, `read_timestamps`, `timestamp_delta_ns`.

### [Presentation and diagnostics](presentation_and_diagnostics.md)

Types: `SwapchainHandle`, `SwapchainDesc`, `SwapchainInfo`,
`SwapchainReadiness`, `AcquiredImage`, `PresentMode`, `PresentModeSupport`,
`DebugMessageCallback`, `DebugMessageSeverity`, `DebugResourceKind`,
`DebugMessageCategory`, `DebugResourceRef`, `DebugMessage`.

Surface-module types: `gpu::surface::wayland::{DisplayHandle,
SurfaceHandle}`, `gpu::surface::x11::{DisplayHandle, WindowHandle}`,
`gpu::surface::win32::{InstanceHandle, WindowHandle}`.

Constants: `SWAPCHAIN_HANDLE_INVALID`, `SWAPCHAIN_READINESS_INVALID`,
`MAX_SWAPCHAINS`.

Methods: `SwapchainHandle.is_valid`, `SwapchainHandle.equals`,
`SwapchainReadiness.is_valid`, `SwapchainReadiness.equals`.

Functions: each platform `create_surface`, `destroy_surface`,
`create_swapchain`, `wait_swapchain_presentations`, `destroy_swapchain`,
`resize_swapchain`, `get_swapchain_info`, `get_present_mode_support`,
`acquire_next_image`, `present`.

`RESOURCE_OWNER_KIND_COUNT` and `MAX_DEVICE_RESOURCE_OWNER` appear in the
interface for handle encoding and are not configuration.
