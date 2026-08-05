# Shaders and pipelines

## Shader input

`ShaderDesc` borrows SPIR-V bytes and an entry-point name for one pipeline
creation call. The bytes and string need not remain live after the call.
Pipeline creation validates the selected entry point's stage, interfaces,
root-push layout, and bindless heap declarations.

Root records and the exact push contract are documented in
[Shader ABI](../shader_abi.md). `RootPush`, `GraphicsRootPush`,
`GeneratedDrawRecord`, `GeneratedDrawIndexedRecord`, and
`GeneratedDispatchRecord` are generated public wire types.

## Compute pipelines

`ComputePipelineDesc` supplies one compute shader and the state that contributes
to deterministic pipeline identity. `create_compute_pipeline` returns a
device-owned `PipelineHandle`. Equivalent descriptions may converge on one
private cached pipeline identity while each successful public owner follows the
documented handle lifetime.

## Graphics pipelines

`GraphicsPipelineDesc` supplies vertex and fragment shaders, primitive
topology, ordered color formats, depth format, and sample count. State that can
change without a new pipeline belongs to `DynamicRasterState` and
`GraphicsState`, not to a native pipeline variant.

Relevant types are:

- `PrimitiveTopology`, `CullMode`, `FrontFace`, and `PolygonMode`;
- `CompareOp` and `DepthState`;
- `BlendFactor`, `BlendOp`, `BlendState`, `ColorWriteMask`,
  `ColorTargetState`, and `ColorState`; and
- `DynamicRasterState`.

`MAX_COLOR_ATTACHMENTS` is the library ceiling; use the lower
`DeviceCaps.max_color_attachments` on the selected device.

`color_blend_disabled`, `alpha_blend`,
`premultiplied_alpha_blend`, and `additive_blend` build common
`ColorTargetState` values. `uniform_color_state` repeats one target state over
an exact color-format count. `COLOR_WRITE_ALL` enables all channels.

## Ray-tracing pipelines

`RayTracingPipelineDesc` supplies ordered ray-generation shaders, miss
shaders, structured hit groups, callable shaders, and a recursion depth. The
supported roles are ray generation, miss, closest hit, any hit, intersection,
and callable. Every shader uses the same eight-byte `RootPush` contract and the
global heap layout; include `ray_tracing.glsl` for binding 5 without enabling
the separate ray-query shader extension.

`RayTracingHitGroupDesc` selects `TRIANGLES` or `PROCEDURAL`. A triangle group
may contain closest-hit and any-hit shaders and must not contain an
intersection shader. A procedural group requires an intersection shader and
may also contain closest-hit and any-hit shaders. Empty optional shader
pointers mean that role is absent.

`create_ray_tracing_pipeline` requires
`DeviceDesc.enable_ray_tracing_pipelines`. Zero recursion depth normalizes to
one; higher values must fit
`DeviceCaps.ray_tracing_pipelines.max_recursion_depth`.
`get_ray_tracing_pipeline_info` returns deterministic ray-generation, miss,
hit, and callable `RayTracingShaderGroupRange` values in that order.
`get_ray_tracing_shader_group_handles` copies one exact contiguous range into
caller storage whose length is `group_count * shader_group_handle_size`.

Ray-tracing pipelines do not support pipeline libraries/linking, capture
replay, deferred creation, dynamic stack sizing, Indirect2, or batched trace
commands. Basic indirect tracing changes only where dimensions are read; SBT
allocation and packing remain application responsibilities.

## Ownership and destruction

`destroy_pipeline` releases one pipeline owner only after all recording,
executable, and submitted references have retired. It does not wait.
Under full validation, a retained command reference returns
`RESOURCE_IN_USE`; under trusted validation the application must still keep the
pipeline live through use.

Pipeline functions are thread-safe. Creation is serialized only around the
device's internal pipeline/cache domain. The descriptor and shader data are
borrowed for the call; the returned handle owns the published result.

## Pipeline cache

`RuntimeDesc.pipeline_cache_data` is a borrowed opaque driver blob copied at
runtime creation and applied to later device creation.
`get_pipeline_cache_size` returns the current byte count.
`get_pipeline_cache_data` copies into caller storage and reports the required
size through the function's normal result contract.

The blob is private to the driver/device compatibility domain. Treat import
failure as a cache miss according to the returned fault and keep application
metadata such as device/driver identity alongside persisted data. Cache size
and usefulness vary by driver.

## Fault behavior

- malformed SPIR-V, missing/wrong entry points, or ABI mismatches return
  `SHADER_INVALID`;
- unsupported formats, sample counts, raster features, or selected-device
  requirements return `UNSUPPORTED_FEATURE`;
- inconsistent descriptors return `INVALID_ARGUMENT`;
- driver pipeline failure returns `PIPELINE_CREATE_FAILED`; and
- fixed capacity exhaustion returns `SLOT_TABLE_FULL`.

Creation is transactional: no output handle or cache entry is published after
a failed validation or native create.
