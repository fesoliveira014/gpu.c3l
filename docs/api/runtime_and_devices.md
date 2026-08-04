# Runtime and devices

## Runtime lifecycle

`RuntimeDesc` configures contract validation, optional Vulkan validation
layers, debug names and callback delivery, public table capacities, an optional
borrowed pipeline-cache blob, and an application name.
`acceleration_structure_heap_capacity` independently sizes the optional
shader-visible TLAS heap. It must be nonzero when a device requests ray
queries or ray-tracing pipelines and is otherwise unused.
`full_validation_runtime_desc()` returns a useful development baseline.
A zero descriptor selects `ContractValidation.TRUSTED` with layers and debug
names disabled.

`create_runtime` copies all configuration it needs after the call.
`create_runtime` and `destroy_runtime` are externally synchronized with all
runtime and surface registry mutation. `destroy_runtime` succeeds only after
its devices, surfaces, and borrowed adapter uses have ended.

Imports are inert: no runtime or backend object exists before
`create_runtime`.

## Adapters

`enumerate_adapters` returns an allocation-free `AdapterList` borrowed from the
runtime. `AdapterList.get` returns a borrowed `Adapter`; `AdapterInfo` and
`AdapterDiagnostics` contain borrowed strings valid until runtime destruction.
Enumeration and immutable queries are thread-safe after runtime publication.

`AdapterInfo` reports semantic class, memory totals, queue roles, and limits.
`AdapterDiagnostics` reports backend and driver identity for logging and
support. Applications should select through `DeviceDesc` rather than branching
on backend versions.

`supports_presentation` tests one adapter/surface pair.
`supports_device_desc` preflights the complete semantic request and returns
`DeviceSupport`; `create_device` remains authoritative because system state can
change.

## Device creation

`DeviceDesc` requests queue roles and optional presentation against one exact
surface. The runtime description supplies heap and resource capacities.
`create_device` verifies the adapter's required feature profile and exact
capacities before publishing a `Device`.

Ray queries and direct ray-tracing pipelines have independent explicit opt-ins:
`DeviceDesc.enable_ray_queries` and
`DeviceDesc.enable_ray_tracing_pipelines`. Either request enables their shared
acceleration-structure foundation; enabling one does not enable the other's
shader execution model.
`supports_device_desc` reports an unsupported request before creation when the
adapter lacks any required ray, acceleration-structure, descriptor, address,
command, queue, or vertex-format capability. A zero-initialized device
description keeps both optional feature families disabled.

Creation is transactional and externally synchronized per runtime. On failure,
no device or child is live. Common faults are `UNSUPPORTED_FEATURE`,
`INVALID_ARGUMENT`, `OUT_OF_HOST_MEMORY`, `OUT_OF_DEVICE_MEMORY`, and
`BACKEND_ERROR`.

`get_device_caps` returns the selected semantic profile, including:

- selected queue roles and asynchronous-compute availability;
- configured texture/sampler heap capacities;
- alignment and workload limits;
- indirect-count, generated-work, wireframe, sparse, and timestamp support;
- actionable `AccelerationStructureCaps` when either ray feature was enabled;
- `RayQueryCaps.enabled` only when ray queries were enabled;
- actionable `RayTracingPipelineCaps` only when ray-tracing pipelines were
  enabled;
- sampler limits; and
- the effective maximum color attachment count.

Query capabilities instead of hardcoding selected-device limits.

`AccelerationStructureCaps` reports the shared enabled state, selected TLAS
heap capacity, maximum geometry, primitive, and instance counts, and scratch
alignment. These are creation and allocator bounds, not suggestions.
`RayQueryCaps` reports its independent enabled bit.
`RayTracingPipelineCaps` reports the total direct-dispatch invocation limit,
`max_ray_dispatch_dimensions` for the selected device's per-axis launch limits,
recursion depth, shader-group handle/alignment/stride requirements, and the
hit-attribute limit. Disabled capability records are fully zero/false.
Exceeding an enabled bound returns `UNSUPPORTED_FEATURE` or `INVALID_ARGUMENT`
as documented by the operation.

`destroy_device` is thread-safe against ordinary device operations but does not
wait. It returns `RESOURCE_IN_USE` while public children remain and
`DEVICE_BUSY` while operations, queued work, or closing state prevent teardown.
On a retryable failure the handle remains live.

## Queues

`QueueKind` names `GRAPHICS`, `COMPUTE`, and `TRANSFER`. `QueueRequest`
controls which semantic roles a device must provide: `required` marks roles
that must be selected, and `distinct` marks required roles that must not alias
another required role.

When `single_queue` is `false` (the zero/default value), existing
`required`/`distinct` validation, default role normalization, and queue-family
preference order are unchanged. A zero queue request still selects the
documented default roles with the existing asynchronous-compute and transfer
preferences.

When `single_queue` is `true`, every role in `required` must resolve to one
exact selected queue identity. A nonzero `distinct` conflicts with this policy,
and an empty `required` set is invalid rather than being normalized to
defaults. `supports_device_desc` and `create_device` return
`INVALID_ARGUMENT` for either invalid policy. If the policy is valid but no
single identity can satisfy all required roles, support evaluation returns
`DeviceSupport.supported == false` and authoritative device creation returns
`UNSUPPORTED_FEATURE`.

With a presentation surface, single-queue selection also requires the
presentation queue to equal the same exact selected graphics identity. A
separate private presentation queue is not used for this policy; it remains a
fallback only when `single_queue` is `false`.

Appending `single_queue` changes the public C3 `QueueRequest` layout, so
consumers must rebuild against the updated package. Existing named and
zero-value initializers remain source-compatible. `QueueRequest` is not a C
ABI or host/shader ABI record, so generated shader records and C3/GLSL offsets
are unchanged.

`get_queue` returns the exact selected queue or `UNSUPPORTED_FEATURE`;
`QueueInfo` reports its semantic roles. A compute role may alias graphics, and
transfer may alias another selected queue.

Queues are borrowed from the device. `Queue.is_valid` and `Queue.equals` are
value operations. Queue-targeting `submit`, sparse bind, and presentation are
externally synchronized on the underlying native queue, so aliased roles share
one synchronization boundary.

## Surfaces

`Surface` is runtime-owned and is created by a platform module; details are in
[Presentation and diagnostics](presentation_and_diagnostics.md). A
`DeviceDesc` borrows its surface during support checks and device creation.
The created device accepts only that exact surface for swapchain creation.

## Handle and value helpers

`Runtime`, `Adapter`, `Device`, `Surface`, and `Queue` provide `is_valid`;
device and queue identities also support equality where declared.
`*_INVALID` constants are zero sentinels. These checks do not extend owner
lifetime or prove that a registry generation remains live.

`Vec2f`, `Vec4f`, and `Vec4u` are shared ABI vector aliases.
`SparseTextureCaps`, `TimestampCaps`, `AccelerationStructureCaps`,
`RayQueryCaps`, and `RayTracingPipelineCaps` are nested capability records
exposed through `DeviceCaps`.

## Fault and concurrency summary

| Operation | Concurrency | Ownership/call order |
|---|---|---|
| runtime create/destroy | externally synchronized process-wide | destroy after all children and borrowed calls |
| adapter enumeration/query | thread-safe | values and strings borrow the runtime |
| presentation/support preflight | externally synchronized when a surface is involved | surface and runtime must be live |
| device create | externally synchronized per runtime | descriptor is borrowed for the call |
| device caps/queue query | thread-safe | device must be live |
| device destroy | thread-safe operation on one target | wait work, destroy children, retry busy faults |

Device loss returns `DEVICE_LOST` from affected operations; peer devices remain
independent.
