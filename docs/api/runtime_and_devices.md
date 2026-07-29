# Runtime and devices

## Runtime lifecycle

`RuntimeDesc` configures contract validation, optional Vulkan validation
layers, debug names and callback delivery, public table capacities, an optional
borrowed pipeline-cache blob, and an application name.
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

Creation is transactional and externally synchronized per runtime. On failure,
no device or child is live. Common faults are `UNSUPPORTED_FEATURE`,
`INVALID_ARGUMENT`, `OUT_OF_HOST_MEMORY`, `OUT_OF_DEVICE_MEMORY`, and
`BACKEND_ERROR`.

`get_device_caps` returns the selected semantic profile, including:

- selected queue roles and asynchronous-compute availability;
- configured texture/sampler heap capacities;
- alignment and workload limits;
- indirect-count, generated-work, wireframe, sparse, and timestamp support;
- sampler limits; and
- the effective maximum color attachment count.

Query capabilities instead of hardcoding selected-device limits.

`destroy_device` is thread-safe against ordinary device operations but does not
wait. It returns `RESOURCE_IN_USE` while public children remain and
`DEVICE_BUSY` while operations, queued work, or closing state prevent teardown.
On a retryable failure the handle remains live.

## Queues

`QueueKind` names `GRAPHICS`, `COMPUTE`, and `TRANSFER`. `get_queue` returns
the exact selected queue or `UNSUPPORTED_FEATURE`; `QueueInfo` reports its
semantic roles. A compute role may alias graphics, and transfer may alias
another selected queue.

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
`SparseTextureCaps` and `TimestampCaps` are nested capability records exposed
through `DeviceCaps`.

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
