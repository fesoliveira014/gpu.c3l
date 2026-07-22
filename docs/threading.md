# Threading model

Every public entry point belongs to one of three tiers. Anything not sanctioned
here is misuse; results and validation verdicts are undefined. The library's
own state remains memory-safe except when concurrent use of the same Tier C
token races its consumption and device destruction: consumption releases the
token's retained device pin while another call can still be in flight.

## Tiers

- **E — externally synchronized.** The caller serializes calls within the
  scope named in the table. Device calls normally share one device-wide scope;
  runtime and surface calls may use a process-wide scope.
- **S — thread-safe.** Any thread while the owning runtime or device is live;
  internally synchronized (see the lock map).
- **C — confined.** A recording or executable command token and its aliases
  are used by one thread at a time. Different tokens may be recorded in
  parallel.

## Per-entry-point table

| Entry point | Tier | Notes |
|---|---|---|
| `create_runtime` / `destroy_runtime` | E | process-wide runtime registry mutation |
| `enumerate_adapters` / `AdapterList.get` / adapter queries | S | immutable cache reads; borrowed strings are read-only |
| `Surface.is_valid` / `strict_device_request` | S | pure value operations; no registry access |
| `surface::{win32,wayland,x11}::create_surface` / `destroy_surface` | E | process-wide surface registry mutation |
| `supports_presentation` / `request_presentation` / `supports_device_request` | E | process-wide surface registry access; a presentation request does not retain its surface |
| `create_device` | E | per runtime; device-registry mutation is synchronized; presentation also uses the surface scope |
| `destroy_device` | S | per target device; success invalidates the caller's token; live children return `RESOURCE_IN_USE`, while active operations, incomplete queue work, or closing state return retryable `DEVICE_BUSY` |
| `submit` / `present` | E | externally synchronize each acquired image; submit consumes readiness, present consumes the image |
| `poll_completion` / `wait_completion` | S | reusable value queries; host waits do not consume the point |
| `create_swapchain` / `destroy_swapchain` | E | process-wide surface registry access; destruction rejects pending acquisition or use and never waits |
| `resize_swapchain` / `get_swapchain_info` / `get_present_mode_support` / `acquire_next_image` | E | per swapchain; resize rejects pending acquisition or use and never waits |
| `allocate_memory` / `free_allocation` | S | internally synchronized; free must happen-after the last use |
| `get_allocation_info` / `get_allocation_span` / `get_span_mapping` / `get_span_address` | S | lock-free slot resolution |
| `flush_mapped_span` / `invalidate_mapped_span` | S | lock-free validation; coherent no-op; native calls are internally synchronized |
| `create_texture` / `destroy_texture` | S | |
| `create_attachment_view` / `destroy_attachment_view` | S | immutable render subresource; destroy happens-after every command reference |
| `create_texture_view` / `create_texture_views` / `destroy_texture_view` | S | owner and generation are validated before heap mutation |
| `intern_sampler` / `publish_sampler` | S | equal interning and repeated publication converge under the device resource lock |
| `prepare_shader_code` | S | pure read of caller-owned immutable bytes and strings |
| `create_compute_pipeline` / `create_graphics_pipeline` / `create_compute_pipelines` / `create_graphics_pipelines` / `destroy_pipeline` | S | single and batch creation serialize on a device-wide creation lock; a same-key request converges to one entry |
| `get_memory_stats` / `build_memory_report` | S | advisory: values may be inconsistent under concurrent mutation; quiesce externally for exact snapshots |
| `begin_commands` / `end_commands` / command discard | C | recording storage is automatic per worker |
| `reserve_generated_scratch` / `release_generated_scratch` | C | cold mutation of one pipeline/kind reservation in the calling thread's device context; the queue selects the device, and the context must be quiescent |
| every `cmd_*` recording call | C | confined to the list's thread |
| `cmd_begin_label` / `cmd_end_label` | C | no-ops without debug-utils |

Most public device operations take a short-lived atomic pin. `begin_commands`
transfers its pin to the recording token and publishes a stable encoder cell.
Recording calls validate that cell directly and do not borrow a registry pin.
Successful end transfers the same cell and pin ownership to the executable
token; successful `submit` or executable discard invalidates the cell before
releasing the pin.
Pin acquisition may return `DEVICE_BUSY`; failed destruction restores the live
state and preserves the token and generation.

`begin_commands` lazily allocates one recording context per thread/device pair.
Each device can allocate 256 recording contexts over its lifetime. A further
distinct recording thread receives
`SLOT_TABLE_FULL`; contexts are released when the device is destroyed.
Completed or discarded native command buffers return to the same context and
are reset and reused only by its owner. A worker must reserve its own generated
scratch before recording generated commands; reservations are not shared or
stolen across worker contexts. Each reservation is also specific to the public
pipeline handle, so aliases of one native pipeline do not share it. Release
every reservation before a worker
exits: an exited worker's reservations stay held by its context, and the
reserved pipelines remain retained until the device is destroyed.

Runtime creation and destruction must not overlap other runtime operations. After
publication, enumeration and adapter queries may run concurrently; all such calls
must finish before runtime destruction.

The surface registry and its runtime retain counts are process-wide and have no
internal synchronization. Calls identified above as registry mutations or
accesses must not overlap each other, `create_runtime`, or `destroy_runtime`,
even for different runtimes, devices, or surfaces. `present`,
`resize_swapchain`, swapchain queries, and acquire do not access that registry
and remain externally synchronized per
device or swapchain.

`create_surface` retains its runtime. A presentation-bearing `DeviceRequest`
and the created device store only the surface token, so the surface must remain
live through support checks, device creation, and swapchain creation. A device
accepts only that exact surface. `create_swapchain` retains the surface until
`destroy_swapchain` or device teardown; `destroy_surface` returns
`RESOURCE_IN_USE` while any swapchain retains it. The application keeps the
native instance, display, and window objects valid until the surface is
destroyed.

## Operation concurrency

There is no global application work phase in the root module. Calls may overlap
only according to their entry-point tiers and token ownership rules. A Tier C
command token and all its aliases stay confined to one thread at a time;
different command tokens may be recorded in parallel. Tier S allocation and
span operations may overlap, but callers synchronize writes to mapped storage
and keep every allocation live through its last submitted use.

Submission consumes executable command tokens only after native acceptance and
returns a reusable `CompletionPoint`. The caller retains that point whenever it
guards resource reuse, command-dependent destruction, or cross-queue ordering.
Before device destruction, wait for the latest point on every used queue and
destroy every swapchain and child resource.

## Lock order

Resource creation and destruction use `resource_mutex`; shader-visible
texture-view cache publication uses `texture_view_cache_mutex`; command-record
allocation, submit claims, and reclamation use `command_mutex`.
When both resource and view-cache locks are needed, resource comes first.
Submission releases `command_mutex` before locking the selected queue, so
command-record and queue locks are not nested.
Queue submission may acquire `resource_mutex` while holding its
`submit_mutex` to publish or retire submitted-command metadata. This is the
only queue/resource nesting order: `submit_mutex` precedes `resource_mutex`,
and resource-locked paths never acquire a queue submission mutex.

## Texture-transition discipline

Texture transitions contain the caller-declared previous and next semantic
uses. Recording does not consult or mutate shared layout state. Concurrent
recording is therefore independent, but the caller must keep each declaration
consistent with execution order and use completion-point waits for cross-queue
ownership transfers.

## Visibility rules

- Resource slot reads (`get` paths), including allocation and span queries,
  are lock-free: tables never reallocate, and a token reaches another thread
  only through your synchronization. That hand-off is the happens-before edge.
- Command records and root encoder cells live in fixed tables. `CommandList`
  carries the owner-bearing handle and an opaque pointer to its exact cell.
  Publication, recording-to-executable transfer, and invalidation follow Tier C
  confinement. Passing the token through caller synchronization is the required
  hand-off and makes the published cell visible; copies remain aliases and must
  not be used concurrently. Its embedded `Device` value is independent of the
  caller variable passed to `begin_commands`.
- Pipeline slots live in a fixed table and carry every native layout needed by
  recording. Pipeline creation may grow packed layout-cache storage while
  another thread records with an existing pipeline because recording does not
  inspect that cache. Bind snapshots the stable slot identity and native state;
  later commands validate the cached generation without resolving the table.
- Destruction must happen-after the last use of the handle on any thread.
- Attachment-view slots live in a fixed table and retain their texture. Render
  recording reads immutable view metadata without the shader-visible view-cache
  lock and records a reference before the handle may be destroyed.

## Worker-thread setup (C3)

C3 only creates the implicit temp allocator on the main thread. Library
paths allocate temporaries internally, so worker threads calling into the
library must wrap their body in `@pool_init(...)` (see `std::core::mem`) or
they abort on first temp allocation.

## Debug callback discipline

Public-contract and ordinary backend `DebugMessage` values are delivered
synchronously on the calling thread. Vulkan debug-utils messages are delivered
synchronously on the arbitrary application or driver thread chosen by Vulkan;
multiple callbacks may run concurrently, with no cross-thread ordering or
library serialization.

The callback must be nonblocking and must not call gpu.c3l. Delivery can occur
while internal resource or queue locks are held, so reentry may deadlock. Copy
borrowed fields into application-owned synchronized storage and return. The
callback and `debug_user_data` remain valid from runtime or device creation entry
through the matching destroy return; no callback occurs afterward.

## Miscellany

- A command token is confined, not locked.
- Executable tokens recorded for the same queue may share one `SubmitDesc`.

## Completion across queues

Each selected queue identity owns one private timeline. A successful `submit`
signals its next value and returns a `CompletionPoint` for that queue. Same-queue
submissions are ordered by the queue. Cross-queue dependencies are explicit in
`SubmitDesc.completion_waits`; each wait names its first destination stages,
and no application work boundary adds waits or signals. Same-queue waits are
still stage-validated before the redundant native wait is elided.

Each queue release-publishes one strong retired prefix. Sequence N is retired
only after native completion and after every published submitted-command batch
through N has released validation references, recycled generated scratch, and
retired its command buffer. Poll and wait acquire-load that prefix after point
validation; an already-retired point performs no native call and acquires
neither the queue nor resource mutex.

Native progress is capped to the acquire-loaded contiguous published prefix
before retirement. This prevents a native submission paused before metadata or
point publication from exposing or releasing its predicted sequence. A first
poll miss may query the queue timeline and retire the observed published
prefix. A successful wait retires exactly its requested sequence; timeout
preserves both the point and retired prefix. Headroom and threshold drains
advance the same strong prefix, with no weaker completion cache.

Resource storage, raw addresses, shader indices, and mapped data remain the
caller's responsibility: retain every owning token until all points covering
its use have completed.

## Submission lifetime

`submit` and `present` are Tier E. Externally synchronize each queue and
acquired image. Resource destruction is immediate and never waits. Discard
recording or executable command tokens and wait for every returned
`CompletionPoint` that may reference a resource before destroying it.
Validation returns `RESOURCE_IN_USE` for detected explicit references.
`destroy_device` queries every published queue sequence without blocking and
returns `DEVICE_BUSY` while any is incomplete.
