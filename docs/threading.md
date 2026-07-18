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
- **S — thread-safe.** Any thread, any time within a frame; internally
  synchronized (see the lock map).
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
| `begin_frame` / `end_frame` / `@with_frame` | E | token-paired `IDLE -> ACTIVE -> IDLE`; quiescence required; helper worker is a direct call |
| `submit` / `present` | E | externally synchronize each acquired image; submit consumes readiness, present consumes the image |
| `poll_completion` / `wait_completion` | S | reusable value queries; host waits do not consume the point |
| `create_swapchain` / `destroy_swapchain` | E | process-wide surface registry access; destruction invalidates a pending acquire and waits its render work |
| `resize_swapchain` / `get_swapchain_info` / `get_present_mode_support` / `acquire_next_image` | E | per swapchain; resize rejects a pending acquire and waits graphics/present work |
| `allocate_memory` / `free_allocation` | S | internally synchronized; free must happen-after the last use |
| `get_allocation_info` / `get_allocation_span` / `get_span_mapping` / `get_span_address` | S | lock-free slot resolution |
| `flush_mapped_span` / `invalidate_mapped_span` | S | lock-free validation; coherent no-op; native calls are internally synchronized |
| `create_texture` / `destroy_texture` | S | |
| `create_texture_descriptor` / `destroy_texture_descriptor` | S | |
| `create_sampler` / `destroy_sampler` | S | |
| `create_shader` / `destroy_shader` | S | |
| `create_compute_pipeline` / `create_graphics_pipeline` / `destroy_pipeline` | S | driver compiles run in parallel; a same-key race compiles twice, converges to one entry |
| `alloc_frame_span` | S | lock-free CAS bump through a current `FrameToken`; token copies may be shared during the active generation |
| `alloc_persistent_span` / `free_persistent_span` | S | VMA virtual blocks are not internally synchronized; the library locks them |
| `upload_buffer_data` / `upload_texture_data` / `readback_buffer_data` / `readback_texture_data` | S | use a separate private pool serialized by `helper_record_mutex` |
| `cmd_readback_buffer` / `cmd_readback_texture` | C | records into the caller's list |
| `poll_readback` | S | lock-free |
| `resolve_readback` | S | |
| `get_memory_stats` / `build_memory_report` / `get_persistent_stats` | S | advisory: values may be inconsistent under concurrent mutation; quiesce externally for exact snapshots |
| `begin_commands` / `end_commands` / command discard | C | recording storage is automatic per worker; discard also cancels attached readback tickets |
| every `cmd_*` recording call | C | confined to the list's thread |
| `cmd_begin_label` / `cmd_end_label` | C | no-ops without debug-utils |

Most public device operations take a short-lived atomic pin. `begin_commands`
transfers its pin to the recording token. Recording calls and end/discard borrow
that pin. Successful end transfers ownership to the executable token; successful
submit or executable discard releases it.
Pin acquisition may return `DEVICE_BUSY`; failed destruction restores the live
state and preserves the token and generation.

`begin_commands` lazily allocates one recording context per thread/device pair.
Each device reserves one helper context and can allocate 256 recording-thread
contexts over its lifetime. A further distinct recording thread receives
`SLOT_TABLE_FULL`; contexts are released when the device is destroyed.

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

## Phase rule

No Tier S or Tier C call may be in flight across `begin_frame` / `end_frame`.
The frame-owner thread begins and ends the token generation; workers may receive
copies or the shared token pointer only after begin and must quiesce before end.
With `enable_validation`, in-flight Tier S calls at the boundary fault
`INVALID_RESOURCE_STATE`. A live unsubmitted command record prevents its frame
slot from resetting; submit or discard it first. Without validation the phase
rule is still contractual — Tier S calls pay nothing.

Frame lifecycle errors are independent of validation: double begin faults
`INVALID_RESOURCE_STATE`; malformed, consumed, and stale frame tokens fault
`INVALID_HANDLE`. Every rejection is mutation-free. A successful end consumes
the frame generation and all token aliases; a failed end retains the caller's
token and boundary state for retry.

`@with_frame(&frame, &device, named_worker, ...args)` is Tier E around the
entire worker. It invokes the symbol directly, with no runtime callback or
virtual dispatch, then attempts end exactly once even when the worker returns a
fault through `!`. An end fault wins and leaves `frame` live for retry.
Off-frame `submit` and `present` remain allowed. Before device destruction,
wait the latest `CompletionPoint` for each affected queue and destroy every
swapchain.

## Lock order

`helper_record_mutex → transfer_mutex → resource_mutex → command_mutex`, one
direction only. Creation and destruction share a single `resource_mutex`
(cold paths); command-record allocation, submit claims, and reclamation share
`command_mutex`; and the transfer arenas share `transfer_mutex`.
`helper_record_mutex` spans a blocking helper's recording window
(`begin_commands` through `end_commands`) and is outermost because that
window takes `transfer_mutex` internally for its allocation.

Each selected queue identity owns one completion timeline and submission mutex.
Roles that alias one native queue share that state. Submit releases
`command_mutex` before locking the selected queue, checks completion headroom,
reserves the next sequence immediately before the native call, and publishes only
after success. Native failure rolls the reservation back while still locked. The
queue mutex is released before public command tokens are invalidated, so
command-record and queue locks are never nested.

Dedicated-fallback staging/readback buffers (the arena ring miss path in
`transfer_alloc`/`ticket_alloc`) create their VMA buffer without holding
`transfer_mutex` — only the ring-capacity check and bookkeeping around it are
locked — so a fallback's backend allocation never blocks concurrent arena
allocations. A relock re-check covers the ring having filled during that
unlocked window; on that fault the fresh buffer is destroyed before the
call faults `ARENA_FULL`.

## Single-recorder texture discipline

Within a frame, one thread owns a given texture's barriers and render
passes. Under this discipline tracked-layout validation is exact and
cross-texture parallel recording is data-race-free. Violating it degrades
layout validation to best-effort (stale verdicts) — never memory unsafety.

Texture-layout transitions are staged per list at record time (`old_layout`
validates against the recording list's own pending transitions first, else
the tracked layout) and only commit onto tracked state when the list
submits, under the queue mutex, in submission order. A list that is recorded
and never submitted has no effect on tracked state. Commit order is submit
order, not cross-queue execution order — per-queue ownership of a texture
stays the caller's responsibility.

## Visibility rules

- Resource slot reads (`get` paths), including allocation and span queries,
  are lock-free: tables never reallocate, and a token reaches another thread
  only through your synchronization. That hand-off is the happens-before edge.
- Command records also live in a fixed table. The public `CommandList` is an
  owner-bearing handle into that table; passing it from a recording thread to
  the submitting thread is the required hand-off. Copies remain aliases and
  must not be used concurrently. Its embedded `Device` value is independent of
  the caller variable passed to `begin_commands`.
- Destruction must happen-after the last use of the handle on any thread.

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

## Frame retirement across queues

Each distinct compute/transfer queue has its own auxiliary timeline
(`aux_compute_timeline`, `aux_transfer_timeline`), signaled from that queue
alone — monotonic by construction, since a single queue's submissions
execute in submission order. `submit` piggybacks one internal aux signal
(`ALL_COMMANDS`, per-queue counter) onto every user submit on a distinct
compute or transfer queue; the counter and its last-signaled value update
only after the submit succeeds, under the queue mutex already held, so a
rejected submit never leaves anything waiting on a value that will never
signal. `end_frame` issues a single empty submit: the graphics-side frame
signal, which waits each used queue's latest recorded aux value before
signaling the frame value. A host-side wait on the frame value therefore
covers every queue's work — arenas, command pools, and
readback tickets stay safe under any queue topology, off-frame submissions
included. `submit` sets the per-queue used flags unconditionally, not just
while a frame is active, so an off-frame submission on a distinct compute or
transfer queue is still waited by the next `end_frame`'s chain (off-frame
graphics-queue submits need no flag: the frame signal already runs on that
queue, and Vulkan's per-queue submission order covers them for free). See
"Off-frame submissions" below for the destruction-side half of this
contract.

`end_frame` builds its prospective frame value and wait chain without mutation,
then commits retirement bookkeeping only after the graphics signal submit is
accepted. A rejected signal leaves the active boundary exactly retryable.

## Helper timeline

Blocking helpers (`upload_buffer_data`, `upload_texture_data`,
`readback_buffer_data`, `readback_texture_data`) never touch `frame_timeline`
or the frame counter. Each reserves a value on a separate `helper_timeline`
under `transfer_mutex`, before its (single) transfer allocation, and tags
that allocation's arena range or dedicated buffer with it. At completion
(after waiting for the helper submission's completion point) it signals
`helper_timeline` to that value — turnstiled: it first waits for `helper_timeline` to reach
`value - 1`, so concurrent helpers' completions land in strictly increasing
order even when they finish out of reservation order, and one helper's
completion can never retire another helper's or an unsubmitted list's
resources. This turnstile wait carries no lock (it would deadlock a slower
predecessor out of the very primitive it needs to reach its own signal — the
wait-for-predecessor is itself the ordering guarantee) and is generous but
finite; a helper that faults after reserving its value still signals it on
every exit path, so one stuck helper costs its immediate successor one
timeout rather than an unbounded stall. Frame-scoped paths
(`cmd_upload_buffer`, `cmd_upload_texture`, `cmd_readback_buffer`,
`cmd_readback_texture`) are unaffected: they still tag `frame_timeline` at
`counter + 1`, retired only by `end_frame`. See docs/memory.md §13.1.
Before recording, each blocking helper also checks under
`helper_record_mutex` whether `helper_timeline` already reads its reserved
value minus one — every predecessor complete — and if so resets the helper
context's pool for the current frame slot, bounding it to one live buffer
under sequential use even when no frame loop ever runs; the phase rule (no
Tier S call in flight across `begin_frame`/`end_frame`) keeps this from
racing `begin_frame`'s own per-slot reset.

## Off-frame submissions

Tier E's `submit`/`present` may run outside a `begin_frame`/`end_frame`
bracket — sanctioned for frame-loop-free apps and one-shot setup work.
Resource destruction is immediate and never waits. Discard recording or
executable command tokens and wait for every returned `CompletionPoint` that
may reference the resource before destroying it. Validation returns
`RESOURCE_IN_USE` for detected explicit references. `destroy_device` queries
every published queue sequence without blocking and returns `DEVICE_BUSY`
while any is incomplete.
