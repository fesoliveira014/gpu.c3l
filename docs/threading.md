# Threading model

Every public entry point belongs to one of three tiers. Anything not
sanctioned here is documented misuse — it will not corrupt memory through the
library's own state, but results and validation verdicts are undefined.

## Tiers

- **E — externally synchronized.** One thread at a time across all Tier E
  calls on a device. Typically the "frame owner" thread.
- **S — thread-safe.** Any thread, any time within a frame; internally
  synchronized (see the lock map).
- **C — confined.** The object (a `CommandList` and the `RecordingContext` it
  came from) is used by one thread at a time. Different contexts record in
  parallel freely.

## Per-entry-point table

| Entry point | Tier | Notes |
|---|---|---|
| `create_device` / `destroy_device` | E | |
| `begin_frame` / `end_frame` | E | quiescence required, see phase rule |
| `submit` / `present` | E | queue-mutex backed, so Tier S private submits interleave safely |
| `wait_queue_idle` | E | queue-mutex backed |
| `create_swapchain` / `destroy_swapchain` / `resize_swapchain` / `acquire_next_image` | E | per swapchain |
| `create_buffer` / `destroy_buffer` | S | |
| `get_buffer_address` / `get_buffer_span` | S | lock-free read |
| `flush_buffer` / `invalidate_buffer` | S | VMA is internally synchronized |
| `create_texture` / `destroy_texture` | S | |
| `create_texture_descriptor` / `destroy_texture_descriptor` | S | |
| `create_sampler` / `destroy_sampler` | S | |
| `create_shader` / `destroy_shader` | S | |
| `create_compute_pipeline` / `create_graphics_pipeline` / `destroy_pipeline` | S | driver compiles run in parallel; a same-key race compiles twice, converges to one entry |
| `create_semaphore` / `destroy_semaphore` | S | |
| `wait_semaphore` | S | lock-free |
| `alloc_frame_span` | S | lock-free CAS bump |
| `alloc_persistent_span` / `free_persistent_span` | S | VMA virtual blocks are not internally synchronized; the library locks them |
| `create_recording_context` / `destroy_recording_context` | S | destroy requires the context's lists retired |
| `upload_buffer_data` / `upload_texture_data` / `readback_buffer_data` / `readback_texture_data` | S | record on a dedicated internal context, serialized by helper_record_mutex; never share a pool with Tier-C recording |
| `cmd_readback_buffer` / `cmd_readback_texture` | C | records into the caller's list |
| `poll_readback` | S | lock-free |
| `resolve_readback` | S | |
| `get_memory_stats` / `build_memory_report` / `get_persistent_stats` | S | advisory: values may be inconsistent under concurrent mutation; quiesce externally for exact snapshots |
| `begin_commands` / `end_commands` | C | confined to the context's thread |
| every `cmd_*` recording call | C | confined to the list's thread |
| `cmd_begin_label` / `cmd_end_label` | C | no-ops without debug-utils |

## Phase rule

No Tier S or Tier C call may be in flight across `begin_frame` / `end_frame`.
With `enable_validation`, in-flight Tier S calls at the boundary fault
`INVALID_RESOURCE_STATE`. An unsubmitted command record is reclaimed when its
frame slot retires and its command pool resets; every alias of that token then
faults `INVALID_HANDLE`. Without validation the phase rule is still contractual —
Tier S calls pay nothing.

## Lock order

`helper_record_mutex → transfer_mutex → resource_mutex → command_mutex`, one
direction only. Creation and destruction share a single `resource_mutex`
(cold paths); command-record allocation, submit claims, and reclamation share
`command_mutex`; and the transfer arenas share `transfer_mutex`.
`helper_record_mutex` spans a blocking helper's recording window
(`begin_commands` through `end_commands`) and is outermost because that
window takes `transfer_mutex` internally for its allocation.

Each queue has its own mutex. Submit releases `command_mutex` before taking a
queue mutex and releases the queue mutex before invalidating command tokens,
so command-record and queue locks are never nested.

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
stays the caller's responsibility (gpu.c3l#36).

## Visibility rules

- Resource slot reads (`get` paths) are lock-free: tables never reallocate, and
  a handle reaches another thread only through your synchronization — that
  hand-off is the happens-before edge.
- Command records also live in a fixed table. The public `CommandList` is an
  owner-bearing handle into that table; passing it from a recording thread to
  the submitting thread is the required hand-off. Copies remain aliases and
  must not be used concurrently.
- Destruction must happen-after the last use of the handle on any thread.

## Worker-thread setup (C3)

C3 only creates the implicit temp allocator on the main thread. Library
paths allocate temporaries internally, so worker threads calling into the
library must wrap their body in `@pool_init(...)` (see `std::core::mem`) or
they abort on first temp allocation.

## Miscellany

- The validation messenger callback fires on arbitrary driver threads; the
  library only writes to stderr from it.
- A context is confined, not locked: two lists from one context may be
  recorded interleaved by the owning thread, never by two threads.
- Command lists from different contexts may be mixed in one `SubmitDesc`.

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
covers every queue's work — arenas, command pools, descriptor retires, and
readback tickets stay safe under any queue topology, off-frame submissions
included. `submit` sets the per-queue used flags unconditionally, not just
while a frame is active, so an off-frame submission on a distinct compute or
transfer queue is still waited by the next `end_frame`'s chain (off-frame
graphics-queue submits need no flag: the frame signal already runs on that
queue, and Vulkan's per-queue submission order covers them for free). See
"Off-frame submissions" below for the destruction-side half of this
contract.

## Helper timeline

Blocking helpers (`upload_buffer_data`, `upload_texture_data`,
`readback_buffer_data`, `readback_texture_data`) never touch `frame_timeline`
or the frame counter. Each reserves a value on a separate `helper_timeline`
under `transfer_mutex`, before its (single) transfer allocation, and tags
that allocation's arena range or dedicated buffer with it. At completion
(after the helper's own queue-idle wait) it signals `helper_timeline` to
that value — turnstiled: it first waits for `helper_timeline` to reach
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
Resources a command list submitted off-frame refers to must not be freed
while that work may still be in flight: destroying such a resource enqueues
it in the deferred-release queue (docs/memory.md §17) rather than freeing it
synchronously, and it stays queued until the next `end_frame`'s cross-queue
chain covers it, or until device teardown if no further frame ever runs.
