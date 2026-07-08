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
| `upload_buffer_data` / `upload_texture_data` / `readback_buffer_data` / `readback_texture_data` | S | serialize on the queue mutex internally |
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
`INVALID_RESOURCE_STATE`; a recording left open across the boundary surfaces
through Vulkan validation (pool reset under an active command buffer).
Without validation the rule is contract only — the boundary pays one branch,
Tier S calls pay nothing.

## Lock order

`transfer_mutex → resource_mutex → queue mutexes`, one direction only.
Creation and destruction share a single `resource_mutex` (cold paths); the
transfer arenas share `transfer_mutex`; each queue has its own mutex.

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

- Slot reads (`get` paths) are lock-free: tables never reallocate, and a
  handle reaches another thread only through your synchronization — that
  hand-off is the happens-before edge. The same applies to passing a
  `CommandList` from a recording thread to the submitting thread.
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

`end_frame`'s fence is a chain of queue-ordered empty submits: distinct
compute/transfer queues used during the frame signal auxiliary timeline
values first, and the graphics-side signal waits on them before signaling
the frame value. A host-side wait on the frame value therefore covers every
queue's frame work — arenas, command pools, descriptor retires, and readback
tickets stay safe under any queue topology.
