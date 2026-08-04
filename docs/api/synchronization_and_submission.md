# Synchronization and submission

## Stages, access, and barriers

`StageMask` describes semantic execution scopes. `Barrier` orders global
execution and memory visibility between a `before` and `after` stage/access
domain. `cmd_barrier` records it on a compatible queue.

Texture transitions use `TextureState`, which combines `TextureLayout`,
`StageMask`, and `TextureAccess`. `sampled_at` and `storage_at` build common
states. `TextureBarrier` additionally identifies a texture and subresource
range.

`texture_transition` builds a barrier for an explicit texture range.
`texture_view_transition` builds one from a view's range.
`cmd_texture_barrier` records the result.

The caller owns texture history. `before` asserts the state established by
earlier ordered work; the library does not look up or repair it. A global
barrier cannot establish a texture layout.

`StageMask.acceleration_structure_build` names both BLAS/TLAS build and update
execution. On either side of a global barrier it carries acceleration-structure
read/write access. On a ray-query-enabled device, compute, vertex, and fragment
stages also include acceleration-structure read access for shader queries.
`StageMask.ray_tracing` names all direct ray-tracing shader stages. As a source
it carries shader-write and acceleration-structure-read access; as a
destination it carries shader read/write and acceleration-structure read
access. The generic shader-read scope covers caller-owned SBT contents without
requiring the out-of-scope ray-tracing-maintenance extension. Sampled and
storage texture states accept `.ray_tracing` on an opted-in device and a
compatible graphics or compute queue.

Order consecutive builds or updates explicitly:

```c3
gpu::Barrier build_to_build = {
    .before = { .acceleration_structure_build },
    .after  = { .acceleration_structure_build },
};
gpu::cmd_barrier(&commands, &build_to_build)!;
```

Before a compute query, use build-to-compute; before a graphics query, use the
calling shader stage or stages:

```c3
gpu::Barrier build_to_query = {
    .before = { .acceleration_structure_build },
    .after  = { .compute },
};
gpu::cmd_barrier(&commands, &build_to_query)!;
```

The same rule applies after an in-place update. Build/update commands insert no
implicit dependency.

Before `cmd_trace_rays`, use build-to-ray-tracing for a newly built or updated
TLAS and use upload-to-ray-tracing for GPU-produced root/SBT data:

```c3
gpu::Barrier build_to_trace = {
    .before = { .acceleration_structure_build },
    .after  = { .ray_tracing },
};
gpu::cmd_barrier(&commands, &build_to_trace)!;
```

Ray-tracing stages are valid only when the device opt-in is enabled and on
selected graphics or compute queues. They are rejected on transfer-only queues.

## Completion points

`CompletionPoint` identifies an ordered timeline value on one selected queue.
`COMPLETION_POINT_INVALID` is the zero sentinel. Points are reusable,
copyable ordering values, not resource owners.

`poll_completion` performs a nonblocking query.
`wait_completion` waits up to the supplied timeout; `TIMEOUT_INFINITE` requests
an unbounded host wait. Timeout returns `WAIT_TIMEOUT` without consuming or
changing the point.

Polling and waiting are thread-safe. They may retire completed command units
and retained references. Keep the device live throughout the call.

## Submission

`SubmitDesc` supplies executable command lists, `CompletionWait` dependencies,
and optional swapchain readiness. `submit` targets the exact `Queue` with which
every command allocator was created.

Each `CompletionWait` pairs a prior point with destination stages supported by
the destination queue. Cross-queue dependencies must name at least one valid
device stage; host and presentation stages are not wait destinations.
Use `.acceleration_structure_build` when the receiving compute/graphics queue
will build or update a structure, and use the receiving shader stage when it
will query or trace one. `.ray_tracing` is a valid wait destination only on an
enabled compatible graphics/compute queue. Transfer-only queues reject both
ray execution stages.

Submission is externally synchronized on the target native queue. It:

1. validates the complete batch and waits;
2. atomically claims every executable token;
3. reserves the next completion value;
4. makes the native queue call;
5. publishes retirement records and the returned point; and
6. consumes command tokens and any supplied one-shot swapchain readiness.

Validation, preparation, or native rejection before acceptance restores the
batch to executable state. Successful submission keeps each allocator unit,
native command buffer, fixed scratch, device pin, and retained reference alive
until ordered retirement.

The application retains the returned point whenever it guards memory reuse,
resource destruction, allocator reuse, presentation, or later queue work.

## Timestamp queries

`create_timestamp_pool` allocates `TimestampPoolDesc.capacity` query slots.
Pools are device-owned generational handles. Destruction is thread-safe but
must happen after every recorded use and host read.

The command sequence is explicit:

1. `cmd_reset_timestamps`;
2. `cmd_write_timestamp` at selected stages;
3. execute and order completion;
4. either `cmd_resolve_timestamps` to GPU storage or `read_timestamps` on the
   host; and
5. use `timestamp_delta_ns` with the queue role's valid-bit width and period.

The library does not track which slots were reset or written and does not
insert implicit resets. `read_timestamps` does not establish GPU completion.
Values are comparable only when written on the same native queue; distinct
queues are not calibrated.

`TimestampCaps` in `DeviceCaps` reports supported roles, valid widths, and the
device period. A semantic transfer role may have no timestamp capability.

## Concurrency and lifetime

Submission and sparse binding are externally synchronized per native queue.
Distinct native queues may submit independently. Completion poll/wait is
thread-safe and uses a separate retirement boundary.

Command recording itself is thread-confined and may occur concurrently through
distinct allocators. Barriers recorded in different lists do not synchronize
host threads; execution order comes only from queue submission order and
completion waits.

Under full validation, submitted lists retain explicitly recorded owners until
retirement. GPU-address targets, texture indices, sampler indices, timestamp
history, and sparse backing remain caller-owned under every policy.

## Fault behavior

- invalid stages, access, layouts, or ranges: `INVALID_ARGUMENT`;
- contradictory or unestablished command state: `INVALID_RESOURCE_STATE`;
- stale/foreign/consumed tokens: `INVALID_HANDLE` or
  `COMMAND_RECORDING_ERROR`;
- elapsed host wait: `WAIT_TIMEOUT`;
- device loss: `DEVICE_LOST`; and
- transient queue/device contention: `DEVICE_BUSY`.

No synchronization call performs hidden resource destruction or device-wide
idle waits.
