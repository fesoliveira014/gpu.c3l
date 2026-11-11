# Unified image layouts

[Documentation](../index.md) › Contributing › Unified image layouts

**Question.** Should the library keep every texture in one image layout and
own the transitions, instead of requiring an application-authored
`TextureBarrier` for every layout change?

**Decision.** Yes, as an opt-in device mode (`DeviceDesc.unified_layouts`).
Explicit mode stays the default and is unchanged.

## What the mode does

- Every `TextureLayout` except `UNDEFINED` lowers to `GENERAL` in barriers,
  descriptors, copies, and attachments. Stage and access masks still come
  from the `TextureState`, so barriers keep describing hazards.
- Validation keeps the stage, access, and queue rules and drops the
  per-layout usage and access rules.
- New textures are queued at creation and transitioned
  `UNDEFINED -> GENERAL` by a library-recorded command list, submitted as a
  batch of its own just before the next submit on a queue whose roles
  overlap the texture's `access`. Submits on other queues leave it queued,
  so no queue family outside the texture's sharing set touches it. Later
  batches on other queues whose roles overlap that `access` wait for the
  initialization batch's point at all commands until it retires. Destroying
  a texture before its initialization is submitted removes it from the
  queue.
- A readiness-carrying submit gets a library-recorded acquire list
  (`UNDEFINED -> GENERAL`) before the application lists and a present list
  (`GENERAL -> PRESENT_SRC`) after them. `AcquiredImage.prior_state`
  reports `GENERAL`. `readiness_before` is ignored; the batch waits the
  acquisition at all commands.
- `VK_KHR_unified_image_layouts` is enabled when present and reported as
  `DeviceCaps.unified_layouts_optimal`.

## Mechanism

Library-recorded lists come from one command allocator per queue that lives
on the queue's completion state, outside the public allocator table, so leak
reports and capacities are unchanged. The lists are ordinary records: they
are published with the batch and retired by completion, which gives buffer
reuse without a second ring.

The pending-texture queue is one array of seven buckets, one per non-empty
subset of the graphics, compute and transfer roles, guarded by a leaf mutex.
A submit takes the buckets that overlap its queue's roles, under the device
resource lock so a concurrent destroy cannot interleave, before it takes its
queue's submission lock. It records them into one initialization list only
after the submit's own inputs are validated and claimed under that lock, so a
rejected submit puts the textures back and leaves no GPU work. That list is
submitted alone, signalling its own point on the queue's timeline: it waits
on nothing but an earlier sparse bind on the same queue, so a queue that
waits for it waits only for the initializing queue's earlier work, not for
the batch submitted after it. Once that native submit succeeds, the point is
stored for every other queue identity whose roles overlap the initialized
textures, and those queues add a timeline wait at all commands while the
point is unretired and not already covered by an application wait at all
stages on the same queue. The wait is repeated on each batch until
retirement because a semaphore wait orders only its own batch. A failed
initialization batch puts its textures back and fails the submit; a failure
of the caller's batch after it leaves the initialization done.

A device-wide init-submit mutex is held from the take through the
initialization batch's publication or the restore, so the two look atomic to
other submits. It sits under the device operation pin and above the resource
lock, and no path takes it while holding a resource, queue, or retirement
lock. A per-role count of textures queued or taken but not yet settled lets
a submit skip the mutex: a take is settled after its point is stored for the
other queues (both sequentially consistent), so a zero count for every role
of the queue means every texture created before the call is either
initialized with its point stored or destroyed. The take records how many
textures it took from each bucket and is settled from those counts, never
from the texture slots, and before the initialization record and its point
are published. Once they are published, any thread that observes completion
may retire the record, which unpins the textures and frees its handle array,
and a texture may then be destroyed; neither can lose or corrupt a count.

Roles, not families, decide compatibility. A role in a texture's `access`
puts that role's family in its sharing set, so a role match is always a
valid family. Matching by family would let, for example, a graphics queue
take a compute-only texture when compute shares the graphics family on
another queue index, and then make the compute queue wait for graphics.

Sparse textures are queued at creation, not after their first bind, and the
initialization is not ordered against a bind on another queue; it is ordered
after an earlier bind on the queue that initializes it.

## Cost status

Unmeasured. Without the extension, `GENERAL` can disable framebuffer or
depth compression on some vendors; with it, the driver guarantees the layout
is as fast as the optimal ones. lavapipe exposes neither the extension nor
a cost. A hardware comparison of a cookbook scene in both modes belongs
here before the default changes.

## Outstanding

- Pre-recorded acquire and present command buffers per swapchain image
  (issue #598 asks for no per-frame recording). The current mode records two
  one-barrier lists per readiness submit from the queue's library allocator.
- The cookbook recipes are still written for explicit mode with a unified
  note; a unified-mode pass over every recipe is pending.

## Alternatives kept in reserve

- Pre-recorded per-image acquire and present command buffers from a private
  pool: records nothing per frame. Deferred (see Outstanding); the per-submit
  lists are one barrier each and reuse the allocator lifetime.
- `vkTransitionImageLayoutEXT` (`VK_EXT_host_image_copy`) at texture
  creation, skipping the pending queue and the cross-queue waits. It needs
  `VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT` on every image, which can disable
  compression on some drivers, and the extension is not universal.
- Removing `SubmitDesc.readiness_before`: it is ignored in the mode and kept
  for explicit mode; the removal question is open.

## Revisit when

A hardware run shows the unified mode slower than explicit mode by a
repeatable margin on a device without the extension, or the extension is
common enough to make the explicit mode the exception.
