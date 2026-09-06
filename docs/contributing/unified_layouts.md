# Unified image layouts

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
  `UNDEFINED -> GENERAL` by a library-recorded command list placed first in
  the next submit on any queue. Destroying a texture before that submit
  removes it from the queue.
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
reuse without a second ring. The pending-texture queue is guarded by a leaf
mutex; the drain swaps it out under the queue's submission mutex and puts it
back if the submit fails.

## Cost status

Unmeasured. Without the extension, `GENERAL` can disable framebuffer or
depth compression on some vendors; with it, the driver guarantees the layout
is as fast as the optimal ones. lavapipe exposes neither the extension nor
a cost. A hardware comparison of a cookbook scene in both modes belongs
here before the default changes.

## Alternatives kept in reserve

- Pre-recorded per-image acquire and present command buffers from a private
  pool: records nothing per frame. Rejected for now; the per-submit lists
  are one barrier each and reuse the allocator lifetime.
- `vkTransitionImageLayoutEXT` (`VK_EXT_host_image_copy`) at texture
  creation, skipping the pending queue. An optimization once the baseline
  is measured.
- Removing `SubmitDesc.readiness_before`: it is ignored in the mode and kept
  for explicit mode; the removal question is open.

## Revisit when

A hardware run shows the unified mode slower than explicit mode by a
repeatable margin on a device without the extension, or the extension is
common enough to make the explicit mode the exception.
