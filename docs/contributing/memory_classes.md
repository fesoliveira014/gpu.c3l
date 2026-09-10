# Memory classes

[Documentation](../index.md) › Contributing › Memory classes

Decision record for the placement each `MemoryClass` requests from VMA and
why `CPU_WRITE_GPU_LOCAL` exists beside `CPU_WRITE`.

## Mapping

| Class | VMA usage | Flags | Required |
|---|---|---|---|
| `CPU_WRITE` | `AUTO_PREFER_HOST` | mapped, sequential write | `HOST_VISIBLE` |
| `CPU_WRITE_GPU_LOCAL` | `AUTO_PREFER_DEVICE` | mapped, sequential write | `HOST_VISIBLE` |
| `CPU_READ` | `AUTO_PREFER_HOST` | mapped, random access | `HOST_VISIBLE` |
| `GPU_PRIVATE` | `AUTO_PREFER_DEVICE` | none | none |
| `TEXTURE` | image requirements | none | none |

`AUTO_PREFER_DEVICE` with `HOST_VISIBLE` required selects a memory type with
`DEVICE_LOCAL | HOST_VISIBLE` when the adapter exposes one and a host type
otherwise. There is no fault path; `AllocationInfo.device_local` reports the
selected type. `AdapterMemoryInfo.device_local_host_visible_bytes` is the
size of the heaps that back such a type: the whole device-local pool on
UMA and ReBAR parts, a 256 MiB window on discrete parts without ReBAR, zero
where no such type exists.

## Observed placement

| Adapter | `CPU_WRITE` | `CPU_WRITE_GPU_LOCAL` |
|---|---|---|
| lavapipe | device-local (all memory is) | device-local |
| Discrete without ReBAR | host | 256 MiB window, host on exhaustion |
| Discrete with ReBAR | host | device-local |
| Integrated | device-local | device-local |

## Decision

`CPU_WRITE` stays host-local. Its consumers are staging buffers read once by
a copy, where device placement buys nothing and would compete for the
window. Data the GPU reads on every command (root records, per-frame
constants, CPU-written indirect arguments) uses `CPU_WRITE_GPU_LOCAL`.

Alternatives considered:

- Change `CPU_WRITE` to prefer device memory. Rejected: staging uploads
  would exhaust a 256 MiB window before the data that benefits from it.
- Fault when no device-local host-visible type exists. Rejected: the class
  is a placement preference, and applications would need a fallback branch
  the library can take for them.
- Require `HOST_COHERENT`. Rejected: VMA's fallback can land on
  non-coherent memory on some integrated parts. `flush_mapped_span` and
  `invalidate_mapped_span` stay required and are no-ops on coherent memory;
  #592 relaxes that contract separately.

The cost of host-local root records was inferred from the mapping, not
measured. The root-record benchmark planned in
[root_pointer_data.md](root_pointer_data.md) runs with both classes.
