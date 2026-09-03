# Style

## 1. Language target

C3 0.8.3. C3 is pre-1.0; check syntax against the installed compiler, not
memory.

## 2. Modules and files

| Module | Contents |
|---|---|
| `module gpu;` | Public API. Non-callables in `gpu/gpu.c3i`; callables, with docstrings and bodies, in `gpu/gpu.c3`. |
| `module gpu::surface::<platform>;` | Native handle typedefs in `surface.c3i`; `create_surface` in `surface.c3`. |
| `module gpu::internal @private;` | Backend-independent implementation, one file per area in `gpu/internal/*.c3`. |
| `module gpu::internal::vk @private;` | Vulkan backend, mirroring those areas in `gpu/internal/vk/*.c3`. Translation helpers go in `helpers.c3`. |

Samples use their own module names. Only shipped helpers go in `module gpu;`.

## 3. Naming

| Kind | Case | Examples |
|---|---|---|
| Variables, fields, parameters | `snake_case` | `queue_index`, `debug_name` |
| Functions | `snake_case` | `allocate_memory`, `cmd_dispatch` |
| Structs, enums, typedefs, aliases | `PascalCase` | `RuntimeDesc`, `GpuSpan` |
| Constants and enum values | `SCREAMING_SNAKE_CASE` | `MAX_SHADER_HEAP_CAPACITY`, `DEVICE_LOST` |
| Modules | lowercase, `::`-separated | `gpu::internal::vk` |
| Files | `snake_case.c3` | `descriptor_heap.c3` |

## 4. Definition order

Within a file, or within each banner section of a file grouped by domain:

```text
1. Typedefs
2. Aliases
3. Constants
4. Enums / bitstructs
5. Structs
6. Struct methods
7. Free functions
```

## 5. Lifecycle functions

Project-owned resources use free functions: `create_x` / `destroy_x`,
`allocate_memory` / `free_allocation`. Not `Device.create` or
`GpuAllocation.free`.

Methods are for operations on an existing receiver that are not lifecycle
operations. Bindings may use method syntax where it mirrors the C API.

## 6. Errors

Fallible operations return `T?` or `void?` and fail with a named fault:

```c3
fn GpuAllocation? allocate_memory(Device* device, AllocationDesc* desc);
return INVALID_HANDLE~;
```

Do not use bool out-parameters, null returns, `-1` sentinels, or global
error state. Use the most specific fault that fits; add a catch-all only
when no category applies.

## 7. Handles

Use the typed handle: `TextureHandle texture`, not `uint texture`.

## 8. Call formatting

A call with four or more arguments, or wider than 120 columns, uses named
arguments, one per line, trailing comma:

```c3
gpu::cmd_draw(
    commands:       &commands,
    vertex_root:    vertex_root,
    fragment_root:  fragment_root,
    vertex_count:   3,
    instance_count: 1,
)!;
```

Calls with three or fewer arguments may stay positional.

## 9. Braces

K&R:

```c3
fn void? free_allocation(Device* device, GpuAllocation* allocation) {
    if (allocation == null || !allocation.is_valid()) {
        return INVALID_HANDLE~;
    }
}
```

## 10. Docstrings and comments

Every public callable has a C3 docstring in this order:

```text
summary, including which recoverable faults it returns

@param entries in declaration order
@return entry
@require entries, only for stable local contracts
```

State ownership, borrowing, token consumption, blocking, and thread
confinement when they matter. Keep the text backend-neutral; a surface module
may name its native handle types.

`@require` is executable. Never put a recoverable condition in it: invalid
handles, missing capabilities, resource in use, exhaustion, timeouts, device
loss. Those go in prose and `@return`.

Example:

```c3
<* Allocate caller-managed GPU storage.
   Free the returned allocation after GPU use is quiescent.

   @param device : "Live device that will own the allocation."
   @param desc : "Borrowed allocation description."
   @return "An owning allocation token, or a recoverable validation, capacity, allocation, or device fault." *>
fn GpuAllocation? allocate_memory(Device* device, AllocationDesc* desc);
```

Inline comments explain why, not what. A field that needs a comment to be
understood should be renamed.

## 11. Current state only

Code and shipped documentation describe current behavior. No schedules,
roadmap labels, ticket ids, milestone names, or history in identifiers,
file names, test names, or `debug_name` values.

## 12. Public signature hygiene

Public `gpu` signatures never contain `vk::`, `vma::`, or `sdl::` types,
platform window structs, or raw OS handles outside the surface typedefs.
Backend files may import `vk` and `vma`. Samples may import `sdl`.

## 13. Shaders

Explicit `set`/`binding` and `location`. `std430` for root and table data.
Shared structs come from the ABI generator or carry size and offset
assertions on both sides. No `vec3` in shared structs.

## 14. Debug names

Every resource descriptor accepts `debug_name`. Use descriptive
`snake_case`: `input_buffer`, `transient_upload_0`, `albedo_texture`,
`pipeline_root_pointer_compute`.

## 15. Tests

`@test` functions with `snake_case` names that state the behavior:
`test_invalid_buffer_handle_rejected`,
`test_root_pointer_compute_writes_output`. Assert the specific fault. CPU
tests are exhaustive; Vulkan tests run validation-clean.

## 16. Formatting tools

No whole-tree auto-formatting. Hand-format to this guide. Avoid
whitespace-only rewrites.

## 17. Checklist

A change is style-compliant when:

- names follow the table in section 3;
- public lifecycle uses free functions;
- faults are specific;
- public signatures leak no bindings;
- calls with four or more arguments use the named multiline form;
- comments explain why, ownership, or invariants;
- no development labels appear in identifiers; and
- CPU tests and the relevant backend tests pass.
