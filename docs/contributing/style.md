# Style and project conventions

## 1. Language target

Target C3 0.8.0. C3 is pre-1.0, so code examples and implementation work must be verified against the target compiler rather than memory of earlier syntax.

## 2. Module names

Public module:

```c3
module gpu;
```

Backend-independent implementation module:

```c3
module gpu::internal @private;
```

Vulkan backend module:

```c3
module gpu::internal::vk @private;
```

Public non-callables belong in `gpu/gpu.c3i`. Public callables, including their
contracts, attributes, defaults, and bodies, belong in `gpu/gpu.c3`. A platform
surface keeps only its public native typedefs in `surface.c3i` and its public
callable in the adjacent `surface.c3`. Private implementation declarations go
in `gpu/internal/*.c3`; Vulkan declarations go in `gpu/internal/vk/*.c3`.

Sample modules may use sample-specific namespaces. Do not put samples in `module gpu;` unless they are shipped helpers.

## 3. Naming

| Kind | Case | Examples |
|---|---|---|
| Variables, fields, parameters | `snake_case` | `queue_index`, `debug_name`, `memory_class` |
| Functions | `snake_case` | `allocate_memory`, `cmd_dispatch`, `wait_completion` |
| Structs, enums, typedefs, aliases | `PascalCase` | `RuntimeDesc`, `GpuSpan`, `TextureUsage` |
| Constants and enum values | `SCREAMING_SNAKE_CASE` | `MAX_SHADER_HEAP_CAPACITY`, `TRANSFER_DST`, `DEVICE_LOST` |
| Modules | lowercase, dotted | `gpu`, `gpu::internal`, `gpu::internal::vk` |
| Files | `snake_case.c3` | `descriptor_heap.c3`, `pipeline_graphics.c3` |

## 4. Definition order

Within each source file:

```text
1. Typedefs
2. Aliases
3. Constants
4. Enums / constdefs / bitstructs
5. Structs
6. Struct methods
7. Free functions
```

Keep type definitions before values and operations that use them.

## 5. Construction and destruction

Project-owned lifecycle uses free functions:

```text
create_device
destroy_device
allocate_memory
free_allocation
create_texture
destroy_texture
```

Avoid:

```text
Device.create
GpuAllocation.free
Texture.create
```

Methods are appropriate when an operation naturally mutates or reads an existing receiver and is not a lifecycle constructor. External bindings may use method syntax when it maps C API structure; backend code may call `vma` wrapper methods because those are part of the binding.

## 6. Error handling

Use C3 optionals/faults for fallible operations.

Good:

```text
allocate_memory(...) -> GpuAllocation?
cmd_dispatch(...) -> void?
return INVALID_HANDLE~
```

Avoid:

```text
bool success out-params
null sentinel returns
-1 resource IDs
global errno-style state
```

Faults should be specific:

```text
INVALID_HANDLE
UNSUPPORTED_FEATURE
RESOURCE_IN_USE
DESCRIPTOR_HEAP_FULL
PIPELINE_CREATE_FAILED
```

Avoid broad catch-all faults such as `FAILED` unless no better category exists.

## 7. Handles

Use typed handles. Do not pass raw `uint`, `int`, or `ulong` where a domain handle exists.

Good:

```text
GpuAllocation allocation
TextureHandle texture
PipelineHandle pipeline
```

Bad:

```text
ulong allocation
uint texture
int pipeline
```

## 8. Call formatting

Calls with four or more arguments, or calls that would exceed 120 characters, should use named arguments, one per line, with a trailing comma.

Preferred:

```c3
gpu::cmd_draw(
    commands:       &commands,
    vertex_root:    vertex_root,
    fragment_root:  fragment_root,
    vertex_count:   3,
    instance_count: 1,
)!;
```

Short calls with three or fewer arguments may stay positional if readable.

## 9. Braces

Use K&R brace style:

```c3
fn void? free_allocation(Device* device, GpuAllocation* allocation) {
    if (allocation == null || !allocation.is_valid()) {
        return INVALID_HANDLE~;
    }
}
```

Do not use Allman braces.

## 10. Comments and docstrings

Every public callable in `gpu`, its platform surface modules, and any other
shipped public module uses a native C3 docstring. Order the content as:

```text
concise summary and recoverable-fault behavior

@param entries in declaration order

@return entry for a value or fault-returning operation

@require entries for stable local caller contracts only
```

Omit a method's receiver when its meaning is self-evident. State ownership,
borrowing and lifetime, token consumption, blocking, and thread confinement
when they affect correct use. Public descriptions stay backend-neutral; a
platform surface may name its public native handle types.

Avoid inline comments that restate the code. Comments should explain why, not
what.

C3 doc contracts are executable. Do not turn recoverable conditions such as
invalid handles, unsupported capabilities, resource use, capacity exhaustion,
timeouts, or device loss into `@require`. Describe those faults in prose and
the `@return` text. Use `@require` only when the condition is a stable local
programming contract and enforcing it preserves the intended API behavior.
Non-owning public utility views must explicitly state that they inherit the
lifetime of their public resources.

Example:

```c3
<* Allocate caller-managed GPU storage.
   Free the returned allocation after GPU use is quiescent.

   @param device : "Live device that will own the allocation."
   @param desc : "Borrowed allocation description."
   @return "An owning allocation token, or a recoverable validation, capacity, allocation, or device fault." *>
fn GpuAllocation? allocate_memory(Device* device, AllocationDesc* desc);
```

Keep docstrings short and precise. Ordinary compilation and `c3c docgen` are
the validation path.

API preconditions, side effects, and ownership rules belong in doc comments. If a field needs a comment to be understood, consider renaming it or restructuring the type.

## 11. Current-state documentation

Shipped code and documentation describe current behavior. Omit schedules,
roadmap labels, ticket identifiers, and implementation history.

Use behavior names such as `test_root_pointer_compute` and
`create_upload_pool`.

## 12. Public API dependency hygiene

Public `gpu` signatures must not expose:

```text
vk:: types
vma:: types
sdl:: types
platform window structs
raw native OS handles unless wrapped in neutral descriptors
```

Backend files may import `vk` and `vma`. Samples may import `sdl`.

## 13. File organization

The root public facade has one authoritative pair:

```text
gpu/gpu.c3i
gpu/gpu.c3
gpu/surface/<platform>/surface.c3i
gpu/surface/<platform>/surface.c3
```

Backend-independent implementation is grouped by area in `gpu/internal/*.c3`.
The Vulkan implementation mirrors those areas below it:

```text
gpu/internal/vk/device.c3
gpu/internal/vk/buffer.c3
gpu/internal/vk/texture.c3
gpu/internal/vk/pipeline_compute.c3
gpu/internal/vk/pipeline_graphics.c3
gpu/internal/vk/command.c3
gpu/internal/vk/sync.c3
```

Translation helpers belong in `gpu/internal/vk/helpers.c3` and should not be duplicated.

## 14. Shader style

Shader source should use explicit layouts:

```text
explicit set/binding
explicit location
std430 for root/table data
stable generated constants
```

Shared structs must be generated or manually mirrored with size checks. Avoid `vec3` in shared ABI structs.

## 15. Resource naming

All resources should accept a `debug_name` where practical.

Debug name conventions:

```text
input_buffer
output_buffer
transient_upload_0
persistent_materials
albedo_texture
swapchain_color_0
pipeline_root_pointer_compute
```

Avoid names that encode schedules or temporary implementation plans.

## 16. Testing style

Tests:

```text
use @test
use snake_case names
assert specific faults
keep pure CPU tests exhaustive
keep Vulkan tests validation-clean
```

Test names describe behavior:

```text
test_invalid_buffer_handle_rejected
test_allocation_extent_overflow_faults
test_root_pointer_compute_writes_output
```

## 17. Formatting tool policy

Do not run whole-tree auto-formatters unless the project explicitly adopts them. Hand-format to this guide. Large whitespace-only rewrites should be avoided.

## 18. Acceptance criteria for style compliance

A change is style-compliant when:

```text
names follow the table above
public lifecycle uses free functions
faults are specific
public signatures do not leak backend bindings
calls with 4+ args use named multiline style
comments document why, ownership, or invariants
no development labels appear in code or test identifiers
pure CPU tests and relevant backend tests pass
```
