# Cookbook

Task-oriented recipes. Each shows the non-obvious lines only and links the
sample that runs the full pattern in CI — read the sample when you want the
whole program. Samples live in
[gpu.c3l-samples](https://github.com/fesoliveira014/gpu.c3l-samples).

Ordered roughly by adoption path.

Unless a recipe says otherwise, ordinary `cmd_*` calls below use CHECKED
syntax and therefore unwrap with `!`. In a `GPU_FAST_COMMANDS` target, omit
that `!` from the 24 ordinary commands. Keep it for the three generated
commands and for every lifecycle, resource, submission, wait, and WSI
operation.

## 1. Author a shader ABI schema

Goal: one source of truth for a struct shared by C3 and GLSL.

```text
// my_pass.abi
abi my_pass;

root MyRoot {
    vec4       tint;
    GpuAddress vertex_gpu;
    uint       count;
    uint       _pad0;
}
```

`gen_shader_abi` emits the C3 twin (with `$assert` size/offset checks) and a
GLSL include; `--check` is the CI drift gate. std430 padding is validated,
not guessed — the generator tells you exactly which `_padN` to insert.
Field names must not be GLSL keywords.

Running example: every sample's `abi/` directory; flow documented in
`docs/shader_abi.md`.

## 2. Upload a texture

Goal: pixels from CPU to a sampled texture.

```c3
gpu::TextureState transfer_destination = {
    .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
    .stages = { .transfer = true },
    .access = { .write = true },
};
gpu::TextureBarrier to_dst = gpu::texture_transition(
    texture: tex,
    before:  { .layout = gpu::TextureLayout.UNDEFINED },
    after:   transfer_destination,
)!;
gpu::cmd_texture_barrier(&cmd, &to_dst)!;

gpu::BufferTextureCopyDesc upload = { .src = upload_span, .texture = tex };
gpu::cmd_copy_buffer_to_texture(&cmd, &upload)!;

gpu::TextureBarrier to_sample = gpu::texture_transition(
    texture: tex,
    before:  transfer_destination,
    after:   gpu::sampled_at({ .fragment_shader = true }),
)!;
gpu::cmd_texture_barrier(&cmd, &to_sample)!;
```

Running example: `textured_cube` (single texture), `pbr_materials`
(several), `bindless_stress` (8192, batched).

## 3. Sample through the bindless heap

Goal: shader picks its texture by an integer you stored anywhere.

```c3
gpu::TextureView view = gpu::create_texture_view(&device, tex, null)!;
root.albedo_tex = view.index;       // raw uint field in your root/table
```

```glsl
vec4 c = sample_texture_2d(mat.albedo_tex, root.heap_sampler, uv);
```

Retain `view` until every command using `view.index` has completed and no
GPU-visible data contains it, then call `destroy_texture_view`.

Running example: `bindless_texture_compute` (compute),
`pbr_materials` (per-material indices in a table).

## 4. Blocking readback

Goal: wait for a result and read it on the CPU.

Allocate `CPU_READ` memory, record the resource transition and copy into its
span, then record a global barrier with `before.transfer` and `after.host`.
Call `submit`, wait for the returned completion point, invalidate the mapped
span, and read it. Free or reuse the allocation only after completion.

Running examples: `offscreen_triangle`, `multithreaded_recording`.

## 5. Non-blocking readback

Goal: overlap GPU work with CPU consumption.

Allocate a `CPU_READ` destination, record the copy and a global barrier with
`before.transfer` and `after.host`, then keep the completion point returned
by `submit`. Once `poll_completion` succeeds, call
`invalidate_mapped_span` and read the mapped span. Reuse or free the allocation
only after completion.

## 6. GPU-driven drawing (indirect + count)

Goal: compute culls, GPU decides the draw count.

```c3
// Compute writes DrawIndirectCommand[] + a count word.
gpu::GraphicsState state =
    gpu::full_render_graphics_state(pass.width, pass.height)!;
gpu::cmd_begin_render_pass_with_state(&cmd, &pass, &state)!;
gpu::cmd_bind_pipeline(&cmd, pipeline)!;
gpu::cmd_draw_indirect(
    commands:      &cmd,
    vertex_root:   vroot,
    fragment_root: froot,
    args:          args_span,
    draw_count:    max_draws,
)!;
// with caps.draw_indirect_count: the _count variant reads the GPU count
```

Shader side indexes per-draw data with `gl_DrawID`.

For several compatible passes, record the complete packet once with
`cmd_set_graphics_state`, then use two-argument `cmd_begin_render_pass` for
each pass. Minimal begin does not replay the packet; record another complete or
partial setter only when state changes. Under `ContractValidation.FULL`, the
first regular or generated draw still requires one successful complete packet
in that command-buffer recording.
Running example: `gpu_driven_draw_sdl`, `frustum_culling`.

## 7. Split submits linked by a completion point

Goal: simulation and rendering as separate submits with explicit ordering.

```c3
gpu::Queue compute = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!;
gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!;

gpu::SubmitDesc sim_submit = { .command_lists = sim_lists[..] };
gpu::CompletionPoint sim_done = gpu::submit(compute, &sim_submit)!;

gpu::CompletionWait[1] draw_waits = {{
    .point  = sim_done,
    .before = { .vertex_shader },
}};
gpu::SubmitDesc draw_submit = {
    .command_lists    = draw_lists[..],
    .completion_waits = draw_waits[..],
    .readiness        = acquired.readiness,
    .readiness_before = { .color_output },
};
gpu::CompletionPoint draw_done = gpu::submit(graphics, &draw_submit)!;
gpu::present(&device, &acquired, draw_done)!;
```

Real overlap happens when `caps.async_compute` is true (distinct compute
queue); resources used by both queues declare `{ .graphics, .compute }` access.
Single-queue devices run the same code serialized.
Running example: `particle_sim`.

When the compute submission writes indirect/count or implicitly preprocessed
generated-command records, wait at their first real consumer instead of using
`.all` or a later shader stage:

```c3
gpu::CompletionWait[1] indirect_waits = {{
    .point     = sim_done,
    .consumers = { .draw_arguments },
}};
```

## 8. Own command recording explicitly

Goal: make command-pool ownership, concurrency, and reuse explicit.

The snippets below show CHECKED calls with `!`. In a target built with
`GPU_FAST_COMMANDS` and `DIRECT_COMMAND_TOKENS`, remove `!` from the 24
ordinary `cmd_*` calls and create the runtime with `TRUSTED` plus tracking off.
Keep `!` on the three generated commands and on lifecycle, resource, submit,
wait, and WSI calls. FAST callers keep referenced resources live until the
covering completion is observed.

### One queue and one worker

```c3
gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!;
gpu::CommandAllocator allocator =
    gpu::create_command_allocator(&device, queue)!;
gpu::CommandList list = gpu::begin_commands(&allocator)!;
record_compute(&list)!;
gpu::ExecutableCommandList executable = gpu::end_commands(&list)!;
gpu::ExecutableCommandList[1] lists = { executable };
gpu::CompletionPoint done = gpu::submit(
    queue,
    &{ .command_lists = lists[..] },
)!;
gpu::wait_completion(done)!;
gpu::destroy_command_allocator(&allocator)!;
```

The allocator is bound to the exact queue and owns fixed recording capacity.
Running example: `root_pointer_compute`.

### Separate graphics and compute allocators

```c3
gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!;
gpu::Queue compute = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!;
gpu::CommandAllocator graphics_allocator =
    gpu::create_command_allocator(&device, graphics)!;
gpu::CommandAllocator compute_allocator =
    gpu::create_command_allocator(&device, compute)!;
```

Use the compute allocator only for `compute` and the graphics allocator only
for `graphics`. Even when both roles alias one native queue, the two allocator
identities keep their pools, capacity, and recording ownership independent.
Destroy each after its last completion. Running example: `particle_sim`.

### Hand executable tokens to a submit thread

Create one allocator per concurrently recording worker. A worker calls
`begin_commands(&worker_allocator)`, records, and calls `end_commands`; it then
publishes the resulting `ExecutableCommandList` through an application-
synchronized queue. The submit thread may combine tokens from different
allocators in one `SubmitDesc` when every allocator is bound to the exact submit
queue. Do not touch the token or its aliases on the worker after publication.

An allocator may move to another recording worker only after its last recording
has ended or been discarded and the application establishes a happens-before
edge. Core command work on a fresh C3 worker needs no implicit temporary-
allocator initialization.
Benchmark with Vulkan validation off because the layer may serialize native
recording; run correctness gates with it on. Running example:
`multithreaded_recording`.

### Gate reuse and destruction on completion

`command_buffer_capacity` is the maximum number of recording, executable, and
in-flight units owned by one allocator. When all are live, `begin_commands`
returns `DEVICE_BUSY`; wait or poll the covering completion, or discard an
unsubmitted token, before retrying. Completion retirement returns each exact
unit to its allocator. `destroy_command_allocator` follows the same rule but
never waits: it returns `RESOURCE_IN_USE` and preserves the allocator until all
units are back. Running example: `offscreen_triangle`.

### Reserve generated scratch on the allocator

```c3
gpu::CommandAllocatorDesc allocator_desc = {
    .command_buffer_capacity = 2,
    .max_generated_preprocess_buffers_per_list = 4,
    .generated_preprocess_bytes = 8 * 1024 * 1024,
    .debug_name = "generated_graphics",
};
gpu::CommandAllocator allocator = gpu::create_command_allocator(
    &device,
    graphics,
    &allocator_desc,
)!;
gpu::GeneratedScratchDesc scratch_desc = {
    .pipeline = pipeline,
    .kind = gpu::GeneratedWorkKind.DRAW,
    .max_commands_per_list = max_draws,
    .preprocess_buffer_count = 4,
};
gpu::reserve_generated_scratch(&allocator, &scratch_desc)!;
```

Reserve and release only while the allocator is quiescent. The backend queries
the exact pipeline/layout/maximum-count requirements once; every smaller
recording uses that reservation without another native query. The descriptor's
byte value is a hard budget, not a substitute for compatibility. Reservation-
table or byte-budget exhaustion returns
`COMMAND_ALLOCATOR_CAPACITY_EXCEEDED`; no matching free reservation during
recording returns `GENERATED_SCRATCH_EXHAUSTED`.
Release the pipeline/kind reservation, then destroy the allocator after its last
completion. Running example: `gpu_driven_draw_sdl`.

## 9. Requesting portable anisotropy

Query the selected device and clamp explicitly. `intern_sampler` rejects values
above the reported maximum instead of silently changing the request.

```c3
gpu::DeviceCaps caps = gpu::get_device_caps(&device)!!;
float preferred = 16.0f;
gpu::SamplerDesc sampler_desc = {
    ...,
    .anisotropy_enable = caps.max_sampler_anisotropy > 0.0f,
    .max_anisotropy = preferred > caps.max_sampler_anisotropy
        ? caps.max_sampler_anisotropy
        : preferred,
};
gpu::SamplerIndex sampler_index = gpu::intern_sampler(&device, &sampler_desc)!!;
```

## 10. Shadow mapping with compare samplers

Goal: depth-only pass, then hardware PCF.

```c3
gpu::SamplerDesc shadow_sampler_desc = {
    ...,
    .compare_enable = true,
    .compare = gpu::CompareOp.LESS_EQUAL,
};
gpu::SamplerIndex shadow_sampler_index =
    gpu::intern_sampler(&device, &shadow_sampler_desc)!;
// The shadow pass has no color targets and uses D32.
gpu::GraphicsState state =
    gpu::full_render_graphics_state(pass.width, pass.height)!;
state.raster = {
    .depth_bias_enable   = true,
    .depth_bias_constant = 1.25f,
    .depth_bias_slope    = 1.75f,
};
state.depth = {
    .test_enable  = true,
    .write_enable = true,
    .compare      = gpu::CompareOp.LESS,
};
gpu::cmd_begin_render_pass_with_state(&cmd, &pass, &state)!;
gpu::cmd_bind_pipeline(&cmd, shadow_pipeline)!;
```

```glsl
float lit = sample_shadow_2d(root.shadow_map, root.shadow_sampler,
    vec3(uv, ndc_depth));
```

Running example: `shadow_mapping` (3×3 PCF).

## 11. Multiple render targets

Goal: G-buffer in one pass.

```c3
gpu::AttachmentViewHandle albedo_view = gpu::create_attachment_view(
    &device,
    &{ .texture = albedo },
)!;
gpu::AttachmentViewHandle normal_view = gpu::create_attachment_view(
    &device,
    &{ .texture = normal },
)!;
gpu::AttachmentViewHandle position_view = gpu::create_attachment_view(
    &device,
    &{ .texture = position },
)!;
gpu::ColorTargetDesc[3] colors = {
    { .view = albedo_view },
    { .view = normal_view },
    { .view = position_view },
};
gpu::RenderPassDesc pass = { .colors = colors[..], .depth = &depth_target, ... };
// pipeline: .colors lists each target's format, blend, and write mask;
// frag writes locations 0..2.
```

Create the depth attachment view the same way. After the covering completion
point finishes, destroy every attachment view before destroying its texture.

Running example: `deferred_shading` (plus the linear→display lessons:
Reinhard + gamma encode when the swapchain is UNORM).

## 12. Persist the pipeline cache

Goal: skip shader compiles on the next run.

```c3
usz size = gpu::get_pipeline_cache_size(&device)!;
char[] blob = mem::new_array(char, (sz)size);
usz written = gpu::get_pipeline_cache_data(&device, blob)!;   // save to disk
// next run:
gpu::RuntimeDesc runtime_desc = {
    .backend             = gpu::BackendKind.VULKAN,
    .pipeline_cache_data = loaded_blob,
};
gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
```

Blob usefulness is driver-dependent (lavapipe: header only); identical
descriptors on one device always dedup in-library regardless.
Running example: `pipeline_cache_timing`.

## 13. Query swapchain runtime state

Goal: build against the selected format and transition acquired images from
their reported prior state.

```c3
gpu::PresentModeSupport support = gpu::get_present_mode_support(&device, swapchain)!;
if (support.mailbox) { /* recreate swapchain with PresentMode.MAILBOX */ }

gpu::SwapchainInfo info = gpu::get_swapchain_info(&device, swapchain)!;
if (info.dormant) { /* wait for a non-zero resize */ }

gpu::ColorTargetState[1] pipeline_colors = {{
    .format     = info.format,
    .write_mask = gpu::COLOR_WRITE_ALL,
}};

gpu::AcquiredImage acquired = gpu::acquire_next_image(
    &device,
    swapchain,
    2_000_000,
)!;
gpu::TextureState color_attachment = {
    .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
    .stages = { .color_output = true },
    .access = { .read = true, .write = true },
};
gpu::TextureBarrier to_color = gpu::texture_transition(
    texture: acquired.texture,
    before:  acquired.prior_state,
    after:   color_attachment,
)!;
gpu::cmd_texture_barrier(&cmd, &to_color)!;
```

Record the render pass, then return the image to the fixed external state:

```c3
gpu::TextureBarrier to_present = gpu::texture_transition(
    texture: acquired.texture,
    before:  color_attachment,
    after:   { .layout = gpu::TextureLayout.PRESENT },
)!;
gpu::cmd_texture_barrier(&cmd, &to_present)!;
```

Re-query `SwapchainInfo` after resize, rebuild pipelines if `format` changed,
and size any per-image data from `image_count`. `prior_state` is directly usable
and distinguishes a newly wrapped image from one returned by a previous
presentation cycle. The caller owns this history; the backend does not infer
the next barrier's `before`. FIFO is always available; other modes remain a
support query away. The graphics submission consumes `acquired.readiness` with
the first concrete consumer stage, such as `{ .color_output = true }`, and
presentation accepts its returned completion point. The public `PRESENT` state
has empty stages/access; Vulkan lowering keeps the fixed
color-attachment-output/no-access presentation scope.
Running example: `present_mode_explorer`.

## 14. Choose a memory class

| Class | For | Pattern |
|---|---|---|
| `MemoryClass.CPU_WRITE` | CPU-written generic data, including roots | map, write, flush, submit, wait or poll, free or reuse |
| `MemoryClass.GPU_PRIVATE` | GPU-only working sets | copy from caller-owned `CPU_WRITE` storage |
| `MemoryClass.CPU_READ` | GPU-to-CPU results | wait, `invalidate_mapped_span`, read |

See `docs/memory.md` for lifetime and visibility rules.

## 15. Allocate generic GPU data

Goal: own an addressable range without exposing backend memory objects.

```c3
gpu::AllocationDesc desc = {
    .size         = 4096,
    .alignment    = 256,
    .memory_class = gpu::MemoryClass.CPU_WRITE,
    .access       = { .compute },
    .debug_name   = "constants",
};
gpu::GpuAllocation allocation = gpu::allocate_memory(&device, &desc)!;
defer (void)gpu::free_allocation(&device, &allocation);

gpu::AllocationInfo info = gpu::get_allocation_info(&device, allocation)!;
gpu::GpuSpan span = gpu::get_allocation_span(&device, allocation)!;
gpu::GpuSpan header = span.checked_subspan(0, 256)!;
char[] mapping = gpu::get_span_mapping(&device, header)!;
mapping[0] = 1;
gpu::flush_mapped_span(&device, header)!;
gpu::GpuAddress address = gpu::get_span_address(&device, header)!;
```

`GpuAllocation` owns storage; its spans borrow ranges. `free_allocation`
invalidates the token only after success and requires all GPU use to be
quiescent. `CPU_WRITE` and `CPU_READ` are mapped; `GPU_PRIVATE` is not.
Use `AllocationInfo` for actual capabilities; do not assume coherence. For
long-lived CPU-written data, borrow the allocation's span, mapping, and address
as needed, write, flush, record and submit, wait for or poll the covering
completion point, then free the owning allocation. For readback, record a
global barrier with `before.transfer` and `after.host`, wait or poll
completion, invalidate the `CPU_READ` span, then read its mapping. Running
example: `memory_report`.

## 16. Retire transient data by completion

Goal: keep caller-owned root data valid until the GPU has finished using it.

```c3
gpu::AllocationDesc root_desc = {
    .size         = RootArgs::size,
    .alignment    = RootArgs::alignment,
    .memory_class = gpu::MemoryClass.CPU_WRITE,
    .access       = { .graphics },
    .debug_name   = "render_root",
};
gpu::GpuAllocation root_allocation =
    gpu::allocate_memory(&device, &root_desc)!;
defer (void)gpu::free_allocation(&device, &root_allocation);

gpu::GpuSpan root_span =
    gpu::get_allocation_span(&device, root_allocation)!;
RootArgs* root =
    (RootArgs*)gpu::get_span_mapping(&device, root_span)!.ptr;
write_root(root);
gpu::flush_mapped_span(&device, root_span)!;

record_rendering(
    commands: &commands,
    root:     gpu::get_span_address(&device, root_span)!,
    state:    &state,
)!;
gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!;
gpu::ExecutableCommandList[1] lists = { executable };
gpu::SubmitDesc submit = { .command_lists = lists[..] };
gpu::CompletionPoint completion = gpu::submit(graphics, &submit)!;
gpu::wait_completion(completion)!;
```

The completion wait makes the deferred free safe. A non-blocking loop retains
both `root_allocation` and `completion`, polls the point, and reuses or frees
the allocation only after the poll succeeds. Applications can build rings or
pools from the same rule; the root module does not choose the number of
concurrent work sets.
