# Cookbook

Task-oriented recipes. Each shows the non-obvious lines only and links the
sample that runs the full pattern in CI — read the sample when you want the
whole program. Samples live in
[gpu.c3l-samples](https://github.com/fesoliveira014/gpu.c3l-samples).

Ordered roughly by adoption path.

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
gpu::TextureBarrier to_dst = gpu::texture_transition(
    texture: tex,
    before:  gpu::TextureUse.UNDEFINED,
    after:   gpu::TextureUse.TRANSFER_DESTINATION,
)!;
gpu::cmd_texture_barrier(&cmd, &to_dst)!;

gpu::BufferTextureCopyDesc upload = { .src = staging_span, .texture = tex };
gpu::cmd_copy_buffer_to_texture(&cmd, &upload)!;

gpu::TextureBarrier to_sample = gpu::texture_transition(
    texture: tex,
    before:  gpu::TextureUse.TRANSFER_DESTINATION,
    after:   gpu::TextureUse.SAMPLED_FRAGMENT,
)!;
gpu::cmd_texture_barrier(&cmd, &to_sample)!;
```

Running example: `textured_cube` (single texture), `pbr_materials`
(several), `bindless_stress` (8192, batched).

## 3. Sample through the bindless heap

Goal: shader picks its texture by an integer you stored anywhere.

```c3
gpu::TextureIndex idx = gpu::create_texture_descriptor(&device, tex, null)!;
root.albedo_tex = idx;              // plain uint field in your root/table
```

```glsl
vec4 c = sample_texture_2d(mat.albedo_tex, root.heap_sampler, uv);
```

Running example: `bindless_texture_compute` (compute),
`pbr_materials` (per-material indices in a table).

## 4. Blocking readback

Goal: get results back on the CPU, simplest form.

```c3
gpu::readback_texture_data(device: &device, src: target, mip: 0,
    out_data: pixels, from_stage: ..., from_hazard: ..., from_layout: ...)!;
// round-trips the layout back to from_layout — the texture is left as found
```

Running example: `offscreen_triangle`, `multithreaded_recording`.

## 5. Non-blocking readback

Goal: overlap GPU work with CPU consumption.

```c3
gpu::ReadbackTicket ticket = gpu::cmd_readback_buffer(&cmd, result_span)!;
```

After submission and frame retirement:

```c3
if (gpu::poll_readback(&device, &ticket)!) {
    gpu::resolve_readback(&device, &ticket, out)!;
}
```

Running example: `image_processing` (histogram readback),
`frustum_culling` (stats).

## 6. GPU-driven drawing (indirect + count)

Goal: compute culls, GPU decides the draw count.

```c3
// compute writes DrawIndirectCommand[] + a count word, then one call:
gpu::cmd_draw_indirect(&cmd, pipeline, vroot, froot, args_span, max_draws)!;
// with caps.draw_indirect_count: the _count variant reads the GPU count
```

Shader side indexes per-draw data with `gl_DrawID`.
Running example: `gpu_driven_draw_sdl`, `frustum_culling`.

## 7. Split submits linked by a completion point

Goal: simulation and rendering as separate submits with explicit ordering.

```c3
gpu::Queue compute = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!;
gpu::Queue graphics = gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!;

gpu::SubmitDesc sim_submit = { .command_lists = sim_lists[..] };
gpu::CompletionPoint sim_done = gpu::submit(compute, &sim_submit)!;

gpu::CompletionPoint[1] draw_waits = { sim_done };
gpu::SubmitDesc draw_submit = {
    .command_lists    = draw_lists[..],
    .completion_waits = draw_waits[..],
    .readiness        = acquired.readiness,
};
gpu::CompletionPoint draw_done = gpu::submit(graphics, &draw_submit)!;
gpu::present(&device, &acquired, draw_done)!;
```

Real overlap happens when `caps.async_compute` is true (distinct compute
queue); resources used by both queues declare `{ .graphics, .compute }` access.
Single-queue devices run the same code serialized.
Running example: `particle_sim`.

## 8. Record command lists from many threads

Goal: scale CPU-side recording.

```c3
// Each worker records against the selected queue.
gpu::CommandList list = gpu::begin_commands(queue)!;
...
gpu::ExecutableCommandList executable = gpu::end_commands(&list)!;

// The main thread submits the executable tokens in the chosen order.
```

Recording storage is cached automatically per worker. Benchmark with validation
off because validation layers may serialize recording. Quiesce every worker
before ending the frame.
Running example: `multithreaded_recording`.

## 9. Shadow mapping with compare samplers

Goal: depth-only pass, then hardware PCF.

```c3
gpu::SamplerDesc shadow_sampler = { ..., .compare_enable = true,
    .compare = gpu::CompareOp.LESS_EQUAL };
// shadow pipeline: no color formats, depth D32, front-face culling +
// .raster = { .depth_bias_slope = 1.0f } against acne
```

```glsl
float lit = sample_shadow_2d(root.shadow_map, root.shadow_sampler,
    vec3(uv, ndc_depth));
```

Running example: `shadow_mapping` (3×3 PCF).

## 10. Multiple render targets

Goal: G-buffer in one pass.

```c3
gpu::ColorTargetDesc[3] colors = { {...albedo}, {...normal}, {...position} };
gpu::RenderPassDesc pass = { .colors = colors[..], .depth = &depth_target, ... };
// pipeline: .color_formats lists all three; frag writes location 0..2
```

Running example: `deferred_shading` (plus the linear→display lessons:
Reinhard + gamma encode when the swapchain is UNORM).

## 11. Persist the pipeline cache

Goal: skip shader compiles on the next run.

```c3
usz size = gpu::get_pipeline_cache_size(&device)!;
char[] blob = mem::new_array(char, (sz)size);
usz written = gpu::get_pipeline_cache_data(&device, blob)!;   // save to disk
// next run:
gpu::DeviceDesc desc = { ..., .pipeline_cache_data = loaded_blob };
```

Blob usefulness is driver-dependent (lavapipe: header only); identical
descriptors on one device always dedup in-library regardless.
Running example: `pipeline_cache_timing`.

## 12. Query swapchain runtime state

Goal: build against the selected format and transition acquired images from
their actual prior layout.

```c3
gpu::PresentModeSupport support = gpu::get_present_mode_support(&device, swapchain)!;
if (support.mailbox) { /* recreate swapchain with PresentMode.MAILBOX */ }

gpu::SwapchainInfo info = gpu::get_swapchain_info(&device, swapchain)!;
if (info.dormant) { /* wait for a non-zero resize */ }

gpu::Format[1] color_formats = { info.format };

gpu::AcquiredImage acquired = gpu::acquire_next_image(&device, swapchain)!;
gpu::TextureUse before = acquired.prior_layout == gpu::TextureLayout.PRESENT
    ? gpu::TextureUse.PRESENT : gpu::TextureUse.UNDEFINED;
gpu::TextureBarrier to_color = gpu::texture_transition(
    acquired.texture,
    before,
    gpu::TextureUse.COLOR_ATTACHMENT,
)!;
```

Re-query `SwapchainInfo` after resize, rebuild pipelines if `format` changed,
and size any per-image data from `image_count`. `prior_layout` removes the
fixed-size seen table normally used to distinguish first acquire from a
previously presented image. FIFO is always available; other modes remain a
support query away. The graphics submission consumes `acquired.readiness`; presentation accepts
its returned completion point. `TextureUse.PRESENT` uses color-attachment
output without presentation-facing access.
Running example: `present_mode_explorer`.

## 13. Choose a memory kind

Goal: right residency per access pattern.

| Kind | For | Pattern |
|---|---|---|
| `FRAME_UPLOAD` | roots, per-frame constants | `alloc_frame_span(&frame, ...)`, valid for that token generation |
| `PERSISTENT_UPLOAD` | tables the CPU rewrites | write + `flush_buffer` |
| `DEVICE` | GPU-only working sets | upload via staging |
| `STAGING` | transfer sources | `cmd_copy_buffer_to_texture` etc. |
| `READBACK` | GPU→CPU results | `invalidate_buffer` before reading |

Running example: `memory_report` prints the arenas live; `docs/memory.md`
has the full model.

## 14. Allocate generic GPU data

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
Use `AllocationInfo` for actual capabilities. Flush CPU writes before GPU use.
For readback, wait or poll completion, invalidate the `CPU_READ` span, then read
its mapping. Running example: `memory_report`.

## 15. Pair fallible frame work

Goal: observe worker and frame-end faults without leaving the frame active on
an early `!`.

```c3
fn void? render_frame(gpu::FrameToken* frame, AppState* state) {
    gpu::GpuSpan root_span = gpu::alloc_frame_span(frame, RootArgs::size, RootArgs::alignment)!;
    record_and_submit(state, root_span)!;
}

gpu::FrameToken frame;
if (catch err = gpu::@with_frame(&frame, &device, render_frame, &state)) {
    if (frame.is_valid()) {
        gpu::end_frame(&frame)!;
    }
    return err~;
}
```

The helper requires a named optional-returning worker and caller-owned token
storage. It calls the worker directly, attempts end exactly once after worker
success or fault, and performs no heap allocation or indirect dispatch. If end
succeeds after a worker fault, the worker fault is returned. If end faults, its
exact fault takes precedence and `frame` remains live for retry; the example
above performs that retry before propagating the original helper fault. Log a
worker fault inside the worker if both diagnostics must be retained.

Prefer `@with_frame` for fallible work. Use explicit begin/end only for a
deliberate recovery flow where the caller-owned token survives the whole flow,
every work fault is caught before leaving the scope, and the end result is
always observed.

Use the existing `root_pointer_compute` and `hello_triangle_sdl` samples as the
headless and windowed lifecycle references; no lifecycle-only sample is needed.
