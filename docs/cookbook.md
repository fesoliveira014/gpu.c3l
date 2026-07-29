# Cookbook

These focused recipes assume a live device, selected queue, and explicit
ownership of every referenced resource. Exact signatures are in the
[API reference](api/index.md).

## Generate a shared shader layout

Define C3/GLSL wire data once:

```text
abi my_app;

root ComputeRoot {
    GpuAddress input_gpu;
    GpuAddress output_gpu;
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}
```

Run `tools/gen_shader_abi` with the application module and output paths. Commit
both generated files and use `--check` in CI. See
[Shader ABI](shader_abi.md#schema-generator).

## Allocate and publish root data

1. Allocate `CPU_WRITE` storage with the generated root's size/alignment.
2. Get its span and mapping.
3. Fill raw `GpuAddress`, `TextureIndex`, and `SamplerIndex` fields.
4. Flush the span.
5. Pass `get_span_address` to the draw or dispatch.
6. Keep every referenced owner live until completion.

```c3
gpu::GpuSpan root_span = gpu::get_allocation_span(&device, root_allocation)!;
DoublerRoot* root = (DoublerRoot*)gpu::get_span_mapping(&device, root_span)!.ptr;
root.input_gpu = gpu::get_span_address(&device, input_span)!;
root.output_gpu = gpu::get_span_address(&device, output_span)!;
gpu::flush_mapped_span(&device, root_span)!;
```

## Upload a texture

1. Query/support-check the `TextureDesc`.
2. Create the texture and an upload allocation.
3. Write and flush upload bytes.
4. Transition `UNDEFINED` to `COPY_DEST`.
5. Record `cmd_copy_buffer_to_texture`.
6. Transition `COPY_DEST` to `SHADER_SAMPLED`.
7. Submit and wait/poll before releasing upload storage.

Keep layout history in application state. A later use supplies the exact prior
state; the library does not remember it.

## Publish a texture to shaders

```c3
gpu::TextureView view = gpu::create_texture_view(
    &device,
    texture,
    &view_desc,
)!;
gpu::TextureIndex texture_index = view.index;
```

Store `texture_index` in root data. Keep `view` alive through every shader use,
then destroy it. The index can be recycled immediately afterward.

Intern reusable sampler state once:

```c3
gpu::SamplerIndex sampler_index = gpu::intern_sampler(&device, &sampler_desc)!;
```

Sampler indices remain stable until device destruction.

## Blocking readback

Record the producer-to-copy barrier, copy into `CPU_READ` storage, submit, wait
for the returned point, invalidate the mapped span, then read the bytes.

`get_span_mapping` does not wait for the GPU and `invalidate_mapped_span` does
not establish completion.

## Nonblocking readback

Retain `{allocation, span, completion}` in an application queue. Poll the
completion each update; only after it reports complete should you invalidate,
consume the result, and recycle/free the allocation.

## Split work across queues

Submit the producer and retain its point. Add a `CompletionWait` to the
consumer submit with destination stages that are supported by the consumer
queue:

```c3
gpu::CompletionWait wait = {
    .point = producer_done,
    .stages = { .compute },
};
gpu::ExecutableCommandList[1] command_lists = { consumer_commands };
gpu::CompletionWait[1] completion_waits = { wait };
gpu::SubmitDesc submit_desc = {
    .command_lists = command_lists[..],
    .completion_waits = completion_waits[..],
};
gpu::CompletionPoint consumer_done = gpu::submit(consumer_queue, &submit_desc)!;
```

The completion wait orders execution and visibility; it does not transfer
ownership of application allocations or raw shader values.

## Record in parallel

Create one `CommandAllocator` per concurrently recording worker. Each allocator
is bound to its exact queue and has one recording owner at a time. Different
allocators can record concurrently.

After `end_commands`, synchronize the handoff of the executable token to a
submit thread. Keep the allocator alive until submission completion retires the
unit. Reuse or destroy only after the covering point completes.

## Draw indirectly

Write one or more `DrawIndirectCommand` or `DrawIndexedIndirectCommand` records
into addressable storage. Barrier GPU-written arguments to the indirect stage,
then call the matching indirect draw.

All draws in a shared-root indirect call receive the same vertex/fragment root
pair. Put per-draw records behind the root and index them with `gl_DrawID`.
Use generated work only after checking `DeviceCaps.generated_work` and
reserving allocator preprocess storage.

## Configure graphics state

Start with `render_geometry_state(width, height)`, then provide the complete
ordered color packet required by the pipeline:

```c3
gpu::GraphicsState state = gpu::render_geometry_state(width, height)!;
state.color = gpu::uniform_color_state(
    color_target_count,
    gpu::alpha_blend(),
)!;
gpu::cmd_set_graphics_state(&commands, &state)!;
```

Apply a complete state before any draw. Later viewport/scissor calls are narrow
overrides. Changing passes or binding a pipeline does not replay state.

## Shadow comparison sampling

Create a `D32_FLOAT` depth attachment and a sampler with comparison enabled and
the desired `CompareOp`. Transition the finished depth texture from
`DEPTH_ATTACHMENT` to its sampled state before the consumer pass. Store the
view and sampler indices in root data.

The public API exposes no stencil attachment state.

## Multiple render targets

Use the same ordered color format list in:

1. `GraphicsPipelineDesc`;
2. `RenderPassDesc.colors`; and
3. `GraphicsState.color.targets`.

The count must not exceed `DeviceCaps.max_color_attachments`. Transition every
attachment explicitly before and after the pass.

## Persist a pipeline cache

At shutdown or a suitable checkpoint:

1. call `get_pipeline_cache_size`;
2. allocate that many bytes;
3. call `get_pipeline_cache_data`; and
4. persist the opaque blob with application metadata for the device/driver.

On the next run, provide the bytes as `RuntimeDesc.pipeline_cache_data`.
Treat rejection or a tiny driver blob as a cache miss, not a correctness
failure.

## Handle swapchain resize

On `SWAPCHAIN_OUT_OF_DATE`, stop acquiring, wait for covering completion
points, then call `resize_swapchain` with the current drawable extent.
Retry `RESOURCE_IN_USE`/`DEVICE_BUSY` only after making progress. Rebuild
format- or extent-dependent application state from `get_swapchain_info`.

Acquisition defaults to nonblocking. Use a finite timeout in an event loop and
handle `WAIT_TIMEOUT` without discarding the swapchain.

## Retire transient data

Associate every transient allocation interval with the last completion point
that can read it. Reuse only when `poll_completion` succeeds or
`wait_completion` returns. This same rule covers root records, indirect
arguments, sparse backing, readback buffers, and data referenced only through a
`GpuAddress`.
