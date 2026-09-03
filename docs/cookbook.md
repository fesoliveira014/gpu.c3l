# Cookbook

Short recipes for common operations. Each assumes a live device, a queue,
and a recording `CommandList` named `commands` unless it says otherwise.
Exact signatures are in the [API reference](api/index.md). Ray-tracing
recipes are at the [end](#ray-tracing).

## Select a device

Iterate adapters and take the first that supports the description:

```c3
gpu::DeviceDesc desc = {
    .queues = {
        .required       = { .graphics, .compute, .transfer },
        .distinct_roles = { .compute },
    },
};
gpu::AdapterList adapters = gpu::enumerate_adapters(runtime)!;
for (uint i = 0; i < adapters.count; i++) {
    gpu::Adapter adapter = adapters.get(i)!;
    gpu::AdapterInfo info = gpu::get_adapter_info(&adapter)!;
    if (info.device_class != gpu::AdapterClass.DISCRETE) continue;
    gpu::DeviceSupport support = gpu::supports_device_desc(&adapter, &desc)!;
    if (!support.supported) {
        io::printfn("%s: %s", info.name, support.unmet_requirement);
        continue;
    }
    return gpu::create_device(&adapter, &desc);
}
return gpu::UNSUPPORTED_FEATURE~;
```

`distinct_roles = { .compute }` demands a separate compute queue. Leave
`queues` zero to accept aliasing; then check `DeviceCaps.async_compute`.

## Upload to GPU-private memory

Write a staging allocation, copy on the GPU, then barrier to the consumer
stage:

```c3
gpu::AllocationDesc staging_desc = {
    .size         = bytes.len,
    .memory_class = gpu::MemoryClass.CPU_WRITE,
    .access       = { .compute },
};
gpu::GpuAllocation staging = gpu::allocate_memory(device, &staging_desc)!;
gpu::GpuSpan staging_span = gpu::get_allocation_span(device, staging)!;
mem::copy(gpu::get_span_mapping(device, staging_span)!.ptr, bytes.ptr, bytes.len);
gpu::flush_mapped_span(device, staging_span)!;

gpu::AllocationDesc private_desc = staging_desc;
private_desc.memory_class = gpu::MemoryClass.GPU_PRIVATE;
gpu::GpuAllocation buffer = gpu::allocate_memory(device, &private_desc)!;
gpu::GpuSpan buffer_span = gpu::get_allocation_span(device, buffer)!;

gpu::BufferCopyDesc copy = { .src = staging_span, .dst = buffer_span };
gpu::cmd_copy_buffer(commands, &copy)!;
gpu::Barrier copy_to_compute = {
    .before = { .transfer },
    .after  = { .compute },
};
gpu::cmd_barrier(commands, &copy_to_compute)!;
```

Free `staging` only after the copy's completion point completes.

## Upload a texture

```c3
gpu::TextureDesc tex_desc = {
    .width  = TEX_SIZE,
    .height = TEX_SIZE,
    .format = gpu::Format.RGBA8_UNORM,
    .usage  = { .sampled, .transfer_dst },
    .access = { .graphics },
};
gpu::TextureHandle texture = gpu::create_texture(device, &tex_desc)!;

gpu::TextureBarrier to_dst = gpu::texture_transition(
    texture: texture,
    before:  { .layout = gpu::TextureLayout.UNDEFINED },
    after:   {
        .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
        .stages = { .transfer },
        .access = { .write },
    },
)!;
gpu::cmd_texture_barrier(commands, &to_dst)!;

gpu::BufferTextureCopyDesc upload = { .src = staging_span, .texture = texture };
gpu::cmd_copy_buffer_to_texture(commands, &upload)!;

gpu::TextureBarrier to_sampled = gpu::texture_transition(
    texture: texture,
    before:  to_dst.after,
    after:   gpu::sampled_at({ .fragment_shader }),
)!;
gpu::cmd_texture_barrier(commands, &to_sampled)!;
```

Zero width and height in the copy descriptor mean the whole mip. The
staging span holds tightly packed rows unless `row_length_texels` says
otherwise. The texture's state is now "sampled at fragment"; the next
transition must name that as `before`.

## Publish a texture and sampler to shaders

```c3
gpu::TextureView view = gpu::create_texture_view(device, texture, null)!;
gpu::SamplerDesc sampler_desc = {
    .min_filter = gpu::Filter.LINEAR,
    .mag_filter = gpu::Filter.LINEAR,
    .mip_filter = gpu::Filter.LINEAR,
    .address_u  = gpu::AddressMode.REPEAT,
    .address_v  = gpu::AddressMode.REPEAT,
    .address_w  = gpu::AddressMode.REPEAT,
    .max_lod    = 16.0f,
};
gpu::SamplerIndex sampler = gpu::intern_sampler(device, &sampler_desc)!;

root.albedo  = view.index;
root.sampler = sampler;
```

Keep `view` alive until the last shader read completes, then
`destroy_texture_view`. The sampler index never needs freeing. Shader side:
[textures and samplers](shader_abi.md#textures-and-samplers).

## Write a storage image, then sample it

```c3
gpu::TextureBarrier to_storage = gpu::texture_transition(
    texture: image,
    before:  { .layout = gpu::TextureLayout.UNDEFINED },
    after:   gpu::storage_at({ .compute }, { .write }),
)!;
gpu::cmd_texture_barrier(commands, &to_storage)!;
// dispatch that writes the image ...
gpu::TextureBarrier to_sampled = gpu::texture_transition(
    texture: image,
    before:  to_storage.after,
    after:   gpu::sampled_at({ .fragment_shader }),
)!;
gpu::cmd_texture_barrier(commands, &to_sampled)!;
```

The texture needs both `.storage` and `.sampled` usage. One texture cannot
be in both layouts at once; split the uses with a transition.

## Render to an offscreen texture

```c3
gpu::TextureDesc target_desc = {
    .width  = WIDTH,
    .height = HEIGHT,
    .format = gpu::Format.RGBA8_UNORM,
    .usage  = { .color_attach, .transfer_src },
    .access = { .graphics },
};
gpu::TextureHandle target = gpu::create_texture(device, &target_desc)!;
gpu::AttachmentViewDesc view_desc = { .texture = target };
gpu::AttachmentViewHandle target_view = gpu::create_attachment_view(device, &view_desc)!;

gpu::TextureBarrier to_attach = gpu::texture_transition(
    texture: target,
    before:  { .layout = gpu::TextureLayout.UNDEFINED },
    after:   {
        .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
        .stages = { .color_output },
        .access = { .read, .write },
    },
)!;
gpu::cmd_texture_barrier(commands, &to_attach)!;

gpu::ColorTargetDesc[1] colors = {{
    .view     = target_view,
    .load_op  = gpu::LoadOp.CLEAR,
    .store_op = gpu::StoreOp.STORE,
    .clear    = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
}};
gpu::RenderPassDesc pass = { .colors = colors[..], .width = WIDTH, .height = HEIGHT };
gpu::GraphicsState state = gpu::render_geometry_state(WIDTH, HEIGHT)!;
gpu::ColorTargetState[1] color_targets = { gpu::color_blend_disabled() };
state.color.targets = color_targets[..];

gpu::cmd_begin_render_pass(commands, &pass)!;
gpu::cmd_bind_pipeline(commands, pipeline)!;
gpu::cmd_set_graphics_state(commands, &state)!;
gpu::cmd_draw(
    commands:       commands,
    vertex_root:    vertex_root,
    fragment_root:  fragment_root,
    vertex_count:   3,
    instance_count: 1,
)!;
gpu::cmd_end_render_pass(commands)!;
```

A swapchain image already has an attachment view in `AcquiredImage`; only
application textures need `create_attachment_view`.

## Draw with a depth buffer

Create a `D32_FLOAT` texture, name it in the pipeline, transition it, and
enable depth in the graphics state:

```c3
gpu::TextureDesc depth_desc = {
    .width  = WIDTH,
    .height = HEIGHT,
    .format = gpu::Format.D32_FLOAT,
    .usage  = { .depth_attach },
    .access = { .graphics },
};
gpu::TextureHandle depth = gpu::create_texture(device, &depth_desc)!;
gpu::AttachmentViewDesc depth_view_desc = { .texture = depth };
gpu::AttachmentViewHandle depth_view = gpu::create_attachment_view(device, &depth_view_desc)!;

gpu::GraphicsPipelineDesc pipe_desc = {
    .vertex_shader   = { .spirv = VERTEX_SPIRV[..] },
    .fragment_shader = { .spirv = FRAGMENT_SPIRV[..] },
    .color_formats   = color_formats[..],
    .depth_format    = gpu::Format.D32_FLOAT,
};
gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(device, &pipe_desc)!;

gpu::TextureBarrier to_depth = gpu::texture_transition(
    texture: depth,
    before:  { .layout = gpu::TextureLayout.UNDEFINED },
    after:   {
        .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
        .stages = { .depth_output },
        .access = { .read, .write },
    },
)!;
gpu::cmd_texture_barrier(commands, &to_depth)!;

gpu::DepthTargetDesc depth_target = {
    .view     = depth_view,
    .load_op  = gpu::LoadOp.CLEAR,
    .store_op = gpu::StoreOp.DONT_CARE,
    .clear    = { .depth = 1.0f },
};
gpu::RenderPassDesc pass = {
    .colors = colors[..],
    .depth  = &depth_target,
    .width  = WIDTH,
    .height = HEIGHT,
};
gpu::GraphicsState state = gpu::render_geometry_state(WIDTH, HEIGHT)!;
state.raster.cull_mode = gpu::CullMode.BACK;
state.depth = {
    .test_enable  = true,
    .write_enable = true,
    .compare      = gpu::CompareOp.LESS,
};
state.color.targets = color_targets[..];
```

An indexed draw takes the index span and type:

```c3
gpu::cmd_draw_indexed(
    commands:       commands,
    vertex_root:    vertex_root,
    fragment_root:  fragment_root,
    index_span:     index_span,
    index_count:    index_count,
    instance_count: 1,
    index_type:     gpu::IndexType.U16,
)!;
```

There is no stencil.

## Configure blending and multiple targets

`GraphicsState.color.targets` has one entry per pipeline color format.
`uniform_color_state` fills a slice with one state:

```c3
gpu::ColorTargetState[3] targets;
state.color = gpu::uniform_color_state(targets[:color_target_count], gpu::alpha_blend());
gpu::cmd_set_graphics_state(commands, &state)!;
```

Or list them explicitly:

```c3
gpu::ColorTargetState[2] mixed = {
    gpu::color_blend_disabled(),
    gpu::additive_blend(),
};
state.color.targets = mixed[..];
```

Presets: `color_blend_disabled`, `alpha_blend`,
`premultiplied_alpha_blend`, `additive_blend`. For multiple render targets,
use the same ordered format list in `GraphicsPipelineDesc.color_formats`,
`RenderPassDesc.colors`, and the color state. The count is limited by
`DeviceCaps.max_color_attachments`.

## Read back a texture

Copy into `CPU_READ` memory, wait, invalidate, read. The producer's
completion point goes into `completion_waits` so the copy is ordered after
the render even on the same queue.

```c3
gpu::AllocationDesc readback_desc = {
    .size         = out.len,
    .memory_class = gpu::MemoryClass.CPU_READ,
    .access       = { .graphics },
};
gpu::GpuAllocation readback = gpu::allocate_memory(device, &readback_desc)!;
defer (void)gpu::free_allocation(device, &readback);
gpu::GpuSpan readback_span = gpu::get_allocation_span(device, readback)!;

gpu::CommandList commands = gpu::begin_commands(allocator)!;
defer (void)gpu::discard_commands(&commands);
gpu::TextureBarrier to_source = gpu::texture_transition(
    texture: target,
    before:  {
        .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
        .stages = { .color_output },
        .access = { .read, .write },
    },
    after:   {
        .layout = gpu::TextureLayout.TRANSFER_SOURCE,
        .stages = { .transfer },
        .access = { .read },
    },
)!;
gpu::cmd_texture_barrier(&commands, &to_source)!;
gpu::TextureBufferCopyDesc copy = { .texture = target, .dst = readback_span };
gpu::cmd_copy_texture_to_buffer(&commands, &copy)!;
gpu::Barrier to_host = { .before = { .transfer }, .after = { .host } };
gpu::cmd_barrier(&commands, &to_host)!;

gpu::ExecutableCommandList[1] lists = { gpu::end_commands(&commands)! };
defer (void)gpu::discard_executable_commands(&lists[0]);
gpu::CompletionWait[1] waits = {{ .point = producer, .before = { .transfer } }};
gpu::SubmitDesc submit = { .command_lists = lists[..], .completion_waits = waits[..] };
gpu::CompletionPoint done = gpu::submit(allocator.queue, &submit)!;

gpu::wait_completion(done)!;
gpu::invalidate_mapped_span(device, readback_span)!;
mem::copy(out.ptr, gpu::get_span_mapping(device, readback_span)!.ptr, out.len);
```

`get_span_mapping` never waits and `invalidate_mapped_span` never orders.
Both belong after `wait_completion`.

## Read back without blocking

Keep the allocation, span, and completion point together. Poll each update:

```c3
struct PendingReadback {
    gpu::GpuAllocation   allocation;
    gpu::GpuSpan         span;
    gpu::CompletionPoint completion;
}

fn bool? try_consume(gpu::Device* device, PendingReadback* pending) {
    if (!gpu::poll_completion(pending.completion)!) return false;
    gpu::invalidate_mapped_span(device, pending.span)!;
    char[] bytes = gpu::get_span_mapping(device, pending.span)!;
    // consume bytes ...
    gpu::free_allocation(device, &pending.allocation)!;
    return true;
}
```

## Reuse per-frame memory

Any memory the GPU reads is reusable only after the covering completion
point completes. The simplest ring is one slot per frame in flight:

```c3
struct FrameSlot {
    gpu::GpuSpan         span;
    gpu::CompletionPoint last_use;
}

fn gpu::GpuSpan? begin_frame(FrameSlot* slot) {
    if (slot.last_use.is_valid()) {
        gpu::wait_completion(slot.last_use)!;
    }
    return slot.span;
}

fn void end_frame(FrameSlot* slot, gpu::CompletionPoint submitted) {
    slot.last_use = submitted;
}
```

The same rule covers root records, indirect arguments, readback buffers,
sparse backing, and command allocators.

## Draw indirectly

A compute pass writes `DrawIndirectCommand` records into addressable
memory. A barrier to the indirect stage makes them visible:

```c3
gpu::cmd_bind_pipeline(commands, cull_pipeline)!;
gpu::cmd_dispatch(
    commands: commands,
    root:     cull_root,
    groups:   { (INSTANCE_COUNT + CULL_TILE - 1) / CULL_TILE, 1, 1 },
)!;
gpu::Barrier args_ready = {
    .before = { .compute },
    .after  = { .indirect },
};
gpu::cmd_barrier(commands, &args_ready)!;

gpu::cmd_begin_render_pass(commands, pass)!;
gpu::cmd_bind_pipeline(commands, draw_pipeline)!;
gpu::cmd_set_graphics_state(commands, state)!;
gpu::cmd_draw_indirect(
    commands:      commands,
    vertex_root:   draw_root,
    fragment_root: draw_root,
    args:          args_span,
    draw_count:    INSTANCE_COUNT,
)!;
gpu::cmd_end_render_pass(commands)!;
```

Every draw in the call sees the same root pair. Index per-draw data with
`gl_DrawID`. `cmd_draw_indexed_indirect_count` reads the draw count from a
second span when `DeviceCaps.draw_indirect_count` is true.

## Draw generated work

Generated records carry their own roots and arguments. Reserve storage on
the allocator once, while it is idle:

```c3
gpu::DeviceCaps caps = gpu::get_device_caps(device)!;
if (!caps.generated_work) return gpu::UNSUPPORTED_FEATURE~;

gpu::GeneratedWorkReservationDesc reservation = {
    .pipeline              = pipeline,
    .kind                  = gpu::GeneratedWorkKind.DRAW_INDEXED,
    .max_commands_per_list = INSTANCE_COUNT,
    .concurrent_lists      = FRAMES_IN_FLIGHT,
};
gpu::reserve_generated_work(allocator, &reservation)!;
```

Then, after the compute producer and an indirect barrier as above:

```c3
gpu::cmd_bind_pipeline(&commands, pipeline)!;
gpu::cmd_draw_indexed_generated(
    commands:       &commands,
    records:        records,
    count_span:     count_span,
    max_draw_count: INSTANCE_COUNT,
    index_span:     index_span,
    index_type:     gpu::IndexType.U32,
)!;
```

`concurrent_lists` counts generated calls in flight, not command lists. Fall
back to shared-root indirect draws when the capability is false.

## Split work across queues

Submit the producer, then pass its point as a wait on the consumer with the
first stage that reads the data:

```c3
gpu::ExecutableCommandList[1] sim_lists = { sim };
gpu::SubmitDesc sim_submit = { .command_lists = sim_lists[..] };
gpu::CompletionPoint sim_done = gpu::submit(compute_queue, &sim_submit)!;

gpu::ExecutableCommandList[1] draw_lists = { draw };
gpu::CompletionWait[1] waits = {{
    .point  = sim_done,
    .before = { .vertex_shader },
}};
gpu::SubmitDesc draw_submit = {
    .command_lists    = draw_lists[..],
    .completion_waits = waits[..],
};
gpu::CompletionPoint draw_done = gpu::submit(graphics_queue, &draw_submit)!;
```

The wait orders execution and visibility. It does not transfer ownership;
allocations must have `access` covering both queues.

## Record on several threads

One allocator per thread. Hand executable lists back to the submitting
thread through a join or another happens-before edge:

```c3
struct WorkerArgs {
    gpu::CommandAllocator      allocator;
    gpu::ExecutableCommandList result;
}

fn int record_worker(void* arg) {
    WorkerArgs* args = (WorkerArgs*)arg;
    gpu::CommandList? commands = gpu::begin_commands(&args.allocator);
    if (catch commands) return 1;
    // record ...
    gpu::ExecutableCommandList? executable = gpu::end_commands(&commands);
    if (catch executable) return 1;
    args.result = executable;
    return 0;
}

WorkerArgs[8] args;
Thread[8] threads;
for (uint w = 0; w < workers; w++) {
    args[w].allocator = gpu::create_command_allocator(device, queue)!;
    threads[w].create(&record_worker, &args[w])!;
}
gpu::ExecutableCommandList[8] lists;
for (uint w = 0; w < workers; w++) {
    threads[w].join()!;
    lists[w] = args[w].result;
}
gpu::SubmitDesc submit = { .command_lists = lists[:workers] };
gpu::CompletionPoint done = gpu::submit(queue, &submit)!;
```

Lists in one submit execute in order. Keep each allocator alive until its
lists retire.

## Measure GPU time

```c3
gpu::DeviceCaps caps = gpu::get_device_caps(device)!;
if (!caps.timestamps.queues.graphics) return gpu::UNSUPPORTED_FEATURE~;
gpu::TimestampPoolDesc pool_desc = { .capacity = 2 };
gpu::TimestampPoolHandle pool = gpu::create_timestamp_pool(device, &pool_desc)!;

gpu::cmd_reset_timestamps(commands, pool, 0, 2)!;
gpu::cmd_write_timestamp(commands, pool, 0, { .all })!;
// work ...
gpu::cmd_write_timestamp(commands, pool, 1, { .all })!;

// after submit:
gpu::wait_completion(done)!;
ulong[2] values;
gpu::read_timestamps(device, pool, 0, 2, values[..])!;
double elapsed_ns = gpu::timestamp_delta_ns(&caps.timestamps, gpu::QueueKind.GRAPHICS, values[0], values[1])!;
```

Reset before every write. `read_timestamps` does not wait; it returns
`DEVICE_BUSY` if the values are not ready. Values from different native
queues are not comparable. `cmd_resolve_timestamps` writes them to GPU
memory instead.

## Persist the pipeline cache

```c3
usz size = gpu::get_pipeline_cache_size(device)!;
char[] blob = mem::new_array(char, (sz)size);
usz written = gpu::get_pipeline_cache_data(device, blob)!;
// write blob[:written] to disk with the adapter's vendor, device, and driver ids
```

On the next run:

```c3
gpu::RuntimeDesc desc = {
    .pipeline_cache_data = blob,
    .application_name    = "my_app",
};
gpu::Runtime runtime = gpu::create_runtime(&desc)!;
```

The blob is opaque driver data. A rejected or tiny blob is a cache miss, not
an error.

## Receive diagnostics

```c3
fn void on_debug_message(gpu::DebugMessage* message, void* user_data) {
    if (message.severity < gpu::DebugMessageSeverity.WARNING) return;
    io::printfn("[%s] %s: %s", message.severity, message.operation, message.invariant);
}

gpu::RuntimeDesc desc = gpu::full_validation_runtime_desc();
desc.debug_callback     = &on_debug_message;
desc.enable_debug_names = true;
gpu::Runtime runtime = gpu::create_runtime(&desc)!;
```

The callback runs synchronously on whatever thread hit the condition. It
must not call the library. Strings are valid only during the call.

## Inspect memory

```c3
gpu::MemoryStats stats = gpu::get_memory_stats(device)!;
for (uint i = 0; i < stats.heap_count; i++) {
    io::printfn("heap %d: %d / %d bytes", i, stats.heaps[i].usage, stats.heaps[i].budget);
}
String text = gpu::build_memory_report(device, true)!;
defer text.free(mem);
io::print(text);
```

## Place several textures in one allocation

```c3
gpu::TextureRequirements[2] reqs = {
    gpu::get_texture_requirements(device, desc_a)!,
    gpu::get_texture_requirements(device, desc_b)!,
};
usz offset_b = (reqs[0].size + reqs[1].alignment - 1) / reqs[1].alignment * reqs[1].alignment;
gpu::AllocationDesc pool_desc = {
    .size                 = offset_b + reqs[1].size,
    .alignment            = reqs[0].alignment,
    .memory_class         = gpu::MemoryClass.TEXTURE,
    .access               = { .graphics },
    .texture_requirements = reqs[..],
};
gpu::GpuAllocation pool = gpu::allocate_memory(device, &pool_desc)!;
gpu::TextureHandle a = gpu::create_placed_texture(device, desc_a, pool, 0)!;
gpu::TextureHandle b = gpu::create_placed_texture(device, desc_b, pool, offset_b)!;
```

Destroy both textures before freeing the pool. `dedicated_only` in the
requirements means the texture cannot be placed.

## Sample a shadow map

Create the depth texture with `.sampled` usage, intern a comparison
sampler, and transition after the depth pass:

```c3
gpu::SamplerDesc desc = {
    .min_filter     = gpu::Filter.LINEAR,
    .mag_filter     = gpu::Filter.LINEAR,
    .address_u      = gpu::AddressMode.CLAMP_TO_EDGE,
    .address_v      = gpu::AddressMode.CLAMP_TO_EDGE,
    .address_w      = gpu::AddressMode.CLAMP_TO_EDGE,
    .compare_enable = true,
    .compare        = gpu::CompareOp.LESS_EQUAL,
};
gpu::SamplerIndex shadow_sampler = gpu::intern_sampler(device, &desc)!;

gpu::TextureBarrier to_sampled = gpu::texture_transition(
    texture: shadow_map,
    before:  {
        .layout = gpu::TextureLayout.DEPTH_ATTACHMENT,
        .stages = { .depth_output },
        .access = { .read, .write },
    },
    after:   gpu::sampled_at({ .fragment_shader }),
)!;
gpu::cmd_texture_barrier(commands, &to_sampled)!;
```

The shader calls `sample_shadow_2d(index, sampler, vec3(uv, depth))`.

## Resize or destroy a swapchain

```c3
gpu::wait_completion(last_graphics)!;
gpu::wait_swapchain_presentations(device, swapchain, RETIRE_TIMEOUT_NS)!;
gpu::resize_swapchain(device, swapchain, width, height)!;   // or destroy_swapchain
gpu::SwapchainInfo info = gpu::get_swapchain_info(device, swapchain)!;
```

Always in that order. `WAIT_TIMEOUT` from the presentation wait is safe to
retry after pumping the event loop. `RESOURCE_IN_USE` after it means an
acquired image was never presented. Rebuild pipelines if `info.format`
changed. The full loop is in
[getting started](getting_started.md#resize-and-teardown).

## Tear down

Children before parents, and nothing while it can still be read:

```c3
gpu::wait_completion(last)!;
gpu::destroy_texture_view(device, view)!;
gpu::destroy_texture(device, texture)!;
gpu::free_allocation(device, allocation)!;
```

Order for a whole application: wait all points, destroy views, pipelines,
textures, acceleration structures, allocators, and timestamp pools, free
allocations, destroy swapchains, then the device, surface, and runtime.
`RESOURCE_IN_USE` names a missed child; `DEVICE_BUSY` means work is still
running.

## Ray tracing

These recipes need `DeviceDesc.enable_ray_queries` or
`enable_ray_tracing_pipelines` and a nonzero
`RuntimeDesc.acceleration_structure_heap_capacity`. The interactive
[`cornell_box`](https://github.com/fesoliveira014/gpu.c3l-samples/tree/main/cornell_box)
sample is the complete reference.

### Build a BLAS and TLAS

Describe capacity, query requirements, allocate scratch, and give the
allocator geometry capacity:

```c3
gpu::AccelerationStructureGeometryDesc[1] geometries = {{
    .kind  = gpu::AccelerationStructureGeometryKind.TRIANGLES,
    .flags = { .opaque },
    .triangles = {
        .max_vertex_count    = vertex_count,
        .max_primitive_count = triangle_count,
        .index_type          = gpu::AccelerationStructureIndexType.U32,
    },
}};
gpu::AccelerationStructureDesc blas_desc = {
    .kind       = gpu::AccelerationStructureKind.BOTTOM_LEVEL,
    .geometries = geometries[..],
};
gpu::AccelerationStructureRequirements reqs =
    gpu::get_acceleration_structure_requirements(&device, &blas_desc)!;
gpu::AccelerationStructureHandle blas =
    gpu::create_acceleration_structure(&device, &blas_desc)!;

gpu::CommandAllocatorDesc allocator_desc = {
    .max_acceleration_structure_geometries_per_build = 1,
};
```

Record the build with caller-owned scratch of `reqs.build_scratch_size`
bytes at `reqs.scratch_alignment`:

```c3
gpu::AccelerationStructureGeometryBuildDesc[1] build_geometries = {{
    .kind = gpu::AccelerationStructureGeometryKind.TRIANGLES,
    .triangles = {
        .vertices        = vertex_span,
        .vertex_stride   = 12,
        .vertex_count    = vertex_count,
        .indices         = index_span,
        .primitive_count = triangle_count,
    },
}};
gpu::AccelerationStructureBuildDesc build = {
    .destination = blas,
    .geometries  = build_geometries[..],
    .scratch     = scratch_span,
};
gpu::cmd_build_acceleration_structure(&commands, &build)!;
```

After the BLAS build completes, pack instances and build the TLAS:

```c3
gpu::AccelerationStructureInstanceDesc instance_desc = {
    .transform_row_0 = { 1, 0, 0, 0 },
    .transform_row_1 = { 0, 1, 0, 0 },
    .transform_row_2 = { 0, 0, 1, 0 },
    .bottom_level    = blas,
    .mask            = 0xff,
};
gpu::AccelerationStructureInstance instance =
    gpu::make_acceleration_structure_instance(&device, &instance_desc)!;
// write `instance` into a 16-byte-aligned CPU_WRITE span, flush

gpu::AccelerationStructureDesc tlas_desc = {
    .kind               = gpu::AccelerationStructureKind.TOP_LEVEL,
    .max_instance_count = 1,
};
gpu::AccelerationStructureHandle tlas =
    gpu::create_acceleration_structure(&device, &tlas_desc)!;
gpu::AccelerationStructureBuildDesc tlas_build = {
    .destination    = tlas,
    .instances      = instance_span,
    .instance_count = 1,
    .scratch        = tlas_scratch_span,
};
gpu::cmd_build_acceleration_structure(&commands, &tlas_build)!;

gpu::AccelerationStructureView tlas_view =
    gpu::create_acceleration_structure_view(&device, tlas)!;
root.tlas_index = tlas_view.index;
```

Between a build and a query in the same list, barrier
`.acceleration_structure_build` to the querying stage. Across submissions,
wait on the build's point at that stage. Teardown: wait, destroy the view,
destroy the TLAS, destroy the BLAS, free scratch.

### Build from GPU-written ranges

When `caps.acceleration_structures.indirect_build` is true, a compute
shader writes one `AccelerationStructureIndirectBuildRange` per geometry.
Descriptor counts become maxima:

```c3
gpu::Barrier ranges_ready = {
    .before = { .compute },
    .after  = { .indirect, .acceleration_structure_build },
};
gpu::cmd_barrier(&commands, &ranges_ready)!;
gpu::cmd_build_acceleration_structure_indirect(&commands, &build, ranges_span)!;
```

After an indirect build the CPU does not know the actual counts, so only
indirect updates or a new direct build may follow.

### Clone a completed structure

Create a destination from the same descriptor, then:

```c3
gpu::cmd_clone_acceleration_structure(&commands, blas, clone)!;
```

No scratch, no barrier, no wait. A cloned BLAS has a new address; existing
instance records still point at the source. A cloned TLAS needs its own
view.

### Confirm procedural intersections

An AABB BLAS reports candidates only. The shader computes the real hit and
confirms it; see the
[ray-query snippet](shader_abi.md#acceleration-structures). Triangles and
AABBs cannot share a BLAS.

### Pack an SBT and trace

```c3
gpu::RayTracingPipelineCaps rt = caps.ray_tracing_pipelines;
gpu::RayTracingPipelineInfo info =
    gpu::get_ray_tracing_pipeline_info(&device, pipeline)!;
gpu::get_ray_tracing_shader_group_handles(
    device:      &device,
    pipeline:    pipeline,
    first_group: 0,
    group_count: info.total_group_count,
    out:         handle_bytes,
)!;
// copy each handle to the start of its record; align record strides to
// rt.shader_group_handle_alignment and region starts to
// rt.shader_group_base_alignment; flush the SBT span

gpu::Barrier build_to_trace = {
    .before = { .acceleration_structure_build },
    .after  = { .ray_tracing },
};
gpu::cmd_barrier(&commands, &build_to_trace)!;
gpu::cmd_bind_pipeline(&commands, pipeline)!;
gpu::cmd_trace_rays(
    commands:             &commands,
    root:                 root_address,
    shader_binding_table: &sbt,
    dimensions:           { width, height, 1 },
)!;
```

The SBT is ordinary addressable memory on a device created with ray-tracing
pipelines. Keep it, the TLAS view, and everything the root reaches alive
through the trace's completion point.

### Set an explicit stack size

For a pipeline created with `dynamic_stack_size = true`, sum the per-group
requirements of the actual shader graph. For one raygen, one miss, one
closest-hit group, and recursion depth one:

```c3
ulong raygen = gpu::get_ray_tracing_shader_group_stack_size(
    device:       &device,
    pipeline:     pipeline,
    group:        info.ray_generation.first_group,
    group_shader: gpu::RayTracingGroupShader.GENERAL,
)!;
ulong miss = gpu::get_ray_tracing_shader_group_stack_size(
    device:       &device,
    pipeline:     pipeline,
    group:        info.miss.first_group,
    group_shader: gpu::RayTracingGroupShader.GENERAL,
)!;
ulong closest_hit = gpu::get_ray_tracing_shader_group_stack_size(
    device:       &device,
    pipeline:     pipeline,
    group:        info.hit.first_group,
    group_shader: gpu::RayTracingGroupShader.CLOSEST_HIT,
)!;
ulong stack_size = raygen + (miss > closest_hit ? miss : closest_hit);

gpu::cmd_bind_pipeline(&commands, pipeline)!;
gpu::cmd_set_ray_tracing_pipeline_stack_size(&commands, stack_size)!;
```

Set it again after binding a static-stack ray pipeline. Deeper graphs must
account for recursion and callables.

### Trace with GPU-written dimensions

With `rt.indirect_dispatch`, a compute shader writes a
`TraceRaysIndirectCommand`:

```c3
gpu::Barrier args_ready = {
    .before = { .compute },
    .after  = { .indirect },
};
gpu::cmd_barrier(&commands, &args_ready)!;
gpu::cmd_bind_pipeline(&commands, pipeline)!;
gpu::cmd_trace_rays_indirect(
    commands:             &commands,
    root:                 root_address,
    shader_binding_table: &sbt,
    args:                 indirect_args,
)!;
```

With `rt.indirect2_dispatch`, the packet also carries the SBT regions:

```c3
gpu::cmd_trace_rays_indirect2(&commands, root_address, indirect2_args)!;
```

The library does not read the packet back. Dimensions must fit
`rt.max_ray_dispatch_dimensions` and `rt.max_ray_dispatch_invocation_count`.
