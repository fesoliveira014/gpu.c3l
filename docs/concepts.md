# Concepts

[Documentation](index.md) › Concepts

`gpu.c3l` is an explicit GPU API. Explicit means the library does not track
what your program did last: it does not remember which layout a texture is
in, which buffer a shader still reads, or whether the GPU has finished the
work that used a piece of memory. Your program states each of those facts at
the point where it matters, and the library turns the statement into backend
calls without adding hidden waits, hidden copies, or hidden reference counts.

This page introduces every concept that [getting started](getting_started.md)
uses, in the order the first program meets them. Each section says what the
concept is, what an implicit API would have done for you and what that costs,
shows the library's own C3 for it, and links to the page that states the
contract. Readers who have
used Vulkan or D3D12 can skip to getting started and return here by the links.

Every snippet below is a complete function. Together they compile against
the library; `device`, `queue`, and `allocator` are the objects getting
started creates.

## Why explicit

An implicit GPU API (OpenGL, or a driver-managed engine layer) answers three
questions on your behalf on every call: is this resource still in use, what
state is it in, and when may its memory be reused. Answering them costs
bookkeeping on every call, forces the driver to guess your intent, and hides
stalls inside calls that look free.

An explicit API asks you to answer them once, where you already know the
answer. The library then does exactly what you said. The trade is a small
amount of extra code for predictable cost: command recording allocates
nothing, destruction never blocks, and no call waits on the GPU unless its
name says so.

The library keeps four promises that shape everything below:

- **Imports create nothing.** The first runtime object is the one
  `create_runtime` returns.
- **Creation is transactional.** A create call returns a complete object or a
  fault, never a half-built one.
- **Destruction never waits.** A destroy call that would have to wait returns
  a fault and leaves the object intact.
- **Completion is the only fence.** The one way to learn that GPU work is
  done is a completion point returned by `submit`.

Contract: [Lifetime rules](architecture.md#lifetime-rules).

## Handles and ownership

Public resources are named by handles: small copyable values such as
`TextureHandle`, `PipelineHandle`, and `SwapchainHandle`. A handle carries
the identity of its device and a generation counter. Copying a handle copies
a name, not ownership; destroying the resource makes every copy stale, and a
stale handle is rejected by library handle resolution.

Objects form a tree. A `Runtime` owns adapters and surfaces; a `Device` owns
allocations, textures, pipelines, command allocators, and swapchains; a
texture owns its views. A parent refuses to be destroyed while a child is
live, so teardown runs children first. `defer` statements in reverse creation
order produce that order for free.

An implicit API would destroy the tree for you at exit and free children
when the last reference dropped. This library has no reference counts to
drop, so it has no ambiguity about when memory returns: a resource is gone
exactly when its destroy call returns without a fault.

```c3
fn void? handles_and_ownership() {
    gpu::RuntimeDesc runtime_desc = { .enable_vulkan_validation = true };
    runtime_desc.application_name = "concepts";
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
    defer (void)gpu::destroy_runtime(&runtime);

    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::Adapter adapter = adapters.get(0)!;
    gpu::Device device = gpu::create_device(&adapter)!;
    defer (void)gpu::destroy_device(&device);

    gpu::TextureDesc texture_desc = {
        .width  = 256,
        .height = 256,
        .format = gpu::Format.RGBA8_UNORM,
        .usage  = { .sampled, .transfer_dst },
        .access = { .graphics },
    };
    gpu::TextureHandle texture = gpu::create_texture(&device, &texture_desc)!;
    defer (void)gpu::destroy_texture(&device, texture);

    gpu::TextureHandle copy = texture;
    assert(copy == texture);
    assert(texture.is_valid());
}
```

`is_valid` checks shape only: a nonzero handle is valid in shape even after
the resource is destroyed. Library handle resolution checks that it still
names a live resource whenever an operation resolves it.

Three public values are not handles: `GpuAddress`, `TextureIndex`, and
`SamplerIndex`. They are plain numbers that shaders read. The
[GPU addresses](#gpu-addresses-and-root-pointers) and
[texture and sampler indices](#texture-and-sampler-indices) sections explain
why they carry no ownership.

Contract: [Object model](architecture.md#object-model),
[Lifetime rules](architecture.md#lifetime-rules).

## Memory, spans, and mapping

GPU memory comes from `allocate_memory`, which returns a `GpuAllocation`: the
owning token you free later. The bytes inside are addressed through a
`GpuSpan`, a non-owning byte range with a bounds check. `get_allocation_span`
gives the span that covers the whole allocation; `checked_subspan` carves a
smaller one.

Every allocation declares a **memory class**, which decides where the memory
lives and whether the CPU can see it:

| Class | CPU can map | Typical use |
|---|---|---|
| `CPU_WRITE` | yes | staging data the GPU reads once |
| `CPU_WRITE_GPU_LOCAL` | yes | small records the GPU reads often: root data, per-frame constants |
| `CPU_READ` | yes | readback of GPU results |
| `GPU_PRIVATE` | no | buffers only shaders touch |
| `TEXTURE` | no | backing memory for placed textures |

The class is fixed at allocation. An implicit API would pick placement from
usage hints and move memory behind your back when it guessed wrong. Here
nothing moves, which is what makes a [GPU address](#gpu-addresses-and-root-pointers)
stable for the allocation's lifetime.

Mapping is the CPU view of a mappable span. `allocate_mapped_memory` returns
the span, its mapping, and its GPU address in one `MappedGpuSpan`. Writes
from the CPU are not visible to the GPU until `flush_mapped_span`; GPU writes
are not visible to the CPU until `invalidate_mapped_span`. On coherent memory
both are no-ops, and calling them is still correct.

```c3
fn void? memory_spans_and_mapping(gpu::Device* device) {
    gpu::AllocationDesc staging_desc = {
        .size         = 1024,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .transfer },
        .debug_name   = "staging",
    };
    gpu::MappedGpuSpan staging = gpu::allocate_mapped_memory(device, &staging_desc)!;
    gpu::GpuAllocation staging_allocation = staging.span.allocation();
    defer (void)gpu::free_allocation(device, &staging_allocation);

    for (usz i = 0; i < staging.bytes.len; i++) staging.bytes[i] = (char)i;
    gpu::flush_mapped_span(device, staging.span)!;

    gpu::GpuSpan first_half = staging.span.checked_subspan(0, 512)!;
    gpu::AllocationDesc private_desc = {
        .size         = 512,
        .memory_class = gpu::MemoryClass.GPU_PRIVATE,
        .access       = { .compute, .transfer },
        .debug_name   = "private",
    };
    gpu::GpuAllocation private_allocation = gpu::allocate_memory(device, &private_desc)!;
    defer (void)gpu::free_allocation(device, &private_allocation);
    gpu::GpuSpan destination = gpu::get_allocation_span(device, private_allocation)!;
    assert(first_half.size == destination.size);
}
```

`access` lists the queue roles that will touch the memory. The library uses
it to place the allocation where every listed queue can reach it. Applications
must restrict GPU use to the listed roles.

Contract: [Memory](architecture.md#memory).

## GPU addresses and root pointers

A `GpuAddress` is the 64-bit device address of a span. Shaders read it as a
pointer: GLSL `buffer_reference` blocks dereference it directly. Because
allocations never move, an address is valid from allocation until free.

This is how every shader receives its data. There are no descriptor sets to
build and no binding slots to assign. A dispatch or draw pushes one address,
the **root**, into the shader; the shader reads a struct through it, and the
struct holds whatever else the shader needs: more addresses, texture indices,
constants.

```glsl
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430) buffer Params {
    uint64_t input_gpu;
    uint64_t output_gpu;
    uint     count;
};

layout(push_constant) uniform Push {
    uint64_t root_gpu;
};

void main() {
    Params params = Params(root_gpu);
    // params.input_gpu and params.output_gpu are addresses of other spans.
}
```

The C3 side writes the same struct into mapped memory and passes its address:

```c3
struct Params {
    gpu::GpuAddress input_gpu;
    gpu::GpuAddress output_gpu;
    uint            count;
}

fn void? root_pointers(
    gpu::Device* device,
    gpu::CommandList* commands,
    gpu::GpuSpan input,
    gpu::GpuSpan output,
    uint count,
) {
    gpu::AllocationDesc root_desc = {
        .size         = Params::size,
        .alignment    = Params::alignment,
        .memory_class = gpu::MemoryClass.CPU_WRITE_GPU_LOCAL,
        .access       = { .compute },
        .debug_name   = "params",
    };
    gpu::MappedGpuSpan root = gpu::allocate_mapped_memory(device, &root_desc)!;
    gpu::GpuAllocation root_allocation = root.span.allocation();
    defer (void)gpu::free_allocation(device, &root_allocation);

    Params* params = (Params*)root.bytes.ptr;
    params.input_gpu  = gpu::get_span_address(device, input)!;
    params.output_gpu = gpu::get_span_address(device, output)!;
    params.count      = count;
    gpu::flush_mapped_span(device, root.span)!;

    gpu::cmd_dispatch(
        commands: commands,
        root:     root.address,
        groups:   { (count + 63) / 64, 1, 1 },
    )!;
}
```

The two structs must agree byte for byte. The library ships a schema
generator that emits both from one `.abi` file; the
[shader ABI](shader_abi.md#schema-generator) page covers it.

What an implicit API hid here is the binding model: descriptor sets, layouts,
and the driver's tracking of which buffer each slot points at. The cost of
doing without it is one rule you must keep yourself: the allocation behind an
address stays alive, and its contents stay unchanged, for as long as a shader
that may still run can read it. The [completion points](#completion-points)
section says how you know when that is.

Contract: [Shaders and pipelines](architecture.md#shaders-and-pipelines),
[Root push](shader_abi.md#root-push).

## Texture and sampler indices

Textures are not addressable by pointer. Shaders reach them through a
device-wide table, the **heap**, by integer index. `create_texture_view`
publishes a texture (or a subrange of its mips and layers) into the heap and
returns a `TextureView` whose `index` field is the `TextureIndex` shaders use.
`intern_sampler` returns a `SamplerIndex` for a sampler description; equal
descriptions get equal indices, and an index lives as long as the device.

Indices travel to shaders inside root data like any other number:

```c3
struct MaterialParams {
    gpu::TextureIndex albedo;
    gpu::SamplerIndex sampler;
    uint              _pad0;
    uint              _pad1;
}

fn MaterialParams? texture_and_sampler_indices(gpu::Device* device, gpu::TextureHandle texture) {
    gpu::TextureViewDesc view_desc = {};
    gpu::TextureView view = gpu::create_texture_view(device, texture, &view_desc)!;
    defer (void)gpu::destroy_texture_view(device, view);

    gpu::SamplerDesc sampler_desc = {
        .min_filter = gpu::Filter.LINEAR,
        .mag_filter = gpu::Filter.LINEAR,
        .address_u  = gpu::AddressMode.REPEAT,
        .address_v  = gpu::AddressMode.REPEAT,
    };
    gpu::SamplerIndex sampler = gpu::intern_sampler(device, &sampler_desc)!;

    return { .albedo = view.index, .sampler = sampler };
}
```

In GLSL the heap is a set of unsized arrays declared by
`descriptor_heap.glsl` (`gpu_texture_heap`, `gpu_sampler_heap`, and their
storage, 3D, and cube counterparts). A shader combines the two indices at the
sample site:

```glsl
#include "descriptor_heap.glsl"

vec4 sample_albedo(uint texture_index, uint sampler_index, vec2 uv) {
    return texture(
        sampler2D(gpu_texture_heap[GPU_HEAP_SLOT(texture_index)],
                  gpu_sampler_heap[GPU_HEAP_SLOT(sampler_index)]),
        uv);
}
```

A `TextureView` owns its heap slot. Destroying the view frees the slot at
once, and the number in your root data now names nothing, or soon names some
other texture. That is the same rule as for addresses: the value is not a
reference, so the thing it names must outlive every shader that can read it.
An implicit API would have refused to free the texture while a draw still
used it; here the refusal comes from you waiting on the right completion
point first.

Contract: [Textures and shader indices](architecture.md#textures-and-shader-indices).

## Commands

GPU work is recorded into a `CommandList`, closed into an
`ExecutableCommandList`, and handed to a queue with `submit`. Recording
happens on the CPU and touches nothing on the GPU; the list is a description
of work, not the work.

Lists come from a `CommandAllocator`. An allocator belongs to one queue and
owns a fixed number of command units. `begin_commands` takes a unit and
returns a recording list; `end_commands` closes it; `submit` consumes it. The
unit returns to the allocator when the work that used it completes.

```c3
fn gpu::CompletionPoint? commands(
    gpu::Device* device,
    gpu::Queue queue,
    gpu::CommandAllocator* allocator,
    gpu::PipelineHandle pipeline,
    gpu::GpuAddress root,
) {
    gpu::CommandList list = gpu::begin_commands(allocator)!;
    defer (void)gpu::discard_commands(&list);
    gpu::cmd_bind_pipeline(&list, pipeline)!;
    gpu::cmd_dispatch(&list, root, { 1, 1, 1 })!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&list)!;
    defer (void)gpu::discard_executable_commands(&executable);

    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit_desc = { .command_lists = lists[..] };
    return gpu::submit(queue, &submit_desc);
}
```

The two `defer` lines are fault paths: after a successful `end_commands` the
first discard is a no-op, and after a successful `submit` so is the second.
On an early fault they return the unit to the allocator.

Recording is cheap by construction. The allocator preallocates its scratch,
so `cmd_*` calls allocate nothing, and a list records on one thread only. To
record on several threads, give each thread its own allocator; distinct
allocators never contend.

An implicit API would let you issue draw calls directly and would batch them
behind the scenes, deciding when to flush. The explicit list makes the batch
boundary yours: one list is one unit of work, one submit is one unit of
completion.

Contract: [Commands](architecture.md#commands),
[Threading](architecture.md#threading).

## Completion points

`submit` returns a `CompletionPoint`. It is the only statement the library
makes about GPU progress: when the point completes, everything in that submit
has finished, and so has everything that submit waited on.

Every reuse question has the same answer. May this staging memory be
overwritten? May this allocator hand out the unit that list used? May this
texture be destroyed? Yes, once the point that covers its last use has
completed. The library does not know which point that is, because it keeps
no per-resource history. You know, because you submitted the work.

```c3
fn void? completion_points(
    gpu::Device* device,
    gpu::GpuAllocation* staging,
    gpu::CompletionPoint upload_done,
) {
    if (gpu::poll_completion(upload_done)!) {
        gpu::free_allocation(device, staging)!;
        return;
    }
    gpu::wait_completion(upload_done)!;
    gpu::free_allocation(device, staging)!;
}
```

`poll_completion` never blocks; `wait_completion` blocks until the point
completes or a timeout passes. Wait for the last GPU use before destroying a
resource or freeing its memory: those calls do not infer application GPU use
or wait. Live library dependents can still cause `RESOURCE_IN_USE`, and
incomplete library-owned work can cause `DEVICE_BUSY`; the object stays
intact for a retry.

Points are values. Copy them, store one per frame, pass them across threads.
A point stays meaningful until its device is destroyed. `CompletionPoint`'s
zero value is invalid and `is_valid` reports it, which lets a frame loop keep
"the last submit" in a variable that starts empty.

An implicit API fences for you by counting: each resource remembers the last
command that touched it, and every call checks. The explicit point removes
the per-resource memory and the per-call check; the cost is that you keep the
points that matter, which is usually one per frame or one per upload batch.

Contract: [Lifetime rules](architecture.md#lifetime-rules),
[Commands](architecture.md#commands).

## Barriers and texture state

Within one queue, commands start in order but may overlap and may see stale
memory. A **barrier** is the statement that a later command depends on an
earlier one. `Barrier` names two stage masks: everything in `before` must
finish, and its writes must be visible, before anything in `after` starts.

```c3
fn void? barriers(gpu::CommandList* commands) {
    gpu::Barrier compute_to_transfer = {
        .before = { .compute },
        .after  = { .transfer },
    };
    gpu::cmd_barrier(commands, &compute_to_transfer)!;

    gpu::Barrier transfer_to_host = {
        .before = { .transfer },
        .after  = { .host },
    };
    gpu::cmd_barrier(commands, &transfer_to_host)!;
}
```

The `host` stage means the CPU: a barrier into `host` at the end of a list,
followed by waiting the completion point and `invalidate_mapped_span`, is the
readback sequence.

Textures add a second dimension. A texture has a **layout**, an
implementation-chosen memory arrangement that differs between "being sampled",
"being rendered to", and "being copied into". Changing it is a GPU operation
with a cost, so it is explicit: a `TextureBarrier` names the exact state the
texture is in (`before`) and the state it should be in (`after`). State is
layout plus the stages and access that use it.

```c3
fn void? texture_state(gpu::CommandList* commands, gpu::TextureHandle texture) {
    gpu::TextureBarrier upload_to_sampled = gpu::texture_transition(
        texture: texture,
        before:  {
            .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
            .stages = { .transfer },
            .access = { .write },
        },
        after:   {
            .layout = gpu::TextureLayout.SAMPLED,
            .stages = { .fragment_shader },
            .access = { .read },
        },
    )!;
    gpu::cmd_texture_barrier(commands, &upload_to_sampled)!;
}
```

The library does not remember a texture's layout. If `before` is wrong, the
transition is wrong. Explicit Vulkan validation can help diagnose it. The
first transition of a new texture starts from the zero state, layout
`UNDEFINED`, whose contents are unspecified.

An implicit API tracks every resource's state and inserts barriers where it
sees a hazard. That tracking is what makes such APIs slow to record and hard
to make multithreaded; it also inserts barriers you did not need. Explicit
barriers cost one line where you already know the hazard exists.

Contract: [Synchronization](architecture.md#synchronization),
[Textures and shader indices](architecture.md#textures-and-shader-indices).

## Queues

A queue is where submits go. The library exposes three roles: `GRAPHICS`,
`COMPUTE`, and `TRANSFER`. A device may back two roles with one native queue;
`DeviceCaps.async_compute` says whether compute is separate. Ask for the
roles you need in `DeviceDesc.queues`, and check support before creating the
device.

Work on different queues runs concurrently and in no defined order. To order
it, a submit lists the completion points it waits on and the stages that must
wait:

```c3
fn gpu::CompletionPoint? queues(
    gpu::Queue graphics,
    gpu::ExecutableCommandList draw,
    gpu::CompletionPoint compute_done,
) {
    gpu::ExecutableCommandList[1] lists = { draw };
    gpu::CompletionWait[1] waits = {{
        .point  = compute_done,
        .before = { .vertex_shader },
    }};
    gpu::SubmitDesc submit_desc = {
        .command_lists    = lists[..],
        .completion_waits = waits[..],
    };
    return gpu::submit(graphics, &submit_desc);
}
```

Memory shared between queues declares both roles in `AllocationDesc.access`.
Command allocators do not cross queues: create one per queue you submit to.

Contract: [Runtime, adapters, devices](architecture.md#runtime-adapters-devices),
[Synchronization](architecture.md#synchronization).

## Render passes and graphics state

Graphics work draws into attachments: color targets and an optional depth
target. An `AttachmentViewHandle` names one mip and layer of a texture as a
target. `cmd_begin_render_pass` takes a `RenderPassDesc` that lists the
targets with their load and store operations (clear, load, store, discard);
draws follow; `cmd_end_render_pass` closes it.

A graphics pipeline fixes only what changes rarely: shaders, target formats,
sample count, polygon mode. Everything else, from viewport and scissor to
topology, culling, depth test, and blending, is one `GraphicsState` packet
set inside the pass with `cmd_set_graphics_state`. `render_geometry_state`
builds a full-area, no-culling, no-depth packet to start from.

```c3
fn void? render_passes(
    gpu::Device* device,
    gpu::CommandList* commands,
    gpu::TextureHandle target,
    gpu::PipelineHandle pipeline,
    uint width,
    uint height,
) {
    gpu::AttachmentViewDesc attachment_desc = { .texture = target };
    gpu::AttachmentViewHandle attachment = gpu::create_attachment_view(device, &attachment_desc)!;
    defer (void)gpu::destroy_attachment_view(device, attachment);

    gpu::ColorTargetDesc[1] colors = {{
        .view     = attachment,
        .load_op  = gpu::LoadOp.CLEAR,
        .store_op = gpu::StoreOp.STORE,
        .clear    = { .rgba = { 0.0f, 0.0f, 0.0f, 1.0f } },
    }};
    gpu::RenderPassDesc pass = { .colors = colors[..] };
    gpu::GraphicsState state = gpu::render_geometry_state(width, height)!;
    gpu::ColorTargetState[1] color_state = { gpu::color_blend_disabled() };
    state.color.targets = color_state[..];

    gpu::cmd_begin_render_pass(commands, &pass)!;
    gpu::cmd_bind_pipeline(commands, pipeline)!;
    gpu::cmd_set_graphics_state(commands, &state)!;
    gpu::cmd_draw(
        commands:       commands,
        vertex_root:    (gpu::GpuAddress)0,
        fragment_root:  (gpu::GpuAddress)0,
        vertex_count:   3,
        instance_count: 1,
    )!;
    gpu::cmd_end_render_pass(commands)!;
}
```

The texture must be in the `COLOR_ATTACHMENT` state when the pass begins;
the [barrier](#barriers-and-texture-state) that puts it there is yours.
An implicit API would create framebuffers and pipeline variants for each
combination of state; here one pipeline serves every state packet.

Contract: [Commands and rendering](api/commands_and_rendering.md).

## Presentation

Drawing to a window means drawing into an image the window system owns. A
`Surface` wraps a native window; a `SwapchainHandle` holds the images. A
frame acquires one image, records into it, submits, and presents.

Two things make this explicit rather than a `swap_buffers` call. First, the
acquired image arrives in whatever state the presentation engine left it, so
the frame transitions it to `COLOR_ATTACHMENT` before drawing and to
`PRESENT` after. Second, the image is not ready to draw into at acquire time;
the engine hands over a `readiness` token that the first submit writing the
image must wait on, and `present` needs the completion point of that submit.

```c3
fn gpu::CompletionPoint? presentation(
    gpu::Device* device,
    gpu::Queue graphics,
    gpu::SwapchainHandle swapchain,
    gpu::CommandAllocator* allocator,
) {
    gpu::AcquiredImage acquired = gpu::acquire_next_image(device, swapchain)!;

    gpu::CommandList commands = gpu::begin_commands(allocator)!;
    defer (void)gpu::discard_commands(&commands);
    gpu::TextureBarrier to_attachment = gpu::texture_transition(
        texture: acquired.texture,
        before:  acquired.prior_state,
        after:   {
            .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
            .stages = { .color_output },
            .access = { .read, .write },
        },
    )!;
    gpu::cmd_texture_barrier(&commands, &to_attachment)!;
    gpu::TextureBarrier to_present = gpu::texture_transition(
        texture: acquired.texture,
        before:  to_attachment.after,
        after:   { .layout = gpu::TextureLayout.PRESENT },
    )!;
    gpu::cmd_texture_barrier(&commands, &to_present)!;

    gpu::ExecutableCommandList[1] lists = { gpu::end_commands(&commands)! };
    defer (void)gpu::discard_executable_commands(&lists[0]);
    gpu::SubmitDesc submit_desc = {
        .command_lists    = lists[..],
        .readiness        = acquired.readiness,
        .readiness_before = { .color_output },
    };
    gpu::CompletionPoint frame = gpu::submit(graphics, &submit_desc)!;
    gpu::present(device, &acquired, frame)!;
    return frame;
}
```

Acquire returns `WAIT_TIMEOUT` when no image is free and
`SWAPCHAIN_OUT_OF_DATE` when the window changed size; both are normal frame
outcomes, not errors. Resizing or destroying a swapchain follows the
completion rule like everything else: wait the last frame's point, wait
pending presentations, then act.

Contract: [Presentation](architecture.md#presentation).

## Validation and diagnostics

Applications own valid GPU usage, ordering, and resource lifetimes. The
library protects its host structures, resolves handles safely, and reports
actual operational failures. `enable_vulkan_validation` requests Vulkan
diagnostics during development; it does not prove all application usage.

Messages arrive through a callback set on `RuntimeDesc`, synchronously, from
whichever thread made the call:

```c3
fn void report(gpu::DebugMessage* message, void* user_data) {
    io::eprintfn("%s: %s (%s)", message.severity, message.operation, message.invariant);
}

fn gpu::Runtime? validation_and_diagnostics() {
    gpu::RuntimeDesc desc = { .enable_vulkan_validation = true };
    desc.application_name = "concepts";
    desc.debug_callback = &report;
    return gpu::create_runtime(&desc);
}
```

Enable Vulkan validation explicitly during development. A zero `RuntimeDesc`
leaves it disabled. Memory reached through a `GpuAddress` or a shader index
remains the application's responsibility.

Contract: [Diagnostics and cost](architecture.md#diagnostics-and-cost).

## Next

[Getting started](getting_started.md) assembles these pieces into a compute
program and a windowed triangle. [Architecture](architecture.md) states each
contract in full.
