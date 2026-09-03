# Getting started

Two programs. The first doubles an array on the GPU with no window. The
second draws a triangle in an SDL3 window. Each is walked through in
sections; the complete sources are linked.

Targets: C3 0.8.3, `linux-x64` or `windows-x64`, Vulkan 1.3.

## Prerequisites

- `c3c --version` reports 0.8.3.
- A Vulkan 1.3 loader and driver. On a headless Linux box, lavapipe works.
- `glslangValidator` or `glslc` to compile GLSL to SPIR-V.
- Git with submodule support, if cloning instead of using a release.

## Install

Lay the application out like this:

```text
hello_gpu/
├── lib/
│   └── gpu.c3l/          release bundle or recursive clone
├── shaders/
│   └── doubler.comp.glsl
├── src/
│   └── main.c3
└── project.json
```

Either extract the
[latest release](https://github.com/fesoliveira014/gpu.c3l/releases/latest)
for your target into `lib/gpu.c3l`, or clone:

```sh
git clone --recurse-submodules \
  https://github.com/fesoliveira014/gpu.c3l.git lib/gpu.c3l
```

Both forms ship `gpu` plus its three binding packages. `project.json`
resolves all four:

```json
{
  "langrev": "1",
  "dependency-search-paths": [ "lib", "lib/gpu.c3l/lib" ],
  "dependencies": [ "gpu", "vk", "vma", "spvreflect" ],
  "output": "build",
  "wincrt": "dynamic",
  "targets": {
    "hello_gpu": {
      "type": "executable",
      "sources": [ "src/main.c3" ]
    }
  }
}
```

Keep `"wincrt": "dynamic"` on Windows. The vendored VMA library is built
against the release CRT.

## Step 1: double an array on the GPU

Complete source:
[`examples/getting_started/src/main.c3`](https://github.com/fesoliveira014/gpu.c3l/blob/main/examples/getting_started/src/main.c3).
Shader:
[`examples/getting_started/shaders/doubler.comp.glsl`](https://github.com/fesoliveira014/gpu.c3l/blob/main/examples/getting_started/shaders/doubler.comp.glsl).

### The shader

The compute shader receives one 64-bit root address by push constant, reads
a root struct through it, and follows two more addresses to the arrays:

```glsl
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "buffer_reference.glsl"

layout(local_size_x = 64) in;

layout(buffer_reference, std430) buffer DoublerRoot {
    uint64_t input_gpu;
    uint64_t output_gpu;
    uint count;
};
GPU_DECLARE_READONLY_ARRAY_REF(InBuf, float);
GPU_DECLARE_WRITEONLY_ARRAY_REF(OutBuf, float);

layout(push_constant) uniform Push {
    uint64_t root_gpu;
};

void main() {
    DoublerRoot root = DoublerRoot(root_gpu);
    uint i = gl_GlobalInvocationID.x;
    if (i < root.count) {
        OutBuf(root.output_gpu).values[i] = InBuf(root.input_gpu).values[i] * 2.0;
    }
}
```

Compile it with the library's include directory on the path:

```sh
glslangValidator -V --target-env vulkan1.3 \
  -I lib/gpu.c3l/include/shaders \
  shaders/doubler.comp.glsl -o shaders/doubler.comp.spv
```

The C3 side declares the same root struct:

```c3
struct DoublerRoot {
    gpu::GpuAddress input_gpu;
    gpu::GpuAddress output_gpu;
    uint            count;
}
```

### Runtime and device

```c3
gpu::RuntimeDesc runtime_desc = gpu::full_validation_runtime_desc();
runtime_desc.application_name = "hello_gpu";
gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
defer (void)gpu::destroy_runtime(&runtime);

gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
gpu::Adapter adapter = adapters.get(0)!;
gpu::Device device = gpu::create_device(&adapter)!;
defer (void)gpu::destroy_device(&device);
```

`full_validation_runtime_desc` turns on contract validation and the Vulkan
validation layer. Use it during development. A zero `RuntimeDesc` turns both
off.

`create_device` with no descriptor selects default queues and no
presentation. The `defer` lines destroy in reverse order, which is the
required order: children before parents.

### Memory

Three allocations: input the CPU writes, output the CPU reads, and the root
record.

```c3
gpu::AllocationDesc input_desc = {
    .size         = COUNT * float::size,
    .alignment    = 16,
    .memory_class = gpu::MemoryClass.CPU_WRITE,
    .access       = { .compute },
    .debug_name   = "input",
};
gpu::AllocationDesc output_desc = input_desc;
output_desc.memory_class = gpu::MemoryClass.CPU_READ;
output_desc.debug_name = "output";

gpu::GpuAllocation input = gpu::allocate_memory(&device, &input_desc)!;
defer (void)gpu::free_allocation(&device, &input);
gpu::GpuAllocation output = gpu::allocate_memory(&device, &output_desc)!;
defer (void)gpu::free_allocation(&device, &output);
```

`access` names the queue roles that will touch the memory. Write the input
through its mapping and flush:

```c3
gpu::GpuSpan in_span = gpu::get_allocation_span(&device, input)!;
gpu::GpuSpan out_span = gpu::get_allocation_span(&device, output)!;
float* in_data = (float*)gpu::get_span_mapping(&device, in_span)!.ptr;
for (uint i = 0; i < COUNT; i++) in_data[i] = (float)i;
gpu::flush_mapped_span(&device, in_span)!;
```

### Pipeline

The SPIR-V is embedded at compile time and borrowed only for the create call:

```c3
const char[*] DOUBLER_SPIRV = $embed("../shaders/doubler.comp.spv");

gpu::ComputePipelineDesc pipe_desc = {
    .shader = { .spirv = DOUBLER_SPIRV[..], .entry_point = "main" },
};
gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pipe_desc)!;
defer (void)gpu::destroy_pipeline(&device, pipeline);
```

### Root data

Allocate the root record in mapped memory, fill it with the two array
addresses, and flush:

```c3
gpu::AllocationDesc root_desc = {
    .size         = DoublerRoot::size,
    .alignment    = DoublerRoot::alignment,
    .memory_class = gpu::MemoryClass.CPU_WRITE,
    .access       = { .compute },
    .debug_name   = "doubler_root",
};
gpu::GpuAllocation root_allocation = gpu::allocate_memory(&device, &root_desc)!;
defer (void)gpu::free_allocation(&device, &root_allocation);
gpu::GpuSpan root_span = gpu::get_allocation_span(&device, root_allocation)!;

DoublerRoot* root = (DoublerRoot*)gpu::get_span_mapping(&device, root_span)!.ptr;
root.input_gpu  = gpu::get_span_address(&device, in_span)!;
root.output_gpu = gpu::get_span_address(&device, out_span)!;
root.count      = COUNT;
gpu::flush_mapped_span(&device, root_span)!;
gpu::GpuAddress root_address = gpu::get_span_address(&device, root_span)!;
```

### Record, submit, wait

```c3
gpu::Queue queue = gpu::get_queue(&device, gpu::QueueKind.COMPUTE)!;
gpu::CommandAllocator allocator = gpu::create_command_allocator(&device, queue)!;
defer (void)gpu::destroy_command_allocator(&allocator);

gpu::CommandList commands = gpu::begin_commands(&allocator)!;
defer (void)gpu::discard_commands(&commands);
gpu::cmd_bind_pipeline(&commands, pipeline)!;
gpu::cmd_dispatch(
    commands: &commands,
    root:     root_address,
    groups:   { (COUNT + 63) / 64, 1, 1 },
)!;
gpu::Barrier to_host = {
    .before = { .compute },
    .after  = { .host },
};
gpu::cmd_barrier(&commands, &to_host)!;
gpu::ExecutableCommandList executable = gpu::end_commands(&commands)!;
defer (void)gpu::discard_executable_commands(&executable);

gpu::ExecutableCommandList[1] lists = { executable };
gpu::SubmitDesc submit = { .command_lists = lists[..] };
gpu::CompletionPoint completion = gpu::submit(queue, &submit)!;
gpu::wait_completion(completion)!;
```

The compute-to-host barrier makes the output visible to the CPU. `submit`
consumes the executable list and returns the point that every later reuse
waits on. The deferred discards are no-ops after a successful end and
submit; they only run on an early fault.

### Read back

```c3
gpu::invalidate_mapped_span(&device, out_span)!;
float* out_data = (float*)gpu::get_span_mapping(&device, out_span)!.ptr;
for (uint i = 0; i < COUNT; i++) {
    if (out_data[i] != (float)i * 2.0f) return gpu::INVALID_ARGUMENT~;
}
```

### Run

```sh
c3c run hello_gpu --path .
```

```text
hello_gpu: all 256 values doubled on the GPU
```

## Step 2: a triangle in an SDL3 window

The complete program lives in the samples repository:
[`hello_triangle_sdl`](https://github.com/fesoliveira014/gpu.c3l-samples/tree/main/hello_triangle_sdl).
Its window and surface helpers are in
[`shared/sample_window_sdl.c3`](https://github.com/fesoliveira014/gpu.c3l-samples/blob/main/shared/sample_window_sdl.c3).
The sample also uploads a texture and captures screenshots; the sections
below cover the presentation path only.

### Project

Add [`sdl3.c3l`](https://github.com/fesoliveira014/sdl3.c3l) beside the GPU
library and list it as a target dependency:

```sh
git clone https://github.com/fesoliveira014/sdl3.c3l.git lib/sdl3.c3l
```

```json
"targets": {
  "hello_triangle": {
    "type": "executable",
    "dependencies": [ "sdl3" ],
    "sources": [ "src/main.c3" ]
  }
}
```

### Shaders

The vertex shader generates positions from `gl_VertexIndex`, so neither
stage reads root data:

```glsl
#version 460
const vec2 POSITIONS[3] = vec2[](vec2(-0.8, -0.7), vec2(0.8, -0.7), vec2(0.0, 0.8));
void main() {
    gl_Position = vec4(POSITIONS[gl_VertexIndex], 0.0, 1.0);
}
```

```glsl
#version 460
layout(location = 0) out vec4 out_color;
void main() {
    out_color = vec4(0.95, 0.35, 0.15, 1.0);
}
```

Compile both with `--target-env vulkan1.3` as in step 1.

### Window and surface

Ask SDL3 for the native handles of the active video driver and pass them to
the matching surface module:

```c3
import gpu::surface::wayland;
import gpu::surface::win32;
import gpu::surface::x11;

fn gpu::Surface? create_sdl_surface(gpu::Runtime* runtime, sdl::Window* window) {
    sdl::PropertiesID props = sdl::get_window_properties(window);
    ZString driver = (ZString)sdl::get_current_video_driver();

    if (driver.str_view() == "x11") {
        return gpu::surface::x11::create_surface(
            runtime,
            (gpu::surface::x11::DisplayHandle)sdl::get_pointer_property(
                props, (char*)sdl::WindowProperties.X11_DISPLAY_POINTER, null),
            (gpu::surface::x11::WindowHandle)sdl::get_number_property(
                props, (char*)sdl::WindowProperties.X11_WINDOW_NUMBER, 0),
        );
    }
    // wayland and windows branches are the same shape; see the sample.
    return UNSUPPORTED_VIDEO_DRIVER~;
}
```

The surface borrows the window. Keep the window alive until
`destroy_surface`.

### Presentation device

A device that presents is created with the surface in its descriptor. Pick
the first adapter that supports it:

```c3
fn gpu::Device? create_presentation_device(gpu::Runtime* runtime, gpu::Surface* surface) {
    gpu::DeviceDesc desc = { .surface = *surface };
    gpu::AdapterList adapters = gpu::enumerate_adapters(runtime)!;
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!;
        if (!gpu::supports_device_desc(&adapter, &desc)!.supported) continue;
        return gpu::create_device(&adapter, &desc);
    }
    return gpu::UNSUPPORTED_FEATURE~;
}
```

### Swapchain

```c3
gpu::SwapchainDesc swapchain_desc = {
    .width            = width,
    .height           = height,
    .preferred_format = gpu::Format.BGRA8_UNORM,
    .present_mode     = gpu::PresentMode.FIFO,
    .debug_name       = "hello_triangle_swapchain",
};
gpu::SwapchainHandle swapchain =
    gpu::create_swapchain(&device, &surface, &swapchain_desc)!;
gpu::SwapchainInfo swapchain_info = gpu::get_swapchain_info(&device, swapchain)!;
```

`width` and `height` come from `sdl::get_window_size_in_pixels`. The backend
may pick a different format; read the actual one from `swapchain_info` and
build the pipeline against it.

### Pipeline

```c3
gpu::Format[1] color_formats = { swapchain_info.format };
gpu::GraphicsPipelineDesc pipeline_desc = {
    .vertex_shader   = { .spirv = VERTEX_SPIRV[..], .entry_point = "main" },
    .fragment_shader = { .spirv = FRAGMENT_SPIRV[..], .entry_point = "main" },
    .color_formats   = color_formats[..],
    .debug_name      = "hello_triangle",
};
gpu::PipelineHandle pipeline = gpu::create_graphics_pipeline(&device, &pipeline_desc)!;
```

Only formats, sample count, and polygon mode are pipeline state. Topology,
culling, depth, and blending are set per pass with `GraphicsState`.

### Frame: acquire

```c3
gpu::AcquiredImage? acquired = gpu::acquire_next_image(
    device:     &device,
    swapchain:  swapchain,
    timeout_ns: ACQUIRE_TIMEOUT_NS,
);
if (catch err = acquired) {
    if (err == gpu::WAIT_TIMEOUT) continue;
    if (err == gpu::SWAPCHAIN_OUT_OF_DATE) {
        swapchain_info = recover_swapchain(&device, swapchain, window, last_graphics)!;
        continue;
    }
    return err~;
}
```

`WAIT_TIMEOUT` means no image was free within the budget. Skip the frame and
pump events. `SWAPCHAIN_OUT_OF_DATE` means the window changed.

### Frame: record

Transition the image from whatever state it was in to color attachment,
render, then transition it to present:

```c3
gpu::CommandList commands = gpu::begin_commands(&allocator)!;
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

gpu::ColorTargetDesc[1] colors = {{
    .view     = acquired.attachment_view,
    .load_op  = gpu::LoadOp.CLEAR,
    .store_op = gpu::StoreOp.STORE,
    .clear    = { .rgba = { 0.04f, 0.05f, 0.10f, 1.0f } },
}};
gpu::RenderPassDesc pass = {
    .colors = colors[..],
    .width  = swapchain_info.width,
    .height = swapchain_info.height,
};
gpu::GraphicsState state = gpu::render_geometry_state(pass.width, pass.height)!;
gpu::ColorTargetState[1] color_state = { gpu::color_blend_disabled() };
state.color.targets = color_state[..];

gpu::cmd_begin_render_pass(&commands, &pass)!;
gpu::cmd_bind_pipeline(&commands, pipeline)!;
gpu::cmd_set_graphics_state(&commands, &state)!;
gpu::cmd_draw(
    commands:       &commands,
    vertex_root:    (gpu::GpuAddress)0,
    fragment_root:  (gpu::GpuAddress)0,
    vertex_count:   3,
    instance_count: 1,
)!;
gpu::cmd_end_render_pass(&commands)!;

gpu::TextureBarrier to_present = gpu::texture_transition(
    texture: acquired.texture,
    before:  to_attachment.after,
    after:   { .layout = gpu::TextureLayout.PRESENT },
)!;
gpu::cmd_texture_barrier(&commands, &to_present)!;
```

`render_geometry_state` gives a full-area viewport and scissor with no
culling and no depth. The color packet must match the pipeline's color
formats, one entry each. A zero root address is legal; these shaders never
dereference it.

### Frame: submit and present

The acquired image's `readiness` goes into the first submit that writes the
image. `present` takes the completion point of that submit.

```c3
gpu::ExecutableCommandList[1] executable = { gpu::end_commands(&commands)! };
defer (void)gpu::discard_executable_commands(&executable[0]);
gpu::SubmitDesc submit_desc = {
    .command_lists    = executable[..],
    .readiness        = acquired.readiness,
    .readiness_before = { .color_output },
};
last_graphics = gpu::submit(graphics_queue, &submit_desc)!;

if (catch err = gpu::present(&device, &acquired, last_graphics)) {
    if (err == gpu::SWAPCHAIN_OUT_OF_DATE) {
        swapchain_info = recover_swapchain(&device, swapchain, window, last_graphics)!;
    } else if (err != gpu::WAIT_TIMEOUT) {
        return err~;
    }
}
```

`WAIT_TIMEOUT` from `present` leaves the acquired image intact. Pump events
and call `present` again with the same image and point.

### Resize and teardown

Both follow one order: wait for the last completion point, wait for pending
presentations, then resize or destroy. Neither `resize_swapchain` nor
`destroy_swapchain` waits by itself.

```c3
fn gpu::SwapchainInfo? recover_swapchain(
    gpu::Device* device,
    gpu::SwapchainHandle swapchain,
    sdl::Window* window,
    gpu::CompletionPoint last_graphics,
) {
    if (last_graphics.is_valid()) {
        gpu::wait_completion(last_graphics, gpu::TIMEOUT_INFINITE)!;
    }
    while (catch err = gpu::wait_swapchain_presentations(device, swapchain, RETIRE_TIMEOUT_NS)) {
        if (err != gpu::WAIT_TIMEOUT) return err~;
        sdl::pump_events();
    }
    uint width;
    uint height;
    get_pixel_size(window, &width, &height);
    gpu::resize_swapchain(device, swapchain, width, height)!;
    return gpu::get_swapchain_info(device, swapchain);
}
```

`WAIT_TIMEOUT` from `wait_swapchain_presentations` leaves everything intact,
so the loop just pumps events and retries. A dormant swapchain (minimized
window) reports `swapchain_info.dormant`; sleep and skip frames until it
comes back.

## Next

- [Architecture](architecture.md): the ownership, memory, and sync model.
- [Shader ABI](shader_abi.md): root structs, buffer references, heap
  indices, and the schema generator.
- [Cookbook](cookbook.md): uploads, readback, textures, indirect draws,
  multiple queues, resize.
- [API reference](api/index.md): every public symbol by domain.
