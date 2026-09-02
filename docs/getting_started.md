# Getting started

This guide has two steps. First, add `gpu.c3l` to a C3 application and run a
minimal headless compute program. Then add SDL3 and render a triangle to a
window.

The examples target **C3 0.8.3**, `linux-x64` or `windows-x64`, and Vulkan 1.3.

## Prerequisites

Install:

- C3 0.8.3 (`c3c --version`);
- a Vulkan 1.3 loader and qualifying driver;
- `glslangValidator` or `glslc`; and
- Git with submodule support.

The vendored `vma.c3l` package must contain a static VMA library under
`linked-libs/<target>/`. The repository includes the supported Linux artifact;
Windows consumers use the matching release-CRT artifact. Keep `"wincrt":
"dynamic"` when using that Windows library.

## Step 1: minimal compute application

### Vendor the library

Use this application layout:

```text
hello_gpu/
├── lib/
│   └── gpu.c3l/             release bundle or recursive clone
├── shaders/
│   └── doubler.comp.glsl
├── src/
│   └── main.c3
└── project.json
```

Download the archive for your target from the
[latest release](https://github.com/fesoliveira014/gpu.c3l/releases/latest) and
extract its `gpu.c3l/` directory under `hello_gpu/lib/`. To track Git instead:

```sh
mkdir -p hello_gpu/lib
git clone --recurse-submodules \
  https://github.com/fesoliveira014/gpu.c3l.git hello_gpu/lib/gpu.c3l
```

The release bundle and recursive clone both provide only the `gpu`, `vk`,
`vma`, and `spvreflect` packages required by the library.

`project.json` resolves the library plus its binding submodules:

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

### Compile the shader

The minimal compute shader receives one root pointer, follows two
`GpuAddress` fields, and doubles an array:

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

Compile it beside the source expected by `$embed`:

```sh
glslangValidator -V --target-env vulkan1.3 \
  shaders/doubler.comp.glsl -o shaders/doubler.comp.spv
```

### Record and submit

The complete maintained program is
[`examples/getting_started/src/main.c3`](https://github.com/fesoliveira014/gpu.c3l/blob/main/examples/getting_started/src/main.c3).
Copy it as the application's `src/main.c3`; it embeds
`../shaders/doubler.comp.spv`.
Its essential flow is:

1. create a runtime, enumerate an adapter, and create a headless device;
2. allocate mapped input, output, and root storage;
3. write the root's input/output addresses and flush host writes;
4. create a compute pipeline from embedded SPIR-V;
5. create a compute-queue command allocator;
6. bind, dispatch, and barrier the output for host access;
7. submit, wait, invalidate, and verify; and
8. destroy owners in dependency order.

Build and run the application:

```sh
cp lib/gpu.c3l/examples/getting_started/src/main.c3 src/main.c3
c3c run hello_gpu --path .
```

Expected output:

```text
hello_gpu: all 256 values doubled on the GPU
```

This is the core model: command completion orders reuse, while the application
retains allocations referenced by raw GPU addresses.

The optional ray-tracing pipeline path uses the same ownership model with an
explicit acceleration-structure opt-in, caller-built BLAS/TLAS values, and a
caller-packed shader binding table. Follow the
[SBT/direct-trace recipe](cookbook.md#pack-an-sbt-and-trace-directly) and the
[GPU-written ray-work recipe](cookbook.md#trace-with-gpu-written-ray-work)
after the compute introduction; the companion samples repository contains the
complete interactive Cornell Box application.

## Step 2: SDL3 hello triangle

Add `sdl3.c3l` beside the GPU library:

```text
hello_triangle/
├── lib/
│   ├── gpu.c3l/
│   └── sdl3.c3l/
├── shaders/
│   ├── triangle.vert.glsl
│   └── triangle.frag.glsl
├── src/
│   └── main.c3
└── project.json
```

Use the same project configuration as step one, add `"sdl3"` to the target's
dependencies, and clone the binding:

```sh
git clone https://github.com/fesoliveira014/sdl3.c3l.git lib/sdl3.c3l
```

```json
{
  "langrev": "1",
  "dependency-search-paths": [ "lib", "lib/gpu.c3l/lib" ],
  "dependencies": [ "gpu", "vk", "vma", "spvreflect" ],
  "output": "build",
  "wincrt": "dynamic",
  "targets": {
    "hello_triangle": {
      "type": "executable",
      "dependencies": [ "sdl3" ],
      "sources": [ "src/main.c3" ]
    }
  }
}
```

The vertex shader generates positions from `gl_VertexIndex`; neither shader
needs root data:

```glsl
#version 460

const vec2 POSITIONS[3] = vec2[](
    vec2(-0.8, -0.7),
    vec2( 0.8, -0.7),
    vec2( 0.0,  0.8)
);

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

Compile both assets:

```sh
glslangValidator -V --target-env vulkan1.3 \
  shaders/triangle.vert.glsl -o shaders/triangle.vert.spv
glslangValidator -V --target-env vulkan1.3 \
  shaders/triangle.frag.glsl -o shaders/triangle.frag.spv
```

The application is intentionally one file. The surface helper converts SDL3's
active video-driver properties to the distinct Wayland, X11, or Win32 public
handle types.

Resize and teardown follow one order: establish command completion with
`wait_completion`, retire pending presentations with
`wait_swapchain_presentations`, then call `resize_swapchain` or
`destroy_swapchain`. Neither lifecycle call waits, so the presentation wait is
what makes the swapchain quiescent. `retire_presentations` runs it with a
finite timeout: `WAIT_TIMEOUT` leaves the swapchain handle and its pending
presentations intact, so the attempt simply repeats, and pumping the platform
queue between attempts keeps presentation progressing while the surface and
window stay alive.

```c3
module hello_triangle;

import gpu;
import gpu::surface::wayland;
import gpu::surface::win32;
import gpu::surface::x11;
import sdl;
import std::io;

faultdef WINDOW_INIT_FAILED, UNSUPPORTED_VIDEO_DRIVER;

const char[*] VERTEX_SPIRV = $embed("../shaders/triangle.vert.spv");
const char[*] FRAGMENT_SPIRV = $embed("../shaders/triangle.frag.spv");
const ulong ACQUIRE_TIMEOUT_NS = 2_000_000;
const ulong RETIRE_TIMEOUT_NS = 2_000_000;

struct WindowEvents {
    bool quit;
    bool resized;
}

fn int main() {
    if (catch err = run()) {
        io::printfn("hello_triangle: FAIL (%s)", err);
        return 1;
    }
    return 0;
}

fn void? run() {
    if (!sdl::init({ .video })) return WINDOW_INIT_FAILED~;
    defer sdl::quit();

    sdl::Window* window =
        sdl::create_window("gpu.c3l hello triangle", 800, 600, { .resizable });
    if (window == null) return WINDOW_INIT_FAILED~;
    defer sdl::destroy_window(window);

    gpu::RuntimeDesc runtime_desc = gpu::full_validation_runtime_desc();
    runtime_desc.application_name = "hello_triangle";
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
    defer (void)gpu::destroy_runtime(&runtime);

    gpu::Surface surface = create_sdl_surface(&runtime, window)!;
    defer (void)gpu::destroy_surface(&surface);

    gpu::Device device = create_presentation_device(&runtime, &surface)!;
    defer (void)gpu::destroy_device(&device);

    gpu::Queue graphics_queue =
        gpu::get_queue(&device, gpu::QueueKind.GRAPHICS)!;
    gpu::CommandAllocator allocator =
        gpu::create_command_allocator(&device, graphics_queue)!;
    defer (void)gpu::destroy_command_allocator(&allocator);

    uint width;
    uint height;
    get_pixel_size(window, &width, &height);
    gpu::SwapchainDesc swapchain_desc = {
        .width = width,
        .height = height,
        .preferred_format = gpu::Format.BGRA8_UNORM,
        .present_mode = gpu::PresentMode.FIFO,
        .debug_name = "hello_triangle_swapchain",
    };
    gpu::SwapchainHandle swapchain =
        gpu::create_swapchain(&device, &surface, &swapchain_desc)!;
    defer (void)destroy_swapchain_when_ready(&device, swapchain);
    gpu::SwapchainInfo swapchain_info =
        gpu::get_swapchain_info(&device, swapchain)!;

    gpu::Format[1] color_formats = { swapchain_info.format };
    gpu::GraphicsPipelineDesc pipeline_desc = {
        .vertex_shader = {
            .spirv = VERTEX_SPIRV[..],
            .entry_point = "main",
        },
        .fragment_shader = {
            .spirv = FRAGMENT_SPIRV[..],
            .entry_point = "main",
        },
        .color_formats = color_formats[..],
        .debug_name = "hello_triangle",
    };
    gpu::PipelineHandle pipeline;
    bool pipeline_live;
    defer {
        if (pipeline_live) (void)gpu::destroy_pipeline(&device, pipeline);
    }

    gpu::CompletionPoint last_graphics;
    bool running = true;
    while (running) {
        WindowEvents events = poll_window_events();
        if (events.quit) break;
        if (events.resized) {
            swapchain_info = recover_swapchain(
                device: &device,
                swapchain: swapchain,
                window: window,
                last_graphics: last_graphics,
            )!;
        }
        if (swapchain_info.dormant) {
            sdl::delay(16);
            continue;
        }
        if (!pipeline_live || color_formats[0] != swapchain_info.format) {
            if (last_graphics.is_valid()) {
                gpu::wait_completion(last_graphics, gpu::TIMEOUT_INFINITE)!;
            }
            if (pipeline_live) gpu::destroy_pipeline(&device, pipeline)!;
            pipeline_live = false;
            color_formats[0] = swapchain_info.format;
            pipeline = gpu::create_graphics_pipeline(&device, &pipeline_desc)!;
            pipeline_live = true;
        }

        gpu::AcquiredImage? acquired = gpu::acquire_next_image(
            device: &device,
            swapchain: swapchain,
            timeout_ns: ACQUIRE_TIMEOUT_NS,
        );
        if (catch err = acquired) {
            if (err == gpu::WAIT_TIMEOUT) continue;
            if (err == gpu::SWAPCHAIN_OUT_OF_DATE) {
                swapchain_info = recover_swapchain(
                    device: &device,
                    swapchain: swapchain,
                    window: window,
                    last_graphics: last_graphics,
                )!;
                continue;
            }
            return err~;
        }

        gpu::CommandList commands = gpu::begin_commands(&allocator)!;
        defer (void)gpu::discard_commands(&commands);

        gpu::TextureBarrier to_attachment = gpu::texture_transition(
            texture: acquired.texture,
            before: acquired.prior_state,
            after: {
                .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
                .stages = { .color_output },
                .access = { .read, .write },
            },
        )!;
        gpu::cmd_texture_barrier(&commands, &to_attachment)!;

        gpu::ColorTargetDesc[1] colors = {{
            .view = acquired.attachment_view,
            .load_op = gpu::LoadOp.CLEAR,
            .store_op = gpu::StoreOp.STORE,
            .clear = { .rgba = { 0.04f, 0.05f, 0.10f, 1.0f } },
        }};
        gpu::RenderPassDesc pass = {
            .colors = colors[..],
            .width = swapchain_info.width,
            .height = swapchain_info.height,
        };
        gpu::GraphicsState state =
            gpu::render_geometry_state(pass.width, pass.height)!;
        gpu::ColorTargetState[1] color_state = {
            gpu::color_blend_disabled(),
        };
        state.color.targets = color_state[..];

        gpu::cmd_begin_render_pass(&commands, &pass)!;
        gpu::cmd_bind_pipeline(&commands, pipeline)!;
        gpu::cmd_set_graphics_state(&commands, &state)!;
        gpu::cmd_draw(
            commands: &commands,
            vertex_root: (gpu::GpuAddress)0,
            fragment_root: (gpu::GpuAddress)0,
            vertex_count: 3,
            instance_count: 1,
        )!;
        gpu::cmd_end_render_pass(&commands)!;

        gpu::TextureBarrier to_present = gpu::texture_transition(
            texture: acquired.texture,
            before: {
                .layout = gpu::TextureLayout.COLOR_ATTACHMENT,
                .stages = { .color_output },
                .access = { .read, .write },
            },
            after: { .layout = gpu::TextureLayout.PRESENT },
        )!;
        gpu::cmd_texture_barrier(&commands, &to_present)!;

        gpu::ExecutableCommandList[1] executable = {
            gpu::end_commands(&commands)!,
        };
        defer (void)gpu::discard_executable_commands(&executable[0]);
        gpu::SubmitDesc submit_desc = {
            .command_lists = executable[..],
            .readiness = acquired.readiness,
            .readiness_before = { .color_output },
        };
        last_graphics = gpu::submit(graphics_queue, &submit_desc)!;

        if (catch err = present_when_ready(
            &device,
            &acquired,
            last_graphics,
        )) {
            if (err != gpu::SWAPCHAIN_OUT_OF_DATE) return err~;
            swapchain_info = recover_swapchain(
                device: &device,
                swapchain: swapchain,
                window: window,
                last_graphics: last_graphics,
            )!;
        }
    }

    if (last_graphics.is_valid()) {
        gpu::wait_completion(last_graphics, gpu::TIMEOUT_INFINITE)!;
    }
}

fn gpu::Surface? create_sdl_surface(
    gpu::Runtime* runtime,
    sdl::Window* window,
) {
    sdl::PropertiesID props = sdl::get_window_properties(window);
    ZString driver = (ZString)sdl::get_current_video_driver();

    if (driver.str_view() == "wayland") {
        return gpu::surface::wayland::create_surface(
            runtime,
            (gpu::surface::wayland::DisplayHandle)sdl::get_pointer_property(
                props,
                (char*)sdl::WindowProperties.WAYLAND_DISPLAY_POINTER,
                null,
            ),
            (gpu::surface::wayland::SurfaceHandle)sdl::get_pointer_property(
                props,
                (char*)sdl::WindowProperties.WAYLAND_SURFACE_POINTER,
                null,
            ),
        );
    }
    if (driver.str_view() == "x11") {
        return gpu::surface::x11::create_surface(
            runtime,
            (gpu::surface::x11::DisplayHandle)sdl::get_pointer_property(
                props,
                (char*)sdl::WindowProperties.X11_DISPLAY_POINTER,
                null,
            ),
            (gpu::surface::x11::WindowHandle)sdl::get_number_property(
                props,
                (char*)sdl::WindowProperties.X11_WINDOW_NUMBER,
                0,
            ),
        );
    }
    if (driver.str_view() == "windows") {
        return gpu::surface::win32::create_surface(
            runtime,
            (gpu::surface::win32::InstanceHandle)sdl::get_pointer_property(
                props,
                (char*)sdl::WindowProperties.WIN32_INSTANCE_POINTER,
                null,
            ),
            (gpu::surface::win32::WindowHandle)sdl::get_pointer_property(
                props,
                (char*)sdl::WindowProperties.WIN32_HWND_POINTER,
                null,
            ),
        );
    }
    return UNSUPPORTED_VIDEO_DRIVER~;
}

fn gpu::Device? create_presentation_device(
    gpu::Runtime* runtime,
    gpu::Surface* surface,
) {
    gpu::DeviceDesc desc = { .surface = *surface };
    gpu::AdapterList adapters = gpu::enumerate_adapters(runtime)!;
    for (uint i = 0; i < adapters.count; i++) {
        gpu::Adapter adapter = adapters.get(i)!;
        if (!gpu::supports_device_desc(&adapter, &desc)!.supported) continue;
        return gpu::create_device(&adapter, &desc);
    }
    return gpu::UNSUPPORTED_FEATURE~;
}

fn void get_pixel_size(
    sdl::Window* window,
    uint* width,
    uint* height,
) {
    int w;
    int h;
    sdl::get_window_size_in_pixels(window, &w, &h);
    *width = w > 0 ? (uint)w : 0;
    *height = h > 0 ? (uint)h : 0;
}

fn void? retire_presentations(
    gpu::Device* device,
    gpu::SwapchainHandle swapchain,
) {
    while (catch err = gpu::wait_swapchain_presentations(
        device,
        swapchain,
        RETIRE_TIMEOUT_NS,
    )) {
        if (err != gpu::WAIT_TIMEOUT) return err~;
        sdl::pump_events();
    }
}

fn gpu::SwapchainInfo? recover_swapchain(
    gpu::Device* device,
    gpu::SwapchainHandle swapchain,
    sdl::Window* window,
    gpu::CompletionPoint last_graphics,
) {
    if (last_graphics.is_valid()) {
        gpu::wait_completion(last_graphics, gpu::TIMEOUT_INFINITE)!;
    }
    retire_presentations(device, swapchain)!;

    uint width;
    uint height;
    get_pixel_size(window, &width, &height);
    gpu::resize_swapchain(
        device: device,
        swapchain: swapchain,
        width: width,
        height: height,
    )!;
    return gpu::get_swapchain_info(device, swapchain);
}

fn void? present_when_ready(
    gpu::Device* device,
    gpu::AcquiredImage* image,
    gpu::CompletionPoint completion,
) {
    while (catch err = gpu::present(device, image, completion)) {
        if (err != gpu::WAIT_TIMEOUT) return err~;
        sdl::delay(1);
    }
}

fn void? destroy_swapchain_when_ready(
    gpu::Device* device,
    gpu::SwapchainHandle swapchain,
) {
    retire_presentations(device, swapchain)!;
    return gpu::destroy_swapchain(device, swapchain);
}

fn WindowEvents poll_window_events() {
    WindowEvents events;
    sdl::Event event;
    while (sdl::poll_event(&event)) {
        switch (event.type) {
            case sdl::EventType.QUIT:
                events.quit = true;
            case sdl::EventType.WINDOW_RESIZED:
            case sdl::EventType.WINDOW_PIXEL_SIZE_CHANGED:
                events.resized = true;
            default:
                break;
        }
    }
    return events;
}
```

Build and run:

```sh
c3c run hello_triangle --path .
```

The maintained
[`hello_triangle_sdl`](https://github.com/fesoliveira014/gpu.c3l-samples/tree/main/hello_triangle_sdl)
sample expands this foundation with per-frame uploads, textures, screenshots,
argument handling, and self-test behavior.

## Next

- [Architecture](architecture.md) for the ownership and synchronization model.
- [Shader ABI](shader_abi.md) before designing application root data.
- [Cookbook](cookbook.md) for uploads, readback, indirect work, and resize
  patterns.
- [Public API](api/index.md) for exact domain contracts.
