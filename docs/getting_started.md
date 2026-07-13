# Getting started

This walkthrough builds a minimal compute program on Linux or Windows. CI
compiles and runs the embedded project on lavapipe.

## 1. Toolchain

You need three things: the **C3 compiler** (0.8.0 — the version this library
is pinned to), a **Vulkan 1.3 loader + driver**, and **glslang** to compile
shaders.

### Linux

Install c3c 0.8.0 from the [C3 releases](https://github.com/c3lang/c3c/releases)
and put it on your PATH. Then:

```sh
sudo apt install -y mesa-vulkan-drivers vulkan-validationlayers glslang-tools
```

`mesa-vulkan-drivers` includes **lavapipe**, a CPU implementation of
Vulkan 1.3, and `vulkan-validationlayers` is required because the program
below turns validation on — the right default while learning an explicit
API — everything in this walkthrough (and the entire library test
suite) runs on it, so a machine with no GPU at all is fine. If you have a
real GPU with a Vulkan driver, nothing changes.

### Windows

- c3c: download the release zip, then fetch its MSVC SDK once:
  `echo Y | c3c.exe fetch-sdk windows` (answers the license prompt), and if
  c3c does not find the SDK afterwards, copy the downloaded `msvc_sdk`
  directory from `%LOCALAPPDATA%` to sit beside `c3c.exe`.
- Vulkan: any current GPU driver ships the runtime. Without a GPU, use
  [mesa-dist-win](https://github.com/pal1000/mesa-dist-win) (lavapipe) — note
  that **elevated shells ignore `VK_DRIVER_FILES`**; register the ICD under
  `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` instead (see
  `docs/platforms_and_dependencies.md`).
- glslang: ships with the [Vulkan SDK](https://vulkan.lunarg.com/), or grab a
  [standalone release](https://github.com/KhronosGroup/glslang/releases).

## 2. Project setup

Create a directory and vendor the library. `gpu.c3l` brings its own
backend bindings (`vk`, `vma`, `spvreflect`) as submodules, so clone
recursively:

```sh run
mkdir -p hello_gpu/lib hello_gpu/src hello_gpu/shaders
cd hello_gpu
git clone --quiet --recurse-submodules "${GPU_C3L_URL:-https://github.com/fesoliveira014/gpu.c3l}" lib/gpu.c3l
```

(The `GPU_C3L_URL` override exists for CI mirrors; you can paste the plain
`git clone --recurse-submodules https://github.com/fesoliveira014/gpu.c3l lib/gpu.c3l`.)

On Windows, build the VMA 3.3.0 static library in the cloned dependency before
building the program. Follow the `windows-x64 setup` section in
[`platforms_and_dependencies.md`](platforms_and_dependencies.md).

Wire it up. Two search paths — your `lib/` for `gpu`, and the library's
own `lib/` for the bindings it vendors:

```json file=hello_gpu/project.json
{
  "langrev": "1",
  "dependency-search-paths": [ "lib", "lib/gpu.c3l/lib" ],
  "dependencies": [ "gpu", "vk", "vma", "spvreflect" ],
  "output": "build",
  "targets": {
    "hello_gpu": {
      "type": "executable",
      "sources": [ "src/main.c3" ]
    }
  }
}
```

## 3. Compute shader and root pointers

If you have not written Vulkan before, here is the problem this library's
execution model removes. In classic Vulkan, a shader cannot just be handed a
buffer. Every resource access goes through **descriptor sets** — driver-owned
tables of resource references. You describe each table's shape up front (a
*descriptor set layout*: "binding 0 is a storage buffer, binding 1 is a
uniform buffer…"), bake that shape into every pipeline (a *pipeline layout*),
allocate tables from a *descriptor pool*, write your buffer handles into
slots with `vkUpdateDescriptorSets`, and bind the right tables before each
draw or dispatch. The shader side then declares matching
`layout(set = 1, binding = 3)` plumbing. That is five API concepts and two
places to keep in sync before the first byte of data reaches a shader — and
changing *which* buffer a dispatch uses means rewriting or re-binding tables.

The root-pointer model replaces all of it with something you already know:
**a pointer to a struct**.

```
classic Vulkan                          root pointer
──────────────                          ────────────
set layouts ─┐                          struct DoublerRoot {
pipeline layout ├─ describe shapes          input_gpu;   ← raw GPU address
descriptor pool ─┤                          output_gpu;  ← raw GPU address
vkUpdateDescriptorSets ─ fill slots         count;
vkCmdBindDescriptorSets ─ bind          }
layout(set=N, binding=M) in shader      push 1 address of that struct
```

Vulkan 1.3 lets a shader dereference raw 64-bit GPU addresses
(`buffer_reference` — *buffer device address* on the API side). So instead
of tables and slots: write your parameters into a plain struct in GPU-visible
memory, push the struct's 64-bit address as the only push constant, and let
the shader cast the address back to the struct type and follow the pointers
inside it. Which buffers a dispatch uses is just *data in a struct* — change
the fields, dispatch again. Nothing to allocate, update, or bind, and the
struct definition is shared between C3 and GLSL, so there is exactly one
shape to keep in sync (and the ABI generator, later, keeps it for you).

Textures are the one thing GPUs still want tables for (samplers and image
descriptors are opaque hardware state, not addresses) — for those the
library manages a single global **bindless heap** and hands you plain
integer indices (`TextureIndex`, `SamplerIndex`) that you put in your root
structs like any other field. You will meet it in the samples; this program
needs no textures.

The root struct below is this program's whole binding model:

```glsl file=hello_gpu/shaders/doubler.comp.glsl
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 64) in;

layout(buffer_reference, std430) buffer DoublerRoot {
    uint64_t input_gpu;
    uint64_t output_gpu;
    uint     count;
};
layout(buffer_reference, std430) readonly  buffer InBuf  { float v[]; };
layout(buffer_reference, std430) writeonly buffer OutBuf { float v[]; };

layout(push_constant) uniform Push { uint64_t root_gpu; };

void main() {
    DoublerRoot root = DoublerRoot(root_gpu);
    uint i = gl_GlobalInvocationID.x;
    if (i < root.count) {
        OutBuf(root.output_gpu).v[i] = InBuf(root.input_gpu).v[i] * 2.0;
    }
}
```

Compile it to SPIR-V:

```sh run
cd hello_gpu
glslangValidator --target-env vulkan1.3 -o shaders/doubler.comp.spv shaders/doubler.comp.glsl
```

## 4. The program

The C3 side declares the same 24-byte root struct, allocates it from the
per-frame arena, and hands its GPU address to `cmd_dispatch`. Errors are
C3 optionals throughout — `!` propagates, no error codes to check:

```c3 file=hello_gpu/src/main.c3
module hello_gpu;

import gpu;
import std::io;

const char[*] DOUBLER_SPIRV = $embed("../shaders/doubler.comp.spv");
const uint COUNT = 256;

struct DoublerRoot {
    gpu::GpuAddress input_gpu;
    gpu::GpuAddress output_gpu;
    uint            count;
}

struct FrameWork {
    gpu::Device*         device;
    gpu::BufferHandle    input;
    gpu::BufferHandle    output;
    gpu::PipelineHandle  pipeline;
}

fn int main() {
    if (catch err = run()) {
        io::printfn("hello_gpu: FAIL (%s)", err);
        return 1;
    }
    io::printn("hello_gpu: all 256 values doubled on the GPU");
    return 0;
}

fn void? run() {
    gpu::DeviceDesc device_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .frames_in_flight  = 2,
        .application_name  = "hello_gpu",
    };
    gpu::Device device = gpu::create_device(&device_desc)!;
    defer (void)gpu::destroy_device(&device);

    gpu::BufferDesc io_desc = {
        .size        = COUNT * float::size,
        .usage       = { .storage, .addressable },
        .memory_kind = gpu::MemoryKind.PERSISTENT_UPLOAD,
        .debug_name  = "io",
    };
    gpu::BufferHandle input = gpu::create_buffer(&device, &io_desc)!;
    defer (void)gpu::destroy_buffer(&device, input);
    gpu::BufferHandle output = gpu::create_buffer(&device, &io_desc)!;
    defer (void)gpu::destroy_buffer(&device, output);

    gpu::GpuSpan in_span = gpu::get_buffer_span(&device, input)!;
    float* in_data = (float*)in_span.cpu;
    for (uint i = 0; i < COUNT; i++) in_data[i] = (float)i;
    gpu::flush_buffer(&device, input, 0, 0)!;

    gpu::ShaderDesc shader_desc = {
        .stage       = gpu::ShaderStage.COMPUTE,
        .spirv       = DOUBLER_SPIRV[..],
        .entry_point = "main",
    };
    gpu::ShaderHandle shader = gpu::create_shader(&device, &shader_desc)!;
    defer (void)gpu::destroy_shader(&device, shader);

    gpu::ComputePipelineDesc pipe_desc = { .shader = shader, .push_constant_size = gpu::RootPush::size };
    gpu::PipelineHandle pipeline = gpu::create_compute_pipeline(&device, &pipe_desc)!;
    defer (void)gpu::destroy_pipeline(&device, pipeline);

    FrameWork frame_work = {
        .device   = &device,
        .input    = input,
        .output   = output,
        .pipeline = pipeline,
    };
    gpu::FrameToken frame;
    if (catch frame_err = gpu::@with_frame(&frame, &device, run_frame, &frame_work)) {
        if (frame.is_valid()) gpu::end_frame(&frame)!;
        return frame_err~;
    }

    gpu::GpuSpan out_span = gpu::get_buffer_span(&device, output)!;
    gpu::invalidate_buffer(&device, output, 0, 0)!;
    float* out_data = (float*)out_span.cpu;
    for (uint i = 0; i < COUNT; i++) {
        if (out_data[i] != (float)i * 2.0f) return gpu::INVALID_ARGUMENT~;
    }
}

fn void? run_frame(gpu::FrameToken* frame, FrameWork* work) {
    gpu::GpuSpan root_span = gpu::alloc_frame_span(frame, DoublerRoot::size, 16)!;
    DoublerRoot* root = (DoublerRoot*)root_span.cpu;
    root.input_gpu  = gpu::get_buffer_address(work.device, work.input)!;
    root.output_gpu = gpu::get_buffer_address(work.device, work.output)!;
    root.count      = COUNT;

    gpu::CommandList cmd = gpu::begin_commands(work.device, gpu::QueueKind.COMPUTE)!;
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: work.pipeline,
        root:     root_span.gpu,
        groups:   { (COUNT + 63) / 64, 1, 1 },
    )!;
    gpu::BufferBarrier to_host = {
        .buffer = work.output, .offset = 0, .size = COUNT * float::size,
        .before_stage = gpu::Stage.COMPUTE_SHADER, .after_stage = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE, .after_hazard = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &to_host)!;
    gpu::end_commands(&cmd)!;

    gpu::CommandList[1] lists = { cmd };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::submit(work.device, &submit)!;
    gpu::wait_queue_idle(work.device, gpu::QueueKind.COMPUTE)!;
}
```

## 5. Build and run

```sh run
cd hello_gpu
c3c build hello_gpu
./build/hello_gpu
```

Expected output:

```text
hello_gpu: all 256 values doubled on the GPU
```

Troubleshooting the two most likely faults:

- `UNSUPPORTED_BACKEND` — the loader found no driver. Point it at lavapipe
  explicitly:
  `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json ./build/hello_gpu`
- `UNSUPPORTED_FEATURE` — `enable_validation = true` but the validation
  layer is not installed (`vulkan-validationlayers` on apt; on Windows it
  ships with the Vulkan SDK). Install it, or set `enable_validation = false`.

## 6. Where to go next

The hand-written root struct above is fine for one shader — and exactly the
kind of thing that silently breaks when two languages each declare it. The
real workflow generates both sides from one schema:

- **Shader ABI generator** — write a `.abi` schema, get the C3 struct
  (with size/offset asserts) and the GLSL include from one source of truth,
  plus a `--check` drift gate for CI. Read `docs/shader_abi.md`.
- **Textures and the bindless heap** — `TextureIndex`/`SamplerIndex` in
  root structs, one global descriptor set you never manage. Read
  `docs/api.md`.
- **The samples repository** —
  [gpu.c3l-samples](https://github.com/fesoliveira014/gpu.c3l-samples):
  eighteen runnable programs from a windowed triangle through GPU-driven
  rendering, deferred shading, PBR, and multithreaded command recording,
  each with a README and a screenshot where it has something to show.
- **The rest of the docs** — `docs/document_index.md` is the map.
