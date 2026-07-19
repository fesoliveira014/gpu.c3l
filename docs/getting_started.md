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
inside it. Which data ranges a dispatch uses is just *data in a struct* — change
the fields, dispatch again. Nothing to allocate, update, or bind, and the
struct definition is shared between C3 and GLSL. The ABI generator emits both
forms from one schema.

Textures are the one thing GPUs still want tables for (samplers and image
descriptors are opaque hardware state, not addresses) — for those the library
manages a single global **bindless heap**. Creating a texture view returns an
owner-bearing `TextureView`; its raw 32-bit `index` field is the value stored in
shader data. Sampler descriptions are first interned as device-owned `Sampler`
identities; `publish_sampler` yields stable `SamplerIndex` values. Put those raw
indices in root structs like any other field. This program needs no textures.

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

The C3 side declares the same 24-byte root struct, stores it in a caller-owned
`CPU_WRITE` allocation, and hands its GPU address to `cmd_dispatch`. Errors are
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

fn int main() {
    if (catch err = run()) {
        io::printfn("hello_gpu: FAIL (%s)", err);
        return 1;
    }
    io::printn("hello_gpu: all 256 values doubled on the GPU");
    return 0;
}

fn void? run() {
    gpu::RuntimeDesc runtime_desc = {
        .backend           = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .application_name  = "hello_gpu",
    };
    gpu::Runtime runtime = gpu::create_runtime(&runtime_desc)!;
    defer (void)gpu::destroy_runtime(&runtime);
    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    gpu::Adapter adapter = adapters.get(0)!;
    gpu::DeviceRequest request = gpu::strict_device_request();
    gpu::DeviceRequestSupport support =
        gpu::supports_device_request(&adapter, &request)!;
    if (!support.supported) return gpu::UNSUPPORTED_FEATURE~;

    gpu::Device device = gpu::create_device(&adapter, &request)!;
    defer (void)gpu::destroy_device(&device);

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

    gpu::GpuAllocation input =
        gpu::allocate_memory(&device, &input_desc)!;
    defer (void)gpu::free_allocation(&device, &input);
    gpu::GpuAllocation output =
        gpu::allocate_memory(&device, &output_desc)!;
    defer (void)gpu::free_allocation(&device, &output);

    gpu::GpuSpan in_span = gpu::get_allocation_span(&device, input)!;
    gpu::GpuSpan out_span = gpu::get_allocation_span(&device, output)!;
    float* in_data = (float*)gpu::get_span_mapping(&device, in_span)!.ptr;
    for (uint i = 0; i < COUNT; i++) in_data[i] = (float)i;
    gpu::flush_mapped_span(&device, in_span)!;

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

    gpu::AllocationDesc root_desc = {
        .size         = DoublerRoot::size,
        .alignment    = DoublerRoot::alignment,
        .memory_class = gpu::MemoryClass.CPU_WRITE,
        .access       = { .compute },
        .debug_name   = "doubler_root",
    };
    gpu::GpuAllocation root_allocation =
        gpu::allocate_memory(&device, &root_desc)!;
    defer (void)gpu::free_allocation(&device, &root_allocation);
    gpu::GpuSpan root_span =
        gpu::get_allocation_span(&device, root_allocation)!;

    run_compute(
        device:      &device,
        pipeline:    pipeline,
        input_span:  in_span,
        output_span: out_span,
        root_span:   root_span,
    )!;
    gpu::invalidate_mapped_span(&device, out_span)!;
    float* out_data = (float*)gpu::get_span_mapping(&device, out_span)!.ptr;
    for (uint i = 0; i < COUNT; i++) {
        if (out_data[i] != (float)i * 2.0f) return gpu::INVALID_ARGUMENT~;
    }
}

fn void? run_compute(
    gpu::Device* device,
    gpu::PipelineHandle pipeline,
    gpu::GpuSpan input_span,
    gpu::GpuSpan output_span,
    gpu::GpuSpan root_span,
) {
    DoublerRoot* root =
        (DoublerRoot*)gpu::get_span_mapping(device, root_span)!.ptr;
    gpu::GpuAddress root_address =
        gpu::get_span_address(device, root_span)!;
    root.input_gpu  = gpu::get_span_address(device, input_span)!;
    root.output_gpu = gpu::get_span_address(device, output_span)!;
    root.count      = COUNT;
    gpu::flush_mapped_span(device, root_span)!;

    gpu::Queue queue = gpu::get_queue(device, gpu::QueueKind.COMPUTE)!;
    gpu::CommandList cmd = gpu::begin_commands(queue)!;
    defer (void)gpu::discard_commands(&cmd);
    gpu::cmd_dispatch(
        commands: &cmd,
        pipeline: pipeline,
        root:     root_address,
        groups:   { (COUNT + 63) / 64, 1, 1 },
    )!;
    gpu::BufferBarrier to_host = {
        .span          = output_span,
        .before_stage  = gpu::Stage.COMPUTE_SHADER,
        .after_stage   = gpu::Stage.HOST,
        .before_hazard = gpu::Hazard.SHADER_WRITE,
        .after_hazard  = gpu::Hazard.HOST_READ,
    };
    gpu::cmd_buffer_barrier(&cmd, &to_host)!;
    gpu::ExecutableCommandList executable = gpu::end_commands(&cmd)!;
    defer (void)gpu::discard_executable_commands(&executable);

    gpu::ExecutableCommandList[1] lists = { executable };
    gpu::SubmitDesc submit = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(queue, &submit)!;
    gpu::wait_completion(completion)!;
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
- **Textures, samplers, and the bindless heap** — `TextureView.index` and
  published `SamplerIndex` values in root structs, with one global descriptor set
  you never manage. Read `docs/api.md`.
- **The samples repository** —
  [gpu.c3l-samples](https://github.com/fesoliveira014/gpu.c3l-samples):
  eighteen runnable programs from a windowed triangle through GPU-driven
  rendering, deferred shading, PBR, and multithreaded command recording,
  each with a README and a screenshot where it has something to show.
- **The rest of the docs** — `docs/document_index.md` is the map.
