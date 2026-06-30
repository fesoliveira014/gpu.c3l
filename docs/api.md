# gpu.c3l Public API

This document defines the intended public API shape. Names are architectural targets and may be adjusted during implementation, but the module boundary and design principles should remain stable.

## 1. Public module

All public API lives in:

```c3
module gpu;
```

Backend and dependency modules are not re-exported as public handle types.

Application code should import:

```c3
import gpu;
```

Windowed samples may also import:

```c3
import sdl;
```

but core `gpu` declarations should not mention `sdl::Window`, `vk::Device`, or `vma::Allocation`.

## 2. Naming rules

Use:

```text
create_device
create_buffer
destroy_buffer
begin_commands
cmd_dispatch
cmd_barrier
```

Do not use project-owned OO-style constructors such as:

```text
Device.create
Buffer.create
Texture.destroy
```

Methods are acceptable only for operations that clearly operate on an existing `&self` receiver and where the style guide allows them. Public lifecycle should prefer free functions.

## 3. Type taxonomy

### Backend and device

```text
BackendKind
    VULKAN

DeviceDesc
    BackendKind backend
    bool enable_validation
    bool enable_debug_names
    bool require_descriptor_buffer
    bool allow_descriptor_indexing_fallback
    uint frames_in_flight
    ZString application_name

DeviceCaps
    bool buffer_device_address
    bool synchronization2
    bool dynamic_rendering
    bool timeline_semaphore
    bool descriptor_buffer
    bool descriptor_indexing
    uint max_texture_descriptors
    uint max_sampler_descriptors
    usz min_uniform_alignment
    usz min_storage_alignment
    usz min_texel_buffer_alignment

Device
    BackendKind backend
    DeviceCaps caps
    void* backend_state
```

Creation:

```text
create_device(DeviceDesc* desc) -> Device?
destroy_device(Device* device) -> void?
```

### Handles

All handles are strongly typed aliases or small wrappers around `ulong`.

```text
BufferHandle
TextureHandle
PipelineHandle
SamplerHandle
SemaphoreHandle
SwapchainHandle
```

Invalid handle value:

```text
*_Handle.INVALID = 0
```

A valid handle packs slot index and generation. Public code should not inspect the packed representation.

### GPU addresses

```text
GpuAddress
    ulong value

GpuSpan
    GpuAddress gpu
    void* cpu
    usz size
    BufferHandle buffer
    usz offset
    MemoryKind kind
```

Rules:

```text
GpuAddress zero is invalid for shader-visible memory.
GpuSpan.cpu may be null.
GpuSpan.gpu must be aligned for the intended shader layout.
GpuSpan.buffer identifies the backing buffer for barriers/copies/debug.
```

## 4. Faults

Public operations use C3 optionals/faults. `faultdef` declares a flat list of globally-unique fault values (there is no braced/named fault group in C3 0.8.0); these live in `module gpu` and are referenced as `gpu::INVALID_HANDLE`, raised with the `~` suffix:

```c3
faultdef
    UNSUPPORTED_BACKEND,
    UNSUPPORTED_FEATURE,
    INVALID_ARGUMENT,
    INVALID_HANDLE,
    INVALID_RESOURCE_STATE,
    OUT_OF_HOST_MEMORY,
    OUT_OF_DEVICE_MEMORY,
    DEVICE_LOST,
    RESOURCE_IN_USE,
    ARENA_FULL,
    DESCRIPTOR_HEAP_FULL,
    PIPELINE_CREATE_FAILED,
    SHADER_INVALID,
    SURFACE_LOST,
    SWAPCHAIN_OUT_OF_DATE,
    COMMAND_RECORDING_ERROR;
```

Backend-local Vulkan/VMA faults should not leak unless they carry useful public meaning. Map them to public faults and log backend details when validation/debug is enabled.

## 5. Memory API

### Memory kinds

```text
MemoryKind.FRAME_UPLOAD
MemoryKind.PERSISTENT_UPLOAD
MemoryKind.DEVICE
MemoryKind.READBACK
MemoryKind.STAGING
```

### Frame spans

Frame spans are transient and invalid after the frame arena resets.

```text
alloc_frame_span(Device* device, usz size, usz align) -> GpuSpan?
```

Use cases:

```text
root structs
per-dispatch data
per-draw data
small per-frame tables
```

### Persistent spans

Persistent spans are suballocations from large VMA-backed buffers.

```text
PersistentAllocDesc
    usz size
    usz align
    BufferUsage usage
    MemoryKind memory_kind
    ZString debug_name

alloc_persistent_span(Device* device, PersistentAllocDesc* desc) -> GpuSpan?
free_persistent_span(Device* device, GpuSpan span) -> void?
```

### Explicit buffers

`BufferUsage` is a bitstruct of bool flags, composed by field-set
(`{ .storage, .addressable }`), not OR-combined enum values.

```text
bitstruct BufferUsage : uint
    bool transfer_src : 0
    bool transfer_dst : 1
    bool uniform      : 2
    bool storage      : 3
    bool addressable  : 4
    bool indirect     : 5
    bool index        : 6
    bool vertex       : 7

BufferDesc
    usz size
    BufferUsage usage
    MemoryKind memory_kind
    ZString debug_name

create_buffer(Device* device, BufferDesc* desc) -> BufferHandle?
destroy_buffer(Device* device, BufferHandle buffer) -> void?
get_buffer_span(Device* device, BufferHandle buffer) -> GpuSpan?
get_buffer_address(Device* device, BufferHandle buffer) -> GpuAddress?
map_buffer(Device* device, BufferHandle buffer) -> char[]?
unmap_buffer(Device* device, BufferHandle buffer) -> void?
flush_buffer(Device* device, BufferHandle buffer, usz offset, usz size) -> void?
invalidate_buffer(Device* device, BufferHandle buffer, usz offset, usz size) -> void?
```

`get_buffer_address` faults if the buffer was not created with the `addressable` usage flag set.

## 6. Texture API

### Formats

The public `Format` enum should contain only formats supported by the library:

```text
UNDEFINED
R8_UNORM
R8_UINT
RG8_UNORM
RGBA8_UNORM
RGBA8_SRGB
R16_UINT
R16_FLOAT
RG16_FLOAT
RGBA16_FLOAT
R32_UINT
R32_FLOAT
RG32_FLOAT
RGBA32_FLOAT
D32_FLOAT
D24_UNORM_S8_UINT
```

### Texture descriptors

`TextureUsage` is likewise a bitstruct of bool flags.

```text
bitstruct TextureUsage : uint
    bool sampled      : 0
    bool storage      : 1
    bool color_attach : 2
    bool depth_attach : 3
    bool transfer_src : 4
    bool transfer_dst : 5

TextureDimension
    TEX_1D
    TEX_2D
    TEX_3D
    CUBE

TextureDesc
    TextureDimension dimension
    uint width
    uint height
    uint depth
    uint mip_levels
    uint array_layers
    Format format
    TextureUsage usage
    ZString debug_name

TextureViewDesc
    Format format
    uint base_mip
    uint mip_count
    uint base_layer
    uint layer_count
```

### Texture functions

```text
create_texture(Device* device, TextureDesc* desc) -> TextureHandle?
destroy_texture(Device* device, TextureHandle texture) -> void?
create_texture_descriptor(Device* device, TextureHandle texture, TextureViewDesc* view) -> TextureIndex?
destroy_texture_descriptor(Device* device, TextureIndex index) -> void?
```

`TextureHandle` owns the image. `TextureIndex` is a descriptor heap entry used by shaders.

## 7. Sampler API

```text
Filter
    NEAREST
    LINEAR

AddressMode
    REPEAT
    MIRRORED_REPEAT
    CLAMP_TO_EDGE
    CLAMP_TO_BORDER

SamplerDesc
    Filter min_filter
    Filter mag_filter
    Filter mip_filter
    AddressMode address_u
    AddressMode address_v
    AddressMode address_w
    float mip_lod_bias
    float min_lod
    float max_lod
    bool anisotropy_enable
    float max_anisotropy
    ZString debug_name

create_sampler(Device* device, SamplerDesc* desc) -> SamplerIndex?
destroy_sampler(Device* device, SamplerIndex sampler) -> void?
```

Samplers are shader-visible indices. The backend may store immutable sampler descriptors or a sampler heap.

## 8. Shader and pipeline API

### Shader modules

```text
ShaderStage
    COMPUTE
    VERTEX
    FRAGMENT

ShaderDesc
    ShaderStage stage
    char[] spirv
    ZString entry_point
    ZString debug_name

create_shader(Device* device, ShaderDesc* desc) -> ShaderHandle?
destroy_shader(Device* device, ShaderHandle shader) -> void?
```

Shader compilation can be handled by tools or samples. The core library consumes SPIR-V bytes.

### Compute pipelines

```text
ComputePipelineDesc
    ShaderHandle shader
    uint push_constant_size
    ZString debug_name

create_compute_pipeline(Device* device, ComputePipelineDesc* desc) -> PipelineHandle?
```

The first ABI requires at least an 8-byte push constant for the root pointer.

### Graphics pipelines

```text
PrimitiveTopology
    TRIANGLES
    LINES
    POINTS

DepthState
    bool test_enable
    bool write_enable
    CompareOp compare

RasterState
    CullMode cull_mode
    FrontFace front_face
    PolygonMode polygon_mode

BlendState
    bool enable
    BlendFactor src_color
    BlendFactor dst_color
    BlendOp color_op
    BlendFactor src_alpha
    BlendFactor dst_alpha
    BlendOp alpha_op

GraphicsPipelineDesc
    ShaderHandle vertex_shader
    ShaderHandle fragment_shader
    PrimitiveTopology topology
    DepthState depth
    RasterState raster
    BlendState blend
    Format[] color_formats
    Format depth_format
    ZString debug_name

create_graphics_pipeline(Device* device, GraphicsPipelineDesc* desc) -> PipelineHandle?
destroy_pipeline(Device* device, PipelineHandle pipeline) -> void?
```

## 9. Command API

### Command lifecycle

```text
CommandList begin_commands(Device* device, QueueKind queue) -> CommandList?
end_commands(Device* device, CommandList* commands) -> void?
submit(Device* device, SubmitDesc* desc) -> void?
wait_queue_idle(Device* device, QueueKind queue) -> void?
```

### Dispatch

```text
cmd_dispatch(
    CommandList* commands,
    PipelineHandle pipeline,
    GpuAddress root,
    Vec3u groups,
) -> void?
```

### Render pass

```text
ColorTargetDesc
    TextureHandle texture
    uint mip_level
    uint array_layer
    LoadOp load_op
    StoreOp store_op
    ClearColor clear

DepthTargetDesc
    TextureHandle texture
    LoadOp load_op
    StoreOp store_op
    ClearDepthStencil clear

RenderPassDesc
    ColorTargetDesc[] colors
    DepthTargetDesc* depth
    uint width
    uint height

cmd_begin_render_pass(CommandList* commands, RenderPassDesc* desc) -> void?
cmd_end_render_pass(CommandList* commands) -> void?
```

### Draw

```text
cmd_draw(
    CommandList* commands,
    PipelineHandle pipeline,
    GpuAddress vertex_root,
    GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) -> void?

cmd_draw_indexed(
    CommandList* commands,
    PipelineHandle pipeline,
    GpuAddress vertex_root,
    GpuAddress fragment_root,
    GpuSpan index_span,
    uint index_count,
    uint instance_count,
) -> void?
```

### Transfer

```text
cmd_copy_buffer(CommandList* commands, BufferCopyDesc* desc) -> void?
cmd_copy_buffer_to_texture(CommandList* commands, BufferTextureCopyDesc* desc) -> void?
cmd_copy_texture_to_buffer(CommandList* commands, TextureBufferCopyDesc* desc) -> void?
cmd_fill_buffer(CommandList* commands, BufferHandle buffer, usz offset, usz size, uint value) -> void?
```

### Barriers

```text
Stage
    HOST
    TRANSFER
    COMPUTE_SHADER
    VERTEX_SHADER
    FRAGMENT_SHADER
    COLOR_ATTACHMENT
    DEPTH_STENCIL
    INDIRECT_COMMAND
    PRESENT

Hazard
    HOST_WRITE
    TRANSFER_READ
    TRANSFER_WRITE
    SHADER_READ
    SHADER_WRITE
    COLOR_READ
    COLOR_WRITE
    DEPTH_READ
    DEPTH_WRITE
    INDIRECT_READ
    PRESENT_READ

BufferBarrier
    BufferHandle buffer
    usz offset
    usz size
    Stage before_stage
    Stage after_stage
    Hazard before_hazard
    Hazard after_hazard

TextureBarrier
    TextureHandle texture
    Stage before_stage
    Stage after_stage
    Hazard before_hazard
    Hazard after_hazard
    TextureLayout old_layout
    TextureLayout new_layout

cmd_buffer_barrier(CommandList* commands, BufferBarrier* barrier) -> void?
cmd_texture_barrier(CommandList* commands, TextureBarrier* barrier) -> void?
```

No command helper should silently insert barriers for a later use.

## 10. Swapchain API

Core WSI types should be platform-neutral.

```text
SurfaceDesc
    PlatformKind platform
    void* native_display
    void* native_window

SwapchainDesc
    uint width
    uint height
    Format preferred_format
    PresentMode present_mode
    uint image_count
    bool srgb

create_swapchain(Device* device, SurfaceDesc* surface, SwapchainDesc* desc) -> SwapchainHandle?
destroy_swapchain(Device* device, SwapchainHandle swapchain) -> void?
acquire_next_image(Device* device, SwapchainHandle swapchain) -> AcquiredImage?
present(Device* device, PresentDesc* desc) -> void?
```

SDL helper functions should live in samples or an optional helper module, not in the core API.

## 11. Example: root-pointer compute

Pseudo-code:

```c3
import gpu;

struct RootArgs {
    gpu::GpuAddress input;
    gpu::GpuAddress output;
    uint count;
    uint _pad0, _pad1, _pad2;
}

fn void? run_compute() {
    gpu::DeviceDesc device_desc = {
        .backend = gpu::BackendKind.VULKAN,
        .enable_validation = true,
        .enable_debug_names = true,
        .frames_in_flight = 2,
        .allow_descriptor_indexing_fallback = true,
        .application_name = "root_pointer_compute",
    };

    gpu::Device device = gpu::create_device(&device_desc)!;
    defer gpu::destroy_device(&device)!!;

    gpu::BufferDesc input_desc = {
        .size = 4096,
        .usage = { .storage, .addressable, .transfer_dst },
        .memory_kind = gpu::MemoryKind.DEVICE,
        .debug_name = "input",
    };

    gpu::BufferHandle input = gpu::create_buffer(&device, &input_desc)!;
    defer gpu::destroy_buffer(&device, input)!!;

    gpu::GpuSpan root_span = gpu::alloc_frame_span(&device, RootArgs::size, RootArgs::alignment)!;
    RootArgs* root = (RootArgs*)root_span.cpu;
    root.input = gpu::get_buffer_address(&device, input)!;
    root.count = 1024;

    gpu::CommandList commands = gpu::begin_commands(&device, gpu::QueueKind.COMPUTE)!;
    gpu::cmd_dispatch(&commands, pipeline, root_span.gpu, { 16, 1, 1 })!;
    gpu::end_commands(&device, &commands)!;
}
```

Exact C3 syntax should be verified during implementation against C3 0.8.0.

## 12. API acceptance criteria

The public API is acceptable when:

```text
no public signature exposes vk::, vma::, or sdl:: types
all fallible operations return optionals/faults
all resources have explicit destruction or frame ownership
root-pointer compute can be written without descriptor-set concepts
texture sampling can be written with TextureIndex and SamplerIndex
barriers are explicit and expressive enough for all samples
headless tests do not depend on SDL3
windowed samples depend on sdl3 only in sample project files
```
