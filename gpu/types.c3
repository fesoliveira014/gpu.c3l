module gpu;

// Shared-ABI vector aliases: plain SIMD vectors (element-aligned in structs;
// std430 vector alignment comes from explicit padding validated by the ABI
// generator, not from these types).
alias Vec2f = float[<2>];
// Same shared-ABI rules as Vec2f.
alias Vec4f = float[<4>];
// Same shared-ABI rules as Vec2f.
alias Vec4u = uint[<4>];
$assert Vec2f::size == 8;
$assert Vec4f::size == 16;
$assert Vec4u::size == 16;

// Device handles are compact registry tokens. Device-owned resources also
// carry an exact owner identity so coincident local slots cannot alias.

bitstruct Device : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

// Fixed-width shader-visible heap values. Value zero is the invalid sentinel;
// live values encode the zero-based backend slot plus one and carry no CPU
// ownership or generation metadata.
bitstruct TextureIndex : uint {
    uint value : 0..31;
}

bitstruct SamplerIndex : uint {
    uint value : 0..31;
}

<* Owning token for device-scoped generic GPU data. *>
struct GpuAllocation {
    ulong owner;
    uint  index;
    uint  generation;
}
$assert GpuAllocation::size <= ulong::size * 2;

struct TextureHandle {
    ulong owner;
    uint  index;
    uint  generation;
}
$assert TextureHandle::size <= ulong::size * 2;

<* Owning CPU token for one published texture view. `index` is the raw value
   stored in shader data; ownership and generation stay CPU-only. *>
struct TextureView {
    ulong        owner;
    TextureIndex index;
    uint         generation;
}
$assert TextureView::size <= ulong::size * 2;

<* Immutable device-interned sampler identity. *>
struct Sampler {
    ulong owner;
    uint  index;
    uint  generation;
}
$assert Sampler::size <= ulong::size * 2;

struct PipelineHandle {
    ulong owner;
    uint  index;
    uint  generation;
}
$assert PipelineHandle::size <= ulong::size * 2;

struct SwapchainHandle {
    ulong owner;
    uint  index;
    uint  generation;
}
$assert SwapchainHandle::size <= ulong::size * 2;

struct ShaderHandle {
    ulong owner;
    uint  index;
    uint  generation;
}
$assert ShaderHandle::size <= ulong::size * 2;

struct CommandListHandle {
    ulong owner;
    uint  index;
    uint  generation;
}
$assert CommandListHandle::size <= ulong::size * 2;

const Device            DEVICE_INVALID              = {};
const GpuAllocation     GPU_ALLOCATION_INVALID      = {};
const TextureHandle     TEXTURE_HANDLE_INVALID      = {};
const TextureView       TEXTURE_VIEW_INVALID        = {};
const Sampler           SAMPLER_INVALID             = {};
const PipelineHandle    PIPELINE_HANDLE_INVALID     = {};
const SwapchainHandle   SWAPCHAIN_HANDLE_INVALID    = {};
const ShaderHandle      SHADER_HANDLE_INVALID       = {};
const CommandListHandle COMMAND_LIST_HANDLE_INVALID = {};

const TextureIndex TEXTURE_INDEX_INVALID = {};
const SamplerIndex SAMPLER_INDEX_INVALID = {};

<* True when the device handle is not the invalid zero sentinel. *>
fn bool Device.is_valid(self)          => (ulong)self != 0;
// Owner-bearing handles require both fields to be nonzero.
fn bool GpuAllocation.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool GpuAllocation.equals(self, GpuAllocation other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool TextureHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool TextureHandle.equals(self, TextureHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool TextureView.is_valid(self) {
    return self.owner != 0 && self.index.is_valid() && self.generation != 0;
}
fn bool TextureView.equals(self, TextureView other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool Sampler.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool Sampler.equals(self, Sampler other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool PipelineHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool PipelineHandle.equals(self, PipelineHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool SwapchainHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool SwapchainHandle.equals(self, SwapchainHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool ShaderHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool ShaderHandle.equals(self, ShaderHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool CommandListHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool CommandListHandle.equals(self, CommandListHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}

<* True for a live heap index; the zero value is the invalid sentinel. *>
fn bool TextureIndex.is_valid(self) => (uint)self != 0;
<* True for a live heap index; the zero value is the invalid sentinel. *>
fn bool SamplerIndex.is_valid(self) => (uint)self != 0;
