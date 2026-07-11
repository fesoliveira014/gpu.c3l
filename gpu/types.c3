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

// Strongly-typed resource handles: one distinct bitstruct per kind over ulong,
// packed index(0..31) | generation(32..55) | reserved. A live handle has
// generation >= 1, so raw 0 is reserved for INVALID.

bitstruct BufferHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

// Same packing as BufferHandle.
bitstruct TextureHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

// Same packing as BufferHandle.
bitstruct PipelineHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

// Same packing as BufferHandle.
bitstruct SemaphoreHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

// Same packing as BufferHandle.
bitstruct SwapchainHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

// Same packing as BufferHandle.
bitstruct ShaderHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

// Same packing as BufferHandle.
bitstruct RecordingContextHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

// Same packing as BufferHandle; identity for a device-owned command record.
bitstruct CommandListHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

// Shader-visible heap indices, packed slot(0..15) | generation(16..31).
// Embedded in GPU records as 32-bit scalars; shaders mask with `& 0xFFFFu`.
// A live index has generation >= 1, so raw 0 is INVALID.

bitstruct TextureIndex : uint {
    ushort slot       : 0..15;
    ushort generation : 16..31;
}

// Same packing as TextureIndex.
bitstruct SamplerIndex : uint {
    ushort slot       : 0..15;
    ushort generation : 16..31;
}

const BufferHandle BUFFER_HANDLE_INVALID       = {};
const TextureHandle TEXTURE_HANDLE_INVALID     = {};
const PipelineHandle PIPELINE_HANDLE_INVALID   = {};
const SemaphoreHandle SEMAPHORE_HANDLE_INVALID = {};
const SwapchainHandle SWAPCHAIN_HANDLE_INVALID = {};
const ShaderHandle SHADER_HANDLE_INVALID       = {};
const RecordingContextHandle RECORDING_CONTEXT_HANDLE_INVALID = {};
const CommandListHandle COMMAND_LIST_HANDLE_INVALID = {};

const TextureIndex TEXTURE_INDEX_INVALID = {};
const SamplerIndex SAMPLER_INDEX_INVALID = {};

<* True for a live handle; the zero value is the invalid sentinel. *>
fn bool BufferHandle.is_valid(self)    => (ulong)self != 0;
<* True for a live handle; the zero value is the invalid sentinel. *>
fn bool TextureHandle.is_valid(self)   => (ulong)self != 0;
<* True for a live handle; the zero value is the invalid sentinel. *>
fn bool PipelineHandle.is_valid(self)  => (ulong)self != 0;
<* True for a live handle; the zero value is the invalid sentinel. *>
fn bool SemaphoreHandle.is_valid(self) => (ulong)self != 0;
<* True for a live handle; the zero value is the invalid sentinel. *>
fn bool SwapchainHandle.is_valid(self) => (ulong)self != 0;
<* True for a live handle; the zero value is the invalid sentinel. *>
fn bool ShaderHandle.is_valid(self)    => (ulong)self != 0;
<* True for a live handle; the zero value is the invalid sentinel. *>
fn bool RecordingContextHandle.is_valid(self) => (ulong)self != 0;
<* True for a live command-list handle; the zero value is the invalid sentinel. *>
fn bool CommandListHandle.is_valid(self) => (ulong)self != 0;

<* True for a live heap index; the zero value is the invalid sentinel. *>
fn bool TextureIndex.is_valid(self) => (uint)self != 0;
<* True for a live heap index; the zero value is the invalid sentinel. *>
fn bool SamplerIndex.is_valid(self) => (uint)self != 0;
