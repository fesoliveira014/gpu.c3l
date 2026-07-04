module gpu;

// Strongly-typed resource handles: one distinct bitstruct per kind over ulong,
// packed index(0..31) | generation(32..55) | reserved. A live handle has
// generation >= 1, so raw 0 is reserved for INVALID.

bitstruct BufferHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

bitstruct TextureHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

bitstruct PipelineHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

bitstruct SemaphoreHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

bitstruct SwapchainHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

bitstruct ShaderHandle : ulong {
    uint index      : 0..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

bitstruct RecordingContextHandle : ulong {
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

const TextureIndex TEXTURE_INDEX_INVALID = {};
const SamplerIndex SAMPLER_INDEX_INVALID = {};

fn bool BufferHandle.is_valid(self)    => (ulong)self != 0;
fn bool TextureHandle.is_valid(self)   => (ulong)self != 0;
fn bool PipelineHandle.is_valid(self)  => (ulong)self != 0;
fn bool SemaphoreHandle.is_valid(self) => (ulong)self != 0;
fn bool SwapchainHandle.is_valid(self) => (ulong)self != 0;
fn bool ShaderHandle.is_valid(self)    => (ulong)self != 0;
fn bool RecordingContextHandle.is_valid(self) => (ulong)self != 0;

fn bool TextureIndex.is_valid(self) => (uint)self != 0;
fn bool SamplerIndex.is_valid(self) => (uint)self != 0;
