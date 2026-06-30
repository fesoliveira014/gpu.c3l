module gpu;

// Strongly-typed resource handles. Each kind is a distinct bitstruct over
// `ulong` so the compiler rejects passing one handle kind where another is
// expected. Packed layout: index (0..31) | generation (32..55) | reserved.
// A live handle always has generation >= 1, so raw 0 is reserved for INVALID.

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

bitstruct SamplerHandle : ulong {
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

const BufferHandle BUFFER_HANDLE_INVALID       = {};
const TextureHandle TEXTURE_HANDLE_INVALID     = {};
const PipelineHandle PIPELINE_HANDLE_INVALID   = {};
const SamplerHandle SAMPLER_HANDLE_INVALID     = {};
const SemaphoreHandle SEMAPHORE_HANDLE_INVALID = {};
const SwapchainHandle SWAPCHAIN_HANDLE_INVALID = {};
const ShaderHandle SHADER_HANDLE_INVALID       = {};

fn bool BufferHandle.is_valid(self)    => (ulong)self != 0;
fn bool TextureHandle.is_valid(self)   => (ulong)self != 0;
fn bool PipelineHandle.is_valid(self)  => (ulong)self != 0;
fn bool SamplerHandle.is_valid(self)   => (ulong)self != 0;
fn bool SemaphoreHandle.is_valid(self) => (ulong)self != 0;
fn bool SwapchainHandle.is_valid(self) => (ulong)self != 0;
fn bool ShaderHandle.is_valid(self)    => (ulong)self != 0;
