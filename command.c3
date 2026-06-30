module gpu;

// Command-area descriptor types (render targets and copies). The command-list
// recorder and submit descriptors depend on backend state and arrive with the
// command milestone, not here.

enum LoadOp : int {
    LOAD,
    CLEAR,
    DONT_CARE,
}

enum StoreOp : int {
    STORE,
    DONT_CARE,
}

struct ClearColor {
    float[4] rgba;
}

struct ClearDepthStencil {
    float depth;
    uint  stencil;
}

struct ColorTargetDesc {
    TextureHandle texture;
    uint          mip_level;
    uint          array_layer;
    LoadOp        load_op;
    StoreOp       store_op;
    ClearColor    clear;
}

struct DepthTargetDesc {
    TextureHandle     texture;
    LoadOp            load_op;
    StoreOp           store_op;
    ClearDepthStencil clear;
}

struct RenderPassDesc {
    ColorTargetDesc[] colors;
    DepthTargetDesc*  depth;
    uint              width;
    uint              height;
}

struct BufferCopyDesc {
    BufferHandle src;
    BufferHandle dst;
    usz          src_offset;
    usz          dst_offset;
    usz          size;
}

struct BufferTextureCopyDesc {
    BufferHandle  buffer;
    usz           buffer_offset;
    TextureHandle texture;
    uint          mip_level;
    uint          array_layer;
}

struct TextureBufferCopyDesc {
    TextureHandle texture;
    uint          mip_level;
    uint          array_layer;
    BufferHandle  buffer;
    usz           buffer_offset;
}
