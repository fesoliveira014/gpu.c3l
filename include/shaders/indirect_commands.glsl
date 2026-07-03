// gpu.c3l indirect argument shader ABI.
//
// Struct layouts must match command.c3 (DrawIndirectCommand,
// DrawIndexedIndirectCommand, DispatchIndirectCommand) and Vulkan's argument
// layouts byte-for-byte: 16 / 20 / 12 bytes, std430, no padding.
//
// A multi-draw shares one vertex/fragment root pair across all draws;
// per-draw variation indexes a table through gl_DrawID (shaderDrawParameters
// is a required device feature).

#ifndef GPU_INDIRECT_COMMANDS_GLSL
#define GPU_INDIRECT_COMMANDS_GLSL

struct DrawIndirectCommand {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};

struct DrawIndexedIndirectCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int  vertex_offset;
    uint first_instance;
};

struct DispatchIndirectCommand {
    uint x;
    uint y;
    uint z;
};

#endif
