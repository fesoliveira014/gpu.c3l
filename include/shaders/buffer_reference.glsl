// gpu.c3l canonical buffer-reference declarations.
//
// The ABI generator owns root records. These macros cover the hand-written
// blocks that view data reached through a root address: one record or a
// runtime array, readable or writable. Each expands to a complete std430
// buffer-reference block with a single member — `value` for a record,
// `values[]` for a runtime array — and the call site supplies the semicolon:
//
//   GPU_DECLARE_READONLY_ARRAY_REF(MaterialTable, Material);
//   Material material = MaterialTable(root.materials_gpu).values[index];
//
// The macros do not set `buffer_reference_align`, so a block keeps the
// extension's default 16-byte reference alignment; an array element access
// uses the element's own alignment instead. Declare the block directly when a
// record sits at a less strictly aligned address, or when it needs a layout
// other than std430, unqualified read-write access, or more than one member.
//
// A reference carries an address and nothing else: no length, no ownership, no
// bounds checking, no robustness. Counts and bounds stay ordinary root fields
// and ordinary shader code, and keeping the allocation alive through GPU
// completion remains an application contract.

#ifndef GPU_BUFFER_REFERENCE_GLSL
#define GPU_BUFFER_REFERENCE_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

#define GPU_DECLARE_READONLY_REF(NAME, TYPE) \
    layout(buffer_reference, std430) readonly buffer NAME { TYPE value; }

#define GPU_DECLARE_WRITEONLY_REF(NAME, TYPE) \
    layout(buffer_reference, std430) writeonly buffer NAME { TYPE value; }

#define GPU_DECLARE_READONLY_ARRAY_REF(NAME, TYPE) \
    layout(buffer_reference, std430) readonly buffer NAME { TYPE values[]; }

#define GPU_DECLARE_WRITEONLY_ARRAY_REF(NAME, TYPE) \
    layout(buffer_reference, std430) writeonly buffer NAME { TYPE values[]; }

#endif
