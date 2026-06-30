module gpu;

// A shader-visible 64-bit GPU address. Distinct from a plain ulong so it can't
// be confused with sizes, offsets, or handles.
typedef GpuAddress = ulong;

enum MemoryKind : int {
    FRAME_UPLOAD,
    PERSISTENT_UPLOAD,
    DEVICE,
    READBACK,
    STAGING,
}

// Host-side view of a GPU-visible range. `cpu` may be null for non-mappable
// memory; `gpu` is non-zero for shader-visible memory.
struct GpuSpan {
    GpuAddress   gpu;
    void*        cpu;
    usz          size;
    BufferHandle buffer;
    usz          offset;
    MemoryKind   kind;
}

// Round `value` up to the next multiple of `align` (a power of two).
fn usz align_up(usz value, usz align) => (value + align - 1) & ~(align - 1);

<* Derive a sub-range starting `offset` bytes into this span. The GPU address
   and (when present) CPU pointer advance by `offset`; the backing buffer and
   memory kind are inherited. *>
fn GpuSpan GpuSpan.subspan(self, usz offset, usz size) {
    return {
        .gpu    = (GpuAddress)((ulong)self.gpu + offset),
        .cpu    = self.cpu == null ? null : (char*)self.cpu + offset,
        .size   = size,
        .buffer = self.buffer,
        .offset = self.offset + offset,
        .kind   = self.kind,
    };
}
