module gpu::utils;

import gpu;

<* Aligned non-owning views of a mapped GPU span.
   All fields inherit the lifetime of the span's owning allocation. *>
struct MappedGpuSpan {
    gpu::GpuSpan    span;
    char[]          bytes;
    gpu::GpuAddress address;
}

<* Derive a non-empty exact-sized mapped range within this immediate parent.
   The returned views inherit the owning allocation's lifetime and perform no
   synchronization.

   @param offset : "Byte offset relative to this mapped span."
   @param size : "Nonzero byte size of the derived mapped span."
   @return "An aligned non-owning mapped subspan, or the recoverable range or overflow fault from the underlying span." *>
fn MappedGpuSpan? MappedGpuSpan.checked_subspan(
    &self,
    usz offset,
    usz size,
) {
    gpu::GpuSpan child = self.span.checked_subspan(offset, size)!;
    return {
        .span    = child,
        .bytes   = self.bytes[offset : size],
        .address = self.address + (gpu::GpuAddress)offset,
    };
}

<* Borrow aligned CPU and GPU views of a live host-visible span.
   The result inherits the owning allocation's lifetime and performs no
   synchronization.

   @param device : "Live device that owns the span."
   @param span : "Non-owning host-visible allocation range."
   @return "Aligned borrowed views, or the recoverable fault from the public mapping or address query." *>
fn MappedGpuSpan? mapped_gpu_span(
    gpu::Device* device,
    gpu::GpuSpan span,
) {
    char[] bytes = gpu::get_span_mapping(device, span)!;
    gpu::GpuAddress address = gpu::get_span_address(device, span)!;
    return {
        .span    = span,
        .bytes   = bytes,
        .address = address,
    };
}
