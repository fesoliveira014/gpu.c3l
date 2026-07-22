module gpu;

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
fn bool AttachmentViewHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool AttachmentViewHandle.equals(self, AttachmentViewHandle other) @operator(==) {
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
fn bool CommandListHandle.is_valid(self) => self.owner != 0 && self.generation != 0;
fn bool CommandListHandle.equals(self, CommandListHandle other) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}
fn bool CommandAllocatorHandle.is_valid(self) =>
    self.owner != 0 && self.generation != 0;
fn bool CommandAllocatorHandle.equals(
    self,
    CommandAllocatorHandle other,
) @operator(==) {
    return self.owner == other.owner
        && self.index == other.index
        && self.generation == other.generation;
}

<* True for a live heap index; the zero value is the invalid sentinel. *>
fn bool TextureIndex.is_valid(self) => (uint)self != 0;
<* True for a live heap index; the zero value is the invalid sentinel. *>
fn bool SamplerIndex.is_valid(self) => (uint)self != 0;
