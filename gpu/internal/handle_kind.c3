module gpu::internal @private;

<* Public type named in a terminating handle message. *>
enum HandleKind : char (String type_name) @private {
    ADAPTER                     { "Adapter" },
    DEVICE                      { "Device" },
    ALLOCATION                  { "GpuAllocation" },
    GPU_SPAN                    { "GpuSpan" },
    TEXTURE                     { "TextureHandle" },
    TEXTURE_VIEW                { "TextureView" },
    TEXTURE_REQUIREMENTS        { "TextureRequirements" },
    ATTACHMENT_VIEW             { "AttachmentViewHandle" },
    ACCELERATION_STRUCTURE      { "AccelerationStructureHandle" },
    ACCELERATION_STRUCTURE_VIEW { "AccelerationStructureView" },
    PIPELINE                    { "PipelineHandle" },
    COMMAND_ALLOCATOR           { "CommandAllocator" },
    COMMAND_LIST                { "CommandList" },
    EXECUTABLE_COMMAND_LIST     { "ExecutableCommandList" },
    COMPLETION_POINT            { "CompletionPoint" },
    TIMESTAMP_POOL              { "TimestampPoolHandle" },
    SWAPCHAIN                   { "SwapchainHandle" },
}
