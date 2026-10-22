module gpu::internal @private;

import std::io;

const usz REJECTED_HANDLE_MESSAGE_CAPACITY @private = 192;

// A rejected handle is an application bug. Internal code propagates this
// fault so every defer (locks, pins, rollback) runs; the public boundary then
// terminates the process with the recorded kind and cause.
faultdef HANDLE_REJECTED;

enum HandleKind : char (String type_name) @private {
    RUNTIME                     { "Runtime" },
    ADAPTER                     { "Adapter" },
    SURFACE                     { "Surface" },
    DEVICE                      { "Device" },
    QUEUE                       { "Queue" },
    ALLOCATION                  { "GpuAllocation" },
    GPU_SPAN                    { "GpuSpan" },
    TEXTURE                     { "TextureHandle" },
    TEXTURE_VIEW                { "TextureView" },
    TEXTURE_INDEX_RANGE         { "TextureIndexRange" },
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
    SWAPCHAIN_READINESS         { "SwapchainReadiness" },
    ACQUIRED_IMAGE              { "AcquiredImage" },
}

enum HandleCause : char (String text) @private {
    NULL_POINTER    { "null" },
    ZERO            { "zero" },
    MALFORMED       { "malformed" },
    STALE           { "stale" },
    FOREIGN         { "foreign" },
    CONSUMED        { "consumed" },
    MISMATCHED      { "mismatched" },
    INVARIANT       { "corrupt" },
    DESTROY_OVERLAP { "overlapping" },
}

struct HandleRejection @private {
    HandleKind  kind;
    HandleCause cause;
}

tlocal HandleRejection last_handle_rejection @private;

<* Record why a handle was rejected and return the private rejection fault. *>
fn fault reject_handle(HandleKind kind, HandleCause cause) @private {
    last_handle_rejection = {
        .kind  = kind,
        .cause = cause,
    };
    return HANDLE_REJECTED;
}

<* Format the most recent rejection on this thread for `operation`. *>
fn String rejected_handle_message(String operation, char[] buffer) @private {
    HandleRejection rejection = last_handle_rejection;
    if (rejection.cause == HandleCause.DESTROY_OVERLAP) {
        return (String)io::bprintf(
            buffer,
            "gpu::%s: %s used concurrently with destroy_device",
            operation,
            rejection.kind.type_name,
        ) ?? operation;
    }
    return (String)io::bprintf(
        buffer,
        "gpu::%s: %s %s",
        operation,
        rejection.cause.text,
        rejection.kind.type_name,
    ) ?? operation;
}

<* Terminate when `err` is a rejected handle; other faults return to the caller. *>
fn void terminate_rejected_handle(fault err, String operation) @private {
    if (err != HANDLE_REJECTED) return;
    char[REJECTED_HANDLE_MESSAGE_CAPACITY] buffer;
    abort("%s", rejected_handle_message(operation, buffer[..]));
}

<* Public boundary: register first with `defer (catch err)` so it runs after
   every other defer of the calling public function. *>
macro void @terminate_rejected_handle(fault err) {
    terminate_rejected_handle(err, $$FUNC);
}

<* Unwrap an internal result in a public function that returns no optional.
   Only a rejected handle is expected; any other fault is a library bug. *>
macro @require_valid(#expr) {
    var value = #expr;
    if (catch err = value) {
        terminate_rejected_handle(err, $$FUNC);
        abort("gpu::%s: unexpected fault %s", $$FUNC, err);
    }
    return value;
}
