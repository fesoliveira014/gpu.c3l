module gpu;

alias DebugMessageCallback = fn void (DebugMessage*, void*);

enum DebugMessageSeverity : int {
    VERBOSE,
    INFO,
    WARNING,
    ERROR,
}

enum DebugResourceKind : int {
    NONE,
    DEVICE,
    BUFFER,
    TEXTURE,
    PIPELINE,
    SEMAPHORE,
    SWAPCHAIN,
    SHADER,
    RECORDING_CONTEXT,
    COMMAND_LIST,
    TEXTURE_DESCRIPTOR,
    SAMPLER,
    FRAME,
    PERSISTENT_SPAN,
}

bitstruct DebugMessageCategory : uint {
    bool public_contract   : 0;
    bool backend           : 1;
    bool validation        : 2;
    bool performance       : 3;
    bool resource_lifetime : 4;
    bool general           : 5;
}

<* Public resource identity when a diagnostic originates from a tracked slot.
   Strings remain valid only for the duration of the callback. *>
struct DebugResourceRef {
    DebugResourceKind kind;
    bool              has_public_identity;
    uint              index;
    uint              generation;
    ZString           debug_name;
}

<* Structured diagnostic delivered synchronously to DeviceDesc.debug_callback.
   All strings remain valid only for the duration of the callback. *>
struct DebugMessage {
    DebugMessageSeverity severity;
    DebugMessageCategory category;
    ZString              operation;
    bool                 has_fault;
    fault                public_fault;
    DebugResourceRef     resource;
    ZString              rejected_field;
    ZString              invariant;
    ZString              backend_text;
    ZString              validation_id_name;
    int                  validation_id_number;
}

<* Internal synchronous callback dispatch; null callbacks are ignored. *>
fn void dispatch_debug_message(DebugMessageCallback callback, void* user_data, DebugMessage* message) @private {
    if (callback == null) return;
    callback(message, user_data);
}

<* Open a debug-utils label region; a no-op when the extension is absent.
   Valid while recording, including inside render passes; balance is the
   caller's responsibility. *>
fn void? cmd_begin_label(CommandList* commands, ZString label, float[4] color = {}) {
    BackendVTable* vtable = command_vtable(commands)!;
    if (label == null) return INVALID_ARGUMENT~;
    return vtable.cmd_begin_label(commands, label, color);
}

<* Counterpart of cmd_begin_label. *>
fn void? cmd_end_label(CommandList* commands) {
    BackendVTable* vtable = command_vtable(commands)!;
    return vtable.cmd_end_label(commands);
}
