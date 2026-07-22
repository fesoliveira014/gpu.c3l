module gpu;


<* Internal synchronous callback dispatch; null callbacks are ignored. *>
fn void dispatch_debug_message(DebugMessageCallback callback, void* user_data, DebugMessage* message) @private {
    if (callback == null) return;
    callback(message, user_data);
}
