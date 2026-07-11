module gpu;

<* Open a debug-utils label region; a no-op when the extension is absent.
   Valid while recording, including inside render passes; balance is the
   caller's responsibility. *>
fn void? cmd_begin_label(CommandList* commands, ZString label, float[4] color = {}) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_begin_label(commands, label, color);
}

<* Counterpart of cmd_begin_label. *>
fn void? cmd_end_label(CommandList* commands) {
    if (commands.device == null || commands.device.vtable == null) return INVALID_ARGUMENT~;
    return commands.device.vtable.cmd_end_label(commands);
}
