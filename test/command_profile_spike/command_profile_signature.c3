module command_profile_signature;

fn void cmd_dispatch() @if($feature(GPU_FAST_COMMANDS)) {}

fn void? cmd_dispatch() @if(!$feature(GPU_FAST_COMMANDS)) {
    return;
}

fn void fast_call() @if($feature(GPU_FAST_COMMANDS)) {
    cmd_dispatch();
}

fn void? checked_call() @if(!$feature(GPU_FAST_COMMANDS)) {
    cmd_dispatch()!;
}
