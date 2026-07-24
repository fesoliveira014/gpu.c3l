module command_asm_surface;

import gpu;

fn void main() @if($feature(GPU_FAST_COMMANDS)) {
    gpu::CommandList commands;
    gpu::BufferCopyDesc copy;
    gpu::Barrier barrier;
    gpu::Viewport viewport;

    gpu::cmd_copy_buffer(&commands, &copy);
    gpu::cmd_barrier(&commands, &barrier);
    gpu::cmd_set_viewport(&commands, &viewport);
    gpu::cmd_dispatch(&commands, {}, {});
    gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        vertex_count:  3,
        instance_count: 1,
    );
}

fn void main() @if(!$feature(GPU_FAST_COMMANDS)) {
    gpu::CommandList commands;
    gpu::BufferCopyDesc copy;
    gpu::Barrier barrier;
    gpu::Viewport viewport;

    gpu::cmd_copy_buffer(&commands, &copy)!!;
    gpu::cmd_barrier(&commands, &barrier)!!;
    gpu::cmd_set_viewport(&commands, &viewport)!!;
    gpu::cmd_dispatch(&commands, {}, {})!!;
    gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        vertex_count:  3,
        instance_count: 1,
    )!!;
}
