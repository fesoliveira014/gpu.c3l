module gpu_span_data_operations;

import gpu;

fn void check_span_data_operations(
    gpu::CommandList* commands,
    gpu::GpuSpan src,
    gpu::GpuSpan dst,
) {
    gpu::BufferCopyDesc copy = {
        .src = src,
        .dst = dst,
    };
    (void)gpu::cmd_copy_buffer(
        commands: commands,
        desc:     &copy,
    );
    (void)gpu::cmd_fill_buffer(
        commands: commands,
        dst:      dst,
        value:    0,
    );

    gpu::Barrier barrier = {
        .before = { .compute = true },
        .after = { .compute = true },
    };
    (void)gpu::cmd_barrier(
        commands: commands,
        barrier:  &barrier,
    );

    gpu::BufferTextureCopyDesc buffer_to_texture = {
        .src     = src,
        .texture = {},
    };
    (void)gpu::cmd_copy_buffer_to_texture(
        commands: commands,
        desc:     &buffer_to_texture,
    );
    gpu::TextureBufferCopyDesc texture_to_buffer = {
        .texture = {},
        .dst     = dst,
    };
    (void)gpu::cmd_copy_texture_to_buffer(
        commands: commands,
        desc:     &texture_to_buffer,
    );

    gpu::PipelineHandle pipeline = {};
    gpu::GpuAddress root = {};
    gpu::GraphicsState graphics = {};
    (void)gpu::cmd_bind_pipeline(commands, pipeline);
    (void)gpu::cmd_set_graphics_state(commands, &graphics);
    (void)gpu::cmd_dispatch(
        commands: commands,
        root:     root,
        groups:   { 1, 1, 1 },
    );
    (void)gpu::cmd_draw(
        commands:       commands,
        vertex_root:    root,
        fragment_root:  root,
        vertex_count:   3,
        instance_count: 1,
    );
    (void)gpu::cmd_draw_indexed(
        commands:       commands,

        vertex_root:    root,
        fragment_root:  root,
        index_span:     src,
        index_count:    1,
        instance_count: 1,
        index_type:     gpu::IndexType.U32,
    );
    (void)gpu::cmd_draw_indirect(
        commands:      commands,

        vertex_root:   root,
        fragment_root: root,
        args:          src,
        draw_count:    1,
    );
    (void)gpu::cmd_draw_indexed_indirect(
        commands:      commands,

        vertex_root:   root,
        fragment_root: root,
        args:          src,
        draw_count:    1,
        index_span:    dst,
        index_type:    gpu::IndexType.U32,
    );
    (void)gpu::cmd_draw_indexed_indirect_count(
        commands:       commands,

        vertex_root:    root,
        fragment_root:  root,
        args:           src,
        count_span:     dst,
        max_draw_count: 1,
        index_span:     dst,
        index_type:     gpu::IndexType.U32,
    );
    (void)gpu::cmd_dispatch_indirect(
        commands: commands,

        root:     root,
        args:     src,
    );
    (void)gpu::cmd_draw_generated(
        commands:       commands,
        records:        src,
        count_span:     dst,
        max_draw_count: 1,
    );
    (void)gpu::cmd_draw_indexed_generated(
        commands:       commands,
        records:        src,
        count_span:     dst,
        max_draw_count: 1,
        index_span:     dst,
        index_type:     gpu::IndexType.U32,
    );
    (void)gpu::cmd_dispatch_generated(
        commands:           commands,
        records:            src,
        count_span:         dst,
        max_dispatch_count: 1,
    );
}
