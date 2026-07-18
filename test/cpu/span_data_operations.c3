module gpu_span_data_operations;

import gpu;

fn void check_span_data_operations(
    gpu::CommandList* commands,
    gpu::Device* device,
    gpu::GpuSpan src,
    gpu::GpuSpan dst,
    char[] data,
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

    gpu::BufferBarrier barrier = { .span = dst };
    (void)gpu::cmd_buffer_barrier(
        commands: commands,
        barrier:  &barrier,
    );
    (void)gpu::cmd_upload_buffer(
        commands: commands,
        dst:      dst,
        data:     data,
    );
    (void)gpu::upload_buffer_data(
        device:      device,
        dst:         dst,
        data:        data,
        next_stage:  {},
        next_hazard: {},
    );
    (void)gpu::readback_buffer_data(
        device:      device,
        src:         src,
        out_data:    data,
        from_stage:  {},
        from_hazard: {},
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
    (void)gpu::cmd_draw_indexed(
        commands:       commands,
        pipeline:       pipeline,
        vertex_root:    root,
        fragment_root:  root,
        index_span:     src,
        index_count:    1,
        instance_count: 1,
        index_type:     gpu::IndexType.U32,
    );
    (void)gpu::cmd_draw_indirect(
        commands:      commands,
        pipeline:      pipeline,
        vertex_root:   root,
        fragment_root: root,
        args:          src,
        draw_count:    1,
    );
    (void)gpu::cmd_draw_indexed_indirect(
        commands:      commands,
        pipeline:      pipeline,
        vertex_root:   root,
        fragment_root: root,
        args:          src,
        draw_count:    1,
        index_span:    dst,
        index_type:    gpu::IndexType.U32,
    );
    (void)gpu::cmd_draw_indexed_indirect_count(
        commands:       commands,
        pipeline:       pipeline,
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
        pipeline: pipeline,
        root:     root,
        args:     src,
    );
}
