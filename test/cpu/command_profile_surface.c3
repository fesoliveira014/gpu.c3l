module gpu_profile_surface;

import gpu;

fn void? compile_fast_command_surface() @if($feature(GPU_FAST_COMMANDS)) {
    gpu::CommandList commands;
    gpu::GpuSpan span;
    gpu::BufferCopyDesc copy;
    gpu::BufferTextureCopyDesc buffer_texture;
    gpu::TextureBufferCopyDesc texture_buffer;
    gpu::Barrier barrier;
    gpu::TextureBarrier texture_barrier;
    gpu::PipelineHandle pipeline;
    gpu::DepthState depth;
    gpu::DynamicRasterState raster;
    gpu::Viewport viewport;
    gpu::ScissorRect scissor;
    gpu::GraphicsState graphics;
    gpu::RenderPassDesc render_pass;

    gpu::cmd_copy_buffer(&commands, &copy);
    gpu::cmd_fill_buffer(&commands, span, 0);
    gpu::cmd_copy_buffer_to_texture(&commands, &buffer_texture);
    gpu::cmd_copy_texture_to_buffer(&commands, &texture_buffer);
    gpu::cmd_barrier(&commands, &barrier);
    gpu::cmd_texture_barrier(&commands, &texture_barrier);
    gpu::cmd_bind_pipeline(&commands, pipeline);
    gpu::cmd_dispatch(&commands, {}, {});
    gpu::cmd_dispatch_indirect(&commands, {}, span);
    gpu::cmd_begin_label(&commands, "profile");
    gpu::cmd_end_label(&commands);
    gpu::cmd_set_depth_state(&commands, &depth);
    gpu::cmd_set_raster_state(&commands, &raster);
    gpu::cmd_set_viewport(&commands, &viewport);
    gpu::cmd_set_scissor(&commands, &scissor);
    gpu::cmd_set_graphics_state(&commands, &graphics);
    gpu::cmd_begin_render_pass(&commands, &render_pass);
    gpu::cmd_begin_render_pass_with_state(
        &commands,
        &render_pass,
        &graphics,
    );
    gpu::cmd_end_render_pass(&commands);
    gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        vertex_count:  3,
        instance_count: 1,
    );
    gpu::cmd_draw_indexed(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        index_span:    span,
        index_count:   3,
        instance_count: 1,
    );
    gpu::cmd_draw_indirect(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        args:          span,
        draw_count:    1,
    );
    gpu::cmd_draw_indexed_indirect(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        args:          span,
        draw_count:    1,
        index_span:    span,
    );
    gpu::cmd_draw_indexed_indirect_count(
        commands:       &commands,
        vertex_root:    {},
        fragment_root:  {},
        args:           span,
        count_span:     span,
        max_draw_count: 1,
        index_span:     span,
    );
    gpu::cmd_draw_generated(
        commands:       &commands,
        records:        span,
        count_span:     span,
        max_draw_count: 1,
    )!;
    gpu::cmd_draw_indexed_generated(
        commands:       &commands,
        records:        span,
        count_span:     span,
        max_draw_count: 1,
        index_span:     span,
    )!;
    gpu::cmd_dispatch_generated(
        commands:           &commands,
        records:            span,
        count_span:         span,
        max_dispatch_count: 1,
    )!;
}

fn void? compile_checked_command_surface() @if(!$feature(GPU_FAST_COMMANDS)) {
    gpu::CommandList commands;
    gpu::GpuSpan span;
    gpu::BufferCopyDesc copy;
    gpu::BufferTextureCopyDesc buffer_texture;
    gpu::TextureBufferCopyDesc texture_buffer;
    gpu::Barrier barrier;
    gpu::TextureBarrier texture_barrier;
    gpu::PipelineHandle pipeline;
    gpu::DepthState depth;
    gpu::DynamicRasterState raster;
    gpu::Viewport viewport;
    gpu::ScissorRect scissor;
    gpu::GraphicsState graphics;
    gpu::RenderPassDesc render_pass;

    gpu::cmd_copy_buffer(&commands, &copy)!;
    gpu::cmd_fill_buffer(&commands, span, 0)!;
    gpu::cmd_copy_buffer_to_texture(&commands, &buffer_texture)!;
    gpu::cmd_copy_texture_to_buffer(&commands, &texture_buffer)!;
    gpu::cmd_barrier(&commands, &barrier)!;
    gpu::cmd_texture_barrier(&commands, &texture_barrier)!;
    gpu::cmd_bind_pipeline(&commands, pipeline)!;
    gpu::cmd_dispatch(&commands, {}, {})!;
    gpu::cmd_dispatch_indirect(&commands, {}, span)!;
    gpu::cmd_begin_label(&commands, "profile")!;
    gpu::cmd_end_label(&commands)!;
    gpu::cmd_set_depth_state(&commands, &depth)!;
    gpu::cmd_set_raster_state(&commands, &raster)!;
    gpu::cmd_set_viewport(&commands, &viewport)!;
    gpu::cmd_set_scissor(&commands, &scissor)!;
    gpu::cmd_set_graphics_state(&commands, &graphics)!;
    gpu::cmd_begin_render_pass(&commands, &render_pass)!;
    gpu::cmd_begin_render_pass_with_state(
        &commands,
        &render_pass,
        &graphics,
    )!;
    gpu::cmd_end_render_pass(&commands)!;
    gpu::cmd_draw(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        vertex_count:  3,
        instance_count: 1,
    )!;
    gpu::cmd_draw_indexed(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        index_span:    span,
        index_count:   3,
        instance_count: 1,
    )!;
    gpu::cmd_draw_indirect(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        args:          span,
        draw_count:    1,
    )!;
    gpu::cmd_draw_indexed_indirect(
        commands:      &commands,
        vertex_root:   {},
        fragment_root: {},
        args:          span,
        draw_count:    1,
        index_span:    span,
    )!;
    gpu::cmd_draw_indexed_indirect_count(
        commands:       &commands,
        vertex_root:    {},
        fragment_root:  {},
        args:           span,
        count_span:     span,
        max_draw_count: 1,
        index_span:     span,
    )!;
    gpu::cmd_draw_generated(
        commands:       &commands,
        records:        span,
        count_span:     span,
        max_draw_count: 1,
    )!;
    gpu::cmd_draw_indexed_generated(
        commands:       &commands,
        records:        span,
        count_span:     span,
        max_draw_count: 1,
        index_span:     span,
    )!;
    gpu::cmd_dispatch_generated(
        commands:           &commands,
        records:            span,
        count_span:         span,
        max_dispatch_count: 1,
    )!;
}
