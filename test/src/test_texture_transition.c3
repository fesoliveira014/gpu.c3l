module gpu_test;

import gpu;

fn gpu::TextureHandle transition_texture() {
    return { .index = 0, .generation = 1 };
}

fn gpu::TextureState transfer_source_state() {
    return {
        .layout = gpu::TextureLayout.TRANSFER_SOURCE,
        .stages = { .transfer },
        .access = { .read },
    };
}

fn gpu::TextureState transfer_destination_state() {
    return {
        .layout = gpu::TextureLayout.TRANSFER_DESTINATION,
        .stages = { .transfer },
        .access = { .write },
    };
}

fn void assert_texture_state(
    gpu::TextureState actual,
    gpu::TextureState expected,
) {
    assert(actual.layout == expected.layout
        && (uint)actual.stages == (uint)expected.stages
        && (uint)actual.access == (uint)expected.access,
        "texture state fields match");
}

fn void texture_state_helpers_compose_exact_states() @test {
    gpu::StageMask sampled_stages = {
        .vertex_shader   = true,
        .fragment_shader = true,
    };
    gpu::TextureState sampled = gpu::sampled_at(sampled_stages);
    assert(sampled.layout == gpu::TextureLayout.SAMPLED,
        "sampled_at selects sampled layout");
    assert((uint)sampled.stages == (uint)sampled_stages,
        "sampled_at preserves every shader stage");
    assert(sampled.access.read && !sampled.access.write,
        "sampled_at selects read-only access");

    gpu::TextureAccess[4] access_modes = {
        {},
        { .read },
        { .write },
        { .read, .write },
    };
    gpu::StageMask storage_stages = { .compute };
    foreach (access : access_modes) {
        gpu::TextureState storage = gpu::storage_at(storage_stages, access);
        assert(storage.layout == gpu::TextureLayout.STORAGE,
            "storage_at selects storage layout");
        assert((uint)storage.stages == (uint)storage_stages,
            "storage_at preserves every shader stage");
        assert((uint)storage.access == (uint)access,
            "storage_at preserves the requested access bits");
    }
}

fn void texture_transition_preserves_compositional_states() @test {
    gpu::TextureHandle texture = transition_texture();
    gpu::TextureState before = transfer_source_state();
    gpu::TextureState after = gpu::sampled_at({ .compute });
    gpu::TextureBarrier barrier = gpu::texture_transition(
        texture: texture,
        before:  before,
        after:   after,
    )!!;

    assert(barrier.texture == texture, "transition preserves the texture handle");
    assert(barrier.view.format == gpu::Format.UNDEFINED
        && barrier.view.base_mip == 0
        && barrier.view.mip_count == 0
        && barrier.view.base_layer == 0
        && barrier.view.layer_count == 0,
        "whole-texture transition uses the full default view");
    assert_texture_state(barrier.before, before);
    assert_texture_state(barrier.after, after);
}

fn void texture_view_transition_preserves_subresource_range() @test {
    gpu::TextureHandle texture = transition_texture();
    gpu::TextureViewDesc view = {
        .base_mip    = 2,
        .mip_count   = 3,
        .base_layer  = 4,
        .layer_count = 5,
    };
    gpu::TextureState before = transfer_destination_state();
    gpu::TextureState after = gpu::sampled_at({ .fragment_shader });
    gpu::TextureBarrier barrier = gpu::texture_view_transition(
        texture: texture,
        view:    view,
        before:  before,
        after:   after,
    )!!;

    assert(barrier.texture == texture, "view transition preserves the texture handle");
    assert(barrier.view.format == view.format
        && barrier.view.base_mip == view.base_mip
        && barrier.view.mip_count == view.mip_count
        && barrier.view.base_layer == view.base_layer
        && barrier.view.layer_count == view.layer_count,
        "view transition preserves the selected subresources");
    assert_texture_state(barrier.before, before);
    assert_texture_state(barrier.after, after);
}

fn void texture_transition_accepts_every_source_layout() @test {
    gpu::TextureLayout[8] layouts = {
        gpu::TextureLayout.UNDEFINED,
        gpu::TextureLayout.TRANSFER_SOURCE,
        gpu::TextureLayout.TRANSFER_DESTINATION,
        gpu::TextureLayout.SAMPLED,
        gpu::TextureLayout.STORAGE,
        gpu::TextureLayout.COLOR_ATTACHMENT,
        gpu::TextureLayout.DEPTH_ATTACHMENT,
        gpu::TextureLayout.PRESENT,
    };
    gpu::TextureState after = transfer_destination_state();

    foreach (layout : layouts) {
        gpu::TextureState before = { .layout = layout };
        gpu::TextureBarrier barrier = gpu::texture_transition(
            texture: transition_texture(),
            before:  before,
            after:   after,
        )!!;
        assert(barrier.before.layout == layout,
            "every public layout is accepted as a structural source");
    }
}

fn void texture_transition_rejects_invalid_layouts() @test {
    gpu::TextureLayout invalid_layout = bitcast(999, gpu::TextureLayout);
    fault invalid_before = @catch(gpu::texture_transition(
        texture: transition_texture(),
        before:  { .layout = invalid_layout },
        after:   transfer_destination_state(),
    ));
    assert(invalid_before == gpu::INVALID_ARGUMENT,
        "invalid source layout faults INVALID_ARGUMENT");

    fault invalid_after = @catch(gpu::texture_transition(
        texture: transition_texture(),
        before:  transfer_source_state(),
        after:   { .layout = invalid_layout },
    ));
    assert(invalid_after == gpu::INVALID_ARGUMENT,
        "invalid destination layout faults INVALID_ARGUMENT");
}

fn void texture_transition_rejects_undefined_destination() @test {
    fault invalid = @catch(gpu::texture_transition(
        texture: transition_texture(),
        before:  transfer_source_state(),
        after:   { .layout = gpu::TextureLayout.UNDEFINED },
    ));
    assert(invalid == gpu::INVALID_ARGUMENT, "UNDEFINED is source-only");
}

fn void texture_view_transition_rejects_invalid_layouts() @test {
    gpu::TextureLayout invalid_layout = bitcast(999, gpu::TextureLayout);
    fault invalid_before = @catch(gpu::texture_view_transition(
        texture: transition_texture(),
        view:    {},
        before:  { .layout = invalid_layout },
        after:   transfer_destination_state(),
    ));
    assert(invalid_before == gpu::INVALID_ARGUMENT,
        "invalid view source layout faults INVALID_ARGUMENT");

    fault invalid_after = @catch(gpu::texture_view_transition(
        texture: transition_texture(),
        view:    {},
        before:  transfer_source_state(),
        after:   { .layout = invalid_layout },
    ));
    assert(invalid_after == gpu::INVALID_ARGUMENT,
        "invalid view destination layout faults INVALID_ARGUMENT");
}

fn void acquired_image_exposes_prior_state() @test {
    gpu::TextureState prior_state = {
        .layout = gpu::TextureLayout.PRESENT,
    };
    gpu::AcquiredImage image = { .prior_state = prior_state };
    assert_texture_state(image.prior_state, prior_state);
}
