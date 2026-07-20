module gpu_test;

import gpu;

fn gpu::TextureHandle transition_texture() {
    return { .index = 0, .generation = 1 };
}

fn void texture_transition_preserves_semantic_uses() @test {
    gpu::TextureHandle texture = transition_texture();
    gpu::TextureBarrier barrier = gpu::texture_transition(
        texture: texture,
        before:  gpu::TextureUse.TRANSFER_SOURCE,
        after:   gpu::TextureUse.SAMPLED_COMPUTE,
    )!!;

    assert(barrier.texture == texture, "transition preserves the texture handle");
    assert(barrier.view.format == gpu::Format.UNDEFINED
        && barrier.view.base_mip == 0
        && barrier.view.mip_count == 0
        && barrier.view.base_layer == 0
        && barrier.view.layer_count == 0,
        "whole-texture transition uses the full default view");
    assert(barrier.before == gpu::TextureUse.TRANSFER_SOURCE, "transition preserves the previous semantic use");
    assert(barrier.after == gpu::TextureUse.SAMPLED_COMPUTE, "transition preserves the next semantic use");
}

fn void texture_view_transition_preserves_subresource_range() @test {
    gpu::TextureHandle texture = transition_texture();
    gpu::TextureViewDesc view = {
        .base_mip    = 2,
        .mip_count   = 3,
        .base_layer  = 4,
        .layer_count = 5,
    };
    gpu::TextureBarrier barrier = gpu::texture_view_transition(
        texture: texture,
        view:    view,
        before:  gpu::TextureUse.TRANSFER_DESTINATION,
        after:   gpu::TextureUse.SAMPLED_FRAGMENT,
    )!!;

    assert(barrier.texture == texture, "view transition preserves the texture handle");
    assert(barrier.view.format == view.format
        && barrier.view.base_mip == view.base_mip
        && barrier.view.mip_count == view.mip_count
        && barrier.view.base_layer == view.base_layer
        && barrier.view.layer_count == view.layer_count,
        "view transition preserves the selected subresources");
    assert(barrier.before == gpu::TextureUse.TRANSFER_DESTINATION, "view transition preserves the previous use");
    assert(barrier.after == gpu::TextureUse.SAMPLED_FRAGMENT, "view transition preserves the next use");
}

fn void texture_transition_accepts_every_source_use() @test {
    gpu::TextureUse[9] uses = {
        gpu::TextureUse.UNDEFINED,
        gpu::TextureUse.TRANSFER_SOURCE,
        gpu::TextureUse.TRANSFER_DESTINATION,
        gpu::TextureUse.SAMPLED_COMPUTE,
        gpu::TextureUse.SAMPLED_FRAGMENT,
        gpu::TextureUse.STORAGE_COMPUTE,
        gpu::TextureUse.COLOR_ATTACHMENT,
        gpu::TextureUse.DEPTH_ATTACHMENT,
        gpu::TextureUse.PRESENT,
    };

    foreach (use : uses) {
        gpu::TextureBarrier barrier = gpu::texture_transition(
            texture: transition_texture(),
            before:  use,
            after:   gpu::TextureUse.TRANSFER_DESTINATION,
        )!!;
        assert(barrier.before == use, "every semantic use is valid as a source");
    }
}

fn void texture_transition_rejects_invalid_uses() @test {
    gpu::TextureUse invalid_use = bitcast(999, gpu::TextureUse);
    fault invalid_before = @catch(gpu::texture_transition(
        texture: transition_texture(),
        before:  invalid_use,
        after:   gpu::TextureUse.PRESENT,
    ));
    assert(invalid_before == gpu::INVALID_ARGUMENT, "invalid source use faults INVALID_ARGUMENT");

    fault invalid_after = @catch(gpu::texture_transition(
        texture: transition_texture(),
        before:  gpu::TextureUse.PRESENT,
        after:   invalid_use,
    ));
    assert(invalid_after == gpu::INVALID_ARGUMENT, "invalid destination use faults INVALID_ARGUMENT");
}

fn void texture_transition_rejects_undefined_destination() @test {
    fault invalid = @catch(gpu::texture_transition(
        texture: transition_texture(),
        before:  gpu::TextureUse.PRESENT,
        after:   gpu::TextureUse.UNDEFINED,
    ));
    assert(invalid == gpu::INVALID_ARGUMENT, "UNDEFINED is source-only");
}

fn void texture_view_transition_rejects_invalid_uses() @test {
    gpu::TextureUse invalid_use = bitcast(999, gpu::TextureUse);
    fault invalid_before = @catch(gpu::texture_view_transition(
        texture: transition_texture(),
        view:    {},
        before:  invalid_use,
        after:   gpu::TextureUse.PRESENT,
    ));
    assert(invalid_before == gpu::INVALID_ARGUMENT, "invalid view source use faults INVALID_ARGUMENT");

    fault invalid_after = @catch(gpu::texture_view_transition(
        texture: transition_texture(),
        view:    {},
        before:  gpu::TextureUse.PRESENT,
        after:   invalid_use,
    ));
    assert(invalid_after == gpu::INVALID_ARGUMENT, "invalid view destination use faults INVALID_ARGUMENT");
}
