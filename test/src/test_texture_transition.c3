module gpu_test;

import gpu::compat;

struct TextureUseCase {
    gpu::compat::TextureUse    use;
    gpu::compat::Stage         stage;
    gpu::compat::Hazard        hazard;
    gpu::compat::TextureLayout layout;
}

fn gpu::compat::TextureHandle transition_texture() {
    return { .index = 0, .generation = 1 };
}

fn void texture_use_mappings_are_exact() @test {
    TextureUseCase[8] cases = {
        { gpu::compat::TextureUse.UNDEFINED,            gpu::compat::Stage.HOST,             gpu::compat::Hazard.NONE,              gpu::compat::TextureLayout.UNDEFINED },
        { gpu::compat::TextureUse.TRANSFER_DESTINATION, gpu::compat::Stage.TRANSFER,         gpu::compat::Hazard.TRANSFER_WRITE,    gpu::compat::TextureLayout.TRANSFER_DST },
        { gpu::compat::TextureUse.SAMPLED_COMPUTE,      gpu::compat::Stage.COMPUTE_SHADER,   gpu::compat::Hazard.SHADER_READ,       gpu::compat::TextureLayout.SHADER_READ },
        { gpu::compat::TextureUse.SAMPLED_FRAGMENT,     gpu::compat::Stage.FRAGMENT_SHADER,  gpu::compat::Hazard.SHADER_READ,       gpu::compat::TextureLayout.SHADER_READ },
        { gpu::compat::TextureUse.STORAGE_COMPUTE,      gpu::compat::Stage.COMPUTE_SHADER,   gpu::compat::Hazard.SHADER_READ_WRITE, gpu::compat::TextureLayout.GENERAL },
        { gpu::compat::TextureUse.COLOR_ATTACHMENT,     gpu::compat::Stage.COLOR_ATTACHMENT, gpu::compat::Hazard.COLOR_READ_WRITE,  gpu::compat::TextureLayout.COLOR_ATTACHMENT },
        { gpu::compat::TextureUse.DEPTH_ATTACHMENT,     gpu::compat::Stage.DEPTH_STENCIL,    gpu::compat::Hazard.DEPTH_READ_WRITE,  gpu::compat::TextureLayout.DEPTH_STENCIL },
        { gpu::compat::TextureUse.PRESENT,              gpu::compat::Stage.COLOR_ATTACHMENT, gpu::compat::Hazard.NONE,              gpu::compat::TextureLayout.PRESENT },
    };
    gpu::compat::TextureHandle texture = transition_texture();

    foreach (&expected : cases) {
        gpu::compat::TextureUse after = expected.use == gpu::compat::TextureUse.UNDEFINED
            ? gpu::compat::TextureUse.TRANSFER_DESTINATION
            : expected.use;
        gpu::compat::TextureBarrier source = gpu::compat::texture_transition(
            texture: texture,
            before:  expected.use,
            after:   after,
        )!!;
        assert(source.texture == texture, "transition preserves the texture handle");
        assert(source.before_stage == expected.stage, "source stage matches the preset");
        assert(source.before_hazard == expected.hazard, "source hazard matches the preset");
        assert(source.old_layout == expected.layout, "source layout matches the preset");

        if (expected.use != gpu::compat::TextureUse.UNDEFINED) {
            gpu::compat::TextureBarrier destination = gpu::compat::texture_transition(
                texture: texture,
                before:  gpu::compat::TextureUse.UNDEFINED,
                after:   expected.use,
            )!!;
            assert(destination.after_stage == expected.stage, "destination stage matches the preset");
            assert(destination.after_hazard == expected.hazard, "destination hazard matches the preset");
            assert(destination.new_layout == expected.layout, "destination layout matches the preset");
        }
    }
}

fn void texture_transition_rejects_invalid_uses() @test {
    gpu::compat::TextureUse invalid_use = bitcast(999, gpu::compat::TextureUse);
    fault invalid_before = @catch(gpu::compat::texture_transition(
        texture: transition_texture(),
        before:  invalid_use,
        after:   gpu::compat::TextureUse.PRESENT,
    ));
    assert(invalid_before == gpu::compat::INVALID_ARGUMENT, "invalid source use faults INVALID_ARGUMENT");

    fault invalid_after = @catch(gpu::compat::texture_transition(
        texture: transition_texture(),
        before:  gpu::compat::TextureUse.PRESENT,
        after:   invalid_use,
    ));
    assert(invalid_after == gpu::compat::INVALID_ARGUMENT, "invalid destination use faults INVALID_ARGUMENT");
}

fn void texture_transition_rejects_undefined_destination() @test {
    fault invalid = @catch(gpu::compat::texture_transition(
        texture: transition_texture(),
        before:  gpu::compat::TextureUse.PRESENT,
        after:   gpu::compat::TextureUse.UNDEFINED,
    ));
    assert(invalid == gpu::compat::INVALID_ARGUMENT, "UNDEFINED is source-only");
}

fn void storage_same_use_transition_keeps_combined_access() @test {
    gpu::compat::TextureBarrier barrier = gpu::compat::texture_transition(
        texture: transition_texture(),
        before:  gpu::compat::TextureUse.STORAGE_COMPUTE,
        after:   gpu::compat::TextureUse.STORAGE_COMPUTE,
    )!!;
    assert(barrier.before_hazard == gpu::compat::Hazard.SHADER_READ_WRITE,
        "storage source includes shader reads and writes");
    assert(barrier.after_hazard == gpu::compat::Hazard.SHADER_READ_WRITE,
        "storage destination includes shader reads and writes");
    assert(barrier.old_layout == gpu::compat::TextureLayout.GENERAL
        && barrier.new_layout == gpu::compat::TextureLayout.GENERAL,
        "same-use storage barrier preserves general layout");
}

fn void presentation_transitions_have_narrow_scopes() @test {
    gpu::compat::TextureHandle texture = transition_texture();
    gpu::compat::TextureBarrier acquire = gpu::compat::texture_transition(
        texture: texture,
        before:  gpu::compat::TextureUse.PRESENT,
        after:   gpu::compat::TextureUse.COLOR_ATTACHMENT,
    )!!;
    assert(acquire.texture == texture, "acquire preserves the texture handle");
    assert(acquire.before_stage == gpu::compat::Stage.COLOR_ATTACHMENT, "acquire source stage matches the WSI wait");
    assert(acquire.after_stage == gpu::compat::Stage.COLOR_ATTACHMENT, "acquire destination stage is color output");
    assert(acquire.before_hazard == gpu::compat::Hazard.NONE, "acquire source access is empty");
    assert(acquire.after_hazard == gpu::compat::Hazard.COLOR_READ_WRITE, "acquire destination access is color read/write");
    assert(acquire.old_layout == gpu::compat::TextureLayout.PRESENT, "acquire source layout is present");
    assert(acquire.new_layout == gpu::compat::TextureLayout.COLOR_ATTACHMENT, "acquire destination layout is color attachment");

    gpu::compat::TextureBarrier release = gpu::compat::texture_transition(
        texture: texture,
        before:  gpu::compat::TextureUse.COLOR_ATTACHMENT,
        after:   gpu::compat::TextureUse.PRESENT,
    )!!;
    assert(release.texture == texture, "release preserves the texture handle");
    assert(release.before_stage == gpu::compat::Stage.COLOR_ATTACHMENT, "release source stage is color output");
    assert(release.after_stage == gpu::compat::Stage.COLOR_ATTACHMENT, "release destination stage matches the WSI signal");
    assert(release.before_hazard == gpu::compat::Hazard.COLOR_READ_WRITE, "release source access is color read/write");
    assert(release.after_hazard == gpu::compat::Hazard.NONE, "release destination access is empty");
    assert(release.old_layout == gpu::compat::TextureLayout.COLOR_ATTACHMENT, "release source layout is color attachment");
    assert(release.new_layout == gpu::compat::TextureLayout.PRESENT, "release destination layout is present");
}
