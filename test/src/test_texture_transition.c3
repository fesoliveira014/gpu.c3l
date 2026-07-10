module gpu_test;

import gpu;

struct TextureUseCase {
    gpu::TextureUse    use;
    gpu::Stage         stage;
    gpu::Hazard        hazard;
    gpu::TextureLayout layout;
}

fn gpu::TextureHandle transition_texture() {
    return { .index = 0, .generation = 1 };
}

fn void texture_use_mappings_are_exact() @test {
    TextureUseCase[8] cases = {
        { gpu::TextureUse.UNDEFINED,            gpu::Stage.HOST,             gpu::Hazard.NONE,              gpu::TextureLayout.UNDEFINED },
        { gpu::TextureUse.TRANSFER_DESTINATION, gpu::Stage.TRANSFER,         gpu::Hazard.TRANSFER_WRITE,    gpu::TextureLayout.TRANSFER_DST },
        { gpu::TextureUse.SAMPLED_COMPUTE,      gpu::Stage.COMPUTE_SHADER,   gpu::Hazard.SHADER_READ,       gpu::TextureLayout.SHADER_READ },
        { gpu::TextureUse.SAMPLED_FRAGMENT,     gpu::Stage.FRAGMENT_SHADER,  gpu::Hazard.SHADER_READ,       gpu::TextureLayout.SHADER_READ },
        { gpu::TextureUse.STORAGE_COMPUTE,      gpu::Stage.COMPUTE_SHADER,   gpu::Hazard.SHADER_READ_WRITE, gpu::TextureLayout.GENERAL },
        { gpu::TextureUse.COLOR_ATTACHMENT,     gpu::Stage.COLOR_ATTACHMENT, gpu::Hazard.COLOR_READ_WRITE,  gpu::TextureLayout.COLOR_ATTACHMENT },
        { gpu::TextureUse.DEPTH_ATTACHMENT,     gpu::Stage.DEPTH_STENCIL,    gpu::Hazard.DEPTH_READ_WRITE,  gpu::TextureLayout.DEPTH_STENCIL },
        { gpu::TextureUse.PRESENT,              gpu::Stage.PRESENT,          gpu::Hazard.PRESENT_READ,      gpu::TextureLayout.PRESENT },
    };
    gpu::TextureHandle texture = transition_texture();

    foreach (&expected : cases) {
        gpu::TextureUse after = expected.use == gpu::TextureUse.UNDEFINED
            ? gpu::TextureUse.TRANSFER_DESTINATION
            : expected.use;
        gpu::TextureBarrier source = gpu::texture_transition(
            texture: texture,
            before:  expected.use,
            after:   after,
        )!!;
        assert(source.texture == texture, "transition preserves the texture handle");
        assert(source.before_stage == expected.stage, "source stage matches the preset");
        assert(source.before_hazard == expected.hazard, "source hazard matches the preset");
        assert(source.old_layout == expected.layout, "source layout matches the preset");

        if (expected.use != gpu::TextureUse.UNDEFINED) {
            gpu::TextureBarrier destination = gpu::texture_transition(
                texture: texture,
                before:  gpu::TextureUse.UNDEFINED,
                after:   expected.use,
            )!!;
            assert(destination.after_stage == expected.stage, "destination stage matches the preset");
            assert(destination.after_hazard == expected.hazard, "destination hazard matches the preset");
            assert(destination.new_layout == expected.layout, "destination layout matches the preset");
        }
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

fn void storage_same_use_transition_keeps_combined_access() @test {
    gpu::TextureBarrier barrier = gpu::texture_transition(
        texture: transition_texture(),
        before:  gpu::TextureUse.STORAGE_COMPUTE,
        after:   gpu::TextureUse.STORAGE_COMPUTE,
    )!!;
    assert(barrier.before_hazard == gpu::Hazard.SHADER_READ_WRITE,
        "storage source includes shader reads and writes");
    assert(barrier.after_hazard == gpu::Hazard.SHADER_READ_WRITE,
        "storage destination includes shader reads and writes");
    assert(barrier.old_layout == gpu::TextureLayout.GENERAL
        && barrier.new_layout == gpu::TextureLayout.GENERAL,
        "same-use storage barrier preserves general layout");
}
