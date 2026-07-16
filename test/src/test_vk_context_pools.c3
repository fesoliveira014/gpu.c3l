module gpu_test;

import gpu;
import gpu::vk @public;
import vk;

const uint CONTEXT_POOL_STAGE_CAP = 6;

const uint FIXTURE_GRAPHICS_FAMILY = 3;
const uint FIXTURE_COMPUTE_FAMILY = 5;
const uint FIXTURE_TRANSFER_FAMILY = 7;

struct ContextPoolFixture {
    uint fail_at;
    uint create_calls;
    uint destroy_calls;
    uint[CONTEXT_POOL_STAGE_CAP] destroyed;
    uint[CONTEXT_POOL_STAGE_CAP] families;
}

fn vk::CommandPool? fake_create_context_pool(void* user_data, vk::Device device, uint family) {
    ContextPoolFixture* fixture = (ContextPoolFixture*)user_data;
    uint ordinal = fixture.create_calls++;
    fixture.families[ordinal] = family;
    if (ordinal == fixture.fail_at) return gpu::OUT_OF_HOST_MEMORY~;
    return (vk::CommandPool)(uptr)(ordinal + 1);
}

fn void fake_destroy_context_pool(void* user_data, vk::Device device, vk::CommandPool pool) {
    ContextPoolFixture* fixture = (ContextPoolFixture*)user_data;
    uint ordinal = (uint)(uptr)pool - 1;
    fixture.destroyed[ordinal]++;
    fixture.destroy_calls++;
}

fn gpu::vk::ContextPoolOps context_pool_ops(ContextPoolFixture* fixture) {
    return {
        .user_data = fixture,
        .create = &fake_create_context_pool,
        .destroy = &fake_destroy_context_pool,
    };
}

fn uint expected_pool_family(bool distinct, uint ordinal) {
    if (distinct) {
        switch (ordinal % 3) {
            case 0: return FIXTURE_GRAPHICS_FAMILY;
            case 1: return FIXTURE_COMPUTE_FAMILY;
            default: return FIXTURE_TRANSFER_FAMILY;
        }
    }
    return ordinal % 2 == 0 ? FIXTURE_GRAPHICS_FAMILY : FIXTURE_TRANSFER_FAMILY;
}

fn void assert_context_empty(gpu::vk::RecordingContextState* context) {
    assert(context.graphics_pools.ptr == null, "graphics array released");
    assert(context.compute_pools.ptr == null, "compute array or alias cleared");
    assert(context.transfer_pools.ptr == null, "transfer array released");
    assert(context.pool_dirty.ptr == null, "dirty-mask array released");
    assert(!context.compute_pools_distinct, "ownership flag cleared");
}

fn void run_context_pool_failure(bool distinct, uint fail_at, uint stage_count) {
    gpu::vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute, .transfer },
        .frames_in_flight = 2,
        .graphics_family = FIXTURE_GRAPHICS_FAMILY,
        .compute_family = distinct ? FIXTURE_COMPUTE_FAMILY : FIXTURE_GRAPHICS_FAMILY,
        .transfer_family = FIXTURE_TRANSFER_FAMILY,
    };
    ContextPoolFixture fixture = { .fail_at = fail_at };
    gpu::vk::ContextPoolOps ops = context_pool_ops(&fixture);
    gpu::vk::RecordingContextState context;

    fault create_fault = @catch(gpu::vk::create_context_pools_with_ops(&state, &context, &ops));
    assert(create_fault == gpu::OUT_OF_HOST_MEMORY, "original create fault propagates");
    assert(fixture.create_calls == fail_at + 1, "creation stops at injected stage");
    assert(fixture.destroy_calls == fail_at, "every prior pool is destroyed once");
    for (uint i = 0; i < fixture.create_calls; i++) {
        assert(fixture.families[i] == expected_pool_family(distinct, i), "pool creation follows family order");
    }
    for (uint i = 0; i < stage_count; i++) {
        uint expected = i < fail_at ? 1 : 0;
        assert(fixture.destroyed[i] == expected, "only successful handles are destroyed");
    }
    assert_context_empty(&context);

    gpu::vk::destroy_context_pools_with_ops(&state, &context, &ops);
    assert(fixture.destroy_calls == fail_at, "second cleanup is a no-op");
}

fn void run_context_pool_success(bool distinct, uint stage_count) {
    gpu::vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute, .transfer },
        .frames_in_flight = 2,
        .graphics_family = FIXTURE_GRAPHICS_FAMILY,
        .compute_family = distinct ? FIXTURE_COMPUTE_FAMILY : FIXTURE_GRAPHICS_FAMILY,
        .transfer_family = FIXTURE_TRANSFER_FAMILY,
    };
    ContextPoolFixture fixture = { .fail_at = uint::max };
    gpu::vk::ContextPoolOps ops = context_pool_ops(&fixture);
    gpu::vk::RecordingContextState context;

    gpu::vk::create_context_pools_with_ops(&state, &context, &ops)!!;
    assert(fixture.create_calls == stage_count, "all topology stages are created");
    assert(context.compute_pools_distinct == distinct, "compute ownership matches topology");
    if (distinct) {
        assert(context.compute_pools.ptr != context.graphics_pools.ptr, "distinct compute owns separate storage");
    } else {
        assert(
            context.compute_pools.ptr == context.graphics_pools.ptr,
            "shared compute aliases graphics after success"
        );
    }
    for (uint i = 0; i < stage_count; i++) {
        assert(fixture.families[i] == expected_pool_family(distinct, i), "successful creation follows family order");
    }

    gpu::vk::destroy_context_pools_with_ops(&state, &context, &ops);
    assert(fixture.destroy_calls == stage_count, "successful pools are destroyed once");
    for (uint i = 0; i < stage_count; i++) {
        assert(fixture.destroyed[i] == 1, "each successful handle is destroyed exactly once");
    }
    assert_context_empty(&context);

    gpu::vk::destroy_context_pools_with_ops(&state, &context, &ops);
    assert(fixture.destroy_calls == stage_count, "repeated success cleanup is a no-op");
}

fn void run_single_role_context_pool_success(
    gpu::QueueRoles selected,
    uint expected_family,
) {
    gpu::vk::VkDeviceState state = {
        .selected_queue_roles = selected,
        .frames_in_flight      = 2,
        .graphics_family      = FIXTURE_GRAPHICS_FAMILY,
        .compute_family       = FIXTURE_COMPUTE_FAMILY,
        .transfer_family      = FIXTURE_TRANSFER_FAMILY,
    };
    ContextPoolFixture fixture = { .fail_at = uint::max };
    gpu::vk::ContextPoolOps ops = context_pool_ops(&fixture);
    gpu::vk::RecordingContextState context;

    gpu::vk::create_context_pools_with_ops(&state, &context, &ops)!!;
    assert(fixture.create_calls == 2, "only the admitted role owns pools");
    assert(
        fixture.families[0] == expected_family
            && fixture.families[1] == expected_family,
        "pool creation uses only the admitted family"
    );
    assert(
        (context.graphics_pools.ptr != null) == selected.graphics,
        "graphics pool presence follows the request"
    );
    assert(
        (context.compute_pools.ptr != null) == selected.compute,
        "compute pool presence follows the request"
    );
    assert(
        (context.transfer_pools.ptr != null) == selected.transfer,
        "transfer pool presence follows the request"
    );

    gpu::vk::destroy_context_pools_with_ops(&state, &context, &ops);
    assert(fixture.destroy_calls == 2, "only admitted pools are destroyed");
    assert_context_empty(&context);
}

fn void graphics_only_context_pools_initialize_one_family() @test {
    run_single_role_context_pool_success({ .graphics }, FIXTURE_GRAPHICS_FAMILY);
}

fn void compute_only_context_pools_initialize_one_family() @test {
    run_single_role_context_pool_success({ .compute }, FIXTURE_COMPUTE_FAMILY);
}

fn void transfer_only_context_pools_initialize_one_family() @test {
    run_single_role_context_pool_success({ .transfer }, FIXTURE_TRANSFER_FAMILY);
}

fn void shared_context_pool_failures_are_transactional() @test {
    for (uint stage = 0; stage < 4; stage++) run_context_pool_failure(false, stage, 4);
}

fn void distinct_context_pool_failures_are_transactional() @test {
    for (uint stage = 0; stage < 6; stage++) run_context_pool_failure(true, stage, 6);
}

fn void successful_context_pool_ownership_is_idempotent() @test {
    run_context_pool_success(false, 4);
    run_context_pool_success(true, 6);
}

fn void built_in_context_insertion_failure_cleans_full_context() @test {
    gpu::vk::VkDeviceState state = {
        .selected_queue_roles = { .graphics, .compute, .transfer },
        .frames_in_flight = 2,
        .graphics_family = FIXTURE_GRAPHICS_FAMILY,
        .compute_family = FIXTURE_GRAPHICS_FAMILY,
        .transfer_family = FIXTURE_TRANSFER_FAMILY,
    };
    ContextPoolFixture fixture = { .fail_at = uint::max };
    gpu::vk::ContextPoolOps ops = context_pool_ops(&fixture);

    fault insert_fault = @catch(gpu::vk::create_builtin_recording_context_with_ops(&state, &ops));
    assert(insert_fault == gpu::SLOT_TABLE_FULL, "built-in insertion fault propagates");
    assert(fixture.create_calls == 4, "full shared pool set was constructed");
    assert(fixture.destroy_calls == 4, "full local pool set is destroyed on insertion failure");
    for (uint i = 0; i < 4; i++) {
        assert(fixture.destroyed[i] == 1, "each unpublished pool is destroyed exactly once");
    }
    assert(state.recording_contexts.next_new == 0, "no recording-context handle is published");
}
