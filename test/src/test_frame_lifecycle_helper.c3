module gpu_test;

import gpu::compat @public;

faultdef
    FRAME_BEGIN_FAILED,
    FRAME_WORK_FAILED,
    FRAME_END_FAILED;

struct FrameFixture {
    uint  begin_calls;
    uint  work_calls;
    uint  end_calls;
    ulong generation;
    bool  fail_begin;
    bool  fail_work;
    bool  fail_end;
}

fn void? fixture_begin(gpu::compat::Device* device, gpu::compat::FrameToken* out_frame) {
    FrameFixture* fixture = (FrameFixture*)gpu::compat::device_backend_state_ptr(device);
    fixture.begin_calls++;
    if (fixture.fail_begin) return FRAME_BEGIN_FAILED~;
    fixture.generation++;
    *out_frame = { .device = *device, .generation = fixture.generation };
}

fn void? fixture_end(gpu::compat::FrameToken* frame) {
    FrameFixture* fixture = (FrameFixture*)gpu::compat::device_backend_state_ptr(&frame.device);
    fixture.end_calls++;
    if (fixture.fail_end) return FRAME_END_FAILED~;
}

fn void? fixture_work(gpu::compat::FrameToken* frame, FrameFixture* fixture) {
    assert(frame.generation == fixture.generation, "worker receives the live frame token");
    fixture.work_calls++;
    if (fixture.fail_work) return FRAME_WORK_FAILED~;
}

fn void? fail_frame_work() {
    return FRAME_WORK_FAILED~;
}

fn void? fixture_rethrowing_work(gpu::compat::FrameToken* frame, FrameFixture* fixture) {
    assert(frame.generation == fixture.generation, "worker receives the live frame token");
    fixture.work_calls++;
    fail_frame_work()!;
}

fn void fixture_destroy(gpu::compat::Device* device) {}

gpu::compat::BackendVTable frame_fixture_vtable = {
    .destroy_device = &fixture_destroy,
    .begin_frame = &fixture_begin,
    .end_frame = &fixture_end,
};

fn gpu::compat::Device fixture_device(FrameFixture* fixture) {
    return gpu::compat::register_device(
        backend:       gpu::compat::BackendKind.VULKAN,
        caps:          {},
        vtable:        &frame_fixture_vtable,
        backend_state: fixture,
    )!!;
}

fn void frame_token_stays_within_two_machine_words() @test {
    assert(gpu::compat::FrameToken::size <= 16, "FrameToken stays within two machine words");
}

fn void frame_token_validity_follows_lifecycle() @test {
    FrameFixture fixture;
    gpu::compat::Device device = fixture_device(&fixture);
    defer gpu::compat::destroy_device(&device)!!;

    gpu::compat::FrameToken frame;
    assert(!frame.is_valid(), "zero frame token is invalid");

    frame = gpu::compat::begin_frame(&device)!!;
    assert(frame.is_valid(), "begun frame token is valid");

    gpu::compat::end_frame(&frame)!!;
    assert(!frame.is_valid(), "ended frame token is invalid");
}

fn void scoped_frame_rejects_null_output_token() @test {
    FrameFixture fixture;
    gpu::compat::Device device = fixture_device(&fixture);
    defer gpu::compat::destroy_device(&device)!!;

    fault result = @catch(gpu::compat::@with_frame(
        (gpu::compat::FrameToken*)null,
        &device,
        fixture_work,
        &fixture,
    ));
    assert(result == gpu::compat::INVALID_HANDLE, "null frame output faults INVALID_HANDLE");
    assert(fixture.begin_calls == 0, "null frame output does not begin a frame");
}

fn void invalid_device_token_is_rejected() @test {
    gpu::compat::Device invalid;
    fault caps_fault = @catch(gpu::compat::get_device_caps(&invalid));
    assert(caps_fault == gpu::compat::INVALID_HANDLE, "zero device token faults INVALID_HANDLE");
}

fn void stale_device_copy_is_rejected() @test {
    FrameFixture fixture;
    gpu::compat::Device device = fixture_device(&fixture);
    gpu::compat::Device stale = device;
    gpu::compat::destroy_device(&device)!!;

    fault stale_fault = @catch(gpu::compat::get_device_caps(&stale));
    assert(stale_fault == gpu::compat::INVALID_HANDLE, "destroyed device alias faults INVALID_HANDLE");

    gpu::compat::Device replacement = fixture_device(&fixture);
    defer gpu::compat::destroy_device(&replacement)!!;
    assert((ulong)replacement != (ulong)stale, "replacement uses a fresh device generation");
    fault reused_fault = @catch(gpu::compat::get_device_caps(&stale));
    assert(reused_fault == gpu::compat::INVALID_HANDLE, "replacement does not revive a stale device alias");
}

fn void one_live_device_is_enforced() @test {
    FrameFixture first_fixture;
    FrameFixture second_fixture;
    gpu::compat::Device first = fixture_device(&first_fixture);
    defer gpu::compat::destroy_device(&first)!!;

    gpu::compat::Device? second = gpu::compat::register_device(
        backend:       gpu::compat::BackendKind.VULKAN,
        caps:          {},
        vtable:        &frame_fixture_vtable,
        backend_state: &second_fixture,
    );
    fault second_fault = @catch(second);
    assert(second_fault == gpu::compat::INVALID_RESOURCE_STATE, "second live device faults INVALID_RESOURCE_STATE");
}

fn void scoped_frame_ends_once_after_success() @test {
    FrameFixture fixture;
    gpu::compat::Device device = fixture_device(&fixture);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::FrameToken frame;

    gpu::compat::@with_frame(&frame, &device, fixture_work, &fixture)!!;

    assert(fixture.begin_calls == 1, "helper begins once");
    assert(fixture.work_calls == 1, "helper invokes the worker once");
    assert(fixture.end_calls == 1, "helper ends once");
    assert((ulong)frame.device == 0 && frame.generation == 0, "successful end consumes the token");
}

fn void scoped_frame_preserves_natural_worker_fault() @test {
    FrameFixture fixture;
    gpu::compat::Device device = fixture_device(&fixture);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::FrameToken frame;

    fault worker_fault = @catch(gpu::compat::@with_frame(&frame, &device, fixture_rethrowing_work, &fixture));

    assert(worker_fault == FRAME_WORK_FAILED, "worker rethrow reaches the helper caller");
    assert(fixture.work_calls == 1 && fixture.end_calls == 1, "worker fault still ends exactly once");
    assert((ulong)frame.device == 0 && frame.generation == 0, "successful cleanup consumes the token");
}

fn void scoped_frame_end_fault_wins_and_retains_retry_token() @test {
    FrameFixture fixture = { .fail_work = true, .fail_end = true };
    gpu::compat::Device device = fixture_device(&fixture);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::FrameToken frame;

    fault end_fault = @catch(gpu::compat::@with_frame(&frame, &device, fixture_work, &fixture));

    assert(end_fault == FRAME_END_FAILED, "end fault takes precedence over the worker fault");
    assert(fixture.end_calls == 1, "helper attempts end exactly once");
    assert(frame.is_valid(), "failed end retains a valid token");
    assert(frame.device == device && frame.generation != 0, "failed end retains the live token");

    fixture.fail_end = false;
    gpu::compat::end_frame(&frame)!!;
    assert(fixture.end_calls == 2, "caller can retry with the retained token");
    assert((ulong)frame.device == 0 && frame.generation == 0, "successful retry consumes the token");
}

fn void scoped_frame_begin_fault_short_circuits() @test {
    FrameFixture fixture = { .fail_begin = true };
    gpu::compat::Device device = fixture_device(&fixture);
    defer gpu::compat::destroy_device(&device)!!;
    gpu::compat::FrameToken frame = { .device = device, .generation = 99 };

    fault begin_fault = @catch(gpu::compat::@with_frame(&frame, &device, fixture_work, &fixture));

    assert(begin_fault == FRAME_BEGIN_FAILED, "begin fault reaches the helper caller");
    assert(fixture.begin_calls == 1, "helper attempts begin once");
    assert(fixture.work_calls == 0 && fixture.end_calls == 0, "failed begin skips worker and end");
    assert((ulong)frame.device == 0 && frame.generation == 0, "failed begin leaves no apparent live token");
}
