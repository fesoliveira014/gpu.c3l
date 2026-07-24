module gpu;

import gpu::internal @public;
import std::io;
import std::time;

const uint WRAPPER_BENCH_ITERATIONS = 20_000;
const uint WRAPPER_BENCH_REPETITIONS = 5;
const ulong DISPATCH_OBSERVATION = 3;
const ulong DRAW_OBSERVATION = 5;
const ulong BARRIER_OBSERVATION = 7;
const ulong VIEWPORT_OBSERVATION = 11;
const ulong COPY_BUFFER_OBSERVATION = 13;

struct WrapperBenchState {
    ulong                        resource_owner;
    ulong                        command_marker;
    gpu::internal::CommandRecord record;
}

struct WrapperBenchSamples {
    double[WRAPPER_BENCH_REPETITIONS] direct;
    double[WRAPPER_BENCH_REPETITIONS] public;
}

ulong wrapper_bench_observation @private;

fn void observe(ulong value) @noinline {
    ulong current = @volatile_load(wrapper_bench_observation);
    @volatile_store(wrapper_bench_observation, current + value);
}

fn void observe_dispatch() @noinline => observe(DISPATCH_OBSERVATION);

fn void observe_draw() @noinline => observe(DRAW_OBSERVATION);

fn void observe_barrier() @noinline => observe(BARRIER_OBSERVATION);

fn void observe_viewport() @noinline => observe(VIEWPORT_OBSERVATION);

fn void observe_copy_buffer() @noinline => observe(COPY_BUFFER_OBSERVATION);

fn void destroy_wrapper_bench_device(void* backend_state) {
    (void)backend_state;
}

fn CommandAllocatorHandle? create_wrapper_bench_allocator(
    Device* device,
    Queue queue,
    CommandAllocatorDesc* desc,
) {
    (void)queue;
    (void)desc;
    WrapperBenchState* state = (WrapperBenchState*)gpu::internal::device_backend_state_ptr(
        device,
    );
    return {
        .owner      = gpu::internal::resource_owner_tag(
            state.resource_owner,
            gpu::internal::ResourceOwnerKind.COMMAND_ALLOCATOR,
        ),
        .index      = 0,
        .generation = 1,
    };
}

fn void? destroy_wrapper_bench_allocator(
    Device* device,
    CommandAllocatorHandle handle,
) {
    (void)device;
    (void)handle;
}

fn void? begin_wrapper_bench_commands(
    CommandAllocator* allocator,
    CommandList* out_commands,
) {
    WrapperBenchState* state = (WrapperBenchState*)gpu::internal::device_backend_state_ptr(
        &allocator.device,
    );
    state.record = {
        .ops             = &wrapper_bench_command_ops,
        .backend_state   = state,
        .backend_command = &state.command_marker,
        .device          = allocator.device,
        .identity        = {
            .owner      = gpu::internal::resource_owner_tag(
                state.resource_owner,
                gpu::internal::ResourceOwnerKind.COMMAND,
            ),
            .index      = 0,
            .generation = 1,
        },
        .state = gpu::internal::CommandState.RECORDING,
    };
    gpu::internal::publish_command_list(&state.record, out_commands);
}

fn void? end_wrapper_bench_commands(
    gpu::internal::CommandRecord* record,
) {
    if (record.state != gpu::internal::CommandState.RECORDING) {
        return COMMAND_RECORDING_ERROR~;
    }
    record.state = gpu::internal::CommandState.EXECUTABLE;
}

fn void? discard_wrapper_bench_commands(
    gpu::internal::CommandRecord* record,
    bool executable,
) {
    (void)executable;
    record.state = gpu::internal::CommandState.INACTIVE;
}

fn void? wrapper_bench_copy_buffer(
    gpu::internal::CommandRecord* record,
    BufferCopyDesc* desc,
) {
    (void)record;
    (void)desc;
    observe_copy_buffer();
}

fn void? wrapper_bench_barrier(
    gpu::internal::CommandRecord* record,
    Barrier* barrier,
) {
    (void)record;
    (void)barrier;
    observe_barrier();
}

fn void? wrapper_bench_dispatch(
    gpu::internal::CommandRecord* record,
    GpuAddress root,
    Vec3u groups,
) {
    (void)record;
    (void)root;
    (void)groups;
    observe_dispatch();
}

fn void? wrapper_bench_viewport(
    gpu::internal::CommandRecord* record,
    Viewport* viewport,
) {
    (void)record;
    (void)viewport;
    observe_viewport();
}

fn void? wrapper_bench_draw(
    gpu::internal::CommandRecord* record,
    GpuAddress vertex_root,
    GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
) {
    (void)record;
    (void)vertex_root;
    (void)fragment_root;
    (void)vertex_count;
    (void)instance_count;
    observe_draw();
}

gpu::internal::CommandOps wrapper_bench_command_ops @private = {
    .copy_buffer  = &wrapper_bench_copy_buffer,
    .barrier      = &wrapper_bench_barrier,
    .dispatch     = &wrapper_bench_dispatch,
    .set_viewport = &wrapper_bench_viewport,
    .draw         = &wrapper_bench_draw,
};

gpu::internal::BackendVTable wrapper_bench_vtable @private = {
    .destroy_device            = &destroy_wrapper_bench_device,
    .create_command_allocator  = &create_wrapper_bench_allocator,
    .destroy_command_allocator = &destroy_wrapper_bench_allocator,
    .begin_commands            = &begin_wrapper_bench_commands,
    .end_commands              = &end_wrapper_bench_commands,
    .discard_commands          = &discard_wrapper_bench_commands,
    .command_ops               = &wrapper_bench_command_ops,
};

fn double sample_elapsed(std::time::Clock start) {
    return (double)(long)start.to_now() / (double)WRAPPER_BENCH_ITERATIONS;
}

fn double sample_min(double[WRAPPER_BENCH_REPETITIONS] values) {
    double result = values[0];
    for (uint index = 1; index < WRAPPER_BENCH_REPETITIONS; index++) {
        if (values[index] < result) result = values[index];
    }
    return result;
}

fn double sample_max(double[WRAPPER_BENCH_REPETITIONS] values) {
    double result = values[0];
    for (uint index = 1; index < WRAPPER_BENCH_REPETITIONS; index++) {
        if (values[index] > result) result = values[index];
    }
    return result;
}

fn double sample_median(double[WRAPPER_BENCH_REPETITIONS] values) {
    double[WRAPPER_BENCH_REPETITIONS] sorted = values;
    for (uint index = 1; index < WRAPPER_BENCH_REPETITIONS; index++) {
        double key = sorted[index];
        int previous = (int)index - 1;
        while (previous >= 0 && sorted[previous] > key) {
            sorted[previous + 1] = sorted[previous];
            previous--;
        }
        sorted[previous + 1] = key;
    }
    return sorted[WRAPPER_BENCH_REPETITIONS / 2];
}

fn void check_sample_observation(ulong before, ulong value) {
    ulong expected = before
        + (ulong)WRAPPER_BENCH_ITERATIONS * 2 * value;
    assert(
        @volatile_load(wrapper_bench_observation) == expected,
        "one paired sample must observe every direct and public invocation"
    );
}

fn void measure_dispatch(
    CommandList* commands,
    WrapperBenchSamples* samples,
) {
    Vec3u groups = { 1, 1, 1 };
    for (uint repetition = 0; repetition < WRAPPER_BENCH_REPETITIONS; repetition++) {
        ulong observation_before = @volatile_load(wrapper_bench_observation);
        if ((repetition & 1) == 0) {
            std::time::Clock direct_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                observe_dispatch();
            }
            samples.direct[repetition] = sample_elapsed(direct_start);
            std::time::Clock public_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                cmd_dispatch(commands, (GpuAddress)1, groups)!!;
            }
            samples.public[repetition] = sample_elapsed(public_start);
        } else {
            std::time::Clock public_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                cmd_dispatch(commands, (GpuAddress)1, groups)!!;
            }
            samples.public[repetition] = sample_elapsed(public_start);
            std::time::Clock direct_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                observe_dispatch();
            }
            samples.direct[repetition] = sample_elapsed(direct_start);
        }
        check_sample_observation(observation_before, DISPATCH_OBSERVATION);
    }
}

fn void measure_draw(
    CommandList* commands,
    WrapperBenchSamples* samples,
) {
    for (uint repetition = 0; repetition < WRAPPER_BENCH_REPETITIONS; repetition++) {
        ulong observation_before = @volatile_load(wrapper_bench_observation);
        if ((repetition & 1) == 0) {
            std::time::Clock direct_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                observe_draw();
            }
            samples.direct[repetition] = sample_elapsed(direct_start);
            std::time::Clock public_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                cmd_draw(
                    commands:      commands,
                    vertex_root:   (GpuAddress)1,
                    fragment_root: (GpuAddress)2,
                    vertex_count:  3,
                    instance_count: 1,
                )!!;
            }
            samples.public[repetition] = sample_elapsed(public_start);
        } else {
            std::time::Clock public_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                cmd_draw(
                    commands:      commands,
                    vertex_root:   (GpuAddress)1,
                    fragment_root: (GpuAddress)2,
                    vertex_count:  3,
                    instance_count: 1,
                )!!;
            }
            samples.public[repetition] = sample_elapsed(public_start);
            std::time::Clock direct_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                observe_draw();
            }
            samples.direct[repetition] = sample_elapsed(direct_start);
        }
        check_sample_observation(observation_before, DRAW_OBSERVATION);
    }
}

fn void measure_barrier(
    CommandList* commands,
    WrapperBenchSamples* samples,
) {
    Barrier barrier = {
        .before = { .compute = true },
        .after  = { .compute = true },
    };
    for (uint repetition = 0; repetition < WRAPPER_BENCH_REPETITIONS; repetition++) {
        ulong observation_before = @volatile_load(wrapper_bench_observation);
        if ((repetition & 1) == 0) {
            std::time::Clock direct_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                observe_barrier();
            }
            samples.direct[repetition] = sample_elapsed(direct_start);
            std::time::Clock public_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                cmd_barrier(commands, &barrier)!!;
            }
            samples.public[repetition] = sample_elapsed(public_start);
        } else {
            std::time::Clock public_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                cmd_barrier(commands, &barrier)!!;
            }
            samples.public[repetition] = sample_elapsed(public_start);
            std::time::Clock direct_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                observe_barrier();
            }
            samples.direct[repetition] = sample_elapsed(direct_start);
        }
        check_sample_observation(observation_before, BARRIER_OBSERVATION);
    }
}

fn void measure_viewport(
    CommandList* commands,
    WrapperBenchSamples* samples,
) {
    Viewport viewport = {
        .width     = 1.0f,
        .height    = 1.0f,
        .max_depth = 1.0f,
    };
    for (uint repetition = 0; repetition < WRAPPER_BENCH_REPETITIONS; repetition++) {
        ulong observation_before = @volatile_load(wrapper_bench_observation);
        if ((repetition & 1) == 0) {
            std::time::Clock direct_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                observe_viewport();
            }
            samples.direct[repetition] = sample_elapsed(direct_start);
            std::time::Clock public_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                cmd_set_viewport(commands, &viewport)!!;
            }
            samples.public[repetition] = sample_elapsed(public_start);
        } else {
            std::time::Clock public_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                cmd_set_viewport(commands, &viewport)!!;
            }
            samples.public[repetition] = sample_elapsed(public_start);
            std::time::Clock direct_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                observe_viewport();
            }
            samples.direct[repetition] = sample_elapsed(direct_start);
        }
        check_sample_observation(observation_before, VIEWPORT_OBSERVATION);
    }
}

fn void measure_copy_buffer(
    CommandList* commands,
    WrapperBenchSamples* samples,
) {
    BufferCopyDesc desc = {
        .src = { .owner = 1, .index = 1, .generation = 1, .size = 4 },
        .dst = { .owner = 1, .index = 2, .generation = 1, .size = 4 },
    };
    for (uint repetition = 0; repetition < WRAPPER_BENCH_REPETITIONS; repetition++) {
        ulong observation_before = @volatile_load(wrapper_bench_observation);
        if ((repetition & 1) == 0) {
            std::time::Clock direct_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                observe_copy_buffer();
            }
            samples.direct[repetition] = sample_elapsed(direct_start);
            std::time::Clock public_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                cmd_copy_buffer(commands, &desc)!!;
            }
            samples.public[repetition] = sample_elapsed(public_start);
        } else {
            std::time::Clock public_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                cmd_copy_buffer(commands, &desc)!!;
            }
            samples.public[repetition] = sample_elapsed(public_start);
            std::time::Clock direct_start = std::time::clock::now();
            for (uint index = 0; index < WRAPPER_BENCH_ITERATIONS; index++) {
                observe_copy_buffer();
            }
            samples.direct[repetition] = sample_elapsed(direct_start);
        }
        check_sample_observation(observation_before, COPY_BUFFER_OBSERVATION);
    }
}

fn void print_samples(
    String operation,
    WrapperBenchSamples* samples,
    ulong observation_value,
) {
    double direct_median = sample_median(samples.direct);
    double public_median = sample_median(samples.public);
    ulong checksum = (ulong)WRAPPER_BENCH_ITERATIONS
        * (ulong)WRAPPER_BENCH_REPETITIONS * 2 * observation_value;
    io::printfn(
        "command_path_cpu operation=%s iterations=%d repetitions=%d checksum=%d direct_min_ns=%.6f direct_median_ns=%.6f direct_max_ns=%.6f public_min_ns=%.6f public_median_ns=%.6f public_max_ns=%.6f ratio=%.6f",
        operation,
        WRAPPER_BENCH_ITERATIONS,
        WRAPPER_BENCH_REPETITIONS,
        checksum,
        sample_min(samples.direct),
        direct_median,
        sample_max(samples.direct),
        sample_min(samples.public),
        public_median,
        sample_max(samples.public),
        public_median / direct_median,
    );
}

fn int main() {
    WrapperBenchState state;
    state.resource_owner = gpu::internal::allocate_device_resource_owner()!!;
    Device device = gpu::internal::register_device(
        backend:        BackendKind.VULKAN,
        caps:           {},
        vtable:         &wrapper_bench_vtable,
        backend_state:  &state,
        command_ops:    &wrapper_bench_command_ops,
        resource_owner: state.resource_owner,
    )!!;
    defer destroy_device(&device)!!;
    Queue queue = get_queue(&device, QueueKind.GRAPHICS)!!;
    CommandAllocator allocator = create_command_allocator(&device, queue)!!;
    defer destroy_command_allocator(&allocator)!!;
    CommandList commands = begin_commands(&allocator)!!;
    defer discard_commands(&commands)!!;

    WrapperBenchSamples dispatch;
    WrapperBenchSamples draw;
    WrapperBenchSamples barrier;
    WrapperBenchSamples viewport;
    WrapperBenchSamples copy_buffer;
    @volatile_store(wrapper_bench_observation, (ulong)0);
    measure_dispatch(&commands, &dispatch);
    measure_draw(&commands, &draw);
    measure_barrier(&commands, &barrier);
    measure_viewport(&commands, &viewport);
    measure_copy_buffer(&commands, &copy_buffer);

    ulong invocations = (ulong)WRAPPER_BENCH_ITERATIONS
        * (ulong)WRAPPER_BENCH_REPETITIONS * 2;
    ulong expected = invocations * (
        DISPATCH_OBSERVATION
        + DRAW_OBSERVATION
        + BARRIER_OBSERVATION
        + VIEWPORT_OBSERVATION
        + COPY_BUFFER_OBSERVATION
    );
    ulong observed = @volatile_load(wrapper_bench_observation);
    assert(observed == expected, "every direct and public invocation is observed");

    print_samples("dispatch", &dispatch, DISPATCH_OBSERVATION);
    print_samples("draw", &draw, DRAW_OBSERVATION);
    print_samples("barrier", &barrier, BARRIER_OBSERVATION);
    print_samples("viewport", &viewport, VIEWPORT_OBSERVATION);
    print_samples("copy_buffer", &copy_buffer, COPY_BUFFER_OBSERVATION);
    io::printfn(
        "command_path_cpu_check operations=5 observed=%d expected=%d status=pass",
        observed,
        expected,
    );
    return 0;
}
