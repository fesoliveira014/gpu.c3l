module gpu_bench;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import vk;
import std::atomic;
import std::io;
import std::thread;
import std::time;
import std::core::mem::allocators;

const uint[3] UPLOAD_BENCH_WORKER_COUNTS = { 1, 2, 4 };
const usz[3] UPLOAD_BENCH_PAYLOAD_SIZES = { 4 * 1024, 256 * 1024, 4 * 1024 * 1024 };
const uint UPLOAD_BENCH_MAX_WORKERS = 4;
const uint UPLOAD_BENCH_WARMUP_ITERATIONS = 1;
const uint[3] UPLOAD_BENCH_ITERATIONS = { 2048, 512, 32 };
const usz UPLOAD_BENCH_THREAD_POOL_SIZE = 64 * 1024;

struct UploadStorage {
    gpu::GpuAllocation allocation;
    gpu::GpuSpan       span;
}

struct UploadWorkerArgs {
    gpu::Device*               device;
    gpu::CommandAllocator*     allocator;
    UploadStorage*             upload;
    UploadStorage*             destination;
    Atomic{uint}*              requested_generation;
    Atomic{uint}               completed_generation;
    Atomic{bool}*              stop;
    uint                       worker_index;
    gpu::ExecutableCommandList executable;
}

fn UploadStorage? create_upload_storage(
    gpu::Device* device,
    usz size,
    gpu::MemoryClass memory_class,
    ZString debug_name,
) {
    gpu::AllocationDesc desc = {
        .size         = size,
        .alignment    = 16,
        .memory_class = memory_class,
        .access       = { .transfer },
        .debug_name   = debug_name,
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(device, &desc)!;
    defer catch (void)gpu::free_allocation(device, &allocation);
    gpu::GpuSpan span = gpu::get_allocation_span(device, allocation)!;
    return { .allocation = allocation, .span = span };
}

fn void? free_upload_storage(gpu::Device* device, UploadStorage* storage) {
    if (storage == null || !storage.allocation.is_valid()) return;
    gpu::free_allocation(device, &storage.allocation)!;
    storage.span = {};
}

fn void print_upload_adapter(gpu::Device* device) {
    vk::VkDeviceState* state = gpu::internal::device_slots[device.index].data.state;
    vk::PhysicalDeviceDriverProperties driver = vk::physical_device_driver_properties();
    vk::PhysicalDeviceProperties2 properties = vk::physical_device_properties2().set_next(&driver);
    vk::get_physical_device_properties2(state.physical_device, &properties);
    io::printfn(
        "adapter_type=%s driver_id=%d lavapipe=%s",
        upload_device_type_name(properties.properties.device_type),
        (uint)driver.driver_id,
        upload_is_lavapipe_driver(driver.driver_id) ? "true" : "false",
    );
}

fn void record_upload_generation(UploadWorkerArgs* args, uint generation) {
    char[] mapping = gpu::get_span_mapping(args.device, args.upload.span)!!;
    for (usz i = 0; i < mapping.len; i++) {
        mapping[i] = (char)((i
            + (usz)args.worker_index * 31
            + (usz)generation * 17) % 251);
    }
    gpu::flush_mapped_span(args.device, args.upload.span)!!;

    gpu::CommandList commands = gpu::begin_commands(args.allocator)!!;
    defer (void)gpu::discard_commands(&commands);
    gpu::BufferCopyDesc copy = {
        .src = args.upload.span,
        .dst = args.destination.span,
    };
    gpu::cmd_copy_buffer(&commands, &copy)!!;
    args.executable = gpu::end_commands(&commands)!!;
}

fn int record_upload(void* raw_args) {
    UploadWorkerArgs* args = (UploadWorkerArgs*)raw_args;
    @pool_init(&allocators::LIBC_ALLOCATOR, UPLOAD_BENCH_THREAD_POOL_SIZE) {
        uint observed_generation = 0;
        while (true) {
            uint generation = args.requested_generation.load(AtomicOrdering.ACQUIRE);
            while (generation == observed_generation) {
                if (args.stop.load(AtomicOrdering.ACQUIRE)) return 0;
                thread::yield();
                generation = args.requested_generation.load(AtomicOrdering.ACQUIRE);
            }
            @pool() {
                record_upload_generation(args, generation);
            };
            observed_generation = generation;
            args.completed_generation.store(generation, AtomicOrdering.RELEASE);
        }
    };
}

<* Publishes one recording generation, submits every worker list, and waits
   before the caller may reuse either allocation set. *>
fn void run_upload_iteration(
    gpu::Device* device,
    UploadWorkerArgs[] args,
    uint workers,
    Atomic{uint}* requested_generation,
    uint generation,
) {
    requested_generation.store(generation, AtomicOrdering.RELEASE);
    for (uint worker = 0; worker < workers; worker++) {
        while (args[worker].completed_generation.load(AtomicOrdering.ACQUIRE)
            != generation) {
            thread::yield();
        }
    }

    gpu::ExecutableCommandList[UPLOAD_BENCH_MAX_WORKERS] lists;
    for (uint worker = 0; worker < workers; worker++) {
        lists[worker] = args[worker].executable;
    }
    gpu::SubmitDesc submit = { .command_lists = lists[:workers] };
    gpu::CompletionPoint completion = submit_bench_queue(
        device,
        gpu::QueueKind.TRANSFER,
        &submit,
    )!!;
    gpu::wait_completion(completion)!!;
}

<* Reuses each caller-owned source and destination only after the prior
   covering completion point has completed. *>
fn double run_upload_combo(
    gpu::Device* device,
    uint workers,
    usz payload_size,
    uint measured_iterations,
) {
    UploadStorage[UPLOAD_BENCH_MAX_WORKERS] uploads = {};
    UploadStorage[UPLOAD_BENCH_MAX_WORKERS] destinations = {};
    defer {
        for (uint i = 0; i < workers; i++) {
            (void)free_upload_storage(device, &destinations[i]);
            (void)free_upload_storage(device, &uploads[i]);
        }
    }

    for (uint i = 0; i < workers; i++) {
        uploads[i] = create_upload_storage(
            device:       device,
            size:         payload_size,
            memory_class: gpu::MemoryClass.CPU_WRITE,
            debug_name:   "upload_bench_source",
        )!!;
        destinations[i] = create_upload_storage(
            device:       device,
            size:         payload_size,
            memory_class: gpu::MemoryClass.GPU_PRIVATE,
            debug_name:   "upload_bench_destination",
        )!!;
    }

    Atomic{uint} requested_generation;
    Atomic{bool} stop;
    requested_generation.store(0, AtomicOrdering.RELAXED);
    stop.store(false, AtomicOrdering.RELAXED);

    UploadWorkerArgs[UPLOAD_BENCH_MAX_WORKERS] args;
    gpu::CommandAllocator[UPLOAD_BENCH_MAX_WORKERS] command_allocators;
    gpu::Queue transfer_queue = gpu::get_queue(
        device,
        gpu::QueueKind.TRANSFER,
    )!!;
    for (uint worker = 0; worker < workers; worker++) {
        command_allocators[worker] = gpu::create_command_allocator(
            device,
            transfer_queue,
        )!!;
    }
    defer {
        for (uint worker = 0; worker < workers; worker++) {
            (void)gpu::destroy_command_allocator(&command_allocators[worker]);
        }
    }
    Thread[UPLOAD_BENCH_MAX_WORKERS] threads;
    for (uint worker = 0; worker < workers; worker++) {
        args[worker] = {
            .device               = device,
            .allocator            = &command_allocators[worker],
            .upload               = &uploads[worker],
            .destination          = &destinations[worker],
            .requested_generation = &requested_generation,
            .stop                 = &stop,
            .worker_index         = worker,
        };
        args[worker].completed_generation.store(0, AtomicOrdering.RELAXED);
        threads[worker].create(&record_upload, &args[worker])!!;
    }

    uint generation = 0;
    for (uint warmup = 0; warmup < UPLOAD_BENCH_WARMUP_ITERATIONS; warmup++) {
        generation++;
        run_upload_iteration(
            device:               device,
            args:                 args[:workers],
            workers:              workers,
            requested_generation: &requested_generation,
            generation:           generation,
        );
    }

    std::time::Clock start = std::time::clock::now();
    for (uint iteration = 0; iteration < measured_iterations; iteration++) {
        generation++;
        run_upload_iteration(
            device:               device,
            args:                 args[:workers],
            workers:              workers,
            requested_generation: &requested_generation,
            generation:           generation,
        );
    }
    long elapsed_ns = (long)start.to_now();

    stop.store(true, AtomicOrdering.RELEASE);
    for (uint worker = 0; worker < workers; worker++) {
        assert(threads[worker].join()!! == 0, "upload worker exits after the final completion");
    }

    uint total_uploads = workers * measured_iterations;
    return (double)total_uploads / ((double)elapsed_ns / 1.0e9);
}

fn int main() {
    io::printfn(
        "iterations=payload-specific warmup=%d units=uploads/s",
        UPLOAD_BENCH_WARMUP_ITERATIONS,
    );
    io::printfn(
        "config: workers=[%d,%d,%d] payload_bytes=[%d,%d,%d] measured_iterations=[%d,%d,%d]",
        UPLOAD_BENCH_WORKER_COUNTS[0],
        UPLOAD_BENCH_WORKER_COUNTS[1],
        UPLOAD_BENCH_WORKER_COUNTS[2],
        UPLOAD_BENCH_PAYLOAD_SIZES[0],
        UPLOAD_BENCH_PAYLOAD_SIZES[1],
        UPLOAD_BENCH_PAYLOAD_SIZES[2],
        UPLOAD_BENCH_ITERATIONS[0],
        UPLOAD_BENCH_ITERATIONS[1],
        UPLOAD_BENCH_ITERATIONS[2],
    );

    gpu::RuntimeDesc desc = {

        .application_name  = "gpu_upload_throughput_bench",
    };
    gpu::Device device = create_bench_device(&desc)!!;
    defer destroy_bench_device(&device)!!;
    print_upload_adapter(&device);

    foreach (payload_index, payload_size : UPLOAD_BENCH_PAYLOAD_SIZES) {
        uint measured_iterations = UPLOAD_BENCH_ITERATIONS[payload_index];
        double one_worker_rate = 0.0;
        foreach (workers : UPLOAD_BENCH_WORKER_COUNTS) {
            double rate = run_upload_combo(
                device:              &device,
                workers:             workers,
                payload_size:        payload_size,
                measured_iterations: measured_iterations,
            );
            if (workers == 1) one_worker_rate = rate;
            io::printfn(
                "workers=%d payload_bytes=%d iterations=%d uploads_per_sec=%.0f scaling_vs_1_worker=%.2f",
                workers,
                payload_size,
                measured_iterations,
                rate,
                upload_scaling_vs_one(rate, one_worker_rate),
            );
        }
    }
    return 0;
}
