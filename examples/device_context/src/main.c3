module context_example;

import gpu;
import gpu::util;
import std::io;

const uint VALUE_COUNT = 16;
const uint FILL_VALUE = 0x13579bdf;

faultdef READBACK_MISMATCH;

fn int main() {
    if (catch err = run()) {
        (void)io::eprintfn("device_context: %s", err);
        return 1;
    }
    io::printn("device_context: GPU fill and readback succeeded");
    return 0;
}

fn void? run() {
    gpu::util::DeviceContextDesc desc = gpu::util::default_device_context_desc();
    desc.runtime.application_name = "device_context";
    desc.device.queues.required = { .compute };
    desc.command_queue = gpu::QueueKind.COMPUTE;

    gpu::util::DeviceContext context = gpu::util::create_device_context(&desc)!;
    defer try gpu::util::destroy_device_context(&context);

    gpu::AllocationDesc allocation_desc = {
        .size = VALUE_COUNT * uint::size,
        .alignment = uint::alignment,
        .memory_class = gpu::MemoryClass.CPU_READ,
        .access = { .compute },
        .debug_name = "readback",
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(&context.device, &allocation_desc)!;
    defer try (void)gpu::free_allocation(&context.device, &allocation);
    gpu::GpuSpan span = gpu::get_allocation_span(&context.device, allocation)!;

    gpu::CommandList commands = gpu::begin_commands(&context.command_allocator)!;
    defer (void)gpu::discard_commands(&commands);
    gpu::cmd_fill_buffer(&commands, span, FILL_VALUE)!;
    gpu::Barrier readable = { .before = { .transfer }, .after = { .host } };
    gpu::cmd_barrier(&commands, &readable)!;

    gpu::ExecutableCommandList[1] lists = { gpu::end_commands(&commands)! };
    defer (void)gpu::discard_executable_commands(&lists[0]);
    gpu::SubmitDesc submission = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(context.queue, &submission)!;
    gpu::wait_completion(completion)!;

    gpu::invalidate_mapped_span(&context.device, span)!;
    uint* values = (uint*)gpu::get_span_mapping(&context.device, span)!.ptr;
    for (uint index = 0; index < VALUE_COUNT; index++) {
        if (values[index] != FILL_VALUE) return READBACK_MISMATCH~;
    }
}
