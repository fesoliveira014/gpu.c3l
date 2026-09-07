module gpu_test;

import gpu;
import gpu::util;
import std::io;

const uint CONTEXT_FILL_VALUE = 0x13579bdf;
const uint CONTEXT_FILL_COUNT = 16;

struct ContextDiagnostics {
    uint errors;
}

fn void count_context_errors(gpu::DebugMessage* message, void* user_data) {
    ContextDiagnostics* diagnostics = (ContextDiagnostics*)user_data;
    if (message.severity != gpu::DebugMessageSeverity.ERROR) return;
    diagnostics.errors++;
    io::printfn("context validation: %s: %s", message.operation, message.backend_text);
}

fn void device_context_submits_and_reads_back_before_teardown() @test {
    ContextDiagnostics diagnostics;
    gpu::util::DeviceContextDesc desc = gpu::util::default_device_context_desc();
    desc.runtime.application_name = "device_context_test";
    desc.runtime.enable_vulkan_validation = true;
    desc.runtime.debug_callback = &count_context_errors;
    desc.runtime.debug_user_data = &diagnostics;
    desc.device.queues.required = { .compute };
    desc.command_queue = gpu::QueueKind.COMPUTE;

    gpu::util::DeviceContext context = gpu::util::create_device_context(&desc)!!;
    defer gpu::util::destroy_device_context(&context);

    gpu::AllocationDesc allocation_desc = {
        .size = CONTEXT_FILL_COUNT * uint::size,
        .alignment = uint::alignment,
        .memory_class = gpu::MemoryClass.CPU_READ,
        .access = { .compute },
        .debug_name = "context_readback",
    };
    gpu::GpuAllocation allocation = gpu::allocate_memory(&context.device, &allocation_desc)!!;
    defer {
        if (allocation.is_valid()) gpu::free_allocation(&context.device, &allocation)!!;
    }
    gpu::GpuSpan span = gpu::get_allocation_span(&context.device, allocation)!!;

    gpu::CommandList commands = gpu::begin_commands(&context.command_allocator)!!;
    defer (void)gpu::discard_commands(&commands);
    gpu::cmd_fill_buffer(&commands, span, CONTEXT_FILL_VALUE)!!;
    gpu::Barrier readable = { .before = { .transfer }, .after = { .host } };
    gpu::cmd_barrier(&commands, &readable)!!;

    gpu::ExecutableCommandList[1] lists = { gpu::end_commands(&commands)!! };
    defer (void)gpu::discard_executable_commands(&lists[0]);
    gpu::SubmitDesc submission = { .command_lists = lists[..] };
    gpu::CompletionPoint completion = gpu::submit(context.queue, &submission)!!;
    gpu::wait_completion(completion)!!;

    gpu::invalidate_mapped_span(&context.device, span)!!;
    uint* values = (uint*)gpu::get_span_mapping(&context.device, span)!!.ptr;
    for (uint index = 0; index < CONTEXT_FILL_COUNT; index++) {
        assert(values[index] == CONTEXT_FILL_VALUE);
    }

    gpu::free_allocation(&context.device, &allocation)!!;
    gpu::Device old_device = context.device;
    gpu::util::destroy_device_context(&context);
    assert(!context.runtime.is_valid());
    assert(!context.device.is_valid());
    assert(@catch(gpu::get_device_caps(&old_device)) == gpu::INVALID_HANDLE);
    assert(diagnostics.errors == 0);
}
