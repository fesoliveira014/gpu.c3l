module gpu::util;

import gpu;
import std::io;

<* Return one fresh surface from the supplied runtime, transferring ownership.
   Clean partial work before faulting; native objects remain caller-owned. *>
alias SurfaceFactory = fn gpu::Surface?(gpu::Runtime* runtime, void* user_data);

<* Select a borrowed adapter of the supplied runtime without modifying inputs
   or creating owners. Bootstrap handles must not escape the callback. *>
alias AdapterSelector = fn gpu::Adapter?(
    gpu::Runtime* runtime,
    gpu::DeviceDesc* requirements,
    void* user_data,
);

const uint SOFTWARE_RANK @private = 1;
const uint OTHER_RANK @private = 2;
const uint VIRTUAL_RANK @private = 3;
const uint INTEGRATED_RANK @private = 4;
const uint DISCRETE_RANK @private = 5;

<* Opt-in setup policy. Nested zero values retain their core defaults.
   device.surface must be zero; create_surface supplies optional presentation. *>
struct DeviceContextDesc {
    gpu::RuntimeDesc runtime;
    gpu::DeviceDesc device;
    gpu::QueueKind command_queue;
    gpu::CommandAllocatorDesc command_allocator;
    gpu::SwapchainDesc swapchain;
    SurfaceFactory create_surface;
    void* surface_user_data;
    AdapterSelector select_adapter;
    void* adapter_user_data;
}

<* Owns setup resources; copies do not create independent ownership.
   Pass by pointer and release through destroy_device_context after all use.
   adapter and queue borrow their runtime and device; userdata outlives teardown.
   swapchain_info is a creation snapshot; refresh it after resize, and ignore it when headless. *>
struct DeviceContext {
    gpu::Runtime runtime;
    gpu::Surface surface;
    gpu::Adapter adapter;
    gpu::Device device;
    gpu::Queue queue;
    gpu::CommandAllocator command_allocator;
    gpu::SwapchainHandle swapchain;
    gpu::SwapchainInfo swapchain_info;
    gpu::DebugMessageCallback debug_callback;
    void* debug_user_data;
}

<* Choose full contract checks, public default capacities, and a graphics queue.
   Presentation requires a surface factory and the application's drawable size.
   @return "An overridable headless setup description with backend layers off." *>
fn DeviceContextDesc default_device_context_desc() {
    return {
        .runtime = {
            .contract_validation = gpu::ContractValidation.FULL,
            .texture_heap_capacity = gpu::DEFAULT_TEXTURE_HEAP_CAPACITY,
            .sampler_heap_capacity = gpu::DEFAULT_SAMPLER_HEAP_CAPACITY,
            .texture_capacity = gpu::DEFAULT_TEXTURE_CAPACITY,
            .pipeline_capacity = gpu::MAX_PIPELINES,
        },
        .device = { .queues = { .required = { .graphics, .compute, .transfer } } },
        .command_queue = gpu::QueueKind.GRAPHICS,
        .command_allocator = {
            .command_buffer_capacity = gpu::DEFAULT_COMMAND_ALLOCATOR_CAPACITY,
        },
        .swapchain = {
            .preferred_format = gpu::Format.BGRA8_UNORM,
            .present_mode = gpu::PresentMode.FIFO,
        },
    };
}

<* Create an owning context without submitting work or retrying setup failures.
   Externally synchronize runtime/surface mutation. Callbacks run synchronously;
   native window objects and debug userdata remain borrowed through teardown.
   @param desc : "Call-scoped settings; never modified. Null is invalid."
   @return "A complete context, or the original fault after best-effort cleanup." *>
fn DeviceContext? create_device_context(DeviceContextDesc* desc) {
    validate_context_desc(desc)!;
    DeviceContextDesc settings = *desc;
    gpu::DeviceDesc device_desc = settings.device;
    if ((uint)device_desc.queues.required == 0
        && (uint)device_desc.queues.distinct_roles == 0
        && !device_desc.queues.single_queue) {
        device_desc.queues.required = { .graphics, .compute, .transfer };
    }

    gpu::Runtime runtime = gpu::create_runtime(&settings.runtime)!;
    defer catch (void)gpu::destroy_runtime(&runtime);

    gpu::AdapterList adapters = gpu::enumerate_adapters(&runtime)!;
    if (adapters.count == 0) return gpu::UNSUPPORTED_FEATURE~;

    gpu::Surface surface;
    if (settings.create_surface != null) {
        surface = create_context_surface(&runtime, &adapters, &settings)!;
        device_desc.surface = surface;
    }
    defer catch {
        if (surface.is_valid()) (void)gpu::destroy_surface(&surface);
    }

    gpu::Adapter adapter = select_context_adapter(
        &adapters,
        &device_desc,
        &settings,
    )!;
    gpu::Device device = gpu::create_device(&adapter, &device_desc)!;
    defer catch (void)gpu::destroy_device(&device);

    gpu::Queue queue = gpu::get_queue(&device, settings.command_queue)!;
    gpu::CommandAllocator command_allocator = gpu::create_command_allocator(
        &device,
        queue,
        &settings.command_allocator,
    )!;
    defer catch (void)gpu::destroy_command_allocator(&command_allocator);

    gpu::SwapchainHandle swapchain;
    gpu::SwapchainInfo swapchain_info;
    if (surface.is_valid()) {
        swapchain = gpu::create_swapchain(&device, &surface, &settings.swapchain)!;
        defer catch (void)gpu::destroy_swapchain(&device, swapchain);
        swapchain_info = gpu::get_swapchain_info(&device, swapchain)!;
    }

    return {
        .runtime = runtime,
        .surface = surface,
        .adapter = adapter,
        .device = device,
        .queue = queue,
        .command_allocator = command_allocator,
        .swapchain = swapchain,
        .swapchain_info = swapchain_info,
        .debug_callback = settings.runtime.debug_callback,
        .debug_user_data = settings.runtime.debug_user_data,
    };
}

<* Attempt reverse-order teardown, reporting faults and continuing without waits.
   Stop concurrent use, discard unsubmitted commands, wait completions and
   presentations, and destroy additional application resources before calling.
   @param context : "Owner to consume and clear; zero or null is harmless. Copies must not be reused." *>
fn void destroy_device_context(DeviceContext* context) {
    if (context == null) return;

    if (context.swapchain.is_valid()) {
        @report_context_teardown(context, "destroy_swapchain",
            gpu::destroy_swapchain(&context.device, context.swapchain));
    }

    if (context.command_allocator.handle.is_valid()) {
        @report_context_teardown(context, "destroy_command_allocator",
            gpu::destroy_command_allocator(&context.command_allocator));
    }

    if (context.device.is_valid()) {
        @report_context_teardown(context, "destroy_device",
            gpu::destroy_device(&context.device));
    }

    if (context.surface.is_valid()) {
        @report_context_teardown(context, "destroy_surface",
            gpu::destroy_surface(&context.surface));
    }

    if (context.runtime.is_valid()) {
        @report_context_teardown(context, "destroy_runtime",
            gpu::destroy_runtime(&context.runtime));
    }

    *context = {};
}

fn void? validate_context_desc(DeviceContextDesc* desc) @private {
    if (desc == null || desc.device.surface.is_valid()) return gpu::INVALID_ARGUMENT~;
    if (desc.create_surface == null && desc.surface_user_data != null) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (desc.select_adapter == null && desc.adapter_user_data != null) {
        return gpu::INVALID_ARGUMENT~;
    }

    gpu::QueueRoles required = desc.device.queues.required;
    if ((uint)required == 0) required = { .graphics, .compute, .transfer };
    bool available;
    switch (desc.command_queue) {
        case gpu::QueueKind.GRAPHICS: available = required.graphics;
        case gpu::QueueKind.COMPUTE: available = required.compute;
        case gpu::QueueKind.TRANSFER: available = required.transfer;
        default: return gpu::INVALID_ARGUMENT~;
    }
    if (!available) return gpu::INVALID_ARGUMENT~;
    if (desc.create_surface != null && !required.graphics) return gpu::INVALID_ARGUMENT~;
}

fn gpu::Surface? create_context_surface(
    gpu::Runtime* runtime,
    gpu::AdapterList* adapters,
    DeviceContextDesc* settings,
) @private {
    gpu::Surface surface = settings.create_surface(runtime, settings.surface_user_data)!;
    gpu::Adapter first_adapter = adapters.get(0)!;
    if (catch err = gpu::supports_presentation(&first_adapter, &surface)) {
        // Identity faults must not destroy a foreign callback result.
        if (err != gpu::INVALID_HANDLE) (void)gpu::destroy_surface(&surface);
        return err~;
    }
    return surface;
}

fn gpu::Adapter? select_context_adapter(
    gpu::AdapterList* adapters,
    gpu::DeviceDesc* requirements,
    DeviceContextDesc* settings,
) @private {
    if (settings.select_adapter != null) {
        gpu::Runtime runtime = adapters.runtime;
        gpu::DeviceDesc callback_requirements = *requirements;
        gpu::Adapter selected = settings.select_adapter(
            &runtime,
            &callback_requirements,
            settings.adapter_user_data,
        )!;
        for (uint index = 0; index < adapters.count; index++) {
            gpu::Adapter candidate = adapters.get(index)!;
            if ((ulong)candidate != (ulong)selected) continue;
            if (!gpu::supports_device_desc(&selected, requirements)!.supported) {
                return gpu::UNSUPPORTED_FEATURE~;
            }
            return selected;
        }
        return gpu::INVALID_HANDLE~;
    }

    gpu::Adapter best;
    uint best_rank;
    for (uint index = 0; index < adapters.count; index++) {
        gpu::Adapter candidate = adapters.get(index)!;
        if (!gpu::supports_device_desc(&candidate, requirements)!.supported) continue;
        gpu::AdapterInfo info = gpu::get_adapter_info(&candidate)!;
        uint rank = adapter_rank(info.device_class);
        if (rank <= best_rank) continue;
        best = candidate;
        best_rank = rank;
    }
    if (!best.is_valid()) return gpu::UNSUPPORTED_FEATURE~;
    return best;
}

fn uint adapter_rank(gpu::AdapterClass device_class) @private {
    switch (device_class) {
        case gpu::AdapterClass.DISCRETE: return DISCRETE_RANK;
        case gpu::AdapterClass.INTEGRATED: return INTEGRATED_RANK;
        case gpu::AdapterClass.VIRTUAL: return VIRTUAL_RANK;
        case gpu::AdapterClass.OTHER: return OTHER_RANK;
        case gpu::AdapterClass.SOFTWARE: return SOFTWARE_RANK;
        default: return OTHER_RANK;
    }
}

macro @report_context_teardown(DeviceContext* context, ZString operation, #release) @private {
    if (catch err = #release) {
        if (context.debug_callback != null) {
            gpu::DebugMessage message = {
                .severity = gpu::DebugMessageSeverity.ERROR,
                .category = { .resource_lifetime },
                .operation = operation,
                .has_fault = true,
                .public_fault = err,
                .invariant = "device context teardown could not release an owned resource",
            };
            context.debug_callback(&message, context.debug_user_data);
        } else {
            (void)io::eprintfn("gpu::util teardown: %s failed: %s", operation, err);
        }
    }
}
