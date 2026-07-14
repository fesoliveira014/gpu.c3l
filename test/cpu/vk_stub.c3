module gpu::vk @private;

import gpu @public;
import std::core::mem;

struct StubRuntimeState {
    gpu::Runtime runtime;
    uint         adapter_count;
}

uint configured_adapter_count @private = 1;
uint runtime_creates @private;
uint runtime_destroys @private;
bool probe_destroy_visibility @private;
fault destroy_visibility_fault @private;

fn void destroy_stub_runtime(void* backend_state) {
    StubRuntimeState* state = (StubRuntimeState*)backend_state;
    if (probe_destroy_visibility) {
        destroy_visibility_fault = @catch(gpu::enumerate_adapters(&state.runtime));
        probe_destroy_visibility = false;
    }
    runtime_destroys++;
    mem::free(state);
}

fn gpu::AdapterInfo? get_stub_adapter_info(void* backend_state, uint index) {
    StubRuntimeState* state = (StubRuntimeState*)backend_state;
    if (state == null || index >= state.adapter_count) return gpu::INVALID_HANDLE~;
    return {
        .name             = index == 0 ? "stub adapter 0" : "stub adapter 1",
        .device_class     = gpu::AdapterClass.SOFTWARE,
        .memory           = {
            .device_local_bytes = 1024,
            .host_visible_bytes = 1024,
        },
        .queues           = {
            .graphics_count = 1,
            .compute_count  = 1,
            .transfer_count = 1,
        },
        .strict_supported = true,
    };
}

fn gpu::AdapterDiagnostics? get_stub_adapter_diagnostics(void* backend_state, uint index) {
    StubRuntimeState* state = (StubRuntimeState*)backend_state;
    if (state == null || index >= state.adapter_count) return gpu::INVALID_HANDLE~;
    return {
        .backend_name = "stub",
        .driver_name  = "stub driver",
    };
}

const gpu::RuntimeBackendVTable STUB_RUNTIME_VTABLE @private = {
    .destroy_runtime         = &destroy_stub_runtime,
    .get_adapter_info        = &get_stub_adapter_info,
    .get_adapter_diagnostics = &get_stub_adapter_diagnostics,
};

fn gpu::Runtime? create_vk_runtime(gpu::RuntimeDesc* desc) {
    StubRuntimeState* state = mem::new(StubRuntimeState);
    state.adapter_count = configured_adapter_count;
    runtime_creates++;
    defer catch destroy_stub_runtime(state);
    gpu::Runtime runtime = gpu::register_runtime(
        backend:       gpu::BackendKind.VULKAN,
        vtable:        &STUB_RUNTIME_VTABLE,
        backend_state: state,
        adapter_count: state.adapter_count,
    )!;
    state.runtime = runtime;
    return runtime;
}

fn void stub_set_adapter_count(uint count) => configured_adapter_count = count;

fn void stub_probe_destroy_visibility() {
    probe_destroy_visibility = true;
}

fn fault stub_destroy_visibility_fault() => destroy_visibility_fault;

fn void stub_reset_runtime_counts() {
    runtime_creates = 0;
    runtime_destroys = 0;
}

fn uint stub_runtime_create_count() => runtime_creates;

fn uint stub_runtime_destroy_count() => runtime_destroys;

// CPU-lane stub: satisfies the public module's backend routing so it
// compiles without the real backend or native libraries. Creating a device
// through it faults; the CPU tests never do.
fn gpu::Device? create_vk_device(gpu::DeviceDesc* desc) {
    return gpu::UNSUPPORTED_BACKEND~;
}
