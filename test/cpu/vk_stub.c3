module gpu::vk @private;

import gpu @public;
import std::core::mem;

struct StubRuntimeState {
    gpu::Runtime runtime;
    uint         adapter_count;
}

struct StubDeviceState {
    gpu::Adapter adapter;
}

uint configured_adapter_count @private = 1;
bool configured_strict_support @private = true;
uint device_support_queries @private;
uint device_temporary_creates @private;
uint device_temporary_destroys @private;
bool fail_device_after_temporary_allocation @private;
gpu::Adapter last_device_adapter @private;
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
    device_support_queries++;
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
        .strict_supported = configured_strict_support,
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

fn void stub_set_adapter_count(uint count) {
    configured_adapter_count = count;
    configured_strict_support = true;
}

fn void stub_set_strict_support(bool supported) => configured_strict_support = supported;

fn void stub_reset_device_counts() {
    device_support_queries = 0;
    device_temporary_creates = 0;
    device_temporary_destroys = 0;
    fail_device_after_temporary_allocation = false;
    last_device_adapter = {};
}

fn uint stub_device_support_query_count() => device_support_queries;

fn uint stub_device_temporary_create_count() => device_temporary_creates;

fn uint stub_device_temporary_destroy_count() => device_temporary_destroys;

fn void stub_fail_device_after_temporary_allocation(bool fail) {
    fail_device_after_temporary_allocation = fail;
}

fn gpu::Adapter stub_last_device_adapter() => last_device_adapter;

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

fn void destroy_stub_device(gpu::Device* device) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    device_temporary_destroys++;
    mem::free(state);
}

const gpu::BackendVTable STUB_DEVICE_VTABLE @private = {
    .destroy_device = &destroy_stub_device,
};

fn gpu::Device? create_vk_device_from_desc(gpu::DeviceDesc* desc) {
    return gpu::UNSUPPORTED_BACKEND~;
}

fn gpu::Device? create_vk_device(
    gpu::Adapter* adapter,
    gpu::DeviceRequest* request,
    gpu::Runtime* retained_runtime,
) {
    gpu::validate_device_request(request)!;
    StubDeviceState* state = mem::new(StubDeviceState);
    device_temporary_creates++;
    defer catch {
        device_temporary_destroys++;
        mem::free(state);
    }
    if (fail_device_after_temporary_allocation) return gpu::BACKEND_ERROR~;

    state.adapter = *adapter;
    last_device_adapter = *adapter;
    gpu::DeviceCaps caps = { .strict_enabled = true };
    return gpu::register_device(
        backend:          gpu::BackendKind.VULKAN,
        caps:             caps,
        vtable:           &STUB_DEVICE_VTABLE,
        backend_state:    state,
        retained_runtime: retained_runtime,
        adapter:          adapter,
    );
}
