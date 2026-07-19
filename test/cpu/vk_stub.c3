module gpu::vk @private;

import gpu @public;
import std::atomic;
import std::core::mem;
import std::thread;

const uint STUB_SAMPLER_CAPACITY @private = 16;

struct StubRuntimeState {
    gpu::Runtime runtime;
    uint         adapter_count;
}

struct StubSamplerSlot {
    SamplerKey   key;
    gpu::Sampler sampler;
}

struct StubDeviceState {
    gpu::Adapter         adapter;
    gpu::Surface         surface;
    gpu::GpuAllocation   allocation;
    gpu::TextureHandle   texture;
    gpu::SwapchainHandle swapchain;
    StubSamplerSlot[STUB_SAMPLER_CAPACITY] samplers;
    Atomic{bool}         sampler_lock;
    ulong                resource_owner;
    uint                 sampler_count;
    uint                 sampler_intern_attempts;
    uint                 sampler_native_creates;
    uint                 sampler_publish_attempts;
    bool                 allocation_live;
    bool                 texture_live;
    bool                 swapchain_live;
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
bool surface_destroy_available @private = true;
bool probe_destroy_visibility @private;
fault destroy_visibility_fault @private;
uint surface_creates @private;
uint surface_destroys @private;
bool presentation_supported @private = true;
uint swapchain_create_attempts @private;
uint allocation_create_attempts @private;
uint allocation_free_attempts @private;
uint texture_destroy_attempts @private;

fn void lock_stub_samplers(StubDeviceState* state) @private {
    while (state.sampler_lock.set(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
}

fn void unlock_stub_samplers(StubDeviceState* state) @private {
    (void)state.sampler_lock.clear(AtomicOrdering.RELEASE);
}

fn gpu::Sampler? intern_stub_sampler(
    gpu::Device* device,
    gpu::SamplerDesc* desc,
) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null || desc == null) return gpu::INVALID_ARGUMENT~;
    lock_stub_samplers(state);
    defer unlock_stub_samplers(state);
    state.sampler_intern_attempts++;
    SamplerKey key = canonical_sampler_key(desc, 16.0f);
    for (uint i = 0; i < state.sampler_count; i++) {
        if (sampler_key_equal(&state.samplers[i].key, &key)) {
            return state.samplers[i].sampler;
        }
    }
    if (state.sampler_count >= STUB_SAMPLER_CAPACITY) {
        return gpu::SLOT_TABLE_FULL~;
    }
    uint index = state.sampler_count;
    gpu::Sampler sampler = {
        .owner = gpu::resource_owner_tag(
            state.resource_owner,
            gpu::ResourceOwnerKind.SAMPLER,
        ),
        .index      = index,
        .generation = 1,
    };
    state.samplers[index] = { .key = key, .sampler = sampler };
    state.sampler_count++;
    state.sampler_native_creates++;
    return sampler;
}

fn gpu::SamplerIndex? publish_stub_sampler(
    gpu::Device* device,
    gpu::Sampler sampler,
) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null) return gpu::INVALID_HANDLE~;
    lock_stub_samplers(state);
    defer unlock_stub_samplers(state);
    state.sampler_publish_attempts++;
    if (sampler.index >= state.sampler_count
        || state.samplers[sampler.index].sampler != sampler) {
        return gpu::INVALID_HANDLE~;
    }
    return {
        .slot       = (ushort)sampler.index,
        .generation = 1,
    };
}

fn uint stub_sampler_intern_attempt_count(gpu::Device* device) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    lock_stub_samplers(state);
    defer unlock_stub_samplers(state);
    return state.sampler_intern_attempts;
}

fn uint stub_sampler_native_create_count(gpu::Device* device) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    lock_stub_samplers(state);
    defer unlock_stub_samplers(state);
    return state.sampler_native_creates;
}

fn uint stub_sampler_publish_attempt_count(gpu::Device* device) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    lock_stub_samplers(state);
    defer unlock_stub_samplers(state);
    return state.sampler_publish_attempts;
}

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

fn void*? create_stub_win32_surface(
    void* backend_state,
    void* instance,
    void* window,
) {
    if (backend_state == null || instance == null || window == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    surface_creates++;
    return mem::new(char);
}

fn void*? create_stub_wayland_surface(
    void* backend_state,
    void* display,
    void* surface,
) {
    return create_stub_win32_surface(
        backend_state,
        display,
        surface,
    );
}

fn void*? create_stub_x11_surface(
    void* backend_state,
    void* display,
    ulong window,
) {
    if (backend_state == null || display == null || window == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    surface_creates++;
    return mem::new(char);
}

fn void destroy_stub_surface(void* backend_state, void* surface) {
    if (backend_state == null || surface == null) return;
    surface_destroys++;
    mem::free(surface);
}

fn bool? supports_stub_presentation(
    void* backend_state,
    uint adapter_index,
    void* surface,
) {
    StubRuntimeState* state = (StubRuntimeState*)backend_state;
    if (state == null || surface == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_HANDLE~;
    }
    return presentation_supported;
}

fn bool? supports_stub_queue_requirements(
    void* backend_state,
    uint adapter_index,
    gpu::QueueRequirements* requirements,
    void* backend_surface,
) {
    (void)backend_surface;
    StubRuntimeState* state = (StubRuntimeState*)backend_state;
    if (state == null || requirements == null || adapter_index >= state.adapter_count) {
        return gpu::INVALID_HANDLE~;
    }
    gpu::QueueSelection? selection = gpu::queue_selection_for_requirements(
        requirements:               *requirements,
        compute_aliases_graphics:    true,
        transfer_aliases_graphics:   true,
        transfer_aliases_compute:    true,
    );
    if (catch selection_fault = selection) {
        if (selection_fault == gpu::UNSUPPORTED_FEATURE) return false;
        return selection_fault~;
    }
    return true;
}

const gpu::RuntimeBackendVTable STUB_RUNTIME_VTABLE @private = {
    .destroy_runtime         = &destroy_stub_runtime,
    .get_adapter_info        = &get_stub_adapter_info,
    .get_adapter_diagnostics = &get_stub_adapter_diagnostics,
    .create_win32_surface    = &create_stub_win32_surface,
    .create_wayland_surface  = &create_stub_wayland_surface,
    .create_x11_surface      = &create_stub_x11_surface,
    .destroy_surface         = &destroy_stub_surface,
    .supports_presentation   = &supports_stub_presentation,
    .supports_queue_requirements = &supports_stub_queue_requirements,
};

const gpu::RuntimeBackendVTable STUB_RUNTIME_VTABLE_NO_SURFACE_DESTROY @private = {
    .destroy_runtime         = &destroy_stub_runtime,
    .get_adapter_info        = &get_stub_adapter_info,
    .get_adapter_diagnostics = &get_stub_adapter_diagnostics,
    .create_win32_surface    = &create_stub_win32_surface,
    .create_wayland_surface  = &create_stub_wayland_surface,
    .create_x11_surface      = &create_stub_x11_surface,
    .supports_presentation   = &supports_stub_presentation,
    .supports_queue_requirements = &supports_stub_queue_requirements,
};

fn gpu::Runtime? create_vk_runtime(gpu::RuntimeDesc* desc) {
    StubRuntimeState* state = mem::new(StubRuntimeState);
    state.adapter_count = configured_adapter_count;
    runtime_creates++;
    defer catch destroy_stub_runtime(state);
    gpu::Runtime runtime = gpu::register_runtime(
        backend:       gpu::BackendKind.VULKAN,
        vtable:        surface_destroy_available
            ? &STUB_RUNTIME_VTABLE
            : &STUB_RUNTIME_VTABLE_NO_SURFACE_DESTROY,
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

fn void stub_set_surface_destroy_available(bool available) {
    surface_destroy_available = available;
}

fn gpu::Adapter stub_last_device_adapter() => last_device_adapter;

fn gpu::Adapter? stub_device_adapter(gpu::Device* device) {
    gpu::DeviceOperation operation = gpu::begin_device_operation(device)!;
    defer gpu::end_device_operation(&operation);
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null) return gpu::INVALID_HANDLE~;
    return state.adapter;
}

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

fn void stub_reset_surface_counts() {
    surface_creates = 0;
    surface_destroys = 0;
    presentation_supported = true;
    swapchain_create_attempts = 0;
}

fn uint stub_surface_create_count() => surface_creates;

fn uint stub_surface_destroy_count() => surface_destroys;

fn uint stub_swapchain_create_attempt_count() => swapchain_create_attempts;

fn void stub_reset_allocation_counts() {
    allocation_create_attempts = 0;
    allocation_free_attempts = 0;
}

fn uint stub_allocation_free_attempt_count() => allocation_free_attempts;

fn void stub_reset_texture_counts() {
    texture_destroy_attempts = 0;
}

fn uint stub_texture_destroy_attempt_count() => texture_destroy_attempts;

fn void stub_set_presentation_support(bool supported) {
    presentation_supported = supported;
}

fn void destroy_stub_device(void* backend_state) {
    StubDeviceState* state = (StubDeviceState*)backend_state;
    if (state.swapchain_live) {
        gpu::surface_release_swapchain(&state.surface)!!;
    }
    device_temporary_destroys++;
    mem::free(state);
}

fn gpu::GpuAllocation? allocate_stub_memory(
    gpu::Device* device,
    gpu::AllocationDesc* desc,
) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    allocation_create_attempts++;
    if (state == null || desc == null) return gpu::INVALID_ARGUMENT~;
    if (state.allocation_live) return gpu::INVALID_RESOURCE_STATE~;
    state.allocation = {
        .owner      = gpu::resource_owner_tag(
            state.resource_owner,
            gpu::ResourceOwnerKind.ALLOCATION,
        ),
        .index      = 0,
        .generation = 1,
    };
    state.allocation_live = true;
    return state.allocation;
}

fn void? free_stub_allocation(
    gpu::Device* device,
    gpu::GpuAllocation allocation,
) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    allocation_free_attempts++;
    if (state == null || !state.allocation_live) return gpu::INVALID_HANDLE~;
    if (state.allocation != allocation) return gpu::INVALID_HANDLE~;
    state.allocation = {};
    state.allocation_live = false;
}
fn gpu::TextureCompatibility stub_texture_compatibility(ulong owner, uint value) {
    return (gpu::TextureCompatibility)(((uint128)owner << 64) | (uint128)value);
}


fn gpu::TextureRequirements? get_stub_texture_requirements(
    gpu::Device* device,
    gpu::TextureDesc* desc,
) {
    StubDeviceState* state =
        (StubDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null || !gpu::texture_desc_valid(desc)) {
        return gpu::INVALID_ARGUMENT~;
    }
    return {
        .size      = 64,
        .alignment = 64,
        .compatibility = stub_texture_compatibility(
            gpu::resource_owner_tag(
                state.resource_owner,
                gpu::ResourceOwnerKind.ALLOCATION,
            ),
            1,
        ),
    };
}

fn gpu::TextureHandle? create_stub_texture(
    gpu::Device* device,
    gpu::TextureDesc* desc,
) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null || desc == null) return gpu::INVALID_ARGUMENT~;
    if (state.texture_live) return gpu::INVALID_RESOURCE_STATE~;
    state.texture = {
        .owner      = gpu::resource_owner_tag(state.resource_owner, gpu::ResourceOwnerKind.TEXTURE),
        .index      = 0,
        .generation = 1,
    };
    state.texture_live = true;
    return state.texture;
}

fn gpu::TextureHandle? create_stub_placed_texture(
    gpu::Device* device,
    gpu::TextureDesc* desc,
    gpu::GpuAllocation allocation,
    usz offset,
) {
    StubDeviceState* state =
        (StubDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null || !state.allocation_live
        || state.allocation != allocation || offset != 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    return create_stub_texture(device, desc);
}

fn void? destroy_stub_texture(
    gpu::Device* device,
    gpu::TextureHandle texture,
) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    texture_destroy_attempts++;
    if (state == null || !state.texture_live) return gpu::INVALID_HANDLE~;
    if (state.texture != texture) return gpu::INVALID_HANDLE~;
    state.texture = {};
    state.texture_live = false;
}

fn gpu::SwapchainHandle? create_stub_swapchain(
    gpu::Device* device,
    gpu::Surface* surface,
    gpu::SwapchainDesc* desc,
) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    swapchain_create_attempts++;
    if (state == null || surface == null || desc == null) return gpu::INVALID_ARGUMENT~;
    if (state.swapchain_live) return gpu::INVALID_RESOURCE_STATE~;

    gpu::surface_retain_swapchain(surface)!;
    state.surface = *surface;
    state.swapchain = {
        .owner      = gpu::resource_owner_tag(state.resource_owner, gpu::ResourceOwnerKind.SWAPCHAIN),
        .index      = 0,
        .generation = 1,
    };
    state.swapchain_live = true;
    return state.swapchain;
}

fn void? destroy_stub_swapchain(
    gpu::Device* device,
    gpu::SwapchainHandle swapchain,
) {
    StubDeviceState* state = (StubDeviceState*)gpu::device_backend_state_ptr(device);
    if (state == null || !state.swapchain_live) return gpu::INVALID_HANDLE~;
    if (state.swapchain != swapchain) return gpu::INVALID_HANDLE~;

    gpu::surface_release_swapchain(&state.surface)!;
    state.surface = {};
    state.swapchain = {};
    state.swapchain_live = false;
}

const gpu::BackendVTable STUB_DEVICE_VTABLE @private = {
    .destroy_device           = &destroy_stub_device,
    .allocate_memory          = &allocate_stub_memory,
    .free_allocation          = &free_stub_allocation,
    .get_texture_requirements = &get_stub_texture_requirements,
    .create_texture           = &create_stub_texture,
    .create_placed_texture    = &create_stub_placed_texture,
    .destroy_texture          = &destroy_stub_texture,
    .intern_sampler           = &intern_stub_sampler,
    .publish_sampler          = &publish_stub_sampler,
    .create_swapchain         = &create_stub_swapchain,
    .destroy_swapchain        = &destroy_stub_swapchain,
};

fn gpu::Device? stub_create_strict_disabled_device() {
    StubDeviceState* state = mem::new(StubDeviceState);
    defer catch mem::free(state);
    state.resource_owner = gpu::allocate_device_resource_owner()!;
    return gpu::register_device(
        backend:        gpu::BackendKind.VULKAN,
        caps:           {},
        vtable:         &STUB_DEVICE_VTABLE,
        backend_state:  state,
        resource_owner: state.resource_owner,
    );
}

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
    state.resource_owner = gpu::allocate_device_resource_owner()!;
    device_temporary_creates++;
    defer catch {
        device_temporary_destroys++;
        mem::free(state);
    }
    if (fail_device_after_temporary_allocation) return gpu::BACKEND_ERROR~;

    state.adapter = *adapter;
    last_device_adapter = *adapter;
    gpu::Surface presentation_surface = gpu::requested_surface(*request);
    gpu::QueueRequirements requirements = gpu::requested_queue_requirements(*request);
    gpu::QueueSelection queues = gpu::queue_selection_for_requirements(
        requirements:               requirements,
        compute_aliases_graphics:    true,
        transfer_aliases_graphics:   true,
        transfer_aliases_compute:    true,
    )!;
    gpu::DeviceCaps caps = {
        .strict_enabled       = true,
        .presentation_enabled = presentation_surface.is_valid(),
    };
    return gpu::register_device(
        backend:              gpu::BackendKind.VULKAN,
        caps:                 caps,
        vtable:               &STUB_DEVICE_VTABLE,
        backend_state:        state,
        resource_owner:       state.resource_owner,
        retained_runtime:     retained_runtime,
        adapter:              adapter,
        presentation_surface: presentation_surface.is_valid()
            ? &presentation_surface
            : null,
        queues:               queues,
    );
}
