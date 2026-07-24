module gpu::internal @private;

import gpu;
import std::atomic;
import std::atomic::types;
import std::thread;
import std::core::mem;

// Backend dispatch functions. Lifecycle entries take validated public tokens;
// recording entries take the authoritative record resolved at begin.
alias CheckDeviceDestroyFn @private  = fn void? (void*);
alias DestroyDeviceFn @private       = fn void (void*);
alias ReportContractFailureFn @private = fn fault (
    void* backend_state,
    ZString operation,
    fault public_fault,
    DebugResourceKind resource_kind,
    ZString rejected_field,
    ZString invariant,
);
alias GetMemoryStatsFn @private      = fn MemoryStats? (Device*);
alias BuildMemoryReportFn @private   = fn String? (Device*, bool detailed);
alias AllocateMemoryFn @private      = fn GpuAllocation? (Device*, AllocationDesc*);
alias FreeAllocationFn @private      = fn void? (Device*, GpuAllocation);
alias GetAllocationInfoFn @private   = fn AllocationInfo? (Device*, GpuAllocation);
alias GetAllocationSpanFn @private   = fn GpuSpan? (Device*, GpuAllocation);
alias GetSpanMappingFn @private      = fn char[]? (Device*, GpuSpan);
alias GetSpanAddressFn @private      = fn GpuAddress? (Device*, GpuSpan);
alias FlushSpanFn @private           = fn void? (Device*, GpuSpan);
alias InvalidateSpanFn @private      = fn void? (Device*, GpuSpan);
// Out-param form avoids a c3c 0.8.0 debug-info codegen failure.
alias CreateCommandAllocatorFn @private = fn CommandAllocatorHandle? (
    Device*,
    Queue,
    CommandAllocatorDesc*,
);
alias DestroyCommandAllocatorFn @private = fn void? (
    Device*,
    CommandAllocatorHandle,
);
alias BeginCommandsFn @private = fn void? (
    CommandAllocator* allocator,
    CommandList* out_commands,
);
alias ResolveCommandFn @private = fn CommandRecord*? (
    void* backend_state,
    CommandListHandle handle,
);
alias ReserveGeneratedScratchFn @private = fn void? (
    CommandAllocator*,
    GeneratedScratchDesc*,
);
alias ReleaseGeneratedScratchFn @private = fn void? (
    CommandAllocator*,
    PipelineHandle,
    GeneratedWorkKind,
);
alias EndCommandsFn @private     = fn void? (CommandRecord*);
alias DiscardCommandsFn @private = fn void? (
    CommandRecord*,
    bool executable,
);
alias SubmitFn @private          = fn CompletionPoint? (Queue, SubmitDesc*);
alias PollCompletionFn @private      = fn bool? (Device*, uint, ulong);
alias WaitCompletionFn @private      = fn void? (Device*, uint, ulong, ulong);
alias CmdCopyBufferFn @private       = fn void? (CommandRecord*, BufferCopyDesc*);
alias CmdFillBufferFn @private       = fn void? (CommandRecord*, GpuSpan, uint);
alias CmdCopyBufferToTextureFn @private = fn void? (CommandRecord*, BufferTextureCopyDesc*);
alias CmdCopyTextureToBufferFn @private = fn void? (CommandRecord*, TextureBufferCopyDesc*);
alias CmdBarrierFn @private          = fn void? (CommandRecord*, Barrier*);
alias CmdTextureBarrierFn @private   = fn void? (CommandRecord*, TextureBarrier*);
alias GetTextureFormatSupportFn @private = fn TextureFormatSupport? (Device*, Format);
alias SupportsTextureDescFn @private     = fn bool? (Device*, TextureDesc*);
alias GetTextureRequirementsFn @private  = fn TextureRequirements? (Device*, TextureDesc*);
alias CreateTextureFn @private           = fn TextureHandle? (Device*, TextureDesc*);
alias CreatePlacedTextureFn @private = fn TextureHandle? (
    Device* device,
    TextureDesc* desc,
    GpuAllocation allocation,
    usz offset,
);
alias CreateDedicatedTextureFn @private = fn DedicatedTexture? (
    Device* device,
    TextureDesc* desc,
    AllocationDesc* allocation_desc,
);
alias CreateAttachmentViewFn @private = fn AttachmentViewHandle? (
    Device*,
    AttachmentViewDesc*,
);
alias DestroyAttachmentViewFn @private = fn void? (
    Device*,
    AttachmentViewHandle,
);
alias CreateTextureViewFn @private       = fn TextureView? (Device*, TextureHandle, TextureViewDesc*);
alias DestroyTextureFn @private          = fn void? (Device*, TextureHandle);
alias DestroyTextureViewFn @private      = fn void? (Device*, TextureView);
alias CreateTextureViewsFn @private      = fn void? (Device*, TextureViewCreateDesc[], TextureView[]);
alias InternSamplerFn @private            = fn SamplerIndex? (Device*, SamplerDesc*);
alias CreateComputePipelineFn @private  = fn PipelineHandle? (Device*, ComputePipelineDesc*);
alias CreateGraphicsPipelineFn @private = fn PipelineHandle? (Device*, GraphicsPipelineDesc*);
alias CreateComputePipelinesFn @private = fn void? (Device*, ComputePipelineDesc[], PipelineHandle[]);
alias CreateGraphicsPipelinesFn @private = fn void? (Device*, GraphicsPipelineDesc[], PipelineHandle[]);
alias DestroyPipelineFn @private        = fn void? (Device*, PipelineHandle);
alias GetPipelineCacheSizeFn @private   = fn usz? (Device*);
alias GetPipelineCacheDataFn @private   = fn usz? (Device*, char[]);
alias CmdBindPipelineFn @private             = fn void? (CommandRecord*, PipelineHandle);
alias CmdSetDepthStateFn @private            = fn void? (CommandRecord*, DepthState*);
alias CmdSetRasterStateFn @private           = fn void? (CommandRecord*, DynamicRasterState*);
alias CmdSetGraphicsStateFn @private         = fn void? (CommandRecord*, GraphicsState*);
alias CmdDispatchFn @private                 = fn void? (CommandRecord*, GpuAddress, Vec3u);
alias CmdBeginRenderPassFn @private          = fn void? (CommandRecord*, RenderPassDesc*);
alias CmdBeginRenderPassWithStateFn @private = fn void? (CommandRecord*, RenderPassDesc*, GraphicsState*);
alias CmdSetViewportFn @private              = fn void? (CommandRecord*, Viewport*);
alias CmdSetScissorFn @private               = fn void? (CommandRecord*, ScissorRect*);
alias CmdEndRenderPassFn @private            = fn void? (CommandRecord*);
alias CmdDrawFn @private = fn void? (
    CommandRecord* record,
    GpuAddress vertex_root,
    GpuAddress fragment_root,
    uint vertex_count,
    uint instance_count,
);
alias CmdDrawIndexedFn @private = fn void? (
    CommandRecord* record,
    GpuAddress vertex_root,
    GpuAddress fragment_root,
    GpuSpan index_span,
    uint index_count,
    uint instance_count,
    IndexType index_type,
);
alias CmdDrawIndirectFn @private = fn void? (
    CommandRecord* record,
    GpuAddress vertex_root,
    GpuAddress fragment_root,
    GpuSpan args,
    uint draw_count,
);
alias CmdDrawIndexedIndirectFn @private = fn void? (
    CommandRecord* record,
    GpuAddress vertex_root,
    GpuAddress fragment_root,
    GpuSpan args,
    uint draw_count,
    GpuSpan index_span,
    IndexType index_type,
);
alias CmdDrawIndexedIndirectCountFn @private = fn void? (
    CommandRecord* record,
    GpuAddress vertex_root,
    GpuAddress fragment_root,
    GpuSpan args,
    GpuSpan count_span,
    uint max_draw_count,
    GpuSpan index_span,
    IndexType index_type,
);
alias CmdDispatchIndirectFn @private         = fn void? (CommandRecord*, GpuAddress, GpuSpan);
alias CmdDrawGeneratedFn @private = fn void? (
    CommandRecord* record,
    GpuSpan records,
    GpuSpan count_span,
    uint max_draw_count,
);
alias CmdDrawIndexedGeneratedFn @private = fn void? (
    CommandRecord* record,
    GpuSpan records,
    GpuSpan count_span,
    uint max_draw_count,
    GpuSpan index_span,
    IndexType index_type,
);
alias CmdDispatchGeneratedFn @private = fn void? (
    CommandRecord* record,
    GpuSpan records,
    GpuSpan count_span,
    uint max_dispatch_count,
);
alias CmdBeginLabelFn @private               = fn void? (CommandRecord*, ZString, float[4]);
alias CmdEndLabelFn @private                 = fn void? (CommandRecord*);
alias CreateSwapchainFn @private       = fn SwapchainHandle? (Device*, Surface*, SwapchainDesc*);
alias DestroySwapchainFn @private      = fn void? (Device*, SwapchainHandle);
alias ResizeSwapchainFn @private       = fn void? (Device*, SwapchainHandle, uint, uint);
alias GetSwapchainInfoFn @private      = fn SwapchainInfo? (Device*, SwapchainHandle);
alias AcquireNextImageFn @private      = fn AcquiredImage? (Device*, SwapchainHandle, ulong);
alias PresentFn @private               = fn void? (Device*, AcquiredImage*, CompletionPoint);
alias GetPresentModeSupportFn @private = fn PresentModeSupport? (Device*, SwapchainHandle);

const uint MAX_DEVICES @private = 64;
const uint MAX_DEVICE_GENERATION @private = 0x00FF_FFFF;
const uint MAX_DEVICE_OPERATION_PINS @private = 0x3FFF_FFFF;
const uint MAX_DEVICE_OPERATION_PIN_ATTEMPTS @private = 64;
const uint MAX_DEVICE_COMMANDS @private = 4096;

const uint DEVICE_SLOT_FREE @private = 0;
const uint DEVICE_SLOT_LIVE @private = 1;
const uint DEVICE_SLOT_CLOSING @private = 2;
const uint DEVICE_SLOT_RETIRED @private = 3;

enum CommandState @private {
    INACTIVE,
    RECORDING,
    RECORDING_RENDER_PASS,
    EXECUTABLE,
    SUBMITTING,
    SUBMITTED,
}

enum ResourceOwnerKind @private {
    ALLOCATION,
    BUFFER,
    TEXTURE,
    ATTACHMENT_VIEW,
    TEXTURE_VIEW,
    PIPELINE,
    SWAPCHAIN,
    COMMAND_ALLOCATOR,
    COMMAND,
    COMPLETION_POINT,
}

$assert (ulong)ResourceOwnerKind.COMPLETION_POINT + 1
    == gpu::RESOURCE_OWNER_KIND_COUNT;


bitstruct DeviceSlotControl : ulong @private {
    uint state      : 0..1;
    uint pins       : 2..31;
    uint generation : 32..55;
    uint reserved   : 56..63;
}

struct CommandRecord @private {
    CommandOps*       ops;
    void*             backend_state;
    void*             backend_command;
    Device            device;
    CommandListHandle identity;
    CommandState      state;
}
$assert CommandRecord::size <= ulong::size * 8;

struct CommandOps @private {
    CmdCopyBufferFn               copy_buffer;
    CmdFillBufferFn               fill_buffer;
    CmdCopyBufferToTextureFn      copy_buffer_to_texture;
    CmdCopyTextureToBufferFn      copy_texture_to_buffer;
    CmdBarrierFn                  barrier;
    CmdTextureBarrierFn           texture_barrier;
    CmdBindPipelineFn             bind_pipeline;
    CmdSetDepthStateFn            set_depth_state;
    CmdSetRasterStateFn           set_raster_state;
    CmdSetGraphicsStateFn         set_graphics_state;
    CmdDispatchFn                 dispatch;
    CmdBeginRenderPassFn          begin_render_pass;
    CmdBeginRenderPassWithStateFn begin_render_pass_with_state;
    CmdSetViewportFn              set_viewport;
    CmdSetScissorFn               set_scissor;
    CmdEndRenderPassFn            end_render_pass;
    CmdDrawFn                     draw;
    CmdDrawIndexedFn              draw_indexed;
    CmdDrawIndirectFn             draw_indirect;
    CmdDrawIndexedIndirectFn      draw_indexed_indirect;
    CmdDrawIndexedIndirectCountFn draw_indexed_indirect_count;
    CmdDispatchIndirectFn         dispatch_indirect;
    CmdDrawGeneratedFn            draw_generated;
    CmdDrawIndexedGeneratedFn     draw_indexed_generated;
    CmdDispatchGeneratedFn        dispatch_generated;
    CmdBeginLabelFn               begin_label;
    CmdEndLabelFn                 end_label;
}

// One immutable table per backend.
struct BackendVTable @private {
    CheckDeviceDestroyFn      check_device_destroy;
    DestroyDeviceFn           destroy_device;
    ReportContractFailureFn   report_contract_failure;
    GetMemoryStatsFn          get_memory_stats;
    BuildMemoryReportFn       build_memory_report;
    AllocateMemoryFn          allocate_memory;
    FreeAllocationFn          free_allocation;
    GetAllocationInfoFn       get_allocation_info;
    GetAllocationSpanFn       get_allocation_span;
    GetSpanMappingFn          get_span_mapping;
    GetSpanAddressFn          get_span_address;
    FlushSpanFn               flush_mapped_span;
    InvalidateSpanFn          invalidate_mapped_span;
    CreateCommandAllocatorFn  create_command_allocator;
    DestroyCommandAllocatorFn destroy_command_allocator;
    ReserveGeneratedScratchFn reserve_generated_scratch;
    ReleaseGeneratedScratchFn release_generated_scratch;
    BeginCommandsFn           begin_commands;
    ResolveCommandFn         resolve_command;
    EndCommandsFn             end_commands;
    DiscardCommandsFn         discard_commands;
    SubmitFn                  submit;
    PollCompletionFn          poll_completion;
    WaitCompletionFn          wait_completion;
    CommandOps*               command_ops;
    GetTextureFormatSupportFn get_texture_format_support;
    SupportsTextureDescFn     supports_texture_desc;
    GetTextureRequirementsFn  get_texture_requirements;
    CreateTextureFn           create_texture;
    CreatePlacedTextureFn     create_placed_texture;
    CreateDedicatedTextureFn  create_dedicated_texture;
    DestroyTextureFn          destroy_texture;
    CreateAttachmentViewFn    create_attachment_view;
    DestroyAttachmentViewFn   destroy_attachment_view;
    CreateTextureViewFn       create_texture_view;
    DestroyTextureViewFn      destroy_texture_view;
    CreateTextureViewsFn      create_texture_views;
    InternSamplerFn           intern_sampler;
    CreateComputePipelineFn   create_compute_pipeline;
    CreateGraphicsPipelineFn  create_graphics_pipeline;
    CreateComputePipelinesFn  create_compute_pipelines;
    CreateGraphicsPipelinesFn create_graphics_pipelines;
    DestroyPipelineFn         destroy_pipeline;
    GetPipelineCacheSizeFn    get_pipeline_cache_size;
    GetPipelineCacheDataFn    get_pipeline_cache_data;
    CreateSwapchainFn         create_swapchain;
    DestroySwapchainFn        destroy_swapchain;
    ResizeSwapchainFn         resize_swapchain;
    GetSwapchainInfoFn        get_swapchain_info;
    AcquireNextImageFn        acquire_next_image;
    PresentFn                 present;
    GetPresentModeSupportFn   get_present_mode_support;
}

struct DeviceData @private {
    BackendKind    backend;
    DeviceCaps     caps;
    QueueSelection queues;
    BackendVTable* vtable;
    CommandOps*    command_ops;
    void*          backend_state;
    ulong          resource_owner;
    Runtime        runtime;
    Adapter        adapter;
    Surface        presentation_surface;
    bool           runtime_retained;
}

struct DeviceSlot @private {
    Atomic{ulong} control;
    Atomic{ulong} live_children;
    Atomic{bool}  lost;
    DeviceData    data;
}

struct DeviceOperation @private {
    DeviceSlot*    slot;
    DeviceData*    data;
    BackendVTable* vtable;
    bool           active;
    bool           owns_pin;
}


<* Private test and benchmark counters for structural recording work. Compare
   relaxed snapshots only around externally synchronized cold or warm work. *>
struct RecordingWorkCounters @private @if(
    env::TESTING
    || $feature(COMMAND_RESOLUTION_STATS)
    || $feature(COMMAND_RECORDING_WORK_STATS)
) {
    ulong host_allocations;
    ulong command_pool_creations;
    ulong command_buffer_allocations;
    ulong command_buffer_frees;
    ulong command_buffer_resets;
    ulong image_view_creations;
    ulong vma_allocations;
    ulong registry_lock_acquisitions;
    ulong generated_scratch_misses;
    ulong generated_dispatch_emissions;
    ulong generated_draw_emissions;
    ulong generated_indexed_draw_emissions;
}

Atomic{ulong} command_device_registry_resolutions @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_retained_pin_operations @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_lifecycle_vtable_dispatches @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_table_lookups @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_pipeline_table_lookups @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_pipeline_cache_lookups @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_policy_selections @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_recording_commands @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_native_commands @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_pipeline_bind_commands @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_strict_heap_set_bind_commands @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_strict_heap_buffer_bind_commands @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));
Atomic{ulong} command_strict_heap_offset_commands @private @if(env::TESTING || $feature(COMMAND_RESOLUTION_STATS));

fn void note_command_device_registry_resolution() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_device_registry_resolutions.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_command_retained_pin_operation() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_retained_pin_operations.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_command_lifecycle_vtable_dispatch() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_lifecycle_vtable_dispatches.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_command_table_lookup() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_table_lookups.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_command_pipeline_table_lookup() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_pipeline_table_lookups.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_command_pipeline_cache_lookup() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_pipeline_cache_lookups.add(
            1,
            AtomicOrdering.RELAXED,
        );
    $endif
}

fn void note_command_policy_selection() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_policy_selections.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_command_recording_call() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_recording_commands.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_command_native_call() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_native_commands.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_command_pipeline_bind() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_pipeline_bind_commands.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_command_strict_heap_set_bind() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_strict_heap_set_bind_commands.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_command_strict_heap_buffer_bind() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_strict_heap_buffer_bind_commands.add(1, AtomicOrdering.RELAXED);
    $endif
}

fn void note_command_strict_heap_offset() @private {
    $if env::TESTING || $feature(COMMAND_RESOLUTION_STATS):
        (void)command_strict_heap_offset_commands.add(1, AtomicOrdering.RELAXED);
    $endif
}


DeviceSlot[MAX_DEVICES] device_slots @private;
Atomic{bool} device_registry_lock @private;
Atomic{ulong} device_resource_owner_counter @private;
Atomic{ulong} device_registry_lock_acquisitions @private;

fn void lock_device_registry() @private {
    while (device_registry_lock.set(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    (void)device_registry_lock_acquisitions.add(1, AtomicOrdering.RELAXED);
}

fn void unlock_device_registry() @private {
    (void)device_registry_lock.clear(AtomicOrdering.RELEASE);
}

fn ulong device_registry_lock_count() @private {
    return device_registry_lock_acquisitions.load(AtomicOrdering.RELAXED);
}

fn void reset_device_registry_lock_count() @private @if(
    env::TESTING
    || $feature(COMMAND_RESOLUTION_STATS)
    || $feature(COMMAND_RECORDING_WORK_STATS)
) {
    device_registry_lock_acquisitions.store(0, AtomicOrdering.RELAXED);
}

fn ulong? allocate_device_resource_owner() @private {
    ulong current = device_resource_owner_counter.load(AtomicOrdering.RELAXED);
    while (true) {
        if (current == gpu::MAX_DEVICE_RESOURCE_OWNER) return gpu::SLOT_TABLE_FULL~;
        ulong seen = mem::compare_exchange(
            &device_resource_owner_counter.data,
            current,
            current + 1,
            AtomicOrdering.ACQUIRE_RELEASE,
            AtomicOrdering.RELAXED,
        );
        if (seen == current) return current + 1;
        current = seen;
    }
}

fn DeviceOperation? begin_device_operation(
    Device* device,
    bool allow_lost = false,
) @private {
    if (device == null || !device.is_valid()) return gpu::INVALID_HANDLE~;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return gpu::INVALID_HANDLE~;

    DeviceSlot* slot = &device_slots[device.index];
    ulong packed = slot.control.load(AtomicOrdering.ACQUIRE);
    for (uint attempt = 0; attempt < MAX_DEVICE_OPERATION_PIN_ATTEMPTS; attempt++) {
        DeviceSlotControl control = (DeviceSlotControl)packed;
        if (control.generation != device.generation) return gpu::INVALID_HANDLE~;
        if (control.state == DEVICE_SLOT_CLOSING) return gpu::DEVICE_BUSY~;
        if (control.state != DEVICE_SLOT_LIVE) return gpu::INVALID_HANDLE~;
        if (control.pins == MAX_DEVICE_OPERATION_PINS) return gpu::DEVICE_BUSY~;

        DeviceSlotControl pinned = control;
        pinned.pins++;
        ulong seen = mem::compare_exchange(
            &slot.control.data,
            packed,
            (ulong)pinned,
            AtomicOrdering.ACQUIRE_RELEASE,
            AtomicOrdering.ACQUIRE,
        );
        if (seen != packed) {
            packed = seen;
            continue;
        }
        if (!allow_lost && slot.lost.load(AtomicOrdering.ACQUIRE)) {
            DeviceSlotControl pin = { .pins = 1 };
            (void)slot.control.sub((ulong)pin, AtomicOrdering.RELEASE);
            return gpu::DEVICE_LOST~;
        }
        return {
            .slot     = slot,
            .data     = &slot.data,
            .vtable   = slot.data.vtable,
            .active   = true,
            .owns_pin = true,
        };
    }
    return gpu::DEVICE_BUSY~;
}

fn DeviceOperation? borrow_retained_device_operation(
    Device* device,
    bool allow_lost = false,
) @private {
    if (device == null || !device.is_valid()) return gpu::INVALID_HANDLE~;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return gpu::INVALID_HANDLE~;

    DeviceSlot* slot = &device_slots[device.index];
    DeviceSlotControl control =
        (DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    if (control.generation != device.generation) return gpu::INVALID_HANDLE~;
    if (control.state == DEVICE_SLOT_CLOSING) return gpu::DEVICE_BUSY~;
    if (control.state != DEVICE_SLOT_LIVE || control.pins == 0) {
        return gpu::INVALID_HANDLE~;
    }
    if (!allow_lost && slot.lost.load(AtomicOrdering.ACQUIRE)) {
        return gpu::DEVICE_LOST~;
    }
    return {
        .slot   = slot,
        .data   = &slot.data,
        .vtable = slot.data.vtable,
        .active = true,
    };
}

fn void end_device_operation(DeviceOperation* operation) @private {
    if (operation == null || !operation.active) return;

    if (operation.owns_pin) {
        DeviceSlotControl pin = { .pins = 1 };
        (void)operation.slot.control.sub((ulong)pin, AtomicOrdering.RELEASE);
    }
    operation.active = false;
    operation.owns_pin = false;
}

fn void transfer_device_operation_pin(DeviceOperation* operation) @private {
    if (operation == null || !operation.active) return;
    operation.owns_pin = false;
}

fn bool release_retained_device_pins(
    DeviceOperation* operation,
    uint count = 1,
) @private {
    if (operation == null || !operation.active || count == 0) return false;

    ulong packed = operation.slot.control.load(AtomicOrdering.ACQUIRE);
    while (true) {
        DeviceSlotControl control = (DeviceSlotControl)packed;
        uint operation_pins = operation.owns_pin ? 1 : 0;
        if (control.pins < count
            || control.pins - count < operation_pins) {
            return false;
        }

        DeviceSlotControl released = control;
        released.pins -= count;
        ulong seen = mem::compare_exchange(
            &operation.slot.control.data,
            packed,
            (ulong)released,
            AtomicOrdering.ACQUIRE_RELEASE,
            AtomicOrdering.ACQUIRE,
        );
        if (seen != packed) {
            packed = seen;
            continue;
        }
        return true;
    }
}

fn void release_command_record_ownership(
    CommandRecord* record,
) @private {
    if (record == null || !record.device.is_valid()
        || record.device.index >= MAX_DEVICES) {
        return;
    }
    DeviceSlot* slot = &device_slots[record.device.index];
    (void)slot.live_children.sub(1, AtomicOrdering.RELEASE);
    DeviceOperation operation = {
        .slot   = slot,
        .active = true,
    };
    (void)release_retained_device_pins(&operation);
    operation.active = false;
}

fn void retain_device_children(DeviceOperation* operation, ulong count = 1) @private {
    if (count == 0) return;
    (void)operation.slot.live_children.add(count, AtomicOrdering.RELEASE);
}

fn void release_device_children(DeviceOperation* operation, ulong count = 1) @private {
    if (count == 0) return;
    (void)operation.slot.live_children.sub(count, AtomicOrdering.RELEASE);
}

fn ulong resource_owner_tag(ulong device_owner, ResourceOwnerKind kind) @private {
    return (device_owner - 1) * gpu::RESOURCE_OWNER_KIND_COUNT + (ulong)kind + 1;
}

fn fault report_device_contract_failure(
    DeviceOperation* operation,
    ZString operation_name,
    fault public_fault,
    DebugResourceKind resource_kind,
    ZString rejected_field,
    ZString invariant,
) @private {
    if (operation != null && operation.active
        && operation.vtable.report_contract_failure != null) {
        return operation.vtable.report_contract_failure(
            backend_state:  operation.data.backend_state,
            operation:      operation_name,
            public_fault:   public_fault,
            resource_kind:  resource_kind,
            rejected_field: rejected_field,
            invariant:      invariant,
        );
    }
    return public_fault;
}

fn void? validate_resource_owner(
    DeviceOperation* operation,
    ulong owner,
    ResourceOwnerKind kind,
) @private {
    if (operation == null || !operation.active) return gpu::INVALID_HANDLE~;
    if (owner != resource_owner_tag(operation.data.resource_owner, kind)) {
        return gpu::INVALID_HANDLE~;
    }
}

<* Caller must hold an operation pin or be this device's destroying thread. *>
fn void mark_device_lost(Device* device) @private {
    if (device == null || !device.is_valid()) return;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return;

    DeviceSlot* slot = &device_slots[device.index];
    DeviceSlotControl control = (DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    if (control.generation != device.generation) return;
    if (control.state != DEVICE_SLOT_LIVE && control.state != DEVICE_SLOT_CLOSING) return;
    slot.lost.store(true, AtomicOrdering.RELEASE);
}

<* Caller must hold an operation pin for this device. *>
fn DeviceData* resolve_pinned_device(Device* device) @private {
    if (device == null || !device.is_valid()) return null;
    if (device.reserved != 0 || device.index >= MAX_DEVICES) return null;

    DeviceSlot* slot = &device_slots[device.index];
    DeviceSlotControl control = (DeviceSlotControl)slot.control.load(AtomicOrdering.ACQUIRE);
    if (control.generation != device.generation) return null;
    if (control.state == DEVICE_SLOT_LIVE) return &slot.data;
    if (control.state == DEVICE_SLOT_CLOSING && control.pins > 0) return &slot.data;
    return null;
}

fn bool device_slot_available() @private {
    lock_device_registry();
    defer unlock_device_registry();
    for (uint i = 0; i < MAX_DEVICES; i++) {
        DeviceSlotControl control = (DeviceSlotControl)device_slots[i].control.load(AtomicOrdering.RELAXED);
        if (control.state == DEVICE_SLOT_FREE) return true;
    }
    return false;
}

fn Device? register_device(
    BackendKind backend,
    DeviceCaps caps,
    BackendVTable* vtable,
    void* backend_state,
    CommandOps* command_ops = null,
    ulong resource_owner = 0,
    Runtime* retained_runtime = null,
    Adapter* adapter = null,
    Surface* presentation_surface = null,
    QueueSelection queues = {},
    Device* owner = null,
) @private {
    if (vtable == null || backend_state == null || vtable.destroy_device == null) return gpu::INVALID_ARGUMENT~;
    CommandOps* selected_ops = command_ops == null
        ? vtable.command_ops
        : command_ops;
    if (resource_owner == 0) resource_owner = allocate_device_resource_owner()!;
    bool has_runtime = retained_runtime != null;
    if (has_runtime != (adapter != null)) return gpu::INVALID_ARGUMENT~;

    Runtime runtime;
    Adapter registered_adapter;
    RuntimeData* runtime_data;
    if (has_runtime) {
        runtime_data = resolve_runtime(retained_runtime);
        RuntimeData* adapter_data = resolve_adapter(adapter);
        if (runtime_data == null || adapter_data != runtime_data) {
            return gpu::INVALID_HANDLE~;
        }
        if (runtime_data.live_device_count.load(AtomicOrdering.ACQUIRE) == 0) return gpu::INVALID_RESOURCE_STATE~;
        runtime = *retained_runtime;
        registered_adapter = *adapter;
    }

    Surface registered_surface;
    if (presentation_surface != null) {
        if (!has_runtime) return gpu::INVALID_ARGUMENT~;
        SurfaceData* surface_data = resolve_surface(presentation_surface);
        if (surface_data == null) return gpu::INVALID_HANDLE~;
        RuntimeData* surface_runtime = resolve_runtime(&surface_data.runtime);
        if (surface_runtime != runtime_data) return gpu::INVALID_HANDLE~;
        registered_surface = *presentation_surface;
    }
    caps.presentation_enabled = registered_surface.is_valid();
    if (queues.counts.graphics == 0
        && queues.counts.compute == 0
        && queues.counts.transfer == 0) {
        queues = default_queue_selection(&caps);
    }
    if (!queue_selection_is_well_formed(&queues)) return gpu::INVALID_ARGUMENT~;
    caps.queues = queues.counts;

    lock_device_registry();
    defer unlock_device_registry();
    for (uint i = 0; i < MAX_DEVICES; i++) {
        DeviceSlot* slot = &device_slots[i];
        DeviceSlotControl control = (DeviceSlotControl)slot.control.load(AtomicOrdering.RELAXED);
        if (control.state != DEVICE_SLOT_FREE) continue;

        control.state = DEVICE_SLOT_LIVE;
        control.pins = 0;
        if (control.generation == 0) control.generation = 1;
        control.reserved = 0;
        Device handle = {
            .index      = i,
            .generation = control.generation,
        };
        slot.live_children.store(0, AtomicOrdering.RELAXED);
        slot.lost.store(false, AtomicOrdering.RELAXED);
        slot.data = {
            .backend              = backend,
            .caps                 = caps,
            .queues               = queues,
            .vtable               = vtable,
            .command_ops          = selected_ops,
            .backend_state        = backend_state,
            .resource_owner       = resource_owner,
            .runtime              = runtime,
            .adapter              = registered_adapter,
            .presentation_surface = registered_surface,
            .runtime_retained     = has_runtime,
        };
        if (owner != null) *owner = handle;
        slot.control.store((ulong)control, AtomicOrdering.RELEASE);
        return handle;
    }
    return gpu::SLOT_TABLE_FULL~;
}

fn void* device_backend_state_ptr(Device* device) @private {
    DeviceData* data = resolve_pinned_device(device);
    return data == null ? null : data.backend_state;
}

fn void* device_backend_state(Device device) @private {
    return device_backend_state_ptr(&device);
}
