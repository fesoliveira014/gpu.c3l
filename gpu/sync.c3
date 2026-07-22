module gpu;

const CompletionPoint COMPLETION_POINT_INVALID = {};
const uint STAGE_MASK_BITS @private = 0x01FF;

<* Execution scopes for a global barrier. `all` and `present` are exclusive. *>
bitstruct StageMask : uint {
    bool all             : 0;
    bool host            : 1;
    bool transfer        : 2;
    bool compute         : 3;
    bool vertex_shader   : 4;
    bool fragment_shader : 5;
    bool color_output    : 6;
    bool depth_output    : 7;
    bool present         : 8;
}

<* Explicit invalidation for special non-coherent GPU data paths. *>
bitstruct HazardFlags : uint {
    bool draw_arguments : 0;
    bool descriptors    : 1;
    bool depth_stencil  : 2;
}

<* Texture layout for explicit transitions. *>
enum TextureLayout : int {
    UNDEFINED,
    TRANSFER_SOURCE,
    TRANSFER_DESTINATION,
    SAMPLED,
    STORAGE,
    COLOR_ATTACHMENT,
    DEPTH_ATTACHMENT,
    PRESENT,
}

<* Read/write intent for a texture state. *>
bitstruct TextureAccess : uint {
    bool read  : 0;
    bool write : 1;
}

<* Queue-owned submission progress. Copy and reuse until its device is destroyed. *>
struct CompletionPoint {
    Device device;
    ulong  payload;
}

<* A published queue completion and the first destination stages that consume
   its work. *>
struct CompletionWait {
    CompletionPoint point;
    StageMask       before;
}

<* Global execution and memory dependency with no resource identity or range. *>
struct Barrier {
    StageMask   before;
    StageMask   after;
    HazardFlags hazards;
}

<* Layout, execution stages, and access intent for one texture use. *>
struct TextureState {
    TextureLayout layout;
    StageMask     stages;
    TextureAccess access;
}

<* Explicit semantic transition for a texture subresource range. A zero `view`
   selects the full texture. *>
struct TextureBarrier {
    TextureHandle   texture;
    TextureViewDesc view;
    TextureState    before;
    TextureState    after;
}

fn bool CompletionPoint.is_valid(self) => self.device.is_valid()
    && completion_point_sequence(self) != 0;
fn bool CompletionPoint.equals(self, CompletionPoint other) @operator(==) {
    return self.device == other.device && self.payload == other.payload;
}

fn bool stage_mask_valid(StageMask stages) @private {
    uint raw = (uint)stages;
    if (raw == 0 || (raw & STAGE_MASK_BITS) != raw) return false;
    if (stages.all && raw != 1) return false;
    if (stages.present && raw != 1u << 8) return false;
    return true;
}

fn bool submit_wait_stage_mask_valid(StageMask stages) @private {
    return stage_mask_valid(stages) && !stages.host && !stages.present;
}

fn bool texture_layout_valid(TextureLayout layout) @private {
    switch (layout) {
        case TextureLayout.UNDEFINED:
        case TextureLayout.TRANSFER_SOURCE:
        case TextureLayout.TRANSFER_DESTINATION:
        case TextureLayout.SAMPLED:
        case TextureLayout.STORAGE:
        case TextureLayout.COLOR_ATTACHMENT:
        case TextureLayout.DEPTH_ATTACHMENT:
        case TextureLayout.PRESENT:
            return true;
        default:
            return false;
    }
}

fn TextureState sampled_at(StageMask stages) {
    return {
        .layout = TextureLayout.SAMPLED,
        .stages = stages,
        .access = { .read },
    };
}

fn TextureState storage_at(StageMask stages, TextureAccess access) {
    return {
        .layout = TextureLayout.STORAGE,
        .stages = stages,
        .access = access,
    };
}

<* Construct a semantic transition for a selected texture range.
   `UNDEFINED` is source-only. *>
fn TextureBarrier? texture_view_transition(
    TextureHandle texture,
    TextureViewDesc view,
    TextureState before,
    TextureState after,
) {
    if (!texture_layout_valid(before.layout)
        || !texture_layout_valid(after.layout)
        || after.layout == TextureLayout.UNDEFINED) {
        return INVALID_ARGUMENT~;
    }
    return {
        .texture = texture,
        .view    = view,
        .before  = before,
        .after   = after,
    };
}

<* Construct a semantic transition for the whole texture.
   `UNDEFINED` is source-only. *>
fn TextureBarrier? texture_transition(
    TextureHandle texture,
    TextureState before,
    TextureState after,
) {
    return texture_view_transition(
        texture: texture,
        view:    {},
        before:  before,
        after:   after,
    );
}

<* Return whether the queue has completed this point. *>
fn bool? poll_completion(CompletionPoint point) {
    Device device = point.device;
    DeviceOperation operation = begin_device_operation(&device)!;
    defer end_device_operation(&operation);
    validate_completion_point(&operation, &device, point)!;
    if (operation.vtable.poll_completion == null) return UNSUPPORTED_FEATURE~;
    return operation.vtable.poll_completion(
        &device,
        completion_point_queue_id(point),
        completion_point_sequence(point),
    );
}

<* Wait up to timeout_ns for the queue to complete this point. A timeout leaves
   the point unchanged and faults WAIT_TIMEOUT, so the wait may be retried. *>
fn void? wait_completion(CompletionPoint point, ulong timeout_ns = ulong::max) {
    Device device = point.device;
    DeviceOperation operation = begin_device_operation(&device)!;
    defer end_device_operation(&operation);
    validate_completion_point(&operation, &device, point)!;
    if (operation.vtable.wait_completion == null) return UNSUPPORTED_FEATURE~;
    return operation.vtable.wait_completion(
        &device,
        completion_point_queue_id(point),
        completion_point_sequence(point),
        timeout_ns,
    );
}
