module gpu;

// Timeline semaphore counter value.
typedef SemaphoreValue = ulong;

const uint MAX_SEMAPHORES = 256;

<* Pipeline stage for barrier before/after pairs and semaphore wait/signal points. *>
enum Stage : int {
    HOST,
    TRANSFER,
    COMPUTE_SHADER,
    VERTEX_SHADER,
    FRAGMENT_SHADER,
    COLOR_ATTACHMENT,
    DEPTH_STENCIL,
    INDIRECT_COMMAND,
    PRESENT,
    NONE,
}

<* Memory access kind, paired with a Stage in barriers: before_* is the access
   to make visible, after_* the access that waits on it. *>
enum Hazard : int {
    HOST_WRITE,
    HOST_READ,
    TRANSFER_READ,
    TRANSFER_WRITE,
    SHADER_READ,
    SHADER_WRITE,
    COLOR_READ,
    COLOR_WRITE,
    DEPTH_READ,
    DEPTH_WRITE,
    INDIRECT_READ,
    PRESENT_READ,
    NONE,
    SHADER_READ_WRITE,
    COLOR_READ_WRITE,
    DEPTH_READ_WRITE,
}

<* Layout a texture must hold for a given use; transitioned via TextureBarrier. *>
enum TextureLayout : int {
    UNDEFINED,
    GENERAL,
    COLOR_ATTACHMENT,
    DEPTH_STENCIL,
    SHADER_READ,
    TRANSFER_SRC,
    TRANSFER_DST,
    PRESENT,
}

<* Common whole-texture use for explicit transition construction. *>
enum TextureUse : int {
    UNDEFINED,
    TRANSFER_DESTINATION,
    SAMPLED_COMPUTE,
    SAMPLED_FRAGMENT,
    STORAGE_COMPUTE,
    COLOR_ATTACHMENT,
    DEPTH_ATTACHMENT,
    PRESENT,
}

<* Orders before_* access against after_* access on a range of one buffer.
   `size` 0 means the whole buffer from `offset` onward (VK_WHOLE_SIZE),
   matching the flush/invalidate/fill convention. *>
struct BufferBarrier {
    BufferHandle buffer;
    usz          offset;
    usz          size;
    Stage        before_stage;
    Stage        after_stage;
    Hazard       before_hazard;
    Hazard       after_hazard;
}

<* Orders access and transitions layout on a whole texture. `old_layout` must match
   the texture's tracked layout or recording faults INVALID_RESOURCE_STATE. *>
struct TextureBarrier {
    TextureHandle texture;
    Stage         before_stage;
    Stage         after_stage;
    Hazard        before_hazard;
    Hazard        after_hazard;
    TextureLayout old_layout;
    TextureLayout new_layout;
}

<* Resource-agnostic barrier: orders all before_* access against all after_* access. *>
struct GlobalBarrier {
    Stage  before_stage;
    Stage  after_stage;
    Hazard before_hazard;
    Hazard after_hazard;
}

<* Creation parameters for a timeline semaphore (monotonic 64-bit counter). *>
struct SemaphoreDesc {
    SemaphoreValue initial_value;
    ZString        debug_name;
}

<* Blocks `stage` in a submit until the timeline reaches `value`. *>
struct SemaphoreWait {
    SemaphoreHandle semaphore;
    SemaphoreValue  value;
    Stage           stage;
}

<* Sets the timeline to value once the submit's work clears stage.
   The value must exceed the counter when the signal executes; ordering pending
   signals across queues is the caller's responsibility. *>
struct SemaphoreSignal {
    SemaphoreHandle semaphore;
    SemaphoreValue  value;
    Stage           stage;
}

fn void? resolve_texture_use(
    TextureUse use,
    Stage* stage,
    Hazard* hazard,
    TextureLayout* layout
) @private {
    switch (use) {
        case TextureUse.UNDEFINED:
            *stage = Stage.HOST;
            *hazard = Hazard.NONE;
            *layout = TextureLayout.UNDEFINED;
        case TextureUse.TRANSFER_DESTINATION:
            *stage = Stage.TRANSFER;
            *hazard = Hazard.TRANSFER_WRITE;
            *layout = TextureLayout.TRANSFER_DST;
        case TextureUse.SAMPLED_COMPUTE:
            *stage = Stage.COMPUTE_SHADER;
            *hazard = Hazard.SHADER_READ;
            *layout = TextureLayout.SHADER_READ;
        case TextureUse.SAMPLED_FRAGMENT:
            *stage = Stage.FRAGMENT_SHADER;
            *hazard = Hazard.SHADER_READ;
            *layout = TextureLayout.SHADER_READ;
        case TextureUse.STORAGE_COMPUTE:
            *stage = Stage.COMPUTE_SHADER;
            *hazard = Hazard.SHADER_READ_WRITE;
            *layout = TextureLayout.GENERAL;
        case TextureUse.COLOR_ATTACHMENT:
            *stage = Stage.COLOR_ATTACHMENT;
            *hazard = Hazard.COLOR_READ_WRITE;
            *layout = TextureLayout.COLOR_ATTACHMENT;
        case TextureUse.DEPTH_ATTACHMENT:
            *stage = Stage.DEPTH_STENCIL;
            *hazard = Hazard.DEPTH_READ_WRITE;
            *layout = TextureLayout.DEPTH_STENCIL;
        case TextureUse.PRESENT:
            *stage = Stage.COLOR_ATTACHMENT;
            *hazard = Hazard.NONE;
            *layout = TextureLayout.PRESENT;
        default:
            return INVALID_ARGUMENT~;
    }
}

<* Construct a common whole-texture transition without recording it.
   UNDEFINED is source-only; use a raw TextureBarrier for uncommon tuples. *>
fn TextureBarrier? texture_transition(
    TextureHandle texture,
    TextureUse before,
    TextureUse after
) {
    if (after == TextureUse.UNDEFINED) return INVALID_ARGUMENT~;
    TextureBarrier barrier = { .texture = texture };
    resolve_texture_use(
        use:    before,
        stage:  &barrier.before_stage,
        hazard: &barrier.before_hazard,
        layout: &barrier.old_layout,
    )!;
    resolve_texture_use(
        use:    after,
        stage:  &barrier.after_stage,
        hazard: &barrier.after_hazard,
        layout: &barrier.new_layout,
    )!;
    return barrier;
}

<* Create a timeline semaphore. Ownership: release with destroy_semaphore. *>
fn SemaphoreHandle? create_semaphore(Device* device, SemaphoreDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    SemaphoreHandle semaphore = operation.vtable.create_semaphore(device, desc)!;
    retain_device_children(&operation);
    return semaphore;
}

<* Counterpart of create_semaphore. *>
fn void? destroy_semaphore(Device* device, SemaphoreHandle semaphore) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, semaphore.owner, ResourceOwnerKind.SEMAPHORE)!;
    operation.vtable.destroy_semaphore(device, semaphore)!;
    release_device_children(&operation);
}

<* Wait on the host until the timeline reaches `value`, up to `timeout_ns`
   (0 polls without blocking). Faults WAIT_TIMEOUT — safe to retry — if
   `timeout_ns` elapses first. *>
fn void? wait_semaphore(Device* device, SemaphoreHandle semaphore, SemaphoreValue value, ulong timeout_ns) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, semaphore.owner, ResourceOwnerKind.SEMAPHORE)!;
    return operation.vtable.wait_semaphore(device, semaphore, value, timeout_ns);
}
