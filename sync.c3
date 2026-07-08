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

<* Sets the timeline to `value` once the submit's work clears `stage`. *>
struct SemaphoreSignal {
    SemaphoreHandle semaphore;
    SemaphoreValue  value;
    Stage           stage;
}

<* Create a timeline semaphore. Ownership: release with destroy_semaphore. *>
fn SemaphoreHandle? create_semaphore(Device* device, SemaphoreDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.create_semaphore(device, desc);
}

<* Counterpart of create_semaphore. *>
fn void? destroy_semaphore(Device* device, SemaphoreHandle semaphore) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.destroy_semaphore(device, semaphore);
}

<* Wait on the host until the timeline reaches `value`, up to `timeout_ns`
   (0 polls without blocking). Faults WAIT_TIMEOUT — safe to retry — if
   `timeout_ns` elapses first. *>
fn void? wait_semaphore(Device* device, SemaphoreHandle semaphore, SemaphoreValue value, ulong timeout_ns) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.wait_semaphore(device, semaphore, value, timeout_ns);
}
