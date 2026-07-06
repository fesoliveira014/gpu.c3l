module gpu;

// Timeline semaphore counter value.
typedef SemaphoreValue = ulong;

const uint MAX_SEMAPHORES = 256;

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

struct BufferBarrier {
    BufferHandle buffer;
    usz          offset;
    usz          size;
    Stage        before_stage;
    Stage        after_stage;
    Hazard       before_hazard;
    Hazard       after_hazard;
}

struct TextureBarrier {
    TextureHandle texture;
    Stage         before_stage;
    Stage         after_stage;
    Hazard        before_hazard;
    Hazard        after_hazard;
    TextureLayout old_layout;
    TextureLayout new_layout;
}

struct GlobalBarrier {
    Stage  before_stage;
    Stage  after_stage;
    Hazard before_hazard;
    Hazard after_hazard;
}

struct SemaphoreDesc {
    SemaphoreValue initial_value;
    ZString        debug_name;
}

struct SemaphoreWait {
    SemaphoreHandle semaphore;
    SemaphoreValue  value;
    Stage           stage;
}

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

<* Wait on the host until the timeline reaches `value`, up to `timeout_ns`. *>
fn void? wait_semaphore(Device* device, SemaphoreHandle semaphore, SemaphoreValue value, ulong timeout_ns) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.wait_semaphore(device, semaphore, value, timeout_ns);
}
