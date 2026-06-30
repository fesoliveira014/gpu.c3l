module gpu;

// Timeline semaphore counter value.
typedef SemaphoreValue = ulong;

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
