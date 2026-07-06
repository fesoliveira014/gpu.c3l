module gpu;

const uint MAX_SHADERS           = 256;
const uint MAX_PIPELINES         = 256;
const uint MAX_COLOR_ATTACHMENTS = 8;

enum ShaderStage : int {
    COMPUTE,
    VERTEX,
    FRAGMENT,
}

enum PrimitiveTopology : int {
    TRIANGLES,
    LINES,
    POINTS,
}

enum CompareOp : int {
    NEVER,
    LESS,
    EQUAL,
    LESS_EQUAL,
    GREATER,
    NOT_EQUAL,
    GREATER_EQUAL,
    ALWAYS,
}

enum CullMode : int {
    NONE,
    FRONT,
    BACK,
}

enum FrontFace : int {
    COUNTER_CLOCKWISE,
    CLOCKWISE,
}

enum PolygonMode : int {
    FILL,
    LINE,
}

enum BlendFactor : int {
    ZERO,
    ONE,
    SRC_COLOR,
    ONE_MINUS_SRC_COLOR,
    SRC_ALPHA,
    ONE_MINUS_SRC_ALPHA,
    DST_ALPHA,
    ONE_MINUS_DST_ALPHA,
}

enum BlendOp : int {
    ADD,
    SUBTRACT,
    REVERSE_SUBTRACT,
    MIN,
    MAX,
}

enum IndexType : int {
    U32,
    U16,
}

struct ShaderDesc {
    ShaderStage stage;
    char[]      spirv;
    ZString     entry_point;
    ZString     debug_name;
}

struct DepthState {
    bool      test_enable;
    bool      write_enable;
    CompareOp compare;
}

struct RasterState {
    CullMode    cull_mode;
    FrontFace   front_face;
    PolygonMode polygon_mode;
    float       depth_bias_constant;
    float       depth_bias_slope;
    float       depth_bias_clamp;
}

struct BlendState {
    bool        enable;
    BlendFactor src_color;
    BlendFactor dst_color;
    BlendOp     color_op;
    BlendFactor src_alpha;
    BlendFactor dst_alpha;
    BlendOp     alpha_op;
}

struct ComputePipelineDesc {
    ShaderHandle shader;
    uint         push_constant_size;
    ZString      debug_name;
}

struct GraphicsPipelineDesc {
    ShaderHandle      vertex_shader;
    ShaderHandle      fragment_shader;
    PrimitiveTopology topology;
    DepthState        depth;
    RasterState       raster;
    BlendState        blend;
    Format[]          color_formats;
    Format            depth_format;
    ZString           debug_name;
}

<* Create a shader module from SPIR-V bytes. Ownership: release with destroy_shader. *>
fn ShaderHandle? create_shader(Device* device, ShaderDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.create_shader(device, desc);
}

fn void? destroy_shader(Device* device, ShaderHandle shader) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.destroy_shader(device, shader);
}

<* Create a compute pipeline. Identical descriptors may alias one backend
   pipeline under distinct handles. Ownership: release with destroy_pipeline,
   once per create. *>
fn PipelineHandle? create_compute_pipeline(Device* device, ComputePipelineDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.create_compute_pipeline(device, desc);
}

<* Create a graphics pipeline targeting dynamic rendering. The layout is fixed:
   the global heap set plus a 16-byte vertex/fragment root push range.
   Descriptors identical in immutable state alias one backend pipeline under
   distinct handles; raster cull/front-face and depth state apply per handle
   at draw time. At most MAX_COLOR_ATTACHMENTS color formats. Ownership:
   release with destroy_pipeline, once per create. *>
fn PipelineHandle? create_graphics_pipeline(Device* device, GraphicsPipelineDesc* desc) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.create_graphics_pipeline(device, desc);
}

fn void? destroy_pipeline(Device* device, PipelineHandle pipeline) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.destroy_pipeline(device, pipeline);
}

<* Byte size of the driver pipeline cache blob. The cache grows as pipelines
   are built — re-query before each export. Must not be called concurrently
   with pipeline creation. *>
fn usz? get_pipeline_cache_size(Device* device) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.get_pipeline_cache_size(device);
}

<* Fill `out` with the driver pipeline cache blob and return the bytes
   written; faults INVALID_ARGUMENT if `out` is smaller than the current
   blob. Feed the blob to DeviceDesc.pipeline_cache_data to warm-start a
   later device. Must not be called concurrently with pipeline creation. *>
fn usz? get_pipeline_cache_data(Device* device, char[] out) {
    if (device.vtable == null) return INVALID_ARGUMENT~;
    return device.vtable.get_pipeline_cache_data(device, out);
}
