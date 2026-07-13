module gpu::compat;

const uint MAX_SHADERS           = 256;
// Default DeviceDesc.pipeline_capacity when zero.
const uint MAX_PIPELINES         = 256;
const uint MAX_COLOR_ATTACHMENTS = 8;

<* Pipeline stage a shader module targets. *>
enum ShaderStage : int {
    COMPUTE,
    VERTEX,
    FRAGMENT,
}

<* How draws assemble vertices into primitives. *>
enum PrimitiveTopology : int {
    TRIANGLES,
    LINES,
    POINTS,
}

<* Comparison function for depth testing. *>
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

<* Triangle facing discarded by the rasterizer. *>
enum CullMode : int {
    NONE,
    FRONT,
    BACK,
}

<* Winding order that counts as front-facing. *>
enum FrontFace : int {
    COUNTER_CLOCKWISE,
    CLOCKWISE,
}

<* Triangle rasterization: filled or wireframe. LINE requires
   DeviceCaps.line_polygon_mode. *>
enum PolygonMode : int {
    FILL,
    LINE,
}

<* Multiplier for the source or destination term of the blend equation. *>
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

<* Operator combining the weighted source and destination blend terms. *>
enum BlendOp : int {
    ADD,
    SUBTRACT,
    REVERSE_SUBTRACT,
    MIN,
    MAX,
}

<* Element width of an index buffer; indexed draws default to U32. *>
enum IndexType : int {
    U32,
    U16,
}

<* Inputs to create_shader; the SPIR-V bytes must match `stage`. *>
struct ShaderDesc {
    ShaderStage stage;
    char[]      spirv;
    // Null selects "main".
    ZString     entry_point;
    ZString     debug_name;
}

<* Depth test/write state; the zero value means depth off. *>
struct DepthState {
    bool      test_enable;
    bool      write_enable;
    CompareOp compare;
}

<* Rasterizer state; cull_mode and front_face apply per handle at draw time. *>
struct RasterState {
    CullMode    cull_mode;
    FrontFace   front_face;
    PolygonMode polygon_mode;
    // Any non-zero bias field enables depth bias.
    float       depth_bias_constant;
    float       depth_bias_slope;
    float       depth_bias_clamp;
}

<* Blend equation, applied to every color target alike; disabled (the zero
   value) writes source unmodified. *>
struct BlendState {
    bool        enable;
    BlendFactor src_color;
    BlendFactor dst_color;
    BlendOp     color_op;
    BlendFactor src_alpha;
    BlendFactor dst_alpha;
    BlendOp     alpha_op;
}

<* Inputs to create_compute_pipeline; the layout is the global heap set plus
   a push range of push_constant_size bytes. *>
struct ComputePipelineDesc {
    ShaderHandle shader;
    // At least RootPush::size, four-byte aligned, and no greater than
    // DeviceCaps.max_push_constant_size.
    uint         push_constant_size;
    ZString      debug_name;
}

<* Inputs to create_graphics_pipeline, targeting dynamic rendering. The layout
   is fixed: the global heap set plus a 16-byte vertex/fragment root push range. *>
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
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    if (desc == null) return INVALID_ARGUMENT~;
    return vtable.create_shader(device, desc);
}

<* Counterpart of create_shader; pipelines already built from it stay valid. *>
fn void? destroy_shader(Device* device, ShaderHandle shader) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.destroy_shader(device, shader);
}

<* Create a compute pipeline. Identical descriptors may alias one backend
   pipeline under distinct handles. Ownership: release with destroy_pipeline,
   once per create. *>
fn PipelineHandle? create_compute_pipeline(Device* device, ComputePipelineDesc* desc) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    if (desc == null) return INVALID_ARGUMENT~;
    return vtable.create_compute_pipeline(device, desc);
}

<* Create a graphics pipeline targeting dynamic rendering. The layout is fixed:
   the global heap set plus a 16-byte vertex/fragment root push range.
   Descriptors identical in immutable state alias one backend pipeline under
   distinct handles; raster cull/front-face and depth state apply per handle
   at draw time. At most MAX_COLOR_ATTACHMENTS color formats. Ownership:
   release with destroy_pipeline, once per create. *>
fn PipelineHandle? create_graphics_pipeline(Device* device, GraphicsPipelineDesc* desc) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    if (desc == null) return INVALID_ARGUMENT~;
    return vtable.create_graphics_pipeline(device, desc);
}

<* Counterpart of the pipeline creates, once per returned handle; descriptors aliasing one backend pipeline are refcounted. *>
fn void? destroy_pipeline(Device* device, PipelineHandle pipeline) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.destroy_pipeline(device, pipeline);
}

<* Byte size of the driver pipeline cache blob. The cache grows as pipelines
   are built — re-query before each export. Must not be called concurrently
   with pipeline creation. *>
fn usz? get_pipeline_cache_size(Device* device) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.get_pipeline_cache_size(device);
}

<* Fill `out` with the driver pipeline cache blob and return the bytes
   written; faults INVALID_ARGUMENT if `out` is smaller than the current
   blob. Feed the blob to DeviceDesc.pipeline_cache_data to warm-start a
   later device. Must not be called concurrently with pipeline creation. *>
fn usz? get_pipeline_cache_data(Device* device, char[] out) {
    BackendVTable* vtable = device_vtable(device);
    if (vtable == null) return INVALID_HANDLE~;
    return vtable.get_pipeline_cache_data(device, out);
}
