module gpu;

import std::core::mem;
import std::hash::fnv64a;

// Default DeviceDesc.pipeline_capacity when zero.
const uint MAX_PIPELINES                  = 256;
const uint MAX_COLOR_ATTACHMENTS          = 8;
const uint SPIRV_MAGIC @private           = 0x07230203;
const usz SPIRV_HEADER_SIZE @private      = 20;
const usz SHADER_ENTRY_POINT_CAP @private = 64;

<* Shader stage associated with SPIR-V code. *>
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

<* Borrowed SPIR-V preparation input; the bytes must match `stage`. *>
struct ShaderDesc {
    ShaderStage stage;
    char[]      spirv;
    // Null selects "main".
    ZString     entry_point;
    ZString     debug_name;
}

<* Borrowed shader IR with library-computed process-local identity. Keep the
   referenced bytes immutable and alive while using this value. *>
struct ShaderCode {
    ShaderStage stage;
    char[]      spirv;
    ZString     entry_point;
    ZString     debug_name;
    ulong       digest;
}

<* Depth test/write state; the zero value means depth off. *>
struct DepthState {
    bool      test_enable;
    bool      write_enable;
    CompareOp compare;
}

<* Immutable rasterizer state in graphics pipeline identity. *>
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
    ShaderCode shader;
    // At least RootPush::size, four-byte aligned, and no greater than
    // DeviceCaps.max_push_constant_size.
    uint         push_constant_size;
    ZString      debug_name;
}

<* Inputs to create_graphics_pipeline, targeting dynamic rendering. The layout
   is fixed: the global heap set plus a 16-byte vertex/fragment root push range. *>
struct GraphicsPipelineDesc {
    ShaderCode        vertex_shader;
    ShaderCode        fragment_shader;
    PrimitiveTopology topology;
    RasterState       raster;
    BlendState        blend;
    Format[]          color_formats;
    Format            depth_format;
    SampleCount       sample_count;
    ZString           debug_name;
}

fn bool ShaderCode.is_valid(self) {
    if (self.digest == 0
        || !shader_stage_valid(self.stage)
        || self.spirv.ptr == null
        || self.spirv.len < SPIRV_HEADER_SIZE
        || self.spirv.len % uint::size != 0
        || self.entry_point == null) return false;

    uint magic;
    mem::copy(&magic, self.spirv.ptr, uint::size);
    usz entry_point_len = self.entry_point.str_view().len;
    return magic == SPIRV_MAGIC
        && entry_point_len > 0
        && entry_point_len < SHADER_ENTRY_POINT_CAP;
}

fn bool ShaderCode.equals(self, ShaderCode other) @operator(==) {
    if (!self.is_valid() || !other.is_valid()) {
        return !self.is_valid()
            && !other.is_valid()
            && self.spirv.ptr == other.spirv.ptr
            && self.spirv.len == other.spirv.len;
    }
    return self.digest == other.digest
        && self.stage == other.stage
        && self.entry_point.str_view() == other.entry_point.str_view()
        && self.spirv.len == other.spirv.len
        && mem::equals(self.spirv, other.spirv);
}

fn bool shader_stage_valid(ShaderStage stage) @private {
    switch (stage) {
        case ShaderStage.COMPUTE:
        case ShaderStage.VERTEX:
        case ShaderStage.FRAGMENT:
            return true;
        default:
            return false;
    }
}

fn ulong shader_code_digest(char[] spirv) @private {
    ulong digest = fnv64a::hash(spirv);
    return digest == 0 ? 1 : digest;
}

<* Validate borrowed SPIR-V and compute its opaque process-local identity. *>
fn ShaderCode? prepare_shader_code(ShaderDesc* desc) {
    if (desc == null) return INVALID_ARGUMENT~;
    if (!shader_stage_valid(desc.stage)) return INVALID_ARGUMENT~;
    if (desc.spirv.ptr == null
        || desc.spirv.len < SPIRV_HEADER_SIZE
        || desc.spirv.len % uint::size != 0) return SHADER_INVALID~;

    uint magic;
    mem::copy(&magic, desc.spirv.ptr, uint::size);
    if (magic != SPIRV_MAGIC) return SHADER_INVALID~;

    ZString entry_point = desc.entry_point == null ? "main" : desc.entry_point;
    usz entry_point_len = entry_point.str_view().len;
    if (entry_point_len == 0 || entry_point_len >= SHADER_ENTRY_POINT_CAP) {
        return INVALID_ARGUMENT~;
    }
    return {
        .stage       = desc.stage,
        .spirv       = desc.spirv,
        .entry_point = entry_point,
        .debug_name  = desc.debug_name,
        .digest      = shader_code_digest(desc.spirv),
    };
}

<* Create a compute pipeline. Identical descriptors may alias one backend
   pipeline under distinct handles. Ownership: release with destroy_pipeline,
   once per create. *>
fn PipelineHandle? create_compute_pipeline(Device* device, ComputePipelineDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    PipelineHandle pipeline = operation.vtable.create_compute_pipeline(device, desc)!;
    retain_device_children(&operation);
    return pipeline;
}

<* Create a graphics pipeline targeting dynamic rendering. The layout is fixed:
   the global heap set plus a 16-byte vertex/fragment root push range.
   Descriptors identical in immutable state alias one backend pipeline under
   distinct handles. Depth state is bound separately during command recording.
   At most MAX_COLOR_ATTACHMENTS color formats. Ownership:
   release with destroy_pipeline, once per create. *>
fn PipelineHandle? create_graphics_pipeline(Device* device, GraphicsPipelineDesc* desc) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (desc == null) return INVALID_ARGUMENT~;
    PipelineHandle pipeline = operation.vtable.create_graphics_pipeline(device, desc)!;
    retain_device_children(&operation);
    return pipeline;
}

<* Create compute pipelines transactionally. Output count must match descriptor
   count; an empty batch succeeds. A fault leaves every output unchanged. *>
fn void? create_compute_pipelines(
    Device* device,
    ComputePipelineDesc[] descs,
    PipelineHandle[] out_pipelines,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (out_pipelines.len != descs.len) return INVALID_ARGUMENT~;
    if (descs.len == 0) return;
    operation.vtable.create_compute_pipelines(device, descs, out_pipelines)!;
    for (usz i = 0; i < descs.len; i++) retain_device_children(&operation);
}

<* Create graphics pipelines transactionally. Output count must match descriptor
   count; an empty batch succeeds. A fault leaves every output unchanged. *>
fn void? create_graphics_pipelines(
    Device* device,
    GraphicsPipelineDesc[] descs,
    PipelineHandle[] out_pipelines,
) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    if (out_pipelines.len != descs.len) return INVALID_ARGUMENT~;
    if (descs.len == 0) return;
    operation.vtable.create_graphics_pipelines(device, descs, out_pipelines)!;
    for (usz i = 0; i < descs.len; i++) retain_device_children(&operation);
}

<* Release one returned handle. Descriptions sharing one backend pipeline are
   refcounted. *>
fn void? destroy_pipeline(Device* device, PipelineHandle pipeline) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    validate_resource_owner(&operation, pipeline.owner, ResourceOwnerKind.PIPELINE)!;
    operation.vtable.destroy_pipeline(device, pipeline)!;
    release_device_children(&operation);
}

<* Byte size of the driver pipeline cache blob. The cache grows as pipelines
   are built — re-query before each export. Must not be called concurrently
   with pipeline creation. *>
fn usz? get_pipeline_cache_size(Device* device) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_pipeline_cache_size(device);
}

<* Fill `out` with the driver pipeline cache blob and return the bytes
   written; faults INVALID_ARGUMENT if `out` is smaller than the current
   blob. Feed the blob to DeviceDesc.pipeline_cache_data to warm-start a
   later device. Must not be called concurrently with pipeline creation. *>
fn usz? get_pipeline_cache_data(Device* device, char[] out) {
    DeviceOperation operation = begin_device_operation(device)!;
    defer end_device_operation(&operation);
    return operation.vtable.get_pipeline_cache_data(device, out);
}
