module gpu;

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
