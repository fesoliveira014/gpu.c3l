module gen_shader_abi;

// GLSL 4.60 reserved words and builtin type names. Schema identifiers are
// emitted into GLSL verbatim, so any collision breaks every consumer shader
// with an unrelated-looking compile error; the generator rejects them at the
// schema line instead. Sorted for readability; lookup is linear (checked
// once per identifier at generation time).
const String[*] GLSL_RESERVED = {
    "active", "asm", "atomic_uint", "attribute", "bool", "break", "buffer",
    "bvec2", "bvec3", "bvec4", "case", "cast", "centroid", "class", "coherent",
    "common", "const", "continue", "default", "discard", "dmat2", "dmat2x2",
    "dmat2x3", "dmat2x4", "dmat3", "dmat3x2", "dmat3x3", "dmat3x4", "dmat4",
    "dmat4x2", "dmat4x3", "dmat4x4", "do", "double", "dvec2", "dvec3", "dvec4",
    "else", "enum", "extern", "external", "false", "filter", "fixed", "flat",
    "float", "for", "fvec2", "fvec3", "fvec4", "goto", "half", "highp", "hvec2",
    "hvec3", "hvec4", "if", "iimage1D", "iimage1DArray", "iimage2D",
    "iimage2DArray", "iimage2DMS", "iimage2DMSArray", "iimage2DRect",
    "iimage3D", "iimageBuffer", "iimageCube", "iimageCubeArray", "image1D",
    "image1DArray", "image2D", "image2DArray", "image2DMS", "image2DMSArray",
    "image2DRect", "image3D", "imageBuffer", "imageCube", "imageCubeArray",
    "in", "inline", "inout", "input", "int", "interface", "invariant",
    "isampler1D", "isampler1DArray", "isampler2D", "isampler2DArray",
    "isampler2DMS", "isampler2DMSArray", "isampler2DRect", "isampler3D",
    "isamplerBuffer", "isamplerCube", "isamplerCubeArray", "isubpassInput",
    "isubpassInputMS", "itexture1D", "itexture1DArray", "itexture2D",
    "itexture2DArray", "itexture2DMS", "itexture2DMSArray", "itexture2DRect",
    "itexture3D", "itextureBuffer", "itextureCube", "itextureCubeArray",
    "ivec2", "ivec3", "ivec4", "layout", "long", "lowp", "mat2", "mat2x2",
    "mat2x3", "mat2x4", "mat3", "mat3x2", "mat3x3", "mat3x4", "mat4", "mat4x2",
    "mat4x3", "mat4x4", "mediump", "namespace", "noinline", "noperspective",
    "out", "output", "partition", "patch", "precise", "precision", "public",
    "readonly", "resource", "restrict", "return", "sample", "sampler",
    "sampler1D", "sampler1DArray", "sampler1DArrayShadow", "sampler1DShadow",
    "sampler2D", "sampler2DArray", "sampler2DArrayShadow", "sampler2DMS",
    "sampler2DMSArray", "sampler2DRect", "sampler2DRectShadow",
    "sampler2DShadow", "sampler3D", "sampler3DRect", "samplerBuffer",
    "samplerCube", "samplerCubeArray", "samplerCubeArrayShadow",
    "samplerCubeShadow", "samplerShadow", "shared", "short", "sizeof",
    "smooth", "static", "struct", "subpassInput", "subpassInputMS",
    "subroutine", "superp", "switch", "template", "texture1D",
    "texture1DArray", "texture2D", "texture2DArray", "texture2DMS",
    "texture2DMSArray", "texture2DRect", "texture3D", "textureBuffer",
    "textureCube", "textureCubeArray", "this", "true", "typedef", "uimage1D",
    "uimage1DArray", "uimage2D", "uimage2DArray", "uimage2DMS",
    "uimage2DMSArray", "uimage2DRect", "uimage3D", "uimageBuffer",
    "uimageCube", "uimageCubeArray", "uint", "uniform", "union", "unsigned",
    "usampler1D", "usampler1DArray", "usampler2D", "usampler2DArray",
    "usampler2DMS", "usampler2DMSArray", "usampler2DRect", "usampler3D",
    "usamplerBuffer", "usamplerCube", "usamplerCubeArray", "using",
    "usubpassInput", "usubpassInputMS", "utexture1D", "utexture1DArray",
    "utexture2D", "utexture2DArray", "utexture2DMS", "utexture2DMSArray",
    "utexture2DRect", "utexture3D", "utextureBuffer", "utextureCube",
    "utextureCubeArray", "uvec2", "uvec3", "uvec4", "varying", "vec2", "vec3",
    "vec4", "void", "volatile", "while", "writeonly",
};

<* Reject identifiers that would break the generated GLSL; the C3 side has
   its own compiler to catch C3 keywords. *>
fn void? validate_glsl_name(String name, Loc loc) {
    foreach (reserved : GLSL_RESERVED) {
        if (name == reserved) {
            diagnostic(loc, "'%s' is a GLSL reserved word and would break generated shaders; rename it (e.g. heap_%s)", name, name);
            return PARSE_ERROR~;
        }
    }
}
