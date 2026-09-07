module gen_shader_abi;

macro void @expect_fault(String source, fault expected) {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    fault err = @catch(build_schema(source, &schema, &types));
    assert(err == expected, "expected %s, got %s", expected, err);
}

fn void vec3_rejected() @test {
    @expect_fault(`
        abi t;
        struct Bad {
            vec3 normal;
        }
    `, UNKNOWN_TYPE);
}

fn void array_suffix_form_rejected() @test {
    @expect_fault(`
        abi t;
        struct Bad {
            float vals[4];
        }
    `, PARSE_ERROR);
}

fn void array_prefix_form_accepted() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        struct Ok {
            float[4] vals;
            mat4     m;
        }
    `, &schema, &types)!!;
    StructDecl* decl = schema.structs.get_ref(0);
    assert(decl.fields.get_ref(0).array_len == 4);
    assert(decl.fields.get_ref(1).array_len == 0);
}

fn void nested_array_rejected() @test {
    @expect_fault(`
        abi t;
        struct Bad {
            float[4][2] vals;
        }
    `, PARSE_ERROR);
}

fn void zero_length_array_rejected() @test {
    @expect_fault(`
        abi t;
        struct Bad {
            float[0] vals;
        }
    `, PARSE_ERROR);
}

fn void mat3_rejected() @test {
    @expect_fault(`
        abi t;
        struct Bad {
            mat3 basis;
        }
    `, UNKNOWN_TYPE);
    @expect_fault(`
        abi t;
        struct Bad {
            mat4x3 basis;
        }
    `, UNKNOWN_TYPE);
}

fn void push_aggregate_rejected() @test {
    @expect_fault(`
        abi t;
        push Bad {
            GpuAddress root_gpu;
            uint[2]    ids;
        }
    `, LAYOUT_ERROR);
    @expect_fault(`
        abi t;
        push Bad {
            mat4 m;
        }
    `, LAYOUT_ERROR);
}

fn void duplicate_symbol_rejected() @test {
    @expect_fault(`
        abi t;
        struct Instance { uint a; }
        root Instance { uint b; }
    `, DUPLICATE_SYMBOL);
}

fn void unknown_type_rejected() @test {
    @expect_fault(`
        abi t;
        struct Bad { Missing a; }
    `, UNKNOWN_TYPE);
}

fn void non_scalar_const_rejected() @test {
    @expect_fault(`
        abi t;
        const vec4 BAD = 1;
    `, PARSE_ERROR);
}

fn void empty_struct_rejected() @test {
    @expect_fault(`
        abi t;
        struct Empty { }
    `, PARSE_ERROR);
}

fn void abi_name_mismatch_across_files() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema("abi one; struct A { uint x; }", &schema, &types, file: "one.abi")!!;
    fault err = @catch(build_schema("abi two; struct B { uint y; }", &schema, &types, file: "two.abi"));
    assert(err == PARSE_ERROR, "expected PARSE_ERROR, got %s", err);
}

fn void cross_file_duplicate() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema("abi t; struct A { uint x; }", &schema, &types, file: "one.abi")!!;
    fault err = @catch(build_schema("abi t; struct A { uint y; }", &schema, &types, file: "two.abi"));
    assert(err == DUPLICATE_SYMBOL, "expected DUPLICATE_SYMBOL, got %s", err);
}

fn void keywords_valid_as_names() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        struct S {
            uint type;
            uint root;
            uint push;
            uint _pad0;
        }
    `, &schema, &types)!!;
    assert(schema.structs.get_ref(0).fields.get_ref(0).name == "type");
}

fn void const_out_of_range_rejected() @test {
    @expect_fault(`
        abi t;
        const uint TOO_BIG = 4294967296;
    `, PARSE_ERROR);
    @expect_fault(`
        abi t;
        const uint NEGATIVE = -1;
    `, PARSE_ERROR);
    @expect_fault(`
        abi t;
        const int FRACTIONAL = 1.5;
    `, PARSE_ERROR);
}

fn void const_in_range_accepted() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        const uint MAX_UINT = 4294967295;
        const int MIN_INT = -2147483648;
        const float HALF = 0.5;
    `, &schema, &types)!!;
    assert(schema.consts.len() == 3);
}

fn void semantic_type_registers() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        type MaterialIndex : uint;
        struct M {
            MaterialIndex index;
            uint _pad0;
            uint _pad1;
            uint _pad2;
        }
    `, &schema, &types)!!;
    StructDecl* decl = schema.structs.get_ref(0);
    assert(decl.size == 16);
    TypeInfo info = types.entries.get("MaterialIndex")!!;
    assert(info.kind == TypeKind.SEMANTIC);
    assert(info.glsl_name == "uint");
}

fn void glsl_reserved_names_rejected() @test {
    // Field name colliding with a GLSL keyword.
    @expect_fault(`
        abi t;
        root R {
            SamplerIndex sampler;
            uint _pad0;
            uint _pad1;
            uint _pad2;
        }
    `, PARSE_ERROR);
    // Field name colliding with a builtin vector type.
    @expect_fault(`
        abi t;
        root R {
            uint vec2;
            uint _pad0;
            uint _pad1;
            uint _pad2;
        }
    `, PARSE_ERROR);
    // Declaration name colliding with a keyword.
    @expect_fault(`
        abi t;
        struct layout {
            uint value;
            uint _pad0;
            uint _pad1;
            uint _pad2;
        }
    `, PARSE_ERROR);
    // Constant name colliding with a builtin type.
    @expect_fault(`
        abi t;
        const uint sampler2D = 1;
    `, PARSE_ERROR);
    // Semantic type name colliding with a keyword.
    @expect_fault(`
        abi t;
        type buffer : uint;
    `, PARSE_ERROR);
}

fn void non_reserved_names_still_accepted() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        root R {
            SamplerIndex heap_sampler;
            uint sampler_count;
            uint _pad0;
            uint _pad1;
        }
    `, &schema, &types)!!;
    assert(schema.structs.get_ref(0).fields.get_ref(0).name == "heap_sampler");
}

fn void push_roles_parse_and_bare_push_keeps_header_meaning() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        push compute Tint {
            uint _pad0;
            uint _pad1;
            vec4 color;
        }
        push graphics Material {
            uint id;
            uint _pad0;
            uint _pad1;
            uint _pad2;
            vec4 albedo;
        }
        push Plain {
            GpuAddress root_gpu;
        }
    `, &schema, &types)!!;
    assert(schema.structs.get_ref(0).push_role == PushRole.COMPUTE);
    assert(schema.structs.get_ref(1).push_role == PushRole.GRAPHICS);
    assert(schema.structs.get_ref(2).push_role == PushRole.NONE);
    assert(schema.structs.get_ref(1).size == 32 && schema.structs.get_ref(1).payload_offset == 16);
}

fn void oversized_push_payload_rejected() @test {
    @expect_fault(`
        abi t;
        push compute Big {
            uint _pad0;
            uint _pad1;
            vec4 a;
            vec4 b;
            vec4 c;
            vec4 d;
            vec4 e;
            vec4 f;
            vec4 g;
            vec4 h;
        }
    `, LAYOUT_ERROR);
}

fn void push_payload_offsets_follow_the_header() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        push compute Scalars {
            uint a;
            uint b;
        }
        push compute Vectors {
            uint _pad0;
            uint _pad1;
            vec4 v;
        }
        push graphics Wide {
            vec4 a;
        }
    `, &schema, &types)!!;
    StructDecl* scalars = schema.structs.get_ref(0);
    assert(scalars.payload_offset == 8 && scalars.size == 8);
    assert(scalars.fields.get_ref(0).offset == 8 && scalars.fields.get_ref(1).offset == 12);
    StructDecl* vectors = schema.structs.get_ref(1);
    assert(vectors.payload_offset == 8 && vectors.size == 24);
    assert(vectors.fields.get_ref(2).offset == 16);
    StructDecl* wide = schema.structs.get_ref(2);
    assert(wide.payload_offset == 16 && wide.fields.get_ref(0).offset == 16 && wide.size == 16);
}

fn void push_payload_needs_explicit_leading_padding() @test {
    @expect_fault(`
        abi t;
        push compute Vectors {
            vec4 v;
        }
    `, LAYOUT_ERROR);
}

fn void push_payload_rejects_header_names() @test {
    @expect_fault(`
        abi t;
        push compute Bad {
            u64 root_gpu;
        }
    `, PARSE_ERROR);
    @expect_fault(`
        abi t;
        push graphics Bad {
            u64 fragment_root_gpu;
        }
    `, PARSE_ERROR);
}
