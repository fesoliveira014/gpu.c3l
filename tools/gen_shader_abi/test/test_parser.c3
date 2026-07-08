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

fn void fixed_array_rejected() @test {
    @expect_fault(`
        abi t;
        struct Bad {
            float vals[4];
        }
    `, PARSE_ERROR);
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
