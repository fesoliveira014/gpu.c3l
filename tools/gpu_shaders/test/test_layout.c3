module gpu_shaders;

fn void compute_root_matches_doc_example() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        root ComputeRoot {
            GpuAddress input_gpu;
            GpuAddress output_gpu;
            uint count;
            uint _pad0;
            uint _pad1;
            uint _pad2;
        }
    `, &schema, &types)!!;
    StructDecl* decl = schema.structs.get_ref(0);
    assert(decl.size == 32);
    assert(decl.alignment == 8);
    assert(decl.fields.get_ref(1).offset == 8);
    assert(decl.fields.get_ref(2).offset == 16);
}

fn void material_matches_doc_example() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        struct Material {
            vec4 base_color_factor;
            vec4 emissive_factor_strength;
            TextureIndex albedo_texture;
            TextureIndex normal_texture;
            TextureIndex roughness_texture;
            SamplerIndex material_sampler;
            uint flags;
            uint _pad0;
            uint _pad1;
            uint _pad2;
        }
    `, &schema, &types)!!;
    StructDecl* decl = schema.structs.get_ref(0);
    assert(decl.size == 64);
    assert(decl.alignment == 16);
    assert(decl.fields.get_ref(1).offset == 16);
    assert(decl.fields.get_ref(2).offset == 32);
    assert(decl.fields.get_ref(6).offset == 48);
}

fn void acceleration_structure_index_is_builtin_semantic() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        struct RayRoot {
            AccelerationStructureIndex scene;
            uint _pad0;
            uint _pad1;
            uint _pad2;
        }
    `, &schema, &types)!!;
    StructDecl* decl = schema.structs.get_ref(0);
    assert(decl.size == 16);
    TypeInfo info = types.entries.get("AccelerationStructureIndex")!!;
    assert(info.kind == TypeKind.SEMANTIC);
    assert(info.size == 4);
    assert(info.c3_name == "AccelerationStructureIndex");
    assert(info.glsl_name == "uint");
}

fn void instance_vec_layout() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        struct Instance {
            vec2 pos;
            float scale;
            float _pad0;
            vec4 color;
        }
    `, &schema, &types)!!;
    StructDecl* decl = schema.structs.get_ref(0);
    assert(decl.size == 32);
    assert(decl.alignment == 16);
    assert(decl.fields.get_ref(1).offset == 8);
    assert(decl.fields.get_ref(3).offset == 16);
}

fn void interior_gap_faults() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    fault err = @catch(build_schema(`
        abi t;
        struct Bad {
            float a;
            vec4 b;
        }
    `, &schema, &types));
    assert(err == LAYOUT_ERROR, "expected LAYOUT_ERROR, got %s", err);
}

fn void trailing_gap_faults() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    fault err = @catch(build_schema(`
        abi t;
        struct Bad {
            vec4 a;
            uint b;
        }
    `, &schema, &types));
    assert(err == LAYOUT_ERROR, "expected LAYOUT_ERROR, got %s", err);
}

fn void u64_after_uint_faults() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    fault err = @catch(build_schema(`
        abi t;
        struct Bad {
            uint a;
            u64 b;
        }
    `, &schema, &types));
    assert(err == LAYOUT_ERROR, "expected LAYOUT_ERROR, got %s", err);
}

fn void nested_struct_alignment_propagates() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        struct Inner {
            vec4 v;
        }
        root Outer {
            GpuAddress a;
            uint b;
            uint _pad0;
            Inner inner;
        }
    `, &schema, &types)!!;
    StructDecl* outer = schema.structs.get_ref(1);
    assert(outer.size == 32);
    assert(outer.alignment == 16);
    assert(outer.fields.get_ref(3).offset == 16);
}

fn void mat4_layouts_and_padding() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        struct Leading {
            mat4 m;
        }
        struct Padded {
            float a;
            uint  _pad0;
            uint  _pad1;
            uint  _pad2;
            mat4  m;
        }
    `, &schema, &types)!!;
    StructDecl* leading = schema.structs.get_ref(0);
    assert(leading.size == 64 && leading.alignment == 16);
    StructDecl* padded = schema.structs.get_ref(1);
    assert(padded.fields.get_ref(4).offset == 16 && padded.size == 80);

    @expect_fault(`
        abi t;
        struct Bad {
            float a;
            mat4  m;
        }
    `, LAYOUT_ERROR);
}

fn void oversized_array_rejected() @test {
    @expect_fault(`
        abi t;
        struct Huge {
            uint[1073741824] values;
        }
    `, LAYOUT_ERROR);
    @expect_fault(`
        abi t;
        struct Huge {
            vec4[268435455] a;
            vec4[2]         b;
        }
    `, LAYOUT_ERROR);
}

fn void array_strides_follow_element_alignment() @test {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(`
        abi t;
        struct Elem {
            u64  a;
            u64  b;
            uint c;
            uint d;
        }
        struct Arrays {
            float[8] floats;
            vec2[3]  pairs;
            uint     _pad0;
            uint     _pad1;
            vec4[4]  planes;
            mat4[2]  bones;
            Elem[2]  elems;
        }
    `, &schema, &types)!!;
    StructDecl* elem = schema.structs.get_ref(0);
    assert(elem.size == 24 && elem.alignment == 8);
    StructDecl* arrays = schema.structs.get_ref(1);
    assert(arrays.fields.get_ref(0).offset == 0);
    assert(arrays.fields.get_ref(1).offset == 32);
    assert(arrays.fields.get_ref(4).offset == 64);
    assert(arrays.fields.get_ref(5).offset == 128);
    assert(arrays.fields.get_ref(6).offset == 256);
    assert(arrays.size == 304 && arrays.alignment == 16);

    @expect_fault(`
        abi t;
        struct Bad {
            vec2[3] pairs;
            vec4[4] planes;
        }
    `, LAYOUT_ERROR);
}
