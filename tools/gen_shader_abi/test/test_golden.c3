module gen_shader_abi;

import std::io::file;

fn void generate(String source, String module_name, DString* c3_text, DString* glsl_text) {
    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();
    build_schema(source, &schema, &types)!!;
    emit_c3(&schema, &types, module_name, c3_text);
    emit_glsl(&schema, &types, glsl_text);
}

fn void golden_full_schema() @test {
    char[] source = file::load(mem, "test/golden/full.abi")!!;
    defer free(source.ptr);
    char[] expected_c3 = file::load(mem, "test/golden/full_expected.c3.txt")!!;
    defer free(expected_c3.ptr);
    char[] expected_glsl = file::load(mem, "test/golden/full_expected.glsl.txt")!!;
    defer free(expected_glsl.ptr);

    DString c3_text;
    c3_text.init(mem);
    defer c3_text.free();
    DString glsl_text;
    glsl_text.init(mem);
    defer glsl_text.free();
    generate((String)source, "gpu", &c3_text, &glsl_text);

    assert(c3_text.str_view() == (String)expected_c3, "generated C3 does not match golden file");
    assert(glsl_text.str_view() == (String)expected_glsl, "generated GLSL does not match golden file");
}

fn void generation_is_deterministic() @test {
    char[] source = file::load(mem, "test/golden/full.abi")!!;
    defer free(source.ptr);

    DString c3_a;
    c3_a.init(mem);
    defer c3_a.free();
    DString glsl_a;
    glsl_a.init(mem);
    defer glsl_a.free();
    generate((String)source, "gpu", &c3_a, &glsl_a);

    DString c3_b;
    c3_b.init(mem);
    defer c3_b.free();
    DString glsl_b;
    glsl_b.init(mem);
    defer glsl_b.free();
    generate((String)source, "gpu", &c3_b, &glsl_b);

    assert(c3_a.str_view() == c3_b.str_view());
    assert(glsl_a.str_view() == glsl_b.str_view());
}

fn void app_module_imports_gpu() @test {
    DString c3_text;
    c3_text.init(mem);
    defer c3_text.free();
    DString glsl_text;
    glsl_text.init(mem);
    defer glsl_text.free();
    generate(`
        abi app;
        root Root {
            GpuAddress data_gpu;
        }
    `, "my_app::abi", &c3_text, &glsl_text);
    assert(c3_text.str_view().contains("module my_app::abi;"));
    assert(c3_text.str_view().contains("import gpu;"));
}

fn void semantic_push_member_traits_follow_base_type() @test {
    DString c3_text;
    c3_text.init(mem);
    defer c3_text.free();
    DString glsl_text;
    glsl_text.init(mem);
    defer glsl_text.free();
    generate(`
        abi app;
        type Address : u64;
        push PushData {
            Address value;
        }
    `, "gpu", &c3_text, &glsl_text);
    String generated = c3_text.str_view();
    assert(generated.contains(".scalar_width  = 64"));
    assert(generated.contains(".scalar_signed = false"));
    assert(generated.contains(".integer       = true"));
}
