module gen_shader_abi;

<* Lex + parse + layout a schema source into the given schema/type table. *>
fn void? build_schema(String source, Schema* schema, TypeTable* types, String file = "test.abi") {
    Lexer lexer;
    lexer.init(source, file);
    TokenList tokens;
    tokens.init(mem);
    defer tokens.free();
    lexer.tokenize(&tokens)!;
    Parser parser;
    parser.init(tokens.array_view());
    parser.parse_file(schema, types)!;
    compute_layouts(schema, types)!;
}
