module gen_shader_abi;

struct Parser {
    Token[] tokens;
    usz     pos;
}

fn void Parser.init(&self, Token[] tokens) {
    *self = { .tokens = tokens, .pos = 0 };
}

fn Token Parser.peek(&self) => self.tokens[self.pos];

fn Token Parser.advance(&self) {
    Token token = self.tokens[self.pos];
    if (token.kind != TokenKind.END_OF_FILE) self.pos++;
    return token;
}

fn Token? Parser.expect(&self, TokenKind kind, String what) {
    Token token = self.peek();
    if (token.kind != kind) {
        diagnostic(token.loc, "expected %s, got '%s'", what, token.text.len > 0 ? token.text : "end of file");
        return PARSE_ERROR~;
    }
    return self.advance();
}

<* Accept an identifier or any keyword — keywords are structural only at
   declaration position, so 'type', 'root' etc. remain valid names. *>
fn Token? Parser.expect_name(&self, String what) {
    Token token = self.peek();
    if (token.kind != TokenKind.IDENT && !is_keyword(token.kind)) {
        diagnostic(token.loc, "expected %s, got '%s'", what, token.text.len > 0 ? token.text : "end of file");
        return PARSE_ERROR~;
    }
    return self.advance();
}

<* Parse one schema file into the shared schema/type table (multi-file merge). *>
fn void? Parser.parse_file(&self, Schema* schema, TypeTable* types) {
    Token abi_kw = self.expect(TokenKind.KW_ABI, "'abi' header")!;
    Token name = self.expect_name("abi name")!;
    self.expect(TokenKind.SEMICOLON, "';'")!;
    if (schema.abi_name.len == 0) {
        schema.abi_name = name.text;
        schema.abi_loc = name.loc;
    } else if (schema.abi_name != name.text) {
        diagnostic(name.loc, "abi name '%s' does not match '%s' (declared at %s:%d:%d); all merged files must share one abi name",
            name.text, schema.abi_name, schema.abi_loc.file, schema.abi_loc.line, schema.abi_loc.col);
        return PARSE_ERROR~;
    }
    while (self.peek().kind != TokenKind.END_OF_FILE) {
        Token token = self.peek();
        switch (token.kind) {
            case KW_CONST:  self.parse_const(schema, types)!;
            case KW_TYPE:   self.parse_typedecl(schema, types)!;
            case KW_STRUCT: self.parse_struct(schema, types, DeclKind.STRUCT)!;
            case KW_ROOT:   self.parse_struct(schema, types, DeclKind.ROOT)!;
            case KW_PUSH:   self.parse_struct(schema, types, DeclKind.PUSH)!;
            case KW_EXTERN:
                self.advance();
                self.expect(TokenKind.KW_STRUCT, "'struct' after 'extern'")!;
                self.parse_struct_body(schema, types, DeclKind.EXTERN_STRUCT)!;
            default:
                diagnostic(token.loc, "expected declaration (const, type, struct, root, push, extern), got '%s'", token.text);
                return PARSE_ERROR~;
        }
    }
}

fn void? Parser.parse_const(&self, Schema* schema, TypeTable* types) {
    self.advance();
    Token type_name = self.expect(TokenKind.IDENT, "scalar type")!;
    if (!types.is_scalar(type_name.text)) {
        diagnostic(type_name.loc, "constants must be primitive-typed (uint, int, float, u64), got '%s'", type_name.text);
        return PARSE_ERROR~;
    }
    Token name = self.expect_name("constant name")!;
    validate_glsl_name(name.text, name.loc)!;
    self.expect(TokenKind.EQUALS, "'='")!;
    bool negative = false;
    if (self.peek().kind == TokenKind.MINUS) {
        self.advance();
        negative = true;
    }
    Token number = self.expect(TokenKind.NUMBER, "literal value")!;
    self.expect(TokenKind.SEMICOLON, "';'")!;
    validate_const_literal(type_name.text, number.text, negative, number.loc)!;
    types.declare(name.text, name.loc)!;
    schema.consts.push({
        .type_name = type_name.text,
        .name = name.text,
        .value = number.text,
        .negative = negative,
        .loc = name.loc,
    });
}

fn void? Parser.parse_typedecl(&self, Schema* schema, TypeTable* types) {
    self.advance();
    Token name = self.expect_name("type name")!;
    validate_glsl_name(name.text, name.loc)!;
    self.expect(TokenKind.COLON, "':'")!;
    Token base = self.expect(TokenKind.IDENT, "base scalar type")!;
    self.expect(TokenKind.SEMICOLON, "';'")!;
    if (!types.is_scalar(base.text)) {
        diagnostic(base.loc, "semantic type base must be a primitive scalar (uint, int, float, u64), got '%s'", base.text);
        return PARSE_ERROR~;
    }
    types.declare(name.text, name.loc)!;
    TypeInfo base_info = types.lookup(base.text, base.loc)!;
    types.entries.set(name.text, {
        .kind = TypeKind.SEMANTIC,
        .size = base_info.size,
        .alignment = base_info.alignment,
        .scalar_width = base_info.scalar_width,
        .scalar_signed = base_info.scalar_signed,
        .integer = base_info.integer,
        .c3_name = name.text,
        .glsl_name = base_info.glsl_name,
    });
    schema.types.push({ .name = name.text, .base_name = base.text, .loc = name.loc });
}

fn void? Parser.parse_struct(&self, Schema* schema, TypeTable* types, DeclKind kind) {
    self.advance();
    return self.parse_struct_body(schema, types, kind);
}

fn void? Parser.parse_struct_body(&self, Schema* schema, TypeTable* types, DeclKind kind) {
    Token name = self.expect_name("struct name")!;
    validate_glsl_name(name.text, name.loc)!;
    types.declare(name.text, name.loc)!;
    self.expect(TokenKind.LBRACE, "'{'")!;
    StructDecl decl = { .kind = kind, .name = name.text, .loc = name.loc };
    decl.fields.init(mem);
    defer catch decl.fields.free();
    while (self.peek().kind != TokenKind.RBRACE) {
        Token type_name = self.expect_name("field type")!;
        uint array_len = 0;
        if (self.peek().kind == TokenKind.LBRACKET) {
            self.advance();
            Token count = self.expect(TokenKind.NUMBER, "array length")!;
            array_len = parse_array_len(count)!;
            self.expect(TokenKind.RBRACKET, "']'")!;
            if (self.peek().kind == TokenKind.LBRACKET) {
                diagnostic(self.peek().loc, "nested arrays are not supported");
                return PARSE_ERROR~;
            }
        }
        Token field_name = self.expect_name("field name")!;
        validate_glsl_name(field_name.text, field_name.loc)!;
        if (self.peek().kind == TokenKind.LBRACKET) {
            diagnostic(self.peek().loc, "declare arrays as 'type[N] name', not 'type name[N]'");
            return PARSE_ERROR~;
        }
        self.expect(TokenKind.SEMICOLON, "';'")!;
        decl.fields.push({
            .type_name = type_name.text,
            .name = field_name.text,
            .loc = field_name.loc,
            .type_loc = type_name.loc,
            .array_len = array_len,
        });
    }
    self.advance();
    if (decl.fields.len() == 0) {
        diagnostic(name.loc, "struct '%s' has no fields", name.text);
        return PARSE_ERROR~;
    }
    schema.structs.push(decl);
}

fn bool is_keyword(TokenKind kind) {
    switch (kind) {
        case KW_ABI:
        case KW_CONST:
        case KW_TYPE:
        case KW_STRUCT:
        case KW_ROOT:
        case KW_PUSH:
        case KW_EXTERN:
            return true;
        default:
            return false;
    }
}

fn void? validate_const_literal(String type_name, String value, bool negative, Loc loc) {
    if (type_name == "float") {
        if (catch err = value.to_double()) {
            diagnostic(loc, "invalid float literal '%s' (%s)", value, err);
            return PARSE_ERROR~;
        }
        return;
    }
    if (value.contains(".")) {
        diagnostic(loc, "%s constant cannot have a fractional literal '%s'", type_name, value);
        return PARSE_ERROR~;
    }
    ulong? magnitude = value.to_ulong();
    if (catch err = magnitude) {
        diagnostic(loc, "integer literal '%s' does not fit in u64 (%s)", value, err);
        return PARSE_ERROR~;
    }
    switch (type_name) {
        case "uint":
            if (negative || magnitude > uint::max) {
                diagnostic(loc, "literal %s%s does not fit in uint", negative ? "-" : "", value);
                return PARSE_ERROR~;
            }
        case "int":
            ulong limit = negative ? (ulong)int::max + 1 : int::max;
            if (magnitude > limit) {
                diagnostic(loc, "literal %s%s does not fit in int", negative ? "-" : "", value);
                return PARSE_ERROR~;
            }
        case "u64":
            if (negative) {
                diagnostic(loc, "u64 constant cannot be negative");
                return PARSE_ERROR~;
            }
        default:
            unreachable();
    }
}

fn uint? parse_array_len(Token count) {
    ulong? value = count.text.to_ulong();
    if (catch value) {
        diagnostic(count.loc, "array length must be a positive integer, got '%s'", count.text);
        return PARSE_ERROR~;
    }
    if (value == 0 || value > uint::max) {
        diagnostic(count.loc, "array length must be at least 1 and fit in 32 bits, got '%s'", count.text);
        return PARSE_ERROR~;
    }
    return (uint)value;
}
