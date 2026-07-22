module gen_shader_abi;

import std::collections::list;
import std::collections::map;

alias FieldList  = List{Field};
alias ConstList  = List{ConstDecl};
alias TypeList   = List{TypeDecl};
alias StructList = List{StructDecl};

enum DeclKind {
    STRUCT,
    ROOT,
    PUSH,
    EXTERN_STRUCT,
}

enum TypeKind {
    SCALAR,
    VECTOR,
    SEMANTIC,
    STRUCT_REF,
}

struct TypeInfo {
    TypeKind kind;
    uint     size;
    uint     alignment;
    uint     scalar_width;
    bool     scalar_signed;
    bool     integer;
    String   c3_name;
    String   glsl_name;
}

struct Field {
    String type_name;
    String name;
    Loc    loc;
    Loc    type_loc;
    uint   offset;
}

struct ConstDecl {
    String type_name;
    String name;
    String value;
    bool   negative;
    Loc    loc;
}

struct TypeDecl {
    String name;
    String base_name;
    Loc    loc;
}

struct StructDecl {
    DeclKind  kind;
    String    name;
    Loc       loc;
    FieldList fields;
    uint      size;
    uint      alignment;
}

struct Schema {
    String     abi_name;
    Loc        abi_loc;
    ConstList  consts;
    TypeList   types;
    StructList structs;
}

struct TypeTable {
    HashMap{String, TypeInfo} entries;
    HashMap{String, Loc}      symbols;
}

fn void Schema.init(&self) {
    self.abi_name = "";
    self.consts.init(mem);
    self.types.init(mem);
    self.structs.init(mem);
}

fn void Schema.free(&self) {
    foreach (&decl : self.structs) decl.fields.free();
    self.consts.free();
    self.types.free();
    self.structs.free();
}

fn void TypeTable.init(&self) {
    self.entries.init(mem);
    self.symbols.init(mem);
    self.entries.set("uint",  { .kind = SCALAR, .size = 4, .alignment = 4, .scalar_width = 32, .integer = true, .c3_name = "uint",  .glsl_name = "uint" });
    self.entries.set("int",   { .kind = SCALAR, .size = 4, .alignment = 4, .scalar_width = 32, .scalar_signed = true, .integer = true, .c3_name = "int", .glsl_name = "int" });
    self.entries.set("float", { .kind = SCALAR, .size = 4, .alignment = 4, .scalar_width = 32, .c3_name = "float", .glsl_name = "float" });
    self.entries.set("u64",   { .kind = SCALAR, .size = 8, .alignment = 8, .scalar_width = 64, .integer = true, .c3_name = "ulong", .glsl_name = "uint64_t" });
    self.entries.set("vec2",  { .kind = VECTOR, .size = 8,  .alignment = 8,  .scalar_width = 32, .c3_name = "Vec2f", .glsl_name = "vec2" });
    self.entries.set("vec4",  { .kind = VECTOR, .size = 16, .alignment = 16, .scalar_width = 32, .c3_name = "Vec4f", .glsl_name = "vec4" });
    self.entries.set("GpuAddress",   { .kind = SEMANTIC, .size = 8, .alignment = 8, .scalar_width = 64, .integer = true, .c3_name = "GpuAddress", .glsl_name = "uint64_t" });
    self.entries.set("TextureIndex", { .kind = SEMANTIC, .size = 4, .alignment = 4, .scalar_width = 32, .integer = true, .c3_name = "TextureIndex", .glsl_name = "uint" });
    self.entries.set("SamplerIndex", { .kind = SEMANTIC, .size = 4, .alignment = 4, .scalar_width = 32, .integer = true, .c3_name = "SamplerIndex", .glsl_name = "uint" });
}

fn void TypeTable.free(&self) {
    self.entries.free();
    self.symbols.free();
}

<* Register a schema-declared symbol; faults on any duplicate across files. *>
fn void? TypeTable.declare(&self, String name, Loc loc) {
    Loc? previous = self.symbols.get(name);
    if (try previous) {
        diagnostic(loc, "duplicate symbol '%s' (previously declared at %s:%d:%d)", name, previous.file, previous.line, previous.col);
        return DUPLICATE_SYMBOL~;
    }
    self.symbols.set(name, loc);
}

fn TypeInfo? TypeTable.lookup(&self, String name, Loc loc) {
    TypeInfo? info = self.entries.get(name);
    if (try info) return info;
    if (name == "vec3") {
        diagnostic(loc, "vec3 is not permitted in ABI structs (docs/shader_abi.md section 5); pack into vec4");
        return UNKNOWN_TYPE~;
    }
    diagnostic(loc, "unknown type '%s'", name);
    return UNKNOWN_TYPE~;
}

fn bool TypeTable.is_scalar(&self, String name) {
    TypeInfo? info = self.entries.get(name);
    if (try info) return info.kind == SCALAR;
    return false;
}
