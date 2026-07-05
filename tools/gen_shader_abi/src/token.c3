module gen_shader_abi;

enum TokenKind {
    KW_ABI,
    KW_CONST,
    KW_TYPE,
    KW_STRUCT,
    KW_ROOT,
    KW_PUSH,
    KW_EXTERN,
    IDENT,
    NUMBER,
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,
    COLON,
    SEMICOLON,
    EQUALS,
    MINUS,
    END_OF_FILE,
}

struct Loc {
    String file;
    uint   line;
    uint   col;
}

struct Token {
    TokenKind kind;
    String    text;
    Loc       loc;
}
