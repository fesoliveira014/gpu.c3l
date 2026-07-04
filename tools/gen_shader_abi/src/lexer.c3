module gen_shader_abi;

import std::collections::list;
import std::io;

alias TokenList = List{Token};

struct Lexer {
    String source;
    String file;
    usz    pos;
    uint   line;
    uint   col;
}

fn void Lexer.init(&self, String source, String file) {
    *self = { .source = source, .file = file, .pos = 0, .line = 1, .col = 1 };
}

fn Loc Lexer.loc(&self) => { .file = self.file, .line = self.line, .col = self.col };

fn char Lexer.peek(&self) => self.pos < self.source.len ? self.source[self.pos] : '\0';

fn char Lexer.peek_next(&self) => self.pos + 1 < self.source.len ? self.source[self.pos + 1] : '\0';

fn void Lexer.advance(&self) {
    if (self.pos >= self.source.len) return;
    if (self.source[self.pos] == '\n') {
        self.line++;
        self.col = 1;
    } else {
        self.col++;
    }
    self.pos++;
}

fn void Lexer.skip_trivia(&self) {
    while (true) {
        char c = self.peek();
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            self.advance();
            continue;
        }
        if (c == '/' && self.peek_next() == '/') {
            while (self.peek() != '\n' && self.peek() != '\0') self.advance();
            continue;
        }
        return;
    }
}

fn bool is_ident_start(char c) => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';

fn bool is_ident_char(char c) => is_ident_start(c) || (c >= '0' && c <= '9');

fn bool is_digit(char c) => c >= '0' && c <= '9';

fn TokenKind classify_word(String word) {
    switch (word) {
        case "abi":    return KW_ABI;
        case "const":  return KW_CONST;
        case "type":   return KW_TYPE;
        case "struct": return KW_STRUCT;
        case "root":   return KW_ROOT;
        case "push":   return KW_PUSH;
        case "extern": return KW_EXTERN;
        default:       return IDENT;
    }
}

<* Lex the whole source into tokens (END_OF_FILE terminated). *>
fn void? Lexer.tokenize(&self, TokenList* out) {
    while (true) {
        self.skip_trivia();
        Loc loc = self.loc();
        char c = self.peek();
        if (c == '\0') {
            out.push({ .kind = END_OF_FILE, .text = "", .loc = loc });
            return;
        }
        if (is_ident_start(c)) {
            usz start = self.pos;
            while (is_ident_char(self.peek())) self.advance();
            String word = self.source[start..self.pos - 1];
            out.push({ .kind = classify_word(word), .text = word, .loc = loc });
            continue;
        }
        if (is_digit(c)) {
            usz start = self.pos;
            while (is_digit(self.peek())) self.advance();
            if (self.peek() == '.' && is_digit(self.peek_next())) {
                self.advance();
                while (is_digit(self.peek())) self.advance();
            }
            out.push({ .kind = NUMBER, .text = self.source[start..self.pos - 1], .loc = loc });
            continue;
        }
        TokenKind kind;
        switch (c) {
            case '{': kind = LBRACE;
            case '}': kind = RBRACE;
            case '[': kind = LBRACKET;
            case ']': kind = RBRACKET;
            case ':': kind = COLON;
            case ';': kind = SEMICOLON;
            case '=': kind = EQUALS;
            case '-': kind = MINUS;
            default:
                diagnostic(loc, "unexpected character '%c'", c);
                return PARSE_ERROR~;
        }
        String text = self.source[self.pos..self.pos];
        self.advance();
        out.push({ .kind = kind, .text = text, .loc = loc });
    }
}

fn void diagnostic(Loc loc, String fmt, args...) {
    io::eprintf("%s:%d:%d: error: ", loc.file, loc.line, loc.col);
    io::eprintfn(fmt, ...args);
}
