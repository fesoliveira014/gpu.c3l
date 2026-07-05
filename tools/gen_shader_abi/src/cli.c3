module gen_shader_abi;

import std::collections::list;
import std::io;
import std::io::file;

alias StringList = List{String};
alias SourceList = List{char[]};

struct Options {
    String     module_name;
    String     c3_out;
    String     glsl_out;
    bool       check;
    StringList files;
}

fn int run(String[] args) {
    if (catch err = run_checked(args)) {
        if (err == USAGE_ERROR) print_usage();
        return 1;
    }
    return 0;
}

fn void? run_checked(String[] args) {
    Options options = parse_args(args)!;
    defer options.files.free();

    Schema schema;
    schema.init();
    defer schema.free();
    TypeTable types;
    types.init();
    defer types.free();

    SourceList sources;
    sources.init(mem);
    defer {
        foreach (source : sources) free(source.ptr);
        sources.free();
    }

    foreach (path : options.files) {
        char[]? source = file::load(mem, path);
        if (catch err = source) {
            io::eprintfn("error: cannot read schema file '%s' (%s)", path, err);
            return IO_ERROR~;
        }
        sources.push(source);

        Lexer lexer;
        lexer.init((String)source, path);
        TokenList tokens;
        tokens.init(mem);
        defer tokens.free();
        lexer.tokenize(&tokens)!;

        Parser parser;
        parser.init(tokens.array_view());
        parser.parse_file(&schema, &types)!;
    }

    compute_layouts(&schema, &types)!;

    DString c3_text;
    c3_text.init(mem);
    defer c3_text.free();
    emit_c3(&schema, &types, options.module_name, &c3_text);

    DString glsl_text;
    glsl_text.init(mem);
    defer glsl_text.free();
    emit_glsl(&schema, &types, &glsl_text);

    if (options.check) {
        bool c3_clean = check_file(options.c3_out, c3_text.str_view());
        bool glsl_clean = check_file(options.glsl_out, glsl_text.str_view());
        if (!c3_clean || !glsl_clean) return DRIFT_DETECTED~;
        return;
    }

    save_file(options.c3_out, c3_text.str_view())!;
    save_file(options.glsl_out, glsl_text.str_view())!;
}

fn Options? parse_args(String[] args) {
    Options options;
    options.files.init(mem);
    defer catch options.files.free();
    usz i = 0;
    while (i < args.len) {
        String arg = args[i];
        switch (arg) {
            case "--module":
                options.module_name = flag_value(args, &i)!;
            case "--c3-out":
                options.c3_out = flag_value(args, &i)!;
            case "--glsl-out":
                options.glsl_out = flag_value(args, &i)!;
            case "--check":
                options.check = true;
            default:
                if (arg.len >= 2 && arg[0..1] == "--") {
                    io::eprintfn("error: unknown option '%s'", arg);
                    return USAGE_ERROR~;
                }
                options.files.push(arg);
        }
        i++;
    }
    if (options.module_name.len == 0 || options.c3_out.len == 0 || options.glsl_out.len == 0 || options.files.len() == 0) {
        io::eprintfn("error: --module, --c3-out, --glsl-out, and at least one schema file are required");
        return USAGE_ERROR~;
    }
    return options;
}

fn String? flag_value(String[] args, usz* i) {
    if (*i + 1 >= args.len) {
        io::eprintfn("error: option '%s' expects a value", args[*i]);
        return USAGE_ERROR~;
    }
    (*i)++;
    return args[*i];
}

fn bool check_file(String path, String expected) {
    char[]? existing = file::load(mem, path);
    if (catch err = existing) {
        io::eprintfn("stale: '%s' is missing; regenerate with scripts/gen_abi.sh", path);
        return false;
    }
    defer free(existing.ptr);
    if ((String)existing != expected) {
        io::eprintfn("stale: '%s' does not match the schema; regenerate with scripts/gen_abi.sh", path);
        return false;
    }
    return true;
}

fn void? save_file(String path, String content) {
    if (catch err = file::save(path, content)) {
        io::eprintfn("error: cannot write '%s' (%s)", path, err);
        return IO_ERROR~;
    }
}

fn void print_usage() {
    io::eprintn("usage: gen_shader_abi --module <c3-module> --c3-out <path> --glsl-out <path> [--check] <files.abi...>");
}
