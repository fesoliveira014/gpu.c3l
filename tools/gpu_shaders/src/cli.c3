module gpu_shaders;

import std::collections::list;
import std::io;
import std::io::file;
import std::io::path;
import std::sort;

alias StringList = List{String};
alias SourceList = List{char[]};

const String ABI_EXTENSION = ".abi";
const String ABI_OUTPUT_SUFFIX = "_abi";

struct Options {
    String     module_name;
    String     abi_dir;
    StringList files;
    String     c3_out;
    String     glsl_out;
    String     shader_dir;
    String     spv_out;
    StringList includes;
    bool       check;
}

fn void Options.init(&self) {
    self.files.init(mem);
    self.includes.init(mem);
}

fn void Options.free(&self) {
    self.files.free();
    self.includes.free();
}

fn bool Options.has_abi_input(&self) => self.abi_dir.len > 0 || self.files.len() > 0;

fn int run(String[] args, String argv0) {
    if (catch err = run_checked(args, argv0)) {
        if (err == USAGE_ERROR) print_usage();
        return 1;
    }
    return 0;
}

fn void? run_checked(String[] args, String argv0) => @pool() {
    Options options = parse_args(args)!;
    defer options.free();

    if (options.abi_dir.len > 0) {
        generate_directory(&options)!;
    } else if (options.files.len() > 0) {
        generate_files(
            module_name: options.module_name,
            files: options.files.array_view(),
            c3_out: options.c3_out,
            glsl_out: options.glsl_out,
            check: options.check,
        )!;
    }
    if (options.shader_dir.len > 0) compile_shaders(&options, argv0)!;
}

<* Merge every schema file into one namespace and write (or check) one C3
   and one GLSL output. *>
fn void? generate_files(String module_name, String[] files, String c3_out, String glsl_out, bool check) {
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

    foreach (path : files) {
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
    emit_c3(&schema, &types, module_name, &c3_text);

    DString glsl_text;
    glsl_text.init(mem);
    defer glsl_text.free();
    emit_glsl(&schema, &types, &glsl_text);

    if (check) {
        bool c3_clean = check_file(c3_out, c3_text.str_view());
        bool glsl_clean = check_file(glsl_out, glsl_text.str_view());
        if (!c3_clean || !glsl_clean) return DRIFT_DETECTED~;
        return;
    }

    save_file(c3_out, c3_text.str_view())!;
    save_file(glsl_out, glsl_text.str_view())!;
}

<* Every `*.abi` directly inside `options.abi_dir` is its own namespace and
   produces `<c3_out>/<stem>_abi.c3` and `<glsl_out>/<stem>_abi.glsl`. In check
   mode every schema is checked so each stale file is reported. *>
fn void? generate_directory(Options* options) {
    StringList names = list_files(tmem, options.abi_dir, "*" +++ ABI_EXTENSION)!;
    if (!options.check) {
        make_directory(options.c3_out)!;
        make_directory(options.glsl_out)!;
    }

    bool drift;
    foreach (name : names) {
        String stem = name[:name.len - ABI_EXTENSION.len];
        String schema_path = string::tformat("%s/%s", options.abi_dir, name);
        String[] files = { schema_path };
        fault err = @catch(generate_files(
            module_name: options.module_name,
            files: files,
            c3_out: string::tformat("%s/%s%s.c3", options.c3_out, stem, ABI_OUTPUT_SUFFIX),
            glsl_out: string::tformat("%s/%s%s.glsl", options.glsl_out, stem, ABI_OUTPUT_SUFFIX),
            check: options.check,
        ));
        if (err == DRIFT_DETECTED) {
            drift = true;
        } else if (err) {
            return err~;
        }
    }
    if (drift) return DRIFT_DETECTED~;
}

<* Names of the files in `dir` matching `mask`, sorted by byte order. *>
fn StringList? list_files(Allocator allocator, String dir, String mask) {
    PathList? entries = path::ls(allocator, path::tnew(dir)!, no_dirs: true, mask: mask);
    if (catch err = entries) {
        io::eprintfn("error: cannot list '%s' (%s)", dir, err);
        return IO_ERROR~;
    }
    StringList names;
    names.init(allocator);
    foreach (entry : entries) names.push(((String)entry).copy(allocator));
    quicksort(&names, &compare_names);
    if (names.len() == 0) {
        io::eprintfn("error: no '%s' files in '%s'", mask, dir);
        return IO_ERROR~;
    }
    return names;
}

fn int compare_names(String a, String b) => a.compare_to(b);

fn void? make_directory(String dir) {
    if (catch err = path::mkdir(dir, recursive: true)) {
        io::eprintfn("error: cannot create '%s' (%s)", dir, err);
        return IO_ERROR~;
    }
}

fn Options? parse_args(String[] args) {
    Options options;
    options.init();
    defer catch options.free();
    usz i = 0;
    while (i < args.len) {
        String arg = args[i];
        switch (arg) {
            case "--module":
                options.module_name = flag_value(args, &i)!;
            case "--abi-dir":
                options.abi_dir = flag_value(args, &i)!;
            case "--c3-out":
                options.c3_out = flag_value(args, &i)!;
            case "--glsl-out":
                options.glsl_out = flag_value(args, &i)!;
            case "--shader-dir":
                options.shader_dir = flag_value(args, &i)!;
            case "--spv-out":
                options.spv_out = flag_value(args, &i)!;
            case "--include":
                options.includes.push(flag_value(args, &i)!);
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
    validate_options(&options)!;
    return options;
}

fn void? validate_options(Options* options) {
    if (options.abi_dir.len > 0 && options.files.len() > 0) {
        io::eprintn("error: --abi-dir and schema files are exclusive");
        return USAGE_ERROR~;
    }
    if (!options.has_abi_input() && options.shader_dir.len == 0) {
        io::eprintn("error: pass --abi-dir, schema files, or --shader-dir");
        return USAGE_ERROR~;
    }
    if (options.has_abi_input() && (options.module_name.len == 0 || options.c3_out.len == 0 || options.glsl_out.len == 0)) {
        io::eprintn("error: --module, --c3-out, and --glsl-out are required with schema input");
        return USAGE_ERROR~;
    }
    if (options.spv_out.len > 0 && options.shader_dir.len == 0) {
        io::eprintn("error: --spv-out requires --shader-dir");
        return USAGE_ERROR~;
    }
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
        io::eprintfn("stale: '%s' is missing; regenerate with scripts/build_shaders.py", path);
        return false;
    }
    defer free(existing.ptr);
    if ((String)existing != expected) {
        io::eprintfn("stale: '%s' does not match the schema; regenerate with scripts/build_shaders.py", path);
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
    io::eprintn("usage: gpu_shaders [--module <c3-module> (--abi-dir <dir> | <files.abi...>) --c3-out <path> --glsl-out <path>]");
    io::eprintn("                   [--shader-dir <dir>] [--spv-out <dir>] [--include <dir>]... [--check]");
    io::eprintn("  --abi-dir    every *.abi in <dir> writes <c3-out>/<stem>_abi.c3 and <glsl-out>/<stem>_abi.glsl");
    io::eprintn("  files.abi    merged into one schema; --c3-out and --glsl-out are file paths");
    io::eprintn("  --shader-dir every *.glsl in <dir> compiles to <spv-out>/<stem>.spv (default: next to the source)");
    io::eprintn("  --include    extra glslc include directory; the library include/shaders is found from the tool path");
    io::eprintn("  --check      report stale ABI outputs instead of writing them; shaders still compile");
}
