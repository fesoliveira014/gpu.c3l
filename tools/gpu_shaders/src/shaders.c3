module gpu_shaders;

import std::collections::list;
import std::io;
import std::io::path;
import std::os::env;
import std::os::process;

const String GLSL_EXTENSION = ".glsl";
const String SPV_EXTENSION = ".spv";
const String DEFAULT_GLSLC = "glslc";
const String GLSLC_VARIABLE = "GLSLC";
const String LIBRARY_INCLUDE_RELATIVE = "../../../include/shaders";

const String[] STAGE_SUFFIXES = { ".comp", ".vert", ".frag", ".mesh", ".task", ".rgen", ".rmiss", ".rchit", ".rahit", ".rint", ".rcall" };
const String[] STAGE_NAMES    = { "compute", "vertex", "fragment", "mesh", "task", "rgen", "rmiss", "rchit", "rahit", "rint", "rcall" };

<* Compile every `*.glsl` directly inside `options.shader_dir` to `.spv`.
   Every stage is resolved before the first glslc call; the run stops at the
   first compiler failure. *>
fn void? compile_shaders(Options* options, String argv0) {
    StringList names = list_files(tmem, options.shader_dir, "*" +++ GLSL_EXTENSION)!;
    StringList stages;
    stages.tinit();
    foreach (name : names) stages.push(stage_for(name)!);

    StringList includes = include_dirs(options, argv0)!;
    String out_dir = options.spv_out.len > 0 ? options.spv_out : options.shader_dir;
    if (catch err = path::mkdir(out_dir, recursive: true)) {
        io::eprintfn("error: cannot create '%s' (%s)", out_dir, err);
        return IO_ERROR~;
    }

    String glslc = env::tget_var(GLSLC_VARIABLE) ?? DEFAULT_GLSLC;
    foreach (i, name : names) {
        String source = string::tformat("%s/%s", options.shader_dir, name);
        String output = string::tformat("%s/%s%s", out_dir, name[:name.len - GLSL_EXTENSION.len], SPV_EXTENSION);
        StringList command = glslc_command(
            allocator: tmem,
            glslc: glslc,
            stage: stages[i],
            includes: includes.array_view(),
            source: source,
            output: output,
        );
        run_glslc(command.array_view(), source)!;
        io::printfn("built %s", output);
    }
}

<* Stage name from the inner suffix of `<stem>.<stage>.glsl`. *>
fn String? stage_for(String file_name) {
    String stem = file_name[:file_name.len - GLSL_EXTENSION.len];
    foreach (i, suffix : STAGE_SUFFIXES) {
        if (stem.ends_with(suffix)) return STAGE_NAMES[i];
    }
    io::eprintfn("error: '%s' has no shader stage suffix (expected one of %s before %s)", file_name, STAGE_SUFFIXES, GLSL_EXTENSION);
    return SHADER_STAGE_ERROR~;
}

<* The library `include/shaders` directory, resolved from the invocation path
   of the tool: `<tool dir>/../../../include/shaders`. *>
fn String? library_include_dir(String argv0) {
    String dir = executable_dir(argv0)!;
    String candidate = string::tformat("%s/%s", dir, LIBRARY_INCLUDE_RELATIVE);
    if (!path::is_dir(path::tnew(candidate)!)) return NOT_FOUND~;
    return candidate;
}

fn String? executable_dir(String argv0) {
    usz end;
    bool found;
    foreach (usz i, c : argv0) {
        if (c == '/' || c == '\\') {
            end = i;
            found = true;
        }
    }
    if (!found) return NOT_FOUND~;
    return argv0[:end];
}

fn StringList? include_dirs(Options* options, String argv0) {
    StringList dirs;
    dirs.tinit();
    String? library = library_include_dir(argv0);
    if (try library) {
        dirs.push(library);
    } else if (options.includes.len() == 0) {
        io::eprintfn("error: library include directory not found at '%s' from '%s'; pass --include <gpu.c3l>/include/shaders", LIBRARY_INCLUDE_RELATIVE, argv0);
        return USAGE_ERROR~;
    }
    foreach (dir : options.includes) dirs.push(dir);
    return dirs;
}

fn StringList glslc_command(Allocator allocator, String glslc, String stage, String[] includes, String source, String output) {
    StringList command;
    command.init(allocator);
    command.push(glslc);
    command.push(string::format(allocator, "-fshader-stage=%s", stage));
    command.push("--target-env=vulkan1.3");
    foreach (dir : includes) {
        command.push("-I");
        command.push(dir);
    }
    command.push(source);
    command.push("-o");
    command.push(output);
    return command;
}

fn void? run_glslc(String[] command, String source) {
    int? status = process::run(command, out: process::INHERIT, err: process::INHERIT);
    if (catch err = status) {
        io::eprintfn("error: cannot start '%s' (%s); set %s or add it to PATH", command[0], err, GLSLC_VARIABLE);
        return TOOL_NOT_FOUND~;
    }
    if (status != 0) {
        io::eprintfn("error: %s failed on '%s' (exit %d)", command[0], source, status);
        return SHADER_COMPILE_ERROR~;
    }
}
