module gpu_shaders;

import std::io::file;
import std::io::path;
import std::os::env;

const String MISSING_GLSLC = "gpu_shaders_missing_glslc_binary";

fn void every_stage_suffix_maps() @test {
    String[] names = {
        "a.comp.glsl", "a.vert.glsl", "a.frag.glsl", "a.mesh.glsl", "a.task.glsl",
        "a.rgen.glsl", "a.rmiss.glsl", "a.rchit.glsl", "a.rahit.glsl", "a.rint.glsl", "a.rcall.glsl",
    };
    foreach (i, name : names) {
        String stage = stage_for(name)!!;
        assert(stage == STAGE_NAMES[i], "%s -> %s, expected %s", name, stage, STAGE_NAMES[i]);
    }
}

fn void unknown_suffix_is_stage_error() @test {
    fault err = @catch(stage_for("noise.glsl"));
    assert(err == SHADER_STAGE_ERROR, "expected SHADER_STAGE_ERROR, got %s", err);
    err = @catch(stage_for("a.compute.glsl"));
    assert(err == SHADER_STAGE_ERROR);
}

fn void glslc_command_argument_order() @test {
    String[] includes = { "inc/lib", "inc/extra" };
    StringList command = glslc_command(
        allocator: tmem,
        glslc: "glslc",
        stage: "compute",
        includes: includes,
        source: "shaders/a.comp.glsl",
        output: "out/a.comp.spv",
    );
    String[] expected = {
        "glslc", "-fshader-stage=compute", "--target-env=vulkan1.3",
        "-I", "inc/lib", "-I", "inc/extra",
        "shaders/a.comp.glsl", "-o", "out/a.comp.spv",
    };
    assert(command.len() == expected.len, "got %d arguments", command.len());
    foreach (i, arg : expected) assert(command[i] == arg, "argument %d: %s != %s", i, command[i], arg);
}

fn void executable_dir_handles_both_separators() @test {
    assert(executable_dir("tools/gpu_shaders/build/gpu_shaders")!! == "tools/gpu_shaders/build");
    assert(executable_dir("build\\gpu_shaders.exe")!! == "build");
    assert(executable_dir("./gpu_shaders")!! == ".");
    fault err = @catch(executable_dir("gpu_shaders"));
    assert(err == NOT_FOUND);
}

fn void library_include_dir_follows_tool_path() @test {
    path::mkdir("build/tmp/tree/include/shaders", recursive: true)!!;
    path::mkdir("build/tmp/tree/tools/gpu_shaders/build", recursive: true)!!;
    String found = library_include_dir("build/tmp/tree/tools/gpu_shaders/build/gpu_shaders")!!;
    assert(found == "build/tmp/tree/tools/gpu_shaders/build/../../../include/shaders", "got %s", found);

    path::mkdir("build/tmp/bare/tools/gpu_shaders/build", recursive: true)!!;
    fault err = @catch(library_include_dir("build/tmp/bare/tools/gpu_shaders/build/gpu_shaders"));
    assert(err == NOT_FOUND, "expected NOT_FOUND, got %s", err);
}

fn String write_shader_dir(String dir, String[] names) {
    path::mkdir(dir, recursive: true)!!;
    foreach (name : names) file::save(dir.tconcat("/").tconcat(name), "#version 460\nvoid main() {}\n")!!;
    return dir;
}

fn void missing_include_without_override_is_usage_error() @test {
    Options options;
    options.init();
    defer options.free();
    options.shader_dir = write_shader_dir("build/tmp/noinclude", { "a.comp.glsl" });
    fault err = @catch(compile_shaders(&options, "gpu_shaders"));
    assert(err == USAGE_ERROR, "expected USAGE_ERROR, got %s", err);
}

fn void unknown_suffix_compiles_nothing() @test {
    Options options;
    options.init();
    defer options.free();
    options.shader_dir = write_shader_dir("build/tmp/badsuffix", { "a.comp.glsl", "noise.glsl" });
    options.includes.push("build/tmp/badsuffix");
    env::set_var(GLSLC_VARIABLE, MISSING_GLSLC);
    defer env::clear_var(GLSLC_VARIABLE);
    fault err = @catch(compile_shaders(&options, "gpu_shaders"));
    assert(err == SHADER_STAGE_ERROR, "expected SHADER_STAGE_ERROR, got %s", err);
    assert(!file::exists("build/tmp/badsuffix/a.comp.spv"));
}

fn void missing_glslc_is_tool_not_found() @test {
    Options options;
    options.init();
    defer options.free();
    options.shader_dir = write_shader_dir("build/tmp/noglslc", { "a.comp.glsl" });
    options.includes.push("build/tmp/noglslc");
    options.spv_out = "build/tmp/noglslc/out";
    env::set_var(GLSLC_VARIABLE, MISSING_GLSLC);
    defer env::clear_var(GLSLC_VARIABLE);
    fault err = @catch(compile_shaders(&options, "gpu_shaders"));
    assert(err == TOOL_NOT_FOUND, "expected TOOL_NOT_FOUND, got %s", err);
    assert(path::is_dir(path::tnew("build/tmp/noglslc/out")!!), "output directory is created before compiling");
}
