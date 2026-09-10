module gpu_shaders;

import std::io::file;
import std::io::path;

const String TOOL_ARGV0 = "build/gpu_shaders";

macro void @expect_cli_fault(String[] args, fault expected) {
    fault err = @catch(run_checked(args, TOOL_ARGV0));
    assert(err == expected, "expected %s, got %s", expected, err);
}

fn void missing_flags_usage_error() @test {
    @expect_cli_fault({ "test/golden/full.abi" }, USAGE_ERROR);
}

fn void unknown_flag_usage_error() @test {
    @expect_cli_fault({ "--bogus", "x" }, USAGE_ERROR);
}

fn void flag_without_value_usage_error() @test {
    @expect_cli_fault({ "--module" }, USAGE_ERROR);
}

fn void no_input_usage_error() @test {
    @expect_cli_fault({ "--check" }, USAGE_ERROR);
}

fn void abi_dir_with_files_usage_error() @test {
    @expect_cli_fault({
        "--abi-dir", "test/golden",
        "test/golden/full.abi",
        "--module", "gpu",
        "--c3-out", "build/tmp",
        "--glsl-out", "build/tmp",
    }, USAGE_ERROR);
}

fn void spv_out_without_shader_dir_usage_error() @test {
    @expect_cli_fault({ "--spv-out", "build/tmp" }, USAGE_ERROR);
}

fn void abi_dir_without_module_usage_error() @test {
    @expect_cli_fault({
        "--abi-dir", "test/golden",
        "--c3-out", "build/tmp",
        "--glsl-out", "build/tmp",
    }, USAGE_ERROR);
}

fn void unreadable_file_io_error() @test {
    @expect_cli_fault({
        "--module", "gpu",
        "--c3-out", "/dev/null",
        "--glsl-out", "/dev/null",
        "test/golden/does_not_exist.abi",
    }, IO_ERROR);
}

fn void missing_abi_dir_io_error() @test {
    @expect_cli_fault({
        "--abi-dir", "build/tmp/does_not_exist",
        "--module", "gpu",
        "--c3-out", "build/tmp",
        "--glsl-out", "build/tmp",
    }, IO_ERROR);
}

fn void compile_only_arguments_parse() @test {
    Options options = parse_args({ "--shader-dir", "shaders", "--include", "a", "--include", "b" })!!;
    defer options.free();
    assert(!options.has_abi_input());
    assert(options.shader_dir == "shaders");
    assert(options.includes.len() == 2);
    assert(options.includes[1] == "b");
}

fn String write_schema_dir(String dir) {
    path::mkdir(dir, recursive: true)!!;
    file::save(dir.tconcat("/beta.abi"), "abi beta;\nroot BetaRoot {\n    GpuAddress data_gpu;\n}\n")!!;
    file::save(dir.tconcat("/alpha.abi"), "abi alpha;\nstruct Alpha {\n    uint value;\n}\n")!!;
    return dir;
}

fn void directory_mode_emits_one_pair_per_schema() @test {
    String abi_dir = write_schema_dir("build/tmp/dir_mode/abi");
    String c3_out = "build/tmp/dir_mode/src";
    String glsl_out = "build/tmp/dir_mode/shaders/generated";
    run_checked({
        "--abi-dir", abi_dir,
        "--module", "dir_test",
        "--c3-out", c3_out,
        "--glsl-out", glsl_out,
    }, TOOL_ARGV0)!!;

    char[] alpha_c3 = file::load(mem, c3_out.tconcat("/alpha_abi.c3"))!!;
    defer free(alpha_c3.ptr);
    char[] beta_c3 = file::load(mem, c3_out.tconcat("/beta_abi.c3"))!!;
    defer free(beta_c3.ptr);
    char[] alpha_glsl = file::load(mem, glsl_out.tconcat("/alpha_abi.glsl"))!!;
    defer free(alpha_glsl.ptr);
    char[] beta_glsl = file::load(mem, glsl_out.tconcat("/beta_abi.glsl"))!!;
    defer free(beta_glsl.ptr);

    assert(((String)alpha_c3).contains("module dir_test;"));
    assert(((String)alpha_c3).contains("struct Alpha {"));
    assert(((String)beta_c3).contains("module dir_test;"));
    assert(((String)beta_c3).contains("struct BetaRoot {"));
    assert(((String)alpha_glsl).contains("ALPHA_SHADER_ABI_GLSL"));
    assert(((String)beta_glsl).contains("buffer_reference"));

    run_checked({
        "--abi-dir", abi_dir,
        "--module", "dir_test",
        "--c3-out", c3_out,
        "--glsl-out", glsl_out,
        "--check",
    }, TOOL_ARGV0)!!;
}

fn void directory_mode_check_reports_every_stale_file() @test {
    String abi_dir = write_schema_dir("build/tmp/dir_check/abi");
    String c3_out = "build/tmp/dir_check/src";
    String glsl_out = "build/tmp/dir_check/shaders/generated";
    run_checked({
        "--abi-dir", abi_dir,
        "--module", "dir_test",
        "--c3-out", c3_out,
        "--glsl-out", glsl_out,
    }, TOOL_ARGV0)!!;
    file::save(c3_out.tconcat("/alpha_abi.c3"), "stale\n")!!;
    file::save(glsl_out.tconcat("/beta_abi.glsl"), "stale\n")!!;

    @expect_cli_fault({
        "--abi-dir", abi_dir,
        "--module", "dir_test",
        "--c3-out", c3_out,
        "--glsl-out", glsl_out,
        "--check",
    }, DRIFT_DETECTED);

    char[] beta_c3 = file::load(mem, c3_out.tconcat("/beta_abi.c3"))!!;
    defer free(beta_c3.ptr);
    assert(((String)beta_c3).contains("struct BetaRoot {"), "check mode must not rewrite outputs");
}

fn void list_files_is_sorted_and_filtered() @test {
    String dir = write_schema_dir("build/tmp/list/abi");
    file::save(dir.tconcat("/notes.txt"), "x")!!;
    StringList names = list_files(tmem, dir, "*.abi")!!;
    assert(names.len() == 2, "expected 2 entries, got %d", names.len());
    assert(names[0] == "alpha.abi");
    assert(names[1] == "beta.abi");
}

fn void list_files_empty_directory_io_error() @test {
    path::mkdir("build/tmp/empty", recursive: true)!!;
    fault err = @catch(list_files(tmem, "build/tmp/empty", "*.abi"));
    assert(err == IO_ERROR, "expected IO_ERROR, got %s", err);
}
