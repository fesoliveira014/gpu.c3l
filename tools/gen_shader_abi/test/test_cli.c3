module gen_shader_abi;

macro void @expect_cli_fault(String[] args, fault expected) {
    fault err = @catch(run_checked(args));
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

fn void unreadable_file_io_error() @test {
    @expect_cli_fault({
        "--module", "gpu",
        "--c3-out", "/dev/null",
        "--glsl-out", "/dev/null",
        "test/golden/does_not_exist.abi",
    }, IO_ERROR);
}
