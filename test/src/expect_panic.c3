module gpu_test;

import std::core::builtin;
import libc;

struct ExpectedPanic @align(16) {
    libc::JmpBuf buf;
    char[512]    message;
    usz          message_len;
}

tlocal ExpectedPanic expected_panic;

fn void capture_expected_panic(String message, String file, String function, int line) {
    (void)file;
    (void)function;
    (void)line;
    usz len = message.len < expected_panic.message.len
        ? message.len
        : expected_panic.message.len;
    expected_panic.message[:len] = message[:len];
    expected_panic.message_len = len;
    libc::longjmp(&expected_panic.buf, 1);
}

<* Run a body that must panic and return its message, which stays valid only
   until the next panic on this thread. The previous panic handler is restored;
   a body that returns normally fails the test. Allocation-free, so it also
   works on threads without a temp allocator. *>
macro String @capture_panic(; @body) {
    builtin::PanicFn previous = builtin::panic;
    builtin::panic = &capture_expected_panic;
    if (libc::setjmp(&expected_panic.buf) == 0) {
        @body();
        builtin::panic = previous;
        abort("expected a panic");
    }
    builtin::panic = previous;
    return (String)expected_panic.message[:expected_panic.message_len];
}

<* Run a body that may panic and report whether it did; the message is then
   available from `last_panic_message`. Allocation-free. *>
macro bool @panicked(; @body) {
    builtin::PanicFn previous = builtin::panic;
    builtin::panic = &capture_expected_panic;
    if (libc::setjmp(&expected_panic.buf) == 0) {
        @body();
        builtin::panic = previous;
        return false;
    }
    builtin::panic = previous;
    return true;
}

<* The most recent captured panic message on this thread. *>
fn String last_panic_message() {
    return (String)expected_panic.message[:expected_panic.message_len];
}

<* Run a body that must panic and return a temporary copy of its message. *>
macro String @expect_panic(; @body) {
    String message = @capture_panic() {
        @body();
    };
    return message.tcopy();
}

<* Run a body that must panic with exactly `expected`. *>
macro void @assert_panics(String expected; @body) {
    String message = @capture_panic() {
        @body();
    };
    assert(message == expected, "unexpected panic message: %s", message);
}

fn void expect_panic_returns_abort_message() @test {
    String message = @expect_panic() {
        abort("boom %d", 7);
    };
    assert(message == "boom 7", "the panic message is returned");
}

fn void expect_panic_restores_handler() @test {
    builtin::PanicFn before = builtin::panic;
    (void)@expect_panic() {
        abort("restore");
    };
    assert(builtin::panic == before, "the previous panic handler is restored");
}

fn void expect_panic_keeps_outer_state() @test {
    int value = 41;
    value++;
    (void)@expect_panic() {
        abort("outer");
    };
    assert(value == 42, "locals assigned before the body survive the panic");
}
