module gpu;

import gpu::internal @public;
struct DiagnosticCapture {
    uint              calls;
    void*             user_data;
    DebugMessage      message;
}

fn void capture_diagnostic(DebugMessage* message, void* user_data) {
    DiagnosticCapture* capture = (DiagnosticCapture*)user_data;
    capture.calls++;
    capture.user_data = user_data;
    capture.message = *message;
}

fn void diagnostic_dispatch_is_null_safe() @test {
    DebugMessage message = {};
    gpu::internal::dispatch_debug_message(null, null, &message);
}

fn void null_command_owner_faults_as_invalid_handle() @test {
    fault command_result = @catch(end_commands((CommandList*)null));

    assert(
        command_result == INVALID_HANDLE,
        "null command token faults INVALID_HANDLE"
    );
}

fn void diagnostic_dispatch_delivers_complete_message_once() @test {
    DiagnosticCapture capture = {};
    DebugMessage message = {
        .severity             = DebugMessageSeverity.ERROR,
        .category             = { .public_contract, .validation },
        .operation            = "allocate_memory",
        .has_fault            = true,
        .public_fault         = INVALID_ARGUMENT,
        .resource             = {
            .kind                = DebugResourceKind.ALLOCATION,
            .has_public_identity = true,
            .index               = 17,
            .generation          = 3,
            .debug_name          = "diagnostic_allocation",
        },
        .rejected_field       = "AllocationDesc.size",
        .invariant            = "allocation size must be greater than zero",
        .backend_text         = "allocation size is zero",
        .validation_id_name   = "VUID-test",
        .validation_id_number = 42,
    };

    gpu::internal::dispatch_debug_message(&capture_diagnostic, &capture, &message);

    assert(capture.calls == 1, "dispatch invokes the callback exactly once");
    assert(capture.user_data == &capture, "dispatch preserves callback userdata");
    assert(capture.message.severity == DebugMessageSeverity.ERROR, "severity is preserved");
    assert(capture.message.category.public_contract, "public-contract category is preserved");
    assert(capture.message.category.validation, "validation category is preserved");
    assert(capture.message.operation.str_view() == "allocate_memory", "operation is preserved");
    assert(capture.message.has_fault, "public fault presence is preserved");
    assert(capture.message.public_fault == INVALID_ARGUMENT, "public fault is preserved");
    assert(capture.message.resource.kind == DebugResourceKind.ALLOCATION, "resource kind is preserved");
    assert(capture.message.resource.has_public_identity, "public identity presence is preserved");
    assert(capture.message.resource.index == 17, "resource index is preserved");
    assert(capture.message.resource.generation == 3, "resource generation is preserved");
    assert(capture.message.resource.debug_name.str_view() == "diagnostic_allocation", "debug name is preserved");
    assert(capture.message.rejected_field.str_view() == "AllocationDesc.size", "rejected field is preserved");
    assert(capture.message.invariant.str_view()
        == "allocation size must be greater than zero",
        "invariant is preserved");
    assert(capture.message.backend_text.str_view() == "allocation size is zero", "text is preserved");
    assert(capture.message.validation_id_name.str_view() == "VUID-test", "validation ID name is preserved");
    assert(capture.message.validation_id_number == 42, "validation ID number is preserved");
}

fn void diagnostic_dispatch_preserves_absent_fault() @test {
    DiagnosticCapture capture = {};
    DebugMessage message = {
        .severity     = DebugMessageSeverity.WARNING,
        .category     = { .performance },
        .operation    = "submit",
        .backend_text = "backend warning",
    };

    gpu::internal::dispatch_debug_message(&capture_diagnostic, &capture, &message);

    assert(capture.calls == 1, "faultless diagnostics are delivered");
    assert(!capture.message.has_fault, "absent public fault remains absent");
    fault no_fault = {};
    assert(capture.message.public_fault == no_fault, "absent public fault remains zero");
}
