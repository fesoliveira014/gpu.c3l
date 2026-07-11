module gpu;

struct DiagnosticCapture {
    uint              calls;
    void*             user_data;
    DebugMessage message;
}

fn void capture_diagnostic(DebugMessage* message, void* user_data) {
    DiagnosticCapture* capture = (DiagnosticCapture*)user_data;
    capture.calls++;
    capture.user_data = user_data;
    capture.message = *message;
}

fn void diagnostic_dispatch_is_null_safe() @test {
    DebugMessage message = {};
    dispatch_debug_message(null, null, &message);
}

fn void diagnostic_dispatch_delivers_complete_message_once() @test {
    DiagnosticCapture capture = {};
    DebugMessage message = {
        .severity   = DebugMessageSeverity.ERROR,
        .category = { .public_contract, .validation },
        .operation  = "create_buffer",
        .has_fault  = true,
        .public_fault = INVALID_ARGUMENT,
        .resource = {
            .kind                = DebugResourceKind.BUFFER,
            .has_public_identity = true,
            .index               = 17,
            .generation          = 3,
            .debug_name          = "diagnostic_buffer",
        },
        .rejected_field       = "BufferDesc.size",
        .invariant            = "buffer size must be greater than zero",
        .backend_text                 = "buffer size is zero",
        .validation_id_name   = "VUID-test",
        .validation_id_number = 42,
    };

    dispatch_debug_message(&capture_diagnostic, &capture, &message);

    assert(capture.calls == 1, "dispatch invokes the callback exactly once");
    assert(capture.user_data == &capture, "dispatch preserves callback userdata");
    assert(capture.message.severity == DebugMessageSeverity.ERROR, "severity is preserved");
    assert(capture.message.category.public_contract, "public-contract category is preserved");
    assert(capture.message.category.validation, "validation category is preserved");
    assert(capture.message.operation.str_view() == "create_buffer", "operation is preserved");
    assert(capture.message.has_fault, "public fault presence is preserved");
    assert(capture.message.public_fault == INVALID_ARGUMENT, "public fault is preserved");
    assert(capture.message.resource.kind == DebugResourceKind.BUFFER, "resource kind is preserved");
    assert(capture.message.resource.has_public_identity, "public identity presence is preserved");
    assert(capture.message.resource.index == 17, "resource index is preserved");
    assert(capture.message.resource.generation == 3, "resource generation is preserved");
    assert(capture.message.resource.debug_name.str_view() == "diagnostic_buffer", "debug name is preserved");
    assert(capture.message.rejected_field.str_view() == "BufferDesc.size", "rejected field is preserved");
    assert(capture.message.invariant.str_view() == "buffer size must be greater than zero", "invariant is preserved");
    assert(capture.message.backend_text.str_view() == "buffer size is zero", "text is preserved");
    assert(capture.message.validation_id_name.str_view() == "VUID-test", "validation ID name is preserved");
    assert(capture.message.validation_id_number == 42, "validation ID number is preserved");
}

fn void diagnostic_dispatch_preserves_absent_fault() @test {
    DiagnosticCapture capture = {};
    DebugMessage message = {
        .severity  = DebugMessageSeverity.WARNING,
        .category = { .performance },
        .operation = "submit",
        .backend_text      = "backend warning",
    };

    dispatch_debug_message(&capture_diagnostic, &capture, &message);

    assert(capture.calls == 1, "faultless diagnostics are delivered");
    assert(!capture.message.has_fault, "absent public fault remains absent");
    fault no_fault = {};
    assert(capture.message.public_fault == no_fault, "absent public fault remains zero");
}

fn void invalid_backend_delivers_diagnostic_before_unchanged_fault() @test {
    DiagnosticCapture capture = {};
    BackendKind invalid_backend = bitcast(99, BackendKind);
    DeviceDesc desc = {
        .backend              = invalid_backend,
        .debug_callback  = &capture_diagnostic,
        .debug_user_data = &capture,
    };

    fault result = @catch(create_device(&desc));

    assert(result == UNSUPPORTED_BACKEND, "invalid backend keeps UNSUPPORTED_BACKEND");
    assert(capture.calls == 1, "invalid backend emits exactly one diagnostic");
    assert(capture.message.severity == DebugMessageSeverity.ERROR, "invalid backend is an error");
    assert(capture.message.category.public_contract, "invalid backend is a public-contract diagnostic");
    assert(capture.message.operation.str_view() == "create_device", "device creation operation is identified");
    assert(capture.message.has_fault, "invalid backend carries its public fault");
    assert(capture.message.public_fault == UNSUPPORTED_BACKEND, "diagnostic fault matches the returned fault");
    assert(capture.message.resource.kind == DebugResourceKind.NONE, "invalid backend has no resource");
    assert(!capture.message.resource.has_public_identity, "invalid backend has no public identity");
    assert(capture.message.rejected_field.str_view() == "DeviceDesc.backend", "rejected backend field is identified");
    assert(capture.message.invariant.str_view() == "backend must be supported", "backend invariant is identified");
}

fn void invalid_backend_fault_is_callback_independent() @test {
    BackendKind invalid_backend = bitcast(99, BackendKind);
    DeviceDesc without_callback = { .backend = invalid_backend };
    fault without_callback_fault = @catch(create_device(&without_callback));

    DiagnosticCapture capture = {};
    DeviceDesc with_callback = {
        .backend              = invalid_backend,
        .debug_callback  = &capture_diagnostic,
        .debug_user_data = &capture,
    };
    fault with_callback_fault = @catch(create_device(&with_callback));

    assert(without_callback_fault == UNSUPPORTED_BACKEND, "null callback keeps the original fault");
    assert(with_callback_fault == without_callback_fault, "callback presence does not alter fault identity");
    assert(capture.calls == 1, "configured callback receives one message");
}
