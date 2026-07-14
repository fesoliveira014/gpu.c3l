module gpu_test;

import gpu @public;

fn void strict_device_request_is_an_immutable_value() @test {
    gpu::DeviceRequest empty = gpu::empty_device_request();
    gpu::DeviceRequest original = empty;
    gpu::DeviceRequest strict = gpu::compose_device_request(
        empty,
        gpu::DeviceCapability.STRICT,
    )!!;
    assert(gpu::DeviceRequest::size == uint128::size, "request storage fits in two machine words");
    assert((uint128)empty == (uint128)original, "composition preserves its input value");
    assert((uint128)strict != (uint128)empty, "composition returns a new request value");
}

fn void strict_device_request_is_explicit_and_non_empty() @test {
    gpu::DeviceRequest request = gpu::strict_device_request();
    assert(
        (uint128)request != (uint128)gpu::empty_device_request(),
        "strict construction explicitly contributes strict semantics"
    );
    gpu::validate_device_request(&request)!!;
}

fn void strict_request_state_is_explicit() @test {
    gpu::DeviceRequest empty = gpu::empty_device_request();
    gpu::DeviceRequest strict = gpu::strict_device_request();

    assert(!gpu::requests_strict(empty), "an empty request does not enable strict state");
    assert(gpu::requests_strict(strict), "the strict request enables strict state");
}

fn void request_composition_rejects_duplicate_and_malformed_inputs() @test {
    gpu::DeviceRequest strict = gpu::strict_device_request();
    assert(
        @catch(gpu::compose_device_request(
            strict,
            gpu::DeviceCapability.STRICT,
        )) == gpu::INVALID_ARGUMENT,
        "duplicate strict composition is rejected"
    );

    gpu::DeviceRequest malformed;
    assert(
        @catch(gpu::compose_device_request(
            malformed,
            gpu::DeviceCapability.STRICT,
        )) == gpu::INVALID_ARGUMENT,
        "malformed request composition is rejected"
    );
}
