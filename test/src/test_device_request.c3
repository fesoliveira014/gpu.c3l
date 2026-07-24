module gpu_test;

import gpu::internal @public;
import gpu @public;

fn void strict_device_request_is_an_immutable_value() @test {
    gpu::DeviceRequest empty = gpu::internal::empty_device_request();
    gpu::DeviceRequest original = empty;
    gpu::DeviceRequest strict = gpu::internal::compose_device_request(
        empty,
        gpu::internal::DeviceCapability.STRICT,
    )!!;
    assert(gpu::DeviceRequest::size == uint128::size, "request storage fits in two machine words");
    assert((uint128)empty == (uint128)original, "composition preserves its input value");
    assert((uint128)strict != (uint128)empty, "composition returns a new request value");
}

fn void strict_device_request_is_explicit_and_non_empty() @test {
    gpu::DeviceRequest request = gpu::strict_device_request();
    assert(
        (uint128)request != (uint128)gpu::internal::empty_device_request(),
        "strict construction explicitly contributes strict semantics"
    );
    gpu::internal::validate_device_request(&request)!!;
}

fn void strict_request_state_is_explicit() @test {
    gpu::DeviceRequest empty = gpu::internal::empty_device_request();
    gpu::DeviceRequest strict = gpu::strict_device_request();

    assert(!gpu::internal::requests_strict(empty), "an empty request does not enable strict state");
    assert(gpu::internal::requests_strict(strict), "the strict request enables strict state");
}

fn void resource_agnostic_texture_sync_request_is_explicit_and_immutable() @test {
    gpu::DeviceRequest strict = gpu::strict_device_request();
    gpu::DeviceRequest original = strict;
    gpu::DeviceRequest requested =
        gpu::request_resource_agnostic_texture_sync(strict)!!;

    assert((uint128)strict == (uint128)original, "composition preserves its input");
    assert(
        gpu::internal::requests_resource_agnostic_texture_sync(requested),
        "the contribution records the semantic request"
    );
    assert(
        !gpu::internal::requests_resource_agnostic_texture_sync(strict),
        "the input remains unrequested"
    );
    gpu::internal::validate_device_request(&requested)!!;
}

fn void dynamic_color_state_request_is_explicit_and_immutable() @test {
    gpu::DeviceRequest strict = gpu::strict_device_request();
    gpu::DeviceRequest original = strict;
    gpu::DeviceRequest requested = gpu::request_dynamic_color_state(strict)!!;

    assert((uint128)strict == (uint128)original, "composition preserves its input");
    assert(
        gpu::internal::requests_dynamic_color_state(requested),
        "the contribution records the semantic request"
    );
    assert(
        !gpu::internal::requests_dynamic_color_state(strict),
        "the input remains unrequested"
    );
    gpu::internal::validate_device_request(&requested)!!;
}

fn void resource_agnostic_texture_sync_request_rejects_duplicate_and_malformed_inputs() @test {
    gpu::DeviceRequest requested = gpu::request_resource_agnostic_texture_sync(
        gpu::strict_device_request(),
    )!!;
    assert(
        @catch(gpu::request_resource_agnostic_texture_sync(requested))
            == gpu::INVALID_ARGUMENT,
        "duplicate composition is rejected"
    );

    gpu::DeviceRequest malformed = (gpu::DeviceRequest)(
        (uint128)gpu::strict_device_request() | ((uint128)1 << 33)
    );
    assert(
        @catch(gpu::request_resource_agnostic_texture_sync(malformed))
            == gpu::INVALID_ARGUMENT,
        "unknown request bits are rejected"
    );
}

fn void dynamic_color_state_request_rejects_duplicate_and_malformed_inputs() @test {
    gpu::DeviceRequest requested =
        gpu::request_dynamic_color_state(gpu::strict_device_request())!!;
    assert(
        @catch(gpu::request_dynamic_color_state(requested))
            == gpu::INVALID_ARGUMENT,
        "duplicate composition is rejected"
    );

    gpu::DeviceRequest malformed = (gpu::DeviceRequest)(
        (uint128)gpu::strict_device_request() | ((uint128)1 << 33)
    );
    assert(
        @catch(gpu::request_dynamic_color_state(malformed))
            == gpu::INVALID_ARGUMENT,
        "unknown request bits are rejected"
    );
}

fn void request_composition_rejects_duplicate_and_malformed_inputs() @test {
    gpu::DeviceRequest strict = gpu::strict_device_request();
    assert(
        @catch(gpu::internal::compose_device_request(
            strict,
            gpu::internal::DeviceCapability.STRICT,
        )) == gpu::INVALID_ARGUMENT,
        "duplicate strict composition is rejected"
    );

    gpu::DeviceRequest malformed;
    assert(
        @catch(gpu::internal::compose_device_request(
            malformed,
            gpu::internal::DeviceCapability.STRICT,
        )) == gpu::INVALID_ARGUMENT,
        "malformed request composition is rejected"
    );
}

fn void queue_requirements_are_immutable_and_round_trip() @test {
    gpu::DeviceRequest strict = gpu::strict_device_request();
    gpu::DeviceRequest original = strict;
    gpu::QueueRequirements requirements = {
        .counts   = { .graphics = 1, .compute = 1 },
        .distinct = { .compute },
    };

    gpu::DeviceRequest configured = gpu::request_queues(strict, requirements)!!;
    gpu::QueueRequirements decoded = gpu::internal::requested_queue_requirements(configured);

    assert((uint128)strict == (uint128)original, "queue composition preserves its input");
    assert((uint128)configured != (uint128)strict, "queue composition returns a new value");
    assert(decoded.counts.graphics == 1, "graphics count round-trips");
    assert(decoded.counts.compute == 1, "compute count round-trips");
    assert(decoded.counts.transfer == 0, "omitted transfer count round-trips");
    assert(decoded.distinct.compute, "distinct compute requirement round-trips");
}

fn void strict_request_has_default_queue_requirements() @test {
    gpu::QueueRequirements requirements =
        gpu::internal::requested_queue_requirements(gpu::strict_device_request());

    assert(requirements.counts.graphics == 1, "strict defaults to one graphics queue");
    assert(requirements.counts.compute == 1, "strict defaults to one compute queue");
    assert(requirements.counts.transfer == 1, "strict defaults to one transfer queue");
    assert((uint)requirements.distinct == 0, "strict defaults permit queue aliases");
}

fn void queue_requirement_composition_rejects_invalid_groups() @test {
    gpu::DeviceRequest strict = gpu::strict_device_request();
    gpu::QueueRequirements none;
    gpu::QueueRequirements too_many = { .counts = { .graphics = 256 } };
    gpu::QueueRequirements missing_role = {
        .counts   = { .graphics = 1 },
        .distinct = { .compute },
    };
    gpu::QueueRequirements valid = { .counts = { .compute = 1 } };
    gpu::DeviceRequest configured = gpu::request_queues(strict, valid)!!;

    assert(@catch(gpu::request_queues(strict, none)) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::request_queues(strict, too_many)) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::request_queues(strict, missing_role)) == gpu::INVALID_ARGUMENT);
    assert(@catch(gpu::request_queues(configured, valid)) == gpu::INVALID_ARGUMENT);
}
