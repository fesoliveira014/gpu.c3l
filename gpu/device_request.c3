module gpu;

<* Immutable semantic requirements for device creation. *>
typedef DeviceRequest = uint128;

const ulong DEVICE_REQUEST_HEADER @private = 0x4750_5501_0000_0000ull;
const ulong DEVICE_REQUEST_STRICT_BIT @private = 1ull;
const ulong DEVICE_REQUEST_PRESENTATION_BIT @private = 2ull;
const ulong DEVICE_REQUEST_KNOWN_BITS @private = DEVICE_REQUEST_HEADER
    | DEVICE_REQUEST_STRICT_BIT
    | DEVICE_REQUEST_PRESENTATION_BIT;
const uint DEVICE_REQUEST_SURFACE_SHIFT @private = 64;

enum DeviceCapability @private {
    STRICT,
    PRESENTATION,
}

fn bool device_request_is_well_formed(DeviceRequest request) @private {
    uint128 raw = (uint128)request;
    ulong flags = (ulong)raw;
    bool has_surface = (ulong)(raw >> DEVICE_REQUEST_SURFACE_SHIFT) != 0;
    bool requests_presentation = (flags & DEVICE_REQUEST_PRESENTATION_BIT) != 0;
    return (flags & DEVICE_REQUEST_HEADER) == DEVICE_REQUEST_HEADER
        && (flags & ~DEVICE_REQUEST_KNOWN_BITS) == 0
        && has_surface == requests_presentation;
}

fn DeviceRequest empty_device_request() @private {
    return (DeviceRequest)(uint128)DEVICE_REQUEST_HEADER;
}

fn Surface requested_surface(DeviceRequest request) @private {
    return (Surface)(ulong)((uint128)request >> DEVICE_REQUEST_SURFACE_SHIFT);
}

fn DeviceRequest? compose_device_request(
    DeviceRequest request,
    DeviceCapability capability,
    Surface* surface = null,
) @private {
    if (!device_request_is_well_formed(request)) return INVALID_ARGUMENT~;

    ulong contribution;
    ulong surface_bits;
    switch (capability) {
        case DeviceCapability.STRICT:
            if (surface != null) return INVALID_ARGUMENT~;
            contribution = DEVICE_REQUEST_STRICT_BIT;
        case DeviceCapability.PRESENTATION:
            if (surface == null) return INVALID_ARGUMENT~;
            if (resolve_surface(surface) == null) return INVALID_HANDLE~;
            contribution = DEVICE_REQUEST_PRESENTATION_BIT;
            surface_bits = (ulong)*surface;
    }
    if (((ulong)(uint128)request & contribution) != 0) {
        return INVALID_ARGUMENT~;
    }

    uint128 result = (uint128)request | (uint128)contribution;
    if (surface_bits != 0) {
        result |= (uint128)surface_bits << DEVICE_REQUEST_SURFACE_SHIFT;
    }
    return (DeviceRequest)result;
}

fn void? validate_device_request(DeviceRequest* request) @private {
    if (request == null || !device_request_is_well_formed(*request)) {
        return INVALID_ARGUMENT~;
    }
    if (((ulong)(uint128)*request & DEVICE_REQUEST_STRICT_BIT) == 0) {
        return INVALID_ARGUMENT~;
    }

    Surface surface = requested_surface(*request);
    if (surface.is_valid() && resolve_surface(&surface) == null) {
        return INVALID_HANDLE~;
    }
}

<* Create a strict semantic device request. *>
fn DeviceRequest strict_device_request() {
    return compose_device_request(
        empty_device_request(),
        DeviceCapability.STRICT,
    )!!;
}

<* Return a new request that requires presentation to the given surface. *>
fn DeviceRequest? request_presentation(
    DeviceRequest request,
    Surface* surface,
) {
    return compose_device_request(
        request,
        DeviceCapability.PRESENTATION,
        surface,
    );
}
