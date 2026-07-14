module gpu;

<* Immutable semantic device requirements with private capability storage. *>
typedef DeviceRequest = inline ulong;

const ulong DEVICE_REQUEST_HEADER @private = 0x4750_5501_0000_0000ull;
const ulong DEVICE_REQUEST_STRICT_BIT @private = 1ull;
const ulong DEVICE_REQUEST_KNOWN_BITS @private = DEVICE_REQUEST_HEADER
    | DEVICE_REQUEST_STRICT_BIT;

enum DeviceCapability @private {
    STRICT,
}

fn bool device_request_is_well_formed(DeviceRequest request) @private {
    ulong bits = (ulong)request;
    return (bits & DEVICE_REQUEST_HEADER) == DEVICE_REQUEST_HEADER
        && (bits & ~DEVICE_REQUEST_KNOWN_BITS) == 0;
}

fn DeviceRequest empty_device_request() @private {
    return (DeviceRequest)DEVICE_REQUEST_HEADER;
}

fn DeviceRequest? compose_device_request(
    DeviceRequest request,
    DeviceCapability capability,
) @private {
    if (!device_request_is_well_formed(request)) return INVALID_ARGUMENT~;

    ulong contribution;
    switch (capability) {
        case DeviceCapability.STRICT:
            contribution = DEVICE_REQUEST_STRICT_BIT;
    }
    if (((ulong)request & contribution) != 0) return INVALID_ARGUMENT~;
    return (DeviceRequest)((ulong)request | contribution);
}

fn void? validate_device_request(DeviceRequest* request) @private {
    if (request == null || !device_request_is_well_formed(*request)) {
        return INVALID_ARGUMENT~;
    }
    if (((ulong)*request & DEVICE_REQUEST_STRICT_BIT) == 0) {
        return INVALID_ARGUMENT~;
    }
}

<* Create a request that explicitly requires strict GPU semantics. *>
fn DeviceRequest strict_device_request() {
    return compose_device_request(
        empty_device_request(),
        DeviceCapability.STRICT,
    )!!;
}
