module gpu::internal @private;

import gpu;

const uint MAX_REQUESTED_QUEUES_PER_ROLE @private = 255;

const ulong DEVICE_REQUEST_HEADER @private = 0x4750_5501_0000_0000ull;
const ulong DEVICE_REQUEST_STRICT_BIT @private = 1ull;
const ulong DEVICE_REQUEST_PRESENTATION_BIT @private = 2ull;
const uint DEVICE_REQUEST_GRAPHICS_QUEUE_SHIFT @private = 2;
const uint DEVICE_REQUEST_COMPUTE_QUEUE_SHIFT @private = 10;
const uint DEVICE_REQUEST_TRANSFER_QUEUE_SHIFT @private = 18;
const uint DEVICE_REQUEST_DISTINCT_QUEUE_SHIFT @private = 26;
const ulong DEVICE_REQUEST_QUEUE_COUNT_MASK @private = 0xFFull;
const ulong DEVICE_REQUEST_QUEUE_ROLE_MASK @private = 0x7ull;
const ulong DEVICE_REQUEST_QUEUE_VALUE_BITS @private =
    (DEVICE_REQUEST_QUEUE_COUNT_MASK << DEVICE_REQUEST_GRAPHICS_QUEUE_SHIFT)
    | (DEVICE_REQUEST_QUEUE_COUNT_MASK << DEVICE_REQUEST_COMPUTE_QUEUE_SHIFT)
    | (DEVICE_REQUEST_QUEUE_COUNT_MASK << DEVICE_REQUEST_TRANSFER_QUEUE_SHIFT)
    | (DEVICE_REQUEST_QUEUE_ROLE_MASK << DEVICE_REQUEST_DISTINCT_QUEUE_SHIFT);
const ulong DEVICE_REQUEST_QUEUES_BIT @private = 1ull << 29;
const ulong DEVICE_REQUEST_RESOURCE_AGNOSTIC_TEXTURE_SYNC_BIT @private = 1ull << 30;
const ulong DEVICE_REQUEST_KNOWN_BITS @private = DEVICE_REQUEST_HEADER
    | DEVICE_REQUEST_STRICT_BIT
    | DEVICE_REQUEST_PRESENTATION_BIT
    | DEVICE_REQUEST_QUEUE_VALUE_BITS
    | DEVICE_REQUEST_QUEUES_BIT
    | DEVICE_REQUEST_RESOURCE_AGNOSTIC_TEXTURE_SYNC_BIT;
const uint DEVICE_REQUEST_SURFACE_SHIFT @private = 64;

enum DeviceCapability @private {
    STRICT,
    PRESENTATION,
    RESOURCE_AGNOSTIC_TEXTURE_SYNC,
}

fn bool queue_requirements_are_valid(QueueRequirements requirements) @private {
    if (requirements.counts.graphics > MAX_REQUESTED_QUEUES_PER_ROLE
        || requirements.counts.compute > MAX_REQUESTED_QUEUES_PER_ROLE
        || requirements.counts.transfer > MAX_REQUESTED_QUEUES_PER_ROLE) {
        return false;
    }
    if (requirements.counts.graphics == 0
        && requirements.counts.compute == 0
        && requirements.counts.transfer == 0) {
        return false;
    }
    if (((uint)requirements.distinct & ~0x7u) != 0) return false;
    if (requirements.distinct.graphics && requirements.counts.graphics == 0) return false;
    if (requirements.distinct.compute && requirements.counts.compute == 0) return false;
    if (requirements.distinct.transfer && requirements.counts.transfer == 0) return false;
    return true;
}

fn QueueRequirements default_queue_requirements() @private {
    return {
        .counts = {
            .graphics = 1,
            .compute  = 1,
            .transfer = 1,
        },
    };
}

fn ulong encode_queue_requirements(QueueRequirements requirements) @private {
    return ((ulong)requirements.counts.graphics << DEVICE_REQUEST_GRAPHICS_QUEUE_SHIFT)
        | ((ulong)requirements.counts.compute << DEVICE_REQUEST_COMPUTE_QUEUE_SHIFT)
        | ((ulong)requirements.counts.transfer << DEVICE_REQUEST_TRANSFER_QUEUE_SHIFT)
        | ((ulong)(uint)requirements.distinct << DEVICE_REQUEST_DISTINCT_QUEUE_SHIFT);
}

fn QueueRequirements decode_queue_requirements(ulong flags) @private {
    return {
        .counts = {
            .graphics = (uint)((flags >> DEVICE_REQUEST_GRAPHICS_QUEUE_SHIFT)
                & DEVICE_REQUEST_QUEUE_COUNT_MASK),
            .compute = (uint)((flags >> DEVICE_REQUEST_COMPUTE_QUEUE_SHIFT)
                & DEVICE_REQUEST_QUEUE_COUNT_MASK),
            .transfer = (uint)((flags >> DEVICE_REQUEST_TRANSFER_QUEUE_SHIFT)
                & DEVICE_REQUEST_QUEUE_COUNT_MASK),
        },
        .distinct = (QueueRoles)(uint)((flags >> DEVICE_REQUEST_DISTINCT_QUEUE_SHIFT)
            & DEVICE_REQUEST_QUEUE_ROLE_MASK),
    };
}

fn bool queue_payload_is_well_formed(ulong flags) @private {
    if ((flags & DEVICE_REQUEST_QUEUES_BIT) == 0) {
        return (flags & DEVICE_REQUEST_QUEUE_VALUE_BITS) == 0;
    }
    return queue_requirements_are_valid(decode_queue_requirements(flags));
}

fn bool device_request_is_well_formed(DeviceRequest request) @private {
    uint128 raw = (uint128)request;
    ulong flags = (ulong)raw;
    QueueRequirements requirements = (flags & DEVICE_REQUEST_QUEUES_BIT) == 0
        ? default_queue_requirements()
        : decode_queue_requirements(flags);
    bool has_surface = (ulong)(raw >> DEVICE_REQUEST_SURFACE_SHIFT) != 0;
    bool requests_presentation = (flags & DEVICE_REQUEST_PRESENTATION_BIT) != 0;
    return (flags & DEVICE_REQUEST_HEADER) == DEVICE_REQUEST_HEADER
        && (flags & ~DEVICE_REQUEST_KNOWN_BITS) == 0
        && has_surface == requests_presentation
        && (!requests_presentation || requirements.counts.graphics > 0)
        && queue_payload_is_well_formed(flags);
}

fn DeviceRequest empty_device_request() @private {
    return (DeviceRequest)(uint128)DEVICE_REQUEST_HEADER;
}

fn Surface requested_surface(DeviceRequest request) @private {
    return (Surface)(ulong)((uint128)request >> DEVICE_REQUEST_SURFACE_SHIFT);
}

fn QueueRequirements requested_queue_requirements(DeviceRequest request) @private {
    ulong flags = (ulong)(uint128)request;
    if ((flags & DEVICE_REQUEST_QUEUES_BIT) == 0) {
        return default_queue_requirements();
    }
    return decode_queue_requirements(flags);
}

fn bool requests_strict(DeviceRequest request) @private {
    return ((ulong)(uint128)request & DEVICE_REQUEST_STRICT_BIT) != 0;
}

fn bool requests_resource_agnostic_texture_sync(DeviceRequest request) @private {
    return ((ulong)(uint128)request
        & DEVICE_REQUEST_RESOURCE_AGNOSTIC_TEXTURE_SYNC_BIT) != 0;
}

fn DeviceRequest? compose_device_request(
    DeviceRequest request,
    DeviceCapability capability,
    Surface* surface = null,
) @private {
    if (!device_request_is_well_formed(request)) return gpu::INVALID_ARGUMENT~;

    ulong contribution;
    ulong surface_bits;
    switch (capability) {
        case DeviceCapability.STRICT:
            if (surface != null) return gpu::INVALID_ARGUMENT~;
            contribution = DEVICE_REQUEST_STRICT_BIT;
        case DeviceCapability.PRESENTATION:
            if (surface == null) return gpu::INVALID_ARGUMENT~;
            if (resolve_surface(surface) == null) return gpu::INVALID_HANDLE~;
            if (requested_queue_requirements(request).counts.graphics == 0) {
                return gpu::INVALID_ARGUMENT~;
            }
            contribution = DEVICE_REQUEST_PRESENTATION_BIT;
            surface_bits = (ulong)*surface;
        case DeviceCapability.RESOURCE_AGNOSTIC_TEXTURE_SYNC:
            if (surface != null) return gpu::INVALID_ARGUMENT~;
            contribution = DEVICE_REQUEST_RESOURCE_AGNOSTIC_TEXTURE_SYNC_BIT;
    }
    if (((ulong)(uint128)request & contribution) != 0) {
        return gpu::INVALID_ARGUMENT~;
    }

    uint128 result = (uint128)request | (uint128)contribution;
    if (surface_bits != 0) {
        result |= (uint128)surface_bits << DEVICE_REQUEST_SURFACE_SHIFT;
    }
    return (DeviceRequest)result;
}

fn void? validate_device_request(DeviceRequest* request) @private {
    if (request == null || !device_request_is_well_formed(*request)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (!requests_strict(*request)) return gpu::INVALID_ARGUMENT~;

    Surface surface = requested_surface(*request);
    if (surface.is_valid() && resolve_surface(&surface) == null) {
        return gpu::INVALID_HANDLE~;
    }
}
