module gpu::internal @private;

import gpu;
import std::math;

fn bool sampler_filter_is_valid(Filter filter) @local {
    return filter == Filter.NEAREST || filter == Filter.LINEAR;
}

fn bool sampler_address_mode_is_valid(AddressMode mode) @local {
    switch (mode) {
        case AddressMode.REPEAT:
        case AddressMode.MIRRORED_REPEAT:
        case AddressMode.CLAMP_TO_EDGE:
        case AddressMode.CLAMP_TO_BORDER:
            return true;
        default:
            return false;
    }
}

fn bool sampler_compare_op_is_valid(CompareOp op) @local {
    switch (op) {
        case CompareOp.NEVER:
        case CompareOp.LESS:
        case CompareOp.EQUAL:
        case CompareOp.LESS_EQUAL:
        case CompareOp.GREATER:
        case CompareOp.NOT_EQUAL:
        case CompareOp.GREATER_EQUAL:
        case CompareOp.ALWAYS:
            return true;
        default:
            return false;
    }
}

fn bool sampler_desc_valid(SamplerDesc* desc) @private {
    if (!sampler_filter_is_valid(desc.min_filter)
        || !sampler_filter_is_valid(desc.mag_filter)
        || !sampler_filter_is_valid(desc.mip_filter)) {
        return false;
    }
    if (!sampler_address_mode_is_valid(desc.address_u)
        || !sampler_address_mode_is_valid(desc.address_v)
        || !sampler_address_mode_is_valid(desc.address_w)) {
        return false;
    }
    if (desc.compare_enable && !sampler_compare_op_is_valid(desc.compare)) return false;
    if (!math::is_finite(desc.mip_lod_bias)) return false;
    if (!math::is_finite(desc.min_lod) || !math::is_finite(desc.max_lod)) return false;
    if (desc.min_lod > desc.max_lod) return false;
    if (!desc.anisotropy_enable) return true;
    return math::is_finite(desc.max_anisotropy) && desc.max_anisotropy >= 1.0f;
}

fn fault sampler_contract_failure(
    DeviceOperation* operation,
    fault public_fault,
    ZString rejected_field,
    ZString invariant,
) @private {
    return report_device_contract_failure(
        operation:      operation,
        operation_name: "intern_sampler",
        public_fault:   public_fault,
        resource_kind:  DebugResourceKind.SAMPLER,
        rejected_field: rejected_field,
        invariant:      invariant,
    );
}

<* The rules that read only the description are `sampler_desc_valid`, required
   by `intern_sampler`. The limits below read the device's caps, so they are
   asserted here and a violation panics in safe builds. *>
fn void? validate_sampler_desc(
    DeviceOperation* operation,
    SamplerDesc* desc,
) @private {
    assert(
        desc.mip_lod_bias >= -operation.data.caps.max_sampler_lod_bias
            && desc.mip_lod_bias <= operation.data.caps.max_sampler_lod_bias,
        "desc.mip_lod_bias magnitude must not exceed DeviceCaps.max_sampler_lod_bias"
    );
    if (!desc.anisotropy_enable) return;
    if (!math::is_finite(operation.data.caps.max_sampler_anisotropy)
        || operation.data.caps.max_sampler_anisotropy <= 0.0f) {
        return sampler_contract_failure(
            operation:      operation,
            public_fault:   gpu::UNSUPPORTED_FEATURE,
            rejected_field: "SamplerDesc.anisotropy_enable",
            invariant:      "device must support sampler anisotropy",
        )~;
    }
    assert(
        desc.max_anisotropy <= operation.data.caps.max_sampler_anisotropy,
        "desc.max_anisotropy must not exceed DeviceCaps.max_sampler_anisotropy"
    );
}
