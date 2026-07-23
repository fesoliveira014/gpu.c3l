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

fn ZString sampler_desc_invalid_enum_field(SamplerDesc* desc) @local {
    if (!sampler_filter_is_valid(desc.min_filter)) return "SamplerDesc.min_filter";
    if (!sampler_filter_is_valid(desc.mag_filter)) return "SamplerDesc.mag_filter";
    if (!sampler_filter_is_valid(desc.mip_filter)) return "SamplerDesc.mip_filter";
    if (!sampler_address_mode_is_valid(desc.address_u)) return "SamplerDesc.address_u";
    if (!sampler_address_mode_is_valid(desc.address_v)) return "SamplerDesc.address_v";
    if (!sampler_address_mode_is_valid(desc.address_w)) return "SamplerDesc.address_w";
    if (desc.compare_enable && !sampler_compare_op_is_valid(desc.compare)) {
        return "SamplerDesc.compare";
    }
    return null;
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

fn void? validate_sampler_desc(
    DeviceOperation* operation,
    SamplerDesc* desc,
) @private {
    ZString invalid_enum_field = sampler_desc_invalid_enum_field(desc);
    if (invalid_enum_field != null) {
        return sampler_contract_failure(
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            rejected_field: invalid_enum_field,
            invariant:      "must name a defined enum value",
        )~;
    }
    if (!math::is_finite(desc.mip_lod_bias)) {
        return sampler_contract_failure(
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            rejected_field: "SamplerDesc.mip_lod_bias",
            invariant:      "must be finite",
        )~;
    }
    if (desc.mip_lod_bias < -operation.data.caps.max_sampler_lod_bias
        || desc.mip_lod_bias > operation.data.caps.max_sampler_lod_bias) {
        return sampler_contract_failure(
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            rejected_field: "SamplerDesc.mip_lod_bias",
            invariant:      "absolute value must not exceed DeviceCaps.max_sampler_lod_bias",
        )~;
    }
    if (!math::is_finite(desc.min_lod)) {
        return sampler_contract_failure(
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            rejected_field: "SamplerDesc.min_lod",
            invariant:      "must be finite",
        )~;
    }
    if (!math::is_finite(desc.max_lod)) {
        return sampler_contract_failure(
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            rejected_field: "SamplerDesc.max_lod",
            invariant:      "must be finite",
        )~;
    }
    if (desc.min_lod > desc.max_lod) {
        return sampler_contract_failure(
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            rejected_field: "SamplerDesc.min_lod",
            invariant:      "must not exceed SamplerDesc.max_lod",
        )~;
    }
    if (!desc.anisotropy_enable) return;
    if (!math::is_finite(desc.max_anisotropy)) {
        return sampler_contract_failure(
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            rejected_field: "SamplerDesc.max_anisotropy",
            invariant:      "must be finite when anisotropy is enabled",
        )~;
    }
    if (desc.max_anisotropy < 1.0f) {
        return sampler_contract_failure(
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            rejected_field: "SamplerDesc.max_anisotropy",
            invariant:      "must be at least 1.0 when anisotropy is enabled",
        )~;
    }
    if (!math::is_finite(operation.data.caps.max_sampler_anisotropy)
        || operation.data.caps.max_sampler_anisotropy <= 0.0f) {
        return sampler_contract_failure(
            operation:      operation,
            public_fault:   gpu::UNSUPPORTED_FEATURE,
            rejected_field: "SamplerDesc.anisotropy_enable",
            invariant:      "device must support sampler anisotropy",
        )~;
    }
    if (desc.max_anisotropy > operation.data.caps.max_sampler_anisotropy) {
        return sampler_contract_failure(
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            rejected_field: "SamplerDesc.max_anisotropy",
            invariant:      "must not exceed DeviceCaps.max_sampler_anisotropy",
        )~;
    }
}
