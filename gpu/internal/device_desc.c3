module gpu::internal @private;

import gpu;
import gpu::internal::vk @public;

const uint MAX_REQUESTED_QUEUES_PER_ROLE @private = 255;

fn bool queue_requirements_are_zero(QueueRequirements requirements) @private {
    return requirements.counts.graphics == 0
        && requirements.counts.compute == 0
        && requirements.counts.transfer == 0
        && (uint)requirements.distinct == 0;
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

fn DeviceDesc? normalize_device_desc(DeviceDesc* desc) @private {
    DeviceDesc normalized = desc == null ? {} : *desc;
    if (queue_requirements_are_zero(normalized.queues)) {
        normalized.queues = default_queue_requirements();
    } else if (!queue_requirements_are_valid(normalized.queues)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (normalized.surface.is_valid()
        && normalized.queues.counts.graphics == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    return normalized;
}

fn DeviceSupport? evaluate_device_desc_support(
    RuntimeData* data,
    Adapter* adapter,
    DeviceDesc normalized,
    bool creation_diagnostics = false,
) @private {
    void* backend_surface = null;
    if (normalized.surface.is_valid()) {
        SurfaceData* surface_data = resolve_surface(&normalized.surface);
        if (surface_data == null) return gpu::INVALID_HANDLE~;
        RuntimeData* surface_runtime = resolve_runtime(&surface_data.runtime);
        if (surface_runtime != data) return gpu::INVALID_HANDLE~;
        backend_surface = surface_data.backend_surface;
        if (!gpu::internal::vk::vk_supports_presentation(
            data.state,
            adapter.adapter_index,
            backend_surface,
        )!) {
            return {
                .supported         = false,
                .unmet_requirement = "presentation",
            };
        }
    }

    AdapterInfo info = gpu::internal::vk::vk_get_adapter_info(
        data.state,
        adapter.adapter_index,
    )!;
    QueueRequirements queues = normalized.queues;
    if (queues.counts.graphics > info.queues.graphics_count) {
        return {
            .supported         = false,
            .unmet_requirement = "graphics queues",
        };
    }
    if (queues.counts.compute > info.queues.compute_count) {
        return {
            .supported         = false,
            .unmet_requirement = "compute queues",
        };
    }
    if (queues.counts.transfer > info.queues.transfer_count) {
        return {
            .supported         = false,
            .unmet_requirement = "transfer queues",
        };
    }

    if (!gpu::internal::vk::vk_supports_queue_requirements(
        state:           data.state,
        adapter_index:   adapter.adapter_index,
        requirements:    &queues,
        backend_surface: backend_surface,
    )!) {
        return {
            .supported         = false,
            .unmet_requirement = "queue topology",
        };
    }
    if (!gpu::internal::vk::vk_adapter_supports_device_baseline(
        data.state,
        adapter.adapter_index,
    )!) {
        if (creation_diagnostics) {
            gpu::internal::vk::validate_device_baseline_support(
                data.state,
                adapter.adapter_index,
            )!;
        }
        return {
            .supported         = false,
            .unmet_requirement = "device baseline",
        };
    }

    return { .supported = true };
}
