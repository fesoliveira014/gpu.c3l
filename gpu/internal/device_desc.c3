module gpu::internal @private;

import gpu;
import gpu::internal::vk @public;

fn bool queue_request_is_zero(QueueRequest request) @private {
    return (uint)request.required == 0
        && (uint)request.distinct == 0;
}

fn bool queue_request_is_valid(QueueRequest request) @private {
    uint required = (uint)request.required;
    uint distinct = (uint)request.distinct;
    return required != 0
        && (required & ~gpu::internal::QUEUE_ROLE_MASK) == 0
        && (distinct & ~gpu::internal::QUEUE_ROLE_MASK) == 0
        && (distinct & ~required) == 0;
}

fn QueueRequest default_queue_request() @private {
    return {
        .required = {
            .graphics,
            .compute,
            .transfer,
        },
    };
}

fn DeviceDesc? normalize_device_desc(DeviceDesc* desc) @private {
    DeviceDesc normalized = desc == null ? {} : *desc;
    if (queue_request_is_zero(normalized.queues)) {
        normalized.queues = default_queue_request();
    } else if (!queue_request_is_valid(normalized.queues)) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (normalized.surface.is_valid()
        && !normalized.queues.required.graphics) {
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
    if (normalized.enable_ray_queries
        && gpu::internal::vk::vk_acceleration_structure_heap_capacity(
            data.state,
        ) == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (normalized.enable_ray_queries
        && !gpu::internal::vk::vk_adapter_supports_ray_queries(
            data.state,
            adapter.adapter_index,
        )!) {
        return {
            .supported         = false,
            .unmet_requirement = "ray queries",
        };
    }
    if (normalized.enable_ray_queries
        && !gpu::internal::vk::vk_adapter_supports_ray_query_heap(
            data.state,
            adapter.adapter_index,
            gpu::internal::vk::vk_acceleration_structure_heap_capacity(
                data.state,
            ),
        )!) {
        return {
            .supported         = false,
            .unmet_requirement = "acceleration structure heap capacity",
        };
    }

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

    if (normalized.enable_sparse_textures
        && !gpu::internal::vk::vk_adapter_supports_sparse_textures(
            data.state,
            adapter.adapter_index,
        )!) {
        return {
            .supported         = false,
            .unmet_requirement = "sparse textures",
        };
    }

    QueueRequest queues = normalized.queues;
    if (!gpu::internal::vk::vk_supports_queue_request(
        state:                  data.state,
        adapter_index:          adapter.adapter_index,
        request:                &queues,
        backend_surface:        backend_surface,
        require_sparse_binding: normalized.enable_sparse_textures,
    )!) {
        return {
            .supported         = false,
            .unmet_requirement = normalized.enable_sparse_textures
                ? "sparse queue topology"
                : "queue topology",
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
