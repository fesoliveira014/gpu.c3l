module gpu;

<* Result of a read-only semantic support query. unmet_requirement is a
   borrowed static label when supported is false. *>
struct DeviceRequestSupport {
    bool   supported;
    String unmet_requirement;
}

<* Query semantic request support without enabling device state.
   unmet_requirement is a borrowed static label. *>
fn DeviceRequestSupport? supports_device_request(
    Adapter* adapter,
    DeviceRequest* request,
) {
    validate_device_request(request)!;
    RuntimeData* data = resolve_adapter(adapter);
    if (data == null) return INVALID_HANDLE~;

    AdapterInfo info = data.vtable.get_adapter_info(
        data.backend_state,
        adapter.adapter_index,
    )!;
    if (!info.strict_supported) {
        return {
            .supported         = false,
            .unmet_requirement = "strict semantics",
        };
    }

    QueueRequirements queues = requested_queue_requirements(*request);
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

    Surface surface = requested_surface(*request);
    void* backend_surface = null;
    if (surface.is_valid() && !supports_presentation(adapter, &surface)!) {
        return {
            .supported         = false,
            .unmet_requirement = "presentation",
        };
    }
    if (surface.is_valid()) {
        backend_surface = surface_backend_handle(&surface)!;
    }

    if (data.vtable.supports_queue_requirements == null
        || !data.vtable.supports_queue_requirements(
            data.backend_state,
            adapter.adapter_index,
            &queues,
            backend_surface,
        )!) {
        return {
            .supported         = false,
            .unmet_requirement = "queue topology",
        };
    }

    return { .supported = true };
}
