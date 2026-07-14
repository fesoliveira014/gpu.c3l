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
    return { .supported = true };
}
