module retired_strict_supported_info;

import gpu;

fn bool use_retired_strict_supported_adapter_info(gpu::AdapterInfo info) {
    return info.strict_supported;
}
