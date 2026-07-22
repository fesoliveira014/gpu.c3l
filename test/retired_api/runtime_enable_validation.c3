module retired_runtime_validation;

import gpu;

fn gpu::RuntimeDesc use_retired_runtime_enable_validation() => {
    .backend           = gpu::BackendKind.VULKAN,
    .enable_validation = true,
};
