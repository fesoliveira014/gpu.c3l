module gpu_utils_import_only;

import gpu::utils;

fn MappedGpuSpan identity(MappedGpuSpan value) => value;

fn int main() {
    MappedGpuSpan value;
    return (int)identity(value).bytes.len;
}
