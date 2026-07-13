module profile_boundary_test;

import gpu;

fn void accept(gpu::PipelineHandle value) => (void)value;

fn int main() {
    gpu::compat::PipelineHandle value = {};
    accept(value);
    return 0;
}
