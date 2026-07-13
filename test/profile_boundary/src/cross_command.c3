module profile_boundary_test;

import gpu;

fn void accept(gpu::CommandList value) => (void)value;

fn int main() {
    gpu::compat::CommandList value = {};
    accept(value);
    return 0;
}
