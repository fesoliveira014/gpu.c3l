module profile_boundary_test;

import gpu;

fn void accept(gpu::Device value) => (void)value;

fn int main() {
    gpu::compat::Device value = {};
    accept(value);
    return 0;
}
