module profile_boundary_test;

import gpu;

fn void accept(gpu::BufferHandle value) => (void)value;

fn int main() {
    gpu::compat::BufferHandle value = {};
    accept(value);
    return 0;
}
