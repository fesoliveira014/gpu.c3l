module profile_boundary_test;

import gpu;

fn void accept(gpu::TextureIndex value) => (void)value;

fn int main() {
    gpu::compat::TextureIndex value = {};
    accept(value);
    return 0;
}
