module profile_boundary_test;

import gpu;

fn void accept(gpu::GlobalBarrier value) => (void)value;

fn int main() {
    gpu::compat::GlobalBarrier value = {};
    accept(value);
    return 0;
}
