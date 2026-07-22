module retired_texture_dimension;

import gpu;

fn void use_retired() {
    gpu::TextureDimension dimension;
    (void)dimension;
}
