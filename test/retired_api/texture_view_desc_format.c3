module retired_view_desc_format;

import gpu;

fn gpu::TextureViewDesc use_retired() => { .format = gpu::Format.RGBA8_UNORM };
