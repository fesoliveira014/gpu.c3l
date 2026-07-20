module retired_create_device_from_desc;

import gpu;

fn void? use_retired_create_device_from_desc() {
    (void)gpu::create_device_from_desc(null)!;
}
