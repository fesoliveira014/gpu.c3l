module retired_frames_in_flight;

import gpu;

fn gpu::DeviceDesc use_retired_frames_in_flight() => { .frames_in_flight = 1 };