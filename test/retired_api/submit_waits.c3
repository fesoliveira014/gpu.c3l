module retired_submit_waits;

import gpu;

fn void use_retired_submit_waits() {
    gpu::SubmitDesc submit = { .waits = {} };
    (void)submit;
}