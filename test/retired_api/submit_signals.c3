module retired_submit_signals;

import gpu;

fn void use_retired_submit_signals() {
    gpu::SubmitDesc submit = { .signals = {} };
    (void)submit;
}