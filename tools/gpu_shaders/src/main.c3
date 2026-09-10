module gpu_shaders::entry;

import gpu_shaders;

fn int main(String[] args) => gpu_shaders::run(args[1..], args[0]);
