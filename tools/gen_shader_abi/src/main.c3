module gen_shader_abi::entry;

import gen_shader_abi;

fn int main(String[] args) => gen_shader_abi::run(args[1..]);
