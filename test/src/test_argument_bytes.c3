module gpu_test;

import gpu::internal @public;
import gpu @public;

fn void argument_bytes_accept_zero_and_products() @test {
    assert(gpu::internal::indirect_argument_bytes(16, 0)!! == 0);
    assert(gpu::internal::indirect_argument_bytes(16, 4)!! == 64);
    assert(gpu::internal::indirect_argument_bytes(usz::max, 1)!! == usz::max);
}

fn void argument_bytes_reject_overflow() @test {
    assert(@catch(gpu::internal::indirect_argument_bytes(usz::max, 2)) == gpu::INVALID_ARGUMENT);
}
