module gpu;

import gpu::internal @public;
import gpu::internal::vk @public;
import gpu_test;

fn void public_limits_match_backend_storage() @test {
    assert(gpu::MAX_SUBMIT_COMMAND_LISTS == 1024);
    assert(gpu::MAX_SUBMIT_COMPLETION_WAITS == 1024);
    assert(gpu::MAX_SPARSE_TILE_BINDS == 1024);
    assert(gpu::MAX_SPARSE_OPAQUE_BINDS == 1024);
    assert(gpu::MAX_SPARSE_BIND_WAITS == 1024);
    assert(gpu::MAX_SHADER_ENTRY_POINT_BYTES + 1 == gpu::internal::SHADER_ENTRY_POINT_CAP);
}
