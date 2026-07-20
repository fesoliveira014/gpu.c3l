from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import check_performance_contract


REQUIRED_PATHS = (
    "gpu/command.c3",
    "gpu/device.c3",
    "gpu/vk/command.c3",
    "gpu/vk/command_state.c3",
    "gpu/vk/lifetime.c3",
    "gpu/vk/queue.c3",
    "gpu/vk/sync.c3",
    "gpu/vk/texture.c3",
    "test/src/command_record_bench.c3",
    "test/src/lifecycle_bench.c3",
)


class PerformanceContractTests(unittest.TestCase):
    def copied_tree(self, destination: Path) -> None:
        for relative in REQUIRED_PATHS:
            source = check_performance_contract.ROOT / relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def mutate(
        self,
        root: Path,
        relative: str,
        old: str,
        new: str,
    ) -> None:
        path = root / relative
        source = path.read_text(encoding="utf-8")
        self.assertIn(old, source)
        path.write_text(source.replace(old, new, 1), encoding="utf-8")

    def test_current_sources_satisfy_contract(self):
        self.assertEqual(check_performance_contract.check(), [])

    def test_registry_lock_on_public_recording_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/command.c3",
                "fn void? cmd_barrier(CommandList* commands, Barrier* barrier) {",
                (
                    "fn void? cmd_barrier(CommandList* commands, Barrier* barrier) {\n"
                    "    lock_device_registry();"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("lock_device_registry(" in error for error in errors))

    def test_completion_point_allocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/queue.c3",
                "return gpu::next_completion_point(queue, &completion.next_sequence);",
                (
                    "mem::new(gpu::CompletionPoint);\n"
                    "    return gpu::next_completion_point("
                    "queue, &completion.next_sequence);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("reserve_queue_completion_locked" in error for error in errors))

    def test_hidden_destruction_wait_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/texture.c3",
                "fn void? vk_destroy_texture(gpu::Device* device, gpu::TextureHandle handle) @private {",
                (
                    "fn void? vk_destroy_texture("
                    "gpu::Device* device, gpu::TextureHandle handle) @private {\n"
                    "    wait_completion(device);"
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("wait_completion(" in error for error in errors))

    def test_missing_completion_wait_instrumentation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/sync.c3",
                "state.submit_stats.completion_wait_calls++;",
                "state.submit_stats.queue_submits++;",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "gpu/vk/sync.c3 is missing performance evidence token"
                    in error
                    for error in errors
                )
            )

    def test_missing_deferred_release_instrumentation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/lifetime.c3",
                "fn CommandBufferStats command_buffer_stats(",
                "fn CommandBufferStats missing_stats(",
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(
                any(
                    "gpu/vk/lifetime.c3 is missing performance evidence token"
                    in error
                    for error in errors
                )
            )

    def test_generated_fallback_before_reuse_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copied_tree(root)
            self.mutate(
                root,
                "gpu/vk/command.c3",
                (
                    "GeneratedPreprocessBuffer* existing = "
                    "find_generated_preprocess_buffer("
                ),
                (
                    "GeneratedPreprocessBuffer* existing = "
                    "missing_generated_preprocess_lookup("
                ),
            )
            errors = check_performance_contract.check(root)
            self.assertTrue(any("acquisition is missing" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
