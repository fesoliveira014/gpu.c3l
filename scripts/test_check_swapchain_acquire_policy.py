from __future__ import annotations

import unittest
from pathlib import Path

from scripts import check_swapchain_acquire_policy


SOURCE = Path("gpu/internal/vk/swapchain.c3").read_text(encoding="utf-8")


class SwapchainAcquirePolicyTests(unittest.TestCase):
    def test_accepts_current_acquisition_transaction(self) -> None:
        self.assertEqual(
            check_swapchain_acquire_policy.validate_acquire_policy(SOURCE),
            [],
        )

    def test_rejects_hidden_timeout_substitution(self) -> None:
        mutated = SOURCE.replace(
            "slot.swapchain,\n        timeout_ns,",
            "slot.swapchain,\n        ACQUIRE_TIMEOUT_NS,",
            1,
        )
        self.assertIn(
            "native acquire must receive timeout_ns unchanged",
            check_swapchain_acquire_policy.validate_acquire_policy(mutated),
        )

    def test_rejects_timeout_rewrite_before_native_acquire(self) -> None:
        mutated = SOURCE.replace(
            "    uint image_index = 0;",
            "    timeout_ns = 1_000_000_000;\n    uint image_index = 0;",
            1,
        )
        self.assertIn(
            "acquisition transaction must not rewrite timeout_ns",
            check_swapchain_acquire_policy.validate_acquire_policy(mutated),
        )

    def test_rejects_precommit_mutation(self) -> None:
        mutated = SOURCE.replace(
            "uint image_index = 0;",
            "slot.acquire_pending = true;\n    uint image_index = 0;",
            1,
        )
        self.assertIn(
            "transaction must not mutate slot.acquire_pending",
            check_swapchain_acquire_policy.validate_acquire_policy(mutated),
        )

    def test_rejects_commit_before_result_mapping(self) -> None:
        mutated = SOURCE.replace(
            "    check_wsi_backend_result(\n"
            "        context: &context,\n"
            "        kind:    WsiResultKind.ACQUIRE,",
            "    return commit_acquired_image(\n"
            "    );\n"
            "    check_wsi_backend_result(\n"
            "        context: &context,\n"
            "        kind:    WsiResultKind.ACQUIRE,",
            1,
        )
        self.assertIn(
            "acquisition commit must follow successful result mapping",
            check_swapchain_acquire_policy.validate_acquire_policy(mutated),
        )

    def test_rejects_missing_commit_field(self) -> None:
        mutated = SOURCE.replace(
            "slot.pending_image = image_index;",
            "(void)image_index;",
            1,
        )
        self.assertIn(
            "commit must assign slot.pending_image exactly once",
            check_swapchain_acquire_policy.validate_acquire_policy(mutated),
        )

    def test_rejects_indexed_retirement_mutation(self) -> None:
        mutated = SOURCE.replace(
            "    slot.pending_acquire_semaphore = candidate.semaphore;",
            "    slot.pending_acquire_semaphore = candidate.semaphore;\n"
            "    slot.acquire_retirements[0] = {};",
            1,
        )
        self.assertIn(
            "acquisition commit must not own retirement entries",
            check_swapchain_acquire_policy.validate_acquire_policy(mutated),
        )

    def test_rejects_waiting_production_poll_substitution(self) -> None:
        mutated = SOURCE.replace(
            "poll_fn:     &poll_swapchain_completion,",
            "poll_fn:     &wait_for_swapchain_completion,",
            1,
        )
        self.assertIn(
            "production acquire must use nonwaiting completion polling",
            check_swapchain_acquire_policy.validate_acquire_policy(mutated),
        )

    def test_rejects_waiting_semaphore_selection(self) -> None:
        mutated = SOURCE.replace(
            "    if (slot.acquire_retirements.len == 0)",
            "    wait_completion(point);\n"
            "    if (slot.acquire_retirements.len == 0)",
            1,
        )
        self.assertIn(
            "acquire semaphore selection must not use wait_completion",
            check_swapchain_acquire_policy.validate_acquire_policy(mutated),
        )


if __name__ == "__main__":
    unittest.main()
