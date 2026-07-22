#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWAPCHAIN_SOURCE = ROOT / "gpu" / "internal" / "vk" / "swapchain.c3"
FUNCTION_DECLARATION = re.compile(
    r"^fn\s+[^\r\n(]*?\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)
COMMIT_FIELDS = (
    "acquire_pending",
    "readiness_consumed",
    "acquisition",
    "next_acquire_semaphore",
    "pending_acquire_semaphore_index",
    "pending_image",
    "pending_acquire_semaphore",
    "render_completion",
)


def function_blocks(source: str) -> dict[str, str]:
    declarations = list(FUNCTION_DECLARATION.finditer(source))
    return {
        declaration.group(1): source[
            declaration.start():
            declarations[index + 1].start()
            if index + 1 < len(declarations)
            else len(source)
        ]
        for index, declaration in enumerate(declarations)
    }


def validate_acquire_policy(source: str) -> list[str]:
    blocks = function_blocks(source)
    failures = []
    required = (
        "vk_acquire_next_image",
        "prepare_swapchain_acquisition",
        "vk_acquire_next_image_with",
        "commit_acquired_image",
        "select_acquire_semaphore_with",
    )
    for name in required:
        if name not in blocks:
            failures.append(f"missing acquisition policy function {name}")
    if failures:
        return failures

    wrapper = blocks["vk_acquire_next_image"]
    if not re.search(r"timeout_ns\s*:\s*timeout_ns", wrapper):
        failures.append("production acquire must forward timeout_ns to the transaction")
    if not re.search(
        r"poll_fn\s*:\s*&poll_swapchain_completion",
        wrapper,
    ):
        failures.append("production acquire must use nonwaiting completion polling")

    transaction = blocks["vk_acquire_next_image_with"]
    native_call = re.compile(
        r"vk::Result\s+result\s*=\s*acquire_fn\s*\(\s*"
        r"state\.device,\s*slot\.swapchain,\s*timeout_ns,\s*"
        r"candidate\.semaphore,",
        re.DOTALL,
    )
    if not native_call.search(transaction):
        failures.append("native acquire must receive timeout_ns unchanged")
    if re.search(r"\btimeout_ns\s*=", transaction):
        failures.append("acquisition transaction must not rewrite timeout_ns")
    result_check = transaction.find("check_wsi_backend_result(")
    commit_call = transaction.find("return commit_acquired_image(")
    if result_check < 0 or commit_call < 0 or result_check > commit_call:
        failures.append("acquisition commit must follow successful result mapping")

    prepare = blocks["prepare_swapchain_acquisition"]
    for name, block in (("prepare", prepare), ("transaction", transaction)):
        for field in COMMIT_FIELDS:
            if re.search(rf"slot\.{field}\s*=", block):
                failures.append(f"{name} must not mutate slot.{field}")

    commit = blocks["commit_acquired_image"]
    for field in COMMIT_FIELDS:
        assignments = re.findall(rf"slot\.{field}\s*=", commit)
        if len(assignments) != 1:
            failures.append(f"commit must assign slot.{field} exactly once")
    if re.search(r"slot\.acquire_retirements(?:\s*\[[^]]+\])?\s*=", commit):
        failures.append("acquisition commit must not own retirement entries")

    selector = blocks["select_acquire_semaphore_with"]
    for token in ("wait_completion", "sleep(", "delay(", "while (", "do {"):
        if token in selector:
            failures.append(f"acquire semaphore selection must not use {token}")

    poll = blocks.get("poll_swapchain_completion", "")
    poll_body = re.compile(
        r"return\s+vk_poll_completion\s*\(\s*device,\s*"
        r"gpu::internal::completion_point_queue_id\(point\),\s*"
        r"gpu::internal::completion_point_sequence\(point\),\s*\)\s*;",
        re.DOTALL,
    )
    if not poll_body.search(poll):
        failures.append("production swapchain polling must remain a direct poll")

    return failures


def main() -> int:
    failures = validate_acquire_policy(
        SWAPCHAIN_SOURCE.read_text(encoding="utf-8")
    )
    if failures:
        print("swapchain acquisition policy violations:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("swapchain acquisition timeout and commit policy is isolated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
