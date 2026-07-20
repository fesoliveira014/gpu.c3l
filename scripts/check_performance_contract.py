#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_COMMANDS = (
    "cmd_barrier",
    "cmd_dispatch_indirect",
    "cmd_draw_indirect",
    "cmd_draw_indexed_indirect",
    "cmd_draw_indexed_indirect_count",
    "cmd_dispatch_generated",
    "cmd_draw_generated",
    "cmd_draw_indexed_generated",
)
BACKEND_COMMANDS = (
    "vk_cmd_dispatch_indirect",
    "vk_cmd_dispatch_generated",
    "execute_generated_work",
)
RECORDING_FORBIDDEN = (
    "lock_device_registry(",
    "alloc::new",
    "mem::new",
    "create_compute_pipeline(",
    "create_graphics_pipeline(",
    "prepare_shader_code(",
)
DESTRUCTION_FORBIDDEN = (
    "wait_completion(",
    "wait_queue_idle(",
    "device_wait_idle(",
    "vk::device_wait_idle(",
    "enqueue_deferred",
    "deferred_release",
)
POINT_ALLOCATION_FORBIDDEN = (
    "alloc::new",
    "mem::new",
    ".alloc(",
)


def function_body(source: str, name: str) -> str:
    declaration = re.search(
        rf"(?m)^fn\s+[^\n]*\b{re.escape(name)}\s*\(",
        source,
    )
    if declaration is None:
        raise ValueError(f"missing function {name}")
    start = source.find("{", declaration.end())
    if start < 0:
        raise ValueError(f"missing body for {name}")
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise ValueError(f"unterminated body for {name}")


def read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def reject_tokens(
    errors: list[str],
    relative: str,
    name: str,
    body: str,
    tokens: tuple[str, ...],
) -> None:
    for token in tokens:
        if token in body:
            errors.append(f"{relative}:{name} contains forbidden {token}")


def check(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    public_source = read(root, "gpu/command.c3")
    backend_source = read(root, "gpu/vk/command.c3")
    sync_source = read(root, "gpu/vk/sync.c3")
    queue_source = read(root, "gpu/vk/queue.c3")
    texture_source = read(root, "gpu/vk/texture.c3")
    device_source = read(root, "gpu/device.c3")
    command_state_source = read(root, "gpu/vk/command_state.c3")
    lifetime_source = read(root, "gpu/vk/lifetime.c3")
    command_bench = read(root, "test/src/command_record_bench.c3")
    lifecycle_bench = read(root, "test/src/lifecycle_bench.c3")

    for name in PUBLIC_COMMANDS:
        reject_tokens(
            errors,
            "gpu/command.c3",
            name,
            function_body(public_source, name),
            RECORDING_FORBIDDEN,
        )
    for name in BACKEND_COMMANDS:
        reject_tokens(
            errors,
            "gpu/vk/command.c3",
            name,
            function_body(backend_source, name),
            RECORDING_FORBIDDEN,
        )

    reject_tokens(
        errors,
        "gpu/vk/sync.c3",
        "vk_cmd_barrier",
        function_body(sync_source, "vk_cmd_barrier"),
        RECORDING_FORBIDDEN,
    )
    reject_tokens(
        errors,
        "gpu/vk/queue.c3",
        "reserve_queue_completion_locked",
        function_body(queue_source, "reserve_queue_completion_locked"),
        POINT_ALLOCATION_FORBIDDEN,
    )
    reject_tokens(
        errors,
        "gpu/vk/texture.c3",
        "vk_destroy_texture",
        function_body(texture_source, "vk_destroy_texture"),
        DESTRUCTION_FORBIDDEN,
    )

    acquire = function_body(
        backend_source,
        "acquire_generated_preprocess_buffer",
    )
    acquire_steps = (
        "find_generated_preprocess_buffer(",
        "take_generated_preprocess_buffer(",
        "create_buffer_with_alignment(",
    )
    try:
        positions = [acquire.index(step) for step in acquire_steps]
        if positions != sorted(positions):
            errors.append(
                "gpu/vk/command.c3 generated preprocess reuse must precede allocation"
            )
    except ValueError as error:
        errors.append(
            "gpu/vk/command.c3 generated preprocess acquisition is missing "
            f"{error.args[0]}"
        )

    required_text = (
        (
            backend_source,
            "gpu/vk/command.c3",
            "generated_recording_allocations",
        ),
        (
            command_state_source,
            "gpu/vk/command_state.c3",
            "recycle_generated_preprocess_buffers_locked(",
        ),
        (
            lifetime_source,
            "gpu/vk/lifetime.c3",
            "recycle_generated_preprocess_buffers_locked(",
        ),
        (
            device_source,
            "gpu/device.c3",
            "device_registry_lock_acquisitions",
        ),
        (
            queue_source,
            "gpu/vk/queue.c3",
            "completion_point_allocations",
        ),
        (
            sync_source,
            "gpu/vk/sync.c3",
            "completion_wait_calls",
        ),
        (
            lifetime_source,
            "gpu/vk/lifetime.c3",
            "command_buffer_stats(",
        ),
        (
            command_bench,
            "test/src/command_record_bench.c3",
            "invariants: registry_locks=%d recording_allocations=%d "
            "draw_compilations=%d preprocess_allocations=%d",
        ),
        (
            lifecycle_bench,
            "test/src/lifecycle_bench.c3",
            "invariants: point_allocations=%d destruction_waits=%d "
            "deferred_releases=%d",
        ),
        (
            lifecycle_bench,
            "test/src/lifecycle_bench.c3",
            "command_buffer_stats(",
        ),
    )
    for source, relative, token in required_text:
        if token not in source:
            errors.append(f"{relative} is missing performance evidence token {token}")

    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("performance contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
