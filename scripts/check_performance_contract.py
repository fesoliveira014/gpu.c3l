#!/usr/bin/env python3
from __future__ import annotations

import hashlib
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
BACKEND_OWNERSHIP_DIGESTS = {
    "vk_begin_commands_with_context": "3292d8a58303e0fc2992ffe89976ecc542e0712da03aecc0efdee0fc4deb37de",
    "generated_preprocess_compatible": "df5842124c9c4e5427be4440e3a6ce3df20282e3d3b7d9d7557ac6bec007b5a0",
    "ensure_generated_preprocess_pool_capacity_locked": "8f07b4af72b8e015ec80aee966099e0a362323110017cff1c37172133ab00569",
    "command_recording_stats": "381e23f43057fb0882396bc074d75c196da6fc3fe2b74547f300223eced04a2e",
    "free_generated_preprocess_buffers": "a4e9fc1d9b5e6d1a010e634ceab4bfd7090707552f5d39abf84d736faa59c744",
    "ensure_generated_preprocess_capacity": "f95bc1998ad4440be1ce002753025c62638647437ae6d5af14a1237effa18b95",
    "execute_generated_work": "2c7d0ee833035e94166a955d64f02ca8addbf9bc8050a343d62ccf15b9b90bf8",
    "recycle_generated_preprocess_buffers_locked": "4710b2f580e8c148f0ae4e13a6454bbd270f5dc6b54ba8b82a218a91c57f40b1",
    "take_generated_preprocess_buffer": "7a0262396bf857c810ab5b08a384bcdb12d404b9da2f05133d0513dde605628e",
    "acquire_generated_preprocess_buffer": "41cdde328b77255c518829a196a22bd8091e12a3361e6d70486cf986fa43a754",
}
COMMAND_STATE_OWNERSHIP_DIGESTS = {
    "release_command": "052c7e5e284197a6d22729dc7045397c37535c0a9c50aaf3ec163211d0859157",
    "destroy_command_table": "45c57c05ed629582c17062c6829d1a2a9146ba44bce9552b186cefc0692723d7",
}
LIFETIME_OWNERSHIP_DIGESTS = {
    "publish_submitted_commands": "cdbc47b4f1b25be4fc79e04f6ae8ff410b219ab6854924df0edabd389376f066",
    "release_submitted_command_batch": "f9b2c92ba67ebdf818a8d6417074d55012b6756dad85106e05593c68748f775f",
    "release_completed_submitted_commands_locked": "bd81cb2ab4514a5428d70fa75883594913b7e77d0c15f123593b008f7e0e50d4",
    "destroy_submitted_commands": "1858d0b6e5a0d7659b1ed6ca6a08abf7e13f5577cdcea9cdc48dfdcde0d95d60",
}
DEVICE_OWNERSHIP_DIGESTS = {
    "destroy_state": "aa9b3c53333b2eb21b29bc54b629f7a184ff66602ed9dfaefb328baf8d93b0b2",
}



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


def function_names(source: str) -> tuple[str, ...]:
    return tuple(
        re.findall(
            r"(?m)^fn\s+[^\n]*\b([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            source,
        )
    )


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
    backend_device_source = read(root, "gpu/vk/device.c3")
    lifetime_source = read(root, "gpu/vk/lifetime.c3")
    command_bench = read(root, "test/src/command_record_bench.c3")
    lifecycle_bench = read(root, "test/src/lifecycle_bench.c3")

    ownership_sources = (
        ("gpu/vk/command.c3", backend_source, BACKEND_OWNERSHIP_DIGESTS),
        (
            "gpu/vk/command_state.c3",
            command_state_source,
            COMMAND_STATE_OWNERSHIP_DIGESTS,
        ),
        ("gpu/vk/lifetime.c3", lifetime_source, LIFETIME_OWNERSHIP_DIGESTS),
        ("gpu/vk/device.c3", backend_device_source, DEVICE_OWNERSHIP_DIGESTS),
    )
    for relative, source, digests in ownership_sources:
        for name, expected in digests.items():
            body = function_body(source, name)
            actual = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if actual != expected:
                errors.append(
                    f"{relative}:{name} generated preprocess ownership flow "
                    "must match reviewed source"
                )

    reviewed_ownership = {
        relative: digests
        for relative, _, digests in ownership_sources
    }
    for path in sorted((root / "gpu/vk").rglob("*.c3")):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        digests = reviewed_ownership.get(relative, {})
        for name in function_names(source):
            body = function_body(source, name)
            touches_ownership = (
                "generated_preprocess" in name
                or "generated_preprocess" in body
            )
            if touches_ownership and name not in digests:
                errors.append(
                    f"{relative}:{name} generated preprocess ownership flow "
                    "is unreviewed"
                )

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
    if "find_generated_preprocess_buffer(" in acquire:
        errors.append(
            "gpu/vk/command.c3 reuses a generated preprocess address "
            "within one command record"
        )
    allowed_record_lines = (
        "ensure_generated_preprocess_capacity(state, record);",
        "uint index = record.generated_preprocess_count++;",
        "record.generated_preprocess[index] = pooled;",
        "return &record.generated_preprocess[index];",
        "ensure_generated_preprocess_capacity(state, record);",
        "uint index = record.generated_preprocess_count++;",
        "record.generated_preprocess[index] = buffer;",
        "return &record.generated_preprocess[index];",
    )
    record_lines = tuple(
        line.strip()
        for line in acquire.splitlines()
        if re.search(r"\brecord\b", line)
    )
    if record_lines != allowed_record_lines:
        errors.append(
            "gpu/vk/command.c3 generated preprocess acquisition must only "
            "append newly acquired buffers"
        )
    acquire_steps = (
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
    execute = function_body(backend_source, "execute_generated_work")
    native_execute_call = (
        "state.device_dispatch.generated_work.cmd_execute_generated_commands(\n"
        "        record.command_buffer,\n"
        "        vk::FALSE,\n"
        "        &info,\n"
        "    );"
    )

    allowed_execute_record_lines = (
        "record,",
        "record.command_buffer,",
    )
    execute_record_lines = tuple(
        line.strip()
        for line in execute.splitlines()
        if re.search(r"\brecord\b", line)
    )
    if (
        execute_record_lines != allowed_execute_record_lines
        or execute.count("acquire_generated_preprocess_buffer(") != 1
    ):
        errors.append(
            "gpu/vk/command.c3 generated execution must acquire a fresh buffer"
        )
    if (
        execute.count(native_execute_call) != 1
        or backend_source.count("cmd_execute_generated_commands") != 1
    ):
        errors.append(
            "gpu/vk/command.c3 generated execution must issue exactly one native call"
        )
    take = function_body(backend_source, "take_generated_preprocess_buffer")
    take_success_block = (
        "*out_buffer = *candidate;\n"
        "        state.generated_preprocess_pool_count--;\n"
        "        *candidate = state.generated_preprocess_pool[\n"
        "            state.generated_preprocess_pool_count\n"
        "        ];\n"
        "        state.generated_preprocess_pool["
        "state.generated_preprocess_pool_count] = {};\n"
        "        (void)state.generated_preprocess_reuses.add("
        "1, AtomicOrdering.RELAXED);\n"
        "        return true;"
    )
    take_steps = (
        "state.resource_mutex.lock()!!;",
        "defer state.resource_mutex.unlock();",
        take_success_block,
    )
    try:
        take_positions = [take.index(step) for step in take_steps]
        take_is_unique_removal = (
            take_positions == sorted(take_positions)
            and all(take.count(step) == 1 for step in take_steps)
            and len(re.findall(r"\bout_buffer\b", take)) == 1
            and take.count("defer") == 1
        )
    except ValueError:
        take_is_unique_removal = False
    if not take_is_unique_removal:
        errors.append(
            "gpu/vk/command.c3 successful pool take must remove "
            "the selected preprocess buffer"
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
