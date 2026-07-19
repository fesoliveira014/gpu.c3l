#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "test" / "retired_api"

FIXTURES = {
    "buffer_handle": "BufferHandle",
    "public_semaphore": "SemaphoreHandle",
    "queue_idle": "wait_queue_idle",
    "timeline_caps": "timeline_semaphore",
    "submit_waits": "waits",
    "submit_signals": "signals",
    "readback_ticket": "ReadbackTicket",
    "cmd_readback_buffer": "cmd_readback_buffer",
    "cmd_readback_texture": "cmd_readback_texture",
    "poll_readback": "poll_readback",
    "resolve_readback": "resolve_readback",
    "readback_not_ready": "READBACK_NOT_READY",
    "cmd_upload_buffer": "cmd_upload_buffer",
    "cmd_upload_texture": "cmd_upload_texture",
    "texture_upload_desc": "TextureUploadDesc",
    "upload_buffer_data": "upload_buffer_data",
    "upload_texture_data": "upload_texture_data",
    "readback_buffer_data": "readback_buffer_data",
    "readback_texture_data": "readback_texture_data",
    "memory_kind_staging": "STAGING",
    "staging_arena_size": "staging_arena_size",
    "readback_arena_size": "readback_arena_size",
    "default_staging_arena_size": "DEFAULT_STAGING_ARENA_SIZE",
    "default_readback_arena_size": "DEFAULT_READBACK_ARENA_SIZE",
    "debug_semaphore": "SEMAPHORE",
}


def main() -> int:
    failures = []
    for target, retired_symbol in FIXTURES.items():
        result = subprocess.run(
            ["c3c", "build", target, "--path", str(PROJECT)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if result.returncode == 0:
            failures.append(f"{target} unexpectedly compiled")
        elif retired_symbol not in output:
            failures.append(
                f"{target} failed without naming {retired_symbol}"
            )

    if failures:
        print("retired API fixture failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("retired API fixtures fail to compile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())