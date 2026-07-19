#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "test" / "retired_api"

FIXTURES = {
    "create_sampler": "create_sampler",
    "destroy_sampler": "destroy_sampler",
    "create_texture_descriptor": "create_texture_descriptor",
    "destroy_texture_descriptor": "destroy_texture_descriptor",
    "create_texture_descriptors": "create_texture_descriptors",
    "texture_descriptor_desc": "TextureDescriptorDesc",
    "descriptor_heap_mode": "DescriptorHeapMode",
    "descriptor_heap_mode_field": "descriptor_heap_mode",
    "texture_descriptor_capacity": "texture_descriptor_capacity",
    "sampler_descriptor_capacity": "sampler_descriptor_capacity",
    "descriptor_buffer_caps": "descriptor_buffer",
    "descriptor_indexing_caps": "descriptor_indexing",
    "max_texture_descriptors_caps": "max_texture_descriptors",
    "max_sampler_descriptors_caps": "max_sampler_descriptors",
    "default_texture_descriptors": "DEFAULT_TEXTURE_DESCRIPTORS",
    "default_sampler_descriptors": "DEFAULT_SAMPLER_DESCRIPTORS",
    "max_descriptor_slots": "MAX_DESCRIPTOR_SLOTS",
    "buffer_handle": "BufferHandle",
    "shader_handle": "ShaderHandle",
    "shader_handle_invalid": "SHADER_HANDLE_INVALID",
    "max_shaders": "MAX_SHADERS",
    "create_shader": "create_shader",
    "destroy_shader": "destroy_shader",
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
    "memory_kind": "MemoryKind",
    "frame_token": "FrameToken",
    "begin_frame": "begin_frame",
    "alloc_frame_span": "alloc_frame_span",
    "end_frame": "end_frame",
    "with_frame": "with_frame",
    "default_frame_arena_size": "DEFAULT_FRAME_ARENA_SIZE",
    "frame_arena_size": "frame_arena_size",
    "frames_in_flight": "frames_in_flight",
    "debug_frame": "FRAME",
    "arena_full": "ARENA_FULL",
    "staging_arena_size": "staging_arena_size",
    "readback_arena_size": "readback_arena_size",
    "default_staging_arena_size": "DEFAULT_STAGING_ARENA_SIZE",
    "default_readback_arena_size": "DEFAULT_READBACK_ARENA_SIZE",
    "debug_semaphore": "SEMAPHORE",
    "persistent_alloc_desc": "PersistentAllocDesc",
    "persistent_arena_stats": "PersistentArenaStats",
    "max_persistent_allocations": "MAX_PERSISTENT_ALLOCATIONS",
    "default_persistent_arena_size": "DEFAULT_PERSISTENT_ARENA_SIZE",
    "persistent_arena_size": "persistent_arena_size",
    "alloc_persistent_span": "alloc_persistent_span",
    "free_persistent_span": "free_persistent_span",
    "get_persistent_stats": "get_persistent_stats",
    "debug_persistent_span": "PERSISTENT_SPAN",
    "retired_cmd_dispatch_pipeline": "pipeline",
    "retired_cmd_dispatch_indirect_pipeline": "pipeline",
    "retired_cmd_draw_pipeline": "pipeline",
    "retired_cmd_draw_indexed_pipeline": "pipeline",
    "retired_cmd_draw_indirect_pipeline": "pipeline",
    "retired_cmd_draw_indexed_indirect_pipeline": "pipeline",
    "retired_cmd_draw_indexed_indirect_count_pipeline": "pipeline",
    "retired_graphics_pipeline_depth": "depth",
}

ERROR_DIAGNOSTIC = re.compile(
    r"\((?P<path>[^()\r\n]+):(?P<line>\d+):(?P<column>\d+)\) "
    r"Error: (?P<message>[^\r\n]+)$"
)

INVALID_MEMBER_TYPES = {
    "submit_waits": "SubmitDesc",
    "submit_signals": "SubmitDesc",
    "descriptor_heap_mode_field": "DeviceDesc",
    "texture_descriptor_capacity": "DeviceDesc",
    "sampler_descriptor_capacity": "DeviceDesc",
    "staging_arena_size": "DeviceDesc",
    "readback_arena_size": "DeviceDesc",
    "persistent_arena_size": "DeviceDesc",
    "frame_arena_size": "DeviceDesc",
    "frames_in_flight": "DeviceDesc",
    "retired_graphics_pipeline_depth": "GraphicsPipelineDesc",
}

ENUM_VALUES = {
    "debug_frame": ("DebugResourceKind", "FRAME"),
    "debug_semaphore": ("DebugResourceKind", "SEMAPHORE"),
    "debug_persistent_span": ("DebugResourceKind", "PERSISTENT_SPAN"),
}

FIELD_OR_METHODS = {
    "timeline_caps": "DeviceCaps.timeline_semaphore",
    "descriptor_buffer_caps": "DeviceCaps.descriptor_buffer",
    "descriptor_indexing_caps": "DeviceCaps.descriptor_indexing",
    "max_texture_descriptors_caps": "DeviceCaps.max_texture_descriptors",
    "max_sampler_descriptors_caps": "DeviceCaps.max_sampler_descriptors",
}

MACRO_SYMBOLS = {
    "with_frame",
}

RETIRED_PIPELINE_SIGNATURES = {
    "retired_cmd_dispatch_pipeline",
    "retired_cmd_dispatch_indirect_pipeline",
    "retired_cmd_draw_pipeline",
    "retired_cmd_draw_indexed_pipeline",
    "retired_cmd_draw_indirect_pipeline",
    "retired_cmd_draw_indexed_indirect_pipeline",
    "retired_cmd_draw_indexed_indirect_count_pipeline",
}


def diagnostic_points_to_retired_member(
    target: str,
    retired_symbol: str,
    diagnostic: re.Match[str],
) -> bool:
    try:
        source_lines = (PROJECT / f"{target}.c3").read_text(
            encoding="utf-8",
        ).splitlines()
    except OSError:
        return False

    line_number = int(diagnostic.group("line"))
    column = int(diagnostic.group("column"))
    if line_number < 1 or line_number > len(source_lines) or column < 1:
        return False

    source_line = source_lines[line_number - 1]
    occurrences = list(
        re.finditer(
            rf"(?<![A-Za-z0-9_]){re.escape(retired_symbol)}"
            r"(?![A-Za-z0-9_])",
            source_line,
        )
    )
    if len(occurrences) != 1:
        return False

    occurrence = occurrences[0]
    first_index = occurrence.start()
    while (
        first_index > 0
        and re.fullmatch(r"[A-Za-z0-9_:.@]", source_line[first_index - 1])
    ):
        first_index -= 1
    first_column = first_index + 1
    last_column = occurrence.end()
    return first_column <= column <= last_column


def has_expected_diagnostic(
    target: str,
    retired_symbol: str,
    output: str,
) -> bool:
    diagnostics = [
        match
        for line in output.splitlines()
        if (match := ERROR_DIAGNOSTIC.search(line)) is not None
    ]
    if len(diagnostics) != 1:
        return False

    diagnostic = diagnostics[-1]
    filename = diagnostic.group("path").replace("\\", "/").rsplit("/", 1)[-1]
    if filename != f"{target}.c3":
        return False

    message = diagnostic.group("message")
    if target in RETIRED_PIPELINE_SIGNATURES:
        return (
            message == (
                "It is not possible to cast 'PipelineHandle' to "
                "'GpuAddress'."
            )
            and diagnostic_points_to_retired_member(
                target,
                retired_symbol,
                diagnostic,
            )
        )
    if member_type := INVALID_MEMBER_TYPES.get(target):
        return (
            message == f"This is not a valid member of '{member_type}'."
            and diagnostic_points_to_retired_member(
                target,
                retired_symbol,
                diagnostic,
            )
        )
    if enum_value := ENUM_VALUES.get(target):
        enum_type, value = enum_value
        return (
            message == f"'{enum_type}' has no enumeration value '{value}'."
            and diagnostic_points_to_retired_member(
                target,
                retired_symbol,
                diagnostic,
            )
        )
    if field_or_method := FIELD_OR_METHODS.get(target):
        return (
            message == f"There is no field or method '{field_or_method}'."
            and diagnostic_points_to_retired_member(
                target,
                retired_symbol,
                diagnostic,
            )
        )

    diagnostic_symbol = (
        f"@{retired_symbol}"
        if target in MACRO_SYMBOLS
        else retired_symbol
    )
    return (
        re.fullmatch(
            rf"'gpu::{re.escape(diagnostic_symbol)}' could not be found, "
            r"(?:did you spell it right\?|did you perhaps want .+\?)",
            message,
        ) is not None
        and diagnostic_points_to_retired_member(
            target,
            retired_symbol,
            diagnostic,
        )
    )


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
        elif not has_expected_diagnostic(target, retired_symbol, output):
            failures.append(
                f"{target} failed without the expected diagnostic for "
                f"{retired_symbol}"
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
