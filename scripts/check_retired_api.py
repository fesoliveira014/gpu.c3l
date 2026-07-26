#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "test" / "retired_api"

FIXTURES = {
    "texture_dimension": "TextureDimension",
    "d24_unorm_s8_uint": "D24_UNORM_S8_UINT",
    "clear_depth_stencil": "ClearDepthStencil",
    "sampler_identity": "Sampler",
    "sampler_invalid": "SAMPLER_INVALID",
    "publish_sampler": "publish_sampler",
    "texture_desc_dimension": "dimension",
    "texture_desc_depth": "depth",
    "texture_view_desc_format": "format",
    "device_request": "DeviceRequest",
    "device_request_support": "DeviceRequestSupport",
    "strict_device_request": "strict_device_request",
    "request_presentation": "request_presentation",
    "request_queues": "request_queues",
    "supports_device_request": "supports_device_request",
    "queue_counts": "QueueCounts",
    "queue_requirements": "QueueRequirements",
    "get_queue_counts": "get_queue_counts",
    "indexed_get_queue": "get_queue",
    "strict_enabled_caps": "strict_enabled",
    "strict_supported_adapter_info": "strict_supported",
    "backend_kind": "BackendKind",
    "runtime_backend": "backend",
    "get_device_backend": "get_device_backend",
    "runtime_enable_validation": "enable_validation",
    "runtime_track_resource_lifetimes": "track_resource_lifetimes",
    "request_resource_agnostic_texture_sync": (
        "request_resource_agnostic_texture_sync"
    ),
    "resource_agnostic_texture_sync_caps": (
        "resource_agnostic_texture_sync"
    ),
    "contract_validation_object_boundaries": "OBJECT_BOUNDARIES",
    "create_device_from_desc": "create_device_from_desc",
    "create_sampler": "create_sampler",
    "destroy_sampler": "destroy_sampler",
    "create_texture_descriptor": "create_texture_descriptor",
    "destroy_texture_descriptor": "destroy_texture_descriptor",
    "create_texture_descriptors": "create_texture_descriptors",
    "texture_descriptor_desc": "TextureDescriptorDesc",
    "descriptor_heap_mode": "DescriptorHeapMode",
    "descriptor_buffer_caps": "descriptor_buffer",
    "descriptor_indexing_caps": "descriptor_indexing",
    "max_texture_descriptors_caps": "max_texture_descriptors",
    "max_sampler_descriptors_caps": "max_sampler_descriptors",
    "default_texture_descriptors": "DEFAULT_TEXTURE_DESCRIPTORS",
    "default_sampler_descriptors": "DEFAULT_SAMPLER_DESCRIPTORS",
    "max_descriptor_slots": "MAX_DESCRIPTOR_SLOTS",
    "buffer_handle": "BufferHandle",
    "buffer_barrier": "BufferBarrier",
    "global_barrier": "GlobalBarrier",
    "stage": "Stage",
    "hazard": "Hazard",
    "hazard_descriptors": "descriptors",
    "texture_use": "TextureUse",
    "prior_use": "prior_use",
    "cmd_buffer_barrier": "cmd_buffer_barrier",
    "cmd_global_barrier": "cmd_global_barrier",
    "shader_handle": "ShaderHandle",
    "shader_handle_invalid": "SHADER_HANDLE_INVALID",
    "max_shaders": "MAX_SHADERS",
    "create_shader": "create_shader",
    "prepare_shader_code": "prepare_shader_code",
    "create_compute_pipelines": "create_compute_pipelines",
    "create_graphics_pipelines": "create_graphics_pipelines",
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
    "debug_frame": "FRAME",
    "arena_full": "ARENA_FULL",
    "default_staging_arena_size": "DEFAULT_STAGING_ARENA_SIZE",
    "default_readback_arena_size": "DEFAULT_READBACK_ARENA_SIZE",
    "debug_semaphore": "SEMAPHORE",
    "persistent_alloc_desc": "PersistentAllocDesc",
    "persistent_arena_stats": "PersistentArenaStats",
    "max_persistent_allocations": "MAX_PERSISTENT_ALLOCATIONS",
    "default_persistent_arena_size": "DEFAULT_PERSISTENT_ARENA_SIZE",
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
    "retired_pipeline_topology": "topology",
    "retired_pipeline_raster": "raster",
    "retired_graphics_pipeline_blend": "blend",
    "retired_pipeline_colors": "colors",
    "retired_compute_push_constant": "push_constant_size",
    "retired_raster_state": "RasterState",
    "retired_cmd_begin_render_pass_state": "state",
    "cmd_begin_render_pass_with_state": "cmd_begin_render_pass_with_state",
    "retired_begin_commands_queue": "queue",
    "retired_reserve_generated_scratch_queue": "queue",
    "retired_release_generated_scratch_queue": "queue",
    "dynamic_graphics_pipeline_desc": "DynamicGraphicsPipelineDesc",
    "color_target_format": "ColorTargetFormat",
    "color_target_blend_state": "ColorTargetBlendState",
    "request_dynamic_color_state": "request_dynamic_color_state",
    "create_dynamic_graphics_pipeline": "create_dynamic_graphics_pipeline",
    "create_dynamic_graphics_pipelines": "create_dynamic_graphics_pipelines",
    "dynamic_color_state_caps": "dynamic_color_state",
    "color_target_state_format": "format",
    "full_render_graphics_state": "full_render_graphics_state",
    "command_list_handle": "CommandListHandle",
    "command_list_handle_invalid": "COMMAND_LIST_HANDLE_INVALID",
    "command_list_device": "device",
    "command_list_handle_member": "handle",
    "executable_command_list_device": "device",
    "executable_command_list_handle": "handle",
}

ERROR_DIAGNOSTIC = re.compile(
    r"\((?P<path>[^()\r\n]+):(?P<line>\d+):(?P<column>\d+)\) "
    r"Error: (?P<message>[^\r\n]+)$"
)

INVALID_MEMBER_TYPES = {
    "runtime_backend": "RuntimeDesc",
    "runtime_enable_validation": "RuntimeDesc",
    "runtime_track_resource_lifetimes": "RuntimeDesc",
    "texture_desc_dimension": "TextureDesc",
    "texture_desc_depth": "TextureDesc",
    "texture_view_desc_format": "TextureViewDesc",
    "submit_waits": "SubmitDesc",
    "submit_signals": "SubmitDesc",
    "retired_graphics_pipeline_depth": "GraphicsPipelineDesc",
    "retired_pipeline_topology": "GraphicsPipelineDesc",
    "retired_pipeline_raster": "GraphicsPipelineDesc",
    "retired_graphics_pipeline_blend": "GraphicsPipelineDesc",
    "retired_pipeline_colors": "GraphicsPipelineDesc",
    "retired_compute_push_constant": "ComputePipelineDesc",
    "color_target_state_format": "ColorTargetState",
    "prior_use": "AcquiredImage",
}

ENUM_VALUES = {
    "contract_validation_object_boundaries": (
        "ContractValidation",
        "OBJECT_BOUNDARIES",
    ),
    "d24_unorm_s8_uint": ("Format", "D24_UNORM_S8_UINT"),
    "debug_frame": ("DebugResourceKind", "FRAME"),
    "debug_semaphore": ("DebugResourceKind", "SEMAPHORE"),
    "debug_persistent_span": ("DebugResourceKind", "PERSISTENT_SPAN"),
}

FIELD_OR_METHODS = {
    "timeline_caps": "DeviceCaps.timeline_semaphore",
    "hazard_descriptors": "HazardFlags.descriptors",
    "descriptor_buffer_caps": "DeviceCaps.descriptor_buffer",
    "descriptor_indexing_caps": "DeviceCaps.descriptor_indexing",
    "max_texture_descriptors_caps": "DeviceCaps.max_texture_descriptors",
    "max_sampler_descriptors_caps": "DeviceCaps.max_sampler_descriptors",
    "dynamic_color_state_caps": "DeviceCaps.dynamic_color_state",
    "resource_agnostic_texture_sync_caps": (
        "DeviceCaps.resource_agnostic_texture_sync"
    ),
    "strict_enabled_caps": "DeviceCaps.strict_enabled",
    "strict_supported_adapter_info": "AdapterInfo.strict_supported",
    "command_list_device": "CommandList.device",
    "command_list_handle_member": "CommandList.handle",
    "executable_command_list_device": "ExecutableCommandList.device",
    "executable_command_list_handle": "ExecutableCommandList.handle",
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

RETIRED_ALLOCATOR_SIGNATURES = {
    "retired_begin_commands_queue",
    "retired_reserve_generated_scratch_queue",
    "retired_release_generated_scratch_queue",
}

RETIRED_RENDER_PASS_SIGNATURES = {
    "retired_cmd_begin_render_pass_state",
}

LIVE_SCAN_ROOTS = (
    ROOT / "gpu",
    ROOT / "test" / "cpu",
    ROOT / "test" / "src",
    ROOT / "docs",
    ROOT / "README.md",
)
LIVE_SCAN_SUFFIXES = {".c3", ".json", ".md", ".txt"}
RETIRED_DEVICE_REQUEST_MARKERS = {
    "DeviceRequest",
    "DeviceRequestSupport",
    "strict_device_request",
    "request_presentation",
    "request_queues",
    "supports_device_request",
    "DeviceCaps.strict_enabled",
    "AdapterInfo.strict_supported",
}
LIVE_RETIRED_PATTERNS = {
    "BackendKind": re.compile(r"\bBackendKind\b"),
    "get_device_backend": re.compile(r"\bget_device_backend\b"),
    "DeviceRequest": re.compile(r"\bDeviceRequest\b"),
    "DeviceRequestSupport": re.compile(r"\bDeviceRequestSupport\b"),
    "strict_device_request": re.compile(r"\bstrict_device_request\b"),
    "request_presentation": re.compile(r"\brequest_presentation\b"),
    "request_queues": re.compile(r"\brequest_queues\b"),
    "supports_device_request": re.compile(r"\bsupports_device_request\b"),
    "DeviceCaps.strict_enabled": re.compile(r"\bstrict_enabled\b"),
    "AdapterInfo.strict_supported": re.compile(r"\bstrict_supported\b"),
    "TextureDimension": re.compile(r"\bTextureDimension\b"),
    "D24_UNORM_S8_UINT": re.compile(r"\bD24_UNORM_S8_UINT\b"),
    "ClearDepthStencil": re.compile(r"\bClearDepthStencil\b"),
    "public Sampler": re.compile(
        r"\bgpu::Sampler\b|\bstruct\s+Sampler\b|(?<!::)\bSampler\s+sampler\b"
    ),
    "SAMPLER_INVALID": re.compile(r"\bSAMPLER_INVALID\b"),
    "publish_sampler": re.compile(r"\bpublish_sampler\b"),
    "request_resource_agnostic_texture_sync": re.compile(
        r"\brequest_resource_agnostic_texture_sync\b"
    ),
    "resource_agnostic_texture_sync": re.compile(
        r"\bresource_agnostic_texture_sync\b"
    ),
    "ShaderCode": re.compile(r"\bShaderCode\b"),
    "ShaderStage": re.compile(r"(?<!spvreflect::)\bShaderStage\b"),
    "prepare_shader_code": re.compile(r"\bprepare_shader_code\b"),
    "create_compute_pipelines": re.compile(
        r"(?<!vk::)\bcreate_compute_pipelines\b"
    ),
    "create_graphics_pipelines": re.compile(
        r"(?<!vk::)\bcreate_graphics_pipelines\b"
    ),
}
README_RETIRED_PATTERNS = {
    "automatic per-worker command pools": re.compile(
        r"\bautomatic per-worker command pools\b"
    ),
}
RETIRED_DESC_FIELDS = {
    "RuntimeDesc.backend": re.compile(
        r"(?:\bRuntimeDesc\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*"
        r"|\bfn\s+(?:gpu::)?RuntimeDesc\b[^\r\n=]*=>\s*)\{"
        r"(?:(?!\};).)*?\.backend\s*=",
        re.DOTALL,
    ),
    "RuntimeDesc.enable_validation": re.compile(
        r"(?:\bRuntimeDesc\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*"
        r"|\bfn\s+(?:gpu::)?RuntimeDesc\b[^\r\n=]*=>\s*)\{"
        r"(?:(?!\};).)*?\.enable_validation\s*=",
        re.DOTALL,
    ),
    "RuntimeDesc.track_resource_lifetimes": re.compile(
        r"(?:\bRuntimeDesc\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*"
        r"|\bfn\s+(?:gpu::)?RuntimeDesc\b[^\r\n=]*=>\s*)\{"
        r"(?:(?!\};).)*?\.track_resource_lifetimes\s*=",
        re.DOTALL,
    ),
    "TextureDesc.dimension/depth": re.compile(
        r"\bTextureDesc\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*\{"
        r"(?:(?!\};).)*?\.(?:dimension|depth)\s*=",
        re.DOTALL,
    ),
    "TextureViewDesc.format": re.compile(
        r"\bTextureViewDesc\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*\{"
        r"(?:(?!\};).)*?\.format\s*=",
        re.DOTALL,
    ),
}
MARKDOWN_C3_FENCE = re.compile(
    r"(?ms)^```(?:c3|c3c)\s*\n(?P<code>.*?)^```\s*$",
)
QUEUE_DECLARATION = re.compile(
    r"\b(?:gpu::)?Queue\s+([A-Za-z_][A-Za-z0-9_]*)\b",
)
RETIRED_ALLOCATOR_CALLS = {
    "begin_commands(Queue)": (
        "begin_commands",
        re.compile(r"\b(?:gpu::)?begin_commands\s*\(\s*{queue}\s*\)"),
    ),
    "reserve_generated_scratch(Queue, ...)": (
        "reserve_generated_scratch",
        re.compile(
            r"\b(?:gpu::)?reserve_generated_scratch\s*\(\s*{queue}\s*,"
        ),
    ),
    "release_generated_scratch(Queue, ...)": (
        "release_generated_scratch",
        re.compile(
            r"\b(?:gpu::)?release_generated_scratch\s*\(\s*{queue}\s*,"
        ),
    ),
}


def mask_c3_non_code(source: str) -> str:
    masked = list(source)
    index = 0
    quote = None
    block_end = None
    block_depth = 0
    while index < len(source):
        if quote is not None:
            if source[index] == "\\":
                masked[index] = " "
                if index + 1 < len(source):
                    masked[index + 1] = " "
                index += 2
            elif source[index] == quote:
                masked[index] = " "
                quote = None
                index += 1
            else:
                if source[index] not in "\r\n":
                    masked[index] = " "
                index += 1
            continue
        if block_end is not None:
            block_start = "/*" if block_end == "*/" else "<*"
            if source.startswith(block_start, index):
                masked[index:index + 2] = "  "
                block_depth += 1
                index += 2
            elif source.startswith(block_end, index):
                masked[index:index + 2] = "  "
                block_depth -= 1
                index += 2
                if block_depth == 0:
                    block_end = None
            else:
                if source[index] not in "\r\n":
                    masked[index] = " "
                index += 1
            continue
        if source.startswith("//", index):
            while index < len(source) and source[index] not in "\r\n":
                masked[index] = " "
                index += 1
        elif source.startswith("/*", index):
            masked[index:index + 2] = "  "
            block_end = "*/"
            block_depth = 1
            index += 2
        elif source.startswith("<*", index):
            masked[index:index + 2] = "  "
            block_end = "*>"
            block_depth = 1
            index += 2
        elif source[index] in "\"'":
            masked[index] = " "
            quote = source[index]
            index += 1
        else:
            index += 1
    return "".join(masked)


def c3_regions(path: Path, source: str):
    if path.suffix == ".c3":
        yield 0, mask_c3_non_code(source)
    elif path.suffix == ".md":
        for fence in MARKDOWN_C3_FENCE.finditer(source):
            yield fence.start("code"), mask_c3_non_code(fence.group("code"))


def retired_allocator_signature_usages(path: Path, source: str):
    for offset, code in c3_regions(path, source):
        queue_names = set(QUEUE_DECLARATION.findall(code))
        for queue_name in queue_names:
            escaped = re.escape(queue_name)
            for marker, (_, pattern) in RETIRED_ALLOCATOR_CALLS.items():
                concrete = re.compile(pattern.pattern.format(queue=escaped))
                for match in concrete.finditer(code):
                    yield offset + match.start(), marker


def live_scan_files(roots: tuple[Path, ...] = LIVE_SCAN_ROOTS):
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix not in LIVE_SCAN_SUFFIXES:
                continue
            yield path


def is_retired_migration_reference(
    relative_path: str,
    source: str,
    position: int,
) -> bool:
    if relative_path != "docs/api.md":
        return False
    section_start = source.find("### Breaking migration")
    if section_start < 0 or position < section_start:
        return False
    section_end = source.find("\n## ", section_start)
    return section_end < 0 or position < section_end


def find_live_retired_usages(
    roots: tuple[Path, ...] = LIVE_SCAN_ROOTS,
) -> list[str]:
    failures = []
    for path in live_scan_files(roots):
        source = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(ROOT).as_posix()
        for marker, pattern in LIVE_RETIRED_PATTERNS.items():
            for match in pattern.finditer(source):
                if (
                    marker in RETIRED_DEVICE_REQUEST_MARKERS
                    and is_retired_migration_reference(
                        relative_path,
                        source,
                        match.start(),
                    )
                ):
                    continue
                line = source.count("\n", 0, match.start()) + 1
                failures.append(f"{relative_path}:{line}: {marker}")
        if path.name == "README.md":
            for marker, pattern in README_RETIRED_PATTERNS.items():
                for match in pattern.finditer(source):
                    line = source.count("\n", 0, match.start()) + 1
                    failures.append(f"{relative_path}:{line}: {marker}")
        for marker, pattern in RETIRED_DESC_FIELDS.items():
            for match in pattern.finditer(source):
                line = source.count("\n", 0, match.start()) + 1
                failures.append(f"{relative_path}:{line}: {marker}")
        for position, marker in retired_allocator_signature_usages(path, source):
            line = source.count("\n", 0, position) + 1
            failures.append(f"{relative_path}:{line}: {marker}")
    return failures


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
    if target in RETIRED_ALLOCATOR_SIGNATURES:
        return (
            message == (
                "It is not possible to cast 'Queue' to 'CommandAllocator*'."
            )
            and diagnostic_points_to_retired_member(
                target,
                retired_symbol,
                diagnostic,
            )
        )
    if target == "indexed_get_queue":
        return (
            message == (
                "This argument would exceed the number of parameters, "
                "did you add too many arguments?"
            )
            and diagnostic_points_to_retired_member(
                target,
                "0",
                diagnostic,
            )
        )
    if target in RETIRED_RENDER_PASS_SIGNATURES:
        return (
            message == (
                "This argument would exceed the number of parameters, "
                "did you add too many arguments?"
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
    failures = find_live_retired_usages()
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
