#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_PROJECT = ROOT / "test" / "cpu"
PRIVATE_BACKEND_DECLARATION = "module gpu::vk @private;"

FORBIDDEN_TEXT = {
    "backend_state": "backend state pointer",
    "backendvtable": "backend dispatch table",
    "bufferhandle": "retired BufferHandle",
    "bufferdesc": "retired BufferDesc",
    "bufferusage": "retired BufferUsage",
    "frametoken": "retired FrameToken",
    "memorykind": "retired MemoryKind",
    "begin_frame": "retired begin_frame",
    "alloc_frame_span": "retired alloc_frame_span",
    "end_frame": "retired end_frame",
    "with_frame": "retired with_frame",
    '"name":"default_frame_arena_size"': (
        "retired DEFAULT_FRAME_ARENA_SIZE"
    ),
    '"name":"frame_arena_size"': "retired DeviceDesc.frame_arena_size",
    '"name":"frames_in_flight"': "retired DeviceDesc.frames_in_flight",
    "arena_full": "retired ARENA_FULL",
    "create_buffer": "retired public buffer lifecycle",
    "destroy_buffer": "retired public buffer lifecycle",
    "get_buffer_address": "retired public buffer lifecycle",
    "get_buffer_span": "retired public buffer lifecycle",
    "flush_buffer": "retired public buffer lifecycle",
    "invalidate_buffer": "retired public buffer lifecycle",
    "platformkind": "retired PlatformKind",
    "presentdesc": "retired PresentDesc",
    "semaphoredesc": "retired public semaphore",
    "semaphorehandle": "retired public semaphore",
    "semaphoresignal": "retired public semaphore",
    "semaphorevalue": "retired public semaphore",
    "semaphorewait": "retired public semaphore",
    "create_semaphore": "retired public semaphore",
    "destroy_semaphore": "retired public semaphore",
    "wait_semaphore": "retired public semaphore",
    "wait_queue_idle": "retired wait_queue_idle",
    "timeline_semaphore": "retired timeline capability",
    "max_timeline_semaphore_value_difference": "retired timeline capability",
    "probe_vulkan_version": "Vulkan loader probe",
    "probe_vma_allocator": "VMA probe",
    "range_end": "readback retirement range",
    "recordingcontexthandle": "retired recording context",
    "recording_context": "retired recording context",
    "create_recording_context": "retired recording context",
    "destroy_recording_context": "retired recording context",
    "shared_queues": "backend queue-sharing policy",
    "surfacedesc": "retired SurfaceDesc",
    "ticket.value": "readback retirement value",
    "vk::": "Vulkan type",
    "vma::": "VMA type",
}

FORBIDDEN_SYMBOLS = {
    "DestroyDeviceFn",
    "GetMemoryStatsFn",
    "CreateBufferFn",
    "BeginCommandsFn",
    "SubmitFn",
    "CmdDispatchFn",
    "CmdDrawFn",
    "CmdReadbackBufferFn",
    "ResolveReadbackFn",
    "create_wayland_surface",
    "create_win32_surface",
    "create_x11_surface",
}

PLATFORM_HANDLE_TYPES = {
    "gpu::surface::win32": ("InstanceHandle", "WindowHandle"),
    "gpu::surface::wayland": ("DisplayHandle", "SurfaceHandle"),
    "gpu::surface::x11": ("DisplayHandle", "WindowHandle"),
}

DEBUG_RESOURCE_KINDS = (
    "NONE",
    "DEVICE",
    "GPU_SPAN",
    "TEXTURE",
    "PIPELINE",
    "SWAPCHAIN",
    "SHADER",
    "COMMAND_LIST",
    "TEXTURE_DESCRIPTOR",
    "SAMPLER",
    "ALLOCATION",
)

RETIRED_SOURCE_SYMBOLS = (
    "PlatformKind",
    "PresentDesc",
    "SurfaceDesc",
    "BufferHandle",
    "BufferDesc",
    "BufferUsage",
    "create_buffer(",
    "destroy_buffer(",
    "create_sampler(",
    "destroy_sampler(",
    "get_buffer_address(",
    "get_buffer_span(",
    "flush_buffer(",
    "invalidate_buffer(",
    "RecordingContextHandle",
    "RECORDING_CONTEXT",
    "SemaphoreDesc",
    "SemaphoreHandle",
    "SemaphoreSignal",
    "SemaphoreValue",
    "SemaphoreWait",
    "create_semaphore(",
    "create_recording_context(",
    "destroy_recording_context(",
    "destroy_semaphore(",
    "wait_semaphore(",
    "wait_queue_idle",
    "timeline_semaphore",
    "max_timeline_semaphore_value_difference",
    "SEMAPHORE,",
    "PersistentAllocDesc",
    "PersistentArenaStats",
    "MAX_PERSISTENT_ALLOCATIONS",
    "DEFAULT_PERSISTENT_ARENA_SIZE",
    "persistent_arena_size",
    "alloc_persistent_span(",
    "free_persistent_span(",
    "get_persistent_stats(",
    "PERSISTENT_SPAN",
    "FrameToken",
    "MemoryKind",
    "begin_frame(",
    "alloc_frame_span(",
    "end_frame(",
    "@with_frame",
    "DEFAULT_FRAME_ARENA_SIZE",
    "frame_arena_size",
    "frames_in_flight",
    "ARENA_FULL",
    "FRAME,",
)

RETIRED_SOURCE_PATTERNS = {
    r"struct\s+SubmitDesc\s*\{[^}]*\bwaits\b": "SubmitDesc.waits",
    r"struct\s+SubmitDesc\s*\{[^}]*\bsignals\b": "SubmitDesc.signals",
}


def public_entries(module: dict) -> dict:
    return {
        section: (
            [entry for entry in contents if entry.get("visibility") != "private"]
            if isinstance(contents, list) else contents
        )
        for section, contents in module.items()
    }


def member_schema(
    definition: dict | None,
) -> tuple[tuple[str | None, str | None], ...]:
    return tuple(
        (
            member.get("name"),
            member.get("type", {}).get("name"),
        )
        for member in (definition or {}).get("members", [])
    )


def validate_document(document: dict) -> list[str]:
    modules = document.get("modules", {})
    public_module = modules.get("gpu")
    if public_module is None:
        return ["missing gpu module"]

    public_surface = public_entries(public_module)
    encoded = json.dumps(public_surface, separators=(",", ":"))
    lowered = encoded.lower()
    failures = [
        label
        for token, label in FORBIDDEN_TEXT.items()
        if token in lowered
    ]
    failures.extend(
        symbol
        for symbol in FORBIDDEN_SYMBOLS
        if f'"{symbol}"' in encoded
    )

    functions = {
        entry.get("name"): entry
        for entry in public_surface.get("functions", [])
    }
    types = {
        entry.get("name"): entry
        for entry in public_surface.get("types", [])
    }

    begin_commands = functions.get("begin_commands")
    if begin_commands is None:
        failures.append("missing begin_commands")
    else:
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in begin_commands.get("members", [])
        )
        if parameter_types != ("Queue",):
            failures.append("begin_commands must take one Queue token")
        if begin_commands.get("return_type", {}).get("name") != "CommandList?":
            failures.append("begin_commands must return CommandList?")

    end_commands = functions.get("end_commands")
    if end_commands is None:
        failures.append("missing end_commands")
    else:
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in end_commands.get("members", [])
        )
        if parameter_types != ("CommandList*",):
            failures.append("end_commands must consume CommandList*")
        if (end_commands.get("return_type", {}).get("name")
                != "ExecutableCommandList?"):
            failures.append(
                "end_commands must return ExecutableCommandList?"
            )

    submit = functions.get("submit")
    if submit is None:
        failures.append("missing submit")
    else:
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in submit.get("members", [])
        )
        if parameter_types != ("Queue", "SubmitDesc*"):
            failures.append("submit must take Queue and SubmitDesc*")

    required_functions = {
        "allocate_memory": (
            ("Device*", "AllocationDesc*"),
            "GpuAllocation?",
        ),
        "intern_sampler": (
            ("Device*", "SamplerDesc*"),
            "Sampler?",
        ),
        "publish_sampler": (
            ("Device*", "Sampler"),
            "SamplerIndex?",
        ),
        "get_texture_requirements": (
            ("Device*", "TextureDesc*"),
            "TextureRequirements?",
        ),
        "create_placed_texture": (
            ("Device*", "TextureDesc*", "GpuAllocation", "usz"),
            "TextureHandle?",
        ),
        "create_dedicated_texture": (
            ("Device*", "TextureDesc*", "AllocationDesc*"),
            "DedicatedTexture?",
        ),
        "free_allocation": (
            ("Device*", "GpuAllocation*"),
            "void?",
        ),
        "get_allocation_info": (
            ("Device*", "GpuAllocation"),
            "AllocationInfo?",
        ),
        "get_allocation_span": (
            ("Device*", "GpuAllocation"),
            "GpuSpan?",
        ),
        "get_span_mapping": (
            ("Device*", "GpuSpan"),
            "char[]?",
        ),
        "get_span_address": (
            ("Device*", "GpuSpan"),
            "GpuAddress?",
        ),
        "flush_mapped_span": (
            ("Device*", "GpuSpan"),
            "void?",
        ),
        "invalidate_mapped_span": (
            ("Device*", "GpuSpan"),
            "void?",
        ),
        "cmd_copy_buffer": (
            ("CommandList*", "BufferCopyDesc*"),
            "void?",
        ),
        "cmd_fill_buffer": (
            ("CommandList*", "GpuSpan", "uint"),
            "void?",
        ),
        "cmd_buffer_barrier": (
            ("CommandList*", "BufferBarrier*"),
            "void?",
        ),
        "cmd_copy_buffer_to_texture": (
            ("CommandList*", "BufferTextureCopyDesc*"),
            "void?",
        ),
        "cmd_copy_texture_to_buffer": (
            ("CommandList*", "TextureBufferCopyDesc*"),
            "void?",
        ),
        "cmd_draw_indexed": (
            (
                "CommandList*",
                "PipelineHandle",
                "GpuAddress",
                "GpuAddress",
                "GpuSpan",
                "uint",
                "uint",
                "IndexType",
            ),
            "void?",
        ),
        "cmd_draw_indirect": (
            (
                "CommandList*",
                "PipelineHandle",
                "GpuAddress",
                "GpuAddress",
                "GpuSpan",
                "uint",
            ),
            "void?",
        ),
        "cmd_draw_indexed_indirect": (
            (
                "CommandList*",
                "PipelineHandle",
                "GpuAddress",
                "GpuAddress",
                "GpuSpan",
                "uint",
                "GpuSpan",
                "IndexType",
            ),
            "void?",
        ),
        "cmd_draw_indexed_indirect_count": (
            (
                "CommandList*",
                "PipelineHandle",
                "GpuAddress",
                "GpuAddress",
                "GpuSpan",
                "GpuSpan",
                "uint",
                "GpuSpan",
                "IndexType",
            ),
            "void?",
        ),
        "cmd_dispatch_indirect": (
            (
                "CommandList*",
                "PipelineHandle",
                "GpuAddress",
                "GpuSpan",
            ),
            "void?",
        ),
    }
    required_parameter_names = {
        "intern_sampler": ("device", "desc"),
        "publish_sampler": ("device", "sampler"),
        "get_texture_requirements": ("device", "desc"),
        "create_placed_texture": ("device", "desc", "allocation", "offset"),
        "create_dedicated_texture": (
            "device",
            "desc",
            "allocation_desc",
        ),
        "cmd_copy_buffer": ("commands", "desc"),
        "cmd_fill_buffer": ("commands", "dst", "value"),
        "cmd_buffer_barrier": ("commands", "barrier"),
        "cmd_copy_buffer_to_texture": ("commands", "desc"),
        "cmd_copy_texture_to_buffer": ("commands", "desc"),
        "cmd_draw_indexed": (
            "commands",
            "pipeline",
            "vertex_root",
            "fragment_root",
            "index_span",
            "index_count",
            "instance_count",
            "index_type",
        ),
        "cmd_draw_indirect": (
            "commands",
            "pipeline",
            "vertex_root",
            "fragment_root",
            "args",
            "draw_count",
        ),
        "cmd_draw_indexed_indirect": (
            "commands",
            "pipeline",
            "vertex_root",
            "fragment_root",
            "args",
            "draw_count",
            "index_span",
            "index_type",
        ),
        "cmd_draw_indexed_indirect_count": (
            "commands",
            "pipeline",
            "vertex_root",
            "fragment_root",
            "args",
            "count_span",
            "max_draw_count",
            "index_span",
            "index_type",
        ),
        "cmd_dispatch_indirect": (
            "commands",
            "pipeline",
            "root",
            "args",
        ),
    }
    for name, contract in required_functions.items():
        expected_parameters, expected_return = contract
        function = functions.get(name)
        if function is None:
            failures.append(f"missing {name}")
            continue
        members = function.get("members", [])
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in members
        )
        parameter_names = tuple(member.get("name") for member in members)
        expected_names = required_parameter_names.get(name)
        if (parameter_types != expected_parameters
                or (expected_names is not None
                    and parameter_names != expected_names)):
            failures.append(f"{name} has the wrong parameters")
        if function.get("return_type", {}).get("name") != expected_return:
            failures.append(f"{name} has the wrong return type")

    for name, kind in (
        ("GpuAllocation", "struct"),
        ("Sampler", "struct"),
        ("GpuSpan", "struct"),
        ("MemoryClass", "enum"),
        ("MemoryStats", "struct"),
        ("DebugResourceKind", "enum"),
        ("AllocationDesc", "struct"),
        ("TextureCompatibility", "distinct type"),
        ("TextureRequirements", "struct"),
        ("DedicatedTexture", "struct"),
        ("AllocationInfo", "struct"),
        ("BufferCopyDesc", "struct"),
        ("BufferTextureCopyDesc", "struct"),
        ("TextureBufferCopyDesc", "struct"),
        ("BufferBarrier", "struct"),
    ):
        definition = types.get(name)
        if definition is None or definition.get("kind") != kind:
            failures.append(f"missing {name} {kind}")

    texture_compatibility = types.get("TextureCompatibility", {})
    if texture_compatibility.get("base_type", {}).get("name") != "uint128":
        failures.append(
            "TextureCompatibility must be an opaque uint128 distinct type"
        )

    public_type_schemas = {
        "GpuSpan": (
            (
                ("owner", "ulong"),
                ("index", "uint"),
                ("generation", "uint"),
                ("offset", "usz"),
                ("size", "usz"),
            ),
            "GpuSpan must match the exact identity/range schema",
        ),
        "GpuAllocation": (
            (
                ("owner", "ulong"),
                ("index", "uint"),
                ("generation", "uint"),
            ),
            "GpuAllocation must match the exact identity schema",
        ),
        "Sampler": (
            (
                ("owner", "ulong"),
                ("index", "uint"),
                ("generation", "uint"),
            ),
            "Sampler must match the exact identity schema",
        ),
        "AllocationDesc": (
            (
                ("size", "usz"),
                ("alignment", "usz"),
                ("memory_class", "MemoryClass"),
                ("access", "QueueRoles"),
                ("texture_requirements", "TextureRequirements[]"),
                ("debug_name", "ZString"),
            ),
            "AllocationDesc must match the exact public schema",
        ),
        "TextureRequirements": (
            (
                ("size", "usz"),
                ("alignment", "usz"),
                ("compatibility", "TextureCompatibility"),
                ("dedicated_only", "bool"),
            ),
            "TextureRequirements must match the exact public schema",
        ),
        "DedicatedTexture": (
            (
                ("texture", "TextureHandle"),
                ("allocation", "GpuAllocation"),
            ),
            "DedicatedTexture must expose exactly two ownership tokens",
        ),
        "AllocationInfo": (
            (
                ("size", "usz"),
                ("alignment", "usz"),
                ("memory_class", "MemoryClass"),
                ("access", "QueueRoles"),
                ("mapped", "bool"),
                ("coherent", "bool"),
                ("addressable", "bool"),
            ),
            "AllocationInfo must match the exact public schema",
        ),
        "BufferCopyDesc": (
            (
                ("src", "GpuSpan"),
                ("dst", "GpuSpan"),
            ),
            "BufferCopyDesc must contain exactly source and destination spans",
        ),
        "BufferTextureCopyDesc": (
            (
                ("src", "GpuSpan"),
                ("row_length_texels", "uint"),
                ("texture", "TextureHandle"),
                ("mip", "uint"),
                ("base_layer", "uint"),
                ("layer_count", "uint"),
                ("x", "uint"),
                ("y", "uint"),
                ("width", "uint"),
                ("height", "uint"),
            ),
            "BufferTextureCopyDesc must contain one source span",
        ),
        "TextureBufferCopyDesc": (
            (
                ("texture", "TextureHandle"),
                ("dst", "GpuSpan"),
                ("row_length_texels", "uint"),
                ("mip", "uint"),
                ("base_layer", "uint"),
                ("layer_count", "uint"),
                ("x", "uint"),
                ("y", "uint"),
                ("width", "uint"),
                ("height", "uint"),
            ),
            "TextureBufferCopyDesc must contain one destination span",
        ),
        "BufferBarrier": (
            (
                ("span", "GpuSpan"),
                ("before_stage", "Stage"),
                ("after_stage", "Stage"),
                ("before_hazard", "Hazard"),
                ("after_hazard", "Hazard"),
            ),
            "BufferBarrier must contain exactly one span and semantic hazards",
        ),
        "MemoryStats": (
            (
                ("heaps", "MemoryHeapBudget[16]"),
                ("heap_count", "uint"),
                ("texture_count", "ulong"),
                ("live_allocation_count", "ulong"),
            ),
            "MemoryStats must match its exact public schema",
        ),
        "MemoryClass": (
            (
                ("CPU_WRITE", "MemoryClass"),
                ("GPU_PRIVATE", "MemoryClass"),
                ("CPU_READ", "MemoryClass"),
                ("TEXTURE", "MemoryClass"),
            ),
            "MemoryClass must expose exactly the four semantic values",
        ),
        "DebugResourceKind": (
            tuple(
                (name, "DebugResourceKind")
                for name in DEBUG_RESOURCE_KINDS
            ),
            "DebugResourceKind must match its exact schema",
        ),
    }
    for name, (expected_schema, failure) in public_type_schemas.items():
        if member_schema(types.get(name)) != expected_schema:
            failures.append(failure)

    present = functions.get("present")
    if present is None:
        failures.append("missing present")
    else:
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in present.get("members", [])
        )
        if parameter_types != (
            "Device*",
            "AcquiredImage*",
            "CompletionPoint",
        ):
            failures.append(
                "present must consume AcquiredImage* with CompletionPoint"
            )

    executable = types.get("ExecutableCommandList")
    if executable is None or executable.get("kind") != "struct":
        failures.append("missing ExecutableCommandList token")

    submit_desc = types.get("SubmitDesc")
    submit_fields = {
        member.get("name"): member.get("type", {}).get("name")
        for member in (submit_desc or {}).get("members", [])
    }
    if submit_fields.get("command_lists") != "ExecutableCommandList[]":
        failures.append(
            "SubmitDesc.command_lists must contain executable tokens"
        )
    if submit_fields.get("readiness") != "SwapchainReadiness":
        failures.append(
            "SubmitDesc.readiness must contain one-shot swapchain readiness"
        )
    if "swapchain" in submit_fields:
        failures.append("SubmitDesc must not expose swapchain coupling")

    readiness = types.get("SwapchainReadiness")
    if readiness is None or readiness.get("kind") != "struct":
        failures.append("missing SwapchainReadiness token")

    acquired = types.get("AcquiredImage")
    acquired_fields = {
        member.get("name"): member.get("type", {}).get("name")
        for member in (acquired or {}).get("members", [])
    }
    if acquired_fields.get("readiness") != "SwapchainReadiness":
        failures.append("AcquiredImage must carry SwapchainReadiness")

    for module_name, handle_names in PLATFORM_HANDLE_TYPES.items():
        module = modules.get(module_name)
        if module is None:
            failures.append(f"missing {module_name}")
            continue

        surface = public_entries(module)
        definitions = {
            entry.get("name"): entry
            for entry in surface.get("types", [])
        }
        for handle_name in handle_names:
            definition = definitions.get(handle_name)
            if definition is None or definition.get("kind") != "distinct type":
                failures.append(
                    f"{module_name}::{handle_name} must be a distinct type"
                )

        create_surface = next(
            (
                entry
                for entry in surface.get("functions", [])
                if entry.get("name") == "create_surface"
            ),
            None,
        )
        if create_surface is None:
            failures.append(f"missing {module_name}::create_surface")
            continue

        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in create_surface.get("members", [])
        )
        expected_types = ("Runtime*", *handle_names)
        if parameter_types != expected_types:
            failures.append(
                f"{module_name}::create_surface must use typed platform handles"
            )

    return failures


def scan_retired_source_symbols() -> list[str]:
    failures = []
    for path in sorted((ROOT / "gpu").rglob("*.c3")):
        if "vk" in path.relative_to(ROOT / "gpu").parts:
            continue
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        for symbol in RETIRED_SOURCE_SYMBOLS:
            if symbol in source:
                failures.append(f"retired {symbol} in {relative}")
        for pattern, label in RETIRED_SOURCE_PATTERNS.items():
            if re.search(pattern, source, re.DOTALL):
                failures.append(f"retired {label} in {relative}")
    return failures


def validate_private_backend_source(relative: Path, source: str) -> list[str]:
    lines = source.lstrip("﻿").splitlines()
    if lines and lines[0].strip() == PRIVATE_BACKEND_DECLARATION:
        return []
    return [
        f"{relative.as_posix()} must declare the private gpu::vk backend module"
    ]


def scan_private_backend_modules() -> list[str]:
    failures = []
    for path in sorted((ROOT / "gpu" / "vk").rglob("*.c3")):
        failures.extend(
            validate_private_backend_source(
                path.relative_to(ROOT),
                path.read_text(encoding="utf-8"),
            )
        )
    return failures


def main() -> int:
    result = subprocess.run(
        ["c3c", "docgen", "--json"],
        cwd=CPU_PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        return result.returncode

    document = json.loads(result.stdout)
    failures = validate_document(document)
    failures.extend(scan_retired_source_symbols())
    failures.extend(scan_private_backend_modules())
    if failures:
        print("public GPU API contract violations:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("public GPU API matches the strict contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
