#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_PROJECT = ROOT / "test" / "cpu"

FORBIDDEN_TEXT = {
    "backend_state": "backend state pointer",
    "backendvtable": "backend dispatch table",
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

RETIRED_SOURCE_SYMBOLS = (
    "PlatformKind",
    "PresentDesc",
    "SurfaceDesc",
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

    allocation_functions = {
        "allocate_memory": (
            ("Device*", "AllocationDesc*"),
            "GpuAllocation?",
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
    }
    for name, contract in allocation_functions.items():
        expected_parameters, expected_return = contract
        function = functions.get(name)
        if function is None:
            failures.append(f"missing {name}")
            continue
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in function.get("members", [])
        )
        if parameter_types != expected_parameters:
            failures.append(f"{name} has the wrong parameters")
        if function.get("return_type", {}).get("name") != expected_return:
            failures.append(f"{name} has the wrong return type")

    for name, kind in (
        ("GpuAllocation", "struct"),
        ("GpuSpan", "struct"),
        ("MemoryClass", "enum"),
        ("AllocationDesc", "struct"),
        ("AllocationInfo", "struct"),
    ):
        definition = types.get(name)
        if definition is None or definition.get("kind") != kind:
            failures.append(f"missing {name} {kind}")

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
        "AllocationDesc": (
            (
                ("size", "usz"),
                ("alignment", "usz"),
                ("memory_class", "MemoryClass"),
                ("access", "QueueRoles"),
                ("debug_name", "ZString"),
            ),
            "AllocationDesc must match the exact public schema",
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
        "MemoryClass": (
            (
                ("CPU_WRITE", "MemoryClass"),
                ("GPU_PRIVATE", "MemoryClass"),
                ("CPU_READ", "MemoryClass"),
            ),
            "MemoryClass must expose exactly the three semantic values",
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
    if failures:
        print("public GPU API contract violations:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("public GPU API matches the strict contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
