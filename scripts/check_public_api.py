#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_PROJECT = ROOT / "test" / "cpu"

FORBIDDEN_TEXT = {
    "backend_state": "backend state pointer",
    "backendvtable": "backend dispatch table",
    "probe_vulkan_version": "Vulkan loader probe",
    "probe_vma_allocator": "VMA probe",
    "range_end": "readback retirement range",
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
}

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
    public_module = document.get("modules", {}).get("gpu")
    if public_module is None:
        print("docgen did not emit module gpu", file=sys.stderr)
        return 1

    public_surface = {
        section: (
            [entry for entry in contents if entry.get("visibility") != "private"]
            if isinstance(contents, list) else contents
        )
        for section, contents in public_module.items()
    }
    encoded = json.dumps(public_surface, separators=(",", ":"))
    lowered = encoded.lower()
    failures = [label for token, label in FORBIDDEN_TEXT.items() if token in lowered]
    failures.extend(symbol for symbol in FORBIDDEN_SYMBOLS if f'"{symbol}"' in encoded)
    if failures:
        print("public gpu documentation leaks internal symbols:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("public gpu documentation is backend-neutral")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
