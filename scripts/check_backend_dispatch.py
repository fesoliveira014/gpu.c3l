#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DISPATCH_REFERENCES = (
    "vk::extensions",
    "vk::load_extensions",
    "vk::try_set_debug_utils_object_name_ext",
    "vk::get_descriptor_set_layout_size_ext",
    "vk::get_descriptor_set_layout_binding_offset_ext",
    "vk::get_descriptor_ext",
    "vk::cmd_bind_descriptor_buffers_ext",
    "vk::cmd_set_descriptor_buffer_offsets_ext",
)


def find_global_dispatch_references(sources: dict[str, str]) -> list[str]:
    failures = []
    for name, source in sorted(sources.items()):
        for line_number, line in enumerate(source.splitlines(), start=1):
            for token in FORBIDDEN_DISPATCH_REFERENCES:
                if token in line:
                    failures.append(f"{name}:{line_number}: {token}")
    return failures


def scan_backend_sources() -> list[str]:
    sources = {
        str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
        for path in (ROOT / "gpu" / "vk").glob("*.c3")
    }
    return find_global_dispatch_references(sources)


def main() -> int:
    failures = scan_backend_sources()
    if failures:
        print("global Vulkan dispatch references:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Vulkan backend dispatch is runtime- or device-owned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
