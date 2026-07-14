#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LITERAL_DISPATCH_REFERENCES = (
    "vk::extensions",
    "vk::load_extensions",
)
FUNCTION_DECLARATION = re.compile(
    r"^fn\s+[^\r\n(]*?\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)


def generated_singleton_wrappers(source: str) -> set[str]:
    declarations = list(FUNCTION_DECLARATION.finditer(source))
    wrappers = set()
    for index, declaration in enumerate(declarations):
        end = declarations[index + 1].start() if index + 1 < len(declarations) else len(source)
        if "extensions." in source[declaration.start():end]:
            wrappers.add(declaration.group(1))
    return wrappers


def load_backend_sources(root: Path) -> dict[str, str]:
    backend = root / "gpu" / "vk"
    paths = sorted(backend.rglob("*.c3"), key=lambda path: path.as_posix())
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in paths
    }


def find_global_dispatch_references(
    sources: dict[str, str],
    generated_wrappers: set[str],
) -> list[str]:
    wrapper_patterns = [
        (
            f"vk::{name}",
            re.compile(rf"(?<![A-Za-z0-9_])vk::{re.escape(name)}(?![A-Za-z0-9_])"),
        )
        for name in sorted(generated_wrappers)
    ]
    failures = []
    for name, source in sorted(sources.items()):
        for line_number, line in enumerate(source.splitlines(), start=1):
            for token in LITERAL_DISPATCH_REFERENCES:
                if token in line:
                    failures.append(f"{name}:{line_number}: {token}")
            for token, pattern in wrapper_patterns:
                if pattern.search(line):
                    failures.append(f"{name}:{line_number}: {token}")
    return failures


def scan_backend_sources(root: Path = ROOT) -> list[str]:
    binding = (root / "lib" / "vk.c3l" / "commands.c3").read_text(encoding="utf-8")
    wrappers = generated_singleton_wrappers(binding)
    return find_global_dispatch_references(load_backend_sources(root), wrappers)


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
