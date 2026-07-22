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
    r"^fn\s+[^\r\n(]*?\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*\(",
    re.MULTILINE,
)
BUILD_METHOD_REFERENCE = re.compile(r"\.build\s*\(")


def top_level_function_blocks(source: str) -> list[tuple[str, str]]:
    declarations = list(FUNCTION_DECLARATION.finditer(source))
    blocks = []
    for index, declaration in enumerate(declarations):
        end = declarations[index + 1].start() if index + 1 < len(declarations) else len(source)
        blocks.append((declaration.group(1), source[declaration.start():end]))
    return blocks


def generated_singleton_wrappers(source: str) -> set[str]:
    return {
        name
        for name, block in top_level_function_blocks(source)
        if "." not in name and "extensions." in block
    }


def build_methods_use_global_dispatch(
    source: str,
    generated_wrappers: set[str],
) -> bool:
    wrapper_calls = [
        re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(")
        for name in sorted(generated_wrappers)
    ]
    return any(
        name.endswith(".build")
        and (
            "extensions." in block
            or any(pattern.search(block) for pattern in wrapper_calls)
        )
        for name, block in top_level_function_blocks(source)
    )


def load_backend_sources(root: Path) -> dict[str, str]:
    backend = root / "gpu" / "internal" / "vk"
    paths = sorted(backend.rglob("*.c3"), key=lambda path: path.as_posix())
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in paths
    }


def find_global_dispatch_references(
    sources: dict[str, str],
    generated_wrappers: set[str],
    reject_build_methods: bool = False,
) -> list[str]:
    wrapper_patterns = [
        (
            f"vk::{name}",
            re.compile(rf"(?<![A-Za-z0-9_])vk::{re.escape(name)}(?![A-Za-z0-9_])"),
        )
        for name in sorted(generated_wrappers)
        if f"vk::{name}" not in LITERAL_DISPATCH_REFERENCES
    ]
    failures = []
    for name, source in sorted(sources.items()):
        for line_number, line in enumerate(source.splitlines(), start=1):
            for token in LITERAL_DISPATCH_REFERENCES:
                if token in line:
                    failures.append(f"{name}:{line_number}: {token}")
            if reject_build_methods and BUILD_METHOD_REFERENCE.search(line):
                failures.append(f"{name}:{line_number}: .build(")
            for token, pattern in wrapper_patterns:
                if pattern.search(line):
                    failures.append(f"{name}:{line_number}: {token}")
    return failures


def scan_backend_sources(root: Path = ROOT) -> list[str]:
    binding_path = root / "lib" / "vk.c3l" / "commands.c3"
    binding = binding_path.read_text(encoding="utf-8")
    wrappers = generated_singleton_wrappers(binding)
    if not wrappers:
        raise RuntimeError(f"{binding_path}: no generated singleton-backed wrappers found")
    return find_global_dispatch_references(
        load_backend_sources(root),
        wrappers,
        reject_build_methods=build_methods_use_global_dispatch(binding, wrappers),
    )


def main() -> int:
    try:
        failures = scan_backend_sources()
    except RuntimeError as error:
        print(f"backend dispatch check failed: {error}", file=sys.stderr)
        return 1
    if failures:
        print("global Vulkan dispatch references:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Vulkan backend dispatch is runtime- or device-owned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
