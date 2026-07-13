#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "test" / "profile_boundary"
PASS_TARGET = "same_profile"
REQUIRED_ROOT_SOURCES = {"gpu.c3", "gpu.c3i", "types.c3"}
FAIL_TARGETS = {
    "cross_device": "Device",
    "cross_handle": "BufferHandle",
    "cross_command": "CommandList",
    "cross_barrier": "GlobalBarrier",
    "cross_descriptor": "TextureIndex",
    "cross_pipeline": "PipelineHandle",
}


def normalized_source(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").strip().replace(
        "gpu::compat",
        "gpu",
    )


def validate_layout(root: Path) -> list[str]:
    gpu = root / "gpu"
    compat = gpu / "compat"
    root_sources = sorted(
        path
        for path in gpu.iterdir()
        if path.is_file() and path.suffix in {".c3", ".c3i"}
    )
    actual_sources = {path.name for path in root_sources}
    failures = []

    missing = sorted(REQUIRED_ROOT_SOURCES - actual_sources)
    if missing:
        failures.append(f"missing root sources: {', '.join(missing)}")

    for path in root_sources:
        text = path.read_text(encoding="utf-8")
        if "gpu::compat" in text:
            failures.append(f"strict source references gpu::compat: {path.name}")

        if path.name in {"gpu.c3", "gpu.c3i"}:
            continue
        compat_path = compat / path.name
        if (
            compat_path.is_file()
            and normalized_source(path) == normalized_source(compat_path)
        ):
            failures.append(
                f"root source duplicates compatibility implementation: {path.name}"
            )

    strict_vk = gpu / "vk"
    compat_vk = compat / "vk"
    if strict_vk.is_dir():
        for path in sorted(strict_vk.rglob("*.c3")):
            relative = path.relative_to(strict_vk)
            if "gpu::compat" in path.read_text(encoding="utf-8"):
                failures.append(
                    f"strict source references gpu::compat: vk/{relative.as_posix()}"
                )

            compat_path = compat_vk / relative
            if (
                compat_path.is_file()
                and normalized_source(path) == normalized_source(compat_path)
            ):
                failures.append(
                    "strict backend duplicates compatibility implementation: "
                    f"vk/{relative.as_posix()}"
                )

    return failures


def compile_target(target: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["c3c", "build", target, "--path", str(PROJECT), "-C"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    layout_failures = validate_layout(ROOT)
    if layout_failures:
        for failure in layout_failures:
            print(failure, file=sys.stderr)
        return 1

    same_profile = compile_target(PASS_TARGET)
    if same_profile.returncode != 0:
        sys.stderr.write(same_profile.stderr or same_profile.stdout)
        return 1

    failures = []
    for target, type_name in FAIL_TARGETS.items():
        result = compile_target(target)
        output = result.stderr + result.stdout
        if result.returncode == 0:
            failures.append(f"{target}: cross-profile conversion compiled")
        elif "Error:" not in output or type_name not in output:
            failures.append(f"{target}: failed for an unexpected reason")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print("profile types are compile-time distinct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
