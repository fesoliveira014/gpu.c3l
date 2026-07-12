from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Callable


SUPPORTED_TARGETS = {"linux-x64", "windows-x64"}


class RuntimeContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(package: Path) -> dict:
    path = package / "runtime.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeContractError(f"cannot read runtime manifest: {error}") from error
    if manifest.get("format") != 1:
        raise RuntimeContractError("unsupported runtime manifest format")
    return manifest


def validate_target(manifest: dict, target: str) -> None:
    if target not in SUPPORTED_TARGETS:
        raise RuntimeContractError(f"unsupported target: {target}")
    if manifest.get("target") != target:
        raise RuntimeContractError(
            f"package target {manifest.get('target')!r} does not match requested target {target!r}"
        )


def discover_library(name: str) -> bool:
    candidate = ctypes.util.find_library(name) or name
    try:
        if sys.platform == "win32":
            ctypes.WinDLL(candidate)
        else:
            ctypes.CDLL(candidate)
    except OSError:
        return False
    return True


def check(
    package: Path,
    target: str,
    discover: Callable[[str], bool] = discover_library,
) -> dict:
    manifest = load_manifest(package)
    validate_target(manifest, target)
    prerequisites = manifest.get("system-prerequisites", [])
    for prerequisite in prerequisites:
        name = prerequisite.get("name", "")
        discovery = prerequisite.get("discovery", "")
        if name == "Vulkan loader" and not discover(discovery):
            raise RuntimeContractError(
                f"Vulkan loader {discovery!r} was not found through normal system loader discovery; "
                "install a target Vulkan loader and driver"
            )
    message = "system prerequisites declared"
    if target == "windows-x64":
        message = (
            "Windows dynamic release CRT is a declared, non-authoritative prerequisite; "
            "verify the linked application's PE imports"
        )
    return {"target": target, "system-prerequisites": prerequisites, "message": message}


def validated_runtime_files(package: Path, manifest: dict) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for item in manifest.get("package-owned-runtime-files", []):
        relative = item.get("path", "")
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != relative
            or "\\" in relative
            or relative in seen
        ):
            raise RuntimeContractError(f"invalid runtime path: {relative!r}")
        seen.add(relative)
        expected = item.get("sha256", "")
        source = package.joinpath(*pure.parts)
        if len(expected) != 64 or sha256(source) != expected:
            raise RuntimeContractError(f"runtime file hash mismatch: {relative}")
        files.append((source, relative))
    return files


def stage(package: Path, destination: Path, target: str) -> dict:
    manifest = load_manifest(package)
    validate_target(manifest, target)
    files = validated_runtime_files(package, manifest)
    if files:
        destination.mkdir(parents=True, exist_ok=True)
        for source, relative in files:
            output = destination.joinpath(*PurePosixPath(relative).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
    else:
        destination.mkdir(parents=True, exist_ok=True)
    message = "no package-owned runtime files; system prerequisites remain application-owned"
    if target == "windows-x64":
        message += "; CRT status is declarative and non-authoritative"
    return {
        "target": target,
        "staged": [relative for _, relative in files],
        "system-prerequisites": manifest.get("system-prerequisites", []),
        "message": message,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or stage gpu.c3l runtime prerequisites")
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "stage"):
        command = subparsers.add_parser(name)
        command.add_argument("--target", choices=sorted(SUPPORTED_TARGETS), required=True)
        if name == "stage":
            command.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            report = check(args.package, args.target)
        else:
            report = stage(args.package, args.destination, args.target)
    except RuntimeContractError as error:
        parser.exit(1, f"runtime contract error: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
