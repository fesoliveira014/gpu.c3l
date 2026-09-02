#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path
from pathlib import PurePosixPath

if __package__:
    from scripts import package_release
else:
    import package_release


DISTRIBUTION_README = """# gpu.c3l distribution

This directory is generated from the Linux and Windows release bundles and is read-only. Do not edit its contents.

Add this distribution to a consumer checkout with:

```sh
git submodule add https://github.com/fesoliveira014/gpu.c3l-dist lib/gpu.c3l
```
"""

REQUIRED_SHARED_MEMBERS = frozenset(
    {
        "gpu.c3l/LICENSE",
        "gpu.c3l/manifest.json",
        "gpu.c3l/gpu/gpu.c3",
        "gpu.c3l/gpu/gpu.c3i",
        "gpu.c3l/docs/api/index.md",
        "gpu.c3l/lib/vk.c3l/manifest.json",
        "gpu.c3l/lib/vk.c3l/LICENSE",
        "gpu.c3l/lib/vma.c3l/manifest.json",
        "gpu.c3l/lib/vma.c3l/LICENSE",
        "gpu.c3l/lib/spvreflect.c3l/manifest.json",
        "gpu.c3l/lib/spvreflect.c3l/LICENSE",
        "gpu.c3l/lib/spvreflect.c3l/LICENSE.spirv-reflect.apache-2.0",
        "gpu.c3l/lib/spvreflect.c3l/NOTICE",
        *(f"gpu.c3l/{document}" for document in package_release.CONSUMER_DOCS),
    }
)
DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[1]


def validate_member_name(name: str) -> None:
    if not name.startswith("gpu.c3l/") or "\\" in name:
        raise RuntimeError(f"unsafe archive member: {name}")
    relative = name[len("gpu.c3l/"):]
    raw_parts = relative.split("/")
    if (
        not relative
        or any(part in {"", ".", ".."} for part in raw_parts)
        or package_release.FORBIDDEN_RELEASE_PATH_PARTS.intersection(raw_parts)
    ):
        raise RuntimeError(f"unsafe archive member: {name}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or package_release.FORBIDDEN_RELEASE_PATH_PARTS.intersection(path.parts)
    ):
        raise RuntimeError(f"unsafe archive member: {name}")


def archive_members(archive_path: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                member_type = stat.S_IFMT(info.external_attr >> 16)
                if info.is_dir() or member_type not in {0, stat.S_IFREG}:
                    raise RuntimeError(f"unsafe archive member type: {info.filename}")
                validate_member_name(info.filename)
                if info.filename in members:
                    raise RuntimeError(f"duplicate archive member: {info.filename}")
                members[info.filename] = archive.open(info).read()
    else:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    raise RuntimeError(f"unsafe archive member type: {member.name}")
                validate_member_name(member.name)
                if member.name in members:
                    raise RuntimeError(f"duplicate archive member: {member.name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"unsafe archive member type: {member.name}")
                members[member.name] = extracted.read()
    return members


def validate_provenance(linux_bundle: dict, windows_bundle: dict) -> None:
    if linux_bundle["source"] != windows_bundle["source"]:
        raise RuntimeError("source provenance mismatch")
    if linux_bundle["version"] != windows_bundle["version"]:
        raise RuntimeError("bundle version mismatch")
    if linux_bundle["components"] != windows_bundle["components"]:
        raise RuntimeError("component provenance mismatch")


def validate_bundle(bundle: dict, target: str) -> None:
    if bundle.get("schema") != 1 or bundle.get("name") != "gpu.c3l":
        raise RuntimeError("unsupported bundle manifest")
    if bundle.get("target") != target:
        raise RuntimeError(f"unexpected bundle target: {bundle.get('target')}")
    if not isinstance(bundle.get("version"), str):
        raise RuntimeError("invalid bundle manifest")
    try:
        package_release.validate_version(bundle["version"])
    except ValueError as error:
        raise RuntimeError("invalid bundle manifest") from error
    source = bundle.get("source")
    if (
        not isinstance(source, dict)
        or source.get("repository") != package_release.REPOSITORY
        or not isinstance(source.get("commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", source["commit"])
    ):
        raise RuntimeError("invalid bundle manifest")
    components = bundle.get("components")
    if not isinstance(components, list) or len(components) != len(package_release.COMPONENTS):
        raise RuntimeError("invalid bundle manifest")
    for component, expected in zip(components, package_release.COMPONENTS):
        if (
            not isinstance(component, dict)
            or component.get("name") != expected["name"]
            or component.get("repository") != expected["repository"]
            or not isinstance(component.get("commit"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", component["commit"])
        ):
            raise RuntimeError("invalid bundle manifest")


def validate_declared_native_files(members: dict[str, bytes], target: str) -> None:
    expected = {f"gpu.c3l/{path}" for path in package_release.NATIVE_FILES[target]}
    missing = expected - set(members)
    if missing:
        raise RuntimeError(f"missing declared native files: {sorted(missing)}")
    actual = {name for name in members if PurePosixPath(name).suffix in {".a", ".lib", ".dll"}}
    unexpected = actual - expected
    if unexpected:
        raise RuntimeError(f"unexpected native file: {sorted(unexpected)}")


def validate_required_members(members: dict[str, bytes]) -> None:
    missing = REQUIRED_SHARED_MEMBERS - set(members)
    if missing:
        raise RuntimeError(f"missing required release members: {sorted(missing)}")


def validate_release_inventory(members: dict[str, bytes], source_root: Path, target: str) -> None:
    expected = set(package_release.collect_release_files(source_root, target))
    expected.add("gpu.c3l/BUNDLE.json")
    actual = set(members)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise RuntimeError(
            f"archive inventory mismatch for {target}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def shared_members(members: dict[str, bytes]) -> set[str]:
    native_members = {
        f"gpu.c3l/{path}"
        for paths in package_release.NATIVE_FILES.values()
        for path in paths
    }
    return set(members) - native_members - {"gpu.c3l/BUNDLE.json"}


def validate_shared_files(linux_members: dict[str, bytes], windows_members: dict[str, bytes]) -> None:
    linux_shared = shared_members(linux_members)
    windows_shared = shared_members(windows_members)
    if linux_shared != windows_shared:
        raise RuntimeError("shared file sets differ")
    for name in linux_shared:
        if linux_members[name] != windows_members[name]:
            raise RuntimeError(f"shared file contents differ: {name}")


def write_distribution(
    staging_root: Path,
    tag: str,
    linux_archive: Path,
    windows_archive: Path,
    linux_bundle: dict,
    linux_members: dict[str, bytes],
    windows_members: dict[str, bytes],
) -> None:
    native_files = {
        f"gpu.c3l/{path}"
        for paths in package_release.NATIVE_FILES.values()
        for path in paths
    }
    for name in sorted(shared_members(linux_members) | native_files):
        relative = name.removeprefix("gpu.c3l/")
        content = linux_members.get(name, windows_members.get(name))
        if content is None:
            raise RuntimeError(f"missing distribution member: {name}")
        destination = staging_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    (staging_root / "README.md").write_text(DISTRIBUTION_README, encoding="utf-8")
    metadata = {
        "schema": 1,
        "version": linux_bundle["version"],
        "targets": ["linux-x64", "windows-x64"],
        "source": {**linux_bundle["source"], "tag": tag},
        "components": linux_bundle["components"],
        "source_archives": [
            {
                "target": target,
                "basename": archive.name,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
            for target, archive in (("linux-x64", linux_archive), ("windows-x64", windows_archive))
        ],
    }
    (staging_root / "DISTRIBUTION.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def build_distribution(
    tag: str,
    linux_archive: Path,
    windows_archive: Path,
    output_dir: Path,
    source_root: Path | None = None,
) -> None:
    if not tag.startswith("v"):
        raise ValueError(f"tag must be a v-prefixed semantic version: {tag}")
    package_release.validate_version(tag[1:])
    expected_linux = f"gpu.c3l-{tag}-linux-x64.tar.gz"
    expected_windows = f"gpu.c3l-{tag}-windows-x64.zip"
    if linux_archive.name != expected_linux:
        raise ValueError(f"unexpected linux archive name: {linux_archive.name}")
    if windows_archive.name != expected_windows:
        raise ValueError(f"unexpected windows archive name: {windows_archive.name}")
    if output_dir.exists():
        raise RuntimeError(f"output path already exists: {output_dir}")
    source_root = (source_root or DEFAULT_SOURCE_ROOT).resolve()
    linux_members = archive_members(linux_archive)
    windows_members = archive_members(windows_archive)
    linux_bundle = json.loads(linux_members["gpu.c3l/BUNDLE.json"])
    windows_bundle = json.loads(windows_members["gpu.c3l/BUNDLE.json"])
    validate_bundle(linux_bundle, "linux-x64")
    validate_bundle(windows_bundle, "windows-x64")
    validate_provenance(linux_bundle, windows_bundle)
    if tag[1:] != linux_bundle["version"]:
        raise ValueError(f"tag version does not match bundle version: {tag}")
    validate_declared_native_files(linux_members, "linux-x64")
    validate_declared_native_files(windows_members, "windows-x64")
    validate_required_members(linux_members)
    validate_required_members(windows_members)
    validate_release_inventory(linux_members, source_root, "linux-x64")
    validate_release_inventory(windows_members, source_root, "windows-x64")
    validate_shared_files(linux_members, windows_members)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        write_distribution(
            staging_root,
            tag,
            linux_archive,
            windows_archive,
            linux_bundle,
            linux_members,
            windows_members,
        )
        if output_dir.exists():
            raise RuntimeError(f"output path already exists: {output_dir}")
        staging_root.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--linux-archive", required=True, type=Path)
    parser.add_argument("--windows-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    build_distribution(args.tag, args.linux_archive, args.windows_archive, args.output_dir)


if __name__ == "__main__":
    main()
