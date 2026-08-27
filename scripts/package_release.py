#!/usr/bin/env python3

import argparse
import gzip
import json
import posixpath
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


REPOSITORY = "https://github.com/fesoliveira014/gpu.c3l"
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")
LINK_PATTERN = re.compile(r"\[[^]]+\]\(([^)]+)\)")

COMPONENTS = (
    {
        "name": "vk.c3l",
        "path": "lib/vk.c3l",
        "repository": "https://github.com/fesoliveira014/vk.c3l",
    },
    {
        "name": "vma.c3l",
        "path": "lib/vma.c3l",
        "repository": "https://github.com/fesoliveira014/vma.c3l",
    },
    {
        "name": "spvreflect.c3l",
        "path": "lib/spvreflect.c3l",
        "repository": "https://github.com/fesoliveira014/spvreflect.c3l",
    },
)

CONSUMER_DOCS = (
    "docs/index.md",
    "docs/getting_started.md",
    "docs/architecture.md",
    "docs/features_and_limitations.md",
    "docs/shader_abi.md",
    "docs/cookbook.md",
)

NATIVE_FILES = {
    "linux-x64": (
        "lib/vma.c3l/linked-libs/linux-x64/libVulkanMemoryAllocator.a",
        "lib/spvreflect.c3l/linux/libspvreflect.a",
    ),
    "windows-x64": (
        "lib/vk.c3l/windows/vulkan-1.lib",
        "lib/vma.c3l/linked-libs/windows-x64/VulkanMemoryAllocator.lib",
        "lib/spvreflect.c3l/windows/spvreflect.lib",
    ),
}


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_version(version: str) -> None:
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"version must be a semantic version without a v prefix: {version}")


def expected_component_commit(root: Path, component_path: str) -> str:
    entry = run_git(root, "ls-tree", "HEAD", "--", component_path)
    fields = entry.split()
    if len(fields) < 3 or fields[1] != "commit":
        raise RuntimeError(f"missing gitlink for {component_path}")
    return fields[2]


def component_metadata(root: Path) -> list[dict[str, str]]:
    metadata = []
    for component in COMPONENTS:
        component_root = root / component["path"]
        expected = expected_component_commit(root, component["path"])
        if not component_root.is_dir():
            raise RuntimeError(f"submodule is not initialized: {component['path']}")
        actual = run_git(component_root, "rev-parse", "HEAD")
        if actual != expected:
            raise RuntimeError(
                f"submodule commit mismatch for {component['path']}: "
                f"expected {expected}, found {actual}"
            )
        dirty = run_git(component_root, "status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise RuntimeError(f"submodule has tracked changes: {component['path']}")
        metadata.append(
            {
                "name": component["name"],
                "repository": component["repository"],
                "commit": expected,
            }
        )
    return metadata


def validate_vma_dependency_boundary(root: Path) -> None:
    vma_root = root / "lib/vma.c3l"
    sdl_entry = run_git(vma_root, "ls-files", "--stage", "--", "test/libs/sdl3.c3l")
    if sdl_entry:
        raise RuntimeError("vma.c3l still declares the SDL3 test submodule")

    prefix = "submodule.test/libs/vk.c3l"
    update = run_git(vma_root, "config", "-f", ".gitmodules", "--get", f"{prefix}.update")
    shallow = run_git(vma_root, "config", "-f", ".gitmodules", "--get", f"{prefix}.shallow")
    if update != "none" or shallow != "true":
        raise RuntimeError("vma.c3l Vulkan test binding must be opt-in and shallow")


def validate_root_dependency_graph(root: Path) -> None:
    entries = run_git(
        root,
        "config",
        "-f",
        ".gitmodules",
        "--get-regexp",
        r"^submodule\..*\.path$",
    )
    actual = {line.split(maxsplit=1)[1] for line in entries.splitlines()}
    expected = {component["path"] for component in COMPONENTS}
    if actual != expected:
        raise RuntimeError(
            "root submodules must be exactly the runtime bindings: "
            f"expected {sorted(expected)}, found {sorted(actual)}"
        )


def add_file(files: dict[str, Path], root: Path, relative: str) -> None:
    source = root / relative
    if not source.is_file():
        raise RuntimeError(f"required release file is missing: {relative}")
    files[f"gpu.c3l/{relative}"] = source


def collect_release_files(root: Path, target: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for relative in ("LICENSE", "README.md", "manifest.json", *CONSUMER_DOCS):
        add_file(files, root, relative)

    for source in sorted((root / "gpu").rglob("*")):
        if source.is_file():
            relative = source.relative_to(root).as_posix()
            files[f"gpu.c3l/{relative}"] = source

    for source in sorted((root / "docs/api").rglob("*")):
        if source.is_file():
            relative = source.relative_to(root).as_posix()
            files[f"gpu.c3l/{relative}"] = source

    for component in COMPONENTS:
        component_root = root / component["path"]
        for name in ("manifest.json", "LICENSE", "LICENSE.spirv-reflect.apache-2.0", "NOTICE"):
            source = component_root / name
            if source.is_file():
                destination = f"gpu.c3l/{component['path']}/{name}"
                files[destination] = source
        for pattern in ("*.c3", "*.c3i"):
            for source in sorted(component_root.glob(pattern)):
                destination = f"gpu.c3l/{component['path']}/{source.name}"
                files[destination] = source

    for relative in NATIVE_FILES[target]:
        add_file(files, root, relative)
    return files


def write_bundle_metadata(
    root: Path,
    staging_root: Path,
    version: str,
    target: str,
    components: list[dict[str, str]],
) -> None:
    bundle = {
        "schema": 1,
        "name": "gpu.c3l",
        "version": version,
        "target": target,
        "source": {
            "repository": REPOSITORY,
            "commit": run_git(root, "rev-parse", "HEAD"),
        },
        "components": components,
    }
    destination = staging_root / "gpu.c3l/BUNDLE.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")


def stage_files(files: dict[str, Path], staging_root: Path) -> None:
    for destination_name, source in sorted(files.items()):
        destination = staging_root / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def write_tar_gz(staging_root: Path, archive_path: Path) -> None:
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for source in sorted(path for path in staging_root.rglob("*") if path.is_file()):
                    archive.add(
                        source,
                        arcname=source.relative_to(staging_root).as_posix(),
                        recursive=False,
                        filter=normalized_tar_info,
                    )


def write_zip(staging_root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(path for path in staging_root.rglob("*") if path.is_file()):
            name = source.relative_to(staging_root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compresslevel=9)


def read_archive_member(archive_path: Path, member: str) -> str:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            return archive.read(member).decode("utf-8")
    with tarfile.open(archive_path, "r:gz") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"archive member is not a file: {member}")
        return extracted.read().decode("utf-8")


def validate_consumer_doc_links(members: set[str], archive_path: Path) -> None:
    for member in sorted(name for name in members if name.endswith(".md")):
        content = read_archive_member(archive_path, member)
        for match in LINK_PATTERN.finditer(content):
            target = match.group(1).strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = posixpath.normpath(
                str(PurePosixPath(member).parent / PurePosixPath(target))
            )
            if resolved not in members:
                raise RuntimeError(f"broken consumer documentation link: {member} -> {target}")


def create_release(root: Path, version: str, target: str, output_dir: Path) -> Path:
    root = root.resolve()
    validate_version(version)
    if target not in NATIVE_FILES:
        raise ValueError(f"unsupported target: {target}")

    components = component_metadata(root)
    validate_root_dependency_graph(root)
    validate_vma_dependency_boundary(root)
    files = collect_release_files(root, target)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".tar.gz" if target == "linux-x64" else ".zip"
    archive_path = output_dir / f"gpu.c3l-v{version}-{target}{suffix}"
    with tempfile.TemporaryDirectory(prefix="gpu-c3l-release-", dir=output_dir) as staging_dir:
        staging_root = Path(staging_dir)
        stage_files(files, staging_root)
        write_bundle_metadata(root, staging_root, version, target, components)
        if target == "linux-x64":
            write_tar_gz(staging_root, archive_path)
        else:
            write_zip(staging_root, archive_path)

    if target == "linux-x64":
        with tarfile.open(archive_path, "r:gz") as archive:
            members = {member.name for member in archive.getmembers() if member.isfile()}
    else:
        with zipfile.ZipFile(archive_path) as archive:
            members = {name for name in archive.namelist() if not name.endswith("/")}
    validate_consumer_doc_links(members, archive_path)
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a runtime-only gpu.c3l release archive")
    parser.add_argument("--version", required=True)
    parser.add_argument("--target", required=True, choices=sorted(NATIVE_FILES))
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    archive = create_release(root, args.version, args.target, args.output_dir)
    print(archive)


if __name__ == "__main__":
    main()
