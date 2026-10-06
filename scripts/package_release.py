#!/usr/bin/env python3

import argparse
import hashlib
import json
import posixpath
import re
import subprocess
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
    "docs/concepts.md",
    "docs/getting_started.md",
    "docs/architecture.md",
    "docs/features_and_limitations.md",
    "docs/shader_abi.md",
    "docs/cookbook.md",
)

FORBIDDEN_RELEASE_PATH_PARTS = frozenset(
    {
        ".git",
        ".github",
        ".gitmodules",
        "test",
        "tests",
        "examples",
        "scripts",
        "openspec",
        "contributing",
        "sdl3.c3l",
    }
)


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


def exact_tag(root: Path) -> str:
    result = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


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
        entry = {
            "name": component["name"],
            "repository": component["repository"],
            "commit": expected,
        }
        tag = exact_tag(component_root)
        if tag:
            entry["tag"] = tag
        metadata.append(entry)
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
    files[relative] = source


def collect_release_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for relative in ("LICENSE", "README.md", "manifest.json", *CONSUMER_DOCS):
        add_file(files, root, relative)

    for source in sorted((root / "gpu").rglob("*")):
        if source.is_file():
            files[source.relative_to(root).as_posix()] = source

    for directory in ("docs/api", "docs/util", "include/shaders", "tools/gpu_shaders/src"):
        for source in sorted((root / directory).rglob("*")):
            if source.is_file() and source.name != ".gitkeep":
                files[source.relative_to(root).as_posix()] = source
    add_file(files, root, "tools/gpu_shaders/project.json")
    return files


def bundle_metadata(root: Path, version: str, components: list[dict[str, str]]) -> bytes:
    bundle = {
        "schema": 2,
        "name": "gpu.c3l",
        "version": version,
        "source": {
            "repository": REPOSITORY,
            "commit": run_git(root, "rev-parse", "HEAD"),
        },
        "components": components,
    }
    return (json.dumps(bundle, indent=2) + "\n").encode("utf-8")


def write_packed_c3l(entries: dict[str, bytes], archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, entries[name], compresslevel=9)


def write_checksums(output_dir: Path, artifacts: list[Path]) -> Path:
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(artifacts)
    ]
    checksums = output_dir / "SHA256SUMS"
    checksums.write_text("".join(lines), encoding="utf-8", newline="\n")
    return checksums


def validate_consumer_doc_links(members: set[str], archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(name for name in members if name.endswith(".md")):
            content = archive.read(member).decode("utf-8")
            for match in LINK_PATTERN.finditer(content):
                target = match.group(1).strip().split("#", 1)[0]
                if not target or target.startswith(("http://", "https://", "mailto:")):
                    continue
                resolved = posixpath.normpath(
                    str(PurePosixPath(member).parent / PurePosixPath(target))
                )
                if resolved not in members:
                    raise RuntimeError(f"broken consumer documentation link: {member} -> {target}")


def create_release(root: Path, version: str, output_dir: Path) -> Path:
    root = root.resolve()
    validate_version(version)

    components = component_metadata(root)
    validate_root_dependency_graph(root)
    validate_vma_dependency_boundary(root)
    files = collect_release_files(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = {name: source.read_bytes() for name, source in files.items()}
    entries["BUNDLE.json"] = bundle_metadata(root, version, components)

    archive_path = output_dir / f"gpu-v{version}.c3l"
    write_packed_c3l(entries, archive_path)
    write_checksums(output_dir, [archive_path])
    validate_consumer_doc_links(set(entries), archive_path)
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the packed gpu .c3l release artifact")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    archive = create_release(root, args.version, args.output_dir)
    print(archive)


if __name__ == "__main__":
    main()
