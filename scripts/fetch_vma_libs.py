#!/usr/bin/env python3

import argparse
import hashlib
import io
import subprocess
import urllib.request
import zipfile
from pathlib import Path


VMA_PATH = "lib/vma.c3l"
# The vma.c3l release built from the commit lib/vma.c3l pins. Update the tag,
# commit, and checksums together whenever the submodule moves.
VMA_RELEASE_TAG = "v0.2.0"
VMA_RELEASE_COMMIT = "b9962af789efd95e3d87d139ccf4d25d53ecdf42"
RELEASE_URL = f"https://github.com/fesoliveira014/vma.c3l/releases/download/{VMA_RELEASE_TAG}"

# Per platform: the packed artifact, its checksum from the release SHA256SUMS,
# and the checksum of the library inside it.
LIBRARIES = {
    "linux-x64": {
        "asset": f"vma-{VMA_RELEASE_TAG}-linux-x64.c3l",
        "asset_sha256": "09d461cfa3b064a6f3a670fc69374864799e85a198c2241c5e722f73f04547db",
        "file": "libVulkanMemoryAllocator.a",
        "sha256": "0ef10119c7848c82605637fbf9d6a6d7bef471ff625bf601232662459afbb2f6",
    },
    "windows-x64": {
        "asset": f"vma-{VMA_RELEASE_TAG}-windows-x64.c3l",
        "asset_sha256": "3cd84f562ae3f9242be9e05f8c1f95a3d0d5b40657bf1f68df33af7299136af8",
        "file": "VulkanMemoryAllocator.lib",
        "sha256": "69b867e1720816a3ac65aab9e8864ea0e5170f309561c71cb3c3ed9c3577ada0",
    },
}


def validate_vma_checkout(root: Path) -> None:
    vma_root = root / VMA_PATH
    if not (vma_root / ".git").exists():
        raise RuntimeError(
            f"submodule is not initialized: {VMA_PATH}; "
            "run git submodule update --init --recursive"
        )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=vma_root,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != VMA_RELEASE_COMMIT:
        raise RuntimeError(
            f"{VMA_PATH} is at {actual}, but vma.c3l {VMA_RELEASE_TAG} "
            f"was built from {VMA_RELEASE_COMMIT}"
        )


def install_library(root: Path, target: str) -> Path:
    library = LIBRARIES[target]
    destination = root / VMA_PATH / "linked-libs" / target / library["file"]
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == library["sha256"]:
        return destination

    url = f"{RELEASE_URL}/{library['asset']}"
    with urllib.request.urlopen(url) as response:
        artifact = response.read()
    digest = hashlib.sha256(artifact).hexdigest()
    if digest != library["asset_sha256"]:
        raise RuntimeError(f"checksum mismatch for {url}: expected {library['asset_sha256']}, found {digest}")

    with zipfile.ZipFile(io.BytesIO(artifact)) as archive:
        payload = archive.read(f"linked-libs/{target}/{library['file']}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != library["sha256"]:
        raise RuntimeError(f"checksum mismatch for {library['file']} in {url}: expected {library['sha256']}, found {digest}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f"{destination.name}.partial")
    staged.write_bytes(payload)
    staged.replace(destination)
    return destination


def main() -> None:
    argparse.ArgumentParser(
        description="Install the prebuilt VMA static libraries of the pinned vma.c3l release"
    ).parse_args()
    root = Path(__file__).resolve().parents[1]
    validate_vma_checkout(root)
    for target in sorted(LIBRARIES):
        print(install_library(root, target))


if __name__ == "__main__":
    main()
