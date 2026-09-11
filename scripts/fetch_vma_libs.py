#!/usr/bin/env python3

import argparse
import hashlib
import subprocess
import urllib.request
from pathlib import Path


VMA_PATH = "lib/vma.c3l"
# The vma.c3l release built from the commit lib/vma.c3l pins. Update the tag,
# commit, and checksums together whenever the submodule moves.
VMA_RELEASE_TAG = "v0.1.0"
VMA_RELEASE_COMMIT = "431f462a69c22cda5091e0d281a83853a94f40d7"
RELEASE_URL = f"https://github.com/fesoliveira014/vma.c3l/releases/download/{VMA_RELEASE_TAG}"

LIBRARIES = {
    "linux-x64": {
        "asset": "libVulkanMemoryAllocator-linux-x64.a",
        "file": "libVulkanMemoryAllocator.a",
        "sha256": "3d4f91af9ee7cdf212a0ac690da6b7428cc16ecda0393953f3ca2b309b549f5d",
    },
    "windows-x64": {
        "asset": "VulkanMemoryAllocator-windows-x64.lib",
        "file": "VulkanMemoryAllocator.lib",
        "sha256": "9b4322deb148b0879be608cfc66aa0ed19ec2493f98ec79929fdc0e2b1a9133a",
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
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != library["sha256"]:
        raise RuntimeError(f"checksum mismatch for {url}: expected {library['sha256']}, found {digest}")

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
