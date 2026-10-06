#!/usr/bin/env python3

import argparse
import hashlib
import io
import subprocess
import urllib.request
import zipfile
from pathlib import Path


SPVREFLECT_PATH = "lib/spvreflect.c3l"
# The spvreflect.c3l release built from the commit lib/spvreflect.c3l pins.
# Update the tag, commit, and checksums together whenever the submodule moves.
SPVREFLECT_RELEASE_TAG = "v0.1.0"
SPVREFLECT_RELEASE_COMMIT = "dae8224602f547628a7b1f4ac0632e97582c92ee"
RELEASE_URL = (
    "https://github.com/fesoliveira014/spvreflect.c3l/releases/download/"
    f"{SPVREFLECT_RELEASE_TAG}"
)

# The Windows import library is not committed; Linux keeps linux/libspvreflect.a in git.
LIBRARY = {
    "asset": f"spvreflect-{SPVREFLECT_RELEASE_TAG}-windows-x64.c3l",
    "asset_sha256": "a1db1eae44710d6f41f5dc7ef7028c60e141933fe9996f7d216838f2b0a7f03f",
    "member": "windows/spvreflect.lib",
    "sha256": "f91c0050fda4019f9de1b95623cd472bcf1f560a77c0ec1092025d9dd830c171",
}


def validate_checkout(root: Path) -> None:
    checkout = root / SPVREFLECT_PATH
    if not (checkout / ".git").exists():
        raise RuntimeError(
            f"submodule is not initialized: {SPVREFLECT_PATH}; "
            "run git submodule update --init --recursive"
        )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != SPVREFLECT_RELEASE_COMMIT:
        raise RuntimeError(
            f"{SPVREFLECT_PATH} is at {actual}, but spvreflect.c3l {SPVREFLECT_RELEASE_TAG} "
            f"was built from {SPVREFLECT_RELEASE_COMMIT}"
        )


def install_library(root: Path) -> Path:
    destination = root / SPVREFLECT_PATH / LIBRARY["member"]
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == LIBRARY["sha256"]:
        return destination

    url = f"{RELEASE_URL}/{LIBRARY['asset']}"
    with urllib.request.urlopen(url) as response:
        artifact = response.read()
    digest = hashlib.sha256(artifact).hexdigest()
    if digest != LIBRARY["asset_sha256"]:
        raise RuntimeError(f"checksum mismatch for {url}: expected {LIBRARY['asset_sha256']}, found {digest}")

    with zipfile.ZipFile(io.BytesIO(artifact)) as archive:
        payload = archive.read(LIBRARY["member"])
    digest = hashlib.sha256(payload).hexdigest()
    if digest != LIBRARY["sha256"]:
        raise RuntimeError(f"checksum mismatch for {LIBRARY['member']} in {url}: expected {LIBRARY['sha256']}, found {digest}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f"{destination.name}.partial")
    staged.write_bytes(payload)
    staged.replace(destination)
    return destination


def main() -> None:
    argparse.ArgumentParser(
        description="Install the prebuilt Windows spvreflect library of the pinned spvreflect.c3l release"
    ).parse_args()
    root = Path(__file__).resolve().parents[1]
    validate_checkout(root)
    print(install_library(root))


if __name__ == "__main__":
    main()
