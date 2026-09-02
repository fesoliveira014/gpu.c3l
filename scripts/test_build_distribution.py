import hashlib
import io
import json
import stat
import subprocess
import tarfile
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

from scripts import build_distribution
from scripts import package_release


COMMIT = "a" * 40
COMPONENT_COMMITS = ("b" * 40, "c" * 40, "d" * 40)
ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/fesoliveira014/gpu.c3l"
COMPONENTS = (
    ("vk.c3l", "https://github.com/fesoliveira014/vk.c3l"),
    ("vma.c3l", "https://github.com/fesoliveira014/vma.c3l"),
    ("spvreflect.c3l", "https://github.com/fesoliveira014/spvreflect.c3l"),
)
LINUX_NATIVE_FILES = (
    "lib/vma.c3l/linked-libs/linux-x64/libVulkanMemoryAllocator.a",
    "lib/spvreflect.c3l/linux/libspvreflect.a",
)
WINDOWS_NATIVE_FILES = (
    "lib/vk.c3l/windows/vulkan-1.lib",
    "lib/vma.c3l/linked-libs/windows-x64/VulkanMemoryAllocator.lib",
    "lib/spvreflect.c3l/windows/spvreflect.lib",
)
CONSUMER_DOCS = (
    "docs/index.md",
    "docs/getting_started.md",
    "docs/architecture.md",
    "docs/features_and_limitations.md",
    "docs/shader_abi.md",
    "docs/cookbook.md",
)
CONSUMER_README = """# gpu.c3l distribution

This directory is generated from the Linux and Windows release bundles and is read-only. Do not edit its contents.

Add this distribution to a consumer checkout with:

```sh
git submodule add https://github.com/fesoliveira014/gpu.c3l-dist lib/gpu.c3l
```
"""


class BuildDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_directory = tempfile.TemporaryDirectory()
        self.source_root = Path(self.source_directory.name)
        source_files = {
            "LICENSE": "license\n",
            "README.md": "source release readme\n",
            "manifest.json": "{}\n",
            "gpu/gpu.c3": "module gpu;\n",
            "gpu/gpu.c3i": "module gpu;\n",
            "gpu/extra_gpu.c3": "module gpu;\n",
            "docs/api/index.md": "# API\n",
            "lib/vk.c3l/manifest.json": "{}\n",
            "lib/vk.c3l/LICENSE": "license\n",
            "lib/vk.c3l/vk.c3": "module vk;\n",
            "lib/vk.c3l/vk.c3i": "module vk;\n",
            "lib/vma.c3l/manifest.json": "{}\n",
            "lib/vma.c3l/LICENSE": "license\n",
            "lib/vma.c3l/vma.c3": "module vma;\n",
            "lib/vma.c3l/vma.c3i": "module vma;\n",
            "lib/spvreflect.c3l/manifest.json": "{}\n",
            "lib/spvreflect.c3l/LICENSE": "license\n",
            "lib/spvreflect.c3l/LICENSE.spirv-reflect.apache-2.0": "license\n",
            "lib/spvreflect.c3l/NOTICE": "notice\n",
            "lib/spvreflect.c3l/spvreflect.c3": "module spvreflect;\n",
            "lib/spvreflect.c3l/spvreflect.c3i": "module spvreflect;\n",
        }
        source_files.update({path: "# document\n" for path in CONSUMER_DOCS})
        source_files.update({path: "native\n" for path in (*LINUX_NATIVE_FILES, *WINDOWS_NATIVE_FILES)})
        for relative, content in source_files.items():
            destination = self.source_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)
        self.source_root_patch = mock.patch.object(
            build_distribution, "DEFAULT_SOURCE_ROOT", self.source_root
        )
        self.source_root_patch.start()

    def tearDown(self) -> None:
        self.source_root_patch.stop()
        self.source_directory.cleanup()

    def bundle(self, target: str, *, version: str = "1.2.3", commit: str = COMMIT) -> bytes:
        return (json.dumps({
            "schema": 1,
            "name": "gpu.c3l",
            "version": version,
            "target": target,
            "source": {
                "repository": REPOSITORY,
                "commit": commit,
            },
            "components": [
                {
                    "name": name,
                    "repository": repository,
                    "commit": component_commit,
                }
                for (name, repository), component_commit in zip(COMPONENTS, COMPONENT_COMMITS)
            ],
        }, indent=2) + "\n").encode()

    def members(self, target: str, **overrides: bytes) -> dict[str, bytes]:
        members = {
            name: source.read_bytes()
            for name, source in package_release.collect_release_files(self.source_root, target).items()
        }
        members["gpu.c3l/BUNDLE.json"] = self.bundle(target)
        members.update(overrides)
        return members

    def write_tar(self, path: Path, members: dict[str, bytes]) -> None:
        with tarfile.open(path, "w:gz") as archive:
            for name, content in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, fileobj=io.BytesIO(content))

    def write_zip(self, path: Path, members: dict[str, bytes]) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)

    def fixtures(self, directory: Path) -> tuple[Path, Path]:
        return self.fixture_members(directory, self.members("linux-x64"), self.members("windows-x64"))

    def fixture_members(
        self, directory: Path, linux_members: dict[str, bytes], windows_members: dict[str, bytes]
    ) -> tuple[Path, Path]:
        linux = directory / "gpu.c3l-v1.2.3-linux-x64.tar.gz"
        windows = directory / "gpu.c3l-v1.2.3-windows-x64.zip"
        self.write_tar(linux, linux_members)
        self.write_zip(windows, windows_members)
        return linux, windows

    def test_builds_combined_tree_with_distribution_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux, windows = self.fixtures(directory)
            output = directory / "combined"

            build_distribution.build_distribution(
                tag="v1.2.3",
                linux_archive=linux,
                windows_archive=windows,
                output_dir=output,
            )

            self.assertEqual(
                set(LINUX_NATIVE_FILES) | set(WINDOWS_NATIVE_FILES),
                {
                    path.relative_to(output).as_posix()
                    for path in output.rglob("*")
                    if path.is_file() and path.suffix in {".a", ".lib", ".dll"}
                },
            )
            self.assertFalse((output / "BUNDLE.json").exists())
            self.assertEqual(CONSUMER_README, (output / "README.md").read_text())
            metadata = json.loads((output / "DISTRIBUTION.json").read_text())
            self.assertEqual(
                {"schema", "version", "targets", "source", "components", "source_archives"},
                set(metadata),
            )
            self.assertEqual(1, metadata["schema"])
            self.assertEqual("1.2.3", metadata["version"])
            self.assertEqual(["linux-x64", "windows-x64"], metadata["targets"])
            self.assertEqual(REPOSITORY, metadata["source"]["repository"])
            self.assertEqual("v1.2.3", metadata["source"]["tag"])
            self.assertEqual(COMMIT, metadata["source"]["commit"])
            self.assertEqual(
                [linux.name, windows.name],
                [archive["basename"] for archive in metadata["source_archives"]],
            )
            self.assertEqual(
                ["linux-x64", "windows-x64"],
                [archive["target"] for archive in metadata["source_archives"]],
            )
            self.assertTrue(
                all({"target", "basename", "sha256"} == set(archive) for archive in metadata["source_archives"])
            )
            self.assertEqual(
                [
                    hashlib.sha256(linux.read_bytes()).hexdigest(),
                    hashlib.sha256(windows.read_bytes()).hexdigest(),
                ],
                [archive["sha256"] for archive in metadata["source_archives"]],
            )
            self.assertEqual(
                [COMPONENT_COMMITS[0], COMPONENT_COMMITS[1], COMPONENT_COMMITS[2]],
                [component["commit"] for component in metadata["components"]],
            )

    def test_rejects_mismatched_bundle_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            windows_members = self.members("windows-x64")
            bundle = json.loads(windows_members["gpu.c3l/BUNDLE.json"])
            bundle["source"]["commit"] = "e" * 40
            windows_members["gpu.c3l/BUNDLE.json"] = (json.dumps(bundle) + "\n").encode()
            linux, windows = self.fixture_members(directory, self.members("linux-x64"), windows_members)

            with self.assertRaisesRegex(RuntimeError, "source provenance mismatch"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_rejects_mismatched_bundle_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            windows_members = self.members("windows-x64")
            bundle = json.loads(windows_members["gpu.c3l/BUNDLE.json"])
            bundle["version"] = "1.2.4"
            windows_members["gpu.c3l/BUNDLE.json"] = (json.dumps(bundle) + "\n").encode()
            linux, windows = self.fixture_members(directory, self.members("linux-x64"), windows_members)

            with self.assertRaisesRegex(RuntimeError, "bundle version mismatch"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_requires_v_tag_to_match_bundle_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux, windows = self.fixtures(directory)
            linux = linux.rename(directory / "gpu.c3l-v1.2.4-linux-x64.tar.gz")
            windows = windows.rename(directory / "gpu.c3l-v1.2.4-windows-x64.zip")

            with self.assertRaisesRegex(ValueError, "tag version does not match"):
                build_distribution.build_distribution("v1.2.4", linux, windows, directory / "combined")

    def test_requires_canonical_tagged_input_asset_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux, windows = self.fixtures(directory)
            renamed_linux = directory / "linux.tar.gz"
            linux.rename(renamed_linux)

            with self.assertRaisesRegex(ValueError, "unexpected linux archive name"):
                build_distribution.build_distribution("v1.2.3", renamed_linux, windows, directory / "combined")

    def test_rejects_swapped_input_asset_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux, windows = self.fixtures(directory)

            with self.assertRaisesRegex(ValueError, "unexpected linux archive name"):
                build_distribution.build_distribution("v1.2.3", windows, linux, directory / "combined")

    def test_rejects_mismatched_component_commits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            windows_members = self.members("windows-x64")
            bundle = json.loads(windows_members["gpu.c3l/BUNDLE.json"])
            bundle["components"][0]["commit"] = "e" * 40
            windows_members["gpu.c3l/BUNDLE.json"] = (json.dumps(bundle) + "\n").encode()
            linux, windows = self.fixture_members(directory, self.members("linux-x64"), windows_members)

            with self.assertRaisesRegex(RuntimeError, "component provenance mismatch"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_rejects_missing_declared_native_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux_members = self.members("linux-x64")
            missing = LINUX_NATIVE_FILES[0]
            del linux_members[f"gpu.c3l/{missing}"]
            linux, windows = self.fixture_members(directory, linux_members, self.members("windows-x64"))

            with self.assertRaisesRegex(RuntimeError, "missing declared native files"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_rejects_duplicate_archive_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux, windows = self.fixtures(directory)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(windows, "a") as archive:
                    archive.writestr("gpu.c3l/README.md", b"duplicate\n")

            with self.assertRaisesRegex(RuntimeError, "duplicate archive member"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_rejects_traversal_and_development_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in (
                "gpu.c3l/../escape",
                "gpu.c3l/scripts/unsafe.py",
                "gpu.c3l/.gitmodules",
                "gpu.c3l//README.md",
                "gpu.c3l/gpu/./gpu.c3",
            ):
                with self.subTest(name=name):
                    linux_members = self.members("linux-x64", **{name: b"unsafe\n"})
                    windows_members = self.members("windows-x64", **{name: b"unsafe\n"})
                    linux, windows = self.fixture_members(directory, linux_members, windows_members)
                    with self.assertRaisesRegex(RuntimeError, "unsafe archive member"):
                        build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")
                    for path in (linux, windows):
                        path.unlink()

    def test_requires_exact_bundle_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            windows_members = self.members("windows-x64")
            bundle = json.loads(windows_members["gpu.c3l/BUNDLE.json"])
            bundle["target"] = "linux-x64"
            windows_members["gpu.c3l/BUNDLE.json"] = (json.dumps(bundle) + "\n").encode()
            linux, windows = self.fixture_members(directory, self.members("linux-x64"), windows_members)

            with self.assertRaisesRegex(RuntimeError, "unexpected bundle target"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_rejects_altered_shared_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            windows_members = self.members("windows-x64", **{"gpu.c3l/gpu/gpu.c3": b"different\n"})
            linux, windows = self.fixture_members(directory, self.members("linux-x64"), windows_members)

            with self.assertRaisesRegex(RuntimeError, "shared file contents differ"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_rejects_unexpected_inventory_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            windows_members = self.members("windows-x64")
            windows_members["gpu.c3l/gpu/optional.c3"] = b"optional\n"
            linux_members = self.members("linux-x64")
            linux_members["gpu.c3l/gpu/optional.c3"] = b"optional\n"
            del windows_members["gpu.c3l/gpu/optional.c3"]
            linux, windows = self.fixture_members(directory, linux_members, windows_members)

            with self.assertRaisesRegex(RuntimeError, "archive inventory mismatch"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_rejects_identical_omission_of_required_release_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux_members = self.members("linux-x64")
            windows_members = self.members("windows-x64")
            del linux_members["gpu.c3l/manifest.json"]
            del windows_members["gpu.c3l/manifest.json"]
            linux, windows = self.fixture_members(directory, linux_members, windows_members)

            with self.assertRaisesRegex(RuntimeError, "missing required release members"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_rejects_identical_omission_of_non_sentinel_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux_members = self.members("linux-x64")
            windows_members = self.members("windows-x64")
            del linux_members["gpu.c3l/gpu/extra_gpu.c3"]
            del windows_members["gpu.c3l/gpu/extra_gpu.c3"]
            linux, windows = self.fixture_members(directory, linux_members, windows_members)

            with self.assertRaisesRegex(RuntimeError, "archive inventory mismatch"):
                build_distribution.build_distribution(
                    "v1.2.3", linux, windows, directory / "combined", source_root=self.source_root
                )

    def test_rejects_extra_native_library_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            extra = "gpu.c3l/lib/vma.c3l/linked-libs/linux-x64/extra.a"
            linux_members = self.members("linux-x64", **{extra: b"extra\n"})
            windows_members = self.members("windows-x64", **{extra: b"extra\n"})
            linux, windows = self.fixture_members(directory, linux_members, windows_members)

            with self.assertRaisesRegex(RuntimeError, "unexpected native file"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_rejects_native_library_in_wrong_target_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            wrong = WINDOWS_NATIVE_FILES[0]
            linux_members = self.members("linux-x64", **{f"gpu.c3l/{wrong}": b"wrong\n"})
            linux, windows = self.fixture_members(directory, linux_members, self.members("windows-x64"))

            with self.assertRaisesRegex(RuntimeError, "unexpected native file"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_refuses_any_existing_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux, windows = self.fixtures(directory)
            for kind in ("directory", "file"):
                with self.subTest(kind=kind):
                    output = directory / f"combined-{kind}"
                    if kind == "directory":
                        output.mkdir()
                    else:
                        output.write_text("existing\n")

                    with self.assertRaisesRegex(RuntimeError, "output path already exists"):
                        build_distribution.build_distribution("v1.2.3", linux, windows, output)

    def test_cleans_staging_when_publication_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux, windows = self.fixtures(directory)
            output = directory / "combined"

            with mock.patch.object(Path, "replace", side_effect=OSError("publication failure")):
                with self.assertRaisesRegex(OSError, "publication failure"):
                    build_distribution.build_distribution("v1.2.3", linux, windows, output)

            self.assertFalse(output.exists())
            self.assertEqual([], list(directory.glob(".combined.staging-*")))

    def test_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux, windows = self.fixtures(directory)
            first, second = directory / "first", directory / "second"
            build_distribution.build_distribution("v1.2.3", linux, windows, first)
            build_distribution.build_distribution("v1.2.3", linux, windows, second)

            self.assertEqual(
                {
                    path.relative_to(first).as_posix(): hashlib.sha256(path.read_bytes()).digest()
                    for path in first.rglob("*") if path.is_file()
                },
                {
                    path.relative_to(second).as_posix(): hashlib.sha256(path.read_bytes()).digest()
                    for path in second.rglob("*") if path.is_file()
                },
            )

    def test_rejects_invalid_bundle_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux_members = self.members("linux-x64")
            windows_members = self.members("windows-x64")
            for members in (linux_members, windows_members):
                bundle = json.loads(members["gpu.c3l/BUNDLE.json"])
                bundle["source"]["commit"] = "not-a-commit"
                members["gpu.c3l/BUNDLE.json"] = (json.dumps(bundle) + "\n").encode()
            linux, windows = self.fixture_members(directory, linux_members, windows_members)

            with self.assertRaisesRegex(RuntimeError, "invalid bundle manifest"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_rejects_non_file_archive_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux = directory / "gpu.c3l-v1.2.3-linux-x64.tar.gz"
            with tarfile.open(linux, "w:gz") as archive:
                for name, content in self.members("linux-x64").items():
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
                link = tarfile.TarInfo("gpu.c3l/gpu/link")
                link.type = tarfile.SYMTYPE
                link.linkname = "gpu.c3"
                archive.addfile(link)
            windows = directory / "gpu.c3l-v1.2.3-windows-x64.zip"
            self.write_zip(windows, self.members("windows-x64"))

            with self.assertRaisesRegex(RuntimeError, "unsafe archive member type"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_rejects_non_regular_zip_unix_member_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            linux, windows = self.fixtures(directory)
            with zipfile.ZipFile(windows, "a") as archive:
                info = zipfile.ZipInfo("gpu.c3l/gpu/fifo")
                info.external_attr = (stat.S_IFIFO | 0o644) << 16
                archive.writestr(info, b"")

            with self.assertRaisesRegex(RuntimeError, "unsafe archive member type"):
                build_distribution.build_distribution("v1.2.3", linux, windows, directory / "combined")

    def test_cli_runs_as_a_script(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archives = directory / "assets"
            linux = package_release.create_release(ROOT, "1.2.3", "linux-x64", archives)
            windows = package_release.create_release(ROOT, "1.2.3", "windows-x64", archives)
            output = directory / "combined"

            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/build_distribution.py"),
                    "--tag", "v1.2.3",
                    "--linux-archive", str(linux),
                    "--windows-archive", str(windows),
                    "--output-dir", str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue((output / "DISTRIBUTION.json").is_file())


if __name__ == "__main__":
    unittest.main()
