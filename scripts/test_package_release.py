import hashlib
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import package_release


ROOT = Path(__file__).resolve().parents[1]


class PackageReleaseTests(unittest.TestCase):
    def test_root_dependency_graph_contains_only_runtime_bindings(self) -> None:
        package_release.validate_root_dependency_graph(ROOT)

    def test_linux_bundle_is_runtime_only_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = package_release.create_release(
                root=ROOT,
                version="0.1.0",
                target="linux-x64",
                output_dir=Path(first_dir),
            )
            second = package_release.create_release(
                root=ROOT,
                version="0.1.0",
                target="linux-x64",
                output_dir=Path(second_dir),
            )

            self.assertEqual(
                hashlib.sha256(first.read_bytes()).digest(),
                hashlib.sha256(second.read_bytes()).digest(),
            )

            with tarfile.open(first, "r:gz") as archive:
                members = {member.name for member in archive.getmembers() if member.isfile()}
                bundle = json.load(archive.extractfile("gpu.c3l/BUNDLE.json"))
                license_text = archive.extractfile("gpu.c3l/LICENSE").read().decode("utf-8")
            package_release.validate_consumer_doc_links(members, first)

        self.assert_required_members(members)
        self.assertIn(
            "gpu.c3l/lib/vma.c3l/linked-libs/linux-x64/libVulkanMemoryAllocator.a",
            members,
        )
        self.assertIn("gpu.c3l/lib/spvreflect.c3l/linux/libspvreflect.a", members)
        self.assertNotIn("gpu.c3l/lib/vk.c3l/windows/vulkan-1.lib", members)
        self.assert_bundle_metadata(bundle, "linux-x64")
        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2026 fesoliveira014", license_text)

    def test_windows_bundle_contains_only_windows_native_files(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            archive_path = package_release.create_release(
                root=ROOT,
                version="0.1.0",
                target="windows-x64",
                output_dir=Path(output_dir),
            )
            with zipfile.ZipFile(archive_path) as archive:
                members = {name for name in archive.namelist() if not name.endswith("/")}
                bundle = json.loads(archive.read("gpu.c3l/BUNDLE.json"))
                license_text = archive.read("gpu.c3l/LICENSE").decode("utf-8")
            package_release.validate_consumer_doc_links(members, archive_path)

        self.assert_required_members(members)
        self.assertIn(
            "gpu.c3l/lib/vma.c3l/linked-libs/windows-x64/VulkanMemoryAllocator.lib",
            members,
        )
        self.assertIn("gpu.c3l/lib/spvreflect.c3l/windows/spvreflect.lib", members)
        self.assertIn("gpu.c3l/lib/vk.c3l/windows/vulkan-1.lib", members)
        self.assertNotIn("gpu.c3l/lib/spvreflect.c3l/linux/libspvreflect.a", members)
        self.assert_bundle_metadata(bundle, "windows-x64")
        self.assertIn("MIT License", license_text)

    def test_rejects_invalid_versions_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "semantic version"):
                package_release.create_release(
                    root=ROOT,
                    version="v0.1",
                    target="linux-x64",
                    output_dir=Path(output_dir),
                )
            with self.assertRaisesRegex(ValueError, "unsupported target"):
                package_release.create_release(
                    root=ROOT,
                    version="0.1.0",
                    target="macos-x64",
                    output_dir=Path(output_dir),
                )

    def assert_required_members(self, members: set[str]) -> None:
        required = {
            "gpu.c3l/BUNDLE.json",
            "gpu.c3l/LICENSE",
            "gpu.c3l/README.md",
            "gpu.c3l/manifest.json",
            "gpu.c3l/gpu/gpu.c3",
            "gpu.c3l/gpu/gpu.c3i",
            "gpu.c3l/docs/api/index.md",
            "gpu.c3l/lib/vk.c3l/LICENSE",
            "gpu.c3l/lib/vk.c3l/manifest.json",
            "gpu.c3l/lib/vma.c3l/LICENSE",
            "gpu.c3l/lib/vma.c3l/manifest.json",
            "gpu.c3l/lib/spvreflect.c3l/LICENSE",
            "gpu.c3l/lib/spvreflect.c3l/LICENSE.spirv-reflect.apache-2.0",
            "gpu.c3l/lib/spvreflect.c3l/NOTICE",
            "gpu.c3l/lib/spvreflect.c3l/manifest.json",
        }
        self.assertTrue(required <= members, required - members)

        forbidden_parts = {
            ".git",
            ".github",
            "test",
            "tests",
            "examples",
            "scripts",
            "openspec",
            "contributing",
            "sdl3.c3l",
        }
        for member in members:
            self.assertTrue(forbidden_parts.isdisjoint(Path(member).parts), member)

    def assert_bundle_metadata(self, bundle: dict, target: str) -> None:
        self.assertEqual(1, bundle["schema"])
        self.assertEqual("gpu.c3l", bundle["name"])
        self.assertEqual("0.1.0", bundle["version"])
        self.assertEqual(target, bundle["target"])
        self.assertEqual(
            ["vk.c3l", "vma.c3l", "spvreflect.c3l"],
            [component["name"] for component in bundle["components"]],
        )
        for component in bundle["components"]:
            self.assertRegex(component["commit"], r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
