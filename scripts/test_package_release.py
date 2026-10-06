import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import package_release


ROOT = Path(__file__).resolve().parents[1]


class PackageReleaseTests(unittest.TestCase):
    def test_root_dependency_graph_contains_only_runtime_bindings(self) -> None:
        package_release.validate_root_dependency_graph(ROOT)

    def test_artifact_is_packed_neutral_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = package_release.create_release(
                root=ROOT,
                version="0.1.0",
                output_dir=Path(first_dir),
            )
            second = package_release.create_release(
                root=ROOT,
                version="0.1.0",
                output_dir=Path(second_dir),
            )
            self.assertEqual("gpu-v0.1.0.c3l", first.name)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            checksums = (first.parent / "SHA256SUMS").read_text(encoding="utf-8")
            self.assertEqual(
                f"{hashlib.sha256(first.read_bytes()).hexdigest()}  {first.name}\n",
                checksums,
            )

            with zipfile.ZipFile(first) as archive:
                names = archive.namelist()
                infos = archive.infolist()
                manifest = archive.read("manifest.json").decode("utf-8")
                bundle = json.loads(archive.read("BUNDLE.json"))
                license_text = archive.read("LICENSE").decode("utf-8")
            members = set(names)
            package_release.validate_consumer_doc_links(members, first)

        self.assertEqual(sorted(names), names)
        self.assertEqual({(1980, 1, 1, 0, 0, 0)}, {info.date_time for info in infos})
        self.assertIn('"provides": "gpu"', manifest)
        self.assert_required_members(members)
        self.assert_bundle_metadata(bundle)
        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2026 fesoliveira014", license_text)

    def test_rejects_invalid_versions(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "semantic version"):
                package_release.create_release(
                    root=ROOT,
                    version="v0.1",
                    output_dir=Path(output_dir),
                )

    def assert_required_members(self, members: set[str]) -> None:
        required = {
            "BUNDLE.json",
            "LICENSE",
            "README.md",
            "manifest.json",
            "gpu/gpu.c3",
            "gpu/gpu.c3i",
            "gpu/util/device_context.c3",
            "docs/concepts.md",
            "docs/api/index.md",
            "docs/util/index.md",
            "docs/util/device_context.md",
            "include/shaders/descriptor_heap.glsl",
            "include/shaders/generated/shader_abi.glsl",
            "tools/gpu_shaders/project.json",
            "tools/gpu_shaders/src/cli.c3",
        }
        self.assertTrue(required <= members, required - members)

        forbidden_parts = {
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
        for member in members:
            self.assertTrue(forbidden_parts.isdisjoint(Path(member).parts), member)
            self.assertFalse(member.endswith((".a", ".lib", ".so", ".dll")), member)
        self.assertEqual([], [member for member in members if member.startswith(("lib/", "linked-libs/"))])

    def assert_bundle_metadata(self, bundle: dict) -> None:
        self.assertEqual(2, bundle["schema"])
        self.assertEqual("gpu.c3l", bundle["name"])
        self.assertEqual("0.1.0", bundle["version"])
        self.assertEqual(
            ["vk.c3l", "vma.c3l", "spvreflect.c3l"],
            [component["name"] for component in bundle["components"]],
        )
        for component in bundle["components"]:
            self.assertRegex(component["commit"], r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
