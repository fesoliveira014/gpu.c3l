import tempfile
import unittest
from pathlib import Path

from scripts.check_profile_boundary import validate_layout


class ProfileBoundaryLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.gpu = self.root / "gpu"
        self.gpu.mkdir()
        self.compat = self.gpu / "compat"
        self.compat.mkdir()
        (self.gpu / "gpu.c3").write_text("module gpu;\n", encoding="utf-8")
        (self.gpu / "gpu.c3i").write_text("module gpu;\n", encoding="utf-8")
        (self.gpu / "types.c3").write_text("module gpu;\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_accepts_minimal_strict_root(self):
        self.assertEqual([], validate_layout(self.root))

    def test_accepts_distinct_strict_source(self):
        (self.gpu / "buffer.c3").write_text(
            "module gpu;\nstruct StrictBuffer {}\n",
            encoding="utf-8",
        )
        (self.compat / "buffer.c3").write_text(
            "module gpu::compat;\nstruct CompatBuffer {}\n",
            encoding="utf-8",
        )

        self.assertEqual([], validate_layout(self.root))

    def test_rejects_duplicate_root_source(self):
        (self.gpu / "buffer.c3").write_text(
            "module gpu;\nfn void create_buffer() {}\n",
            encoding="utf-8",
        )
        (self.compat / "buffer.c3").write_text(
            "module gpu::compat;\nfn void create_buffer() {}\n",
            encoding="utf-8",
        )

        self.assertIn(
            "root source duplicates compatibility implementation: buffer.c3",
            validate_layout(self.root),
        )

    def test_accepts_distinct_strict_backend(self):
        strict_vk = self.gpu / "vk"
        strict_vk.mkdir()
        compat_vk = self.compat / "vk"
        compat_vk.mkdir()
        (strict_vk / "device.c3").write_text(
            "module gpu::vk @private;\nstruct StrictDevice {}\n",
            encoding="utf-8",
        )
        (compat_vk / "device.c3").write_text(
            "module gpu::compat::vk @private;\nstruct CompatDevice {}\n",
            encoding="utf-8",
        )

        self.assertEqual([], validate_layout(self.root))

    def test_rejects_duplicate_backend_source(self):
        strict_vk = self.gpu / "vk"
        strict_vk.mkdir()
        compat_vk = self.compat / "vk"
        compat_vk.mkdir()
        (strict_vk / "device.c3").write_text(
            "module gpu::vk @private;\nfn void create_device() {}\n",
            encoding="utf-8",
        )
        (compat_vk / "device.c3").write_text(
            "module gpu::compat::vk @private;\nfn void create_device() {}\n",
            encoding="utf-8",
        )

        self.assertIn(
            "strict backend duplicates compatibility implementation: vk/device.c3",
            validate_layout(self.root),
        )

    def test_rejects_compatibility_bridge(self):
        (self.gpu / "gpu.c3").write_text(
            "module gpu;\nimport gpu::compat;\n",
            encoding="utf-8",
        )

        self.assertIn(
            "strict source references gpu::compat: gpu.c3",
            validate_layout(self.root),
        )


if __name__ == "__main__":
    unittest.main()
