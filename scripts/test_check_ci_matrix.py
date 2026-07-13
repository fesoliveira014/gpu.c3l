import importlib.util
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path


SCRIPT = Path(__file__).with_name("check_ci_matrix.py")


class MatrixCheckTests(unittest.TestCase):
    def load_checker(self):
        self.assertTrue(SCRIPT.is_file(), "matrix checker exists")
        spec = importlib.util.spec_from_file_location("check_ci_matrix", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_project_targets_preserve_declared_test_order(self):
        checker = self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project.json"
            project.write_text(
                '{\n'
                '  // ignored executable\n'
                '  "targets": {\n'
                '    "smoke": { "type": "executable" },\n'
                '    "vk_a": { "type": "test" },\n'
                '    "vk_b": { "type": "test" }\n'
                '  }\n'
                '}\n',
                encoding="utf-8",
            )
            self.assertEqual(["vk_a", "vk_b"], checker.read_project_targets(project))

    def test_documented_targets_read_matrix_block(self):
        checker = self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            documentation = Path(directory) / "testing.md"
            documentation.write_text(
                "The blocking headless matrix is shared by Linux and Windows:\n\n"
                "```text\n"
                "vk_a vk_b\n"
                "vk_c\n"
                "```\n",
                encoding="utf-8",
            )
            self.assertEqual(["vk_a", "vk_b", "vk_c"], checker.read_documented_targets(documentation))

    def test_main_rejects_matrix_drift(self):
        checker = self.load_checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test").mkdir()
            (root / "docs").mkdir()
            (root / "test" / "project.json").write_text(
                '{ "targets": { "vk_a": { "type": "test" }, "vk_b": { "type": "test" } } }',
                encoding="utf-8",
            )
            (root / "docs" / "testing.md").write_text(
                "The blocking headless matrix is shared by Linux and Windows:\n\n"
                "```text\nvk_a vk_b\n```\n",
                encoding="utf-8",
            )
            previous = os.environ.get("HEADLESS_TEST_TARGETS")
            os.environ["HEADLESS_TEST_TARGETS"] = "vk_a"
            try:
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(1, checker.main(root))
            finally:
                if previous is None:
                    os.environ.pop("HEADLESS_TEST_TARGETS", None)
                else:
                    os.environ["HEADLESS_TEST_TARGETS"] = previous


if __name__ == "__main__":
    unittest.main()
