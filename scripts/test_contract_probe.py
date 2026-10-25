"""Safe builds of a consumer panic with the contract message."""

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "test" / "cpu"
EXECUTABLE = PROJECT / "build" / (
    "contract_probe.exe" if os.name == "nt" else "contract_probe"
)

BUILDS = {
    "default": [],
    "O1": ["-O1"],
}

PANIC_CASES = {
    "null_runtime_desc": "desc must not be null",
    "adapter_index": "index must be less than the adapter count",
    "queue_role_not_exposed": "kind must be a queue role the device exposes",
    "subspan_out_of_range": "offset + size must not exceed the span size",
}


def run(command):
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )


class ContractProbeTest(unittest.TestCase):
    def check_build(self, build):
        built = run(
            ["c3c", "build", "contract_probe", "--path", str(PROJECT), *BUILDS[build]]
        )
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)

        valid = run([str(EXECUTABLE), "valid"])
        self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)

        for case, message in PANIC_CASES.items():
            with self.subTest(build=build, case=case):
                result = run([str(EXECUTABLE), case])
                output = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, output)
                self.assertIn(message, output)

    def test_default_build_panics(self):
        self.check_build("default")

    def test_o1_build_panics(self):
        self.check_build("O1")


if __name__ == "__main__":
    unittest.main()
