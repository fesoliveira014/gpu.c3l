from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DECLARATIONS = (ROOT / "gpu" / "gpu.c3i").read_text()
PUBLIC_IMPLEMENTATION = (ROOT / "gpu" / "gpu.c3").read_text()


class RayQueryPublicSurfaceTests(unittest.TestCase):
    def test_acceleration_structure_declarations_are_backend_neutral(self) -> None:
        names = (
            "AccelerationStructureHandle",
            "AccelerationStructureView",
            "AccelerationStructureIndex",
            "AccelerationStructureDesc",
            "AccelerationStructureRequirements",
            "AccelerationStructureInstanceDesc",
            "AccelerationStructureBuildDesc",
            "DedicatedAccelerationStructure",
            "RayQueryCaps",
        )
        for name in names:
            declaration = re.search(
                rf"(?:struct|bitstruct)\s+{name}\b.*?\n}}",
                PUBLIC_DECLARATIONS,
                re.DOTALL,
            )
            self.assertIsNotNone(declaration, name)
            self.assertNotRegex(declaration.group(0), r"\b(?:vk|vma)::")

    def test_excluded_acceleration_structure_operations_are_absent(self) -> None:
        public_source = PUBLIC_DECLARATIONS + "\n" + PUBLIC_IMPLEMENTATION
        excluded = (
            "compact_acceleration_structure",
            "copy_acceleration_structure",
            "clone_acceleration_structure",
            "serialize_acceleration_structure",
            "deserialize_acceleration_structure",
            "cmd_build_acceleration_structure_indirect",
            "build_acceleration_structure_host",
            "cmd_update_acceleration_structure_to",
        )
        for name in excluded:
            self.assertNotRegex(public_source, rf"\bfn\s+[^\n]*\b{name}\s*\(")


if __name__ == "__main__":
    unittest.main()
