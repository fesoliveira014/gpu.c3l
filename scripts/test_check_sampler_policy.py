from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from scripts import check_sampler_policy


class SamplerPolicyCheckTests(unittest.TestCase):
    def copy_policy_sources(self, root: Path) -> None:
        for relative in check_sampler_policy.POLICY_FILES:
            source = check_sampler_policy.ROOT / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    def mutate(self, relative: str, old: str, new: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_policy_sources(root)
            path = root / relative
            source = path.read_text(encoding="utf-8")
            self.assertIn(old, source)
            path.write_text(source.replace(old, new, 1), encoding="utf-8")
            return check_sampler_policy.check(root)

    def mutate_pattern(self, relative: str, pattern: str, new: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_policy_sources(root)
            path = root / relative
            source = path.read_text(encoding="utf-8")
            mutated, count = re.subn(pattern, new, source, count=1)
            self.assertEqual(count, 1)
            path.write_text(mutated, encoding="utf-8")
            return check_sampler_policy.check(root)

    def test_current_sources_satisfy_contract(self) -> None:
        self.assertEqual(check_sampler_policy.check(), [])

    def test_rejects_validation_after_vtable_dispatch(self) -> None:
        errors = self.mutate(
            "gpu/gpu.c3",
            "    gpu::internal::validate_sampler_desc(&operation, desc)!;\n"
            "    return operation.vtable.intern_sampler(device, desc);",
            "    return operation.vtable.intern_sampler(device, desc);\n"
            "    gpu::internal::validate_sampler_desc(&operation, desc)!;",
        )
        self.assertIn("sampler validation must precede vtable dispatch", errors)

    def test_rejects_missing_over_limit_branch(self) -> None:
        errors = self.mutate(
            "gpu/internal/sampler.c3",
            "desc.max_anisotropy > operation.data.caps.max_sampler_anisotropy",
            "desc.max_anisotropy < operation.data.caps.max_sampler_anisotropy",
        )
        self.assertIn(
            "active anisotropy must explicitly reject values above the device cap",
            errors,
        )

    def test_rejects_validation_clamping_before_rejection(self) -> None:
        errors = self.mutate(
            "gpu/internal/sampler.c3",
            "    if (desc.max_anisotropy > "
            "operation.data.caps.max_sampler_anisotropy) {",
            "    desc.max_anisotropy = operation.data.caps.max_sampler_anisotropy;\n"
            "    if (desc.max_anisotropy > "
            "operation.data.caps.max_sampler_anisotropy) {",
        )
        self.assertIn(
            "sampler validation must not mutate or clamp max_anisotropy",
            errors,
        )

    def test_rejects_canonical_device_cap_input(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/internal.c3",
            "fn SamplerKey canonical_sampler_key(gpu::SamplerDesc* desc) {",
            "fn SamplerKey canonical_sampler_key(\n"
            "    gpu::SamplerDesc* desc,\n"
            "    float device_max_anisotropy,\n"
            ") {",
        )
        self.assertIn(
            "canonical sampler identity must not accept a device-cap input",
            errors,
        )

    def test_rejects_clamping_restoration(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/internal.c3",
            "? canonical_sampler_float(desc.max_anisotropy)\n"
            "            : 0.0f,",
            "? canonical_sampler_float(desc.max_anisotropy > 16.0f\n"
            "                ? 16.0f\n"
            "                : desc.max_anisotropy)\n"
            "            : 0.0f,",
        )
        self.assertIn("canonical sampler identity must not clamp anisotropy", errors)

    def test_rejects_native_field_drift(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/sampler.c3",
            ".set_max_anisotropy(key.max_anisotropy)",
            ".set_max_anisotropy(desc.max_anisotropy)",
        )
        self.assertIn("native sampler creation must consume key.max_anisotropy", errors)

    def test_rejects_device_cap_at_canonical_call_site(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/sampler.c3",
            "SamplerKey key = canonical_sampler_key(desc);",
            "SamplerKey key = canonical_sampler_key(\n"
            "        desc,\n"
            "        state.max_sampler_anisotropy,\n"
            "    );",
        )
        self.assertIn(
            "Vulkan sampler interning must canonicalize only the description",
            errors,
        )

    def test_rejects_public_source_policy_drift(self) -> None:
        errors = self.mutate(
            "gpu/gpu.c3i",
            "above caps.max_sampler_anisotropy faults INVALID_ARGUMENT.",
            "above caps.max_sampler_anisotropy is implementation-defined.",
        )
        self.assertIn(
            "SamplerDesc public contract must state over-cap INVALID_ARGUMENT rejection",
            errors,
        )

    def test_rejects_documentation_policy_drift(self) -> None:
        errors = self.mutate_pattern(
            "docs/api.md",
            r"is\s+not\s+implicitly\s+clamped",
            "is implicitly clamped",
        )
        self.assertIn(
            "API docs must state that sampler anisotropy is not implicitly clamped",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
