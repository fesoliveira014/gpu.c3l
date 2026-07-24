from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import check_shader_reflection_policy


class ShaderReflectionPolicyCheckTests(unittest.TestCase):
    def copy_policy_sources(self, root: Path) -> None:
        for relative in check_shader_reflection_policy.POLICY_FILES:
            source = check_shader_reflection_policy.ROOT / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        shader_source = check_shader_reflection_policy.ROOT / "test/shaders/root_pointer.comp.glsl"
        shader_destination = root / "test/shaders/root_pointer.comp.glsl"
        shader_destination.parent.mkdir(parents=True, exist_ok=True)
        shader_destination.write_text(
            shader_source.read_text(encoding="utf-8"),
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
            return check_shader_reflection_policy.check(root)

    def test_current_sources_satisfy_contract(self) -> None:
        self.assertEqual(check_shader_reflection_policy.check(), [])

    def test_rejects_module_wide_descriptor_reflection(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/shader.c3",
            ".enumerate_entry_point_descriptor_bindings(",
            ".enumerate_descriptor_bindings(",
        )
        self.assertTrue(any("module-wide reflection call" in error for error in errors))

    def test_rejects_native_creation_before_exact_root_check(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/shader.c3",
            "RootAbiMismatch root_mismatch = root_push_abi_mismatch(",
            "create_pipeline_shader_module_native(state, null, null);\n"
            "    RootAbiMismatch root_mismatch = root_push_abi_mismatch(",
        )
        self.assertTrue(any("before native creation" in error for error in errors))

    def test_requires_no_push_block_acceptance(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/shader.c3",
            "if (count == 0) return RootAbiMismatch.NONE;\n"
            "    if (count != 1) return RootAbiMismatch.BLOCK_COUNT;",
            "if (count == 0) return RootAbiMismatch.BLOCK_COUNT;\n"
            "    if (count != 1) return RootAbiMismatch.BLOCK_COUNT;",
        )
        self.assertTrue(any("count == 0" in error for error in errors))

    def test_requires_integer_numeric_kind(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/shader.c3",
            "integer != expected.integer",
            "integer != integer",
        )
        self.assertTrue(any("integer != expected.integer" in error for error in errors))

    def test_requires_float_numeric_kind(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/shader.c3",
            "float_scalar == expected.integer",
            "float_scalar == float_scalar",
        )
        self.assertTrue(any("float_scalar == expected.integer" in error for error in errors))

    def test_rejects_unguarded_alternate_reference_acceptance(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/shader.c3",
            "if (allow_alternate && reference) {",
            "if (reference) {",
        )
        self.assertTrue(any(
            "alternate reference acceptance must stay behind allow_alternate" in error
            for error in errors
        ))

    def test_requires_property_specific_root_diagnostic(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/shader.c3",
            "invariant:      root_abi_mismatch_invariant(root_mismatch)",
            'invariant:      "selected-entry push constants must match"',
        )
        self.assertTrue(any("exact root mismatch" in error for error in errors))

    def test_requires_reflection_fault_mapping(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/shader.c3",
            "public_fault:   gpu::SHADER_INVALID",
            "public_fault:   gpu::INVALID_ARGUMENT",
        )
        self.assertTrue(any("map to SHADER_INVALID" in error for error in errors))

    def test_rejects_public_generated_metadata(self) -> None:
        errors = self.mutate(
            "gpu/internal/shader_abi.c3",
            "const RootAbiSpec ROOT_PUSH_ABI @private",
            "const RootAbiSpec ROOT_PUSH_ABI",
        )
        self.assertTrue(any("ROOT_PUSH_ABI @private" in error for error in errors))

    def test_generated_metadata_in_comment_does_not_satisfy_contract(self) -> None:
        errors = self.mutate(
            "gpu/internal/shader_abi.c3",
            "struct RootAbiMemberSpec @private",
            "struct RootAbiMemberSpec\n// struct RootAbiMemberSpec @private",
        )
        self.assertTrue(any("RootAbiMemberSpec @private" in error for error in errors))

    def test_generated_metadata_in_string_does_not_satisfy_contract(self) -> None:
        errors = self.mutate(
            "gpu/internal/shader_abi.c3",
            "struct RootAbiSpec @private",
            'struct RootAbiSpec\nconst String ROOT_ABI_POLICY = "struct RootAbiSpec @private";',
        )
        self.assertTrue(any("RootAbiSpec @private" in error for error in errors))

    def test_emitter_marker_in_comment_does_not_satisfy_contract(self) -> None:
        errors = self.mutate(
            "tools/gen_shader_abi/src/emit_c3.c3",
            "emit_root_abi_spec(decl, types, max_push_members, out);",
            "// emit_root_abi_spec(decl, types, max_push_members, out);",
        )
        self.assertTrue(any("emit_root_abi_spec" in error for error in errors))

    def test_shader_counter_in_comment_does_not_satisfy_contract(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/shader.c3",
            "state.shader_reflection_validations++;",
            "// state.shader_reflection_validations++;",
        )
        self.assertTrue(any("reflection validation counter" in error for error in errors))

    def test_shader_counter_in_string_does_not_satisfy_contract(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/shader.c3",
            "state.pipeline_shader_create_attempts++;",
            'ZString policy_example = "state.pipeline_shader_create_attempts++;";',
        )
        self.assertTrue(any("shader-create attempt counter" in error for error in errors))

    def test_rejects_backend_foreign_redeclaration(self) -> None:
        errors = self.mutate(
            "gpu/internal/vk/shader.c3",
            "module gpu::internal::vk @private;",
            "module gpu::internal::vk @private;\nextern fn void local_reflect() @cname(\"spvReflectLocal\");",
        )
        self.assertTrue(any("redeclares a foreign" in error for error in errors))

    def test_rejects_nested_generated_push_struct(self) -> None:
        errors = self.mutate(
            "test/shaders/root_pointer.comp.glsl",
            "uint64_t root_gpu;",
            "RootPush pc;",
        )
        self.assertTrue(any("nests a generated root struct" in error for error in errors))

    def test_ignores_policy_markers_in_comments_and_strings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_policy_sources(root)
            path = root / "gpu/internal/vk/shader.c3"
            source = path.read_text(encoding="utf-8")
            source += (
                '\n// extern fn void fake() @cname("spvReflectFake");\n'
                'const ZString POLICY_EXAMPLE = ".enumerate_descriptor_bindings(";\n'
            )
            path.write_text(source, encoding="utf-8")
            self.assertEqual(check_shader_reflection_policy.check(root), [])


if __name__ == "__main__":
    unittest.main()
