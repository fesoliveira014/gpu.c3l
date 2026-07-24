from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import check_command_policy


STRUCT_SOURCE = """
module gpu::internal;

struct CommandOps @private {
    CopyFn copy;
    DrawFn draw;
}
"""

TABLE_NAMES = (
    "TRUSTED_COMMAND_OPS",
    "TRUSTED_TRACKING_COMMAND_OPS",
    "CHECKED_COMMAND_OPS",
    "CHECKED_TRACKING_COMMAND_OPS",
)


def table_source(
    name: str,
    copy: str = "&copy_command",
    draw: str = "&draw_command",
) -> str:
    return f"""
const gpu::internal::CommandOps {name} @private = {{
    .copy = {copy},
    .draw = {draw},
}};
"""


def all_tables_source() -> str:
    return "module gpu::internal::vk;\n" + "".join(
        table_source(name) for name in TABLE_NAMES
    )


class CommandPolicyCheckTests(unittest.TestCase):
    def write_source(
        self,
        root: Path,
        relative: str,
        source: str,
    ) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    def write_fixture(
        self,
        root: Path,
        tables: str | None = None,
        helpers: str = "",
    ) -> None:
        self.write_source(root, "gpu/internal/device.c3", STRUCT_SOURCE)
        self.write_source(
            root,
            "gpu/internal/vk/device.c3",
            (all_tables_source() if tables is None else tables) + helpers,
        )

    def test_current_sources_satisfy_contract(self) -> None:
        self.assertEqual(check_command_policy.check(), [])

    def test_accepts_complete_direct_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_fewer_runtime_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = all_tables_source().replace(
                table_source("CHECKED_TRACKING_COMMAND_OPS"),
                "",
            )
            self.write_fixture(root, source)
            self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_renamed_or_additional_runtime_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = all_tables_source() + table_source("EXPERIMENTAL_COMMAND_OPS")
            self.write_fixture(root, source)
            self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_complete_tables_across_backend_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            self.write_source(
                root,
                "gpu/internal/vk/nested/duplicate.c3",
                table_source("TRUSTED_COMMAND_OPS"),
            )
            self.assertEqual(check_command_policy.check(root), [])

    def test_rejects_duplicate_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = all_tables_source().replace(
                "    .copy = &copy_command,\n",
                (
                    "    .copy = &copy_command,\n"
                    "    .copy = &other_copy_command,\n"
                ),
                1,
            )
            self.write_fixture(root, source)
            self.assertIn(
                "TRUSTED_COMMAND_OPS has duplicate fields: copy",
                check_command_policy.check(root),
            )

    def test_rejects_duplicate_command_ops_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            struct_source = STRUCT_SOURCE.replace(
                "    CopyFn copy;\n",
                "    CopyFn copy;\n    OtherCopyFn copy;\n",
            )
            self.write_source(root, "gpu/internal/device.c3", struct_source)
            self.write_source(
                root,
                "gpu/internal/vk/device.c3",
                all_tables_source(),
            )
            self.assertIn(
                "CommandOps has duplicate fields: copy",
                check_command_policy.check(root),
            )

    def test_reads_field_name_before_trailing_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            struct_source = STRUCT_SOURCE.replace(
                "    CopyFn copy;\n",
                "    CopyFn copy @deprecated;\n",
            )
            self.write_source(root, "gpu/internal/device.c3", struct_source)
            self.write_source(
                root,
                "gpu/internal/vk/device.c3",
                all_tables_source(),
            )
            self.assertEqual(check_command_policy.check(root), [])

    def test_rejects_missing_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = all_tables_source().replace(
                "    .draw = &draw_command,\n",
                "",
                1,
            )
            self.write_fixture(root, source)
            self.assertIn(
                "TRUSTED_COMMAND_OPS is missing CommandOps fields: draw",
                check_command_policy.check(root),
            )

    def test_rejects_extra_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = all_tables_source().replace(
                "    .draw = &draw_command,\n",
                (
                    "    .draw = &draw_command,\n"
                    "    .dispatch = &dispatch_command,\n"
                ),
                1,
            )
            self.write_fixture(root, source)
            self.assertIn(
                "TRUSTED_COMMAND_OPS has unknown CommandOps fields: dispatch",
                check_command_policy.check(root),
            )

    def test_accepts_indirect_table_entries(self) -> None:
        initializers = (
            "copy_command",
            "copy_command()",
            "(&copy_command)",
            "&commands::copy_command",
            "select_copy_command()",
            "use_fast_copy ? &copy_command : &other_copy_command",
        )
        for initializer in initializers:
            with self.subTest(initializer=initializer):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = all_tables_source().replace(
                        "    .copy = &copy_command,\n",
                        f"    .copy = {initializer},\n",
                        1,
                    )
                    self.write_fixture(root, source)
                    self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_table_free_specialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root, "module gpu::internal::vk;\n")
            self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_indirectly_constructed_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(
                root,
                (
                    "module gpu::internal::vk;\n"
                    "const gpu::internal::CommandOps COMBINED = make_ops();\n"
                ),
            )
            self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_helper_rename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = all_tables_source().replace(
                "&copy_command",
                "&renamed_copy_command",
            )
            self.write_fixture(
                root,
                source,
                "\nfn void renamed_copy_command() {}\n",
            )
            self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_helper_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(
                root,
                helpers="""
fn void copy_command() {
    extracted_copy_helper();
}
fn void extracted_copy_helper() {}
""",
            )
            self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_helper_inlining(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(
                root,
                helpers="\nfn void copy_command() { native_copy(); }\n",
            )
            self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_helper_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            self.write_source(
                root,
                "gpu/internal/vk/commands/copy.c3",
                "fn void copy_command() {}\n",
            )
            self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_definition_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = (
                "module gpu::internal::vk;\n"
                "fn void copy_command() {}\n"
                + "".join(reversed([
                    table_source(name) for name in TABLE_NAMES
                ]))
                + "fn void draw_command() {}\n"
            )
            self.write_fixture(root, source)
            self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_expression_bodied_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(
                root,
                helpers="\nfn void copy_command() => native_copy();\n",
            )
            self.assertEqual(check_command_policy.check(root), [])

    def test_ignores_retired_patterns_in_comments_and_strings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(
                root,
                helpers=r'''
fn void copy_command() {
    io::printn("@pool() tlocal mem::new_array vk::allocate_command_buffers");
    io::printn("const gpu::internal::CommandOps STRING_TABLE = {");
    // validation_policy track_command_reference scratch.reference_count
    // const gpu::internal::CommandOps COMMENT_TABLE = {
    /* vk_cmd_copy_buffer RecordingContextTable */
}
''',
            )
            self.assertEqual(check_command_policy.check(root), [])

    def test_accepts_reuse_of_former_internal_helper_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = all_tables_source().replace(
                "&copy_command",
                "&vk_cmd_copy_buffer",
                1,
            )
            self.write_fixture(
                root,
                tables=source,
                helpers="\nfn void vk_cmd_copy_buffer() {}\n",
            )
            self.assertEqual(check_command_policy.check(root), [])


if __name__ == "__main__":
    unittest.main()
