#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import scripts.check_docs as check_docs


class DocumentationCheckTests(unittest.TestCase):
    def test_rejects_missing_relative_markdown_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            relative = Path("docs/guide.md")
            source = "[missing](other.md)\n[external](https://example.com)\n"

            self.assertEqual(
                check_docs.validate_links(root, relative, source),
                ["docs/guide.md:1: missing link target docs/other.md"],
            )

    def test_document_index_covers_current_topic_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "api.md").write_text("# API\n", encoding="utf-8")
            (docs / "memory.md").write_text("# Memory\n", encoding="utf-8")
            (docs / "document_index.md").write_text(
                "[API](api.md)\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_docs.validate_document_index(root),
                ["docs/document_index.md missing topic link memory.md"],
            )

    def test_manifest_sources_cover_every_gpu_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / "gpu" / "vk"
            backend.mkdir(parents=True)
            (root / "gpu" / "gpu.c3").write_text("", encoding="utf-8")
            (root / "gpu" / "gpu.c3i").write_text("", encoding="utf-8")
            (backend / "backend.c3").write_text("", encoding="utf-8")
            (root / "gpu" / "missing.c3").write_text("", encoding="utf-8")

            self.assertEqual(
                check_docs.validate_manifest_sources(
                    root,
                    ["gpu/gpu.c3", "gpu/gpu.c3i", "gpu/vk/**"],
                ),
                ["manifest sources omit gpu/missing.c3"],
            )

    def test_manifest_rejects_stale_literal_and_empty_glob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gpu").mkdir()
            (root / "gpu" / "gpu.c3").write_text("", encoding="utf-8")

            self.assertEqual(
                check_docs.validate_manifest_sources(
                    root,
                    ["gpu/gpu.c3", "gpu/removed.c3", "gpu/vk/**"],
                ),
                [
                    "manifest source does not exist: gpu/removed.c3",
                    "manifest source pattern matches nothing: gpu/vk/**",
                ],
            )

    def test_rejects_project_history_in_current_state_docs(self) -> None:
        source = (
            "# API\n"
            "Current behavior.\n"
            "Fixed by PR #123 during Milestone 7.\n"
        )

        self.assertEqual(
            check_docs.validate_current_state_text(
                Path("docs/api.md"),
                source,
            ),
            [
                "docs/api.md:3: current-state documentation references PR #123",
                "docs/api.md:3: current-state documentation uses development label Milestone 7",
            ],
        )

    def test_rejects_project_history_in_public_source_text(self) -> None:
        self.assertEqual(
            check_docs.validate_current_state_text(
                Path("gpu/api.c3"),
                "<* Added for M7.3. *>\n",
            ),
            [
                (
                    "gpu/api.c3:1: public source text uses "
                    "development label M7.3"
                ),
            ],
        )

    def test_allows_target_document_status_language(self) -> None:
        self.assertEqual(
            check_docs.validate_current_state_text(
                Path("docs/strict_gpu_profile.md"),
                "Milestone 8 is future work.\n",
            ),
            [],
        )

    def test_vk_named_public_directory_remains_current_state(self) -> None:
        relative = Path("gpu/surface/vk/escape.c3")
        source = "<* Added for M7.4. *>\n"

        self.assertFalse(check_docs.is_private_backend_source(relative))
        self.assertEqual(
            check_docs.validate_current_state_text(relative, source),
            [
                (
                    "gpu/surface/vk/escape.c3:1: public source text uses "
                    "development label M7.4"
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
