#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_INDEX = Path("docs/document_index.md")
TARGET_DOCUMENT = Path("docs/strict_gpu_profile.md")
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\((?P<target>[^)\s]+)")
PROJECT_HISTORY_PATTERNS = (
    (
        re.compile(r"\b(?:PR|pull request)\s+#\d+\b", re.IGNORECASE),
        "references",
    ),
    (
        re.compile(r"\bissue\s+#\d+\b", re.IGNORECASE),
        "references",
    ),
    (
        re.compile(r"\bMilestone\s+\d+(?:\.\d+)?\b", re.IGNORECASE),
        "uses development label",
    ),
    (
        re.compile(r"\bPhase\s+\d+(?:\.\d+)?\b", re.IGNORECASE),
        "uses development label",
    ),
    (
        re.compile(r"\bM\d+(?:\.\d+)?\b"),
        "uses development label",
    ),
)


def validate_links(
    root: Path,
    relative: Path,
    source: str,
) -> list[str]:
    failures = []
    parent = (root / relative).parent
    for line_number, line in enumerate(source.splitlines(), start=1):
        for match in MARKDOWN_LINK.finditer(line):
            target = match.group("target").strip("<>")
            if (
                not target
                or target.startswith("#")
                or "://" in target
                or target.startswith("mailto:")
            ):
                continue
            path_text = unquote(target.split("#", 1)[0])
            if not path_text:
                continue
            resolved = parent / path_text
            if not resolved.exists():
                display = resolved.relative_to(root).as_posix()
                failures.append(
                    f"{relative.as_posix()}:{line_number}: "
                    f"missing link target {display}"
                )
    return failures


def validate_document_index(root: Path) -> list[str]:
    index = root / DOCUMENT_INDEX
    source = index.read_text(encoding="utf-8")
    targets = {
        Path(match.group("target").strip("<>").split("#", 1)[0]).name
        for match in MARKDOWN_LINK.finditer(source)
    }
    topic_names = {
        path.name
        for path in (root / "docs").glob("*.md")
        if path.name != DOCUMENT_INDEX.name
    }
    return [
        f"{DOCUMENT_INDEX.as_posix()} missing topic link {name}"
        for name in sorted(topic_names - targets)
    ]


def validate_manifest_sources(
    root: Path,
    source_entries: list[str],
) -> list[str]:
    source_files = {
        path.relative_to(root).as_posix()
        for path in (root / "gpu").rglob("*")
        if path.is_file() and path.suffix in {".c3", ".c3i"}
    }
    covered = set()
    failures = []
    for entry in source_entries:
        normalized = entry.replace("\\", "/")
        if normalized.endswith("/**"):
            prefix = normalized[:-3].rstrip("/")
            matches = {
                path
                for path in source_files
                if path.startswith(prefix + "/")
            }
            if not matches:
                failures.append(
                    f"manifest source pattern matches nothing: {normalized}"
                )
            covered.update(matches)
            continue
        if not (root / normalized).is_file():
            failures.append(
                f"manifest source does not exist: {normalized}"
            )
            continue
        covered.add(normalized)
    failures.extend(
        f"manifest sources omit {path}"
        for path in sorted(source_files - covered)
    )
    return failures


def validate_current_state_text(
    relative: Path,
    source: str,
) -> list[str]:
    if relative == TARGET_DOCUMENT or "specs" in relative.parts:
        return []
    failures = []
    subject = (
        "current-state documentation"
        if relative.suffix == ".md"
        else "public source text"
    )
    for line_number, line in enumerate(source.splitlines(), start=1):
        for pattern, action in PROJECT_HISTORY_PATTERNS:
            match = pattern.search(line)
            if match is not None:
                failures.append(
                    f"{relative.as_posix()}:{line_number}: "
                    f"{subject} {action} {match.group(0)}"
                )
    return failures


def strip_json_comments(source: str) -> str:
    return "\n".join(
        line.split("//", 1)[0]
        for line in source.splitlines()
    )


def is_private_backend_source(relative: Path) -> bool:
    return relative.parts[:3] == ("gpu", "internal", "vk")


NEGATION_CUE = re.compile(r"\b(?:not|never|neither|without)\b")
OBJECT_OVERSTATEMENT = re.compile(
    r"\bobject_boundaries\b.{0,120}(?:(?:selects?|uses?|enables?|runs?|"
    r"performs?|provides?|adds?).{0,60}(?:checked commands?|full commands?|"
    r"command (?:semantics?|validation|checks?|diagnostics?))|"
    r"(?:validates?|checks?|diagnoses?).{0,30}commands?)\b"
)
UNIVERSAL_OVERSTATEMENT = re.compile(
    r"\b(?:(?:every|all) (?:validation )?(?:polic(?:y|ies)|modes?).{0,180}"
    r"(?:diagnoses?|rejects?|checks?|faults?|validates?).{0,100}"
    r"(?:viewport|limit|usage|enum|semantic misuse)|(?:diagnoses?|rejects?|"
    r"checks?|faults?|validates?).{0,100}(?:viewport|limit|usage|enum|"
    r"semantic misuse).{0,180}(?:every|all) (?:validation )?"
    r"(?:polic(?:y|ies)|modes?))\b"
)


def has_nonnegated_match(
    source: str, pattern: str | re.Pattern[str], prefix: int = 24
) -> bool:
    for match in re.finditer(pattern, source):
        window = source[max(0, match.start() - prefix):match.end()]
        if NEGATION_CUE.search(window) is None:
            return True
    return False


def validate_semantic_markdown(sources: dict[Path, str]) -> list[str]:
    failures = []
    normalized = {
        relative: " ".join(re.findall(r"[a-z0-9_]+", source.lower()))
        for relative, source in sources.items()
        if "specs" not in relative.parts
    }
    policy = normalized.get(Path("docs/api.md"))
    if policy and has_nonnegated_match(
        policy,
        r"\bzero initialized runtime ?desc\b.{0,180}"
        r"\b(?:selects|defaults to|uses) (?:the )?(?:contractvalidation )?"
        r"(?:object_boundaries|full)\b",
    ):
        failures.append("docs/api.md: zero-initialized RuntimeDesc must select TRUSTED")
    for relative, source in normalized.items():
        if has_nonnegated_match(source, OBJECT_OVERSTATEMENT):
            failures.append(
                f"{relative}: OBJECT_BOUNDARIES must not select checked "
                f"command semantics"
            )
        if has_nonnegated_match(source, UNIVERSAL_OVERSTATEMENT):
            failures.append(
                f"{relative}: FULL-only semantic diagnostics must not be "
                f"promised by every policy"
            )
    return failures


def collect_failures(root: Path) -> list[str]:
    failures = []
    markdown_sources = {}
    for path in sorted((root / "docs").rglob("*.md")):
        relative = path.relative_to(root)
        source = path.read_text(encoding="utf-8")
        markdown_sources[relative] = source
        failures.extend(validate_links(root, relative, source))
        failures.extend(validate_current_state_text(relative, source))
    failures.extend(validate_document_index(root))
    failures.extend(validate_semantic_markdown(markdown_sources))

    for path in sorted((root / "gpu").rglob("*")):
        relative = path.relative_to(root)
        if (
            path.is_file()
            and path.suffix in {".c3", ".c3i"}
            and not is_private_backend_source(relative)
        ):
            source = path.read_text(encoding="utf-8")
            failures.extend(
                validate_current_state_text(relative, source)
            )

    manifest = json.loads(strip_json_comments(
        (root / "manifest.json").read_text(encoding="utf-8")
    ))
    failures.extend(
        validate_manifest_sources(root, manifest.get("sources", []))
    )
    return failures


def main() -> int:
    failures = collect_failures(ROOT)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("documentation links and source lists match the repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
