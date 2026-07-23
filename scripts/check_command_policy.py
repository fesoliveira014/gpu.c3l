#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TABLES = frozenset({
    "TRUSTED_COMMAND_OPS",
    "TRUSTED_TRACKING_COMMAND_OPS",
    "CHECKED_COMMAND_OPS",
    "CHECKED_TRACKING_COMMAND_OPS",
})
IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
COMMAND_OPS_DECLARATION = re.compile(r"\bstruct\s+CommandOps\b[^{;]*\{")
TABLE_DECLARATION = re.compile(
    rf"\bconst\s+gpu::internal::CommandOps\s+({IDENTIFIER})\b[^=;{{]*=\s*\{{",
)
TABLE_FIELD = re.compile(rf"\s*\.({IDENTIFIER})\s*=")
DIRECT_TABLE_ENTRY = re.compile(
    rf"\s*\.({IDENTIFIER})\s*=\s*&({IDENTIFIER})\s*\Z",
)


@dataclass(frozen=True)
class TableEntry:
    field: str | None
    target: str | None
    direct: bool


@dataclass(frozen=True)
class CommandTable:
    relative: str
    name: str
    entries: tuple[TableEntry, ...]


def mask_non_code(source: str) -> str:
    masked = list(source)
    index = 0
    quote: str | None = None
    block_end: str | None = None
    block_depth = 0

    while index < len(source):
        if quote is not None:
            if source[index] == "\\":
                masked[index] = " "
                if index + 1 < len(source):
                    masked[index + 1] = " "
                index += 2
            elif source[index] == quote:
                masked[index] = " "
                quote = None
                index += 1
            else:
                if source[index] not in "\r\n":
                    masked[index] = " "
                index += 1
            continue

        if block_end is not None:
            block_start = "/*" if block_end == "*/" else "<*"
            if source.startswith(block_start, index):
                masked[index:index + 2] = "  "
                block_depth += 1
                index += 2
            elif source.startswith(block_end, index):
                masked[index:index + 2] = "  "
                block_depth -= 1
                index += 2
                if block_depth == 0:
                    block_end = None
            else:
                if source[index] not in "\r\n":
                    masked[index] = " "
                index += 1
            continue

        if source.startswith("//", index):
            while index < len(source) and source[index] not in "\r\n":
                masked[index] = " "
                index += 1
        elif source.startswith("/*", index):
            masked[index:index + 2] = "  "
            block_end = "*/"
            block_depth = 1
            index += 2
        elif source.startswith("<*", index):
            masked[index:index + 2] = "  "
            block_end = "*>"
            block_depth = 1
            index += 2
        elif source[index] in "\"'":
            masked[index] = " "
            quote = source[index]
            index += 1
        else:
            index += 1

    return "".join(masked)


def matching_delimiter(source: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(source)):
        if source[index] == opening:
            depth += 1
        elif source[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unterminated {opening}{closing} block")


def top_level_segments(source: str, delimiter: str) -> list[str]:
    depths = {"(": 0, "[": 0, "{": 0}
    pairs = {")": "(", "]": "[", "}": "{"}
    segments = []
    start = 0
    for index, char in enumerate(source):
        if char in depths:
            depths[char] += 1
        elif char in pairs:
            depths[pairs[char]] -= 1
        elif char == delimiter and all(depth == 0 for depth in depths.values()):
            segments.append(source[start:index])
            start = index + 1
    if source[start:].strip():
        segments.append(source[start:])
    return segments


def command_ops_fields(root: Path) -> tuple[str, ...]:
    path = root / "gpu" / "internal" / "device.c3"
    source = mask_non_code(path.read_text(encoding="utf-8"))
    declarations = list(COMMAND_OPS_DECLARATION.finditer(source))
    if len(declarations) != 1:
        raise ValueError(
            "expected exactly one CommandOps struct in gpu/internal/device.c3"
        )

    start = source.find("{", declarations[0].start())
    end = matching_delimiter(source, start, "{", "}")
    fields = []
    for declaration in top_level_segments(source[start + 1:end], ";"):
        identifiers = re.findall(IDENTIFIER, declaration)
        if not identifiers:
            raise ValueError("CommandOps contains an unrecognized field declaration")
        fields.append(identifiers[-1])
    if not fields:
        raise ValueError("CommandOps has no fields")
    return tuple(fields)


def table_entries(body: str) -> tuple[TableEntry, ...]:
    entries = []
    for expression in top_level_segments(body, ","):
        direct = DIRECT_TABLE_ENTRY.fullmatch(expression)
        if direct is not None:
            entries.append(TableEntry(
                field=direct.group(1),
                target=direct.group(2),
                direct=True,
            ))
            continue
        field = TABLE_FIELD.match(expression)
        entries.append(TableEntry(
            field=field.group(1) if field is not None else None,
            target=None,
            direct=False,
        ))
    return tuple(entries)


def command_tables(root: Path) -> tuple[CommandTable, ...]:
    backend = root / "gpu" / "internal" / "vk"
    tables = []
    for path in sorted(backend.rglob("*.c3"), key=lambda item: item.as_posix()):
        source = mask_non_code(path.read_text(encoding="utf-8"))
        relative = path.relative_to(root).as_posix()
        for declaration in TABLE_DECLARATION.finditer(source):
            start = source.find("{", declaration.start())
            end = matching_delimiter(source, start, "{", "}")
            tables.append(CommandTable(
                relative=relative,
                name=declaration.group(1),
                entries=table_entries(source[start + 1:end]),
            ))
    return tuple(tables)


def structural_table_errors(
    fields: tuple[str, ...],
    tables: tuple[CommandTable, ...],
) -> list[str]:
    errors = []
    field_counts = Counter(fields)
    duplicate_command_fields = sorted(
        field for field, count in field_counts.items() if count > 1
    )
    if duplicate_command_fields:
        errors.append(
            "CommandOps has duplicate fields: "
            + ", ".join(duplicate_command_fields)
        )
    expected_fields = set(fields)
    table_counts = Counter(table.name for table in tables)

    missing_tables = sorted(EXPECTED_TABLES - table_counts.keys())
    if missing_tables:
        errors.append(
            "missing command policy tables: " + ", ".join(missing_tables)
        )
    unexpected_tables = sorted(table_counts.keys() - EXPECTED_TABLES)
    if unexpected_tables:
        errors.append(
            "unexpected command policy tables: " + ", ".join(unexpected_tables)
        )
    duplicate_tables = sorted(
        name for name, count in table_counts.items()
        if name in EXPECTED_TABLES and count > 1
    )
    if duplicate_tables:
        errors.append(
            "duplicate command policy tables: " + ", ".join(duplicate_tables)
        )

    for table in tables:
        if table.name not in EXPECTED_TABLES:
            continue
        malformed = [
            entry for entry in table.entries
            if entry.field is None
        ]
        if malformed:
            errors.append(
                f"{table.name} has an unrecognized table entry in {table.relative}"
            )

        entry_fields = [
            entry.field for entry in table.entries
            if entry.field is not None
        ]
        counts = Counter(entry_fields)
        duplicate_fields = sorted(
            field for field, count in counts.items() if count > 1
        )
        if duplicate_fields:
            errors.append(
                f"{table.name} has duplicate fields: "
                + ", ".join(duplicate_fields)
            )

        missing_fields = [field for field in fields if field not in counts]
        if missing_fields:
            errors.append(
                f"{table.name} is missing CommandOps fields: "
                + ", ".join(missing_fields)
            )
        extra_fields = sorted(counts.keys() - expected_fields)
        if extra_fields:
            errors.append(
                f"{table.name} has unknown CommandOps fields: "
                + ", ".join(extra_fields)
            )

        for entry in table.entries:
            if entry.field is not None and not entry.direct:
                errors.append(
                    f"{table.name}.{entry.field} must be initialized with "
                    "a direct &function_name reference"
                )

    return errors


def check(root: Path = ROOT) -> list[str]:
    try:
        fields = command_ops_fields(root)
        tables = command_tables(root)
    except (OSError, ValueError) as error:
        return [str(error)]
    return structural_table_errors(fields, tables)


def main() -> int:
    errors = check()
    if errors:
        print("command policy source contract failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("command policy source contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
