#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "trusted_command_ops": False,
    "trusted_tracking_command_ops": True,
    "checked_command_ops": False,
    "checked_tracking_command_ops": True,
}
POLICY_FIELDS = (
    "validation_policy",
    "track_lifetimes",
    "vulkan_layers",
    "debug_callback",
    "debug_names",
)
TRACKING_FUNCTIONS = frozenset((
    "track_command_reference",
    "resolve_tracked_texture_command_reference",
    "ensure_command_reference_capacity",
    "rollback_command_references",
    "note_command_reference_allocation",
))
TRACKING_CALLS = tuple(f"{name}(" for name in TRACKING_FUNCTIONS)
SUPERSEDED_COMMAND_FUNCTIONS = frozenset((
    "bind_pipeline_state",
    "resolve_generated_work_records",
    "resolve_index_span",
    "resolve_indirect_span",
    "validate_span_recording_access",
    "validate_texture_recording_access",
    "vk_cmd_begin_render_pass",
    "vk_cmd_bind_pipeline",
    "vk_cmd_copy_buffer",
    "vk_cmd_copy_buffer_to_texture",
    "vk_cmd_copy_texture_to_buffer",
    "vk_cmd_draw",
    "vk_cmd_draw_generated",
    "vk_cmd_draw_indexed",
    "vk_cmd_draw_indexed_generated",
    "vk_cmd_draw_indexed_indirect",
    "vk_cmd_draw_indexed_indirect_count",
    "vk_cmd_draw_indirect",
    "vk_cmd_fill_buffer",
    "vk_cmd_texture_barrier",
))
FUNCTION_DECLARATION = re.compile(
    r"(?m)^fn\s+[^\r\n(]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\(",
)
TABLE_DECLARATION = re.compile(
    r"(?m)^gpu::CommandOps\s+([A-Za-z_][A-Za-z0-9_]*)\b[^=]*=\s*\{",
)
TABLE_ENTRY = re.compile(
    r"\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*&([A-Za-z_][A-Za-z0-9_]*)\s*,",
)
CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass(frozen=True)
class Function:
    relative: str
    name: str
    line: int
    body: str


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


def expression_end(source: str, start: int) -> int:
    depths = {"(": 0, "[": 0, "{": 0}
    pairs = {")": "(", "]": "[", "}": "{"}
    for index in range(start, len(source)):
        char = source[index]
        if char in depths:
            depths[char] += 1
        elif char in pairs:
            depths[pairs[char]] -= 1
        elif char == ";" and all(depth == 0 for depth in depths.values()):
            return index
    raise ValueError("unterminated expression-bodied function")


def source_functions(relative: str, source: str) -> list[Function]:
    masked = mask_non_code(source)
    declarations = list(FUNCTION_DECLARATION.finditer(masked))
    functions = []
    for declaration_index, declaration in enumerate(declarations):
        parameter_start = masked.find("(", declaration.start())
        parameter_end = matching_delimiter(masked, parameter_start, "(", ")")
        limit = (
            declarations[declaration_index + 1].start()
            if declaration_index + 1 < len(declarations)
            else len(masked)
        )
        brace = masked.find("{", parameter_end, limit)
        arrow = masked.find("=>", parameter_end, limit)
        if arrow >= 0 and (brace < 0 or arrow < brace):
            end = expression_end(masked, arrow + 2)
            body = masked[arrow:end + 1]
        elif brace >= 0:
            end = matching_delimiter(masked, brace, "{", "}")
            body = masked[brace:end + 1]
        else:
            raise ValueError(
                f"{relative}:{declaration.group(1)} has no function body"
            )
        functions.append(Function(
            relative=relative,
            name=declaration.group(1),
            line=source.count("\n", 0, declaration.start()) + 1,
            body=body,
        ))
    return functions


def load_functions(root: Path) -> dict[str, list[Function]]:
    functions: dict[str, list[Function]] = {}
    backend = root / "gpu" / "vk"
    for path in sorted(backend.rglob("*.c3"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        for function in source_functions(relative, source):
            functions.setdefault(function.name, []).append(function)
    return functions


def command_tables(root: Path) -> dict[str, dict[str, str]]:
    path = root / "gpu" / "vk" / "device.c3"
    source = path.read_text(encoding="utf-8")
    masked = mask_non_code(source)
    tables: dict[str, dict[str, str]] = {}
    for declaration in TABLE_DECLARATION.finditer(masked):
        name = declaration.group(1)
        if name not in TABLES:
            continue
        start = masked.find("{", declaration.start())
        end = matching_delimiter(masked, start, "{", "}")
        tables[name] = dict(TABLE_ENTRY.findall(masked[start:end + 1]))
    return tables


def reachable_functions(
    functions: dict[str, list[Function]],
    roots: set[str],
) -> set[Function]:
    pending = list(roots)
    visited_names = set()
    reachable = set()
    candidates = set(functions)
    while pending:
        name = pending.pop()
        if name in visited_names:
            continue
        visited_names.add(name)
        for function in functions.get(name, ()):
            reachable.add(function)
            for called in set(CALL.findall(function.body)) & candidates:
                if called not in visited_names:
                    pending.append(called)
    return reachable


def check(root: Path = ROOT) -> list[str]:
    errors = []
    try:
        functions = load_functions(root)
        tables = command_tables(root)
    except (OSError, ValueError) as error:
        return [str(error)]

    missing = sorted(set(TABLES) - set(tables))
    if missing:
        errors.append(f"missing command policy tables: {', '.join(missing)}")
        return errors

    superseded = sorted(SUPERSEDED_COMMAND_FUNCTIONS & set(functions))
    if superseded:
        errors.append(
            "superseded command functions remain declared: "
            + ", ".join(superseded)
        )

    expected_fields = set(tables["trusted_command_ops"])
    if not expected_fields:
        errors.append("trusted_command_ops has no command entries")
        return errors
    for table_name, entries in sorted(tables.items()):
        fields = set(entries)
        if fields != expected_fields:
            errors.append(
                f"{table_name} command fields differ from trusted_command_ops"
            )
        for field, entry in sorted(entries.items()):
            if entry not in functions:
                errors.append(
                    f"{table_name}.{field} references missing function {entry}"
                )

    for table_name, tracking in TABLES.items():
        roots = set(tables[table_name].values())
        for function in sorted(
            reachable_functions(functions, roots),
            key=lambda item: (item.relative, item.line, item.name),
        ):
            for field in POLICY_FIELDS:
                if re.search(rf"\b{re.escape(field)}\b", function.body):
                    errors.append(
                        f"{table_name} reaches {function.relative}:{function.line}:"
                        f"{function.name}, which reads {field}"
                    )
            if tracking:
                continue
            if function.name in TRACKING_FUNCTIONS or any(
                call in function.body for call in TRACKING_CALLS
            ):
                errors.append(
                    f"{table_name} reaches tracking work at "
                    f"{function.relative}:{function.line}:{function.name}"
                )

    return errors


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
