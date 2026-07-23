#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "TRUSTED_COMMAND_OPS": False,
    "TRUSTED_TRACKING_COMMAND_OPS": True,
    "CHECKED_COMMAND_OPS": False,
    "CHECKED_TRACKING_COMMAND_OPS": True,
}
# TABLES enumerates every command-policy table; POLICY_FIELDS covers backend
# policy reads; SUPERSEDED_COMMAND_FUNCTIONS records removed dispatch roots.
# Update the matching inventory whenever any of those sets changes.
POLICY_FIELDS = (
    "validation_policy",
    "track_lifetimes",
    "vulkan_layers",
    "debug_callback",
    "debug_names",
)
# Delivery reads are not policy selection; keep each exception narrow and named.
POLICY_FIELD_ALLOWED_READS = frozenset((
    ("report_contract_failure", "debug_callback"),
))
TRACKING_FUNCTIONS = frozenset((
    "track_command_reference",
    "resolve_tracked_texture_command_reference",
    "ensure_command_reference_capacity",
    "rollback_command_references",
    "note_command_reference_allocation",
))
TRACKING_CALLS = tuple(f"{name}(" for name in TRACKING_FUNCTIONS)
CORE_COMMAND_ROOTS = frozenset((
    "vk_create_command_allocator",
    "vk_destroy_command_allocator",
    "vk_reserve_generated_scratch",
    "vk_release_generated_scratch",
    "vk_begin_commands",
    "vk_end_commands",
    "vk_discard_commands",
    "vk_submit",
    "retire_observed_completion_and_drain_with_query",
    "drain_completed_submitted_commands",
))
ALLOCATION_FREE_ROOTS = frozenset((
    "vk_begin_commands",
    "vk_end_commands",
    "vk_discard_commands",
    "vk_submit",
    "retire_observed_completion_and_drain_with_query",
    "drain_completed_submitted_commands",
))
AMBIENT_COMMAND_PATTERNS = {
    "temporary pool": re.compile(r"@pool\s*\(|\bmem::talloc[A-Za-z0-9_]*\s*\("),
    "thread-local state": re.compile(r"\btlocal\b"),
    "recording context": re.compile(
        r"\b(?:ThreadRecordingContext|thread_recording_contexts|"
        r"RecordingContextTable|RecordingContextState|"
        r"RecordingContextHandle|MAX_RECORDING_CONTEXTS|"
        r"MAX_THREAD_DEVICE_CONTEXTS)\b"
    ),
}
WARM_ALLOCATION_PATTERNS = {
    "host allocation": re.compile(
        r"\b(?:mem|alloc)::(?:new[A-Za-z0-9_]*|realloc|"
        r"resize[A-Za-z0-9_]*|alloc)\s*\("
    ),
    "native command allocation": re.compile(
        r"\bvk::(?:allocate_command_buffers|create_command_pool)\s*\("
    ),
    "VMA allocation": re.compile(
        r"(?:\bvma::|\ballocator\.)"
        r"(?:allocate_memory|create_buffer(?:_with_alignment)?|create_image)\s*\("
    ),
}
WARM_STACK_PATTERNS = {
    "capacity-sized stack storage": re.compile(
        r"\[[^\]\r\n]*(?:MAX_SUBMIT_COMMAND_LISTS|"
        r"MAX_SUBMIT_COMPLETION_WAITS)[^\]\r\n]*\]"
    ),
}
REFERENCE_INDEX_HOT_FUNCTIONS = frozenset((
    "preflight_command_references",
    "track_command_reference",
    "resolve_tracked_texture_command_reference",
))
ACCUMULATED_REFERENCE_SCAN_PATTERNS = {
    "reference-count loop": re.compile(
        r"\bfor\s*\([^)]*\breference_count\b"
    ),
    "reference-slice loop": re.compile(
        r"\bforeach\s*\([^)]*\breferences\s*\["
    ),
}
REFERENCE_IDENTITY_FIELDS = ("owner", "index", "generation")
POST_RETAIN_PUBLICATION_MARKERS = {
    "track_command_reference": re.compile(
        r"\bretain_tracked_command_reference\s*\("
    ),
    "resolve_tracked_texture_command_reference": re.compile(
        r"\btracked_command_references\.add\s*\("
    ),
}
REFERENCE_PUBLICATION_CALLERS = frozenset((
    "track_command_reference",
    "resolve_tracked_texture_command_reference",
    "rollback_command_references",
))
ENCODER_PROOF_PATTERNS = {
    "stored device comparison": re.compile(r"\bencoder\.device\b"),
    "stored handle comparison": re.compile(r"\bencoder\.handle\b"),
    "device-loss load": re.compile(
        r"\b(?:device_lost|lost)\b[^;{}]*\bload\s*\("
    ),
    "operation-table null check": re.compile(
        r"\bencoder\.ops\s*(?:==|!=)\s*null"
    ),
    "backend-state null check": re.compile(
        r"\bencoder\.backend_state\s*(?:==|!=)\s*null"
    ),
    "backend-command null check": re.compile(
        r"\bencoder\.backend_command\s*(?:==|!=)\s*null"
    ),
}
FRONTEND_ENCODER_ROOTS = frozenset((
    "command_encoder",
    "recording_encoder",
    "executable_encoder",
    "command_operation",
    "executable_command_operation",
))
TRUSTED_CAPABILITY_PATTERNS = {
    "encoder null check": re.compile(r"\bcommands\s*(?:==|!=)\s*null"),
    "command-record null check": re.compile(r"\brecord\s*(?:==|!=)\s*null"),
    "backend-state null check": re.compile(
        r"\bcommands\.backend_state\s*(?:==|!=)\s*null"
    ),
    "backend-command null check": re.compile(
        r"\bcommands\.backend_command\s*(?:==|!=)\s*null"
    ),
}
COLD_ALLOCATION_FUNCTIONS = frozenset((
    "allocate_command_buffers_real",
    "allocate_generated_preprocess_buffer",
    "create_command_pool_real",
    "initialize_command_allocator_slot",
    "initialize_command_allocator_slot_with_ops",
))
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
    r"(?m)^const\s+gpu::internal::CommandOps\s+([A-Za-z_][A-Za-z0-9_]*)\b[^=]*=\s*\{",
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
    backend = root / "gpu" / "internal" / "vk"
    for path in sorted(backend.rglob("*.c3"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        for function in source_functions(relative, source):
            functions.setdefault(function.name, []).append(function)
    return functions


def command_tables(root: Path) -> dict[str, dict[str, str]]:
    path = root / "gpu" / "internal" / "vk" / "device.c3"
    source = path.read_text(encoding="utf-8")
    masked = mask_non_code(source)
    tables: dict[str, dict[str, str]] = {}
    for declaration in TABLE_DECLARATION.finditer(masked):
        name = declaration.group(1)
        if name not in TABLES:
            raise ValueError(f"unexpected command policy table: {name}")
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

    for function in sorted(
        reachable_functions(functions, set(REFERENCE_INDEX_HOT_FUNCTIONS)),
        key=lambda item: (item.relative, item.line, item.name),
    ):
        for label, pattern in ACCUMULATED_REFERENCE_SCAN_PATTERNS.items():
            if pattern.search(function.body):
                errors.append(
                    "reference index hot path contains " + label + " at "
                    f"{function.relative}:{function.line}:{function.name}"
                )

    for function in functions.get("indexed_resource_reference_matches", ()):
        for field in REFERENCE_IDENTITY_FIELDS:
            exact_comparison = re.compile(
                rf"\bcell\.{field}\s*==\s*reference\.{field}\b"
            )
            if exact_comparison.search(function.body) is None:
                errors.append(
                    "reference index equality omits exact " + field + " at "
                    f"{function.relative}:{function.line}:{function.name}"
                )

    for function_name, retain_pattern in POST_RETAIN_PUBLICATION_MARKERS.items():
        for function in functions.get(function_name, ()):
            retained = retain_pattern.search(function.body)
            published = re.search(
                r"\bpublish_command_reference\s*\(",
                function.body,
            )
            if (retained is None or published is None
                    or published.start() < retained.end()):
                errors.append(
                    "reference index publication is not after retain at "
                    f"{function.relative}:{function.line}:{function.name}"
                )

    for declarations in functions.values():
        for function in declarations:
            if (function.name in REFERENCE_PUBLICATION_CALLERS):
                continue
            if re.search(
                r"\bpublish_command_reference\s*\(",
                function.body,
            ) is not None:
                errors.append(
                    "reference index publication has unauthorized caller at "
                    f"{function.relative}:{function.line}:{function.name}"
                )

    rollback_functions = sorted(
        reachable_functions(functions, {"rollback_command_references"}),
        key=lambda item: (item.relative, item.line, item.name),
    )
    for function in functions.get("rollback_command_references", ()):
        reset = re.search(
            r"\breset_command_reference_index\s*\(",
            function.body,
        )
        rebuilt = re.search(
            r"\bpublish_command_reference\s*\(",
            function.body,
        )
        if (reset is None or rebuilt is None or rebuilt.start() < reset.end()
                ):
            errors.append(
                "reference rollback must reset and rebuild the retained prefix at "
                f"{function.relative}:{function.line}:{function.name}"
            )
    for function in rollback_functions:
        if re.search(
            r"\breference_index\.cells\s*\[[^]]+\]\s*\.epoch\s*=",
            function.body,
        ) is not None:
            errors.append(
                "reference rollback reaches unsafe cell deletion at "
                f"{function.relative}:{function.line}:{function.name}"
            )

    expected_fields = set(tables["TRUSTED_COMMAND_OPS"])
    if not expected_fields:
        errors.append("TRUSTED_COMMAND_OPS has no command entries")
        return errors
    for table_name, entries in sorted(tables.items()):
        fields = set(entries)
        if fields != expected_fields:
            errors.append(
                f"{table_name} command fields differ from TRUSTED_COMMAND_OPS"
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
                if (function.name, field) in POLICY_FIELD_ALLOWED_READS:
                    continue
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

    trusted_roots = set(tables["TRUSTED_COMMAND_OPS"].values())
    trusted_roots.update(tables["TRUSTED_TRACKING_COMMAND_OPS"].values())
    for function in sorted(
        reachable_functions(functions, trusted_roots),
        key=lambda item: (item.relative, item.line, item.name),
    ):
        for label, pattern in TRUSTED_CAPABILITY_PATTERNS.items():
            if pattern.search(function.body):
                errors.append(
                    "trusted command path reaches " + label + " at "
                    f"{function.relative}:{function.line}:{function.name}"
                )

    encoder_path = root / "gpu" / "internal" / "command.c3"
    public_path = root / "gpu" / "gpu.c3"
    if encoder_path.exists():
        try:
            encoder_source = encoder_path.read_text(encoding="utf-8")
            encoder_functions = source_functions(
                "gpu/internal/command.c3",
                encoder_source,
            )
            public_functions = []
            if public_path.exists():
                public_functions = source_functions(
                    "gpu/gpu.c3",
                    public_path.read_text(encoding="utf-8"),
                )
        except (OSError, ValueError) as error:
            errors.append(str(error))
            encoder_functions = []
            public_functions = []
        frontend_functions: dict[str, list[Function]] = {}
        for function in encoder_functions + public_functions:
            frontend_functions.setdefault(function.name, []).append(function)
        frontend_roots = set(FRONTEND_ENCODER_ROOTS)
        frontend_roots.update(
            function.name for function in public_functions
            if function.name.startswith("cmd_")
        )
        for function in sorted(
            reachable_functions(frontend_functions, frontend_roots),
            key=lambda item: (item.relative, item.line, item.name),
        ):
            for label, pattern in ENCODER_PROOF_PATTERNS.items():
                if pattern.search(function.body):
                    errors.append(
                        "frontend command resolution performs " + label + " at "
                        f"{function.relative}:{function.line}:{function.name}"
                    )
        command_encoders = [
            function for function in encoder_functions
            if function.name == "command_encoder"
        ]
        if len(command_encoders) != 1:
            errors.append(
                "command_encoder proof root is missing or duplicated"
            )
        else:
            encoder = command_encoders[0]
            proof_notes = (
                "note_command_encoder_cell_computation(",
                "note_command_encoder_lease_comparison(",
            )
            for note in proof_notes:
                if encoder.body.count(note) != 1:
                    errors.append(
                        "command_encoder must record exactly one "
                        + note.removesuffix("(")
                    )

    missing_roots = sorted(CORE_COMMAND_ROOTS - set(functions))
    if missing_roots:
        errors.append(
            "missing command-path roots: " + ", ".join(missing_roots)
        )
        return errors

    command_roots = set(CORE_COMMAND_ROOTS)
    for entries in tables.values():
        command_roots.update(entries.values())
    for function in sorted(
        reachable_functions(functions, command_roots),
        key=lambda item: (item.relative, item.line, item.name),
    ):
        for label, pattern in AMBIENT_COMMAND_PATTERNS.items():
            if pattern.search(function.body):
                errors.append(
                    "command path reaches " + label + " at "
                    f"{function.relative}:{function.line}:{function.name}"
                )

    allocation_free_roots = set(ALLOCATION_FREE_ROOTS)
    for entries in tables.values():
        allocation_free_roots.update(entries.values())
    for function in sorted(
        reachable_functions(functions, allocation_free_roots),
        key=lambda item: (item.relative, item.line, item.name),
    ):
        if function.name in COLD_ALLOCATION_FUNCTIONS:
            errors.append(
                "warm command path reaches cold allocation helper at "
                f"{function.relative}:{function.line}:{function.name}"
            )
        for label, pattern in WARM_ALLOCATION_PATTERNS.items():
            if pattern.search(function.body):
                errors.append(
                    "warm command path reaches " + label + " at "
                    f"{function.relative}:{function.line}:{function.name}"
                )
        for label, pattern in WARM_STACK_PATTERNS.items():
            if pattern.search(function.body):
                errors.append(
                    "warm command path reaches " + label + " at "
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
