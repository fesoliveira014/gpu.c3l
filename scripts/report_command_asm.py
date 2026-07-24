#!/usr/bin/env python3
"""Report broad generated-assembly observations for representative commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


EXPECTATION_VERSION = 1
OPERATIONS = {
    "dispatch": "cmd_dispatch",
    "draw": "cmd_draw",
    "barrier": "cmd_barrier",
    "viewport": "cmd_set_viewport",
    "copy_buffer": "cmd_copy_buffer",
}
NATIVE_OPERATIONS = {
    "dispatch": "trusted_dispatch",
    "draw": "trusted_draw",
    "barrier": "trusted_barrier",
    "viewport": "trusted_set_viewport",
    "copy_buffer": "trusted_copy_buffer",
}
FUNCTION_TYPE = re.compile(
    r"^\s*\.type\s+(?P<symbol>[^,\s]+)\s*,\s*[@%]function\s*$"
)
LABEL = re.compile(
    r"^(?P<symbol>[A-Za-z_.$][A-Za-z0-9_.$:]*)\s*:\s*(?:[#;].*)?$"
)
SIZE = re.compile(r"^\s*\.size\s+(?P<symbol>[^,\s]+)\s*,")
INSTRUCTION = re.compile(
    r"^\s*(?P<mnemonic>[A-Za-z][A-Za-z0-9_.]*)"
    r"(?:\s+(?P<operands>[^#;]*?))?\s*(?:[#;].*)?$"
)
INDIRECT_CALL_REGISTER = re.compile(
    r"^%?(?:"
    r"r(?:[0-9]+|[abcd]x|[sb]p|[sd]i)"
    r"|e(?:[abcd]x|[sb]p|[sd]i)"
    r"|[abcd][lh]"
    r"|[sb]pl|[sd]il"
    r"|[wx](?:[0-9]+|zr)"
    r")$",
    re.IGNORECASE,
)
KNOWN_PREFIXES = (
    "add", "and", "b", "call", "cmp", "dec", "inc", "j", "lea", "load",
    "mov", "nop", "or", "pop", "push", "ret", "set", "sh", "store", "sub",
    "test", "xor", "xchg", "cmpxchg", "lock",
)


@dataclass(frozen=True)
class Observation:
    symbol: str
    instructions: int
    calls: int
    indirect_calls: int
    atomics: int
    branches: int
    loads: int
    stores: int
    native_dispatch: int
    unknown: int


def function_bodies(source: str) -> dict[str, list[str]]:
    bodies: dict[str, list[str]] = {}
    pending: str | None = None
    active: str | None = None
    for line in source.splitlines():
        function_type = FUNCTION_TYPE.match(line)
        if function_type is not None:
            pending = function_type.group("symbol")
            continue
        label = LABEL.match(line)
        if (
            label is not None
            and pending is not None
            and label.group("symbol") == pending
        ):
            active = label.group("symbol")
            bodies.setdefault(active, [])
            pending = None
            continue
        size = SIZE.match(line)
        if size is not None and size.group("symbol") == active:
            active = None
            continue
        if active is not None:
            bodies[active].append(line)
    return bodies


def memory_direction(mnemonic: str, operands: str) -> tuple[int, int]:
    if not mnemonic.startswith(("mov", "load", "store")):
        return 0, 0
    parts = [part.strip() for part in operands.split(",")]
    memory = lambda value: "(" in value or "[" in value
    if mnemonic.startswith("load"):
        return 1, 0
    if mnemonic.startswith("store"):
        return 0, 1
    if len(parts) >= 2:
        return (1, 0) if memory(parts[0]) else ((0, 1) if memory(parts[-1]) else (0, 0))
    return (1, 0) if parts and memory(parts[0]) else (0, 0)


def is_indirect_call_operand(operands: str) -> bool:
    value = operands.strip()
    return (
        value.startswith("*")
        or INDIRECT_CALL_REGISTER.fullmatch(value) is not None
    )


def observe(symbol: str, lines: list[str]) -> Observation:
    counts = {
        "instructions": 0,
        "calls": 0,
        "indirect_calls": 0,
        "atomics": 0,
        "branches": 0,
        "loads": 0,
        "stores": 0,
        "native_dispatch": 0,
        "unknown": 0,
    }
    for line in lines:
        instruction = INSTRUCTION.match(line)
        if instruction is None or line.lstrip().startswith("."):
            continue
        mnemonic = instruction.group("mnemonic").lower()
        operands = (instruction.group("operands") or "").strip()
        counts["instructions"] += 1
        is_call = mnemonic.startswith(("call", "bl"))
        if is_call:
            counts["calls"] += 1
            if is_indirect_call_operand(operands):
                counts["indirect_calls"] += 1
            if re.search(r"\bvk(?:_|::)?cmd", operands, re.IGNORECASE):
                counts["native_dispatch"] += 1
        if mnemonic.startswith(("lock", "xchg", "cmpxchg", "atomic")):
            counts["atomics"] += 1
        if mnemonic.startswith("j") or mnemonic in {"b", "br", "cbz", "cbnz"}:
            counts["branches"] += 1
        loads, stores = memory_direction(mnemonic, operands)
        counts["loads"] += loads
        counts["stores"] += stores
        if not mnemonic.startswith(KNOWN_PREFIXES):
            counts["unknown"] += 1
    return Observation(symbol=symbol, **counts)


def collect(asm_dir: Path) -> dict[str, Observation | None]:
    bodies: dict[str, list[str]] = {}
    for path in sorted(asm_dir.rglob("*.s")):
        bodies.update(
            function_bodies(path.read_text(encoding="utf-8", errors="replace"))
        )
    observations: dict[str, Observation | None] = {}
    for operation, fragment in OPERATIONS.items():
        candidates = sorted(
            (symbol for symbol in bodies if fragment in symbol),
            key=lambda symbol: (len(symbol), symbol),
        )
        if not candidates:
            observations[operation] = None
            continue
        observation = observe(candidates[0], bodies[candidates[0]])
        native_fragment = NATIVE_OPERATIONS[operation]
        native_candidates = sorted(
            (
                symbol for symbol in bodies
                if symbol.startswith("gpu.internal.vk.")
                and native_fragment in symbol
            ),
            key=lambda symbol: (len(symbol), symbol),
        )
        native_dispatch = (
            observe(
                native_candidates[0],
                bodies[native_candidates[0]],
            ).native_dispatch
            if native_candidates else observation.native_dispatch
        )
        observations[operation] = replace(
            observation,
            native_dispatch=native_dispatch,
        )
    return observations


def validate_limits(
    observations: dict[str, Observation | None],
    limits: dict[str, dict[str, int | dict[str, int]]],
) -> list[str]:
    failures = []
    for operation in OPERATIONS:
        observation = observations[operation]
        if observation is None:
            failures.append(f"{operation}: representative symbol is missing")
            continue
        for field, rule in limits.get(operation, {}).items():
            if not hasattr(observation, field):
                failures.append(f"{operation}: unknown limit field {field}")
                continue
            actual = getattr(observation, field)
            if isinstance(rule, int):
                minimum = None
                maximum = rule
            else:
                minimum = rule.get("minimum")
                maximum = rule.get("maximum")
            if minimum is not None and actual < minimum:
                failures.append(
                    f"{operation}: {field} {actual} is below {minimum}"
                )
            if maximum is None:
                continue
            if actual > maximum:
                failures.append(
                    f"{operation}: {field} {actual} exceeds {maximum}"
                )
    return failures


def parse_compiler_version(output: str) -> str:
    match = re.search(
        r"^C3 Compiler Version:\s*"
        r"(?P<version>[0-9]+\.[0-9]+\.[0-9]+)"
        r"(?:_[0-9]+)?(?:\s.*)?$",
        output,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError("could not determine the C3 compiler version")
    return match.group("version")


def compiler_version() -> str:
    result = subprocess.run(
        ("c3c", "--version"),
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_compiler_version(result.stdout)


def validate_profile(
    profile: dict,
    pinned_compiler: str,
    pinned_target: str,
    comparison_profile: str,
) -> list[str]:
    identity = profile.get("identity", {})
    expected = {
        "compiler": pinned_compiler,
        "target": pinned_target,
        "comparison_profile": comparison_profile,
        "optimization": "O1",
    }
    failures = []
    if profile.get("expectation_version") != EXPECTATION_VERSION:
        failures.append(
            "profile expectation_version does not match reporter"
        )
    for field, value in expected.items():
        if identity.get(field) != value:
            failures.append(
                f"profile identity {field} {identity.get(field)!r} != {value!r}"
            )
    if not isinstance(profile.get("limits"), dict):
        failures.append("profile limits are missing or malformed")
        return failures
    required_fields = {
        "instructions",
        "calls",
        "indirect_calls",
        "atomics",
        "branches",
        "loads",
        "stores",
        "native_dispatch",
    }
    for operation in OPERATIONS:
        operation_limits = profile["limits"].get(operation)
        if not isinstance(operation_limits, dict):
            failures.append(f"profile limits for {operation} are missing")
            continue
        missing = sorted(required_fields - operation_limits.keys())
        if missing:
            failures.append(
                f"profile limits for {operation} are missing "
                + ", ".join(missing)
            )
    return failures


def emit_assembly(
    root: Path,
    asm_dir: Path,
    target: str | None = None,
) -> None:
    command = [
        "c3c",
        "build",
        "command_path_baseline_bench",
        "--path",
        "test",
        "-O1",
        "--emit-asm",
        "--asm-out",
        str(asm_dir),
    ]
    if target is not None:
        command.extend(("--target", target))
    subprocess.run(
        tuple(command),
        cwd=root,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asm-dir", type=Path)
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--pinned-compiler")
    parser.add_argument("--pinned-target")
    parser.add_argument("--comparison-profile")
    parser.add_argument("--limits", type=Path)
    args = parser.parse_args()
    pinned_fields = (
        args.pinned_compiler,
        args.pinned_target,
        args.comparison_profile,
    )
    if any(pinned_fields) and not all(pinned_fields):
        parser.error(
            "--pinned-compiler, --pinned-target, and --comparison-profile "
            "must be supplied together"
        )
    pinned = all(pinned_fields)
    if pinned != (args.limits is not None):
        parser.error("pinned assembly enforcement requires --limits")
    if args.asm_dir is None and not args.emit:
        parser.error("supply --asm-dir or --emit")
    if pinned and not args.emit:
        parser.error("pinned assembly enforcement requires --emit")

    root = Path(__file__).resolve().parents[1]
    profile = None
    if pinned:
        actual_compiler = compiler_version()
        if actual_compiler != args.pinned_compiler:
            parser.error(
                f"c3c version {actual_compiler} does not match "
                f"--pinned-compiler {args.pinned_compiler}"
            )
        profile = json.loads(args.limits.read_text(encoding="utf-8"))
        profile_failures = validate_profile(
            profile,
            args.pinned_compiler,
            args.pinned_target,
            args.comparison_profile,
        )
        if profile_failures:
            parser.error("; ".join(profile_failures))
    temporary = None
    asm_dir = args.asm_dir
    if asm_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="gpu-command-asm-")
        asm_dir = Path(temporary.name)
    if args.emit:
        asm_dir.mkdir(parents=True, exist_ok=True)
        emit_assembly(
            root,
            asm_dir,
            args.pinned_target if pinned else None,
        )

    observations = collect(asm_dir)
    mode = "blocking" if pinned else "advisory"
    print(
        f"asm_expectation version={EXPECTATION_VERSION} mode={mode} "
        "representation=bounded"
    )
    if pinned:
        print(
            f"asm_identity compiler={args.pinned_compiler} "
            f"target={args.pinned_target} "
            f"profile={args.comparison_profile} optimization=O1"
        )
    for operation, observation in observations.items():
        if observation is None:
            print(f"asm operation={operation} status=missing advisory=true")
            continue
        values = " ".join(
            f"{field}={getattr(observation, field)}"
            for field in (
                "instructions",
                "calls",
                "indirect_calls",
                "atomics",
                "branches",
                "loads",
                "stores",
                "native_dispatch",
                "unknown",
            )
        )
        print(
            f"asm operation={operation} symbol={observation.symbol} "
            f"{values} status=observed"
        )

    if not pinned:
        return 0
    assert profile is not None
    failures = validate_limits(observations, profile["limits"])
    if failures:
        for failure in failures:
            print(
                f"assembly limit failed (expectation {EXPECTATION_VERSION}): "
                f"{failure}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
