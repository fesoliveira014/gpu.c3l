#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "test/shaders/root_abi_audit.json"
OUTPUT_DIR = ROOT / "test/src/shaders"


def run(*args: str) -> str:
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def first_line(output: str) -> str:
    return output.splitlines()[0] if output else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild and validate the bounded root-ABI fixture audit.",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="validate existing generated fixtures without rebuilding them",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise SystemExit("unsupported root ABI audit manifest version")
    if not args.no_rebuild:
        run(sys.executable, "scripts/build_shaders.py")

    tools = {
        "glslc": first_line(run("glslc", "--version")),
        "spirv-as": first_line(run("spirv-as", "--version")),
        "spirv-dis": first_line(run("spirv-dis", "--version")),
        "spirv-val": first_line(run("spirv-val", "--version")),
    }
    records: list[dict[str, object]] = []
    for fixture in manifest["fixtures"]:
        source = ROOT / "test/shaders" / fixture["source"]
        output = OUTPUT_DIR / fixture["output"]
        if not source.is_file():
            raise SystemExit(f"missing audit source: {source.relative_to(ROOT)}")
        if not output.is_file():
            raise SystemExit(f"missing rebuilt fixture: {output.relative_to(ROOT)}")
        run(
            "spirv-val",
            "--target-env",
            manifest["target_env"],
            str(output),
        )
        disassembly = run("spirv-dis", "--raw-id", "-o", "-", str(output))
        records.append(
            {
                "id": fixture["id"],
                "source": fixture["source"],
                "output": fixture["output"],
                "expected_library_verdict": fixture["expected"],
                "isolates": fixture["isolates"],
                "spirv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "disassembly_sha256": hashlib.sha256(
                    disassembly.encode("utf-8"),
                ).hexdigest(),
                "spirv_validation": "pass",
            }
        )

    run(sys.executable, "scripts/check_shader_reflection_policy.py")
    report = {
        "schema_version": manifest["schema_version"],
        "target_env": manifest["target_env"],
        "tools": tools,
        "predicates": manifest["predicates"],
        "fixtures": records,
        "library_verdict_command":
            "c3c test vk_shader_reflection --path test",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
