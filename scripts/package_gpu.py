from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable


SUPPORTED_TARGETS = {"linux-x64", "windows-x64"}
TARGET_LINK_CONTRACT = {
    "linux-x64": [":libvulkan.so.1", "VulkanMemoryAllocator", "spvreflect", "stdc++"],
    "windows-x64": ["vulkan-1", "VulkanMemoryAllocator", "spvreflect"],
}
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
TOOLCHAIN_KEYS = {"c3c", "platform", "compiler", "compiler-version", "vulkan-sdk"}


class PackageError(RuntimeError):
    pass


def duplicate_key_guard(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise PackageError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=duplicate_key_guard)
    except (OSError, json.JSONDecodeError) as error:
        raise PackageError(f"cannot read {path}: {error}") from error


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_canonical_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PackageError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def canonical_text_bytes(path: Path) -> bytes:
    try:
        return path.read_text(encoding="utf-8").encode("utf-8")
    except (OSError, UnicodeError) as error:
        raise PackageError(f"cannot read text input {path}: {error}") from error


def sha256_text(path: Path) -> str:
    return sha256_bytes(canonical_text_bytes(path))


def canonical_shell_command(path: Path) -> list[str]:
    return ["sh", "-c", canonical_text_bytes(path).decode("utf-8"), str(path)]


def normalize_relative(path: str) -> str:
    if not isinstance(path, str) or not path or "\\" in path:
        raise PackageError(f"noncanonical relative path: {path!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts or str(pure) != path:
        raise PackageError(f"path escapes package root or is noncanonical: {path!r}")
    return path


def reject_glob(path: str) -> None:
    if any(character in path for character in "*?["):
        raise PackageError(f"source glob is forbidden: {path}")


def validate_mapping(mapping: dict, root: Path, destinations: set[str]) -> None:
    if set(mapping) != {"source", "destination"}:
        raise PackageError(f"mapping must contain source and destination: {mapping!r}")
    source = normalize_relative(mapping["source"])
    destination = normalize_relative(mapping["destination"])
    reject_glob(source)
    if destination in destinations:
        raise PackageError(f"duplicate destination: {destination}")
    destinations.add(destination)
    if not (root / source).is_file():
        raise PackageError(f"missing package input: {source}")


def discovered_binding_sources(root: Path, binding: dict) -> set[str]:
    binding_path = normalize_relative(binding["path"])
    suffixes = set(binding.get("production-suffixes", []))
    directory = root / binding_path
    return {
        path.relative_to(root).as_posix()
        for path in directory.iterdir()
        if path.is_file() and path.suffix in suffixes
    }


def validate_recipe(recipe: dict, root: Path, check_discovery: bool = True) -> None:
    if recipe.get("format") != 1 or recipe.get("c3-version") != "0.8.0":
        raise PackageError("unsupported package recipe format or C3 version")
    targets = recipe.get("targets", {})
    unsupported = set(targets) - SUPPORTED_TARGETS
    missing = SUPPORTED_TARGETS - set(targets)
    if unsupported:
        raise PackageError(f"unsupported target in recipe: {sorted(unsupported)}")
    if missing:
        raise PackageError(f"missing supported target in recipe: {sorted(missing)}")
    for tool_input in recipe.get("tool-inputs", []):
        path = normalize_relative(tool_input)
        reject_glob(path)
        if not (root / path).is_file():
            raise PackageError(f"missing package tool input: {path}")
    destinations: set[str] = set()
    for mapping in recipe.get("sources", []) + recipe.get("assets", []) + recipe.get("licenses", []):
        validate_mapping(mapping, root, destinations)
    allowed_sources = {mapping["source"] for mapping in recipe.get("sources", [])}
    if check_discovery:
        discovered_gpu = {
            path.relative_to(root).as_posix()
            for path in (root / "gpu").rglob("*")
            if path.is_file() and path.suffix in {".c3", ".c3i"}
        }
        allowed_gpu = {source for source in allowed_sources if source.startswith("gpu/")}
        unexpected_gpu = discovered_gpu - allowed_gpu
        if unexpected_gpu:
            raise PackageError(f"unlisted GPU production source: {sorted(unexpected_gpu)}")
    for name, binding in recipe.get("bindings", {}).items():
        normalize_relative(binding["path"])
        if check_discovery:
            discovered = discovered_binding_sources(root, binding)
            allowed = {source for source in allowed_sources if source.startswith(binding["path"] + "/")}
            unexpected = discovered - allowed
            missing_allowed = allowed - discovered
            if unexpected:
                raise PackageError(f"unlisted production source in {name}: {sorted(unexpected)}")
            if missing_allowed:
                raise PackageError(f"allowed production source missing in {name}: {sorted(missing_allowed)}")
    for target, target_recipe in targets.items():
        native_destinations: set[str] = set()
        for mapping in target_recipe.get("native", []):
            validate_mapping(mapping, root, native_destinations)
        generated = target_recipe.get("generated-native")
        if generated:
            source = normalize_relative(generated.get("source", ""))
            destination = normalize_relative(generated.get("destination", ""))
            reject_glob(source)
            if destination in native_destinations:
                raise PackageError(f"duplicate destination: {destination}")
    build = recipe.get("windows-vma-build", {})
    if not HEX_40.fullmatch(build.get("upstream-commit", "")):
        raise PackageError("invalid Windows VMA upstream commit")
    if not HEX_64.fullmatch(build.get("upstream-header-sha256", "")):
        raise PackageError("invalid Windows VMA header hash")
    for key in ("wrapper-source", "size-probe-source", "build-script"):
        path = normalize_relative(build.get(key, ""))
        reject_glob(path)
        if not (root / path).is_file():
            raise PackageError(f"missing Windows VMA build input: {path}")


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip().lower()
    if result.returncode != 0 or not HEX_40.fullmatch(commit):
        raise PackageError(f"cannot determine binding commit for {path}: {result.stderr.strip()}")
    return commit


def hash_mappings(root: Path, mappings: Iterable[dict], text: bool = False) -> list[dict]:
    hash_file = sha256_text if text else sha256
    return [
        {
            "source": mapping["source"],
            "destination": mapping["destination"],
            "sha256": hash_file(root / mapping["source"]),
        }
        for mapping in sorted(mappings, key=lambda item: item["destination"])
    ]


def build_lock(root: Path, recipe: dict | None = None) -> dict:
    recipe = recipe or load_json(root / "packaging" / "package.json")
    validate_recipe(recipe, root)
    bindings = {
        name: {"path": binding["path"], "commit": git_commit(root / binding["path"])}
        for name, binding in sorted(recipe["bindings"].items())
    }
    committed_native: dict[str, list[dict]] = {}
    for target, target_recipe in sorted(recipe["targets"].items()):
        committed_native[target] = hash_mappings(root, target_recipe.get("native", []))
    build = recipe["windows-vma-build"]
    build_inputs = {
        key: {"path": build[key], "sha256": sha256_text(root / build[key])}
        for key in ("build-script", "size-probe-source", "wrapper-source")
    }
    tool_inputs = [
        {"path": path, "sha256": sha256_text(root / path)}
        for path in sorted(recipe.get("tool-inputs", []))
    ]
    return {
        "format": 1,
        "c3-version": recipe["c3-version"],
        "recipe-sha256": sha256_bytes(canonical_json_bytes(recipe)),
        "tool-inputs": tool_inputs,
        "bindings": bindings,
        "sources": hash_mappings(root, recipe["sources"], text=True),
        "assets": hash_mappings(root, recipe["assets"], text=True),
        "licenses": hash_mappings(root, recipe["licenses"], text=True),
        "committed-native": committed_native,
        "windows-vma": {
            "upstream-commit": build["upstream-commit"],
            "upstream-header-sha256": build["upstream-header-sha256"],
            "vulkan-sdk-version": build["vulkan-sdk-version"],
            "inputs": build_inputs,
        },
    }


def check_lock(root: Path, lock: dict | None = None) -> dict:
    lock = lock or load_json(root / "packaging" / "package-lock.json")
    actual = build_lock(root)
    if lock != actual:
        raise PackageError("package lock differs from source, binding, native, or build inputs; refresh intentionally")
    return lock


def normalize_toolchain(toolchain: dict, target: str) -> dict:
    if set(toolchain) - TOOLCHAIN_KEYS:
        raise PackageError(f"toolchain provenance contains noncanonical fields: {sorted(set(toolchain) - TOOLCHAIN_KEYS)}")
    if toolchain.get("platform") != target or not toolchain.get("c3c"):
        raise PackageError("toolchain provenance must declare matching platform and c3c")
    normalized: dict[str, str] = {}
    for key in sorted(toolchain):
        value = toolchain[key]
        if not isinstance(value, str) or not value or value != value.strip() or "\n" in value or "\r" in value:
            raise PackageError(f"toolchain provenance is not normalized: {key}")
        if value.casefold() in {"unknown", "unspecified", "n/a"}:
            raise PackageError(f"toolchain provenance contains placeholder identity: {key}")
        normalized[key] = value
    if target == "windows-x64":
        required = {"compiler", "compiler-version", "vulkan-sdk"}
        if not required <= set(normalized):
            raise PackageError(f"Windows toolchain provenance missing: {sorted(required - set(normalized))}")
    return normalized


def validate_windows_toolchain(recipe: dict, toolchain: dict) -> None:
    expected = recipe["windows-vma-build"]["vulkan-sdk-version"]
    if toolchain.get("vulkan-sdk") != expected:
        raise PackageError(f"Windows Vulkan SDK identity must match locked version {expected}")


def vulkan_header_path(environment: dict[str, str]) -> Path:
    root = environment.get("VULKAN_HEADERS") or environment.get("VULKAN_SDK")
    if not root:
        raise PackageError("VULKAN_HEADERS or VULKAN_SDK must identify the locked Vulkan headers")
    header = Path(root) / "include" / "vulkan" / "vulkan_core.h"
    if not header.is_file():
        raise PackageError(f"Vulkan header is missing: {header}")
    return header


def validate_vulkan_header_identity(header: Path, expected_sdk: str) -> None:
    try:
        source = header.read_text(encoding="utf-8")
    except OSError as error:
        raise PackageError(f"cannot read Vulkan header {header}: {error}") from error
    header_version = re.search(r"^\s*#\s*define\s+VK_HEADER_VERSION\s+(\d+)\s*$", source, re.MULTILINE)
    complete = re.search(
        r"VK_HEADER_VERSION_COMPLETE\s+VK_MAKE_API_VERSION\(\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*VK_HEADER_VERSION\s*\)",
        source,
    )
    parts = expected_sdk.split(".")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        raise PackageError(f"locked Vulkan SDK identity is invalid: {expected_sdk}")
    expected_major, expected_minor, expected_header = map(int, parts[:3])
    if (
        header_version is None
        or complete is None
        or int(header_version.group(1)) != expected_header
        or int(complete.group(1)) != expected_major
        or int(complete.group(2)) != expected_minor
    ):
        raise PackageError(f"Vulkan header does not match locked SDK identity {expected_sdk}: {header}")


def derive_windows_toolchain(
    recipe: dict,
    c3c_version: str,
    environment: dict[str, str] | None = None,
    run_command=subprocess.run,
) -> dict:
    environment = environment or dict(os.environ)
    expected_sdk = recipe["windows-vma-build"]["vulkan-sdk-version"]
    validate_vulkan_header_identity(vulkan_header_path(environment), expected_sdk)
    try:
        result = run_command(["cl"], check=False, capture_output=True, text=True)
    except OSError as error:
        raise PackageError(f"cannot execute MSVC cl: {error}") from error
    output = result.stdout + result.stderr
    version = re.search(r"Compiler Version\s+([0-9]+(?:\.[0-9]+)+)", output, re.IGNORECASE)
    if version is None:
        raise PackageError("cannot derive the MSVC compiler version from cl output")
    return normalize_toolchain(
        {
            "platform": "windows-x64",
            "c3c": c3c_version,
            "compiler": "msvc",
            "compiler-version": version.group(1),
            "vulkan-sdk": expected_sdk,
        },
        "windows-x64",
    )


def validate_fixture_project(project: dict) -> None:
    if project.get("dependencies") != ["gpu"]:
        raise PackageError("consumer fixture must declare only gpu")
    if project.get("dependency-search-paths") != ["lib"]:
        raise PackageError("consumer fixture must use only its generated lib package root")
    if "wincrt" in project:
        raise PackageError("consumer fixture must not own the Windows CRT setting")
    serialized = json.dumps(project)
    for forbidden in ("vk", "vma", "spvreflect", "../gpu", "linked-libs", "link-args", "linked-libraries"):
        if forbidden in serialized:
            raise PackageError(f"consumer fixture leaks forbidden package detail: {forbidden}")


def check_fixture_policy(fixture: Path) -> None:
    validate_fixture_project(load_json(fixture / "project.json"))
    source = (fixture / "src" / "main.c3").read_text(encoding="utf-8")
    required = (
        "import gpu;",
        "$embed(\"shader.comp.spv\")",
        "gpu::create_device",
        "gpu::destroy_device",
        "gpu::create_shader",
        "gpu::destroy_shader",
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise PackageError(f"consumer fixture lacks lifecycle proof: {missing}")
    for forbidden in ("import vk", "import vma", "import spvreflect", "import gpu::vk"):
        if forbidden in source:
            raise PackageError(f"consumer fixture imports backend detail: {forbidden}")
    spirv_path = fixture / "src" / "shader.comp.spv"
    try:
        spirv = spirv_path.read_bytes()
    except OSError as error:
        raise PackageError(
            f"build the consumer fixture shader.comp.spv before policy validation: {spirv_path}: {error}"
        ) from error
    if len(spirv) < 20 or spirv[:4] != b"\x03\x02\x23\x07":
        raise PackageError("consumer fixture does not embed valid SPIR-V")
    shader_source = (fixture / "src" / "shader.comp.glsl").read_text(encoding="utf-8")
    if '#include "generated/shader_abi.glsl"' not in shader_source:
        raise PackageError("consumer fixture must compile against the packaged generated shader ABI")


def fixture_red(root: Path, c3c: str = "c3c") -> str:
    check_fixture_policy(root / "test" / "consumer")
    with tempfile.TemporaryDirectory(prefix="gpu-consumer-red-") as temp_dir:
        fixture = Path(temp_dir) / "consumer"
        shutil.copytree(root / "test" / "consumer", fixture, ignore=shutil.ignore_patterns("lib", "build"))
        development = fixture / "lib" / "gpu.c3l"
        development.mkdir(parents=True)
        shutil.copy2(root / "manifest.json", development / "manifest.json")
        shutil.copytree(root / "gpu", development / "gpu")
        result = subprocess.run(
            [c3c, "build", "consumer", "--path", str(fixture)],
            check=False,
            capture_output=True,
            text=True,
        )
    output = result.stdout + result.stderr
    expected = "Required library 'vk' could not be found"
    if result.returncode == 0 or expected not in output:
        raise PackageError(f"invalid RED baseline; expected {expected!r}, got:\n{output}")
    return expected


def package_manifest(recipe: dict, target: str) -> dict:
    manifest = {
        "provides": "gpu",
        "linklib-dir": "linked-libs",
        "sources": sorted(mapping["destination"] for mapping in recipe["sources"]),
        "targets": {
            target: {
                "link-args": [],
                "linked-libraries": recipe["targets"][target]["linked-libraries"],
            }
        },
    }
    if target == "windows-x64":
        manifest["wincrt"] = "dynamic"
    return manifest


def validate_package_manifest(manifest: dict, target: str) -> None:
    if manifest.get("provides") != "gpu" or "dependencies" in json.dumps(manifest):
        raise PackageError("generated C3 manifest leaks package dependencies")
    if manifest.get("linklib-dir") != "linked-libs" or set(manifest.get("targets", {})) != {target}:
        raise PackageError("generated C3 manifest target link metadata is invalid")
    target_manifest = manifest["targets"][target]
    if target_manifest.get("link-args") != [] or target_manifest.get("linked-libraries") != TARGET_LINK_CONTRACT[target]:
        raise PackageError("generated C3 manifest link contract is invalid")
    sources = manifest.get("sources", [])
    if not sources or sources != sorted(sources) or len(sources) != len(set(sources)):
        raise PackageError("generated C3 manifest source closure is invalid")
    for source in sources:
        normalize_relative(source)
    if target == "windows-x64" and manifest.get("wincrt") != "dynamic":
        raise PackageError("Windows package must select the dynamic CRT")
    if target == "linux-x64" and "wincrt" in manifest:
        raise PackageError("Linux package must not declare a Windows CRT")


def runtime_manifest(recipe: dict, target: str) -> dict:
    target_runtime = recipe["targets"][target]["runtime"]
    return {
        "format": 1,
        "target": target,
        "package-owned-runtime-files": target_runtime["package-owned-runtime-files"],
        "system-prerequisites": target_runtime["system-prerequisites"],
    }


def copy_mapping(root: Path, bundle: Path, mapping: dict, text: bool = False) -> None:
    destination = bundle.joinpath(*PurePosixPath(mapping["destination"]).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if text:
        destination.write_bytes(canonical_text_bytes(root / mapping["source"]))
    else:
        shutil.copy2(root / mapping["source"], destination)


def payload_entries(bundle: Path) -> list[dict]:
    excluded = bundle / "artifact-manifest.json"
    files = sorted(
        (path for path in bundle.rglob("*") if path.is_file() and path != excluded),
        key=lambda path: path.relative_to(bundle).as_posix(),
    )
    return [
        {"path": path.relative_to(bundle).as_posix(), "sha256": sha256(path)}
        for path in files
    ]


def payload_digest(entries: list[dict]) -> str:
    encoded = "".join(f"{entry['path']}\0{entry['sha256']}\n" for entry in entries).encode("utf-8")
    return sha256_bytes(encoded)


def validate_artifact_manifest_shape(manifest: dict) -> None:
    required = {"format", "target", "locked-input-digest", "toolchain", "payload", "payload-digest"}
    allowed = required | {"generated-artifacts"}
    if set(manifest) - allowed or not required <= set(manifest):
        raise PackageError("artifact manifest has missing or unknown fields")
    if manifest["format"] != 1 or manifest["target"] not in SUPPORTED_TARGETS:
        raise PackageError("artifact manifest format or target is invalid")
    if not HEX_64.fullmatch(manifest.get("locked-input-digest", "")):
        raise PackageError("artifact manifest locked input digest is invalid")
    normalize_toolchain(manifest.get("toolchain", {}), manifest["target"])
    seen: set[str] = set()
    previous = ""
    for entry in manifest.get("payload", []):
        if set(entry) != {"path", "sha256"}:
            raise PackageError("payload entry must contain path and hash")
        path = normalize_relative(entry["path"])
        if path == "artifact-manifest.json":
            raise PackageError("artifact manifest must not hash itself")
        if path in seen or path <= previous:
            raise PackageError("duplicate or unsorted payload path")
        if not HEX_64.fullmatch(entry["sha256"]):
            raise PackageError(f"missing or invalid payload hash: {path}")
        seen.add(path)
        previous = path
    if not HEX_64.fullmatch(manifest.get("payload-digest", "")):
        raise PackageError("payload digest is invalid")
    payload_hashes = {entry["path"]: entry["sha256"] for entry in manifest.get("payload", [])}
    generated_paths: list[str] = []
    for generated in manifest.get("generated-artifacts", []):
        if set(generated) != {"path", "sha256"}:
            raise PackageError("generated artifact record is noncanonical")
        generated_path = normalize_relative(generated["path"])
        if (
            generated_path not in seen
            or not HEX_64.fullmatch(generated["sha256"])
            or payload_hashes.get(generated_path) != generated["sha256"]
        ):
            raise PackageError("generated artifact record does not reference a payload hash")
        generated_paths.append(generated_path)
    if generated_paths != sorted(set(generated_paths)):
        raise PackageError("generated artifact records are duplicate or unsorted")


def load_artifact_manifest(path: Path) -> dict:
    raw = path.read_bytes()
    manifest = load_json(path)
    if raw != canonical_json_bytes(manifest):
        raise PackageError("artifact manifest JSON is not canonical")
    validate_artifact_manifest_shape(manifest)
    return manifest


def verify_bundle(bundle: Path, target: str, expected_lock: Path | None = None) -> dict:
    manifest = load_artifact_manifest(bundle / "artifact-manifest.json")
    if manifest["target"] != target:
        raise PackageError(f"bundle target {manifest['target']} does not match {target}")
    expected = manifest["payload"]
    actual = payload_entries(bundle)
    expected_paths = [entry["path"] for entry in expected]
    actual_paths = [entry["path"] for entry in actual]
    if expected_paths != actual_paths:
        raise PackageError(f"payload set differs: expected {expected_paths}, got {actual_paths}")
    for expected_entry, actual_entry in zip(expected, actual):
        if expected_entry["sha256"] != actual_entry["sha256"]:
            raise PackageError(f"payload hash differs: {expected_entry['path']}")
    if payload_digest(actual) != manifest["payload-digest"]:
        raise PackageError("aggregate payload digest differs")
    lock_path = bundle / "package-lock.json"
    lock_bytes = lock_path.read_bytes()
    lock = load_json(lock_path)
    if lock_bytes != canonical_json_bytes(lock):
        raise PackageError("bundled package lock is not canonical")
    if sha256_bytes(lock_bytes) != manifest["locked-input-digest"]:
        raise PackageError("artifact locked input digest does not match bundled package lock")
    if expected_lock is not None:
        expected_bytes = expected_lock.read_bytes()
        expected = load_json(expected_lock)
        if expected_bytes != canonical_json_bytes(expected):
            raise PackageError("checked-in package lock is not canonical")
        if lock_bytes != expected_bytes:
            raise PackageError("bundle does not match the current checked-in package lock")
    package = load_json(bundle / "manifest.json")
    validate_package_manifest(package, target)
    runtime = load_json(bundle / "runtime.json")
    if runtime.get("target") != target:
        raise PackageError("runtime manifest target differs")
    generated_by_path = {item["path"]: item["sha256"] for item in manifest.get("generated-artifacts", [])}
    if target == "windows-x64":
        vma = "linked-libs/windows-x64/VulkanMemoryAllocator.lib"
        payload_by_path = {item["path"]: item["sha256"] for item in actual}
        if generated_by_path.get(vma) != payload_by_path.get(vma):
            raise PackageError("Windows VMA generated provenance is missing")
    return manifest


def verify_vma_header(root: Path, recipe: dict) -> None:
    include = os.environ.get("VMA_INCLUDE")
    if not include:
        raise PackageError("VMA_INCLUDE must identify the locked VMA header root")
    header = Path(include) / "vma" / "vk_mem_alloc.h"
    expected = recipe["windows-vma-build"]["upstream-header-sha256"]
    if not header.is_file() or sha256(header) != expected:
        raise PackageError(f"VMA header is missing or differs from locked upstream commit: {header}")


def validate_vma_directives(output: str, recipe: dict) -> None:
    normalized = output.upper().replace('"', "")
    build = recipe["windows-vma-build"]
    for forbidden in build["forbidden-directives"]:
        if forbidden.upper() in normalized:
            raise PackageError(f"VMA archive advertises forbidden CRT directive {forbidden}")
    required = build["required-directive"].upper()
    if required not in normalized:
        raise PackageError(f"VMA archive lacks /MD-compatible release CRT directive {required}")


def inspect_vma_directives(archive: Path, recipe: dict, dumpbin: str = "dumpbin") -> str:
    result = subprocess.run(
        [dumpbin, "/directives", str(archive)],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).upper()
    if result.returncode != 0:
        raise PackageError(f"cannot inspect VMA linker directives:\n{output}")
    validate_vma_directives(output, recipe)
    return output


def build_windows_vma(root: Path, recipe: dict) -> Path:
    verify_vma_header(root, recipe)
    if not (os.environ.get("VULKAN_HEADERS") or os.environ.get("VULKAN_SDK")):
        raise PackageError("VULKAN_HEADERS or VULKAN_SDK is required for Windows VMA")
    script = root / recipe["windows-vma-build"]["build-script"]
    result = subprocess.run(canonical_shell_command(script), cwd=root, check=False)
    if result.returncode != 0:
        raise PackageError("locked Windows VMA build failed")
    archive = root / recipe["targets"]["windows-x64"]["generated-native"]["source"]
    inspect_vma_directives(archive, recipe)
    return archive


def path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_location(root: Path, recipe: dict, output: Path) -> Path:
    output = output.resolve()
    if output.name != "gpu.c3l":
        raise PackageError("package output directory must be named gpu.c3l")
    protected = [
        (root / "gpu").resolve(),
        (root / "include").resolve(),
        (root / "packaging").resolve(),
        (root / "scripts").resolve(),
        *((root / binding["path"]).resolve() for binding in recipe["bindings"].values()),
    ]
    for input_root in protected:
        if output == input_root or path_contains(input_root, output) or path_contains(output, input_root):
            raise PackageError(f"package output overlaps package input root: {input_root}")
    return output


def verify_managed_bundle(path: Path, target: str) -> None:
    try:
        verify_bundle(path, target)
    except (OSError, PackageError) as error:
        raise PackageError(f"pre-existing unmanaged package directory: {path}: {error}") from error


def recover_interrupted_output(output: Path, target: str) -> None:
    backup = output.with_name(output.name + ".previous")
    if backup.exists():
        verify_managed_bundle(backup, target)
        if output.exists():
            verify_managed_bundle(output, target)
            shutil.rmtree(backup)
        else:
            backup.rename(output)
    if output.exists():
        verify_managed_bundle(output, target)


def assemble(
    root: Path,
    target: str,
    output: Path,
    toolchain: dict,
    build_windows: bool = False,
) -> dict:
    recipe = load_json(root / "packaging" / "package.json")
    validate_recipe(recipe, root)
    check_lock(root)
    if target not in SUPPORTED_TARGETS:
        raise PackageError(f"unsupported target: {target}")
    normalized_toolchain = normalize_toolchain(toolchain, target)
    if target == "windows-x64":
        validate_windows_toolchain(recipe, normalized_toolchain)
    if target == "windows-x64" and build_windows:
        build_windows_vma(root, recipe)
    target_recipe = recipe["targets"][target]
    generated = target_recipe.get("generated-native")
    if generated and not (root / generated["source"]).is_file():
        raise PackageError(f"generated native artifact is missing: {generated['source']}")
    output = validate_output_location(root, recipe, output)
    lock_path = root / "packaging" / "package-lock.json"
    recover_interrupted_output(output, target)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    bundle = temporary_parent / output.name
    backup = output.with_name(output.name + ".previous")
    try:
        bundle.mkdir()
        for mapping in recipe["sources"] + recipe["assets"] + recipe["licenses"]:
            copy_mapping(root, bundle, mapping, text=True)
        for mapping in target_recipe.get("native", []):
            copy_mapping(root, bundle, mapping)
        if generated:
            copy_mapping(root, bundle, generated)
        shutil.copy2(lock_path, bundle / "package-lock.json")
        write_canonical_json(bundle / "manifest.json", package_manifest(recipe, target))
        write_canonical_json(bundle / "runtime.json", runtime_manifest(recipe, target))
        tools = bundle / "tools"
        tools.mkdir()
        (tools / "runtime.py").write_bytes(canonical_text_bytes(root / "packaging" / "runtime.py"))
        payload = payload_entries(bundle)
        artifact = {
            "format": 1,
            "target": target,
            "locked-input-digest": sha256(bundle / "package-lock.json"),
            "toolchain": normalized_toolchain,
            "payload": payload,
            "payload-digest": payload_digest(payload),
        }
        if generated:
            generated_path = generated["destination"]
            generated_hash = next(item["sha256"] for item in payload if item["path"] == generated_path)
            artifact["generated-artifacts"] = sorted(
                [{"path": generated_path, "sha256": generated_hash}], key=lambda item: item["path"]
            )
        write_canonical_json(bundle / "artifact-manifest.json", artifact)
        verify_bundle(bundle, target, expected_lock=lock_path)
        if backup.exists():
            verify_managed_bundle(backup, target)
            shutil.rmtree(backup)
        if output.exists():
            output.rename(backup)
        try:
            bundle.rename(output)
        except Exception:
            if backup.exists() and not output.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return artifact
    finally:
        if temporary_parent.exists():
            shutil.rmtree(temporary_parent)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Assemble and verify target-scoped gpu.c3l packages")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lock_parser = subparsers.add_parser("lock")
    lock_mode = lock_parser.add_mutually_exclusive_group(required=True)
    lock_mode.add_argument("--check", action="store_true")
    lock_mode.add_argument("--refresh", action="store_true")

    subparsers.add_parser("fixture-policy")
    red_parser = subparsers.add_parser("fixture-red")
    red_parser.add_argument("--c3c", default="c3c")

    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--target", choices=sorted(SUPPORTED_TARGETS), required=True)
    assemble_parser.add_argument("--output", type=Path)
    assemble_parser.add_argument("--c3c-version", default="0.8.0_2")
    assemble_parser.add_argument("--skip-windows-vma-build", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--bundle", type=Path, required=True)
    verify_parser.add_argument("--target", choices=sorted(SUPPORTED_TARGETS), required=True)

    crt_parser = subparsers.add_parser("verify-vma-crt")
    crt_parser.add_argument("--archive", type=Path, required=True)
    crt_parser.add_argument("--dumpbin", default="dumpbin")

    args = parser.parse_args()
    try:
        if args.command == "lock":
            if args.check:
                check_lock(root)
            else:
                write_canonical_json(root / "packaging" / "package-lock.json", build_lock(root))
        elif args.command == "fixture-policy":
            check_fixture_policy(root / "test" / "consumer")
        elif args.command == "fixture-red":
            print(fixture_red(root, args.c3c))
        elif args.command == "assemble":
            output = args.output or root / "dist" / args.target / "gpu.c3l"
            if args.target == "windows-x64":
                recipe = load_json(root / "packaging" / "package.json")
                toolchain = derive_windows_toolchain(recipe, args.c3c_version)
            else:
                toolchain = {"platform": args.target, "c3c": args.c3c_version}
            artifact = assemble(
                root,
                args.target,
                output,
                toolchain,
                build_windows=not args.skip_windows_vma_build,
            )
            print(artifact["payload-digest"])
        elif args.command == "verify":
            print(
                verify_bundle(
                    args.bundle, args.target, expected_lock=root / "packaging" / "package-lock.json"
                )["payload-digest"]
            )
        else:
            recipe = load_json(root / "packaging" / "package.json")
            inspect_vma_directives(args.archive, recipe, args.dumpbin)
    except PackageError as error:
        parser.exit(1, f"package error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
