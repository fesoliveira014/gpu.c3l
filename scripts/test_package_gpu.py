from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import sys
import tempfile
import unittest
from subprocess import CompletedProcess
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PackageToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_module(ROOT / "scripts" / "package_gpu.py", "package_gpu")

    def copy_policy_fixture(self, destination: Path) -> Path:
        fixture = destination / "consumer"
        shutil.copytree(
            ROOT / "test" / "consumer",
            fixture,
            ignore=shutil.ignore_patterns("lib", "build", "shader.comp.spv"),
        )
        (fixture / "src" / "shader.comp.spv").write_bytes(
            struct.pack("<5I", 0x07230203, 0x00010000, 0, 1, 0)
        )
        return fixture

    def test_fixture_policy_accepts_gpu_only_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.tool.check_fixture_policy(self.copy_policy_fixture(Path(temp_dir)))

    def test_fixture_policy_reports_missing_or_unreadable_spirv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self.copy_policy_fixture(Path(temp_dir))
            (fixture / "src" / "shader.comp.spv").unlink()
            with self.assertRaisesRegex(self.tool.PackageError, "build.*shader.comp.spv"):
                self.tool.check_fixture_policy(fixture)

    def test_fixture_policy_rejects_backend_dependency(self) -> None:
        fixture = json.loads((ROOT / "test" / "consumer" / "project.json").read_text())
        fixture["dependencies"].append("vk")
        with self.assertRaisesRegex(self.tool.PackageError, "only gpu"):
            self.tool.validate_fixture_project(fixture)

    def test_fixture_policy_rejects_crt_and_nested_binding_roots(self) -> None:
        fixture = json.loads((ROOT / "test" / "consumer" / "project.json").read_text())
        fixture["wincrt"] = "dynamic"
        fixture["dependency-search-paths"].append("../../lib")
        with self.assertRaises(self.tool.PackageError):
            self.tool.validate_fixture_project(fixture)

    def test_recipe_rejects_unsupported_target_and_duplicate_destination(self) -> None:
        recipe = self.tool.load_json(ROOT / "packaging" / "package.json")
        broken = json.loads(json.dumps(recipe))
        broken["targets"]["macos-x64"] = broken["targets"]["linux-x64"]
        with self.assertRaisesRegex(self.tool.PackageError, "unsupported target"):
            self.tool.validate_recipe(broken, ROOT)
        broken = json.loads(json.dumps(recipe))
        broken["sources"].append(dict(broken["sources"][0]))
        with self.assertRaisesRegex(self.tool.PackageError, "duplicate destination"):
            self.tool.validate_recipe(broken, ROOT)

    def test_recipe_rejects_path_escape_and_binding_glob(self) -> None:
        recipe = self.tool.load_json(ROOT / "packaging" / "package.json")
        broken = json.loads(json.dumps(recipe))
        broken["sources"][0]["source"] = "../secret.c3"
        with self.assertRaises(self.tool.PackageError):
            self.tool.validate_recipe(broken, ROOT)
        broken = json.loads(json.dumps(recipe))
        broken["sources"][0]["source"] = "lib/vk.c3l/*.c3"
        with self.assertRaisesRegex(self.tool.PackageError, "glob"):
            self.tool.validate_recipe(broken, ROOT)

    def test_lock_check_detects_changed_source_and_submodule_pin(self) -> None:
        lock = self.tool.load_json(ROOT / "packaging" / "package-lock.json")
        changed = json.loads(json.dumps(lock))
        changed["sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(self.tool.PackageError, "lock differs"):
            self.tool.check_lock(ROOT, lock=changed)
        changed = json.loads(json.dumps(lock))
        changed["bindings"]["vk"]["commit"] = "0" * 40
        with self.assertRaisesRegex(self.tool.PackageError, "lock differs"):
            self.tool.check_lock(ROOT, lock=changed)

    def test_text_inputs_are_checkout_line_ending_independent(self) -> None:
        mapping = {"source": "source.c3", "destination": "gpu/source.c3"}
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            lf_root = temp / "lf"
            crlf_root = temp / "crlf"
            lf_root.mkdir()
            crlf_root.mkdir()
            (lf_root / "source.c3").write_bytes(b"module gpu;\nfn void example() {}\n")
            (crlf_root / "source.c3").write_bytes(b"module gpu;\r\nfn void example() {}\r\n")

            lf_lock = self.tool.hash_mappings(lf_root, [mapping], text=True)
            crlf_lock = self.tool.hash_mappings(crlf_root, [mapping], text=True)
            self.assertEqual(lf_lock, crlf_lock)

            bundle = temp / "bundle"
            self.tool.copy_mapping(crlf_root, bundle, mapping, text=True)
            self.assertEqual(
                b"module gpu;\nfn void example() {}\n",
                (bundle / "gpu" / "source.c3").read_bytes(),
            )

    def test_build_windows_vma_executes_canonical_crlf_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            recipe = self.tool.load_json(ROOT / "packaging" / "package.json")
            script = root / recipe["windows-vma-build"]["build-script"]
            script.parent.mkdir(parents=True)
            script.write_bytes(
                b"#!/usr/bin/env sh\r\n"
                b"set -eu\r\n"
                b'ROOT="$(cd "$(dirname "$0")/.." && pwd)"\r\n'
                b'printf "%s" "$ROOT" > "$ROOT/root.txt"\r\n'
            )
            archive = root / recipe["targets"]["windows-x64"]["generated-native"]["source"]
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"archive")
            calls = []

            def run(command, **kwargs):
                calls.append((command, kwargs))
                return CompletedProcess(command, 0, "", "")

            with (
                mock.patch.object(self.tool, "verify_vma_header"),
                mock.patch.object(self.tool, "inspect_vma_directives"),
                mock.patch.object(self.tool.subprocess, "run", side_effect=run),
                mock.patch.dict(self.tool.os.environ, {"VULKAN_HEADERS": "sdk"}),
            ):
                result = self.tool.build_windows_vma(root, recipe)

            self.assertEqual(archive, result)
            self.assertEqual(1, len(calls))
            command, kwargs = calls[0]
            self.assertEqual(["sh", "-c"], command[:2])
            self.assertNotIn("\r", command[2])
            self.assertIn('dirname "$0"', command[2])
            self.assertEqual(str(script), command[3])
            self.assertEqual(root, kwargs["cwd"])

    def test_lock_covers_recipe_link_metadata(self) -> None:
        lock = self.tool.load_json(ROOT / "packaging" / "package-lock.json")
        recipe = self.tool.load_json(ROOT / "packaging" / "package.json")
        recipe["targets"]["linux-x64"]["linked-libraries"].append("unexpected")
        self.assertNotEqual(lock, self.tool.build_lock(ROOT, recipe))

    def test_lock_check_detects_unlisted_production_binding_source(self) -> None:
        recipe = self.tool.load_json(ROOT / "packaging" / "package.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "repo"
            (repo / "lib").mkdir(parents=True)
            shutil.copytree(ROOT / "lib" / "vk.c3l", repo / "lib" / "vk.c3l")
            for tool_input in recipe["tool-inputs"]:
                source_path = ROOT / tool_input
                destination = repo / tool_input
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
            for source in recipe["sources"] + recipe["assets"] + recipe["licenses"]:
                if source["source"].startswith("lib/vk.c3l/"):
                    continue
                source_path = ROOT / source["source"]
                destination = repo / source["source"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
            (repo / "lib" / "vk.c3l" / "new_binding.c3").write_text("module vk;\n")
            with self.assertRaisesRegex(self.tool.PackageError, "unlisted production source"):
                self.tool.validate_recipe(recipe, repo, check_discovery=True)

    def test_lock_check_detects_unlisted_gpu_source_recursively(self) -> None:
        recipe = self.tool.load_json(ROOT / "packaging" / "package.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "repo"
            for tool_input in recipe["tool-inputs"]:
                destination = repo / tool_input
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / tool_input, destination)
            for mapping in recipe["sources"] + recipe["assets"] + recipe["licenses"]:
                destination = repo / mapping["source"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / mapping["source"], destination)
            for target_recipe in recipe["targets"].values():
                for mapping in target_recipe.get("native", []):
                    destination = repo / mapping["source"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / mapping["source"], destination)
            for key in ("wrapper-source", "size-probe-source", "build-script"):
                source = recipe["windows-vma-build"][key]
                destination = repo / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / source, destination)
            for binding in recipe["bindings"].values():
                source = ROOT / binding["path"]
                destination = repo / binding["path"]
                if not destination.exists():
                    shutil.copytree(source, destination)
            (repo / "gpu" / "nested" / "new_backend.c3").parent.mkdir(parents=True)
            (repo / "gpu" / "nested" / "new_backend.c3").write_text("module gpu::nested;\n")
            with self.assertRaisesRegex(self.tool.PackageError, "unlisted GPU production source"):
                self.tool.validate_recipe(recipe, repo, check_discovery=True)

    def test_artifact_manifest_rejects_self_inclusion_and_noncanonical_paths(self) -> None:
        base = {
            "format": 1,
            "target": "linux-x64",
            "locked-input-digest": "a" * 64,
            "toolchain": {"c3c": "0.8.0_2", "platform": "linux-x64"},
            "payload": [{"path": "manifest.json", "sha256": "b" * 64}],
            "payload-digest": "c" * 64,
        }
        for path in ("artifact-manifest.json", "a/../manifest.json", "a\\manifest.json"):
            broken = json.loads(json.dumps(base))
            broken["payload"][0]["path"] = path
            with self.assertRaises(self.tool.PackageError):
                self.tool.validate_artifact_manifest_shape(broken)

    def test_artifact_manifest_rejects_duplicate_or_missing_payload_hash(self) -> None:
        base = {
            "format": 1,
            "target": "linux-x64",
            "locked-input-digest": "a" * 64,
            "toolchain": {"c3c": "0.8.0_2", "platform": "linux-x64"},
            "payload": [{"path": "manifest.json", "sha256": "b" * 64}],
            "payload-digest": "c" * 64,
        }
        broken = json.loads(json.dumps(base))
        broken["payload"].append(dict(broken["payload"][0]))
        with self.assertRaises(self.tool.PackageError):
            self.tool.validate_artifact_manifest_shape(broken)
        del base["payload"][0]["sha256"]
        with self.assertRaises(self.tool.PackageError):
            self.tool.validate_artifact_manifest_shape(base)

    def test_artifact_manifest_rejects_unnormalized_toolchain(self) -> None:
        manifest = {
            "format": 1,
            "target": "windows-x64",
            "locked-input-digest": "a" * 64,
            "toolchain": {"c3c": " 0.8.0_2", "platform": "windows-x64", "workspace": "C:/repo"},
            "payload": [{"path": "manifest.json", "sha256": "b" * 64}],
            "payload-digest": "c" * 64,
        }
        with self.assertRaisesRegex(self.tool.PackageError, "toolchain"):
            self.tool.validate_artifact_manifest_shape(manifest)

    def test_windows_toolchain_rejects_placeholder_identity(self) -> None:
        with self.assertRaisesRegex(self.tool.PackageError, "placeholder"):
            self.tool.normalize_toolchain(
                {
                    "c3c": "0.8.0_2",
                    "platform": "windows-x64",
                    "compiler": "msvc",
                    "compiler-version": "unknown",
                    "vulkan-sdk": "1.3.290.0",
                },
                "windows-x64",
            )

    def test_package_manifest_rejects_changed_native_link_contract(self) -> None:
        recipe = self.tool.load_json(ROOT / "packaging" / "package.json")
        manifest = self.tool.package_manifest(recipe, "linux-x64")
        self.tool.validate_package_manifest(manifest, "linux-x64")
        manifest["targets"]["linux-x64"]["linked-libraries"].append("unexpected")
        with self.assertRaisesRegex(self.tool.PackageError, "link contract"):
            self.tool.validate_package_manifest(manifest, "linux-x64")

    def test_windows_toolchain_must_match_locked_vulkan_sdk(self) -> None:
        recipe = self.tool.load_json(ROOT / "packaging" / "package.json")
        toolchain = {
            "c3c": "0.8.0_2",
            "platform": "windows-x64",
            "compiler": "msvc",
            "compiler-version": "19.40.33811",
            "vulkan-sdk": "1.3.250.1",
        }
        with self.assertRaisesRegex(self.tool.PackageError, "Vulkan SDK"):
            self.tool.validate_windows_toolchain(recipe, toolchain)

    def test_quoted_msvc_directives_prove_dynamic_release_crt(self) -> None:
        recipe = self.tool.load_json(ROOT / "packaging" / "package.json")
        self.tool.validate_vma_directives('/DEFAULTLIB:"MSVCRT" /DEFAULTLIB:"MSVCPRT"', recipe)
        with self.assertRaisesRegex(self.tool.PackageError, "forbidden CRT"):
            self.tool.validate_vma_directives('/DEFAULTLIB:"LIBCMT"', recipe)

    def test_windows_toolchain_is_derived_from_cl_and_locked_headers(self) -> None:
        recipe = self.tool.load_json(ROOT / "packaging" / "package.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            sdk = Path(temp_dir) / "sdk"
            header = sdk / "include" / "vulkan" / "vulkan_core.h"
            header.parent.mkdir(parents=True)
            header.write_text(
                "#define VK_HEADER_VERSION 290\n"
                "#define VK_HEADER_VERSION_COMPLETE VK_MAKE_API_VERSION(0, 1, 3, VK_HEADER_VERSION)\n"
            )
            calls = []
            def run_cl(command, **kwargs):
                calls.append(command)
                return CompletedProcess(command, 0, "Microsoft (R) C/C++ Optimizing Compiler Version 19.40.33811 for x64\n", "")
            toolchain = self.tool.derive_windows_toolchain(
                recipe,
                "0.8.0_2",
                {"VULKAN_SDK": str(sdk)},
                run_cl,
            )
            self.assertEqual(["cl"], calls[0])
            self.assertEqual("19.40.33811", toolchain["compiler-version"])
            self.assertEqual("1.3.290.0", toolchain["vulkan-sdk"])
            header.write_text(
                "#define VK_HEADER_VERSION 250\n"
                "#define VK_HEADER_VERSION_COMPLETE VK_MAKE_API_VERSION(0, 1, 3, VK_HEADER_VERSION)\n"
            )
            with self.assertRaisesRegex(self.tool.PackageError, "Vulkan header"):
                self.tool.derive_windows_toolchain(
                    recipe,
                    "0.8.0_2",
                    {"VULKAN_HEADERS": str(sdk)},
                    run_cl,
                )

    def test_fixture_shader_uses_packaged_generated_abi(self) -> None:
        source = (ROOT / "test" / "consumer" / "src" / "shader.comp.glsl").read_text()
        self.assertIn('#include "generated/shader_abi.glsl"', source)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.tool.check_fixture_policy(self.copy_policy_fixture(Path(temp_dir)))

    def test_linux_assembly_is_verified_and_deterministic(self) -> None:
        self.tool.check_lock(ROOT)
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first" / "gpu.c3l"
            second = Path(temp_dir) / "second" / "gpu.c3l"
            self.tool.assemble(ROOT, "linux-x64", first, {"c3c": "0.8.0_2", "platform": "linux-x64"})
            self.tool.assemble(ROOT, "linux-x64", second, {"platform": "linux-x64", "c3c": "0.8.0_2"})
            first_manifest = self.tool.verify_bundle(first, "linux-x64")
            second_manifest = self.tool.verify_bundle(second, "linux-x64")
            self.assertEqual(first_manifest["payload-digest"], second_manifest["payload-digest"])
            package_manifest = json.loads((first / "manifest.json").read_text())
            self.assertEqual("gpu", package_manifest["provides"])
            self.assertNotIn("dependencies", json.dumps(package_manifest))

    def test_bundle_ships_assets_and_canonical_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "gpu.c3l"
            self.tool.assemble(ROOT, "linux-x64", bundle, {"c3c": "0.8.0_2", "platform": "linux-x64"})
            artifact = self.tool.verify_bundle(
                bundle,
                "linux-x64",
                expected_lock=ROOT / "packaging" / "package-lock.json",
            )
            lock_bytes = (bundle / "package-lock.json").read_bytes()
            self.assertEqual(self.tool.sha256_bytes(lock_bytes), artifact["locked-input-digest"])
            self.assertTrue((bundle / "include" / "shaders" / "descriptor_heap.glsl").is_file())
            self.assertTrue((bundle / "include" / "shaders" / "generated" / "shader_abi.glsl").is_file())

    def test_repository_verifier_rejects_stale_and_fabricated_bundle_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "gpu.c3l"
            self.tool.assemble(ROOT, "linux-x64", bundle, {"c3c": "0.8.0_2", "platform": "linux-x64"})
            lock = self.tool.load_json(bundle / "package-lock.json")
            lock["c3-version"] = "fabricated"
            self.tool.write_canonical_json(bundle / "package-lock.json", lock)
            artifact = self.tool.load_json(bundle / "artifact-manifest.json")
            payload = self.tool.payload_entries(bundle)
            artifact["payload"] = payload
            artifact["payload-digest"] = self.tool.payload_digest(payload)
            artifact["locked-input-digest"] = self.tool.sha256(bundle / "package-lock.json")
            self.tool.write_canonical_json(bundle / "artifact-manifest.json", artifact)
            with self.assertRaisesRegex(self.tool.PackageError, "checked-in package lock"):
                self.tool.verify_bundle(
                    bundle,
                    "linux-x64",
                    expected_lock=ROOT / "packaging" / "package-lock.json",
                )
            artifact["locked-input-digest"] = "0" * 64
            self.tool.write_canonical_json(bundle / "artifact-manifest.json", artifact)
            with self.assertRaisesRegex(self.tool.PackageError, "locked input digest"):
                self.tool.verify_bundle(bundle, "linux-x64")

    def test_nested_artifact_manifest_is_not_hidden_from_exact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "gpu.c3l"
            self.tool.assemble(ROOT, "linux-x64", bundle, {"c3c": "0.8.0_2", "platform": "linux-x64"})
            nested = bundle / "nested" / "artifact-manifest.json"
            nested.parent.mkdir()
            nested.write_text("{}\n")
            with self.assertRaisesRegex(self.tool.PackageError, "payload set"):
                self.tool.verify_bundle(bundle, "linux-x64")

    def test_output_safety_rejects_unmanaged_and_protected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            unmanaged = temp / "gpu.c3l"
            unmanaged.mkdir()
            marker = unmanaged / "keep.txt"
            marker.write_text("keep")
            with self.assertRaisesRegex(self.tool.PackageError, "unmanaged"):
                self.tool.assemble(ROOT, "linux-x64", unmanaged, {"c3c": "0.8.0_2", "platform": "linux-x64"})
            self.assertEqual("keep", marker.read_text())
            with self.assertRaisesRegex(self.tool.PackageError, "gpu.c3l"):
                self.tool.assemble(ROOT, "linux-x64", temp / "wrong-name", {"c3c": "0.8.0_2", "platform": "linux-x64"})
            with self.assertRaisesRegex(self.tool.PackageError, "package input"):
                self.tool.assemble(ROOT, "linux-x64", ROOT / "gpu" / "gpu.c3l", {"c3c": "0.8.0_2", "platform": "linux-x64"})

    def test_interrupted_managed_backup_is_recovered_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "gpu.c3l"
            self.tool.assemble(ROOT, "linux-x64", output, {"c3c": "0.8.0_2", "platform": "linux-x64"})
            backup = output.with_name("gpu.c3l.previous")
            output.rename(backup)
            result = self.tool.assemble(ROOT, "linux-x64", output, {"c3c": "0.8.0_2", "platform": "linux-x64"})
            self.assertTrue(output.is_dir())
            self.assertFalse(backup.exists())
            self.assertEqual(result["payload-digest"], self.tool.verify_bundle(output, "linux-x64")["payload-digest"])

    def test_old_lock_managed_backup_is_upgraded_to_current_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "gpu.c3l"
            self.tool.assemble(ROOT, "linux-x64", output, {"c3c": "0.8.0_2", "platform": "linux-x64"})
            lock = self.tool.load_json(output / "package-lock.json")
            lock["c3-version"] = "0.7.0"
            self.tool.write_canonical_json(output / "package-lock.json", lock)
            artifact = self.tool.load_json(output / "artifact-manifest.json")
            payload = self.tool.payload_entries(output)
            artifact["payload"] = payload
            artifact["payload-digest"] = self.tool.payload_digest(payload)
            artifact["locked-input-digest"] = self.tool.sha256(output / "package-lock.json")
            self.tool.write_canonical_json(output / "artifact-manifest.json", artifact)
            backup = output.with_name("gpu.c3l.previous")
            output.rename(backup)
            self.tool.assemble(ROOT, "linux-x64", output, {"c3c": "0.8.0_2", "platform": "linux-x64"})
            self.assertFalse(backup.exists())
            self.tool.verify_bundle(
                output,
                "linux-x64",
                expected_lock=ROOT / "packaging" / "package-lock.json",
            )

    def test_verifier_rejects_corrupt_and_extra_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "gpu.c3l"
            self.tool.assemble(ROOT, "linux-x64", bundle, {"c3c": "0.8.0_2", "platform": "linux-x64"})
            (bundle / "manifest.json").write_text("{}\n")
            with self.assertRaisesRegex(self.tool.PackageError, "hash"):
                self.tool.verify_bundle(bundle, "linux-x64")
            bundle = Path(temp_dir) / "second" / "gpu.c3l"
            self.tool.assemble(ROOT, "linux-x64", bundle, {"c3c": "0.8.0_2", "platform": "linux-x64"})
            (bundle / "extra.txt").write_text("unexpected")
            with self.assertRaisesRegex(self.tool.PackageError, "payload set"):
                self.tool.verify_bundle(bundle, "linux-x64")


class RuntimeToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_module(ROOT / "packaging" / "runtime.py", "gpu_package_runtime")

    def runtime_manifest(self, target: str = "windows-x64") -> dict:
        return {
            "format": 1,
            "target": target,
            "package-owned-runtime-files": [],
            "system-prerequisites": [
                {"name": "Vulkan loader", "discovery": "vulkan-1.dll" if target == "windows-x64" else "libvulkan.so.1"},
                {"name": "Microsoft Visual C++ dynamic release runtime", "discovery": "declarative-only"},
            ] if target == "windows-x64" else [
                {"name": "Vulkan loader", "discovery": "libvulkan.so.1"},
                {"name": "C++ runtime", "discovery": "libstdc++.so.6"},
            ],
        }

    def test_no_runtime_file_stage_succeeds_without_copying_system_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package = Path(temp_dir) / "gpu.c3l"
            destination = Path(temp_dir) / "app"
            package.mkdir()
            (package / "runtime.json").write_text(json.dumps(self.runtime_manifest()))
            report = self.runtime.stage(package, destination, "windows-x64")
            self.assertEqual([], report["staged"])
            self.assertEqual([], list(destination.iterdir()))
            self.assertIn("non-authoritative", report["message"])

    def test_wrong_target_and_missing_loader_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package = Path(temp_dir)
            (package / "runtime.json").write_text(json.dumps(self.runtime_manifest()))
            with self.assertRaisesRegex(self.runtime.RuntimeContractError, "target"):
                self.runtime.check(package, "linux-x64", discover=lambda _: True)
            with self.assertRaisesRegex(self.runtime.RuntimeContractError, "Vulkan loader"):
                self.runtime.check(package, "windows-x64", discover=lambda _: False)

    def test_stage_rejects_corrupt_runtime_file(self) -> None:
        manifest = self.runtime_manifest("linux-x64")
        manifest["package-owned-runtime-files"] = [{"path": "runtime/example.so", "sha256": "0" * 64}]
        with tempfile.TemporaryDirectory() as temp_dir:
            package = Path(temp_dir) / "gpu.c3l"
            destination = Path(temp_dir) / "app"
            (package / "runtime").mkdir(parents=True)
            (package / "runtime" / "example.so").write_text("corrupt")
            (package / "runtime.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(self.runtime.RuntimeContractError, "hash"):
                self.runtime.stage(package, destination, "linux-x64")
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
