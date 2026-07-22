#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_FILES = (
    "gpu/internal/shader_abi.c3",
    "gpu/internal/vk/shader.c3",
    "tools/gen_shader_abi/src/emit_c3.c3",
    "scripts/build_shaders.py",
)


def mask_non_code(source: str) -> str:
    masked = list(source)
    index = 0
    quote: str | None = None
    block_end: str | None = None
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
            if source.startswith(block_end, index):
                masked[index:index + 2] = "  "
                block_end = None
                index += 2
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
            index += 2
        elif source.startswith("<*", index):
            masked[index:index + 2] = "  "
            block_end = "*>"
            index += 2
        elif source[index] in "\"'":
            masked[index] = " "
            quote = source[index]
            index += 1
        else:
            index += 1
    return "".join(masked)


def function_body(source: str, name: str) -> str:
    masked = mask_non_code(source)
    declaration = re.search(
        rf"(?m)^fn\s+[^\r\n(]*\b{re.escape(name)}\s*\(",
        masked,
    )
    if declaration is None:
        return ""
    brace = masked.find("{", declaration.end())
    if brace < 0:
        return ""
    depth = 0
    for index in range(brace, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return masked[brace:index + 1]
    return ""


def require(source: str, fragment: str, error: str, errors: list[str]) -> None:
    if fragment not in source:
        errors.append(error)


def check(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    try:
        generated = (root / "gpu/internal/shader_abi.c3").read_text(encoding="utf-8")
        shader = (root / "gpu/internal/vk/shader.c3").read_text(encoding="utf-8")
        emitter = (root / "tools/gen_shader_abi/src/emit_c3.c3").read_text(
            encoding="utf-8"
        )
        builder = (root / "scripts/build_shaders.py").read_text(encoding="utf-8")
    except OSError as error:
        return [str(error)]

    masked_generated = mask_non_code(generated)
    masked_emitter = mask_non_code(emitter)

    for fragment in (
        "struct RootAbiMemberSpec @private",
        "struct RootAbiSpec @private",
        "const RootAbiSpec ROOT_PUSH_ABI @private",
        "const RootAbiSpec GRAPHICS_ROOT_PUSH_ABI @private",
        "ROOT_PUSH_ABI.block_size == RootPush::size",
        "ROOT_PUSH_ABI.member_count == RootPush::members.len",
        "ROOT_PUSH_ABI.members[0].offset == $reflect(RootPush.root_gpu).offset",
        "ROOT_PUSH_ABI.members[0].size == GpuAddress::size",
        "GRAPHICS_ROOT_PUSH_ABI.block_size == GraphicsRootPush::size",
        "GRAPHICS_ROOT_PUSH_ABI.member_count == GraphicsRootPush::members.len",
        ".scalar_width",
        ".scalar_signed",
        ".integer",
    ):
        require(
            masked_generated,
            fragment,
            f"generated root reflection metadata missing: {fragment}",
            errors,
        )
    for fragment in (
        "max_push_member_count(schema)",
        "decl.kind == DeclKind.PUSH",
        "emit_root_abi_spec(decl, types, max_push_members, out)",
        "types.entries.get(field.type_name)",
    ):
        require(
            masked_emitter,
            fragment,
            f"root reflection metadata is not schema-derived: {fragment}",
            errors,
        )

    masked_shader = mask_non_code(shader)
    if re.search(r"\bextern\s+fn\b[^\n]*spvReflect|@cname\s*\(", masked_shader):
        errors.append("shader backend redeclares a foreign reflection function")
    for retired in (
        ".enumerate_descriptor_bindings(",
        ".enumerate_push_constant_blocks(",
    ):
        if retired in masked_shader:
            errors.append(f"module-wide reflection call remains: {retired}")
    for selected in (
        ".enumerate_entry_point_descriptor_bindings(",
        ".enumerate_entry_point_push_constant_blocks(",
    ):
        require(
            masked_shader,
            selected,
            f"selected-entry reflection call missing: {selected}",
            errors,
        )

    creation = function_body(shader, "create_pipeline_shader_module")
    order = (
        "get_entry_point(",
        "check_heap_convention(",
        "check_root_push_abi(",
        "create_pipeline_shader_module_native(",
    )
    positions = [creation.find(fragment) for fragment in order]
    if not creation or any(position < 0 for position in positions):
        errors.append("shader preparation is missing an exact reflection phase")
    elif positions != sorted(positions):
        errors.append("shader preparation does not validate selected entry before native creation")

    root_check = function_body(shader, "check_root_push_abi")
    member_check = function_body(shader, "root_member_shape_matches")
    for fragment in (
        "if (count == 0) return;",
        "count != 1",
        "block.byte_offset() != 0",
        "block.byte_size() != expected.block_size",
        "block.member_count() != expected.member_count",
    ):
        require(root_check, fragment, f"exact root block check missing: {fragment}", errors)
    for fragment in (
        "member.byte_offset() != expected.offset",
        "member.byte_size() != expected.size",
        "member.scalar_width() != expected.scalar_width",
        "member.scalar_is_signed() != expected.scalar_signed",
        "integer == expected.integer",
        "float_scalar != expected.integer",
        "TYPE_FLAG_VECTOR",
        "TYPE_FLAG_MATRIX",
        "TYPE_FLAG_ARRAY",
        "TYPE_FLAG_STRUCT",
        "TYPE_FLAG_BOOL",
        "TYPE_FLAG_REF",
    ):
        require(member_check, fragment, f"exact root member check missing: {fragment}", errors)
    if creation.count("public_fault:   gpu::SHADER_INVALID") != 4:
        errors.append("reflected shader failures must map to SHADER_INVALID")
    require(
        masked_shader,
        "state.pipeline_shader_create_attempts++;",
        "native shader-create attempt counter missing",
        errors,
    )
    require(
        masked_shader,
        "state.shader_reflection_validations++;",
        "reflection validation counter missing",
        errors,
    )

    for source_path in sorted((root / "test/shaders").glob("*.glsl")):
        source = mask_non_code(source_path.read_text(encoding="utf-8"))
        if re.search(r"\b(?:RootPush|GraphicsRootPush)\s+[A-Za-z_]", source):
            errors.append(
                f"{source_path.relative_to(root)} nests a generated root struct in a push block"
            )
    require(builder, 'glob("*.spvasm")', "SPIR-V assembly fixtures are not discovered", errors)
    require(builder, '"spirv-as"', "SPIR-V assembly fixtures do not use spirv-as", errors)
    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(f"shader-reflection-policy: {error}", file=sys.stderr)
        return 1
    print("shader reflection policy matches the exact selected-entry contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
