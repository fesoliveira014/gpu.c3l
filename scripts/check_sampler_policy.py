#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_FILES = (
    "gpu/gpu.c3",
    "gpu/gpu.c3i",
    "gpu/internal/sampler.c3",
    "gpu/internal/vk/internal.c3",
    "gpu/internal/vk/sampler.c3",
    "docs/api.md",
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


def declaration_and_body(source: str, kind: str, name: str) -> tuple[str, str]:
    masked = mask_non_code(source)
    declaration = re.search(
        rf"(?m)^{kind}\s+[^\r\n{{]*\b{re.escape(name)}\s*(?:\(|{{)",
        masked,
    )
    if declaration is None:
        return "", ""
    brace = masked.find("{", declaration.start())
    if brace < 0:
        return "", ""
    depth = 0
    for index in range(brace, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return masked[declaration.start():brace], masked[brace:index + 1]
    return "", ""


def sampler_contract_section(source: str) -> str:
    start = source.find("`intern_sampler` returns")
    if start < 0:
        return ""
    end = source.find("\n### ", start)
    return source[start:end if end >= 0 else len(source)]


def check(root: Path = ROOT) -> list[str]:
    try:
        sources = {
            relative: (root / relative).read_text(encoding="utf-8")
            for relative in POLICY_FILES
        }
    except OSError as error:
        return [str(error)]

    errors: list[str] = []
    _, frontend = declaration_and_body(
        sources["gpu/gpu.c3"],
        "fn",
        "intern_sampler",
    )
    validation_call = "gpu::internal::validate_sampler_desc(&operation, desc)!;"
    dispatch_call = (
        "return gpu::internal::vk::vk_intern_sampler(operation.state, desc);"
    )
    validation_position = frontend.find(validation_call)
    dispatch_position = frontend.find(dispatch_call)
    if validation_position < 0:
        errors.append("intern_sampler must call backend-independent sampler validation")
    if dispatch_position < 0:
        errors.append("intern_sampler direct call is missing")
    if (
        validation_position >= 0
        and dispatch_position >= 0
        and validation_position > dispatch_position
    ):
        errors.append("sampler validation must precede the direct backend call")

    _, validation = declaration_and_body(
        sources["gpu/internal/sampler.c3"],
        "fn",
        "validate_sampler_desc",
    )
    over_limit = (
        "desc.max_anisotropy > operation.data.caps.max_sampler_anisotropy"
    )
    over_limit_position = validation.find(over_limit)
    if over_limit_position < 0:
        errors.append(
            "active anisotropy must explicitly reject values above the device cap"
        )
    elif "gpu::INVALID_ARGUMENT" not in validation[over_limit_position:]:
        errors.append("over-limit anisotropy must fault INVALID_ARGUMENT")
    if re.search(r"\bdesc\.max_anisotropy\s*=", validation):
        errors.append("sampler validation must not mutate or clamp max_anisotropy")

    canonical_declaration, canonical = declaration_and_body(
        sources["gpu/internal/vk/internal.c3"],
        "fn",
        "canonical_sampler_key",
    )
    parameters = re.search(
        r"canonical_sampler_key\s*\((.*)\)\s*$",
        canonical_declaration,
        re.DOTALL,
    )
    normalized_parameters = (
        re.sub(r"\s+", " ", parameters.group(1)).strip().rstrip(",")
        if parameters is not None
        else ""
    )
    if normalized_parameters != "gpu::SamplerDesc* desc":
        errors.append("canonical sampler identity must not accept a device-cap input")
    if not canonical:
        errors.append("canonical sampler key construction is missing")
    else:
        if re.search(r"\b(?:device|caps?|limit|clamp)\b", canonical):
            errors.append(
                "canonical sampler identity must not contain device-limit policy"
            )
        if re.search(r"desc\.max_anisotropy\s*[<>]", canonical):
            errors.append("canonical sampler identity must not clamp anisotropy")
        exact_assignment = re.compile(
            r"\.max_anisotropy\s*=\s*desc\.anisotropy_enable\s*\?\s*"
            r"canonical_sampler_float\s*\(\s*desc\.max_anisotropy\s*\)\s*:\s*0\.0f",
            re.DOTALL,
        )
        if not exact_assignment.search(canonical):
            errors.append(
                "canonical sampler identity must preserve accepted anisotropy exactly"
            )

    _, native = declaration_and_body(
        sources["gpu/internal/vk/sampler.c3"],
        "fn",
        "vk_intern_sampler",
    )
    if "SamplerKey key = canonical_sampler_key(desc);" not in native:
        errors.append("Vulkan sampler interning must canonicalize only the description")
    if ".set_max_anisotropy(key.max_anisotropy)" not in native:
        errors.append("native sampler creation must consume key.max_anisotropy")

    _, sampler_desc = declaration_and_body(
        sources["gpu/gpu.c3i"],
        "struct",
        "SamplerDesc",
    )
    public_comment = sources["gpu/gpu.c3i"]
    public_start = public_comment.find("<* Semantic parameters for intern_sampler. *>")
    public_end = public_comment.find("\n}", public_start)
    public_contract = (
        public_comment[public_start:public_end]
        if public_start >= 0 and public_end >= 0
        else sampler_desc
    )
    public_rejection = re.compile(
        r"(?:above|exceed\w*)[^\n]{0,120}caps\.max_sampler_anisotropy"
        r"[\s\S]{0,160}INVALID_ARGUMENT",
        re.IGNORECASE,
    )
    if not public_rejection.search(public_contract):
        errors.append(
            "SamplerDesc public contract must state over-cap INVALID_ARGUMENT rejection"
        )

    api_contract = sampler_contract_section(sources["docs/api.md"])
    docs_rejection = re.compile(
        r"(?:above|exceed\w*)[\s\S]{0,160}DeviceCaps\.max_sampler_anisotropy"
        r"[\s\S]{0,160}(?:faults?|returns?)[\s\S]{0,60}`?INVALID_ARGUMENT`?",
        re.IGNORECASE,
    )
    no_implicit_clamp = re.compile(
        r"(?:not|never)\s+(?:implicitly\s+)?clamp\w*|"
        r"rather\s+than\s+(?:being\s+)?clamp\w*",
        re.IGNORECASE,
    )
    if not docs_rejection.search(api_contract):
        errors.append("API docs must state over-cap INVALID_ARGUMENT rejection")
    if not no_implicit_clamp.search(api_contract):
        errors.append(
            "API docs must state that sampler anisotropy is not implicitly clamped"
        )
    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(f"sampler-policy: {error}", file=sys.stderr)
        return 1
    print("sampler anisotropy rejection and exact-lowering policy is preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
