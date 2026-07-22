#!/usr/bin/env python3
from __future__ import annotations

from functools import lru_cache

import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_COMMANDS = (
    "cmd_barrier",
    "cmd_dispatch_indirect",
    "cmd_draw_indirect",
    "cmd_draw_indexed_indirect",
    "cmd_draw_indexed_indirect_count",
    "cmd_dispatch_generated",
    "cmd_draw_generated",
    "cmd_draw_indexed_generated",
)
BACKEND_COMMANDS = (
    "vk_cmd_dispatch_indirect",
    "vk_cmd_dispatch_generated",
    "execute_generated_work",
)
RECORDING_FORBIDDEN = (
    "lock_device_registry(",
    "alloc::new",
    "mem::new",
    "create_compute_pipeline(",
    "create_graphics_pipeline(",
    "prepare_shader_code(",
)
PUBLIC_RECORDING_FORBIDDEN = (
    "command_operation(",
    "executable_command_operation(",
    ".vtable.",
)
BACKEND_RECORDING_FORBIDDEN = (
    "device_backend_state_ptr(",
    "resolve_command(",
    ".commands.get(",
    "alloc::new",
    "mem::new",
    "vk::allocate_command_buffers(",
    "vk::free_command_buffers(",
    "vk::create_image_view(",
    "create_buffer_with_alignment(",
)
POST_BIND_PIPELINE_PATTERNS = (
    re.compile(r"\.pipelines\s*\.\s*get(?:_cell)?\s*\("),
    re.compile(r"\.pipeline_cache\s*\."),
)
RECORDING_PIPELINE_RESOLUTION_ALLOWLIST = frozenset((
    "bind_pipeline_state",
    "retain_validation_reference",
    "release_validation_reference",
))
BOUND_PIPELINE_REVALIDATION_FORBIDDEN = (
    "validation_cell",
    "expected_generation",
    "validate_bound_pipeline_identity",
)
BOUND_PIPELINE_SNAPSHOT_FIELDS = (
    "handle",
    "pipeline",
    "layout",
    "generated_dispatch_layout",
    "cache_entry",
    "kind",
    "render",
)
PIPELINE_KEY_FIELDS = (
    "vertex_shader",
    "fragment_shader",
    "color_target_count",
    "color_targets",
    "depth_format",
    "sample_count",
    "polygon_mode",
    "kind",
    "reserved",
)
PIPELINE_CACHE_ENTRY_FIELDS = (
    "hash",
    "key",
    "pipeline",
    "layout",
    "generated_dispatch_layout",
    "refcount",
    "next_free",
    "next_in_bucket",
    "used",
)
SAMPLER_CELL_FIELDS = (
    "hash",
    "key",
    "native",
    "index",
    "next_in_bucket",
)
SAMPLER_TABLE_FIELDS = (
    "slots",
    "bucket_heads",
    "count",
)
RECORDING_COLD_GROWTH_ALLOWLIST = frozenset((
    "ensure_command_reference_capacity",
))
DESTRUCTION_FORBIDDEN = (
    "wait_completion(",
    "wait_queue_idle(",
    "device_wait_idle(",
    "vk::device_wait_idle(",
    "enqueue_deferred",
    "deferred_release",
)
RETIRED_TEXTURE_BARRIER_PATHS = (
    "TextureUseScope",
    "texture_use_",
    "fn TextureBarrierRejection texture_barrier_rejection(",
    "fn TextureBarrierRejection texture_barrier_queue_rejection(",
    "texture_transition_range(",
    "texture_barrier_to_vk(",
)
POINT_ALLOCATION_FORBIDDEN = (
    "alloc::new",
    "mem::new",
    ".alloc(",
)
OWNERSHIP_TRANSFER_CALLEES = (
    "release_submitted_command_batch",
    "release_completed_submitted_commands_locked",
    "retire_queue_through_locked",
    "retire_queue_through",
    "drain_completed_submitted_commands_with_query",
    "publish_submitted_commands",
    "release_command",
    "vk_submit_with_queries",
    "vk_submit_with",
)
BACKEND_OWNERSHIP_DIGESTS = {
    "destroy_context_pools_with_ops": "3557e590bc22dc2c68fbce07a214ea56ffd136c1203b339cdac4db3c7edca975",
    "generated_scratch_requirements": "0d2fc883196ed20b3592c3c8141668f7f43047858614e6592eb836a3d837fb29",
    "vk_reserve_generated_scratch": "aa90c235e2fe339391bcf97831b6f0d325b9e2ae817393916e07a7e25af3ece6",
    "vk_release_generated_scratch": "2daebee02d4d33e23ffc870a86b879243c735f55226b8a0254b337ad3e060104",
    "take_available_command_buffer": "862b1ca13ac29dd551057bdf41896f22fb57cffebaaf19bc63440cbc07eab238",
    "vk_begin_commands_with_context": "993b27c291128abaf97f92195b0ec28cc15b7b6fd9c5b606a5bbb87f1f19911d",
    "vk_discard_commands": "de31365009abdfaa8746e216c78f84a814626bdfc89b5d0e88336a37fb3a33e1",
    "execute_generated_work": "0c1fb3884d112926a75f6cc5bee54722a74868f1ab7aae69bb69a4d8af0360be",
    "generated_preprocess_compatible": "df5842124c9c4e5427be4440e3a6ce3df20282e3d3b7d9d7557ac6bec007b5a0",
    "recycle_generated_preprocess_buffers_locked": "38b1c600ce0ea88ec08d5fa56b3e61eaf97007c10afc94aec6893d1dc9a5da8b",
    "command_recording_stats": "381e23f43057fb0882396bc074d75c196da6fc3fe2b74547f300223eced04a2e",
    "free_generated_preprocess_buffers": "ff81c290557da4ecc7722c172fc4a3904655642220de6056e8c9c3412eea0874",
    "free_reserved_generated_preprocess_buffers": "a4e9fc1d9b5e6d1a010e634ceab4bfd7090707552f5d39abf84d736faa59c744",
    "allocate_generated_preprocess_buffer": "5e726f32786526ffc3c674902100be1a15c430ec71a13c0e476e0b78dde97e48",
    "acquire_generated_preprocess_buffer": "d941f2be232aaaee3236a9a041104c793766c136640bbfd87627c3144357107d",
}
COMMAND_STATE_OWNERSHIP_DIGESTS = {
    "release_command": "052c7e5e284197a6d22729dc7045397c37535c0a9c50aaf3ec163211d0859157",
    "destroy_command_table": "45c57c05ed629582c17062c6829d1a2a9146ba44bce9552b186cefc0692723d7",
}
LIFETIME_OWNERSHIP_DIGESTS = {
    "publish_submitted_commands": "cdbc47b4f1b25be4fc79e04f6ae8ff410b219ab6854924df0edabd389376f066",
    "release_submitted_command_batch": "62e62aa38aadd18bf69a42b62d3851be0ce30b9007e948f9123261bcf091656e",
    "release_completed_submitted_commands_locked": "bd81cb2ab4514a5428d70fa75883594913b7e77d0c15f123593b008f7e0e50d4",
    "retire_queue_through_locked": "20a6ac45e04f5ef6af39283a325adbda8ede10c48439b4189b49db726afd03da",
    "retire_queue_through": "f156456f232cebdcd4a42940d4bfcde11fbe13fb97e0e8b7954502ca630b022e",
    "destroy_submitted_commands": "1858d0b6e5a0d7659b1ed6ca6a08abf7e13f5577cdcea9cdc48dfdcde0d95d60",
    "drain_completed_submitted_commands_with_query": "72af65609acdec98806062f89af9b3381a402d28d29729fe8a8829699a88db19",
    "drain_completed_submitted_commands": "182036fc91b30152a2f76fd8a300f6b25efc1ff4d2260c4eafaa60fc00ba4314",
    "drain_submitted_commands_if_needed_with_query": "4bfb18e43bb0a25f357179f3e0d84bb1baed481b10fb5738e30ffd98e4d7ed10",
    "drain_submitted_commands_if_needed": "c97ab94eaa1307124bce6b53002c80b262624054629ad715c900ab1f53e73930",
}
DEVICE_OWNERSHIP_DIGESTS = {
    "destroy_state": "b3fbea364f5723c79a3804460bedd5a96eae9fe636d44e78677b6d9fddea0a19",
}

SYNC_OWNERSHIP_DIGESTS = {
    "vk_poll_completion_with_query": "412d2e08adca4b1ba1d490369eea100e679c0f166bfe3fbda72a43714d0a9ce4",
    "vk_wait_completion_with_wait": "b237ddbea1156e1a5ae3d814c60b9fecdb50f29ee5105038ec799f66d43ecb6a",
}
QUEUE_OWNERSHIP_DIGESTS = {
    "require_queue_completion_headroom_with_query": "c3f89d8dc421137f79ce5954587dd384cfe88059eda20e30216f7099cbaaaa07",
    "vk_submit_with_queries": "c78440eaacec31ec5a8e6648404bda079b58ca9ee47ed93828eac74007204329",
    "vk_submit_with": "773cfe5518f228b74c72265329eeb46a1db722334cabe1f57b0f0f4a5e3dc9e1",
    "vk_submit": "383f58739300051514fa01012860a73d7b96601c6d8040ffcbf1f864d111196a",
}


@lru_cache(maxsize=128)
def mask_c3_comments(source: str) -> str:
    masked = list(source)
    index = 0
    quote = None
    block_depth = 0

    while index < len(source):
        if quote is not None:
            if source[index] == "\\":
                index += 2
            elif source[index] == quote:
                quote = None
                index += 1
            else:
                index += 1
            continue

        if block_depth > 0:
            if source.startswith("/*", index):
                masked[index:index + 2] = "  "
                block_depth += 1
                index += 2
            elif source.startswith("*/", index):
                masked[index:index + 2] = "  "
                block_depth -= 1
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
            block_depth = 1
            index += 2
        elif source[index] in "\"'":
            quote = source[index]
            index += 1
        else:
            index += 1

    return "".join(masked)


def function_body(source: str, name: str) -> str:
    masked_source = mask_c3_comments(source)
    declaration = re.search(
        rf"(?m)^fn\s+[^\n]*\b{re.escape(name)}\s*\(",
        masked_source,
    )
    if declaration is None:
        raise ValueError(f"missing function {name}")
    start = masked_source.find("{", declaration.end())
    if start < 0:
        raise ValueError(f"missing body for {name}")
    depth = 0
    for index in range(start, len(masked_source)):
        if masked_source[index] == "{":
            depth += 1
        elif masked_source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise ValueError(f"unterminated body for {name}")


def function_names(source: str) -> tuple[str, ...]:
    return tuple(
        re.findall(
            r"(?m)^fn\s+[^\n]*\b([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            mask_c3_comments(source),
        )
    )


def struct_body(source: str, name: str) -> str:
    masked_source = mask_c3_comments(source)
    declaration = re.search(
        rf"(?m)^struct\s+{re.escape(name)}\s*\{{",
        masked_source,
    )
    if declaration is None:
        raise ValueError(f"missing struct {name}")
    start = masked_source.find("{", declaration.start())
    depth = 0
    for index in range(start, len(masked_source)):
        if masked_source[index] == "{":
            depth += 1
        elif masked_source[index] == "}":
            depth -= 1
            if depth == 0:
                return masked_source[start + 1:index]
    raise ValueError(f"unterminated struct {name}")


def struct_field_names(source: str, name: str) -> tuple[str, ...]:
    return tuple(re.findall(
        r"(?m)^\s*[^\n;{}]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
        struct_body(source, name),
    ))


def read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def source_functions(
    root: Path,
    source_root: str,
) -> dict[str, list[tuple[str, str]]]:
    functions: dict[str, list[tuple[str, str]]] = {}
    for path in sorted((root / source_root).glob("*.c3")):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        for name in function_names(source):
            functions.setdefault(name, []).append(
                (relative, function_body(source, name))
            )
    return functions


def reachable_recording_functions(
    functions: dict[str, list[tuple[str, str]]],
    root_prefix: str,
) -> set[tuple[str, str]]:
    candidates = tuple(functions)
    pending = [name for name in candidates if name.startswith(root_prefix)]
    visited_names: set[str] = set()
    reachable: set[tuple[str, str]] = set()
    while pending:
        name = pending.pop()
        if name in visited_names:
            continue
        visited_names.add(name)
        for relative, body in functions.get(name, ()):
            reachable.add((relative, name))
            code = mask_c3_comments(body)
            for candidate in candidates:
                if candidate not in visited_names and re.search(
                    rf"\b{re.escape(candidate)}\s*\(",
                    code,
                ):
                    pending.append(candidate)
    return reachable


def reject_tokens(
    errors: list[str],
    relative: str,
    name: str,
    body: str,
    tokens: tuple[str, ...],
) -> None:
    for token in tokens:
        if token in body:
            errors.append(f"{relative}:{name} contains forbidden {token}")


def require_token_order(
    errors: list[str],
    relative: str,
    name: str,
    body: str,
    tokens: tuple[str, ...],
) -> None:
    position = -1
    for token in tokens:
        next_position = body.find(token, position + 1)
        if next_position < 0:
            errors.append(f"{relative}:{name} is missing ordered token {token}")
            return
        if next_position <= position:
            errors.append(f"{relative}:{name} has invalid completion ordering")
            return
        position = next_position


def scans_slot_table(body: str) -> bool:
    code = mask_c3_comments(body)
    if not re.search(r"\b(?:for|foreach|while)\b", code):
        return False
    if re.search(r"\.slots\s*\[\s*:\s*[^\]]*\.count\s*\]", code):
        return True
    receivers = re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\.slots\b",
        code,
    )
    return any(
        re.search(rf"\b{re.escape(receiver)}\.count\b", code)
        for receiver in receivers
    )


def check(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    public_source = read(root, "gpu/command.c3")
    backend_source = read(root, "gpu/vk/command.c3")
    render_source = read(root, "gpu/vk/render_pass.c3")
    attachment_source = read(root, "gpu/vk/attachment_view.c3")
    sync_source = read(root, "gpu/vk/sync.c3")
    queue_source = read(root, "gpu/vk/queue.c3")
    texture_source = read(root, "gpu/vk/texture.c3")
    device_source = read(root, "gpu/device.c3")
    command_state_source = read(root, "gpu/vk/command_state.c3")
    backend_device_source = read(root, "gpu/vk/device.c3")
    pipeline_cache_source = read(root, "gpu/vk/pipeline_cache.c3")
    shader_source = read(root, "gpu/vk/shader.c3")
    pipeline_compute_source = read(root, "gpu/vk/pipeline_compute.c3")
    sampler_source = read(root, "gpu/vk/sampler.c3")
    lifetime_source = read(root, "gpu/vk/lifetime.c3")
    command_bench = read(root, "test/src/command_record_bench.c3")
    lifecycle_bench = read(root, "test/src/lifecycle_bench.c3")

    if struct_field_names(sampler_source, "SamplerCell") != SAMPLER_CELL_FIELDS:
        errors.append(
            "gpu/vk/sampler.c3:SamplerCell must contain the reviewed "
            "hash/key/native/index/link fields"
        )
    if struct_field_names(sampler_source, "SamplerTable") != SAMPLER_TABLE_FIELDS:
        errors.append(
            "gpu/vk/sampler.c3:SamplerTable must contain fixed slots, "
            "bucket heads, and count"
        )
    for token in (
        "next_pow2(capacity * 2)",
        "table.bucket_heads[bucket] = index + 1;",
        "cell.hash == hash && sampler_key_equal(&cell.key, key)",
        "ulong hash = sampler_key_hash(state, &key);",
        "link_sampler_cell(table, cell_index);",
    ):
        if token not in sampler_source:
            errors.append(
                "gpu/vk/sampler.c3 is missing hashed sampler-index token "
                f"{token}"
            )

    backend_functions = source_functions(root, "gpu/vk")
    sampler_reachable = reachable_recording_functions(
        backend_functions,
        "vk_intern_sampler",
    )
    for relative, name in sorted(sampler_reachable):
        body = next(
            body
            for candidate_relative, body in backend_functions[name]
            if candidate_relative == relative
        )
        code = mask_c3_comments(body)
        reviewed_bucket_lookup = (
            relative == "gpu/vk/sampler.c3"
            and name == "find_sampler_cell"
        )
        if (
            not reviewed_bucket_lookup
            and re.search(r"\b(?:for|foreach|while)\b", code)
        ):
            errors.append(
                f"{relative}:{name} performs forbidden whole-table sampler scan"
            )
        if scans_slot_table(body):
            errors.append(
                f"{relative}:{name} traverses the published sampler prefix"
            )

    poll_completion = function_body(sync_source, "vk_poll_completion_with_query")
    require_token_order(
        errors,
        "gpu/vk/sync.c3",
        "vk_poll_completion_with_query",
        poll_completion,
        (
            "retired_sequence.load(AtomicOrdering.ACQUIRE)",
            "note_completion_counter_query(state);",
            "query_fn(",
            "retire_queue_through(",
        ),
    )
    wait_completion = function_body(sync_source, "vk_wait_completion_with_wait")
    require_token_order(
        errors,
        "gpu/vk/sync.c3",
        "vk_wait_completion_with_wait",
        wait_completion,
        (
            "retired_sequence.load(AtomicOrdering.ACQUIRE)",
            "vk::semaphore_wait_info()",
            "note_completion_wait_call(state);",
            "wait_fn(",
            "retire_queue_through(",
        ),
    )
    retire_locked = function_body(lifetime_source, "retire_queue_through_locked")
    require_token_order(
        errors,
        "gpu/vk/lifetime.c3",
        "retire_queue_through_locked",
        retire_locked,
        (
            "published_sequence.load(",
            "AtomicOrdering.ACQUIRE",
            "retired_sequence.load(AtomicOrdering.RELAXED)",
            "release_completed_submitted_commands_locked(",
            "retired_sequence.store(target, AtomicOrdering.RELEASE)",
        ),
    )
    retire_outer = function_body(lifetime_source, "retire_queue_through")
    require_token_order(
        errors,
        "gpu/vk/lifetime.c3",
        "retire_queue_through",
        retire_outer,
        (
            "retired_sequence.load(AtomicOrdering.ACQUIRE)",
            "state.resource_mutex.lock()",
            "retire_queue_through_locked(",
        ),
    )
    headroom = function_body(
        queue_source,
        "require_queue_completion_headroom_with_query",
    )
    require_token_order(
        errors,
        "gpu/vk/queue.c3",
        "require_queue_completion_headroom_with_query",
        headroom,
        (
            "retired_sequence.load(",
            "AtomicOrdering.ACQUIRE",
            "query_fn(",
            "retire_queue_through(",
            "retired_sequence.load(AtomicOrdering.ACQUIRE)",
        ),
    )

    masked_sync = mask_c3_comments(sync_source)
    for token in RETIRED_TEXTURE_BARRIER_PATHS:
        if token in masked_sync:
            errors.append(
                "gpu/vk/sync.c3 contains retired texture-barrier path "
                f"{token}"
            )

    lower_texture_barrier = function_body(
        sync_source,
        "validate_and_lower_texture_barrier",
    )
    expected_lowering_work = (
        ("note_texture_barrier_helper(state);", 1),
        ("resolve_texture_command_reference(", 1),
        ("note_texture_barrier_access_validation(state);", 1),
        ("validate_texture_queue_access(", 1),
        ("note_texture_barrier_range_resolution(state);", 1),
        ("resolve_texture_barrier_range(", 1),
        ("note_texture_barrier_state_validation(state);", 2),
        ("texture_state_rejection(", 2),
        ("texture_state_to_vk(", 2),
        ("note_texture_barrier_native_assembly(state);", 1),
        ("vk::image_memory_barrier2()", 1),
    )
    for token, expected_count in expected_lowering_work:
        if lower_texture_barrier.count(token) != expected_count:
            errors.append(
                "gpu/vk/sync.c3:validate_and_lower_texture_barrier must "
                f"perform {token} exactly {expected_count} time(s)"
            )

    texture_barrier_command = function_body(
        sync_source,
        "vk_cmd_texture_barrier",
    )
    expected_command_work = (
        ("validate_and_lower_texture_barrier(", 1),
        ("set_image_memory_barriers((&lowered.native)[:1])", 1),
        ("note_texture_barrier_native_emission(state);", 1),
    )
    for token, expected_count in expected_command_work:
        if texture_barrier_command.count(token) != expected_count:
            errors.append(
                "gpu/vk/sync.c3:vk_cmd_texture_barrier must perform "
                f"{token} exactly {expected_count} time(s)"
            )
    reject_tokens(
        errors,
        "gpu/vk/sync.c3",
        "vk_cmd_texture_barrier",
        texture_barrier_command,
        (
            "state.textures.get(",
            "validate_texture_recording_access(",
            "resolve_texture_barrier_range(",
            "texture_state_rejection(",
            "texture_state_to_vk(",
        ),
    )

    public_functions = source_functions(root, "gpu")
    public_reachable = reachable_recording_functions(
        public_functions,
        "cmd_",
    )
    for relative, name in sorted(public_reachable):
        body = next(
            body
            for candidate_relative, body in public_functions[name]
            if candidate_relative == relative
        )
        reject_tokens(
            errors,
            relative,
            name,
            mask_c3_comments(body),
            PUBLIC_RECORDING_FORBIDDEN,
        )

    reachable = reachable_recording_functions(backend_functions, "vk_cmd_")
    for relative, name in sorted(reachable):
        body = mask_c3_comments(
            next(
                body
                for candidate_relative, body in backend_functions[name]
                if candidate_relative == relative
            )
        )
        forbidden = BACKEND_RECORDING_FORBIDDEN
        if name in RECORDING_COLD_GROWTH_ALLOWLIST:
            forbidden = tuple(
                token for token in forbidden
                if token not in ("alloc::new", "mem::new")
            )
        reject_tokens(
            errors,
            relative,
            name,
            body,
            forbidden,
        )
        if name in RECORDING_PIPELINE_RESOLUTION_ALLOWLIST:
            continue
        for pattern in POST_BIND_PIPELINE_PATTERNS:
            if pattern.search(body):
                errors.append(
                    f"{relative}:{name} performs forbidden post-bind "
                    "pipeline resolution on a recording path"
                )

    bound_pipeline_source = mask_c3_comments(
        backend_source + "\n" + command_state_source + "\n" + render_source
    )
    for token in BOUND_PIPELINE_REVALIDATION_FORBIDDEN:
        if token in bound_pipeline_source:
            errors.append(
                "Vulkan command recording retains forbidden bound-pipeline "
                f"revalidation token {token}"
            )
    try:
        bound_pipeline_fields = struct_field_names(
            command_state_source,
            "BoundPipeline",
        )
    except ValueError as error:
        errors.append(f"gpu/vk/command_state.c3:{error}")
    else:
        if bound_pipeline_fields != BOUND_PIPELINE_SNAPSHOT_FIELDS:
            errors.append(
                "gpu/vk/command_state.c3:BoundPipeline must contain only the "
                "reviewed native snapshot fields"
            )

    try:
        command_record_body = struct_body(command_state_source, "CommandRecord")
    except ValueError as error:
        errors.append(f"gpu/vk/command_state.c3:{error}")
    else:
        if re.search(
            r"(?m)^\s*PipelineCell\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*;",
            command_record_body,
        ):
            errors.append(
                "gpu/vk/command_state.c3:CommandRecord must not retain a "
                "PipelineCell pointer"
            )

    backend_root = root / "gpu/vk"
    retired_layout_cache_references = 0
    for path in sorted(backend_root.glob("*.c3")):
        retired_layout_cache_references += mask_c3_comments(
            path.read_text(encoding="utf-8")
        ).count("compute_layout_cache")
    if retired_layout_cache_references != 0:
        errors.append(
            "gpu/vk must use singleton compute layouts, not compute-layout "
            "cache storage"
        )

    if backend_device_source.count(
        "vk::PipelineLayout         compute_layout;"
    ) != 1:
        errors.append(
            "gpu/vk/device.c3 must own exactly one compute pipeline layout"
        )
    if backend_device_source.count(
        "vk::IndirectCommandsLayoutEXT generated_dispatch_layout;"
    ) != 1:
        errors.append(
            "gpu/vk/device.c3 must own exactly one generated dispatch layout"
        )

    shared_creation = function_body(
        pipeline_cache_source,
        "create_pipeline_shared",
    )
    if shared_creation.count("&state.compute_layout") != 1:
        errors.append(
            "gpu/vk/pipeline_cache.c3 must create one fixed compute layout "
            "during device setup"
        )
    if shared_creation.count(
        "state.generated_dispatch_layout = create_generated_work_layout("
    ) != 1:
        errors.append(
            "gpu/vk/pipeline_cache.c3 must create one generated dispatch "
            "layout during device setup"
        )

    compute_creation = function_body(
        pipeline_compute_source,
        "create_compute_pipeline_from_module",
    )
    if compute_creation.count("pipe_info.layout = state.compute_layout;") != 1:
        errors.append(
            "gpu/vk/pipeline_compute.c3 must use the singleton compute layout"
        )
    if "create_pipeline_layout(" in compute_creation:
        errors.append(
            "gpu/vk/pipeline_compute.c3 must not create per-pipeline layouts"
        )

    required_pipeline_key_tokens = (
        "ColorTargetKey[gpu::MAX_COLOR_ATTACHMENTS] color_targets;",
        "$assert ColorTargetKey::size == 36;",
        "$assert PipelineKey::size == 320;",
        "ShaderId vertex_shader;",
        "ShaderId fragment_shader;",
    )
    for token in required_pipeline_key_tokens:
        if token not in pipeline_cache_source:
            errors.append(
                "gpu/vk/pipeline_cache.c3 is missing immutable key shape "
                f"token {token}"
            )
    graphics_key = function_body(pipeline_cache_source, "build_graphics_key")
    for token in ("desc.colors", "target.blend", "target.write_mask", "desc.polygon_mode"):
        if token not in graphics_key:
            errors.append(
                "gpu/vk/pipeline_cache.c3 graphics key is missing immutable "
                f"state {token}"
            )
    reject_tokens(
        errors,
        "gpu/vk/pipeline_cache.c3",
        "build_graphics_key",
        graphics_key,
        ("desc.topology", "desc.raster", "depth_bias", "cull_mode", "front_face"),
    )
    pipeline_key_fields = struct_field_names(
        pipeline_cache_source,
        "PipelineKey",
    )
    if pipeline_key_fields != PIPELINE_KEY_FIELDS:
        errors.append(
            "gpu/vk/pipeline_cache.c3:PipelineKey must contain only the "
            "reviewed ID and immutable-state fields"
        )
    pipeline_cache_entry_fields = struct_field_names(
        pipeline_cache_source,
        "PipelineCacheEntry",
    )
    if pipeline_cache_entry_fields != PIPELINE_CACHE_ENTRY_FIELDS:
        errors.append(
            "gpu/vk/pipeline_cache.c3:PipelineCacheEntry must contain only "
            "the reviewed ID-keyed cache fields"
        )

    compute_key = function_body(pipeline_cache_source, "build_compute_key")
    if (".vertex_shader   = shader" not in compute_key
            or ".fragment_shader = SHADER_ID_INVALID" not in compute_key
            or "push_constant" in compute_key):
        errors.append(
            "gpu/vk/pipeline_cache.c3 compute key must use shader identity "
            "with the fixed root layout"
        )

    find_pipeline_entry = function_body(pipeline_cache_source, "find_entry")
    reject_tokens(
        errors,
        "gpu/vk/pipeline_cache.c3",
        "find_entry",
        find_pipeline_entry,
        (
            "gpu::ShaderCode",
            ".spirv",
            "mem::equals",
            "clone_shader",
            "free_cloned_shader",
            "shader_store",
        ),
    )
    find_entry_calls = set(re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        mask_c3_comments(find_pipeline_entry),
    )) - {"for", "if"}
    unexpected_find_entry_calls = find_entry_calls - {
        "bucket_for",
        "note_pipeline_key_probe",
        "key_equals",
    }
    if unexpected_find_entry_calls:
        errors.append(
            "gpu/vk/pipeline_cache.c3:find_entry must only hash/probe the "
            "canonical ID key; unexpected calls: "
            + ", ".join(sorted(unexpected_find_entry_calls))
        )

    shader_identity_equals = function_body(
        shader_source,
        "shader_store_entry_equals",
    )
    for token in (
        "entry.stage != code.stage",
        "entry.spirv.len != code.spirv.len",
        "entry_point",
        "mem::equals(entry.spirv, code.spirv)",
    ):
        if token not in shader_identity_equals:
            errors.append(
                "gpu/vk/shader.c3 collision verification is missing "
                f"{token}"
            )

    ownership_sources = (
        ("gpu/vk/command.c3", backend_source, BACKEND_OWNERSHIP_DIGESTS),
        (
            "gpu/vk/command_state.c3",
            command_state_source,
            COMMAND_STATE_OWNERSHIP_DIGESTS,
        ),
        ("gpu/vk/lifetime.c3", lifetime_source, LIFETIME_OWNERSHIP_DIGESTS),
        ("gpu/vk/device.c3", backend_device_source, DEVICE_OWNERSHIP_DIGESTS),
        ("gpu/vk/sync.c3", sync_source, SYNC_OWNERSHIP_DIGESTS),
        ("gpu/vk/queue.c3", queue_source, QUEUE_OWNERSHIP_DIGESTS),
    )
    for relative, source, digests in ownership_sources:
        for name, expected in digests.items():
            body = function_body(source, name)
            actual = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if actual != expected:
                errors.append(
                    f"{relative}:{name} generated preprocess ownership flow "
                    "must match reviewed source"
                )

    reviewed_ownership = {
        relative: digests
        for relative, _, digests in ownership_sources
    }
    for path in sorted((root / "gpu/vk").rglob("*.c3")):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        digests = reviewed_ownership.get(relative, {})
        for name in function_names(source):
            body = function_body(source, name)
            code_body = mask_c3_comments(body)
            touches_ownership = (
                "generated_preprocess" in name
                or "generated_preprocess" in code_body
                or any(
                    re.search(rf"\b{re.escape(callee)}\s*\(", code_body)
                    for callee in OWNERSHIP_TRANSFER_CALLEES
                )
            )
            if touches_ownership and name not in digests:
                errors.append(
                    f"{relative}:{name} generated preprocess ownership flow "
                    "is unreviewed"
                )

    for name in PUBLIC_COMMANDS:
        reject_tokens(
            errors,
            "gpu/command.c3",
            name,
            function_body(public_source, name),
            RECORDING_FORBIDDEN,
        )
    for name in BACKEND_COMMANDS:
        reject_tokens(
            errors,
            "gpu/vk/command.c3",
            name,
            function_body(backend_source, name),
            RECORDING_FORBIDDEN,
        )

    reject_tokens(
        errors,
        "gpu/vk/sync.c3",
        "vk_cmd_barrier",
        function_body(sync_source, "vk_cmd_barrier"),
        RECORDING_FORBIDDEN,
    )
    reject_tokens(
        errors,
        "gpu/vk/queue.c3",
        "reserve_queue_completion_locked",
        function_body(queue_source, "reserve_queue_completion_locked"),
        POINT_ALLOCATION_FORBIDDEN,
    )
    reject_tokens(
        errors,
        "gpu/vk/texture.c3",
        "vk_destroy_texture",
        function_body(texture_source, "vk_destroy_texture"),
        DESTRUCTION_FORBIDDEN,
    )

    acquire = function_body(
        backend_source,
        "acquire_generated_preprocess_buffer",
    )
    reject_tokens(
        errors,
        "gpu/vk/command.c3",
        "acquire_generated_preprocess_buffer",
        acquire,
        (
            "alloc::",
            "mem::new",
            "create_buffer_with_alignment(",
            "take_generated_preprocess_buffer(",
        ),
    )
    for token in (
        "generated_reservation_matches(reserved, pipeline, kind)",
        "max_count > reserved.reservation_max_commands",
        "reserved.reservation_in_use",
        "generated_preprocess_compatible(",
        "record.generated_preprocess_count++",
        "return gpu::GENERATED_SCRATCH_EXHAUSTED~;",
    ):
        if token not in acquire:
            errors.append(
                "gpu/vk/command.c3 generated scratch acquisition is missing "
                f"the bounded reservation step {token}"
            )
    execute = function_body(backend_source, "execute_generated_work")
    native_execute_call = (
        "state.device_dispatch.generated_work.cmd_execute_generated_commands(\n"
        "        record.command_buffer,\n"
        "        vk::FALSE,\n"
        "        &info,\n"
        "    );"
    )

    allowed_execute_record_lines = (
        "record:       record,",
        "record.command_buffer,",
    )
    execute_record_lines = tuple(
        line.strip()
        for line in execute.splitlines()
        if re.search(r"\brecord\b", line)
    )
    if (
        execute_record_lines != allowed_execute_record_lines
        or execute.count("acquire_generated_preprocess_buffer(") != 1
    ):
        errors.append(
            "gpu/vk/command.c3 generated execution must acquire a fresh buffer"
        )
    if (
        execute.count(native_execute_call) != 1
        or backend_source.count("cmd_execute_generated_commands") != 1
    ):
        errors.append(
            "gpu/vk/command.c3 generated execution must issue exactly one native call"
        )
    render = function_body(render_source, "vk_cmd_begin_render_pass")
    reject_tokens(
        errors,
        "gpu/vk/render_pass.c3",
        "vk_cmd_begin_render_pass",
        render,
        (
            "alloc::",
            "mem::new",
            "create_image_view(",
            "resolve_texture_view_tracked(",
        ),
    )
    begin = function_body(backend_source, "vk_begin_commands_with_context")
    try:
        reuse = begin.index("take_available_command_buffer(")
        allocate = begin.index("vk::allocate_command_buffers(")
        reset = begin.index("vk::reset_command_buffer(")
        if not (reuse < reset < allocate):
            errors.append(
                "gpu/vk/command.c3 warm command reuse must precede cold allocation"
            )
    except ValueError as error:
        errors.append(
            "gpu/vk/command.c3 command recycling is missing "
            f"{error.args[0]}"
        )
    if backend_source.count("vk::free_command_buffers(") != 1:
        errors.append(
            "gpu/vk/command.c3 may free command buffers only in failed cold begin"
        )
    seam_checks = (
        (
            "gpu/vk/command.c3:vk_begin_commands_with_context",
            begin,
            "vk::allocate_command_buffers(",
            "note_command_buffer_allocation(state);",
        ),
        (
            "gpu/vk/command.c3:vk_begin_commands_with_context",
            begin,
            "vk::free_command_buffers(",
            "note_command_buffer_free(state);",
        ),
        (
            "gpu/vk/command.c3:vk_begin_commands_with_context",
            begin,
            "vk::reset_command_buffer(",
            "note_command_buffer_reset(state);",
        ),
        (
            "gpu/vk/attachment_view.c3:vk_create_attachment_view",
            function_body(attachment_source, "vk_create_attachment_view"),
            "vk::create_image_view(",
            "state.recording_image_view_creations.add(",
        ),
        (
            "gpu/vk/command.c3:allocate_generated_preprocess_buffer",
            function_body(backend_source, "allocate_generated_preprocess_buffer"),
            "create_buffer_with_alignment(",
            "note_recording_vma_allocation(state);",
        ),
    )
    for label, body, native, counter in seam_checks:
        if body.count(native) != 1 or body.count(counter) != 1:
            errors.append(
                f"{label} must count its single native work seam {native}"
            )

    required_text = (
        (
            backend_source,
            "gpu/vk/command.c3",
            "generated_recording_allocations",
        ),
        (
            command_state_source,
            "gpu/vk/command_state.c3",
            "recycle_generated_preprocess_buffers_locked(",
        ),
        (
            lifetime_source,
            "gpu/vk/lifetime.c3",
            "recycle_generated_preprocess_buffers_locked(",
        ),
        (
            device_source,
            "gpu/device.c3",
            "device_registry_lock_acquisitions",
        ),
        (
            queue_source,
            "gpu/vk/queue.c3",
            "completion_point_allocations",
        ),
        (
            sync_source,
            "gpu/vk/sync.c3",
            "completion_wait_calls",
        ),
        (
            lifetime_source,
            "gpu/vk/lifetime.c3",
            "command_buffer_stats(",
        ),
        (
            command_bench,
            "test/src/command_record_bench.c3",
            "invariants: registry_locks=%d recording_allocations=%d "
            "draw_compilations=%d preprocess_allocations=%d",
        ),
        (
            command_bench,
            "test/src/command_record_bench.c3",
            "resolution: recording_commands=%d native_commands=%d device_registry=%d "
            "retained_pins=%d lifecycle_vtable=%d command_table=%d "
            "pipeline_table=%d pipeline_cache=%d policy=%d",
        ),
        (
            lifecycle_bench,
            "test/src/lifecycle_bench.c3",
            "invariants: point_allocations=%d destruction_waits=%d "
            "deferred_releases=%d cached_poll_queries=%d retirement_locks=%d",
        ),
        (
            lifecycle_bench,
            "test/src/lifecycle_bench.c3",
            "command_buffer_stats(",
        ),
    )
    for source, relative, token in required_text:
        if token not in source:
            errors.append(f"{relative} is missing performance evidence token {token}")

    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("performance contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
