#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_PROJECT = ROOT / "test" / "cpu"
CANONICAL_STRICT_SURFACE = CPU_PROJECT / "canonical_strict_surface.c3"
CANONICAL_STRICT_MANIFEST = CPU_PROJECT / "canonical_strict_surface.txt"
ROOT_FUNCTION_REFERENCE = re.compile(r"\bgpu::([a-z_][a-z0-9_]*)\s*\(")
MODULE_DECLARATION = re.compile(
    r"(?m)^\s*module\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*"
    r"(?:::[A-Za-z_][A-Za-z0-9_]*)*)\s*(?P<attributes>[^;]*);"
)

FORBIDDEN_TEXT = {
    "devicedesc": "retired transitional DeviceDesc",
    '"name":"create_device_from_desc"': "retired direct device creation",
    "backend_state": "backend state pointer",
    "backendvtable": "backend dispatch table",
    "descriptorheapmode": "backend heap strategy type",
    '"name":"descriptor_heap_mode"': "backend heap strategy field",
    '"name":"descriptor_buffer"': "backend descriptor-buffer capability",
    '"name":"descriptor_indexing"': "backend descriptor-indexing capability",
    '"name":"texture_descriptor_capacity"': "backend-shaped texture capacity",
    '"name":"sampler_descriptor_capacity"': "backend-shaped sampler capacity",
    '"name":"max_texture_descriptors"': "backend-shaped texture limit",
    '"name":"max_sampler_descriptors"': "backend-shaped sampler limit",
    '"name":"device_generated_commands"': "device_generated_commands",
    '"name":"supported_indirect_commands_shader_stages"': (
        "supported_indirect_commands_shader_stages"
    ),
    '"name":"max_indirect_sequence_count"': "max_indirect_sequence_count",
    '"name":"max_indirect_commands_token_count"': (
        "max_indirect_commands_token_count"
    ),
    '"name":"max_indirect_commands_token_offset"': (
        "max_indirect_commands_token_offset"
    ),
    '"name":"max_indirect_commands_indirect_stride"': (
        "max_indirect_commands_indirect_stride"
    ),
    "bufferhandle": "retired BufferHandle",
    "bufferdesc": "retired BufferDesc",
    "bufferusage": "retired BufferUsage",
    "bufferbarrier": "retired resource-scoped BufferBarrier",
    "globalbarrier": "retired Vulkan-shaped GlobalBarrier",
    "textureuse": "retired TextureUse cross-product",
    '"name":"prior_use"': "retired AcquiredImage.prior_use",
    '"name":"cmd_buffer_barrier"': "retired cmd_buffer_barrier",
    '"name":"cmd_global_barrier"': "retired cmd_global_barrier",
    "texturedescriptordesc": "retired TextureDescriptorDesc",
    '"name":"create_texture_descriptor"': (
        "retired texture descriptor lifecycle"
    ),
    '"name":"destroy_texture_descriptor"': (
        "retired texture descriptor lifecycle"
    ),
    '"name":"create_texture_descriptors"': (
        "retired texture descriptor lifecycle"
    ),
    "frametoken": "retired FrameToken",
    "memorykind": "retired MemoryKind",
    "begin_frame": "retired begin_frame",
    "alloc_frame_span": "retired alloc_frame_span",
    "end_frame": "retired end_frame",
    "with_frame": "retired with_frame",
    '"name":"default_frame_arena_size"': (
        "retired DEFAULT_FRAME_ARENA_SIZE"
    ),
    '"name":"frame_arena_size"': "retired DeviceDesc.frame_arena_size",
    '"name":"frames_in_flight"': "retired DeviceDesc.frames_in_flight",
    "arena_full": "retired ARENA_FULL",
    "create_buffer": "retired public buffer lifecycle",
    "destroy_buffer": "retired public buffer lifecycle",
    "get_buffer_address": "retired public buffer lifecycle",
    "get_buffer_span": "retired public buffer lifecycle",
    "flush_buffer": "retired public buffer lifecycle",
    "invalidate_buffer": "retired public buffer lifecycle",
    "platformkind": "retired PlatformKind",
    "presentdesc": "retired PresentDesc",
    "semaphoredesc": "retired public semaphore",
    "semaphorehandle": "retired public semaphore",
    "semaphoresignal": "retired public semaphore",
    "semaphorevalue": "retired public semaphore",
    "semaphorewait": "retired public semaphore",
    "create_semaphore": "retired public semaphore",
    "destroy_semaphore": "retired public semaphore",
    "wait_semaphore": "retired public semaphore",
    "wait_queue_idle": "retired wait_queue_idle",
    "timeline_semaphore": "retired timeline capability",
    "max_timeline_semaphore_value_difference": "retired timeline capability",
    "probe_vulkan_version": "Vulkan loader probe",
    "probe_vma_allocator": "VMA probe",
    "range_end": "readback retirement range",
    "recordingcontexthandle": "retired recording context",
    "recording_context": "retired recording context",
    "create_recording_context": "retired recording context",
    "destroy_recording_context": "retired recording context",
    "shared_queues": "backend queue-sharing policy",
    "surfacedesc": "retired SurfaceDesc",
    "ticket.value": "readback retirement value",
    "vk::": "Vulkan type",
    "vma::": "VMA type",
}

FORBIDDEN_SYMBOLS = {
    "DestroyDeviceFn",
    "GetMemoryStatsFn",
    "CreateBufferFn",
    "BeginCommandsFn",
    "SubmitFn",
    "CmdDispatchFn",
    "CmdDrawFn",
    "Stage",
    "Hazard",
    "CmdReadbackBufferFn",
    "ResolveReadbackFn",
    "create_wayland_surface",
    "create_win32_surface",
    "create_x11_surface",
    "RenderPass",
    "RenderPassHandle",
    "Framebuffer",
    "FramebufferHandle",
    "create_render_pass",
    "destroy_render_pass",
    "create_framebuffer",
    "destroy_framebuffer",
}

PLATFORM_HANDLE_TYPES = {
    "gpu::surface::win32": ("InstanceHandle", "WindowHandle"),
    "gpu::surface::wayland": ("DisplayHandle", "SurfaceHandle"),
    "gpu::surface::x11": ("DisplayHandle", "WindowHandle"),
}

DEBUG_RESOURCE_KINDS = (
    "NONE",
    "DEVICE",
    "GPU_SPAN",
    "TEXTURE",
    "PIPELINE",
    "SWAPCHAIN",
    "SHADER",
    "COMMAND_ALLOCATOR",
    "COMMAND_LIST",
    "TEXTURE_DESCRIPTOR",
    "SAMPLER",
    "ALLOCATION",
    "ATTACHMENT_VIEW",
)

RETIRED_SOURCE_SYMBOLS = (
    "DeviceDesc",
    "create_device_from_desc(",
    "PlatformKind",
    "PresentDesc",
    "SurfaceDesc",
    "BufferHandle",
    "BufferDesc",
    "BufferUsage",
    "BufferBarrier",
    "GlobalBarrier",
    "TextureUse",
    "prior_use",
    "before_stage",
    "after_stage",
    "before_hazard",
    "after_hazard",
    "old_layout",
    "new_layout",
    "prior_layout",
    "cmd_buffer_barrier(",
    "cmd_global_barrier(",
    "TextureDescriptorDesc",
    "DescriptorHeapMode",
    "descriptor_heap_mode",
    "texture_descriptor_capacity",
    "sampler_descriptor_capacity",
    "max_texture_descriptors",
    "max_sampler_descriptors",
    "descriptor_buffer",
    "descriptor_indexing",
    "DEFAULT_TEXTURE_DESCRIPTORS",
    "DEFAULT_SAMPLER_DESCRIPTORS",
    "MAX_DESCRIPTOR_SLOTS",
    "create_texture_descriptor(",
    "destroy_texture_descriptor(",
    "create_texture_descriptors(",
    "create_buffer(",
    "destroy_buffer(",
    "create_sampler(",
    "destroy_sampler(",
    "get_buffer_address(",
    "get_buffer_span(",
    "flush_buffer(",
    "invalidate_buffer(",
    "RecordingContextHandle",
    "RECORDING_CONTEXT",
    "SemaphoreDesc",
    "SemaphoreHandle",
    "SemaphoreSignal",
    "SemaphoreValue",
    "SemaphoreWait",
    "create_semaphore(",
    "create_recording_context(",
    "destroy_recording_context(",
    "destroy_semaphore(",
    "wait_semaphore(",
    "wait_queue_idle",
    "timeline_semaphore",
    "max_timeline_semaphore_value_difference",
    "SEMAPHORE,",
    "PersistentAllocDesc",
    "PersistentArenaStats",
    "MAX_PERSISTENT_ALLOCATIONS",
    "DEFAULT_PERSISTENT_ARENA_SIZE",
    "persistent_arena_size",
    "alloc_persistent_span(",
    "free_persistent_span(",
    "get_persistent_stats(",
    "PERSISTENT_SPAN",
    "FrameToken",
    "MemoryKind",
    "begin_frame(",
    "alloc_frame_span(",
    "end_frame(",
    "@with_frame",
    "DEFAULT_FRAME_ARENA_SIZE",
    "frame_arena_size",
    "frames_in_flight",
    "ARENA_FULL",
    "FRAME,",
    "TextureDimension",
    "D24_UNORM_S8_UINT",
    "ClearDepthStencil",
    "SAMPLER_INVALID",
    "publish_sampler(",
)

RETIRED_BACKEND_SOURCE_SYMBOLS = (
    "StandaloneDeviceConfig",
    "create_standalone_device_with_probe",
    "TextureUseScope",
    "texture_use_",
    "fn TextureBarrierRejection texture_barrier_rejection(",
    "fn TextureBarrierRejection texture_barrier_queue_rejection(",
    "texture_transition_range(",
    "texture_barrier_to_vk(",
    "ThreadRecordingContext",
    "thread_recording_contexts",
    "RecordingContextTable",
    "RECORDING_CONTEXTS",
    "MAX_RECORDING_CONTEXTS",
    "MAX_THREAD_DEVICE_CONTEXTS",
    "RetiredCommandBuffer",
)

RETIRED_SOURCE_PATTERNS = {
    r"struct\s+SubmitDesc\s*\{[^}]*\bwaits\b": "SubmitDesc.waits",
    r"struct\s+SubmitDesc\s*\{[^}]*\bsignals\b": "SubmitDesc.signals",
}


def public_entries(module: dict) -> dict:
    return {
        section: (
            [entry for entry in contents if entry.get("visibility") != "private"]
            if isinstance(contents, list) else contents
        )
        for section, contents in module.items()
    }


def validate_canonical_function_fixture(
    document: dict,
    source: str,
) -> list[str]:
    module = document.get("modules", {}).get("gpu")
    if module is None:
        return []
    public_functions = {
        entry.get("name")
        for entry in public_entries(module).get("functions", [])
        if entry.get("name")
    }
    referenced_functions = {
        match.group(1)
        for match in ROOT_FUNCTION_REFERENCE.finditer(source)
    }
    return [
        f"canonical strict fixture missing gpu::{name}"
        for name in sorted(public_functions - referenced_functions)
    ]


def canonical_public_entries(document: dict) -> set[str]:
    module = document.get("modules", {}).get("gpu")
    if module is None:
        return set()
    entries = public_entries(module)
    return {
        f"{entry['kind']} {entry['uid']}"
        for category in ("functions", "methods", "types", "variables")
        for entry in entries.get(category, [])
        if entry.get("kind") and entry.get("uid")
    }


def validate_canonical_surface_manifest(
    document: dict,
    source: str,
) -> list[str]:
    expected = {
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    actual = canonical_public_entries(document)
    failures = [
        f"canonical manifest entry missing from generated API: {entry}"
        for entry in sorted(expected - actual)
    ]
    failures.extend(
        f"generated public entry missing from canonical manifest: {entry}"
        for entry in sorted(actual - expected)
    )
    return failures


def member_schema(
    definition: dict | None,
) -> tuple[tuple[str | None, str | None], ...]:
    return tuple(
        (
            member.get("name"),
            member.get("type", {}).get("name"),
        )
        for member in (definition or {}).get("members", [])
    )


def validate_generated_backend_privacy(document: dict) -> list[str]:
    failures = []
    for module_name, module in document.get("modules", {}).items():
        if not (
            module_name == "gpu::vk"
            or module_name.startswith("gpu::vk::")
        ):
            continue
        for contents in module.values():
            if not isinstance(contents, list):
                continue
            for entry in contents:
                if not isinstance(entry, dict):
                    continue
                if entry.get("visibility") == "private":
                    continue
                identity = entry.get("uid") or (
                    f"{module_name}::{entry.get('name', '<anonymous>')}"
                )
                failures.append(f"generated {identity} must remain private")
    return failures


def validate_document(document: dict) -> list[str]:
    modules = document.get("modules", {})
    public_module = modules.get("gpu")
    if public_module is None:
        return ["missing gpu module"]

    public_surface = public_entries(public_module)
    encoded = json.dumps(public_surface, separators=(",", ":"))
    lowered = encoded.lower()
    failures = validate_generated_backend_privacy(document)
    failures.extend([
        label
        for token, label in FORBIDDEN_TEXT.items()
        if token in lowered
    ])
    failures.extend(
        symbol
        for symbol in FORBIDDEN_SYMBOLS
        if f'"{symbol}"' in encoded
    )

    functions = {
        entry.get("name"): entry
        for entry in public_surface.get("functions", [])
    }
    types = {
        entry.get("name"): entry
        for entry in public_surface.get("types", [])
    }
    variables = {
        entry.get("name"): entry
        for entry in public_surface.get("variables", [])
    }

    for retired_type in (
        "TextureDimension",
        "TextureDimensionSupport",
        "ClearDepthStencil",
        "Sampler",
    ):
        if retired_type in types:
            failures.append(f"retired public type {retired_type}")
    if "SAMPLER_INVALID" in variables:
        failures.append("retired public constant SAMPLER_INVALID")
    if "publish_sampler" in functions:
        failures.append("retired public function publish_sampler")

    format_definition = types.get("Format")
    expected_formats = (
        "UNDEFINED",
        "R8_UNORM",
        "R8_UINT",
        "RG8_UNORM",
        "RGBA8_UNORM",
        "RGBA8_SRGB",
        "BGRA8_UNORM",
        "BGRA8_SRGB",
        "R16_UINT",
        "R16_FLOAT",
        "RG16_FLOAT",
        "RGBA16_FLOAT",
        "R32_UINT",
        "R32_FLOAT",
        "RG32_FLOAT",
        "RGBA32_FLOAT",
        "D32_FLOAT",
    )
    if (
        format_definition is None
        or format_definition.get("kind") != "enum"
        or tuple(
            member.get("name")
            for member in format_definition.get("members", [])
        ) != expected_formats
    ):
        failures.append("Format must match the strict backend profile")

    sample_count = types.get("SampleCount")
    expected_sample_counts = (
        "ONE",
        "TWO",
        "FOUR",
        "EIGHT",
        "SIXTEEN",
        "THIRTY_TWO",
        "SIXTY_FOUR",
    )
    if (
        sample_count is None
        or sample_count.get("kind") != "enum"
        or tuple(
            member.get("name")
            for member in sample_count.get("members", [])
        ) != expected_sample_counts
    ):
        failures.append("SampleCount must define the strict sample counts")

    contract_validation = types.get("ContractValidation")
    if (
        contract_validation is None
        or contract_validation.get("kind") != "enum"
        or tuple(
            member.get("name")
            for member in contract_validation.get("members", [])
        ) != ("TRUSTED", "OBJECT_BOUNDARIES", "FULL")
    ):
        failures.append("ContractValidation must match the strict policy order")

    pipeline_schemas = (
        (
            "BlendState",
            "struct",
            (
                ("enable", "bool"),
                ("src_color", "BlendFactor"),
                ("dst_color", "BlendFactor"),
                ("color_op", "BlendOp"),
                ("src_alpha", "BlendFactor"),
                ("dst_alpha", "BlendFactor"),
                ("alpha_op", "BlendOp"),
            ),
        ),
        (
            "ColorTargetState",
            "struct",
            (
                ("format", "Format"),
                ("blend", "BlendState"),
                ("write_mask", "ColorWriteMask"),
            ),
        ),
        (
            "DynamicRasterState",
            "struct",
            (
                ("topology", "PrimitiveTopology"),
                ("cull_mode", "CullMode"),
                ("front_face", "FrontFace"),
                ("depth_bias_enable", "bool"),
                ("depth_bias_constant", "float"),
                ("depth_bias_slope", "float"),
                ("depth_bias_clamp", "float"),
            ),
        ),
        (
            "ComputePipelineDesc",
            "struct",
            (
                ("shader", "ShaderCode"),
                ("debug_name", "ZString"),
            ),
        ),
        (
            "GraphicsPipelineDesc",
            "struct",
            (
                ("vertex_shader", "ShaderCode"),
                ("fragment_shader", "ShaderCode"),
                ("colors", "ColorTargetState[]"),
                ("depth_format", "Format"),
                ("sample_count", "SampleCount"),
                ("polygon_mode", "PolygonMode"),
                ("debug_name", "ZString"),
            ),
        ),
    )
    for type_name, expected_kind, expected_members in pipeline_schemas:
        definition = types.get(type_name)
        if (
            definition is None
            or definition.get("kind") != expected_kind
            or member_schema(definition) != expected_members
        ):
            failures.append(f"{type_name} must match the strict schema")

    color_write_mask = types.get("ColorWriteMask", {})
    color_write_members = color_write_mask.get("members", [])
    color_write_mask_matches = (
        color_write_mask.get("kind") == "bitstruct"
        and color_write_mask.get("base_type", {}).get("name") == "uint"
        and tuple(
            (
                member.get("name"),
                member.get("type", {}).get("name"),
                member.get("bit_range"),
            )
            for member in color_write_members
        ) == (
            ("red", "bool", [0, 0]),
            ("green", "bool", [1, 1]),
            ("blue", "bool", [2, 2]),
            ("alpha", "bool", [3, 3]),
        )
    )
    if not color_write_mask_matches:
        failures.append("ColorWriteMask must match the strict channel bits")

    color_write_all = variables.get("COLOR_WRITE_ALL", {})
    if (
        color_write_all.get("kind") != "constant"
        or color_write_all.get("type", {}).get("name") != "ColorWriteMask"
        or color_write_all.get("value") != (
            "{ .red = true, .green = true, .blue = true, "
            ".alpha = true, }"
        )
    ):
        failures.append("COLOR_WRITE_ALL must enable every color channel")

    texture_schemas = (
        (
            "TextureDesc",
            (
                ("width", "uint"),
                ("height", "uint"),
                ("mip_levels", "uint"),
                ("array_layers", "uint"),
                ("format", "Format"),
                ("usage", "TextureUsage"),
                ("access", "QueueRoles"),
                ("sample_count", "SampleCount"),
                ("debug_name", "ZString"),
            ),
        ),
        (
            "TextureViewDesc",
            (
                ("base_mip", "uint"),
                ("mip_count", "uint"),
                ("base_layer", "uint"),
                ("layer_count", "uint"),
            ),
        ),
        (
            "TextureFormatSupport",
            (
                ("features", "TextureFormatFeatures"),
                ("sample_counts", "TextureSampleCountSupport"),
            ),
        ),
    )
    for type_name, expected_members in texture_schemas:
        if member_schema(types.get(type_name)) != expected_members:
            failures.append(f"{type_name} must match the strict schema")

    strict_render_enums = (
        (
            "LoadOp",
            ("LOAD", "CLEAR", "DONT_CARE"),
            "LoadOp must define the strict load operations",
        ),
        (
            "StoreOp",
            ("STORE", "DONT_CARE"),
            "StoreOp must define the strict store operations",
        ),
    )
    for type_name, expected_members, failure in strict_render_enums:
        definition = types.get(type_name)
        if (
            definition is None
            or definition.get("kind") != "enum"
            or tuple(
                member.get("name")
                for member in definition.get("members", [])
            ) != expected_members
        ):
            failures.append(failure)

    strict_render_schemas = (
        (
            "ClearColor",
            "union",
            (("rgba", "float[4]"), ("uint_rgba", "uint[4]")),
            "ClearColor must expose typed color values",
        ),
        (
            "ClearDepth",
            "struct",
            (("depth", "float"),),
            "ClearDepth must match the strict schema",
        ),
        (
            "AttachmentViewDesc",
            "struct",
            (
                ("texture", "TextureHandle"),
                ("mip_level", "uint"),
                ("array_layer", "uint"),
            ),
            "AttachmentViewDesc must match the strict schema",
        ),
        (
            "ColorTargetDesc",
            "struct",
            (
                ("view", "AttachmentViewHandle"),
                ("resolve_view", "AttachmentViewHandle"),
                ("load_op", "LoadOp"),
                ("store_op", "StoreOp"),
                ("clear", "ClearColor"),
            ),
            "ColorTargetDesc must match the strict schema",
        ),
        (
            "DepthTargetDesc",
            "struct",
            (
                ("view", "AttachmentViewHandle"),
                ("load_op", "LoadOp"),
                ("store_op", "StoreOp"),
                ("clear", "ClearDepth"),
            ),
            "DepthTargetDesc must match the strict schema",
        ),
        (
            "RenderPassDesc",
            "struct",
            (
                ("colors", "ColorTargetDesc[]"),
                ("depth", "DepthTargetDesc*"),
                ("width", "uint"),
                ("height", "uint"),
            ),
            "RenderPassDesc must match the strict schema",
        ),
    )
    for (
        type_name,
        expected_kind,
        expected_members,
        failure,
    ) in strict_render_schemas:
        definition = types.get(type_name)
        if (
            definition is None
            or definition.get("kind") != expected_kind
            or member_schema(definition) != expected_members
        ):
            failures.append(failure)

    runtime_desc_schema = (
        ("backend", "BackendKind"),
        ("contract_validation", "ContractValidation"),
        ("track_resource_lifetimes", "bool"),
        ("enable_vulkan_validation", "bool"),
        ("enable_debug_names", "bool"),
        ("texture_heap_capacity", "uint"),
        ("sampler_heap_capacity", "uint"),
        ("texture_capacity", "uint"),
        ("pipeline_capacity", "uint"),
        ("pipeline_cache_data", "char[]"),
        ("application_name", "ZString"),
        ("debug_callback", "DebugMessageCallback"),
        ("debug_user_data", "void*"),
    )
    if member_schema(types.get("RuntimeDesc")) != runtime_desc_schema:
        failures.append("RuntimeDesc must match the strict schema")

    device_caps_fields = dict(member_schema(types.get("DeviceCaps")))
    for field in (
        "texture_heap_capacity",
        "sampler_heap_capacity",
        "max_generated_work_count",
    ):
        if device_caps_fields.get(field) != "uint":
            failures.append(f"DeviceCaps.{field} must be a uint")
    if device_caps_fields.get("generated_work") != "bool":
        failures.append("DeviceCaps.generated_work must be a bool")
    if device_caps_fields.get("max_sampler_lod_bias") != "float":
        failures.append("DeviceCaps.max_sampler_lod_bias must be a float")

    begin_commands = functions.get("begin_commands")
    if begin_commands is None:
        failures.append("missing begin_commands")
    else:
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in begin_commands.get("members", [])
        )
        if parameter_types != ("CommandAllocator*",):
            failures.append(
                "begin_commands must take one CommandAllocator*"
            )
        if begin_commands.get("return_type", {}).get("name") != "CommandList?":
            failures.append("begin_commands must return CommandList?")

    end_commands = functions.get("end_commands")
    if end_commands is None:
        failures.append("missing end_commands")
    else:
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in end_commands.get("members", [])
        )
        if parameter_types != ("CommandList*",):
            failures.append("end_commands must consume CommandList*")
        if (end_commands.get("return_type", {}).get("name")
                != "ExecutableCommandList?"):
            failures.append(
                "end_commands must return ExecutableCommandList?"
            )

    begin_render_pass = functions.get("cmd_begin_render_pass")
    if begin_render_pass is None:
        failures.append("missing cmd_begin_render_pass")
    else:
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in begin_render_pass.get("members", [])
        )
        if parameter_types != ("CommandList*", "RenderPassDesc*"):
            failures.append(
                "cmd_begin_render_pass has the wrong parameters"
            )
        if begin_render_pass.get("return_type", {}).get("name") != "void?":
            failures.append(
                "cmd_begin_render_pass must return void?"
            )

    end_render_pass = functions.get("cmd_end_render_pass")
    if end_render_pass is None:
        failures.append("missing cmd_end_render_pass")
    else:
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in end_render_pass.get("members", [])
        )
        if parameter_types != ("CommandList*",):
            failures.append(
                "cmd_end_render_pass has the wrong parameters"
            )
        if end_render_pass.get("return_type", {}).get("name") != "void?":
            failures.append("cmd_end_render_pass must return void?")

    submit = functions.get("submit")
    if submit is None:
        failures.append("missing submit")
    else:
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in submit.get("members", [])
        )
        if parameter_types != ("Queue", "SubmitDesc*"):
            failures.append("submit must take Queue and SubmitDesc*")

    required_functions = {
        "create_command_allocator": (
            ("Device*", "Queue", "CommandAllocatorDesc*"),
            "CommandAllocator?",
        ),
        "destroy_command_allocator": (
            ("CommandAllocator*",),
            "void?",
        ),
        "allocate_memory": (
            ("Device*", "AllocationDesc*"),
            "GpuAllocation?",
        ),
        "intern_sampler": (
            ("Device*", "SamplerDesc*"),
            "SamplerIndex?",
        ),
        "create_texture_view": (
            ("Device*", "TextureHandle", "TextureViewDesc*"),
            "TextureView?",
        ),
        "destroy_texture_view": (
            ("Device*", "TextureView"),
            "void?",
        ),
        "create_texture_views": (
            ("Device*", "TextureViewCreateDesc[]", "TextureView[]"),
            "void?",
        ),
        "create_attachment_view": (
            ("Device*", "AttachmentViewDesc*"),
            "AttachmentViewHandle?",
        ),
        "destroy_attachment_view": (
            ("Device*", "AttachmentViewHandle"),
            "void?",
        ),
        "reserve_generated_scratch": (
            ("CommandAllocator*", "GeneratedScratchDesc*"),
            "void?",
        ),
        "release_generated_scratch": (
            (
                "CommandAllocator*",
                "PipelineHandle",
                "GeneratedWorkKind",
            ),
            "void?",
        ),
        "get_texture_requirements": (
            ("Device*", "TextureDesc*"),
            "TextureRequirements?",
        ),
        "create_placed_texture": (
            ("Device*", "TextureDesc*", "GpuAllocation", "usz"),
            "TextureHandle?",
        ),
        "create_dedicated_texture": (
            ("Device*", "TextureDesc*", "AllocationDesc*"),
            "DedicatedTexture?",
        ),
        "free_allocation": (
            ("Device*", "GpuAllocation*"),
            "void?",
        ),
        "get_allocation_info": (
            ("Device*", "GpuAllocation"),
            "AllocationInfo?",
        ),
        "get_allocation_span": (
            ("Device*", "GpuAllocation"),
            "GpuSpan?",
        ),
        "get_span_mapping": (
            ("Device*", "GpuSpan"),
            "char[]?",
        ),
        "get_span_address": (
            ("Device*", "GpuSpan"),
            "GpuAddress?",
        ),
        "flush_mapped_span": (
            ("Device*", "GpuSpan"),
            "void?",
        ),
        "invalidate_mapped_span": (
            ("Device*", "GpuSpan"),
            "void?",
        ),
        "cmd_bind_pipeline": (
            ("CommandList*", "PipelineHandle"),
            "void?",
        ),
        "cmd_set_depth_state": (
            ("CommandList*", "DepthState*"),
            "void?",
        ),
        "cmd_set_raster_state": (
            ("CommandList*", "DynamicRasterState*"),
            "void?",
        ),
        "cmd_dispatch": (
            ("CommandList*", "GpuAddress", "Vec3u"),
            "void?",
        ),
        "cmd_draw": (
            ("CommandList*", "GpuAddress", "GpuAddress", "uint", "uint"),
            "void?",
        ),
        "cmd_copy_buffer": (
            ("CommandList*", "BufferCopyDesc*"),
            "void?",
        ),
        "cmd_fill_buffer": (
            ("CommandList*", "GpuSpan", "uint"),
            "void?",
        ),
        "cmd_barrier": (
            ("CommandList*", "Barrier*"),
            "void?",
        ),
        "cmd_texture_barrier": (
            ("CommandList*", "TextureBarrier*"),
            "void?",
        ),
        "sampled_at": (
            ("StageMask",),
            "TextureState",
        ),
        "storage_at": (
            ("StageMask", "TextureAccess"),
            "TextureState",
        ),
        "texture_transition": (
            ("TextureHandle", "TextureState", "TextureState"),
            "TextureBarrier?",
        ),
        "texture_view_transition": (
            ("TextureHandle", "TextureViewDesc", "TextureState", "TextureState"),
            "TextureBarrier?",
        ),
        "cmd_copy_buffer_to_texture": (
            ("CommandList*", "BufferTextureCopyDesc*"),
            "void?",
        ),
        "cmd_copy_texture_to_buffer": (
            ("CommandList*", "TextureBufferCopyDesc*"),
            "void?",
        ),
        "cmd_draw_indexed": (
            (
                "CommandList*",

                "GpuAddress",
                "GpuAddress",
                "GpuSpan",
                "uint",
                "uint",
                "IndexType",
            ),
            "void?",
        ),
        "cmd_draw_indirect": (
            (
                "CommandList*",

                "GpuAddress",
                "GpuAddress",
                "GpuSpan",
                "uint",
            ),
            "void?",
        ),
        "cmd_draw_indexed_indirect": (
            (
                "CommandList*",

                "GpuAddress",
                "GpuAddress",
                "GpuSpan",
                "uint",
                "GpuSpan",
                "IndexType",
            ),
            "void?",
        ),
        "cmd_draw_indexed_indirect_count": (
            (
                "CommandList*",

                "GpuAddress",
                "GpuAddress",
                "GpuSpan",
                "GpuSpan",
                "uint",
                "GpuSpan",
                "IndexType",
            ),
            "void?",
        ),
        "cmd_dispatch_indirect": (
            (
                "CommandList*",

                "GpuAddress",
                "GpuSpan",
            ),
            "void?",
        ),
    }
    required_parameter_names = {
        "intern_sampler": ("device", "desc"),
        "create_texture_view": ("device", "texture", "desc"),
        "destroy_texture_view": ("device", "view"),
        "create_texture_views": ("device", "descs", "out_views"),
        "create_attachment_view": ("device", "desc"),
        "destroy_attachment_view": ("device", "view"),
        "reserve_generated_scratch": ("allocator", "desc"),
        "release_generated_scratch": ("allocator", "pipeline", "kind"),
        "get_texture_requirements": ("device", "desc"),
        "create_placed_texture": ("device", "desc", "allocation", "offset"),
        "create_dedicated_texture": (
            "device",
            "desc",
            "allocation_desc",
        ),
        "cmd_bind_pipeline": ("commands", "pipeline"),
        "cmd_set_depth_state": ("commands", "depth"),
        "cmd_set_raster_state": ("commands", "raster"),
        "cmd_dispatch": ("commands", "root", "groups"),
        "cmd_draw": (
            "commands",
            "vertex_root",
            "fragment_root",
            "vertex_count",
            "instance_count",
        ),
        "cmd_copy_buffer": ("commands", "desc"),
        "cmd_fill_buffer": ("commands", "dst", "value"),
        "cmd_barrier": ("commands", "barrier"),
        "cmd_texture_barrier": ("commands", "barrier"),
        "sampled_at": ("stages",),
        "storage_at": ("stages", "access"),
        "texture_transition": ("texture", "before", "after"),
        "texture_view_transition": ("texture", "view", "before", "after"),
        "cmd_copy_buffer_to_texture": ("commands", "desc"),
        "cmd_copy_texture_to_buffer": ("commands", "desc"),
        "cmd_draw_indexed": (
            "commands",

            "vertex_root",
            "fragment_root",
            "index_span",
            "index_count",
            "instance_count",
            "index_type",
        ),
        "cmd_draw_indirect": (
            "commands",

            "vertex_root",
            "fragment_root",
            "args",
            "draw_count",
        ),
        "cmd_draw_indexed_indirect": (
            "commands",

            "vertex_root",
            "fragment_root",
            "args",
            "draw_count",
            "index_span",
            "index_type",
        ),
        "cmd_draw_indexed_indirect_count": (
            "commands",

            "vertex_root",
            "fragment_root",
            "args",
            "count_span",
            "max_draw_count",
            "index_span",
            "index_type",
        ),
        "cmd_dispatch_indirect": (
            "commands",

            "root",
            "args",
        ),
    }
    for name, contract in required_functions.items():
        expected_parameters, expected_return = contract
        function = functions.get(name)
        if function is None:
            failures.append(f"missing {name}")
            continue
        members = function.get("members", [])
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in members
        )
        parameter_names = tuple(member.get("name") for member in members)
        expected_names = required_parameter_names.get(name)
        if (parameter_types != expected_parameters
                or (expected_names is not None
                    and parameter_names != expected_names)):
            failures.append(f"{name} has the wrong parameters")
        if function.get("return_type", {}).get("name") != expected_return:
            failures.append(f"{name} has the wrong return type")

    for name, kind in (
        ("GpuAllocation", "struct"),
        ("TextureView", "struct"),
        ("TextureIndex", "bitstruct"),
        ("SamplerIndex", "bitstruct"),
        ("TextureViewCreateDesc", "struct"),
        ("AttachmentViewHandle", "struct"),
        ("AttachmentViewDesc", "struct"),
        ("GeneratedWorkKind", "enum"),
        ("GeneratedScratchDesc", "struct"),
        ("CommandAllocatorHandle", "struct"),
        ("CommandAllocator", "struct"),
        ("CommandAllocatorDesc", "struct"),
        ("GpuSpan", "struct"),
        ("MemoryClass", "enum"),
        ("MemoryStats", "struct"),
        ("DebugResourceKind", "enum"),
        ("AllocationDesc", "struct"),
        ("TextureCompatibility", "distinct type"),
        ("TextureRequirements", "struct"),
        ("DedicatedTexture", "struct"),
        ("AllocationInfo", "struct"),
        ("BufferCopyDesc", "struct"),
        ("BufferTextureCopyDesc", "struct"),
        ("TextureBufferCopyDesc", "struct"),
        ("StageMask", "bitstruct"),
        ("HazardFlags", "bitstruct"),
        ("Barrier", "struct"),
        ("TextureLayout", "enum"),
        ("TextureAccess", "bitstruct"),
        ("TextureState", "struct"),
        ("TextureBarrier", "struct"),
        ("GeneratedDrawRecord", "struct"),
        ("GeneratedDrawIndexedRecord", "struct"),
        ("GeneratedDispatchRecord", "struct"),
    ):
        definition = types.get(name)
        if definition is None or definition.get("kind") != kind:
            failures.append(f"missing {name} {kind}")

    texture_compatibility = types.get("TextureCompatibility", {})
    if texture_compatibility.get("base_type", {}).get("name") != "uint128":
        failures.append(
            "TextureCompatibility must be an opaque uint128 distinct type"
        )

    public_type_schemas = {
        "GpuSpan": (
            (
                ("owner", "ulong"),
                ("index", "uint"),
                ("generation", "uint"),
                ("offset", "usz"),
                ("size", "usz"),
            ),
            "GpuSpan must match the exact identity/range schema",
        ),
        "GpuAllocation": (
            (
                ("owner", "ulong"),
                ("index", "uint"),
                ("generation", "uint"),
            ),
            "GpuAllocation must match the exact identity schema",
        ),
        "AttachmentViewHandle": (
            (
                ("owner", "ulong"),
                ("index", "uint"),
                ("generation", "uint"),
            ),
            "AttachmentViewHandle must match the exact identity schema",
        ),
        "GeneratedScratchDesc": (
            (
                ("pipeline", "PipelineHandle"),
                ("kind", "GeneratedWorkKind"),
                ("max_commands_per_list", "uint"),
                ("preprocess_buffer_count", "uint"),
            ),
            "GeneratedScratchDesc must match the exact public schema",
        ),
        "CommandAllocatorHandle": (
            (
                ("owner", "ulong"),
                ("index", "uint"),
                ("generation", "uint"),
            ),
            "CommandAllocatorHandle must match the exact identity schema",
        ),
        "CommandAllocator": (
            (
                ("device", "Device"),
                ("queue", "Queue"),
                ("handle", "CommandAllocatorHandle"),
            ),
            "CommandAllocator must match the exact public schema",
        ),
        "CommandAllocatorDesc": (
            (
                ("command_buffer_capacity", "uint"),
                ("max_resource_references_per_list", "uint"),
                (
                    "max_generated_preprocess_buffers_per_list",
                    "uint",
                ),
                ("generated_preprocess_bytes", "usz"),
                ("debug_name", "ZString"),
            ),
            "CommandAllocatorDesc must match the exact public schema",
        ),
        "TextureView": (
            (
                ("owner", "ulong"),
                ("index", "TextureIndex"),
                ("generation", "uint"),
            ),
            "TextureView must match the exact identity schema",
        ),
        "TextureViewCreateDesc": (
            (
                ("texture", "TextureHandle"),
                ("view", "TextureViewDesc"),
            ),
            "TextureViewCreateDesc must match the exact public schema",
        ),
        "AllocationDesc": (
            (
                ("size", "usz"),
                ("alignment", "usz"),
                ("memory_class", "MemoryClass"),
                ("access", "QueueRoles"),
                ("texture_requirements", "TextureRequirements[]"),
                ("debug_name", "ZString"),
            ),
            "AllocationDesc must match the exact public schema",
        ),
        "TextureRequirements": (
            (
                ("size", "usz"),
                ("alignment", "usz"),
                ("compatibility", "TextureCompatibility"),
                ("dedicated_only", "bool"),
            ),
            "TextureRequirements must match the exact public schema",
        ),
        "DedicatedTexture": (
            (
                ("texture", "TextureHandle"),
                ("allocation", "GpuAllocation"),
            ),
            "DedicatedTexture must expose exactly two ownership tokens",
        ),
        "AllocationInfo": (
            (
                ("size", "usz"),
                ("alignment", "usz"),
                ("memory_class", "MemoryClass"),
                ("access", "QueueRoles"),
                ("mapped", "bool"),
                ("coherent", "bool"),
                ("addressable", "bool"),
            ),
            "AllocationInfo must match the exact public schema",
        ),
        "BufferCopyDesc": (
            (
                ("src", "GpuSpan"),
                ("dst", "GpuSpan"),
            ),
            "BufferCopyDesc must contain exactly source and destination spans",
        ),
        "GeneratedDrawRecord": (
            (
                ("vertex_root_gpu", "GpuAddress"),
                ("fragment_root_gpu", "GpuAddress"),
                ("arguments", "DrawIndirectCommand"),
            ),
            "GeneratedDrawRecord must match the generated ABI schema",
        ),
        "GeneratedDrawIndexedRecord": (
            (
                ("vertex_root_gpu", "GpuAddress"),
                ("fragment_root_gpu", "GpuAddress"),
                ("arguments", "DrawIndexedIndirectCommand"),
                ("_pad0", "uint"),
            ),
            "GeneratedDrawIndexedRecord must match the generated ABI schema",
        ),
        "GeneratedDispatchRecord": (
            (
                ("root_gpu", "GpuAddress"),
                ("arguments", "DispatchIndirectCommand"),
                ("_pad0", "uint"),
            ),
            "GeneratedDispatchRecord must match the generated ABI schema",
        ),
        "BufferTextureCopyDesc": (
            (
                ("src", "GpuSpan"),
                ("row_length_texels", "uint"),
                ("texture", "TextureHandle"),
                ("mip", "uint"),
                ("base_layer", "uint"),
                ("layer_count", "uint"),
                ("x", "uint"),
                ("y", "uint"),
                ("width", "uint"),
                ("height", "uint"),
            ),
            "BufferTextureCopyDesc must contain one source span",
        ),
        "TextureBufferCopyDesc": (
            (
                ("texture", "TextureHandle"),
                ("dst", "GpuSpan"),
                ("row_length_texels", "uint"),
                ("mip", "uint"),
                ("base_layer", "uint"),
                ("layer_count", "uint"),
                ("x", "uint"),
                ("y", "uint"),
                ("width", "uint"),
                ("height", "uint"),
            ),
            "TextureBufferCopyDesc must contain one destination span",
        ),
        "Barrier": (
            (
                ("before", "StageMask"),
                ("after", "StageMask"),
                ("hazards", "HazardFlags"),
            ),
            "Barrier must contain only semantic stage masks and hazard flags",
        ),
        "TextureLayout": (
            (
                ("UNDEFINED", "TextureLayout"),
                ("TRANSFER_SOURCE", "TextureLayout"),
                ("TRANSFER_DESTINATION", "TextureLayout"),
                ("SAMPLED", "TextureLayout"),
                ("STORAGE", "TextureLayout"),
                ("COLOR_ATTACHMENT", "TextureLayout"),
                ("DEPTH_ATTACHMENT", "TextureLayout"),
                ("PRESENT", "TextureLayout"),
            ),
            "TextureLayout must match the exact semantic layout schema",
        ),
        "TextureAccess": (
            (
                ("read", "bool"),
                ("write", "bool"),
            ),
            "TextureAccess must match the exact semantic access schema",
        ),
        "TextureState": (
            (
                ("layout", "TextureLayout"),
                ("stages", "StageMask"),
                ("access", "TextureAccess"),
            ),
            "TextureState must contain exactly layout, stages, and access",
        ),
        "TextureBarrier": (
            (
                ("texture", "TextureHandle"),
                ("view", "TextureViewDesc"),
                ("before", "TextureState"),
                ("after", "TextureState"),
            ),
            "TextureBarrier must contain only a texture range and compositional states",
        ),
        "MemoryStats": (
            (
                ("heaps", "MemoryHeapBudget[16]"),
                ("heap_count", "uint"),
                ("texture_count", "ulong"),
                ("live_allocation_count", "ulong"),
            ),
            "MemoryStats must match its exact public schema",
        ),
        "MemoryClass": (
            (
                ("CPU_WRITE", "MemoryClass"),
                ("GPU_PRIVATE", "MemoryClass"),
                ("CPU_READ", "MemoryClass"),
                ("TEXTURE", "MemoryClass"),
            ),
            "MemoryClass must expose exactly the four semantic values",
        ),
        "DebugResourceKind": (
            tuple(
                (name, "DebugResourceKind")
                for name in DEBUG_RESOURCE_KINDS
            ),
            "DebugResourceKind must match its exact schema",
        ),
    }
    for name, (expected_schema, failure) in public_type_schemas.items():
        if member_schema(types.get(name)) != expected_schema:
            failures.append(failure)

    texture_layout = types.get("TextureLayout", {})
    if texture_layout.get("base_type", {}).get("name") != "int":
        failures.append("TextureLayout must use int as its base type")

    for name in ("TextureIndex", "SamplerIndex"):
        definition = types.get(name, {})
        members = definition.get("members", [])
        raw_index_matches = (
            definition.get("kind") == "bitstruct"
            and definition.get("base_type", {}).get("name") == "uint"
            and len(members) == 1
            and members[0].get("name") == "value"
            and members[0].get("type", {}).get("name") == "uint"
            and members[0].get("bit_range") == [0, 31]
        )
        if not raw_index_matches:
            failures.append(
                f"{name} must be a generation-free 32-bit shader index"
            )

    expected_flag_schemas = {
        "StageMask": (
            "all",
            "host",
            "transfer",
            "compute",
            "vertex_shader",
            "fragment_shader",
            "color_output",
            "depth_output",
            "present",
        ),
        "HazardFlags": (
            "draw_arguments",
            "descriptors",
            "depth_stencil",
        ),
        "TextureAccess": (
            "read",
            "write",
        ),
    }
    for name, expected_names in expected_flag_schemas.items():
        definition = types.get(name, {})
        members = definition.get("members", [])
        exact_flags = (
            definition.get("kind") == "bitstruct"
            and definition.get("base_type", {}).get("name") == "uint"
            and tuple(member.get("name") for member in members)
                == expected_names
            and all(
                member.get("type", {}).get("name") == "bool"
                and member.get("bit_range") == [bit, bit]
                for bit, member in enumerate(members)
            )
        )
        if not exact_flags:
            failures.append(f"{name} must match the exact semantic flag schema")

    present = functions.get("present")
    if present is None:
        failures.append("missing present")
    else:
        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in present.get("members", [])
        )
        if parameter_types != (
            "Device*",
            "AcquiredImage*",
            "CompletionPoint",
        ):
            failures.append(
                "present must consume AcquiredImage* with CompletionPoint"
            )

    executable = types.get("ExecutableCommandList")
    if executable is None or executable.get("kind") != "struct":
        failures.append("missing ExecutableCommandList token")

    if member_schema(types.get("CompletionWait")) != (
        ("point", "CompletionPoint"),
        ("before", "StageMask"),
    ):
        failures.append("CompletionWait must match the exact stage-scoped schema")

    submit_desc = types.get("SubmitDesc")
    submit_schema = member_schema(submit_desc)
    if submit_schema != (
        ("command_lists", "ExecutableCommandList[]"),
        ("completion_waits", "CompletionWait[]"),
        ("readiness", "SwapchainReadiness"),
        ("readiness_before", "StageMask"),
    ):
        failures.append("SubmitDesc must match the exact stage-scoped schema")
    readiness = types.get("SwapchainReadiness")
    if readiness is None or readiness.get("kind") != "struct":
        failures.append("missing SwapchainReadiness token")

    acquired = types.get("AcquiredImage")
    if member_schema(acquired) != (
        ("texture", "TextureHandle"),
        ("attachment_view", "AttachmentViewHandle"),
        ("readiness", "SwapchainReadiness"),
        ("index", "uint"),
        ("suboptimal", "bool"),
        ("prior_state", "TextureState"),
    ):
        failures.append(
            "AcquiredImage must carry borrowed render handles, readiness, and compositional prior state"
        )

    for module_name, handle_names in PLATFORM_HANDLE_TYPES.items():
        module = modules.get(module_name)
        if module is None:
            failures.append(f"missing {module_name}")
            continue

        surface = public_entries(module)
        definitions = {
            entry.get("name"): entry
            for entry in surface.get("types", [])
        }
        for handle_name in handle_names:
            definition = definitions.get(handle_name)
            if definition is None or definition.get("kind") != "distinct type":
                failures.append(
                    f"{module_name}::{handle_name} must be a distinct type"
                )

        create_surface = next(
            (
                entry
                for entry in surface.get("functions", [])
                if entry.get("name") == "create_surface"
            ),
            None,
        )
        if create_surface is None:
            failures.append(f"missing {module_name}::create_surface")
            continue

        parameter_types = tuple(
            member.get("type", {}).get("name")
            for member in create_surface.get("members", [])
        )
        expected_types = ("Runtime*", *handle_names)
        if parameter_types != expected_types:
            failures.append(
                f"{module_name}::create_surface must use typed platform handles"
            )

    return failures


def scan_retired_source_symbols() -> list[str]:
    failures = []
    for path in sorted((ROOT / "gpu").rglob("*.c3")):
        if "vk" in path.relative_to(ROOT / "gpu").parts:
            continue
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        for symbol in RETIRED_SOURCE_SYMBOLS:
            if symbol in source:
                failures.append(f"retired {symbol} in {relative}")
        for pattern, label in RETIRED_SOURCE_PATTERNS.items():
            if re.search(pattern, source, re.DOTALL):
                failures.append(f"retired {label} in {relative}")
    return failures


def validate_private_backend_source(relative: Path, source: str) -> list[str]:
    failures = []
    normalized = source.lstrip("﻿")
    module_declarations = list(MODULE_DECLARATION.finditer(normalized))
    if not module_declarations:
        failures.append(
            f"{relative.as_posix()} must declare the private gpu::vk backend module"
        )
    for declaration in module_declarations:
        name = declaration.group("name")
        line_number = normalized.count(
            "\n",
            0,
            declaration.start(),
        ) + 1
        if name != "gpu::vk" and not name.startswith("gpu::vk::"):
            failures.append(
                f"{relative.as_posix()}:{line_number} "
                "backend file may only declare gpu::vk modules, "
                f"found {name}"
            )
            continue
        if "@private" not in declaration.group("attributes").split():
            failures.append(
                f"{relative.as_posix()} must declare the private gpu::vk backend module"
            )

    for line_number, line in enumerate(normalized.splitlines(), start=1):
        stripped = line.strip()
        if (
            "@public" in stripped
            and not stripped.startswith("import ")
            and not stripped.startswith("module ")
        ):
            failures.append(
                f"{relative.as_posix()}:{line_number} "
                "backend declaration may not use @public"
            )

    failures.extend(
        f"retired backend {symbol} in {relative.as_posix()}"
        for symbol in RETIRED_BACKEND_SOURCE_SYMBOLS
        if symbol in source
    )
    return failures


def expected_public_module(relative: Path) -> str:
    parts = relative.parts
    if (
        len(parts) >= 4
        and parts[:2] == ("gpu", "surface")
        and parts[2] in {"wayland", "win32", "x11"}
    ):
        return f"gpu::surface::{parts[2]}"
    return "gpu"


def validate_public_module_source(
    relative: Path,
    source: str,
) -> list[str]:
    normalized = source.lstrip("﻿")
    module_declarations = list(MODULE_DECLARATION.finditer(normalized))
    expected = expected_public_module(relative)
    if not module_declarations:
        return [
            f"{relative.as_posix()} must declare public module {expected}"
        ]
    failures = []
    for declaration in module_declarations:
        name = declaration.group("name")
        if name == expected:
            continue
        line_number = normalized.count(
            "\n",
            0,
            declaration.start(),
        ) + 1
        failures.append(
            f"{relative.as_posix()}:{line_number} "
            f"public source may only declare {expected}, found {name}"
        )
    return failures


def is_private_backend_source(relative: Path) -> bool:
    return relative.parts[:2] == ("gpu", "vk")


def scan_public_module_sources() -> list[str]:
    failures = []
    for path in sorted((ROOT / "gpu").rglob("*")):
        relative = path.relative_to(ROOT)
        if (
            path.is_file()
            and path.suffix in {".c3", ".c3i"}
            and not is_private_backend_source(relative)
        ):
            failures.extend(
                validate_public_module_source(
                    relative,
                    path.read_text(encoding="utf-8"),
                )
            )
    return failures


def scan_private_backend_modules() -> list[str]:
    failures = []
    for path in sorted((ROOT / "gpu" / "vk").rglob("*")):
        if not path.is_file() or path.suffix not in {".c3", ".c3i"}:
            continue
        failures.extend(
            validate_private_backend_source(
                path.relative_to(ROOT),
                path.read_text(encoding="utf-8"),
            )
        )
    return failures


def main() -> int:
    result = subprocess.run(
        ["c3c", "docgen", "--json"],
        cwd=CPU_PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        return result.returncode

    document = json.loads(result.stdout)
    failures = validate_document(document)
    failures.extend(
        validate_canonical_function_fixture(
            document,
            CANONICAL_STRICT_SURFACE.read_text(encoding="utf-8"),
        )
    )
    failures.extend(
        validate_canonical_surface_manifest(
            document,
            CANONICAL_STRICT_MANIFEST.read_text(encoding="utf-8"),
        )
    )
    failures.extend(scan_retired_source_symbols())
    failures.extend(scan_public_module_sources())
    failures.extend(scan_private_backend_modules())
    if failures:
        print("public GPU API contract violations:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("public GPU API matches the strict contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
