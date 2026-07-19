from __future__ import annotations

import unittest
from pathlib import Path

from scripts import check_public_api


def api_function(
    name: str,
    return_type: str,
    *parameters: tuple[str, str],
) -> dict:
    return {
        "name": name,
        "return_type": {"name": return_type},
        "members": [
            {"name": parameter, "type": {"name": type_name}}
            for parameter, type_name in parameters
        ],
    }


def surface_module(*handles: str) -> dict:
    return {
        "functions": [{
            "name": "create_surface",
            "members": [
                {"name": "runtime", "type": {"name": "Runtime*"}},
                *[
                    {"name": name.lower(), "type": {"name": name}}
                    for name in handles
                ],
            ],
        }],
        "types": [
            {"name": name, "kind": "distinct type"}
            for name in handles
        ],
    }


def valid_document() -> dict:
    return {
        "modules": {
            "gpu": {
                "functions": [
                    {
                        "name": "begin_commands",
                        "return_type": {"name": "CommandList?"},
                        "members": [
                            {"name": "queue", "type": {"name": "Queue"}},
                        ],
                    },
                    {
                        "name": "end_commands",
                        "return_type": {
                            "name": "ExecutableCommandList?",
                        },
                        "members": [{
                            "name": "commands",
                            "type": {"name": "CommandList*"},
                        }],
                    },
                    {
                        "name": "submit",
                        "return_type": {"name": "CompletionPoint?"},
                        "members": [
                            {"name": "queue", "type": {"name": "Queue"}},
                            {
                                "name": "desc",
                                "type": {"name": "SubmitDesc*"},
                            },
                        ],
                    },
                    {
                        "name": "present",
                        "return_type": {"name": "void?"},
                        "members": [
                            {"name": "device", "type": {"name": "Device*"}},
                            {
                                "name": "image",
                                "type": {"name": "AcquiredImage*"},
                            },
                            {
                                "name": "render_completion",
                                "type": {"name": "CompletionPoint"},
                            },
                        ],
                    },
                    {
                        "name": "allocate_memory",
                        "return_type": {"name": "GpuAllocation?"},
                        "members": [
                            {"name": "device", "type": {"name": "Device*"}},
                            {
                                "name": "desc",
                                "type": {"name": "AllocationDesc*"},
                            },
                        ],
                    },
                    api_function(
                        "intern_sampler",
                        "Sampler?",
                        ("device", "Device*"),
                        ("desc", "SamplerDesc*"),
                    ),
                    api_function(
                        "publish_sampler",
                        "SamplerIndex?",
                        ("device", "Device*"),
                        ("sampler", "Sampler"),
                    ),
                    api_function(
                        "create_texture_view",
                        "TextureView?",
                        ("device", "Device*"),
                        ("texture", "TextureHandle"),
                        ("desc", "TextureViewDesc*"),
                    ),
                    api_function(
                        "destroy_texture_view",
                        "void?",
                        ("device", "Device*"),
                        ("view", "TextureView"),
                    ),
                    api_function(
                        "create_texture_views",
                        "void?",
                        ("device", "Device*"),
                        ("descs", "TextureViewCreateDesc[]"),
                        ("out_views", "TextureView[]"),
                    ),
                    api_function(
                        "get_texture_requirements",
                        "TextureRequirements?",
                        ("device", "Device*"),
                        ("desc", "TextureDesc*"),
                    ),
                    api_function(
                        "create_placed_texture",
                        "TextureHandle?",
                        ("device", "Device*"),
                        ("desc", "TextureDesc*"),
                        ("allocation", "GpuAllocation"),
                        ("offset", "usz"),
                    ),
                    api_function(
                        "create_dedicated_texture",
                        "DedicatedTexture?",
                        ("device", "Device*"),
                        ("desc", "TextureDesc*"),
                        ("allocation_desc", "AllocationDesc*"),
                    ),
                    {
                        "name": "free_allocation",
                        "return_type": {"name": "void?"},
                        "members": [
                            {"name": "device", "type": {"name": "Device*"}},
                            {
                                "name": "allocation",
                                "type": {"name": "GpuAllocation*"},
                            },
                        ],
                    },
                    {
                        "name": "get_allocation_info",
                        "return_type": {"name": "AllocationInfo?"},
                        "members": [
                            {"name": "device", "type": {"name": "Device*"}},
                            {
                                "name": "allocation",
                                "type": {"name": "GpuAllocation"},
                            },
                        ],
                    },
                    {
                        "name": "get_allocation_span",
                        "return_type": {"name": "GpuSpan?"},
                        "members": [
                            {"name": "device", "type": {"name": "Device*"}},
                            {
                                "name": "allocation",
                                "type": {"name": "GpuAllocation"},
                            },
                        ],
                    },
                    {
                        "name": "get_span_mapping",
                        "return_type": {"name": "char[]?"},
                        "members": [
                            {"name": "device", "type": {"name": "Device*"}},
                            {"name": "span", "type": {"name": "GpuSpan"}},
                        ],
                    },
                    {
                        "name": "get_span_address",
                        "return_type": {"name": "GpuAddress?"},
                        "members": [
                            {"name": "device", "type": {"name": "Device*"}},
                            {"name": "span", "type": {"name": "GpuSpan"}},
                        ],
                    },
                    {
                        "name": "flush_mapped_span",
                        "return_type": {"name": "void?"},
                        "members": [
                            {"name": "device", "type": {"name": "Device*"}},
                            {"name": "span", "type": {"name": "GpuSpan"}},
                        ],
                    },
                    {
                        "name": "invalidate_mapped_span",
                        "return_type": {"name": "void?"},
                        "members": [
                            {"name": "device", "type": {"name": "Device*"}},
                            {"name": "span", "type": {"name": "GpuSpan"}},
                        ],
                    },
                    api_function(
                        "cmd_copy_buffer",
                        "void?",
                        ("commands", "CommandList*"),
                        ("desc", "BufferCopyDesc*"),
                    ),
                    api_function(
                        "cmd_fill_buffer",
                        "void?",
                        ("commands", "CommandList*"),
                        ("dst", "GpuSpan"),
                        ("value", "uint"),
                    ),
                    api_function(
                        "cmd_buffer_barrier",
                        "void?",
                        ("commands", "CommandList*"),
                        ("barrier", "BufferBarrier*"),
                    ),
                    api_function(
                        "cmd_copy_buffer_to_texture",
                        "void?",
                        ("commands", "CommandList*"),
                        ("desc", "BufferTextureCopyDesc*"),
                    ),
                    api_function(
                        "cmd_copy_texture_to_buffer",
                        "void?",
                        ("commands", "CommandList*"),
                        ("desc", "TextureBufferCopyDesc*"),
                    ),
                    api_function(
                        "cmd_bind_pipeline",
                        "void?",
                        ("commands", "CommandList*"),
                        ("pipeline", "PipelineHandle"),
                    ),
                    api_function(
                        "cmd_set_depth_state",
                        "void?",
                        ("commands", "CommandList*"),
                        ("depth", "DepthState*"),
                    ),
                    api_function(
                        "cmd_dispatch",
                        "void?",
                        ("commands", "CommandList*"),
                        ("root", "GpuAddress"),
                        ("groups", "Vec3u"),
                    ),
                    api_function(
                        "cmd_draw",
                        "void?",
                        ("commands", "CommandList*"),
                        ("vertex_root", "GpuAddress"),
                        ("fragment_root", "GpuAddress"),
                        ("vertex_count", "uint"),
                        ("instance_count", "uint"),
                    ),
                    api_function(
                        "cmd_draw_indexed",
                        "void?",
                        ("commands", "CommandList*"),
                        ("vertex_root", "GpuAddress"),
                        ("fragment_root", "GpuAddress"),
                        ("index_span", "GpuSpan"),
                        ("index_count", "uint"),
                        ("instance_count", "uint"),
                        ("index_type", "IndexType"),
                    ),
                    api_function(
                        "cmd_draw_indirect",
                        "void?",
                        ("commands", "CommandList*"),
                        ("vertex_root", "GpuAddress"),
                        ("fragment_root", "GpuAddress"),
                        ("args", "GpuSpan"),
                        ("draw_count", "uint"),
                    ),
                    api_function(
                        "cmd_draw_indexed_indirect",
                        "void?",
                        ("commands", "CommandList*"),
                        ("vertex_root", "GpuAddress"),
                        ("fragment_root", "GpuAddress"),
                        ("args", "GpuSpan"),
                        ("draw_count", "uint"),
                        ("index_span", "GpuSpan"),
                        ("index_type", "IndexType"),
                    ),
                    api_function(
                        "cmd_draw_indexed_indirect_count",
                        "void?",
                        ("commands", "CommandList*"),
                        ("vertex_root", "GpuAddress"),
                        ("fragment_root", "GpuAddress"),
                        ("args", "GpuSpan"),
                        ("count_span", "GpuSpan"),
                        ("max_draw_count", "uint"),
                        ("index_span", "GpuSpan"),
                        ("index_type", "IndexType"),
                    ),
                    api_function(
                        "cmd_dispatch_indirect",
                        "void?",
                        ("commands", "CommandList*"),
                        ("root", "GpuAddress"),
                        ("args", "GpuSpan"),
                    ),
                ],
                "types": [
                    {
                        "name": "GraphicsPipelineDesc",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "vertex_shader",
                                "type": {"name": "ShaderCode"},
                            },
                            {
                                "name": "fragment_shader",
                                "type": {"name": "ShaderCode"},
                            },
                            {
                                "name": "topology",
                                "type": {"name": "PrimitiveTopology"},
                            },
                            {
                                "name": "raster",
                                "type": {"name": "RasterState"},
                            },
                            {
                                "name": "blend",
                                "type": {"name": "BlendState"},
                            },
                            {
                                "name": "color_formats",
                                "type": {"name": "Format[]"},
                            },
                            {
                                "name": "depth_format",
                                "type": {"name": "Format"},
                            },
                            {
                                "name": "debug_name",
                                "type": {"name": "ZString"},
                            },
                        ],
                    },
                    {
                        "name": "GpuAllocation",
                        "kind": "struct",
                        "members": [
                            {"name": "owner", "type": {"name": "ulong"}},
                            {"name": "index", "type": {"name": "uint"}},
                            {"name": "generation", "type": {"name": "uint"}},
                        ],
                    },
                    {
                        "name": "Sampler",
                        "kind": "struct",
                        "members": [
                            {"name": "owner", "type": {"name": "ulong"}},
                            {"name": "index", "type": {"name": "uint"}},
                            {"name": "generation", "type": {"name": "uint"}},
                        ],
                    },
                    {
                        "name": "TextureIndex",
                        "kind": "bitstruct",
                        "base_type": {"name": "uint"},
                        "members": [{
                            "name": "value",
                            "type": {"name": "uint"},
                            "bit_range": [0, 31],
                        }],
                    },
                    {
                        "name": "SamplerIndex",
                        "kind": "bitstruct",
                        "base_type": {"name": "uint"},
                        "members": [{
                            "name": "value",
                            "type": {"name": "uint"},
                            "bit_range": [0, 31],
                        }],
                    },
                    {
                        "name": "TextureView",
                        "kind": "struct",
                        "members": [
                            {"name": "owner", "type": {"name": "ulong"}},
                            {
                                "name": "index",
                                "type": {"name": "TextureIndex"},
                            },
                            {"name": "generation", "type": {"name": "uint"}},
                        ],
                    },
                    {
                        "name": "TextureViewCreateDesc",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "texture",
                                "type": {"name": "TextureHandle"},
                            },
                            {
                                "name": "view",
                                "type": {"name": "TextureViewDesc"},
                            },
                        ],
                    },
                    {
                        "name": "DeviceDesc",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "texture_heap_capacity",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "sampler_heap_capacity",
                                "type": {"name": "uint"},
                            },
                        ],
                    },
                    {
                        "name": "DeviceCaps",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "texture_heap_capacity",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "sampler_heap_capacity",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "generated_work",
                                "type": {"name": "bool"},
                            },
                            {
                                "name": "max_generated_work_count",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "max_sampler_lod_bias",
                                "type": {"name": "float"},
                            },
                        ],
                    },
                    {
                        "name": "MemoryClass",
                        "kind": "enum",
                        "members": [
                            {
                                "name": "CPU_WRITE",
                                "type": {"name": "MemoryClass"},
                            },
                            {
                                "name": "GPU_PRIVATE",
                                "type": {"name": "MemoryClass"},
                            },
                            {
                                "name": "CPU_READ",
                                "type": {"name": "MemoryClass"},
                            },
                            {
                                "name": "TEXTURE",
                                "type": {"name": "MemoryClass"},
                            },
                        ],
                    },
                    {
                        "name": "DebugResourceKind",
                        "kind": "enum",
                        "members": [
                            {"name": name, "type": {"name": "DebugResourceKind"}}
                            for name in check_public_api.DEBUG_RESOURCE_KINDS
                        ],
                    },
                    {
                        "name": "MemoryStats",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "heaps",
                                "type": {"name": "MemoryHeapBudget[16]"},
                            },
                            {"name": "heap_count", "type": {"name": "uint"}},
                            {
                                "name": "texture_count",
                                "type": {"name": "ulong"},
                            },
                            {
                                "name": "live_allocation_count",
                                "type": {"name": "ulong"},
                            },
                        ],
                    },
                    {
                        "name": "AllocationDesc",
                        "kind": "struct",
                        "members": [
                            {"name": "size", "type": {"name": "usz"}},
                            {"name": "alignment", "type": {"name": "usz"}},
                            {
                                "name": "memory_class",
                                "type": {"name": "MemoryClass"},
                            },
                            {"name": "access", "type": {"name": "QueueRoles"}},
                            {
                                "name": "texture_requirements",
                                "type": {"name": "TextureRequirements[]"},
                            },
                            {"name": "debug_name", "type": {"name": "ZString"}},
                        ],
                    },
                    {
                        "name": "TextureCompatibility",
                        "kind": "distinct type",
                        "base_type": {"name": "uint128"},
                    },
                    {
                        "name": "TextureRequirements",
                        "kind": "struct",
                        "members": [
                            {"name": "size", "type": {"name": "usz"}},
                            {"name": "alignment", "type": {"name": "usz"}},
                            {
                                "name": "compatibility",
                                "type": {"name": "TextureCompatibility"},
                            },
                            {
                                "name": "dedicated_only",
                                "type": {"name": "bool"},
                            },
                        ],
                    },
                    {
                        "name": "DedicatedTexture",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "texture",
                                "type": {"name": "TextureHandle"},
                            },
                            {
                                "name": "allocation",
                                "type": {"name": "GpuAllocation"},
                            },
                        ],
                    },
                    {
                        "name": "AllocationInfo",
                        "kind": "struct",
                        "members": [
                            {"name": "size", "type": {"name": "usz"}},
                            {"name": "alignment", "type": {"name": "usz"}},
                            {
                                "name": "memory_class",
                                "type": {"name": "MemoryClass"},
                            },
                            {"name": "access", "type": {"name": "QueueRoles"}},
                            {"name": "mapped", "type": {"name": "bool"}},
                            {"name": "coherent", "type": {"name": "bool"}},
                            {"name": "addressable", "type": {"name": "bool"}},
                        ],
                    },
                    {
                        "name": "GpuSpan",
                        "kind": "struct",
                        "members": [
                            {"name": "owner", "type": {"name": "ulong"}},
                            {"name": "index", "type": {"name": "uint"}},
                            {"name": "generation", "type": {"name": "uint"}},
                            {"name": "offset", "type": {"name": "usz"}},
                            {"name": "size", "type": {"name": "usz"}},
                        ],
                    },
                    {
                        "name": "BufferCopyDesc",
                        "kind": "struct",
                        "members": [
                            {"name": "src", "type": {"name": "GpuSpan"}},
                            {"name": "dst", "type": {"name": "GpuSpan"}},
                        ],
                    },
                    {
                        "name": "BufferTextureCopyDesc",
                        "kind": "struct",
                        "members": [
                            {"name": "src", "type": {"name": "GpuSpan"}},
                            {
                                "name": "row_length_texels",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "texture",
                                "type": {"name": "TextureHandle"},
                            },
                            {"name": "mip", "type": {"name": "uint"}},
                            {
                                "name": "base_layer",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "layer_count",
                                "type": {"name": "uint"},
                            },
                            {"name": "x", "type": {"name": "uint"}},
                            {"name": "y", "type": {"name": "uint"}},
                            {"name": "width", "type": {"name": "uint"}},
                            {"name": "height", "type": {"name": "uint"}},
                        ],
                    },
                    {
                        "name": "TextureBufferCopyDesc",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "texture",
                                "type": {"name": "TextureHandle"},
                            },
                            {"name": "dst", "type": {"name": "GpuSpan"}},
                            {
                                "name": "row_length_texels",
                                "type": {"name": "uint"},
                            },
                            {"name": "mip", "type": {"name": "uint"}},
                            {
                                "name": "base_layer",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "layer_count",
                                "type": {"name": "uint"},
                            },
                            {"name": "x", "type": {"name": "uint"}},
                            {"name": "y", "type": {"name": "uint"}},
                            {"name": "width", "type": {"name": "uint"}},
                            {"name": "height", "type": {"name": "uint"}},
                        ],
                    },
                    {
                        "name": "BufferBarrier",
                        "kind": "struct",
                        "members": [
                            {"name": "span", "type": {"name": "GpuSpan"}},
                            {
                                "name": "before_stage",
                                "type": {"name": "Stage"},
                            },
                            {
                                "name": "after_stage",
                                "type": {"name": "Stage"},
                            },
                            {
                                "name": "before_hazard",
                                "type": {"name": "Hazard"},
                            },
                            {
                                "name": "after_hazard",
                                "type": {"name": "Hazard"},
                            },
                        ],
                    },
                    {
                        "name": "GeneratedDrawRecord",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "vertex_root_gpu",
                                "type": {"name": "GpuAddress"},
                            },
                            {
                                "name": "fragment_root_gpu",
                                "type": {"name": "GpuAddress"},
                            },
                            {
                                "name": "arguments",
                                "type": {"name": "DrawIndirectCommand"},
                            },
                        ],
                    },
                    {
                        "name": "GeneratedDrawIndexedRecord",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "vertex_root_gpu",
                                "type": {"name": "GpuAddress"},
                            },
                            {
                                "name": "fragment_root_gpu",
                                "type": {"name": "GpuAddress"},
                            },
                            {
                                "name": "arguments",
                                "type": {
                                    "name": "DrawIndexedIndirectCommand",
                                },
                            },
                            {"name": "_pad0", "type": {"name": "uint"}},
                        ],
                    },
                    {
                        "name": "GeneratedDispatchRecord",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "root_gpu",
                                "type": {"name": "GpuAddress"},
                            },
                            {
                                "name": "arguments",
                                "type": {"name": "DispatchIndirectCommand"},
                            },
                            {"name": "_pad0", "type": {"name": "uint"}},
                        ],
                    },
                    {"name": "ExecutableCommandList", "kind": "struct"},
                    {
                        "name": "SubmitDesc",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "command_lists",
                                "type": {"name": "ExecutableCommandList[]"},
                            },
                            {
                                "name": "readiness",
                                "type": {"name": "SwapchainReadiness"},
                            },
                        ],
                    },
                    {"name": "SwapchainReadiness", "kind": "struct"},
                    {
                        "name": "AcquiredImage",
                        "kind": "struct",
                        "members": [{
                            "name": "readiness",
                            "type": {"name": "SwapchainReadiness"},
                        }],
                    },
                ],
            },
            "gpu::surface::win32": surface_module(
                "InstanceHandle",
                "WindowHandle",
            ),
            "gpu::surface::wayland": surface_module(
                "DisplayHandle",
                "SurfaceHandle",
            ),
            "gpu::surface::x11": surface_module(
                "DisplayHandle",
                "WindowHandle",
            ),
        },
    }


class PublicApiCheckTests(unittest.TestCase):
    def test_accepts_distinct_platform_handle_modules(self) -> None:
        self.assertEqual(check_public_api.validate_document(valid_document()), [])

    def test_rejects_backend_sharing_flags(self) -> None:
        document = valid_document()
        document["modules"]["gpu"]["types"].append(
            {"name": "BufferUsage", "members": [{"name": "shared_queues"}]}
        )
        self.assertIn(
            "backend queue-sharing policy",
            check_public_api.validate_document(document),
        )

    def test_rejects_retired_root_surface_types(self) -> None:
        document = valid_document()
        document["modules"]["gpu"]["types"].append({"name": "PlatformKind"})
        self.assertIn(
            "retired PlatformKind",
            check_public_api.validate_document(document),
        )

    def test_rejects_untyped_root_surface_constructors(self) -> None:
        document = valid_document()
        document["modules"]["gpu"]["functions"].append(
            {"name": "create_win32_surface"}
        )
        self.assertIn(
            "create_win32_surface",
            check_public_api.validate_document(document),
        )

    def test_rejects_transparent_platform_handles(self) -> None:
        document = valid_document()
        win32_types = document["modules"]["gpu::surface::win32"]["types"]
        win32_types[0]["kind"] = "inline type"
        self.assertIn(
            "gpu::surface::win32::InstanceHandle must be a distinct type",
            check_public_api.validate_document(document),
        )

    def test_rejects_retired_public_synchronization(self) -> None:
        document = valid_document()
        document["modules"]["gpu"]["types"].append(
            {"name": "SemaphoreHandle"}
        )
        document["modules"]["gpu"]["functions"].append(
            {"name": "wait_queue_idle"}
        )
        document["modules"]["gpu"]["types"].append({
            "name": "DeviceCaps",
            "members": [{"name": "timeline_semaphore"}],
        })
        failures = check_public_api.validate_document(document)
        self.assertIn("retired public semaphore", failures)
        self.assertIn("retired wait_queue_idle", failures)
        self.assertIn("retired timeline capability", failures)

    def test_rejects_each_retired_frame_policy_symbol(self) -> None:
        cases = (
            (
                "FrameToken",
                "types",
                {"name": "FrameToken"},
                "retired FrameToken",
            ),
            (
                "MemoryKind",
                "types",
                {"name": "MemoryKind"},
                "retired MemoryKind",
            ),
            (
                "begin_frame",
                "functions",
                {"name": "begin_frame"},
                "retired begin_frame",
            ),
            (
                "alloc_frame_span",
                "functions",
                {"name": "alloc_frame_span"},
                "retired alloc_frame_span",
            ),
            (
                "end_frame",
                "functions",
                {"name": "end_frame"},
                "retired end_frame",
            ),
            (
                "with_frame",
                "functions",
                {"name": "with_frame"},
                "retired with_frame",
            ),
            (
                "DeviceDesc.frame_arena_size",
                "types",
                {
                    "name": "FramePolicyProbe",
                    "members": [{"name": "frame_arena_size"}],
                },
                "retired DeviceDesc.frame_arena_size",
            ),
            (
                "DeviceDesc.frames_in_flight",
                "types",
                {
                    "name": "FramePolicyProbe",
                    "members": [{"name": "frames_in_flight"}],
                },
                "retired DeviceDesc.frames_in_flight",
            ),
            (
                "DEFAULT_FRAME_ARENA_SIZE",
                "types",
                {"name": "DEFAULT_FRAME_ARENA_SIZE"},
                "retired DEFAULT_FRAME_ARENA_SIZE",
            ),
            (
                "ARENA_FULL",
                "types",
                {"name": "ARENA_FULL"},
                "retired ARENA_FULL",
            ),
        )
        for symbol, section, entry, expected_failure in cases:
            with self.subTest(symbol=symbol):
                document = valid_document()
                document["modules"]["gpu"][section].append(entry)
                self.assertEqual(
                    check_public_api.validate_document(document),
                    [expected_failure],
                    f"{symbol} must produce exactly its retirement failure",
                )

    def test_rejects_debug_resource_kind_frame(self) -> None:
        document = valid_document()
        resource_kind = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "DebugResourceKind"
        )
        resource_kind["members"] = [
            {"name": name, "type": {"name": "DebugResourceKind"}}
            for name in (
                "NONE",
                "DEVICE",
                "GPU_SPAN",
                "TEXTURE",
                "PIPELINE",
                "SWAPCHAIN",
                "SHADER",
                "COMMAND_LIST",
                "TEXTURE_DESCRIPTOR",
                "SAMPLER",
                "ALLOCATION",
                "FRAME",
            )
        ]
        self.assertEqual(
            check_public_api.validate_document(document),
            ["DebugResourceKind must match its exact schema"],
            "DebugResourceKind.FRAME must produce exactly its schema failure",
        )

    def test_rejects_public_recording_contexts(self) -> None:
        document = valid_document()
        document["modules"]["gpu"]["types"].append(
            {"name": "RecordingContextHandle"}
        )
        document["modules"]["gpu"]["types"].append({
            "name": "DebugResourceKind",
            "members": [{"name": "RECORDING_CONTEXT"}],
        })
        failures = check_public_api.validate_document(document)
        self.assertIn("retired recording context", failures)

    def test_rejects_swapchain_handle_coupling(self) -> None:
        document = valid_document()
        submit_desc = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "SubmitDesc"
        )
        submit_desc["members"] = [{
            "name": "swapchain",
            "type": {"name": "SwapchainHandle"},
        }]
        failures = check_public_api.validate_document(document)
        self.assertIn(
            "SubmitDesc must not expose swapchain coupling",
            failures,
        )
        self.assertIn(
            "SubmitDesc.readiness must contain one-shot swapchain readiness",
            failures,
        )

    def test_rejects_wrong_debug_resource_kind_member(self) -> None:
        document = valid_document()
        resource_kind = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "DebugResourceKind"
        )
        resource_kind["members"][2]["name"] = "BUFFER"
        self.assertIn(
            "DebugResourceKind must match its exact schema",
            check_public_api.validate_document(document),
        )

    def test_rejects_wrong_debug_resource_kind_order(self) -> None:
        document = valid_document()
        resource_kind = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "DebugResourceKind"
        )
        resource_kind["members"][1], resource_kind["members"][2] = (
            resource_kind["members"][2],
            resource_kind["members"][1],
        )
        self.assertIn(
            "DebugResourceKind must match its exact schema",
            check_public_api.validate_document(document),
        )

    def test_rejects_memory_stats_buffer_count(self) -> None:
        document = valid_document()
        memory_stats = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "MemoryStats"
        )
        memory_stats["members"].insert(2, {
            "name": "buffer_count",
            "type": {"name": "ulong"},
        })
        self.assertIn(
            "MemoryStats must match its exact public schema",
            check_public_api.validate_document(document),
        )

    def test_rejects_missing_allocation_contract(self) -> None:
        document = valid_document()
        functions = document["modules"]["gpu"]["functions"]
        functions[:] = [
            function for function in functions
            if function["name"] != "allocate_memory"
        ]
        self.assertIn(
            "missing allocate_memory",
            check_public_api.validate_document(document),
        )

    def test_rejects_missing_mapped_visibility_contract(self) -> None:
        for name in ("flush_mapped_span", "invalidate_mapped_span"):
            with self.subTest(name=name):
                document = valid_document()
                functions = document["modules"]["gpu"]["functions"]
                functions[:] = [
                    function for function in functions
                    if function["name"] != name
                ]
                self.assertIn(
                    f"missing {name}",
                    check_public_api.validate_document(document),
                )

    def test_rejects_span_operation_parameter_renaming(self) -> None:
        for name in (
            "cmd_copy_buffer",
            "cmd_fill_buffer",
            "cmd_buffer_barrier",
            "cmd_copy_buffer_to_texture",
            "cmd_copy_texture_to_buffer",
            "cmd_draw_indexed",
            "cmd_draw_indirect",
            "cmd_draw_indexed_indirect",
            "cmd_draw_indexed_indirect_count",
            "cmd_dispatch_indirect",
        ):
            with self.subTest(name=name):
                document = valid_document()
                function = next(
                    entry for entry in document["modules"]["gpu"]["functions"]
                    if entry["name"] == name
                )
                function["members"][0]["name"] = "renamed"
                self.assertIn(
                    f"{name} has the wrong parameters",
                    check_public_api.validate_document(document),
                )

    def test_rejects_non_span_index_indirect_and_texture_copy(self) -> None:
        for name in (
            "cmd_draw_indexed",
            "cmd_draw_indirect",
            "cmd_draw_indexed_indirect",
            "cmd_draw_indexed_indirect_count",
            "cmd_dispatch_indirect",
        ):
            with self.subTest(name=name):
                document = valid_document()
                function = next(
                    entry for entry in document["modules"]["gpu"]["functions"]
                    if entry["name"] == name
                )
                span = next(
                    member for member in function["members"]
                    if member["type"]["name"] == "GpuSpan"
                )
                span["type"]["name"] = "BufferHandle"
                self.assertIn(
                    f"{name} has the wrong parameters",
                    check_public_api.validate_document(document),
                )

    def test_rejects_non_span_buffer_texture_copy_schemas(self) -> None:
        for type_name, field_name, failure in (
            (
                "BufferTextureCopyDesc",
                "src",
                "BufferTextureCopyDesc must contain one source span",
            ),
            (
                "TextureBufferCopyDesc",
                "dst",
                "TextureBufferCopyDesc must contain one destination span",
            ),
        ):
            with self.subTest(type_name=type_name):
                document = valid_document()
                desc = next(
                    entry for entry in document["modules"]["gpu"]["types"]
                    if entry["name"] == type_name
                )
                field = next(
                    member for member in desc["members"]
                    if member["name"] == field_name
                )
                field["type"]["name"] = "BufferHandle"
                self.assertIn(
                    failure,
                    check_public_api.validate_document(document),
                )

    def test_rejects_dynamic_depth_in_graphics_pipeline_desc(self) -> None:
        document = valid_document()
        desc = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "GraphicsPipelineDesc"
        )
        desc["members"].insert(3, {
            "name": "depth",
            "type": {"name": "DepthState"},
        })
        self.assertIn(
            "GraphicsPipelineDesc must not contain dynamic depth state",
            check_public_api.validate_document(document),
        )

    def test_rejects_pipeline_bearing_execution_signatures(self) -> None:
        for name in (
            "cmd_dispatch",
            "cmd_dispatch_indirect",
            "cmd_draw",
            "cmd_draw_indexed",
            "cmd_draw_indirect",
            "cmd_draw_indexed_indirect",
            "cmd_draw_indexed_indirect_count",
        ):
            with self.subTest(name=name):
                document = valid_document()
                function = next(
                    entry for entry in document["modules"]["gpu"]["functions"]
                    if entry["name"] == name
                )
                function["members"].insert(1, {
                    "name": "pipeline",
                    "type": {"name": "PipelineHandle"},
                })
                self.assertIn(
                    f"{name} has the wrong parameters",
                    check_public_api.validate_document(document),
                )

    def test_rejects_legacy_fill_signature(self) -> None:
        legacy_parameters = {
            "cmd_fill_buffer": (
                "CommandList*",
                "BufferHandle",
                "usz",
                "usz",
                "uint",
            ),
        }
        for name, parameter_types in legacy_parameters.items():
            with self.subTest(name=name):
                document = valid_document()
                function = next(
                    entry for entry in document["modules"]["gpu"]["functions"]
                    if entry["name"] == name
                )
                function["members"] = [
                    {
                        "name": f"parameter_{index}",
                        "type": {"name": type_name},
                    }
                    for index, type_name in enumerate(parameter_types)
                ]
                self.assertIn(
                    f"{name} has the wrong parameters",
                    check_public_api.validate_document(document),
                )

    def test_rejects_legacy_buffer_copy_schema(self) -> None:
        document = valid_document()
        copy_desc = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "BufferCopyDesc"
        )
        copy_desc["members"] = [
            {"name": "src", "type": {"name": "BufferHandle"}},
            {"name": "dst", "type": {"name": "BufferHandle"}},
            {"name": "src_offset", "type": {"name": "usz"}},
            {"name": "dst_offset", "type": {"name": "usz"}},
            {"name": "size", "type": {"name": "usz"}},
        ]
        self.assertIn(
            "BufferCopyDesc must contain exactly source and destination spans",
            check_public_api.validate_document(document),
        )

    def test_rejects_legacy_buffer_barrier_schema(self) -> None:
        document = valid_document()
        barrier = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "BufferBarrier"
        )
        barrier["members"] = [
            {"name": "buffer", "type": {"name": "BufferHandle"}},
            {"name": "offset", "type": {"name": "usz"}},
            {"name": "size", "type": {"name": "usz"}},
            *barrier["members"][1:],
        ]
        self.assertIn(
            "BufferBarrier must contain exactly one span and semantic hazards",
            check_public_api.validate_document(document),
        )

    def test_rejects_public_buffer_types(self) -> None:
        for name, failure in (
            ("BufferHandle", "retired BufferHandle"),
            ("BufferDesc", "retired BufferDesc"),
            ("BufferUsage", "retired BufferUsage"),
        ):
            with self.subTest(name=name):
                document = valid_document()
                document["modules"]["gpu"]["types"].append({
                    "name": name,
                    "kind": "struct",
                })
                self.assertIn(
                    failure,
                    check_public_api.validate_document(document),
                )

    def test_rejects_retired_texture_descriptor_surface(self) -> None:
        for name, failure in (
            ("TextureDescriptorDesc", "retired TextureDescriptorDesc"),
            (
                "create_texture_descriptor",
                "retired texture descriptor lifecycle",
            ),
            (
                "destroy_texture_descriptor",
                "retired texture descriptor lifecycle",
            ),
            (
                "create_texture_descriptors",
                "retired texture descriptor lifecycle",
            ),
        ):
            with self.subTest(name=name):
                document = valid_document()
                section = (
                    "types" if name == "TextureDescriptorDesc"
                    else "functions"
                )
                document["modules"]["gpu"][section].append({
                    "name": name,
                    "kind": "struct",
                })
                self.assertIn(
                    failure,
                    check_public_api.validate_document(document),
                )

    def test_rejects_public_buffer_lifecycle(self) -> None:
        for name in (
            "create_buffer",
            "destroy_buffer",
            "get_buffer_address",
            "get_buffer_span",
            "flush_buffer",
            "invalidate_buffer",
        ):
            with self.subTest(name=name):
                document = valid_document()
                document["modules"]["gpu"]["functions"].append({
                    "name": name,
                })
                self.assertIn(
                    "retired public buffer lifecycle",
                    check_public_api.validate_document(document),
                )

    def test_requires_private_vulkan_backend_modules(self) -> None:
        relative = Path("gpu/vk/buffer.c3")
        self.assertEqual(
            check_public_api.validate_private_backend_source(
                relative,
                "module gpu::vk @private;",
            ),
            [],
        )
        for declaration in (
            "module gpu::vk;",
            "module gpu::vk @public;",
            "module gpu;",
        ):
            with self.subTest(declaration=declaration):
                self.assertEqual(
                    check_public_api.validate_private_backend_source(
                        relative,
                        declaration,
                    ),
                    [
                        "gpu/vk/buffer.c3 must declare the private "
                        "gpu::vk backend module"
                    ],
                )

    def test_rejects_span_backend_details(self) -> None:
        document = valid_document()
        document["modules"]["gpu"]["types"].append({
            "name": "GpuSpan",
            "kind": "struct",
            "members": [
                {"name": "allocation", "type": {"name": "GpuAllocation"}},
                {"name": "buffer", "type": {"name": "BufferHandle"}},
                {"name": "cpu", "type": {"name": "void*"}},
                {"name": "gpu", "type": {"name": "GpuAddress"}},
            ],
        })
        self.assertIn(
            "GpuSpan must match the exact identity/range schema",
            check_public_api.validate_document(document),
        )

    def test_rejects_wrong_span_field_type(self) -> None:
        document = valid_document()
        span = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "GpuSpan"
        )
        span["members"][0]["type"]["name"] = "uint"
        self.assertIn(
            "GpuSpan must match the exact identity/range schema",
            check_public_api.validate_document(document),
        )

    def test_rejects_wrong_allocation_identity_schema(self) -> None:
        document = valid_document()
        allocation = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "GpuAllocation"
        )
        allocation["members"][1]["name"] = "slot"
        self.assertIn(
            "GpuAllocation must match the exact identity schema",
            check_public_api.validate_document(document),
        )

    def test_rejects_wrong_allocation_desc_schema(self) -> None:
        document = valid_document()
        desc = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "AllocationDesc"
        )
        desc["members"][3]["type"]["name"] = "QueueKind"
        self.assertIn(
            "AllocationDesc must match the exact public schema",
            check_public_api.validate_document(document),
        )

    def test_rejects_wrong_texture_compatibility_storage(self) -> None:
        document = valid_document()
        compatibility = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "TextureCompatibility"
        )
        compatibility["base_type"]["name"] = "ulong"
        self.assertIn(
            "TextureCompatibility must be an opaque uint128 distinct type",
            check_public_api.validate_document(document),
        )

    def test_rejects_wrong_texture_requirements_schema(self) -> None:
        document = valid_document()
        requirements = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "TextureRequirements"
        )
        requirements["members"].pop()
        self.assertIn(
            "TextureRequirements must match the exact public schema",
            check_public_api.validate_document(document),
        )

    def test_rejects_wrong_dedicated_texture_schema(self) -> None:
        document = valid_document()
        result = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "DedicatedTexture"
        )
        result["members"].pop()
        self.assertIn(
            "DedicatedTexture must expose exactly two ownership tokens",
            check_public_api.validate_document(document),
        )

    def test_rejects_wrong_placed_texture_signature(self) -> None:
        document = valid_document()
        function = next(
            entry for entry in document["modules"]["gpu"]["functions"]
            if entry["name"] == "create_placed_texture"
        )
        function["members"].pop()
        self.assertIn(
            "create_placed_texture has the wrong parameters",
            check_public_api.validate_document(document),
        )

    def test_rejects_wrong_dedicated_texture_signature(self) -> None:
        document = valid_document()
        function = next(
            entry for entry in document["modules"]["gpu"]["functions"]
            if entry["name"] == "create_dedicated_texture"
        )
        function["members"].pop()
        self.assertIn(
            "create_dedicated_texture has the wrong parameters",
            check_public_api.validate_document(document),
        )

    def test_rejects_wrong_allocation_info_schema(self) -> None:
        document = valid_document()
        info = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "AllocationInfo"
        )
        info["members"].pop()
        self.assertIn(
            "AllocationInfo must match the exact public schema",
            check_public_api.validate_document(document),
        )

    def test_rejects_wrong_memory_class_values(self) -> None:
        document = valid_document()
        memory_class = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "MemoryClass"
        )
        memory_class["members"].reverse()
        self.assertIn(
            "MemoryClass must expose exactly the four semantic values",
            check_public_api.validate_document(document),
        )

    def test_rejects_old_command_lifecycle_shape(self) -> None:
        document = valid_document()
        functions = document["modules"]["gpu"]["functions"]
        begin_commands = next(
            entry for entry in functions
            if entry["name"] == "begin_commands"
        )
        begin_commands["members"] = [
            {"name": "device", "type": {"name": "Device*"}},
            {"name": "queue", "type": {"name": "QueueKind"}},
        ]
        end_commands = next(
            entry for entry in functions
            if entry["name"] == "end_commands"
        )
        end_commands["return_type"] = {"name": "void?"}
        failures = check_public_api.validate_document(document)
        self.assertIn("begin_commands must take one Queue token", failures)
        self.assertIn(
            "end_commands must return ExecutableCommandList?",
            failures,
        )

    def test_rejects_wrong_sampler_identity_schema(self) -> None:
        document = valid_document()
        sampler = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "Sampler"
        )
        sampler["members"][2]["name"] = "revision"
        self.assertIn(
            "Sampler must match the exact identity schema",
            check_public_api.validate_document(document),
        )

    def test_requires_sampler_lod_bias_capability(self) -> None:
        document = valid_document()
        caps = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "DeviceCaps"
        )
        caps["members"].clear()
        self.assertIn(
            "DeviceCaps.max_sampler_lod_bias must be a float",
            check_public_api.validate_document(document),
        )

    def test_requires_semantic_heap_capacities(self) -> None:
        document = valid_document()
        types = document["modules"]["gpu"]["types"]
        desc = next(entry for entry in types if entry["name"] == "DeviceDesc")
        caps = next(entry for entry in types if entry["name"] == "DeviceCaps")
        desc["members"][0]["type"]["name"] = "usz"
        caps["members"] = [
            member for member in caps["members"]
            if member["name"] != "sampler_heap_capacity"
        ]
        failures = check_public_api.validate_document(document)
        self.assertIn("DeviceDesc.texture_heap_capacity must be a uint", failures)
        self.assertIn("DeviceCaps.sampler_heap_capacity must be a uint", failures)

    def test_requires_generated_work_capability_and_limit(self) -> None:
        document = valid_document()
        caps = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "DeviceCaps"
        )
        generated = next(
            member for member in caps["members"]
            if member["name"] == "generated_work"
        )
        generated["type"]["name"] = "uint"
        caps["members"] = [
            member for member in caps["members"]
            if member["name"] != "max_generated_work_count"
        ]
        failures = check_public_api.validate_document(document)
        self.assertIn("DeviceCaps.generated_work must be a bool", failures)
        self.assertIn(
            "DeviceCaps.max_generated_work_count must be a uint",
            failures,
        )

    def test_rejects_generated_record_abi_drift(self) -> None:
        document = valid_document()
        records = {
            entry["name"]: entry
            for entry in document["modules"]["gpu"]["types"]
        }
        records["GeneratedDrawRecord"]["members"][0]["name"] = "vertex_root"
        records["GeneratedDrawIndexedRecord"]["members"].pop()
        records["GeneratedDispatchRecord"]["members"][0]["type"]["name"] = "ulong"
        failures = check_public_api.validate_document(document)
        self.assertIn(
            "GeneratedDrawRecord must match the generated ABI schema",
            failures,
        )
        self.assertIn(
            "GeneratedDrawIndexedRecord must match the generated ABI schema",
            failures,
        )
        self.assertIn(
            "GeneratedDispatchRecord must match the generated ABI schema",
            failures,
        )

    def test_rejects_backend_shaped_heap_configuration(self) -> None:
        document = valid_document()
        types = document["modules"]["gpu"]["types"]
        desc = next(entry for entry in types if entry["name"] == "DeviceDesc")
        caps = next(entry for entry in types if entry["name"] == "DeviceCaps")
        types.append({"name": "DescriptorHeapMode", "kind": "enum"})
        desc["members"].extend([
            {
                "name": "descriptor_heap_mode",
                "type": {"name": "DescriptorHeapMode"},
            },
            {
                "name": "texture_descriptor_capacity",
                "type": {"name": "uint"},
            },
            {
                "name": "sampler_descriptor_capacity",
                "type": {"name": "uint"},
            },
        ])
        caps["members"].extend([
            {"name": "descriptor_buffer", "type": {"name": "bool"}},
            {"name": "descriptor_indexing", "type": {"name": "bool"}},
            {
                "name": "max_texture_descriptors",
                "type": {"name": "uint"},
            },
            {
                "name": "max_sampler_descriptors",
                "type": {"name": "uint"},
            },
        ])
        failures = check_public_api.validate_document(document)
        self.assertIn("backend heap strategy type", failures)
        self.assertIn("backend heap strategy field", failures)
        self.assertIn("backend descriptor-buffer capability", failures)
        self.assertIn("backend descriptor-indexing capability", failures)
        self.assertIn("backend-shaped texture capacity", failures)
        self.assertIn("backend-shaped sampler capacity", failures)
        self.assertIn("backend-shaped texture limit", failures)
        self.assertIn("backend-shaped sampler limit", failures)

    def test_rejects_wrong_sampler_operation_contracts(self) -> None:
        document = valid_document()
        functions = document["modules"]["gpu"]["functions"]
        intern = next(entry for entry in functions
            if entry["name"] == "intern_sampler")
        publish = next(entry for entry in functions
            if entry["name"] == "publish_sampler")
        intern["return_type"]["name"] = "SamplerIndex?"
        publish["members"][1]["type"]["name"] = "SamplerIndex"
        failures = check_public_api.validate_document(document)
        self.assertIn("intern_sampler has the wrong return type", failures)
        self.assertIn("publish_sampler has the wrong parameters", failures)

    def test_rejects_wrong_texture_view_operation_contracts(self) -> None:
        document = valid_document()
        functions = document["modules"]["gpu"]["functions"]
        create = next(entry for entry in functions
            if entry["name"] == "create_texture_view")
        destroy = next(entry for entry in functions
            if entry["name"] == "destroy_texture_view")
        batch = next(entry for entry in functions
            if entry["name"] == "create_texture_views")
        create["return_type"]["name"] = "TextureIndex?"
        destroy["members"][1]["type"]["name"] = "TextureIndex"
        batch["members"][2]["type"]["name"] = "TextureIndex[]"
        failures = check_public_api.validate_document(document)
        self.assertIn("create_texture_view has the wrong return type", failures)
        self.assertIn("destroy_texture_view has the wrong parameters", failures)
        self.assertIn("create_texture_views has the wrong parameters", failures)

    def test_rejects_wrong_texture_view_identity_schema(self) -> None:
        document = valid_document()
        view = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "TextureView"
        )
        view["members"][1]["type"]["name"] = "uint"
        self.assertIn(
            "TextureView must match the exact identity schema",
            check_public_api.validate_document(document),
        )

    def test_rejects_generation_packed_shader_indices(self) -> None:
        document = valid_document()
        types = document["modules"]["gpu"]["types"]
        texture_index = next(
            entry for entry in types if entry["name"] == "TextureIndex"
        )
        sampler_index = next(
            entry for entry in types if entry["name"] == "SamplerIndex"
        )
        texture_index["members"][0]["bit_range"] = [0, 15]
        sampler_index["base_type"]["name"] = "ulong"
        failures = check_public_api.validate_document(document)
        self.assertIn(
            "TextureIndex must be a generation-free 32-bit shader index",
            failures,
        )
        self.assertIn(
            "SamplerIndex must be a generation-free 32-bit shader index",
            failures,
        )

if __name__ == "__main__":
    unittest.main()
