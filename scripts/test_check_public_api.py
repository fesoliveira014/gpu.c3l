from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    specs = {
        ("InstanceHandle", "WindowHandle"): (
            ("instance", "window"),
            ("void*", "void*"),
        ),
        ("DisplayHandle", "SurfaceHandle"): (
            ("display", "surface"),
            ("void*", "void*"),
        ),
        ("DisplayHandle", "WindowHandle"): (
            ("display", "window"),
            ("void*", "ulong"),
        ),
    }
    parameter_names, base_types = specs[handles]
    return {
        "functions": [{
            "name": "create_surface",
            "return_type": {"name": "Surface?"},
            "members": [
                {"name": "runtime", "type": {"name": "Runtime*"}},
                *[
                    {"name": parameter, "type": {"name": handle}}
                    for parameter, handle in zip(parameter_names, handles)
                ],
            ],
        }],
        "types": [
            {
                "name": name,
                "kind": "distinct type",
                "base_type": {"name": base_type},
            }
            for name, base_type in zip(handles, base_types)
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
                            {
                                "name": "allocator",
                                "type": {"name": "CommandAllocator*"},
                            },
                        ],
                    },
                    api_function(
                        "create_command_allocator",
                        "CommandAllocator?",
                        ("device", "Device*"),
                        ("queue", "Queue"),
                        ("desc", "CommandAllocatorDesc*"),
                    ),
                    api_function(
                        "destroy_command_allocator",
                        "void?",
                        ("allocator", "CommandAllocator*"),
                    ),
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
                        "name": "acquire_next_image",
                        "return_type": {"name": "AcquiredImage?"},
                        "members": [
                            {"name": "device", "type": {"name": "Device*"}},
                            {
                                "name": "swapchain",
                                "type": {"name": "SwapchainHandle"},
                            },
                            {
                                "name": "timeout_ns",
                                "type": {"name": "ulong"},
                                "default_value": "0",
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
                        "SamplerIndex?",
                        ("device", "Device*"),
                        ("desc", "SamplerDesc*"),
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
                        "create_attachment_view",
                        "AttachmentViewHandle?",
                        ("device", "Device*"),
                        ("desc", "AttachmentViewDesc*"),
                    ),
                    api_function(
                        "destroy_attachment_view",
                        "void?",
                        ("device", "Device*"),
                        ("view", "AttachmentViewHandle"),
                    ),
                    api_function(
                        "reserve_generated_scratch",
                        "void?",
                        ("allocator", "CommandAllocator*"),
                        ("desc", "GeneratedScratchDesc*"),
                    ),
                    api_function(
                        "release_generated_scratch",
                        "void?",
                        ("allocator", "CommandAllocator*"),
                        ("pipeline", "PipelineHandle"),
                        ("kind", "GeneratedWorkKind"),
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
                        "cmd_barrier",
                        "void?",
                        ("commands", "CommandList*"),
                        ("barrier", "Barrier*"),
                    ),
                    api_function(
                        "cmd_texture_barrier",
                        "void?",
                        ("commands", "CommandList*"),
                        ("barrier", "TextureBarrier*"),
                    ),
                    api_function(
                        "sampled_at",
                        "TextureState",
                        ("stages", "StageMask"),
                    ),
                    api_function(
                        "storage_at",
                        "TextureState",
                        ("stages", "StageMask"),
                        ("access", "TextureAccess"),
                    ),
                    api_function(
                        "texture_transition",
                        "TextureBarrier?",
                        ("texture", "TextureHandle"),
                        ("before", "TextureState"),
                        ("after", "TextureState"),
                    ),
                    api_function(
                        "texture_view_transition",
                        "TextureBarrier?",
                        ("texture", "TextureHandle"),
                        ("view", "TextureViewDesc"),
                        ("before", "TextureState"),
                        ("after", "TextureState"),
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
                        "cmd_begin_render_pass",
                        "void?",
                        ("commands", "CommandList*"),
                        ("desc", "RenderPassDesc*"),
                    ),
                    api_function(
                        "cmd_begin_render_pass_with_state",
                        "void?",
                        ("commands", "CommandList*"),
                        ("desc", "RenderPassDesc*"),
                        ("state", "GraphicsState*"),
                    ),
                    api_function(
                        "cmd_set_graphics_state",
                        "void?",
                        ("commands", "CommandList*"),
                        ("state", "GraphicsState*"),
                    ),
                    api_function(
                        "full_render_graphics_state",
                        "GraphicsState?",
                        ("width", "uint"),
                        ("height", "uint"),
                    ),
                    api_function(
                        "cmd_end_render_pass",
                        "void?",
                        ("commands", "CommandList*"),
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
                        "cmd_set_raster_state",
                        "void?",
                        ("commands", "CommandList*"),
                        ("raster", "DynamicRasterState*"),
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
                "variables": [
                    {
                        "name": "COLOR_WRITE_ALL",
                        "kind": "constant",
                        "type": {"name": "ColorWriteMask"},
                        "value": (
                            "{ .red = true, .green = true, .blue = true, "
                            ".alpha = true, }"
                        ),
                    },
                    {
                        "name": "TIMEOUT_INFINITE",
                        "kind": "constant",
                        "type": {"name": "ulong"},
                        "value": "18446744073709551615",
                    },
                ],
                "types": [
                    {
                        "name": "ColorWriteMask",
                        "kind": "bitstruct",
                        "base_type": {"name": "uint"},
                        "members": [
                            {
                                "name": "red",
                                "type": {"name": "bool"},
                                "bit_range": [0, 0],
                            },
                            {
                                "name": "green",
                                "type": {"name": "bool"},
                                "bit_range": [1, 1],
                            },
                            {
                                "name": "blue",
                                "type": {"name": "bool"},
                                "bit_range": [2, 2],
                            },
                            {
                                "name": "alpha",
                                "type": {"name": "bool"},
                                "bit_range": [3, 3],
                            },
                        ],
                    },
                    {
                        "name": "BlendState",
                        "kind": "struct",
                        "members": [
                            {"name": "enable", "type": {"name": "bool"}},
                            {
                                "name": "src_color",
                                "type": {"name": "BlendFactor"},
                            },
                            {
                                "name": "dst_color",
                                "type": {"name": "BlendFactor"},
                            },
                            {
                                "name": "color_op",
                                "type": {"name": "BlendOp"},
                            },
                            {
                                "name": "src_alpha",
                                "type": {"name": "BlendFactor"},
                            },
                            {
                                "name": "dst_alpha",
                                "type": {"name": "BlendFactor"},
                            },
                            {
                                "name": "alpha_op",
                                "type": {"name": "BlendOp"},
                            },
                        ],
                    },
                    {
                        "name": "ColorTargetState",
                        "kind": "struct",
                        "members": [
                            {"name": "format", "type": {"name": "Format"}},
                            {
                                "name": "blend",
                                "type": {"name": "BlendState"},
                            },
                            {
                                "name": "write_mask",
                                "type": {"name": "ColorWriteMask"},
                            },
                        ],
                    },
                    {
                        "name": "DepthState",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "test_enable",
                                "type": {"name": "bool"},
                            },
                            {
                                "name": "write_enable",
                                "type": {"name": "bool"},
                            },
                            {
                                "name": "compare",
                                "type": {"name": "CompareOp"},
                            },
                        ],
                    },
                    {
                        "name": "DynamicRasterState",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "topology",
                                "type": {"name": "PrimitiveTopology"},
                            },
                            {
                                "name": "cull_mode",
                                "type": {"name": "CullMode"},
                            },
                            {
                                "name": "front_face",
                                "type": {"name": "FrontFace"},
                            },
                            {
                                "name": "depth_bias_enable",
                                "type": {"name": "bool"},
                            },
                            {
                                "name": "depth_bias_constant",
                                "type": {"name": "float"},
                            },
                            {
                                "name": "depth_bias_slope",
                                "type": {"name": "float"},
                            },
                            {
                                "name": "depth_bias_clamp",
                                "type": {"name": "float"},
                            },
                        ],
                    },
                    {
                        "name": "Viewport",
                        "kind": "struct",
                        "members": [
                            {"name": "x", "type": {"name": "float"}},
                            {"name": "y", "type": {"name": "float"}},
                            {
                                "name": "width",
                                "type": {"name": "float"},
                            },
                            {
                                "name": "height",
                                "type": {"name": "float"},
                            },
                            {
                                "name": "min_depth",
                                "type": {"name": "float"},
                            },
                            {
                                "name": "max_depth",
                                "type": {"name": "float"},
                            },
                        ],
                    },
                    {
                        "name": "ScissorRect",
                        "kind": "struct",
                        "members": [
                            {"name": "x", "type": {"name": "int"}},
                            {"name": "y", "type": {"name": "int"}},
                            {
                                "name": "width",
                                "type": {"name": "int"},
                            },
                            {
                                "name": "height",
                                "type": {"name": "int"},
                            },
                        ],
                    },
                    {
                        "name": "GraphicsState",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "viewport",
                                "type": {"name": "Viewport"},
                            },
                            {
                                "name": "scissor",
                                "type": {"name": "ScissorRect"},
                            },
                            {
                                "name": "raster",
                                "type": {"name": "DynamicRasterState"},
                            },
                            {
                                "name": "depth",
                                "type": {"name": "DepthState"},
                            },
                        ],
                    },
                    {
                        "name": "ComputePipelineDesc",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "shader",
                                "type": {"name": "ShaderCode"},
                            },
                            {
                                "name": "debug_name",
                                "type": {"name": "ZString"},
                            },
                        ],
                    },
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
                                "name": "colors",
                                "type": {"name": "ColorTargetState[]"},
                            },
                            {
                                "name": "depth_format",
                                "type": {"name": "Format"},
                            },
                            {
                                "name": "sample_count",
                                "type": {"name": "SampleCount"},
                            },
                            {
                                "name": "polygon_mode",
                                "type": {"name": "PolygonMode"},
                            },
                            {
                                "name": "debug_name",
                                "type": {"name": "ZString"},
                            },
                        ],
                    },
                    {
                        "name": "SampleCount",
                        "kind": "enum",
                        "members": [
                            {"name": name, "type": {"name": "SampleCount"}}
                            for name in (
                                "ONE",
                                "TWO",
                                "FOUR",
                                "EIGHT",
                                "SIXTEEN",
                                "THIRTY_TWO",
                                "SIXTY_FOUR",
                            )
                        ],
                    },
                    {
                        "name": "Format",
                        "kind": "enum",
                        "members": [
                            {"name": name, "type": {"name": "Format"}}
                            for name in (
                                "UNDEFINED", "R8_UNORM", "R8_UINT",
                                "RG8_UNORM", "RGBA8_UNORM", "RGBA8_SRGB",
                                "BGRA8_UNORM", "BGRA8_SRGB", "R16_UINT",
                                "R16_FLOAT", "RG16_FLOAT", "RGBA16_FLOAT",
                                "R32_UINT", "R32_FLOAT", "RG32_FLOAT",
                                "RGBA32_FLOAT", "D32_FLOAT",
                            )
                        ],
                    },
                    {
                        "name": "TextureDesc",
                        "kind": "struct",
                        "members": [
                            {"name": "width", "type": {"name": "uint"}},
                            {"name": "height", "type": {"name": "uint"}},
                            {"name": "mip_levels", "type": {"name": "uint"}},
                            {"name": "array_layers", "type": {"name": "uint"}},
                            {"name": "format", "type": {"name": "Format"}},
                            {"name": "usage", "type": {"name": "TextureUsage"}},
                            {"name": "access", "type": {"name": "QueueRoles"}},
                            {"name": "sample_count", "type": {"name": "SampleCount"}},
                            {"name": "debug_name", "type": {"name": "ZString"}},
                        ],
                    },
                    {
                        "name": "TextureViewDesc",
                        "kind": "struct",
                        "members": [
                            {"name": "base_mip", "type": {"name": "uint"}},
                            {"name": "mip_count", "type": {"name": "uint"}},
                            {"name": "base_layer", "type": {"name": "uint"}},
                            {"name": "layer_count", "type": {"name": "uint"}},
                        ],
                    },
                    {
                        "name": "TextureFormatSupport",
                        "kind": "struct",
                        "members": [
                            {"name": "features", "type": {"name": "TextureFormatFeatures"}},
                            {"name": "sample_counts", "type": {"name": "TextureSampleCountSupport"}},
                        ],
                    },
                    {
                        "name": "LoadOp",
                        "kind": "enum",
                        "members": [
                            {"name": name, "type": {"name": "LoadOp"}}
                            for name in ("LOAD", "CLEAR", "DONT_CARE")
                        ],
                    },
                    {
                        "name": "StoreOp",
                        "kind": "enum",
                        "members": [
                            {"name": name, "type": {"name": "StoreOp"}}
                            for name in ("STORE", "DONT_CARE")
                        ],
                    },
                    {
                        "name": "ClearColor",
                        "kind": "union",
                        "members": [
                            {"name": "rgba", "type": {"name": "float[4]"}},
                            {"name": "uint_rgba", "type": {"name": "uint[4]"}},
                        ],
                    },
                    {
                        "name": "ClearDepth",
                        "kind": "struct",
                        "members": [
                            {"name": "depth", "type": {"name": "float"}},
                        ],
                    },
                    {
                        "name": "AttachmentViewDesc",
                        "kind": "struct",
                        "members": [
                            {"name": "texture", "type": {"name": "TextureHandle"}},
                            {"name": "mip_level", "type": {"name": "uint"}},
                            {"name": "array_layer", "type": {"name": "uint"}},
                        ],
                    },
                    {
                        "name": "ColorTargetDesc",
                        "kind": "struct",
                        "members": [
                            {"name": "view", "type": {"name": "AttachmentViewHandle"}},
                            {"name": "resolve_view", "type": {"name": "AttachmentViewHandle"}},
                            {"name": "load_op", "type": {"name": "LoadOp"}},
                            {"name": "store_op", "type": {"name": "StoreOp"}},
                            {"name": "clear", "type": {"name": "ClearColor"}},
                        ],
                    },
                    {
                        "name": "DepthTargetDesc",
                        "kind": "struct",
                        "members": [
                            {"name": "view", "type": {"name": "AttachmentViewHandle"}},
                            {"name": "load_op", "type": {"name": "LoadOp"}},
                            {"name": "store_op", "type": {"name": "StoreOp"}},
                            {"name": "clear", "type": {"name": "ClearDepth"}},
                        ],
                    },
                    {
                        "name": "RenderPassDesc",
                        "kind": "struct",
                        "members": [
                            {"name": "colors", "type": {"name": "ColorTargetDesc[]"}},
                            {"name": "depth", "type": {"name": "DepthTargetDesc*"}},
                            {"name": "width", "type": {"name": "uint"}},
                            {"name": "height", "type": {"name": "uint"}},
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
                        "name": "AttachmentViewHandle",
                        "kind": "struct",
                        "members": [
                            {"name": "owner", "type": {"name": "ulong"}},
                            {"name": "index", "type": {"name": "uint"}},
                            {"name": "generation", "type": {"name": "uint"}},
                        ],
                    },
                    {
                        "name": "CommandAllocatorHandle",
                        "kind": "struct",
                        "members": [
                            {"name": "owner", "type": {"name": "ulong"}},
                            {"name": "index", "type": {"name": "uint"}},
                            {"name": "generation", "type": {"name": "uint"}},
                        ],
                    },
                    {
                        "name": "CommandAllocator",
                        "kind": "struct",
                        "members": [
                            {"name": "device", "type": {"name": "Device"}},
                            {"name": "queue", "type": {"name": "Queue"}},
                            {
                                "name": "handle",
                                "type": {"name": "CommandAllocatorHandle"},
                            },
                        ],
                    },
                    {
                        "name": "CommandAllocatorDesc",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "command_buffer_capacity",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "max_resource_references_per_list",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "max_generated_preprocess_buffers_per_list",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "generated_preprocess_bytes",
                                "type": {"name": "usz"},
                            },
                            {"name": "debug_name", "type": {"name": "ZString"}},
                        ],
                    },
                    {
                        "name": "GeneratedWorkKind",
                        "kind": "enum",
                        "members": [],
                    },
                    {
                        "name": "GeneratedScratchDesc",
                        "kind": "struct",
                        "members": [
                            {"name": "pipeline", "type": {"name": "PipelineHandle"}},
                            {"name": "kind", "type": {"name": "GeneratedWorkKind"}},
                            {"name": "max_commands_per_list", "type": {"name": "uint"}},
                            {"name": "preprocess_buffer_count", "type": {"name": "uint"}},
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
                        "name": "ContractValidation",
                        "kind": "enum",
                        "members": [
                            {
                                "name": name,
                                "type": {"name": "ContractValidation"},
                            }
                            for name in (
                                "TRUSTED",
                                "OBJECT_BOUNDARIES",
                                "FULL",
                            )
                        ],
                    },
                    {
                        "name": "RuntimeDesc",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "contract_validation",
                                "type": {"name": "ContractValidation"},
                            },
                            {
                                "name": "track_resource_lifetimes",
                                "type": {"name": "bool"},
                            },
                            {
                                "name": "enable_vulkan_validation",
                                "type": {"name": "bool"},
                            },
                            {
                                "name": "enable_debug_names",
                                "type": {"name": "bool"},
                            },
                            {
                                "name": "texture_heap_capacity",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "sampler_heap_capacity",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "texture_capacity",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "pipeline_capacity",
                                "type": {"name": "uint"},
                            },
                            {
                                "name": "pipeline_cache_data",
                                "type": {"name": "char[]"},
                            },
                            {
                                "name": "application_name",
                                "type": {"name": "ZString"},
                            },
                            {
                                "name": "debug_callback",
                                "type": {"name": "DebugMessageCallback"},
                            },
                            {
                                "name": "debug_user_data",
                                "type": {"name": "void*"},
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
                            {
                                "name": "max_sampler_anisotropy",
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
                        "name": "StageMask",
                        "kind": "bitstruct",
                        "base_type": {"name": "uint"},
                        "members": [
                            {
                                "name": name,
                                "type": {"name": "bool"},
                                "bit_range": [bit, bit],
                            }
                            for bit, name in enumerate((
                                "all",
                                "host",
                                "transfer",
                                "compute",
                                "vertex_shader",
                                "fragment_shader",
                                "color_output",
                                "depth_output",
                                "present",
                            ))
                        ],
                    },
                    {
                        "name": "HazardFlags",
                        "kind": "bitstruct",
                        "base_type": {"name": "uint"},
                        "members": [
                            {
                                "name": name,
                                "type": {"name": "bool"},
                                "bit_range": [bit, bit],
                            }
                            for bit, name in enumerate((
                                "draw_arguments",
                                "descriptors",
                                "depth_stencil",
                            ))
                        ],
                    },
                    {
                        "name": "Barrier",
                        "kind": "struct",
                        "members": [
                            {"name": "before", "type": {"name": "StageMask"}},
                            {"name": "after", "type": {"name": "StageMask"}},
                            {"name": "hazards", "type": {"name": "HazardFlags"}},
                        ],
                    },
                    {
                        "name": "TextureLayout",
                        "kind": "enum",
                        "base_type": {"name": "int"},
                        "members": [
                            {"name": name, "type": {"name": "TextureLayout"}}
                            for name in (
                                "UNDEFINED",
                                "TRANSFER_SOURCE",
                                "TRANSFER_DESTINATION",
                                "SAMPLED",
                                "STORAGE",
                                "COLOR_ATTACHMENT",
                                "DEPTH_ATTACHMENT",
                                "PRESENT",
                            )
                        ],
                    },
                    {
                        "name": "TextureAccess",
                        "kind": "bitstruct",
                        "base_type": {"name": "uint"},
                        "members": [
                            {
                                "name": name,
                                "type": {"name": "bool"},
                                "bit_range": [bit, bit],
                            }
                            for bit, name in enumerate(("read", "write"))
                        ],
                    },
                    {
                        "name": "TextureState",
                        "kind": "struct",
                        "members": [
                            {"name": "layout", "type": {"name": "TextureLayout"}},
                            {"name": "stages", "type": {"name": "StageMask"}},
                            {"name": "access", "type": {"name": "TextureAccess"}},
                        ],
                    },
                    {
                        "name": "TextureBarrier",
                        "kind": "struct",
                        "members": [
                            {"name": "texture", "type": {"name": "TextureHandle"}},
                            {"name": "view", "type": {"name": "TextureViewDesc"}},
                            {"name": "before", "type": {"name": "TextureState"}},
                            {"name": "after", "type": {"name": "TextureState"}},
                        ],
                    },                    {
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
                        "name": "CompletionConsumerFlags",
                        "kind": "bitstruct",
                        "base_type": {"name": "uint"},
                        "members": [{
                            "name": "draw_arguments",
                            "type": {"name": "bool"},
                            "bit_range": [0, 0],
                        }],
                    },
                    {
                        "name": "CompletionWait",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "point",
                                "type": {"name": "CompletionPoint"},
                            },
                            {
                                "name": "before",
                                "type": {"name": "StageMask"},
                            },
                            {
                                "name": "consumers",
                                "type": {
                                    "name": "CompletionConsumerFlags",
                                },
                            },
                        ],
                    },
                    {
                        "name": "SubmitDesc",
                        "kind": "struct",
                        "members": [
                            {
                                "name": "command_lists",
                                "type": {"name": "ExecutableCommandList[]"},
                            },
                            {
                                "name": "completion_waits",
                                "type": {"name": "CompletionWait[]"},
                            },
                            {
                                "name": "readiness",
                                "type": {"name": "SwapchainReadiness"},
                            },
                            {
                                "name": "readiness_before",
                                "type": {"name": "StageMask"},
                            },
                        ],
                    },
                    {"name": "SwapchainReadiness", "kind": "struct"},
                    {
                        "name": "AcquiredImage",
                        "kind": "struct",
                        "members": [
                            {"name": "texture", "type": {"name": "TextureHandle"}},
                            {
                                "name": "attachment_view",
                                "type": {"name": "AttachmentViewHandle"},
                            },
                            {
                                "name": "readiness",
                                "type": {"name": "SwapchainReadiness"},
                            },
                            {"name": "index", "type": {"name": "uint"}},
                            {"name": "suboptimal", "type": {"name": "bool"}},
                            {"name": "prior_state", "type": {"name": "TextureState"}},
                        ],
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


def semantic_document() -> dict:
    root_docs = {"text": "Roots may be zero and are forwarded unchanged."}
    root = {"name": "root", "type": {"name": "GpuAddress"}}
    gpu = {
        "functions": [
            {"name": "cmd_dispatch", "members": [root], "docs": dict(root_docs)},
            {"name": "cmd_draw_generated", "docs": dict(root_docs)},
        ],
        "types": [{"name": "GeneratedDrawRecord", "members": [root]}],
    }
    return {"modules": {"gpu": gpu}}


class PublicApiCheckTests(unittest.TestCase):
    def test_rejects_root_documentation_mutations(self) -> None:
        cases = (
            ("cmd_draw_generated", "A nonzero root is forwarded unchanged.", "nonzero"),
            ("cmd_dispatch", "The root is forwarded unchanged.", "zero roots"),
            ("cmd_dispatch", "Zero is valid; the root is not forwarded unchanged.", "forwarded unchanged"),
        )
        for name, docs, expected in cases:
            with self.subTest(name=name, docs=docs):
                document = semantic_document()
                functions = document["modules"]["gpu"]["functions"]
                function = next(entry for entry in functions if entry["name"] == name)
                function["docs"]["text"] = docs
                failures = check_public_api.validate_generated_semantic_contracts(document)
                self.assertTrue(any(expected in failure for failure in failures))

    def test_rejects_retired_command_render_state_symbols(self) -> None:
        self.assertEqual(
            check_public_api.retired_command_render_state_errors(
                """
// bool active_render_width;
const char* note = "active_render_height";
bool depth_state_set;
"""
            ),
            ["retired command render state depth_state_set"],
        )

    def test_canonical_fixture_requires_every_public_root_function(self) -> None:
        document = {
            "modules": {
                "gpu": {
                    "functions": [
                        {"name": "create_runtime"},
                        {"name": "destroy_runtime"},
                        {
                            "name": "register_runtime",
                            "visibility": "private",
                        },
                    ],
                },
            },
        }
        source = "(void)gpu::create_runtime(&desc);\n"

        self.assertEqual(
            check_public_api.validate_canonical_function_fixture(
                document,
                source,
            ),
            ["canonical strict fixture missing gpu::destroy_runtime"],
        )

    def test_canonical_manifest_pins_every_public_entry_kind(self) -> None:
        document = {
            "modules": {
                "gpu": {
                    "functions": [
                        {
                            "name": "create_runtime",
                            "kind": "function",
                            "uid": "gpu::create_runtime",
                        },
                        {
                            "name": "register_runtime",
                            "kind": "function",
                            "uid": "gpu::register_runtime",
                            "visibility": "private",
                        },
                    ],
                    "methods": [{
                        "name": "is_valid",
                        "kind": "method",
                        "uid": "gpu::Runtime.is_valid",
                    }],
                    "types": [
                        {
                            "name": "Runtime",
                            "kind": "bitstruct",
                            "uid": "gpu::Runtime",
                        },
                        {
                            "name": "INVALID_HANDLE",
                            "kind": "fault",
                            "uid": "gpu::INVALID_HANDLE",
                        },
                    ],
                    "variables": [{
                        "name": "RUNTIME_INVALID",
                        "kind": "constant",
                        "uid": "gpu::RUNTIME_INVALID",
                    }],
                },
            },
        }
        source = """
bitstruct gpu::Runtime
constant gpu::RUNTIME_INVALID
fault gpu::INVALID_HANDLE
function gpu::create_runtime
method gpu::Runtime.is_valid
"""

        self.assertEqual(
            check_public_api.validate_canonical_surface_manifest(
                document,
                source,
            ),
            [],
        )
        self.assertEqual(
            check_public_api.validate_canonical_surface_manifest(
                document,
                source.replace(
                    "bitstruct gpu::Runtime",
                    "constant gpu::REMOVED",
                ),
            ),
            [
                (
                    "canonical manifest entry missing from generated API: "
                    "constant gpu::REMOVED"
                ),
                (
                    "generated public entry missing from canonical manifest: "
                    "bitstruct gpu::Runtime"
                ),
            ],
        )

    def test_accepts_distinct_platform_handle_modules(self) -> None:
        self.assertEqual(check_public_api.validate_document(valid_document()), [])

    def test_accepts_local_but_rejects_public_backend_declaration(self) -> None:
        document = valid_document()
        document["modules"]["gpu::internal::vk"] = {
            "functions": [
                {
                    "name": "create_native_device",
                    "uid": "gpu::internal::vk::create_native_device",
                    "visibility": "public",
                },
                {
                    "name": "sampler_filter_is_valid",
                    "uid": "gpu::internal::vk::sampler_filter_is_valid",
                    "visibility": "local",
                },
            ],
            "types": [{
                "name": "PrivateState",
                "uid": "gpu::internal::vk::PrivateState",
                "visibility": "private",
            }],
        }
        self.assertEqual(
            check_public_api.validate_generated_backend_privacy(document),
            [
                "generated gpu::internal::vk::create_native_device "
                "must remain private"
            ],
        )

    def test_rejects_generated_public_nested_backend_declaration(self) -> None:
        document = valid_document()
        document["modules"]["gpu::internal::vk::testing"] = {
            "functions": [{
                "name": "open_backend_escape",
                "uid": "gpu::internal::vk::testing::open_backend_escape",
                "visibility": "public",
            }],
        }
        self.assertIn(
            (
                "generated gpu::internal::vk::testing::open_backend_escape "
                "must remain private"
            ),
            check_public_api.validate_document(document),
        )

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
            "gpu::surface::win32::InstanceHandle must be a distinct void* type",
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
                "COMMAND_ALLOCATOR",
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

    def test_rejects_retired_recording_context_checkpoint(self) -> None:
        failures = check_public_api.validate_private_backend_source(
            Path("gpu/internal/vk/device.c3"),
            "module gpu::internal::vk @private;\nenum Checkpoint { RECORDING_CONTEXTS }\n",
        )
        self.assertIn(
            "retired backend RECORDING_CONTEXTS in gpu/internal/vk/device.c3",
            failures,
        )

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
            "SubmitDesc must match the exact stage-scoped schema",
            failures,
        )

    def test_requires_exact_stage_scoped_submit_schema(self) -> None:
        document = valid_document()
        types = document["modules"]["gpu"]["types"]
        completion_wait = next(
            entry for entry in types
            if entry["name"] == "CompletionWait"
        )
        completion_wait["members"][1]["type"]["name"] = "HazardFlags"
        submit_desc = next(
            entry for entry in types
            if entry["name"] == "SubmitDesc"
        )
        submit_desc["members"][1]["type"]["name"] = "CompletionPoint[]"
        failures = check_public_api.validate_document(document)
        self.assertIn(
            "CompletionWait must match the exact consumer-scoped schema",
            failures,
        )
        self.assertIn(
            "SubmitDesc must match the exact stage-scoped schema",
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
            "cmd_barrier",
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

    def test_requires_sample_count_enum(self) -> None:
        document = valid_document()
        sample_count = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "SampleCount"
        )
        sample_count["members"].pop()
        self.assertIn(
            "SampleCount must define the strict sample counts",
            check_public_api.validate_document(document),
        )

    def test_requires_render_sample_fields(self) -> None:
        for type_name, field_name, failure in (
            (
                "TextureDesc",
                "sample_count",
                "TextureDesc must match the strict schema",
            ),
            (
                "GraphicsPipelineDesc",
                "sample_count",
                "GraphicsPipelineDesc must match the strict schema",
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
                field["type"]["name"] = "uint"
                self.assertIn(
                    failure,
                    check_public_api.validate_document(document),
                )

    def test_requires_pipeline_identity_schemas(self) -> None:
        for type_name in (
            "BlendState",
            "ColorTargetState",
            "DynamicRasterState",
            "ComputePipelineDesc",
            "GraphicsPipelineDesc",
        ):
            with self.subTest(type_name=type_name):
                document = valid_document()
                definition = next(
                    entry for entry in document["modules"]["gpu"]["types"]
                    if entry["name"] == type_name
                )
                definition["members"].pop()
                self.assertIn(
                    f"{type_name} must match the strict schema",
                    check_public_api.validate_document(document),
                )

    def test_requires_color_write_mask_contract(self) -> None:
        document = valid_document()
        color_write_mask = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "ColorWriteMask"
        )
        color_write_mask["members"][3]["bit_range"] = [4, 4]
        self.assertIn(
            "ColorWriteMask must match the strict channel bits",
            check_public_api.validate_document(document),
        )

        document = valid_document()
        color_write_all = document["modules"]["gpu"]["variables"][0]
        color_write_all["value"] = "{}"
        self.assertIn(
            "COLOR_WRITE_ALL must enable every color channel",
            check_public_api.validate_document(document),
        )

    def test_requires_explicit_swapchain_acquisition_timeout_contract(self) -> None:
        document = valid_document()
        acquire = next(
            entry for entry in document["modules"]["gpu"]["functions"]
            if entry["name"] == "acquire_next_image"
        )
        acquire["members"][2]["default_value"] = "1000000000"
        self.assertIn(
            "acquire_next_image must default timeout_ns to zero",
            check_public_api.validate_document(document),
        )

        document = valid_document()
        acquire = next(
            entry for entry in document["modules"]["gpu"]["functions"]
            if entry["name"] == "acquire_next_image"
        )
        acquire["members"].pop()
        self.assertIn(
            "acquire_next_image has the wrong parameters",
            check_public_api.validate_document(document),
        )

        document = valid_document()
        timeout = next(
            entry for entry in document["modules"]["gpu"]["variables"]
            if entry["name"] == "TIMEOUT_INFINITE"
        )
        timeout["value"] = "1000000000"
        self.assertIn(
            "TIMEOUT_INFINITE must equal ulong::max",
            check_public_api.validate_document(document),
        )

    def test_rejects_retired_backend_acquisition_timeout_policy(self) -> None:
        failures = check_public_api.validate_private_backend_source(
            Path("gpu/internal/vk/swapchain.c3"),
            "const ulong ACQUIRE_TIMEOUT_NS = 1_000_000_000;\n",
        )
        self.assertIn(
            "retired backend ACQUIRE_TIMEOUT_NS in gpu/internal/vk/swapchain.c3",
            failures,
        )

    def test_requires_full_render_pass_schema(self) -> None:
        mutations = (
            (
                "LoadOp",
                lambda entry: entry["members"].pop(),
                "LoadOp must define the strict load operations",
            ),
            (
                "StoreOp",
                lambda entry: entry["members"].pop(),
                "StoreOp must define the strict store operations",
            ),
            (
                "ClearColor",
                lambda entry: entry["members"].pop(),
                "ClearColor must expose typed color values",
            ),
            (
                "ClearDepth",
                lambda entry: entry["members"].pop(),
                "ClearDepth must match the strict schema",
            ),
            (
                "AttachmentViewDesc",
                lambda entry: entry["members"].pop(),
                "AttachmentViewDesc must match the strict schema",
            ),
            (
                "ColorTargetDesc",
                lambda entry: entry["members"].pop(),
                "ColorTargetDesc must match the strict schema",
            ),
            (
                "DepthTargetDesc",
                lambda entry: entry["members"].pop(),
                "DepthTargetDesc must match the strict schema",
            ),
            (
                "RenderPassDesc",
                lambda entry: entry["members"].pop(),
                "RenderPassDesc must match the strict schema",
            ),
        )
        for type_name, mutate, failure in mutations:
            with self.subTest(type_name=type_name):
                document = valid_document()
                entry = next(
                    item for item in document["modules"]["gpu"]["types"]
                    if item["name"] == type_name
                )
                mutate(entry)
                self.assertIn(
                    failure,
                    check_public_api.validate_document(document),
                )

    def test_rejects_public_render_pass_objects(self) -> None:
        document = valid_document()
        document["modules"]["gpu"]["types"].extend((
            {"name": "RenderPassHandle", "kind": "struct", "members": []},
            {"name": "FramebufferHandle", "kind": "struct", "members": []},
        ))
        document["modules"]["gpu"]["functions"].extend((
            api_function("create_render_pass", "RenderPassHandle?"),
            api_function("create_framebuffer", "FramebufferHandle?"),
        ))
        failures = check_public_api.validate_document(document)
        for symbol in (
            "RenderPassHandle",
            "FramebufferHandle",
            "create_render_pass",
            "create_framebuffer",
        ):
            self.assertIn(symbol, failures)

    def test_requires_resolve_attachment_fields(self) -> None:
        document = valid_document()
        desc = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "ColorTargetDesc"
        )
        desc["members"] = [
            member for member in desc["members"]
            if member["name"] != "resolve_view"
        ]
        self.assertIn(
            "ColorTargetDesc must match the strict schema",
            check_public_api.validate_document(document),
        )

    def test_requires_render_pass_command_signatures(self) -> None:
        document = valid_document()
        begin = next(
            entry for entry in document["modules"]["gpu"]["functions"]
            if entry["name"] == "cmd_begin_render_pass"
        )
        begin["members"][1]["type"]["name"] = "RenderPassHandle"
        self.assertIn(
            "cmd_begin_render_pass has the wrong parameters",
            check_public_api.validate_document(document),
        )

        document = valid_document()
        combined = next(
            entry for entry in document["modules"]["gpu"]["functions"]
            if entry["name"] == "cmd_begin_render_pass_with_state"
        )
        combined["members"][2]["name"] = "initial_state"
        self.assertIn(
            "cmd_begin_render_pass_with_state has the wrong parameters",
            check_public_api.validate_document(document),
        )

    def test_requires_graphics_state_member_schemas(self) -> None:
        for type_name in ("Viewport", "ScissorRect", "DepthState"):
            for mutation in ("order", "type"):
                with self.subTest(type_name=type_name, mutation=mutation):
                    document = valid_document()
                    definition = next(
                        entry
                        for entry in document["modules"]["gpu"]["types"]
                        if entry["name"] == type_name
                    )
                    if mutation == "order":
                        definition["members"][0:2] = reversed(
                            definition["members"][0:2]
                        )
                    else:
                        definition["members"][0]["type"]["name"] = "uint"
                    self.assertIn(
                        f"{type_name} must match the strict schema",
                        check_public_api.validate_document(document),
                    )

    def test_requires_complete_graphics_state_schema(self) -> None:
        for member_name in ("viewport", "scissor", "raster", "depth"):
            with self.subTest(member_name=member_name):
                document = valid_document()
                state = next(
                    entry for entry in document["modules"]["gpu"]["types"]
                    if entry["name"] == "GraphicsState"
                )
                state["members"] = [
                    member for member in state["members"]
                    if member["name"] != member_name
                ]
                self.assertIn(
                    "GraphicsState must match the strict schema",
                    check_public_api.validate_document(document),
                )

    def test_requires_explicit_graphics_state_function_signatures(self) -> None:
        mutations = (
            (
                "minimal begin parameter",
                "cmd_begin_render_pass",
                lambda function: function["members"].pop(),
                "cmd_begin_render_pass has the wrong parameters",
            ),
            (
                "combined begin state parameter",
                "cmd_begin_render_pass_with_state",
                lambda function: function["members"].pop(),
                "cmd_begin_render_pass_with_state has the wrong parameters",
            ),
            (
                "graphics state setter parameter",
                "cmd_set_graphics_state",
                lambda function: function["members"].pop(),
                "cmd_set_graphics_state has the wrong parameters",
            ),
            (
                "graphics state setter return",
                "cmd_set_graphics_state",
                lambda function: function["return_type"].update(
                    {"name": "void"}
                ),
                "cmd_set_graphics_state has the wrong return type",
            ),
            (
                "full render state dimensions",
                "full_render_graphics_state",
                lambda function: function["members"].pop(),
                "full_render_graphics_state has the wrong parameters",
            ),
            (
                "full render state optional return",
                "full_render_graphics_state",
                lambda function: function["return_type"].update(
                    {"name": "GraphicsState"}
                ),
                "full_render_graphics_state has the wrong return type",
            ),
        )
        for label, function_name, mutate, failure in mutations:
            with self.subTest(label=label):
                document = valid_document()
                function = next(
                    entry
                    for entry in document["modules"]["gpu"]["functions"]
                    if entry["name"] == function_name
                )
                mutate(function)
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
            "GraphicsPipelineDesc must match the strict schema",
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

    def test_accepts_global_semantic_barrier_contract(self) -> None:
        self.assertEqual(
            [],
            check_public_api.validate_document(valid_document()),
        )

    def test_requires_compositional_texture_helper_signatures(self) -> None:
        for name in (
            "sampled_at",
            "storage_at",
            "texture_transition",
            "texture_view_transition",
        ):
            with self.subTest(name=name):
                document = valid_document()
                function = next(
                    entry for entry in document["modules"]["gpu"]["functions"]
                    if entry["name"] == name
                )
                function["members"][-1]["type"]["name"] = "TextureUse"
                self.assertIn(
                    f"{name} has the wrong parameters",
                    check_public_api.validate_document(document),
                )

    def test_rejects_global_barrier_flag_schema_drift(self) -> None:
        for name in (
            "StageMask",
            "HazardFlags",
            "CompletionConsumerFlags",
        ):
            with self.subTest(name=name):
                document = valid_document()
                flags = next(
                    entry for entry in document["modules"]["gpu"]["types"]
                    if entry["name"] == name
                )
                flags["members"][0]["bit_range"] = [1, 1]
                self.assertIn(
                    f"{name} must match the exact semantic flag schema",
                    check_public_api.validate_document(document),
                )

    def test_rejects_retired_generic_barrier_symbols(self) -> None:
        for name, kind, failure in (
            (
                "BufferBarrier",
                "struct",
                "retired resource-scoped BufferBarrier",
            ),
            (
                "GlobalBarrier",
                "struct",
                "retired Vulkan-shaped GlobalBarrier",
            ),
        ):
            with self.subTest(name=name):
                document = valid_document()
                document["modules"]["gpu"]["types"].append({
                    "name": name,
                    "kind": kind,
                })
                self.assertIn(
                    failure,
                    check_public_api.validate_document(document),
                )

    def test_rejects_retired_texture_transition_types(self) -> None:
        for name, failure in (
            ("Stage", "Stage"),
            ("Hazard", "Hazard"),
            ("TextureUse", "retired TextureUse cross-product"),
        ):
            with self.subTest(name=name):
                document = valid_document()
                document["modules"]["gpu"]["types"].append({
                    "name": name,
                    "kind": "enum",
                })
                self.assertIn(
                    failure,
                    check_public_api.validate_document(document),
                )

    def test_rejects_backend_shaped_texture_barrier_schema(self) -> None:
        document = valid_document()
        barrier = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "TextureBarrier"
        )
        barrier["members"] = [
            {"name": "texture", "type": {"name": "TextureHandle"}},
            {"name": "old_layout", "type": {"name": "TextureLayout"}},
            {"name": "new_layout", "type": {"name": "TextureLayout"}},
        ]
        self.assertIn(
            "TextureBarrier must contain only a texture range and compositional states",
            check_public_api.validate_document(document),
        )

    def test_rejects_texture_state_schema_drift(self) -> None:
        for name, expected_failure in (
            ("TextureLayout", "TextureLayout must match the exact semantic layout schema"),
            ("TextureAccess", "TextureAccess must match the exact semantic access schema"),
            ("TextureState", "TextureState must contain exactly layout, stages, and access"),
        ):
            with self.subTest(name=name):
                document = valid_document()
                definition = next(
                    entry for entry in document["modules"]["gpu"]["types"]
                    if entry["name"] == name
                )
                definition["members"].pop()
                self.assertIn(
                    expected_failure,
                    check_public_api.validate_document(document),
                )

    def test_rejects_texture_access_bit_drift(self) -> None:
        document = valid_document()
        access = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "TextureAccess"
        )
        access["members"][1]["bit_range"] = [2, 2]
        self.assertIn(
            "TextureAccess must match the exact semantic flag schema",
            check_public_api.validate_document(document),
        )

    def test_rejects_texture_layout_base_type_drift(self) -> None:
        document = valid_document()
        layout = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "TextureLayout"
        )
        layout["base_type"]["name"] = "uint"
        self.assertIn(
            "TextureLayout must use int as its base type",
            check_public_api.validate_document(document),
        )

    def test_rejects_retired_acquired_prior_use(self) -> None:
        document = valid_document()
        acquired = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "AcquiredImage"
        )
        acquired["members"][-1] = {
            "name": "prior_use",
            "type": {"name": "TextureUse"},
        }
        failures = check_public_api.validate_document(document)
        self.assertIn("retired TextureUse cross-product", failures)
        self.assertIn("retired AcquiredImage.prior_use", failures)
        self.assertIn(
            "AcquiredImage must carry borrowed render handles, readiness, and compositional prior state",
            failures,
        )

    def test_rejects_resource_shaped_barrier_schema(self) -> None:
        document = valid_document()
        barrier = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "Barrier"
        )
        barrier["members"] = [
            {"name": "span", "type": {"name": "GpuSpan"}},
            *barrier["members"],
        ]
        self.assertIn(
            "Barrier must contain only semantic stage masks and hazard flags",
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

    def test_enforces_root_interface_and_implementation_roles(self) -> None:
        interface_failures = check_public_api.validate_root_facade_source(
            Path("gpu/gpu.c3i"),
            "module gpu;\nfn void misplaced() {}\n",
        )
        self.assertIn(
            "gpu/gpu.c3i:2 public interface may not contain callable declarations",
            interface_failures,
        )

        implementation_failures = check_public_api.validate_root_facade_source(
            Path("gpu/gpu.c3"),
            "module gpu;\nstruct Misplaced {}\n"
            "$assert Misplaced::size == 0;\n"
            "fn void hidden() @private {}\n",
        )
        self.assertIn(
            "gpu/gpu.c3:2 public implementation may not contain non-callable declarations",
            implementation_failures,
        )
        self.assertIn(
            "gpu/gpu.c3:3 public implementation may not contain layout assertions",
            implementation_failures,
        )
        self.assertIn(
            "gpu/gpu.c3:4 public implementation may not contain private declarations",
            implementation_failures,
        )

    def test_classifies_macro_and_definition_declarations(self) -> None:
        interface_failures = check_public_api.validate_root_facade_source(
            Path("gpu/gpu.c3i"),
            "module gpu;\nmacro misplaced() {}\n",
        )
        self.assertIn(
            "gpu/gpu.c3i:2 public interface may not contain callable declarations",
            interface_failures,
        )
        implementation_failures = check_public_api.validate_root_facade_source(
            Path("gpu/gpu.c3"),
            "module gpu;\nfaultdef BAD;\nconstdef LIMIT = 1;\n",
        )
        self.assertEqual(
            sum(
                "non-callable declarations" in failure
                for failure in implementation_failures
            ),
            2,
        )

    def test_enforces_exact_surface_source_roles(self) -> None:
        interface = (
            "module gpu::surface::win32;\n"
            "import gpu @public;\n"
            "typedef InstanceHandle = void*;\n"
            "typedef WindowHandle = void*;\n"
        )
        self.assertEqual(
            check_public_api.validate_surface_source(
                Path("gpu/surface/win32/surface.c3i"),
                interface,
            ),
            [],
        )
        failures = check_public_api.validate_surface_source(
            Path("gpu/surface/win32/surface.c3i"),
            interface + "struct Escape {}\nfn void misplaced() {}\n",
        )
        self.assertIn(
            "gpu/surface/win32/surface.c3i may not contain callable declarations",
            failures,
        )
        self.assertIn(
            "gpu/surface/win32/surface.c3i may only contain surface handle typedefs",
            failures,
        )

        implementation_failures = check_public_api.validate_surface_source(
            Path("gpu/surface/win32/surface.c3"),
            "module gpu::surface::win32;\n"
            "import gpu @public;\n"
            "import gpu::internal @public;\n"
            "fn void create_surface() {}\n"
            "fn void extra() {}\n",
        )
        self.assertIn(
            "gpu/surface/win32/surface.c3 must contain exactly create_surface",
            implementation_failures,
        )
        reordered_implementation = (
            "module gpu::surface::win32;\n"
            "import gpu::internal @public;\n"
            "import gpu @public;\n"
            "fn void create_surface() {}\n"
        )
        self.assertEqual(
            check_public_api.validate_surface_source(
                Path("gpu/surface/win32/surface.c3"),
                reordered_implementation,
            ),
            [],
        )

    def test_rejects_internal_and_binding_types_in_all_public_metadata(self) -> None:
        document = valid_document()
        document["modules"]["gpu"]["functions"][0]["return_type"] = {
            "name": "gpu::internal::RuntimeData*",
        }
        document["modules"]["gpu::surface::win32"]["functions"][0][
            "return_type"
        ] = {"name": "vk::Device"}
        failures = check_public_api.validate_public_metadata_boundaries(document)
        self.assertIn(
            "gpu public metadata contains internal gpu type",
            failures,
        )
        self.assertIn(
            "gpu::surface::win32 public metadata contains Vulkan binding type",
            failures,
        )

    def test_allows_typed_private_record_only_in_command_tokens(self) -> None:
        document = valid_document()
        document["modules"]["gpu"]["types"] = [
            {
                "name": token,
                "kind": "struct",
                "members": [{
                    "name": "record",
                    "type": {
                        "name": "CommandRecord*",
                        "uid": "gpu::internal::CommandRecord",
                    },
                }],
            }
            for token in ("CommandList", "ExecutableCommandList")
        ]
        self.assertEqual(
            check_public_api.validate_public_metadata_boundaries(document),
            [],
        )

        document["modules"]["gpu"]["types"][0]["members"].append({
            "name": "leak",
            "type": {
                "name": "DeviceData*",
                "uid": "gpu::internal::DeviceData",
            },
        })
        self.assertIn(
            "gpu public metadata contains internal gpu type",
            check_public_api.validate_public_metadata_boundaries(document),
        )

    def test_rejects_nested_internal_and_backend_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested_internal = root / "gpu" / "internal" / "probe" / "escape.c3"
            nested_internal.parent.mkdir(parents=True)
            nested_internal.write_text(
                "module gpu::internal @private;\n",
                encoding="utf-8",
            )
            nested_backend = root / "gpu" / "internal" / "vk" / "probe" / "escape.c3"
            nested_backend.parent.mkdir(parents=True)
            nested_backend.write_text(
                "module gpu::internal::vk @private;\n",
                encoding="utf-8",
            )
            with mock.patch.object(check_public_api, "ROOT", root):
                self.assertEqual(
                    check_public_api.scan_private_internal_modules(),
                    [
                        "unexpected backend-independent internal source "
                        "gpu/internal/probe/escape.c3"
                    ],
                )
                self.assertEqual(
                    check_public_api.scan_private_backend_modules(),
                    [
                        "unexpected Vulkan backend source "
                        "gpu/internal/vk/probe/escape.c3"
                    ],
                )

    def test_rejects_retired_vulkan_namespace_in_live_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                Path("gpu/probe.c3"),
                Path("test/probe.c3"),
                Path("scripts/probe.py"),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("gpu::" + "vk", encoding="utf-8")
            (root / "manifest.json").write_text(
                "gpu/" + "vk",
                encoding="utf-8",
            )
            with mock.patch.object(check_public_api, "ROOT", root):
                self.assertEqual(
                    check_public_api.scan_retired_vulkan_namespace(),
                    [
                        "retired Vulkan namespace in gpu/probe.c3",
                        "retired Vulkan namespace in manifest.json",
                        "retired Vulkan namespace in scripts/probe.py",
                        "retired Vulkan namespace in test/probe.c3",
                    ],
                )

    def test_requires_private_vulkan_backend_modules(self) -> None:
        relative = Path("gpu/internal/vk/buffer.c3")
        self.assertEqual(
            check_public_api.validate_private_backend_source(
                relative,
                "module gpu::internal::vk @private;",
            ),
            [],
        )
        for declaration in (
            "module gpu::internal::vk;",
            "module gpu::internal::vk @public;",
        ):
            with self.subTest(declaration=declaration):
                self.assertEqual(
                    check_public_api.validate_private_backend_source(
                        relative,
                        declaration,
                    ),
                    [
                        "gpu/internal/vk/buffer.c3 must declare the private "
                        "gpu::internal::vk backend module"
                    ],
                )

        self.assertEqual(
            check_public_api.validate_private_backend_source(
                relative,
                "module gpu;",
            ),
            [
                (
                    "gpu/internal/vk/buffer.c3:1 backend file may only declare "
                    "gpu::internal::vk modules, found gpu"
                ),
            ],
        )
        self.assertEqual(
            check_public_api.validate_private_backend_source(
                relative,
                "module gpu::internal::vk::probe @private;",
            ),
            [
                "gpu/internal/vk/buffer.c3:1 backend file may only declare "
                "gpu::internal::vk modules, found gpu::internal::vk::probe"
            ],
        )

    def test_requires_private_internal_module(self) -> None:
        relative = Path("gpu/internal/device.c3")
        self.assertEqual(
            check_public_api.validate_private_internal_source(
                relative,
                "module gpu::internal @private;\nimport gpu @public;\n",
            ),
            [],
        )
        self.assertEqual(
            check_public_api.validate_private_internal_source(
                relative,
                "module gpu::internal;\n",
            ),
            [
                "gpu/internal/device.c3 must declare private module "
                "gpu::internal"
            ],
        )
        self.assertEqual(
            check_public_api.validate_private_internal_source(
                relative,
                "module gpu::probe @private;\n",
            ),
            [
                "gpu/internal/device.c3:1 internal file may only declare "
                "gpu::internal, found gpu::probe"
            ],
        )

    def test_rejects_internal_source_visibility_escape(self) -> None:
        relative = Path("gpu/internal/helpers.c3")
        source = (
            "module gpu::internal @private;\n"
            "import gpu @public;\n"
            "fn void leaked_helper() @public {}\n"
        )
        self.assertEqual(
            check_public_api.validate_private_internal_source(relative, source),
            [
                "gpu/internal/helpers.c3:3 internal declaration may not use "
                "@public"
            ],
        )

    def test_rejects_backend_source_visibility_escapes(self) -> None:
        relative = Path("gpu/internal/vk/helpers.c3")
        nested_source = (
            "module gpu::internal::vk @private;\n"
            "import gpu @public;\n"
            "module gpu::internal::vk::probe;\n"
            "fn void open_backend_escape() {}\n"
        )
        self.assertEqual(
            check_public_api.validate_private_backend_source(
                relative,
                nested_source,
            ),
            [
                "gpu/internal/vk/helpers.c3:3 backend file may only declare "
                "gpu::internal::vk modules, found gpu::internal::vk::probe"
            ],
        )

        public_declaration_source = (
            "module gpu::internal::vk @private;\n"
            "import gpu @public;\n"
            "fn void open_backend_escape() @public {}\n"
        )
        self.assertEqual(
            check_public_api.validate_private_backend_source(
                relative,
                public_declaration_source,
            ),
            [
                "gpu/internal/vk/helpers.c3:3 "
                "backend declaration may not use @public"
            ],
        )

        wrong_module_source = (
            "module gpu::internal::vk @private;\n"
            "import gpu @public;\n"
            "module gpu::probe;\n"
            "fn int probe_leak() { return 7; }\n"
        )
        self.assertEqual(
            check_public_api.validate_private_backend_source(
                relative,
                wrong_module_source,
            ),
            [
                (
                    "gpu/internal/vk/helpers.c3:3 backend file may only declare "
                    "gpu::internal::vk modules, found gpu::probe"
                ),
            ],
        )

    def test_rejects_retired_standalone_backend_lifecycle(self) -> None:
        relative = Path("gpu/internal/vk/device.c3")
        source = (
            "module gpu::internal::vk @private;\n"
            "struct StandaloneDeviceConfig {}\n"
            "fn void create_standalone_device_with_probe() {}\n"
        )

        self.assertEqual(
            check_public_api.validate_private_backend_source(relative, source),
            [
                (
                    "retired backend StandaloneDeviceConfig "
                    "in gpu/internal/vk/device.c3"
                ),
                (
                    "retired backend create_standalone_device_with_probe "
                    "in gpu/internal/vk/device.c3"
                ),
            ],
        )

    def test_rejects_retired_texture_lowering_paths(self) -> None:
        relative = Path("gpu/internal/vk/sync.c3")
        source = (
            "module gpu::internal::vk @private;\n"
            "struct TextureUseScope {}\n"
            "fn void texture_use_to_vk() {}\n"
            "fn TextureBarrierRejection texture_barrier_rejection() {}\n"
            "fn TextureBarrierRejection texture_barrier_queue_rejection() {}\n"
            "fn void texture_transition_range() {}\n"
            "fn void texture_barrier_to_vk() {}\n"
        )

        failures = check_public_api.validate_private_backend_source(
            relative,
            source,
        )
        self.assertIn(
            "retired backend TextureUseScope in gpu/internal/vk/sync.c3",
            failures,
        )
        self.assertIn(
            "retired backend texture_use_ in gpu/internal/vk/sync.c3",
            failures,
        )
        self.assertIn(
            "retired backend fn TextureBarrierRejection "
            "texture_barrier_rejection( in gpu/internal/vk/sync.c3",
            failures,
        )
        self.assertIn(
            "retired backend fn TextureBarrierRejection "
            "texture_barrier_queue_rejection( in gpu/internal/vk/sync.c3",
            failures,
        )
        self.assertIn(
            "retired backend texture_transition_range( in gpu/internal/vk/sync.c3",
            failures,
        )
        self.assertIn(
            "retired backend texture_barrier_to_vk( in gpu/internal/vk/sync.c3",
            failures,
        )

    def test_rejects_sibling_modules_in_public_sources(self) -> None:
        relative = Path("gpu/gpu.c3")
        self.assertEqual(
            check_public_api.validate_public_module_source(
                relative,
                "struct AllocationInfo {}\n",
            ),
            [
                "gpu/gpu.c3 must declare public module gpu",
            ],
        )

        source = (
            "module gpu;\n"
            "struct AllocationInfo {}\n"
            "module gpu::util;\n"
            "fn int probe_leak() { return 7; }\n"
        )
        self.assertEqual(
            check_public_api.validate_public_module_source(
                relative,
                source,
            ),
            [
                (
                    "gpu/gpu.c3:3 public source may only declare "
                    "gpu, found gpu::util"
                ),
            ],
        )

        self.assertEqual(
            check_public_api.validate_public_module_source(
                Path("gpu/surface/win32/surface.c3"),
                "module gpu::surface::win32;\n",
            ),
            [],
        )
        self.assertEqual(
            check_public_api.validate_public_module_source(
                Path("gpu/surface/util/surface.c3"),
                "module gpu::surface::util;\n",
            ),
            [
                (
                    "gpu/surface/util/surface.c3:1 public source may only "
                    "declare gpu, found gpu::surface::util"
                ),
            ],
        )

    def test_scans_only_the_canonical_backend_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_root = root / "gpu"
            gpu_root.mkdir(parents=True)
            (gpu_root / "gpu.c3").write_text(
                "module gpu;\nfn void probe() {}\n",
                encoding="utf-8",
            )
            (gpu_root / "gpu.c3i").write_text(
                "module gpu;\nstruct Probe {}\n",
                encoding="utf-8",
            )
            for platform, types in check_public_api.SURFACE_TYPES.items():
                surface_root = gpu_root / "surface" / platform
                surface_root.mkdir(parents=True)
                module = f"gpu::surface::{platform}"
                typedefs = "".join(
                    f"typedef {name} = {target};\n"
                    for name, target in types
                )
                (surface_root / "surface.c3i").write_text(
                    f"module {module};\nimport gpu @public;\n{typedefs}",
                    encoding="utf-8",
                )
                (surface_root / "surface.c3").write_text(
                    f"module {module};\n"
                    "import gpu @public;\n"
                    "import gpu::internal @public;\n"
                    "fn void create_surface() {}\n",
                    encoding="utf-8",
                )
            sibling = root / "gpu" / "surface" / "vk" / "escape.c3"
            sibling.parent.mkdir(parents=True)
            sibling.write_text(
                "module gpu::escape;\n",
                encoding="utf-8",
            )
            backend = root / "gpu" / "internal" / "vk" / "escape.c3i"
            backend.parent.mkdir(parents=True)
            backend.write_text(
                "module gpu::escape;\n",
                encoding="utf-8",
            )

            with mock.patch.object(check_public_api, "ROOT", root):
                self.assertEqual(
                    check_public_api.scan_public_module_sources(),
                    [
                        "unexpected public source gpu/surface/vk/escape.c3",
                    ],
                )
                self.assertEqual(
                    check_public_api.scan_private_backend_modules(),
                    [
                        "unexpected Vulkan backend source "
                        "gpu/internal/vk/escape.c3i",
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
        self.assertIn("begin_commands must take one CommandAllocator*", failures)
        self.assertIn(
            "end_commands must return ExecutableCommandList?",
            failures,
        )

    def test_rejects_retired_sampler_identity_schema(self) -> None:
        document = valid_document()
        document["modules"]["gpu"]["types"].append({
            "name": "Sampler",
            "kind": "struct",
            "members": [],
        })
        self.assertIn(
            "retired public type Sampler",
            check_public_api.validate_document(document),
        )

    def test_requires_sampler_capabilities(self) -> None:
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
        self.assertIn(
            "DeviceCaps.max_sampler_anisotropy must be a float",
            check_public_api.validate_document(document),
        )

    def test_requires_runtime_device_defaults_and_capacities(self) -> None:
        document = valid_document()
        types = document["modules"]["gpu"]["types"]
        desc = next(entry for entry in types if entry["name"] == "RuntimeDesc")
        caps = next(entry for entry in types if entry["name"] == "DeviceCaps")
        desc["members"][3]["type"]["name"] = "usz"
        caps["members"] = [
            member for member in caps["members"]
            if member["name"] != "sampler_heap_capacity"
        ]
        failures = check_public_api.validate_document(document)
        self.assertIn("RuntimeDesc must match the strict schema", failures)
        self.assertIn("DeviceCaps.sampler_heap_capacity must be a uint", failures)

    def test_requires_contract_validation_policy_order(self) -> None:
        document = valid_document()
        policy = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "ContractValidation"
        )
        policy["members"].reverse()
        self.assertIn(
            "ContractValidation must match the strict policy order",
            check_public_api.validate_document(document),
        )

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

    def test_rejects_vulkan_shaped_generated_work_capabilities(self) -> None:
        document = valid_document()
        caps = next(
            entry for entry in document["modules"]["gpu"]["types"]
            if entry["name"] == "DeviceCaps"
        )
        native_fields = (
            "device_generated_commands",
            "supported_indirect_commands_shader_stages",
            "max_indirect_sequence_count",
            "max_indirect_commands_token_count",
            "max_indirect_commands_token_offset",
            "max_indirect_commands_indirect_stride",
        )
        caps["members"].extend(
            {"name": name, "type": {"name": "uint"}}
            for name in native_fields
        )
        failures = check_public_api.validate_document(document)
        for name in native_fields:
            with self.subTest(name=name):
                self.assertIn(name, failures)

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
        desc = next(entry for entry in types if entry["name"] == "RuntimeDesc")
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
        intern["return_type"]["name"] = "Sampler?"
        functions.append(api_function(
            "publish_sampler",
            "SamplerIndex?",
            ("device", "Device*"),
            ("sampler", "Sampler"),
        ))
        failures = check_public_api.validate_document(document)
        self.assertIn("intern_sampler has the wrong return type", failures)
        self.assertIn("retired public function publish_sampler", failures)

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

    def test_rejects_wrong_attachment_and_scratch_contracts(self) -> None:
        document = valid_document()
        functions = document["modules"]["gpu"]["functions"]
        create = next(entry for entry in functions
            if entry["name"] == "create_attachment_view")
        destroy = next(entry for entry in functions
            if entry["name"] == "destroy_attachment_view")
        reserve = next(entry for entry in functions
            if entry["name"] == "reserve_generated_scratch")
        create["return_type"]["name"] = "TextureView?"
        destroy["members"][1]["type"]["name"] = "TextureView"
        reserve["members"][1]["type"]["name"] = "GeneratedScratchDesc"
        failures = check_public_api.validate_document(document)
        self.assertIn("create_attachment_view has the wrong return type", failures)
        self.assertIn("destroy_attachment_view has the wrong parameters", failures)
        self.assertIn("reserve_generated_scratch has the wrong parameters", failures)

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
