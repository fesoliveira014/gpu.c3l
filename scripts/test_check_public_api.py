from __future__ import annotations

import unittest

from scripts import check_public_api


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
                ],
                "types": [
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


if __name__ == "__main__":
    unittest.main()
