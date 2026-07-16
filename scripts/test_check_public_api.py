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
                "functions": [],
                "types": [],
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


if __name__ == "__main__":
    unittest.main()
