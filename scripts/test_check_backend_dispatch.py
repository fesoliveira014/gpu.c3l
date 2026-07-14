from __future__ import annotations

import unittest

from scripts import check_backend_dispatch


class BackendDispatchCheckTests(unittest.TestCase):
    def test_accepts_owned_dispatch(self) -> None:
        sources = {
            "gpu/vk/device.c3": "state.dispatch.create_swapchain(...);",
        }
        self.assertEqual(
            check_backend_dispatch.find_global_dispatch_references(sources),
            [],
        )

    def test_rejects_generated_global_dispatch(self) -> None:
        sources = {
            "gpu/vk/debug.c3": "vk::try_set_debug_utils_object_name_ext(...);",
            "gpu/vk/device.c3": "vk::load_extensions(state.instance);",
            "gpu/vk/swapchain.c3": "vk::extensions.vk_queue_present_khr(...);",
        }
        self.assertEqual(
            check_backend_dispatch.find_global_dispatch_references(sources),
            [
                "gpu/vk/debug.c3:1: vk::try_set_debug_utils_object_name_ext",
                "gpu/vk/device.c3:1: vk::load_extensions",
                "gpu/vk/swapchain.c3:1: vk::extensions",
            ],
        )


if __name__ == "__main__":
    unittest.main()
