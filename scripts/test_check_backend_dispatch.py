from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import check_backend_dispatch


class BackendDispatchCheckTests(unittest.TestCase):
    def test_accepts_owned_dispatch(self) -> None:
        sources = {
            "gpu/vk/device.c3": "state.dispatch.create_swapchain(...);",
        }
        self.assertEqual(
            check_backend_dispatch.find_global_dispatch_references(sources, set()),
            [],
        )

    def test_derives_generated_singleton_wrappers(self) -> None:
        generated = """
fn DebugUtilsLabelEXT debug_utils_label_ext() => {};
fn void get_descriptor_ext(Device device) => extensions.vk_get_descriptor_ext(device);
fn void? try_create_swapchain_khr(Device device) {
    Result result = extensions.vk_create_swapchain_khr(device);
}
"""
        self.assertEqual(
            check_backend_dispatch.generated_singleton_wrappers(generated),
            {"get_descriptor_ext", "try_create_swapchain_khr"},
        )

    def test_rejects_generated_global_dispatch(self) -> None:
        wrappers = {"try_create_swapchain_khr"}
        sources = {
            "gpu/vk/compat/swapchain.c3": "Callback fn = vk::try_create_swapchain_khr;",
            "gpu/vk/device.c3": "vk::load_extensions(state.instance);",
            "gpu/vk/swapchain.c3": "vk::extensions.vk_queue_present_khr(...);",
        }
        self.assertEqual(
            check_backend_dispatch.find_global_dispatch_references(sources, wrappers),
            [
                "gpu/vk/compat/swapchain.c3:1: vk::try_create_swapchain_khr",
                "gpu/vk/device.c3:1: vk::load_extensions",
                "gpu/vk/swapchain.c3:1: vk::extensions",
            ],
        )

    def test_scan_loads_nested_backend_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = root / "lib" / "vk.c3l"
            nested = root / "gpu" / "vk" / "compat"
            binding.mkdir(parents=True)
            nested.mkdir(parents=True)
            (binding / "commands.c3").write_text(
                "fn void try_create_swapchain_khr() { extensions.vk_create_swapchain_khr(); }\n",
                encoding="utf-8",
            )
            (nested / "swapchain.c3").write_text(
                "module gpu::vk::compat;\nvk::try_create_swapchain_khr();\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_backend_dispatch.scan_backend_sources(root),
                ["gpu/vk/compat/swapchain.c3:2: vk::try_create_swapchain_khr"],
            )


if __name__ == "__main__":
    unittest.main()
