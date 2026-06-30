module gpu::vk;

import gpu;
import vk;
import std::core::mem;

const ZString VALIDATION_LAYER      = "VK_LAYER_KHRONOS_validation";
const ZString DEBUG_UTILS_EXTENSION = "VK_EXT_debug_utils";

<* Create the Vulkan instance at API 1.3. When validation is requested, enable
   the Khronos validation layer and the debug-utils extension. *>
fn vk::Instance? create_vk_instance(gpu::DeviceDesc* desc) {
    vk::Instance instance;
    @pool() {
        vk::ApplicationInfo app = vk::application_info()
            .set_application_name(desc.application_name == null ? (char*)"gpu.c3l" : (char*)desc.application_name)
            .set_api_version(API_VERSION_1_3);

        vk::InstanceCreateInfo create_info = vk::instance_create_info().set_application_info(&app);

        ZString[1] layers     = { VALIDATION_LAYER };
        ZString[1] extensions = { DEBUG_UTILS_EXTENSION };
        if (desc.enable_validation) {
            create_info = create_info
                .set_enabled_layer_names(layers[..])
                .set_enabled_extension_names(extensions[..]);
        }

        check(vk::create_instance(&create_info, null, &instance))!;
    };
    return instance;
}
