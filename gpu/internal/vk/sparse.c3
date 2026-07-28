module gpu::internal::vk @private;

import gpu;
import vk;

struct SparseTextureFeaturePlan {
    bool enabled;
    bool image_2d;
    bool image_3d;
}

fn SparseTextureFeaturePlan? resolve_sparse_texture_feature_plan(
    bool requested,
    vk::PhysicalDeviceFeatures* features,
) {
    if (!requested) return {};
    if (features == null
        || features.sparse_binding == 0
        || (features.sparse_residency_image2d == 0
            && features.sparse_residency_image3d == 0)) {
        return gpu::UNSUPPORTED_FEATURE~;
    }
    return {
        .enabled  = true,
        .image_2d = features.sparse_residency_image2d != 0,
        .image_3d = features.sparse_residency_image3d != 0,
    };
}

fn bool queue_family_supports_sparse_binding(
    vk::QueueFamilyProperties family,
) {
    return ((uint)family.queue_flags
        & (uint)vk::QueueFlagBits.SPARSE_BINDING_BIT) != 0;
}

fn gpu::QueueRoles sparse_binding_queue_roles(
    QueueFamilies* selection,
    vk::QueueFamilyProperties[] families,
) {
    gpu::QueueRoles roles;
    if (selection.selected.graphics
        && selection.graphics < families.len
        && queue_family_supports_sparse_binding(families[selection.graphics])) {
        roles.graphics = true;
    }
    if (selection.selected.compute
        && selection.compute < families.len
        && queue_family_supports_sparse_binding(families[selection.compute])) {
        roles.compute = true;
    }
    if (selection.selected.transfer
        && selection.transfer < families.len
        && queue_family_supports_sparse_binding(families[selection.transfer])) {
        roles.transfer = true;
    }
    return roles;
}

fn bool? vk_adapter_supports_sparse_textures(
    VkRuntimeState* state,
    uint adapter_index,
) {
    if (state == null || adapter_index >= state.adapters.len) {
        return gpu::INVALID_HANDLE~;
    }
    vk::PhysicalDeviceFeatures2 features2 =
        vk::physical_device_features2();
    vk::get_physical_device_features2(
        state.adapters[adapter_index].physical_device,
        &features2,
    );
    SparseTextureFeaturePlan? plan = resolve_sparse_texture_feature_plan(
        true,
        &features2.features,
    );
    if (catch plan_fault = plan) {
        if (plan_fault == gpu::UNSUPPORTED_FEATURE) return false;
        return plan_fault~;
    }
    return true;
}
