module gpu::internal::vk @private;

import gpu @public;
import vk;
import std::core::mem;

fn bool queue_family_supports_timestamp_workflow(
    vk::QueueFamilyProperties* properties,
) {
    if (properties.timestamp_valid_bits == 0) return false;
    uint executable_flags = (uint)vk::QueueFlagBits.GRAPHICS_BIT
        | (uint)vk::QueueFlagBits.COMPUTE_BIT;
    return ((uint)properties.queue_flags & executable_flags) != 0;
}

fn void add_timestamp_role(
    gpu::TimestampCaps* caps,
    gpu::QueueKind role,
    vk::QueueFamilyProperties* properties,
) {
    if (!queue_family_supports_timestamp_workflow(properties)) return;
    switch (role) {
        case gpu::QueueKind.GRAPHICS:
            caps.queues.graphics = true;
            caps.graphics_valid_bits = properties.timestamp_valid_bits;
        case gpu::QueueKind.COMPUTE:
            caps.queues.compute = true;
            caps.compute_valid_bits = properties.timestamp_valid_bits;
        case gpu::QueueKind.TRANSFER:
            caps.queues.transfer = true;
            caps.transfer_valid_bits = properties.timestamp_valid_bits;
    }
}

fn gpu::TimestampCaps resolve_timestamp_caps(
    QueueFamilies* selection,
    vk::QueueFamilyProperties[] properties,
    float timestamp_period,
) {
    gpu::TimestampCaps caps;
    if (selection.selected.graphics) {
        add_timestamp_role(
            caps:       &caps,
            role:       gpu::QueueKind.GRAPHICS,
            properties: &properties[selection.graphics],
        );
    }
    if (selection.selected.compute) {
        add_timestamp_role(
            caps:       &caps,
            role:       gpu::QueueKind.COMPUTE,
            properties: &properties[selection.compute],
        );
    }
    if (selection.selected.transfer) {
        add_timestamp_role(
            caps:       &caps,
            role:       gpu::QueueKind.TRANSFER,
            properties: &properties[selection.transfer],
        );
    }
    if ((uint)caps.queues != 0) caps.period_ns = timestamp_period;
    return caps;
}

fn gpu::TimestampCaps query_timestamp_caps(
    vk::PhysicalDevice physical_device,
    QueueFamilies* selection,
) {
    uint count;
    vk::get_physical_device_queue_family_properties(
        physical_device,
        &count,
        null,
    );
    if (count == 0) return {};

    vk::PhysicalDeviceProperties device_properties;
    vk::get_physical_device_properties(
        physical_device,
        &device_properties,
    );
    @pool() {
        vk::QueueFamilyProperties[] properties =
            mem::talloc_array(vk::QueueFamilyProperties, (sz)count);
        vk::get_physical_device_queue_family_properties(
            physical_device,
            &count,
            properties.ptr,
        );
        return resolve_timestamp_caps(
            selection,
            properties[:count],
            device_properties.limits.timestamp_period,
        );
    };
}
