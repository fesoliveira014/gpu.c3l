module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::io;
import std::core::mem;

// Heap binding convention (set 0), mirrored by include/shaders/descriptor_heap.glsl.
const uint HEAP_SET = 0;
const uint HEAP_BINDING_SAMPLED_2D = 0;
const uint HEAP_BINDING_STORAGE_2D = 1;
const uint HEAP_BINDING_SAMPLERS = 2;
const uint HEAP_BINDING_SAMPLED_3D = 3;
const uint HEAP_BINDING_STORAGE_3D = 4;
const uint HEAP_BINDING_ACCELERATION_STRUCTURES = 5;
const uint HEAP_BASE_BINDING_COUNT = 5;
const uint HEAP_BINDING_COUNT = 6;
const uint HEAP_BASE_POOL_SIZE_COUNT = 3;
const uint HEAP_POOL_SIZE_COUNT = 4;
const uint TEXTURE_DESCRIPTOR_BATCH_WRITE_STACK_CAP = 64;

enum TextureDescriptorKind : char {
    SAMPLED,
    STORAGE,
}

// Generations bump at destroy (stale indices reject immediately) and at
// alloc (a live index is never generation 0).
struct TextureDescriptorCell {
    gpu::TextureHandle owner;
    uint               generation;
    bool               used;
}

struct AccelerationStructureDescriptorCell {
    gpu::AccelerationStructureHandle owner;
    uint                             generation;
    bool                             used;
}

alias AccelerationStructureDescriptorWriteFn = fn void (
    void* user_data,
    VkDeviceState* state,
    uint slot,
    vk::AccelerationStructureKHR acceleration_structure,
);

struct AccelerationStructureDescriptorOps {
    void*                                  user_data;
    AccelerationStructureDescriptorWriteFn write;
}

struct DescriptorHeapState {
    ulong                   texture_view_owner;
    TextureDescriptorCell[] texture_cells;
    uint                    texture_next_new;
    uint                    sampler_next_new;
    uint[]                  texture_free;
    uint                    texture_free_count;
    ulong                   acceleration_structure_view_owner;
    AccelerationStructureDescriptorCell[] acceleration_structure_cells;
    uint[]                  acceleration_structure_free;
    uint                    acceleration_structure_next_new;
    uint                    acceleration_structure_free_count;
    vk::DescriptorSetLayout set_layout;
    vk::DescriptorPool      descriptor_pool;
    vk::DescriptorSet       descriptor_set;
}

struct PendingTextureWrite {
    TextureSlot*  texture_slot;
    vk::ImageView view;
}

struct DescriptorHeapLayoutPlan {
    vk::DescriptorSetLayoutBinding[HEAP_BINDING_COUNT] bindings;
    vk::DescriptorBindingFlags[HEAP_BINDING_COUNT]     binding_flags;
    uint                                               count;
}

struct DescriptorHeapPoolPlan {
    vk::DescriptorPoolSize[HEAP_POOL_SIZE_COUNT] sizes;
    uint                                         count;
}

fn DescriptorHeapLayoutPlan descriptor_heap_layout_plan(
    uint tex_capacity,
    uint smp_capacity,
    uint acceleration_structure_capacity = 0,
) @private {
    uint binding_flag_bits =
        (uint)vk::DescriptorBindingFlagBits.UPDATE_AFTER_BIND_BIT
        | (uint)vk::DescriptorBindingFlagBits.UPDATE_UNUSED_WHILE_PENDING_BIT
        | (uint)vk::DescriptorBindingFlagBits.PARTIALLY_BOUND_BIT;
    vk::DescriptorBindingFlags binding_flags =
        (vk::DescriptorBindingFlags)binding_flag_bits;
    DescriptorHeapLayoutPlan plan = {
        .bindings = {
            {
                .binding          = HEAP_BINDING_SAMPLED_2D,
                .descriptor_type  = vk::DescriptorType.SAMPLED_IMAGE,
                .descriptor_count = tex_capacity,
                .stage_flags      = vk::ShaderStageFlagBits.ALL,
            },
            {
                .binding          = HEAP_BINDING_STORAGE_2D,
                .descriptor_type  = vk::DescriptorType.STORAGE_IMAGE,
                .descriptor_count = tex_capacity,
                .stage_flags      = vk::ShaderStageFlagBits.ALL,
            },
            {
                .binding          = HEAP_BINDING_SAMPLERS,
                .descriptor_type  = vk::DescriptorType.SAMPLER,
                .descriptor_count = smp_capacity,
                .stage_flags      = vk::ShaderStageFlagBits.ALL,
            },
            {
                .binding          = HEAP_BINDING_SAMPLED_3D,
                .descriptor_type  = vk::DescriptorType.SAMPLED_IMAGE,
                .descriptor_count = tex_capacity,
                .stage_flags      = vk::ShaderStageFlagBits.ALL,
            },
            {
                .binding          = HEAP_BINDING_STORAGE_3D,
                .descriptor_type  = vk::DescriptorType.STORAGE_IMAGE,
                .descriptor_count = tex_capacity,
                .stage_flags      = vk::ShaderStageFlagBits.ALL,
            },
            {
                .binding = HEAP_BINDING_ACCELERATION_STRUCTURES,
                .descriptor_type =
                    vk::DescriptorType.ACCELERATION_STRUCTURE_KHR,
                .descriptor_count = acceleration_structure_capacity,
                .stage_flags = vk::ShaderStageFlagBits.ALL,
            },
        },
        .binding_flags = {
            binding_flags,
            binding_flags,
            binding_flags,
            binding_flags,
            binding_flags,
            binding_flags,
        },
        .count = acceleration_structure_capacity > 0
            ? HEAP_BINDING_COUNT
            : HEAP_BASE_BINDING_COUNT,
    };
    return plan;
}

fn DescriptorHeapPoolPlan descriptor_heap_pool_plan(
    uint tex_capacity,
    uint smp_capacity,
    uint acceleration_structure_capacity = 0,
) @private {
    return {
        .sizes = {
            {
                .type             = vk::DescriptorType.SAMPLED_IMAGE,
                .descriptor_count = tex_capacity * 2,
            },
            {
                .type             = vk::DescriptorType.STORAGE_IMAGE,
                .descriptor_count = tex_capacity * 2,
            },
            {
                .type             = vk::DescriptorType.SAMPLER,
                .descriptor_count = smp_capacity,
            },
            {
                .type = vk::DescriptorType.ACCELERATION_STRUCTURE_KHR,
                .descriptor_count = acceleration_structure_capacity,
            },
        },
        .count = acceleration_structure_capacity > 0
            ? HEAP_POOL_SIZE_COUNT
            : HEAP_BASE_POOL_SIZE_COUNT,
    };
}

fn void? report_descriptor_backend_failure(
    VkDeviceState* state,
    ZString operation,
    fault public_fault,
    ZString backend_text,
    int backend_result,
) {
    if (state.debug_callback != null) {
        gpu::DebugMessage message = {
            .severity             = gpu::DebugMessageSeverity.ERROR,
            .category             = { .backend },
            .operation            = operation,
            .has_fault            = true,
            .public_fault         = public_fault,
            .backend_text         = backend_text,
            .validation_id_number = backend_result,
        };
        gpu::internal::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
    }
    return public_fault~;
}

fn void? check_descriptor_pool_backend_result(
    VkDeviceState* state,
    ZString operation,
    vk::Result result,
) {
    if (result != vk::Result.ERROR_FRAGMENTATION) {
        return check_backend_result(state, operation, result);
    }
    return report_descriptor_backend_failure(
        state:          state,
        operation:      operation,
        public_fault:   gpu::DESCRIPTOR_HEAP_FULL,
        backend_text:   "Vulkan descriptor pool is fragmented",
        backend_result: (int)result,
    );
}

fn void? check_descriptor_allocate_backend_result(
    VkDeviceState* state,
    ZString operation,
    vk::Result result,
) {
    if (result != vk::Result.ERROR_OUT_OF_POOL_MEMORY
        && result != vk::Result.ERROR_FRAGMENTED_POOL) {
        return check_backend_result(state, operation, result);
    }
    return report_descriptor_backend_failure(
        state:          state,
        operation:      operation,
        public_fault:   gpu::DESCRIPTOR_HEAP_FULL,
        backend_text:   "Vulkan descriptor allocation exhausted the pool",
        backend_result: (int)result,
    );
}

fn vk::Result create_descriptor_heap_layout(
    vk::Device device,
    uint tex_capacity,
    uint smp_capacity,
    uint acceleration_structure_capacity,
    vk::DescriptorSetLayout* out_layout,
) @private {
    DescriptorHeapLayoutPlan plan = descriptor_heap_layout_plan(
        tex_capacity,
        smp_capacity,
        acceleration_structure_capacity,
    );
    vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info =
        vk::descriptor_set_layout_binding_flags_create_info()
            .set_binding_flags(plan.binding_flags[:plan.count]);

    vk::DescriptorSetLayoutCreateInfo layout_info =
        vk::descriptor_set_layout_create_info()
            .set_next(&flags_info)
            .set_flags(vk::DescriptorSetLayoutCreateFlagBits.UPDATE_AFTER_BIND_POOL_BIT)
            .set_bindings(plan.bindings[:plan.count]);
    return vk::create_descriptor_set_layout(device, &layout_info, null, out_layout);
}

fn void? create_descriptor_heap(VkDeviceState* state) {
    DescriptorHeapState* heap = &state.shader_heap.descriptors;
    uint tex_capacity = state.shader_heap.texture_capacity;
    uint smp_capacity = state.shader_heap.sampler_capacity;
    uint acceleration_structure_capacity = state.acceleration_structures_enabled
        ? state.acceleration_structure_heap_capacity
        : 0;

    heap.texture_view_owner     = gpu::internal::resource_owner_tag(
        state.resource_owner,
        gpu::internal::ResourceOwnerKind.TEXTURE_VIEW,
    );
    heap.texture_cells          = mem::new_array(TextureDescriptorCell, tex_capacity);
    heap.texture_free           = mem::new_array(uint, tex_capacity);
    if (state.acceleration_structures_enabled) {
        heap.acceleration_structure_view_owner =
            gpu::internal::resource_owner_tag(
                state.resource_owner,
                gpu::internal::ResourceOwnerKind.ACCELERATION_STRUCTURE_VIEW,
            );
        heap.acceleration_structure_cells = mem::new_array(
            AccelerationStructureDescriptorCell,
            acceleration_structure_capacity,
        );
        heap.acceleration_structure_free = mem::new_array(
            uint,
            acceleration_structure_capacity,
        );
    }

    check_backend_result(
        state:     state,
        operation: "create_device",
        result:    create_descriptor_heap_layout(
            device:       state.device,
            tex_capacity: tex_capacity,
            smp_capacity: smp_capacity,
            acceleration_structure_capacity:
                acceleration_structure_capacity,
            out_layout:   &heap.set_layout,
        ),
    )!;

    DescriptorHeapPoolPlan pool = descriptor_heap_pool_plan(
        tex_capacity,
        smp_capacity,
        acceleration_structure_capacity,
    );
    vk::DescriptorPoolCreateInfo pool_info = vk::descriptor_pool_create_info()
        .set_flags(vk::DescriptorPoolCreateFlagBits.UPDATE_AFTER_BIND_BIT)
        .set_max_sets(1)
        .set_pool_sizes(pool.sizes[:pool.count]);
    check_descriptor_pool_backend_result(
        state:     state,
        operation: "create_device",
        result:    vk::create_descriptor_pool(
            device:            state.device,
            p_create_info:     &pool_info,
            p_allocator:       null,
            p_descriptor_pool: &heap.descriptor_pool,
        ),
    )!;

    vk::DescriptorSetAllocateInfo alloc_info = vk::descriptor_set_allocate_info()
        .set_descriptor_pool(heap.descriptor_pool)
        .set_set_layouts((&heap.set_layout)[:1]);
    check_descriptor_allocate_backend_result(
        state:     state,
        operation: "create_device",
        result:    vk::allocate_descriptor_sets(
            device:            state.device,
            p_allocate_info:   &alloc_info,
            p_descriptor_sets: &heap.descriptor_set,
        ),
    )!;
}

<* Safe on a partial or zeroed heap. *>
fn void destroy_descriptor_heap(VkDeviceState* state) {
    DescriptorHeapState* heap = &state.shader_heap.descriptors;
    if (heap.descriptor_pool != null) {
        vk::destroy_descriptor_pool(state.device, heap.descriptor_pool, null);
        heap.descriptor_pool = (vk::DescriptorPool)null;
    }
    heap.descriptor_set = (vk::DescriptorSet)null;
    if (heap.set_layout != null) {
        vk::destroy_descriptor_set_layout(state.device, heap.set_layout, null);
        heap.set_layout = (vk::DescriptorSetLayout)null;
    }
    if (heap.texture_cells.ptr != null)          { mem::free(heap.texture_cells.ptr); heap.texture_cells = {}; }
    if (heap.texture_free.ptr != null)           { mem::free(heap.texture_free.ptr); heap.texture_free = {}; }
    if (heap.acceleration_structure_cells.ptr != null) {
        mem::free(heap.acceleration_structure_cells.ptr);
        heap.acceleration_structure_cells = {};
    }
    if (heap.acceleration_structure_free.ptr != null) {
        mem::free(heap.acceleration_structure_free.ptr);
        heap.acceleration_structure_free = {};
    }
    heap.acceleration_structure_view_owner = 0;
}

fn void report_descriptor_leaks(VkDeviceState* state) {
    DescriptorHeapState* heap = &state.shader_heap.descriptors;
    if (heap.texture_cells.ptr != null) {
        foreach (i, &cell : heap.texture_cells[:heap.texture_next_new]) {
            if (!cell.used) continue;
            if (!report_resource_lifetime(
                state:               state,
                resource_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
                has_public_identity: true,
                index:               (uint)i,
                generation:          (uint)cell.generation,
                debug_name:          null,
                backend_text:        "texture descriptor remains live",
            )) {
                io::eprintfn("gpu: leaked texture descriptor slot %d (owner texture index %d)", i, cell.owner.index);
            }
        }
    }
    if (heap.acceleration_structure_cells.ptr != null) {
        foreach (i, &cell : heap.acceleration_structure_cells[
            :heap.acceleration_structure_next_new
        ]) {
            if (!cell.used) continue;
            if (!report_resource_lifetime(
                state:               state,
                resource_kind:       gpu::DebugResourceKind.ACCELERATION_STRUCTURE_VIEW,
                has_public_identity: true,
                index:               (uint)i,
                generation:          (uint)cell.generation,
                debug_name:          null,
                backend_text:        "acceleration-structure descriptor remains live",
            )) {
                io::eprintfn(
                    "gpu: leaked acceleration-structure descriptor slot %d (owner acceleration-structure index %d)",
                    i,
                    cell.owner.index,
                );
            }
        }
    }
}

fn uint texture_index_slot(gpu::TextureIndex index) @private {
    return index.value - 1;
}

fn gpu::TextureIndex texture_index_from_slot(uint slot) @private {
    return { .value = slot + 1 };
}

fn uint acceleration_structure_index_slot(
    gpu::AccelerationStructureIndex index,
) @private {
    return index.value - 1;
}

fn gpu::AccelerationStructureIndex acceleration_structure_index_from_slot(
    uint slot,
) @private {
    return { .value = slot + 1 };
}

fn AccelerationStructureDescriptorCell*?
acceleration_structure_descriptor_cell_get(
    VkDeviceState* state,
    gpu::AccelerationStructureView view,
) @private {
    DescriptorHeapState* heap = &state.shader_heap.descriptors;
    if (view.owner != heap.acceleration_structure_view_owner
        || !view.index.is_valid()) return gpu::INVALID_HANDLE~;
    uint slot = acceleration_structure_index_slot(view.index);
    if (slot >= heap.acceleration_structure_cells.len
        || !heap.acceleration_structure_cells[slot].used
        || heap.acceleration_structure_cells[slot].generation
            != view.generation) return gpu::INVALID_HANDLE~;
    return &heap.acceleration_structure_cells[slot];
}

fn void write_acceleration_structure_descriptor_real(
    void* user_data,
    VkDeviceState* state,
    uint slot,
    vk::AccelerationStructureKHR acceleration_structure,
) {
    (void)user_data;
    DescriptorHeapState* heap = &state.shader_heap.descriptors;
    vk::WriteDescriptorSetAccelerationStructureKHR acceleration_info =
        vk::write_descriptor_set_acceleration_structure_khr()
            .set_acceleration_structures((&acceleration_structure)[:1]);
    vk::WriteDescriptorSet write = vk::write_descriptor_set()
        .set_next(&acceleration_info)
        .set_dst_set(heap.descriptor_set)
        .set_dst_binding(HEAP_BINDING_ACCELERATION_STRUCTURES)
        .set_dst_array_element(slot)
        .set_descriptor_count(1)
        .set_descriptor_type(
            vk::DescriptorType.ACCELERATION_STRUCTURE_KHR,
        );
    vk::update_descriptor_sets(state.device, 1, &write, 0, null);
}

fn gpu::AccelerationStructureView?
create_acceleration_structure_descriptor_view_with_ops(
    VkDeviceState* state,
    gpu::AccelerationStructureHandle acceleration_structure,
    AccelerationStructureDescriptorOps* ops,
) {
    if (state == null || ops == null || ops.write == null) {
        return gpu::INVALID_ARGUMENT~;
    }
    if (!state.acceleration_structures_enabled) return gpu::UNSUPPORTED_FEATURE~;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();

    AccelerationStructureSlot* owner =
        state.acceleration_structures.get(acceleration_structure)!;
    if (owner.desc.kind != gpu::AccelerationStructureKind.TOP_LEVEL) {
        return gpu::INVALID_ARGUMENT~;
    }
    DescriptorHeapState* heap = &state.shader_heap.descriptors;
    uint headroom = heap.acceleration_structure_free_count
        + (uint)heap.acceleration_structure_cells.len
        - heap.acceleration_structure_next_new;
    if (headroom == 0) return gpu::DESCRIPTOR_HEAP_FULL~;
    if (owner.live_view_count
        >= (uint)heap.acceleration_structure_cells.len) {
        return gpu::BACKEND_ERROR~;
    }

    bool recycled = heap.acceleration_structure_free_count > 0;
    uint slot = recycled
        ? heap.acceleration_structure_free[
            heap.acceleration_structure_free_count - 1
        ]
        : heap.acceleration_structure_next_new;
    AccelerationStructureDescriptorCell* cell =
        &heap.acceleration_structure_cells[slot];
    uint generation = next_handle_generation(cell.generation)!;

    ops.write(ops.user_data, state, slot, owner.native);
    if (recycled) {
        heap.acceleration_structure_free_count--;
    } else {
        heap.acceleration_structure_next_new++;
    }
    cell.owner = acceleration_structure;
    cell.generation = generation;
    cell.used = true;
    owner.live_view_count++;
    return {
        .owner = heap.acceleration_structure_view_owner,
        .index = acceleration_structure_index_from_slot(slot),
        .generation = generation,
    };
}

fn void? destroy_acceleration_structure_descriptor_view(
    VkDeviceState* state,
    gpu::AccelerationStructureView view,
) {
    if (state == null) return gpu::INVALID_ARGUMENT~;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    DescriptorHeapState* heap = &state.shader_heap.descriptors;
    AccelerationStructureDescriptorCell* cell =
        acceleration_structure_descriptor_cell_get(state, view)!;
    AccelerationStructureSlot* owner =
        state.acceleration_structures.get(cell.owner)!;
    if (owner.live_view_count == 0) return gpu::BACKEND_ERROR~;

    cell.used = false;
    owner.live_view_count--;
    if (!should_recycle_handle_generation(cell.generation)) {
        cell.generation = HANDLE_GENERATION_MAX;
        return;
    }
    cell.generation = next_handle_generation(cell.generation)!;
    heap.acceleration_structure_free[
        heap.acceleration_structure_free_count++
    ] = acceleration_structure_index_slot(view.index);
}

fn vk::ImageLayout texture_descriptor_layout(
    TextureSlot* texture,
    TextureDescriptorKind kind,
) {
    return texture_layout_to_vk(
        kind == TextureDescriptorKind.STORAGE
            ? gpu::TextureLayout.STORAGE
            : gpu::TextureLayout.SAMPLED,
        texture.format,
    );
}

fn uint sampled_texture_binding(TextureSlot* texture) @private {
    return texture_slot_is_3d(texture)
        ? HEAP_BINDING_SAMPLED_3D
        : HEAP_BINDING_SAMPLED_2D;
}

fn uint storage_texture_binding(TextureSlot* texture) @private {
    return texture_slot_is_3d(texture)
        ? HEAP_BINDING_STORAGE_3D
        : HEAP_BINDING_STORAGE_2D;
}

fn TextureDescriptorCell*? texture_cell_get(
    VkDeviceState* state,
    gpu::TextureView view,
) @private {
    DescriptorHeapState* heap = &state.shader_heap.descriptors;
    if (!view.index.is_valid()) return gpu::INVALID_HANDLE~;
    uint slot = texture_index_slot(view.index);
    if (slot >= heap.texture_cells.len || !heap.texture_cells[slot].used) {
        return gpu::INVALID_HANDLE~;
    }
    if (heap.texture_cells[slot].generation != view.generation) {
        return gpu::INVALID_HANDLE~;
    }
    return &heap.texture_cells[slot];
}

fn bool texture_has_live_descriptor_views(
    TextureSlot* texture,
) @private {
    return texture.live_descriptor_views != 0;
}

fn void? report_texture_view_owner_invariant_failure(
    VkDeviceState* state,
    ZString operation,
    ZString backend_text,
) @private {
    return report_descriptor_backend_failure(
        state:          state,
        operation:      operation,
        public_fault:   gpu::BACKEND_ERROR,
        backend_text:   backend_text,
        backend_result: 0,
    );
}

fn void? validate_texture_view_increment(
    VkDeviceState* state,
    TextureSlot* owner,
    uint pending_count,
    ZString operation,
) @private {
    if (!contract_uses_full_validation(state.validation_policy)) return;
    uint capacity = (uint)state.shader_heap.descriptors.texture_cells.len;
    if (pending_count > capacity
        || owner.live_descriptor_views > capacity - pending_count) {
        return report_texture_view_owner_invariant_failure(
            state,
            operation,
            "texture view owner count exceeds descriptor heap capacity",
        );
    }
}

fn void? validate_texture_view_decrement(
    VkDeviceState* state,
    TextureSlot* owner,
) @private {
    if (!contract_uses_full_validation(state.validation_policy)
        || owner.live_descriptor_views != 0) return;
    return report_texture_view_owner_invariant_failure(
        state,
        "destroy_texture_view",
        "texture view owner count is zero for a live descriptor",
    );
}

fn void recycle_texture_descriptor_slot(
    DescriptorHeapState* heap,
    gpu::TextureView view,
    TextureDescriptorCell* cell,
) @private {
    if (!should_recycle_handle_generation(cell.generation)) {
        cell.generation = HANDLE_GENERATION_MAX;
        return;
    }
    cell.generation = next_handle_generation(cell.generation)!!;
    heap.texture_free[heap.texture_free_count++] =
        texture_index_slot(view.index);
}

<* Sampled and/or storage descriptor per the texture's usage, same slot in
   both bindings, so one TextureIndex serves both access kinds. *>
fn void write_texture_descriptor(VkDeviceState* state, uint slot, TextureSlot* texture, vk::ImageView view) @private {
    DescriptorHeapState* heap = &state.shader_heap.descriptors;
    vk::WriteDescriptorSet[2] writes;
    vk::DescriptorImageInfo[2] image_infos;
    uint write_count = 0;
    if (texture.usage.sampled) {
        image_infos[write_count] = {
            .image_view   = view,
            .image_layout = texture_descriptor_layout(
                texture,
                TextureDescriptorKind.SAMPLED,
            ),
        };
        writes[write_count] = vk::write_descriptor_set()
            .set_dst_set(heap.descriptor_set)
            .set_dst_binding(sampled_texture_binding(texture))
            .set_dst_array_element(slot)
            .set_descriptor_type(vk::DescriptorType.SAMPLED_IMAGE)
            .set_image_info((&image_infos[write_count])[:1]);
        write_count++;
    }
    if (texture.usage.storage) {
        image_infos[write_count] = {
            .image_view   = view,
            .image_layout = texture_descriptor_layout(
                texture,
                TextureDescriptorKind.STORAGE,
            ),
        };
        writes[write_count] = vk::write_descriptor_set()
            .set_dst_set(heap.descriptor_set)
            .set_dst_binding(storage_texture_binding(texture))
            .set_dst_array_element(slot)
            .set_descriptor_type(vk::DescriptorType.STORAGE_IMAGE)
            .set_image_info((&image_infos[write_count])[:1]);
        write_count++;
    }
    if (write_count > 0) {
        vk::update_descriptor_sets(state.device, write_count, &writes[0], 0, null);
    }
}

fn void write_sampler_descriptor(VkDeviceState* state, uint slot, vk::Sampler sampler) @private {
    DescriptorHeapState* heap = &state.shader_heap.descriptors;

    vk::DescriptorImageInfo image_info = { .sampler = sampler };
    vk::WriteDescriptorSet write = vk::write_descriptor_set()
        .set_dst_set(heap.descriptor_set)
        .set_dst_binding(HEAP_BINDING_SAMPLERS)
        .set_dst_array_element(slot)
        .set_descriptor_type(vk::DescriptorType.SAMPLER)
        .set_image_info((&image_info)[:1]);
    vk::update_descriptor_sets(state.device, 1, &write, 0, null);
}

fn void? validate_descriptor_heap_binding(VkDeviceState* state) {
    if (state == null) return gpu::INVALID_ARGUMENT~;
}

fn void? bind_descriptor_heap(
    VkDeviceState* state,
    CommandRecord* record,
    vk::PipelineLayout layout,
    vk::PipelineBindPoint bind_point,
) {
    validate_descriptor_heap_binding(state)!;
    DescriptorHeapCommandBinding* binding = &record.descriptor_heap_binding;
    bool* set_bound;
    switch (bind_point) {
        case vk::PipelineBindPoint.COMPUTE:
            set_bound = &binding.compute_set_bound;
        case vk::PipelineBindPoint.GRAPHICS:
            set_bound = &binding.graphics_set_bound;
        case vk::PipelineBindPoint.RAY_TRACING_KHR:
            set_bound = &binding.ray_tracing_set_bound;
        default:
            return gpu::INVALID_ARGUMENT~;
    }
    if (*set_bound) return;
    vk::cmd_bind_descriptor_sets(
        command_buffer:        record.command_buffer,
        pipeline_bind_point:   bind_point,
        layout:                layout,
        first_set:             HEAP_SET,
        descriptor_set_count:  1,
        p_descriptor_sets:     &state.shader_heap.descriptors.descriptor_set,
        dynamic_offset_count:  0,
        p_dynamic_offsets:     null,
    );
    *set_bound = true;
}

fn TextureSlot*? resolve_descriptor_texture(
    VkDeviceState* state,
    gpu::TextureHandle texture,
    ZString operation,
    ZString texture_field,
    ZString usage_field,
) @private {
    TextureSlot*? resolved = state.textures.get(texture);
    if (catch f = resolved) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        f,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      texture_field,
            invariant:           "texture must identify a live generation",
            backend_text:        null,
        )~;
    }
    TextureSlot* slot = resolved;
    if (!slot.usage.sampled && !slot.usage.storage) {
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               texture.index,
            generation:          texture.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      usage_field,
            invariant:           "texture must have sampled or storage usage",
            backend_text:        null,
        )~;
    }
    return slot;
}

fn vk::ImageView? resolve_descriptor_texture_view(
    VkDeviceState* state,
    TextureSlot* slot,
    gpu::TextureHandle texture,
    gpu::TextureViewDesc* view,
    ZString operation,
    ZString view_field,
    CachedViewCreation* creation,
) @private {
    vk::ImageView? resolved = resolve_texture_view_tracked(
        state:     state,
        slot:      slot,
        texture:   texture,
        view:      view,
        operation: operation,
        creation:  creation,
    );
    if (catch f = resolved) {
        if (f != gpu::INVALID_ARGUMENT) return f~;
        return report_contract_failure(
            state:               state,
            operation:           operation,
            public_fault:        f,
            resource_kind:       gpu::DebugResourceKind.TEXTURE,
            has_public_identity: true,
            index:               texture.index,
            generation:          texture.generation,
            debug_name:          slot_name(&slot.debug_name[0]),
            rejected_field:      view_field,
            invariant:           "view must select a valid texture subresource range",
            backend_text:        null,
        )~;
    }
    return resolved;
}

fn gpu::TextureView? vk_create_texture_view(
    VkDeviceState* state,
    gpu::TextureHandle texture,
    gpu::TextureViewDesc* view,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    state.texture_view_cache_mutex.lock()!!;
    defer state.texture_view_cache_mutex.unlock();
    DescriptorHeapState* heap = &state.shader_heap.descriptors;
    TextureSlot* texture_slot = resolve_descriptor_texture(
        state:         state,
        texture:       texture,
        operation:     "create_texture_view",
        texture_field: "texture",
        usage_field:   "TextureDesc.usage",
    )!;

    uint headroom = heap.texture_free_count
        + (uint)heap.texture_cells.len - heap.texture_next_new;
    if (headroom == 0) {
        return report_contract_failure(
            state:               state,
            operation:           "create_texture_view",
            public_fault:        gpu::DESCRIPTOR_HEAP_FULL,
            resource_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "RuntimeDesc.texture_heap_capacity",
            invariant:           "must exceed live texture descriptors",
            backend_text:        null,
        )~;
    }
    validate_texture_view_increment(
        state:         state,
        owner:         texture_slot,
        pending_count: 1,
        operation:     "create_texture_view",
    )!;

    CachedViewCreation creation;
    vk::ImageView image_view = resolve_descriptor_texture_view(
        state:     state,
        slot:      texture_slot,
        texture:   texture,
        view:      view,
        operation: "create_texture_view",
        view_field: "view",
        creation:  &creation,
    )!;

    gpu::TextureView result;
    commit_texture_descriptor_item(
        heap:         heap,
        texture:      texture,
        texture_slot: texture_slot,
        out_view:     &result,
    );
    write_texture_descriptor(
        state,
        texture_index_slot(result.index),
        texture_slot,
        image_view,
    );
    commit_texture_view_creations(state, (&creation)[:1]);
    return result;
}

<* Validate and resolve one batch item without touching descriptor allocation
   or output state. The caller owns any returned cache creation until commit. *>
fn void? prepare_texture_descriptor_item(
    VkDeviceState* state,
    gpu::TextureViewCreateDesc* desc,
    uint pending_count,
    PendingTextureWrite* pending,
    CachedViewCreation* creation,
) @private {
    TextureSlot* texture_slot = resolve_descriptor_texture(
        state:         state,
        texture:       desc.texture,
        operation:     "create_texture_views",
        texture_field: "descs[].texture",
        usage_field:   "TextureDesc.usage",
    )!;
    validate_texture_view_increment(
        state:         state,
        owner:         texture_slot,
        pending_count: pending_count,
        operation:     "create_texture_views",
    )!;
    vk::ImageView image_view = resolve_descriptor_texture_view(
        state:      state,
        slot:       texture_slot,
        texture:    desc.texture,
        view:       &desc.view,
        operation:  "create_texture_views",
        view_field: "descs[].view",
        creation:   creation,
    )!;
    *pending = { .texture_slot = texture_slot, .view = image_view };
}

<*
 @require heap.texture_free_count > 0 || heap.texture_next_new < heap.texture_cells.len : "descriptor headroom"
*>
fn void commit_texture_descriptor_item(
    DescriptorHeapState* heap,
    gpu::TextureHandle texture,
    TextureSlot* texture_slot,
    gpu::TextureView* out_view,
) @private {
    uint slot;
    if (heap.texture_free_count > 0) {
        slot = heap.texture_free[--heap.texture_free_count];
    } else {
        slot = heap.texture_next_new++;
    }

    TextureDescriptorCell* cell = &heap.texture_cells[slot];
    cell.owner = texture;
    cell.used = true;
    cell.generation = next_handle_generation(cell.generation)!!;
    texture_slot.live_descriptor_views++;
    *out_view = {
        .owner      = heap.texture_view_owner,
        .index      = texture_index_from_slot(slot),
        .generation = cell.generation,
    };
}

fn uint accumulate_texture_descriptor_write(
    DescriptorHeapState* heap,
    uint slot,
    TextureSlot* texture_slot,
    vk::ImageView view,
    vk::DescriptorImageInfo[] image_infos,
    vk::WriteDescriptorSet[] writes,
    uint write_count,
) {
    if (texture_slot.usage.sampled) {
        image_infos[write_count] = {
            .image_view   = view,
            .image_layout = texture_descriptor_layout(
                texture_slot,
                TextureDescriptorKind.SAMPLED,
            ),
        };
        writes[write_count] = vk::write_descriptor_set()
            .set_dst_set(heap.descriptor_set)
            .set_dst_binding(sampled_texture_binding(texture_slot))
            .set_dst_array_element(slot)
            .set_descriptor_type(vk::DescriptorType.SAMPLED_IMAGE)
            .set_image_info((&image_infos[write_count])[:1]);
        write_count++;
    }
    if (texture_slot.usage.storage) {
        image_infos[write_count] = {
            .image_view   = view,
            .image_layout = texture_descriptor_layout(
                texture_slot,
                TextureDescriptorKind.STORAGE,
            ),
        };
        writes[write_count] = vk::write_descriptor_set()
            .set_dst_set(heap.descriptor_set)
            .set_dst_binding(storage_texture_binding(texture_slot))
            .set_dst_array_element(slot)
            .set_descriptor_type(vk::DescriptorType.STORAGE_IMAGE)
            .set_image_info((&image_infos[write_count])[:1]);
        write_count++;
    }
    return write_count;
}

<* All validation and fallible view creation precedes descriptor mutation.
   A preparation fault restores only views created by this transaction. *>
fn void? vk_create_texture_views(
    VkDeviceState* state,
    gpu::TextureViewCreateDesc[] descs,
    gpu::TextureView[] out_views,
) {
    if (out_views.len != descs.len) {
        return report_contract_failure(
            state:               state,
            operation:           "create_texture_views",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "out_views",
            invariant:           "output count must match descriptor count",
            backend_text:        null,
        )~;
    }
    if (descs.len == 0) return;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    state.texture_view_cache_mutex.lock()!!;
    defer state.texture_view_cache_mutex.unlock();
    DescriptorHeapState* heap = &state.shader_heap.descriptors;

    uint headroom = heap.texture_free_count
        + (uint)heap.texture_cells.len - heap.texture_next_new;
    if ((uint)descs.len > headroom) {
        return report_contract_failure(
            state:               state,
            operation:           "create_texture_views",
            public_fault:        gpu::DESCRIPTOR_HEAP_FULL,
            resource_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "RuntimeDesc.texture_heap_capacity",
            invariant:           "must fit the complete descriptor batch",
            backend_text:        null,
        )~;
    }

    @pool() {
        PendingTextureWrite[] pending = mem::talloc_array(PendingTextureWrite, descs.len);
        CachedViewCreation[] creations = mem::talloc_array(CachedViewCreation, descs.len);
        uint creation_count = 0;
        for (uint i = 0; i < descs.len; i++) {
            CachedViewCreation creation;
            if (catch f = prepare_texture_descriptor_item(
                state:         state,
                desc:          &descs[i],
                pending_count: (uint)descs.len,
                pending:       &pending[i],
                creation:      &creation,
            )) {
                rollback_texture_view_creations(state, creations[:creation_count]);
                return f~;
            }
            if (creation.cache != null) creations[creation_count++] = creation;
        }

        for (uint i = 0; i < descs.len; i++) {
            commit_texture_descriptor_item(
                heap:         heap,
                texture:      descs[i].texture,
                texture_slot: pending[i].texture_slot,
                out_view:     &out_views[i],
            );
        }

        uint max_writes = (uint)descs.len * 2;
        vk::WriteDescriptorSet[TEXTURE_DESCRIPTOR_BATCH_WRITE_STACK_CAP] writes_stack;
        vk::DescriptorImageInfo[TEXTURE_DESCRIPTOR_BATCH_WRITE_STACK_CAP] image_infos_stack;
        vk::WriteDescriptorSet[] writes = max_writes <= TEXTURE_DESCRIPTOR_BATCH_WRITE_STACK_CAP
            ? writes_stack[:max_writes]
            : mem::talloc_array(vk::WriteDescriptorSet, max_writes);
        vk::DescriptorImageInfo[] image_infos = max_writes <= TEXTURE_DESCRIPTOR_BATCH_WRITE_STACK_CAP
            ? image_infos_stack[:max_writes]
            : mem::talloc_array(vk::DescriptorImageInfo, max_writes);

        uint write_count = 0;
        foreach (i, &item : pending) {
            write_count = accumulate_texture_descriptor_write(
                heap:              heap,
                slot:              texture_index_slot(out_views[i].index),
                texture_slot:      item.texture_slot,
                view:              item.view,
                image_infos:       image_infos,
                writes:            writes,
                write_count:       write_count,
            );
        }
        if (write_count > 0) vk::update_descriptor_sets(state.device, write_count, &writes[0], 0, null);
        commit_texture_view_creations(state, creations[:creation_count]);
    };
}

fn void? vk_destroy_texture_view(
    VkDeviceState* state,
    gpu::TextureView view,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    TextureDescriptorCell*? resolved = texture_cell_get(state, view);
    if (catch f = resolved) {
        return report_contract_failure(
            state:               state,
            operation:           "destroy_texture_view",
            public_fault:        f,
            resource_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "view",
            invariant:           "view must identify a live generation",
            backend_text:        null,
        )~;
    }
    TextureDescriptorCell* cell = resolved;
    TextureSlot*? resolved_owner = state.textures.get(cell.owner);
    if (catch resolved_owner) {
        return report_texture_view_owner_invariant_failure(
            state,
            "destroy_texture_view",
            "live texture descriptor owner is missing",
        );
    }
    TextureSlot* owner = resolved_owner;
    validate_texture_view_decrement(state, owner)!;
    cell.used = false;
    owner.live_descriptor_views--;
    recycle_texture_descriptor_slot(
        &state.shader_heap.descriptors,
        view,
        cell,
    );
}
