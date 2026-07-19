module gpu::vk @private;

import gpu @public;
import vk;
import std::thread;
import vma;
import std::io;
import std::core::mem;

// Heap binding convention (set 0), mirrored by include/shaders/descriptor_heap.glsl.
const uint HEAP_SET = 0;
const uint HEAP_BINDING_SAMPLED = 0;
const uint HEAP_BINDING_STORAGE = 1;
const uint HEAP_BINDING_SAMPLERS = 2;

// Generations bump at destroy (stale indices reject immediately) and at
// alloc (a live index is never generation 0).

struct TextureDescriptorCell {
    gpu::TextureHandle owner;
    ushort             generation;
    bool               used;
}

// Device limits governing the active heap mode. Indexing fields come from
// PhysicalDeviceVulkan12Properties; buffer fields come from
// PhysicalDeviceDescriptorBufferPropertiesEXT plus the computed layout size.
struct HeapLimits {
    uint           max_sampled_images;
    uint           max_storage_images;
    uint           max_samplers;
    uint           max_per_stage_sampled_images;
    uint           max_per_stage_storage_images;
    uint           max_per_stage_samplers;
    uint           max_per_stage_resources;
    uint           max_all_pools_descriptors;
    vk::DeviceSize max_resource_descriptor_buffer_range;
    vk::DeviceSize max_sampler_descriptor_buffer_range;
    vk::DeviceSize resource_descriptor_address_space_size;
    vk::DeviceSize sampler_descriptor_address_space_size;
    vk::DeviceSize descriptor_buffer_address_space_size;
    vk::DeviceSize layout_size;
}

struct DescriptorHeapState {
    TextureDescriptorCell[] texture_cells;
    uint                    texture_next_new;
    uint                    sampler_next_new;
    uint[]                  texture_free;
    uint                    texture_free_count;
    vk::DescriptorSetLayout set_layout;
    // Indexing path.
    vk::DescriptorPool      descriptor_pool;
    vk::DescriptorSet       descriptor_set;
    // Buffer path: one host-visible buffer holds the whole set-0 layout.
    vk::Buffer              descriptor_buffer;
    vma::Allocation         descriptor_buffer_allocation;
    void*                   descriptor_buffer_cpu;
    vk::DeviceAddress       descriptor_buffer_gpu;
    vk::DeviceSize          binding_offset_sampled;
    vk::DeviceSize          binding_offset_storage;
    vk::DeviceSize          binding_offset_samplers;
    usz                     sampled_descriptor_size;
    usz                     storage_descriptor_size;
    usz                     sampler_descriptor_size;
}

const uint TEXTURE_DESCRIPTOR_BATCH_WRITE_STACK_CAP = 64;

struct PendingTextureWrite {
    TextureSlot*  texture_slot;
    vk::ImageView view;
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
        gpu::dispatch_debug_message(state.debug_callback, state.debug_user_data, &message);
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


fn ushort bump_generation(ushort generation) @private {
    uint next = ((uint)generation + 1) & 0xFFFF;
    return next == 0 ? (ushort)1 : (ushort)next;
}

<* Texture capacity binds both SAMPLED_IMAGE and STORAGE_IMAGE, so on the
   indexing path it must satisfy both image limits. On the buffer path the
   bound quantity is the descriptor set's byte layout, not slot counts. *>
fn bool heap_fits_limits(gpu::DescriptorHeapMode mode, uint tex_capacity, uint smp_capacity, HeapLimits limits) {
    if (mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
        IndexingHeapLimits indexing_limits = {
            .max_sampled_images           = limits.max_sampled_images,
            .max_storage_images           = limits.max_storage_images,
            .max_samplers                 = limits.max_samplers,
            .max_per_stage_sampled_images = limits.max_per_stage_sampled_images,
            .max_per_stage_storage_images = limits.max_per_stage_storage_images,
            .max_per_stage_samplers       = limits.max_per_stage_samplers,
            .max_per_stage_resources      = limits.max_per_stage_resources,
            .max_all_pools_descriptors    = limits.max_all_pools_descriptors,
        };
        return indexing_heap_fits_limits(
            tex_capacity,
            smp_capacity,
            indexing_limits,
        );
    }
    // One buffer carries both usage bits, so it counts against each per-type
    // address space as well as the combined budget.
    return limits.layout_size <= limits.max_resource_descriptor_buffer_range
        && limits.layout_size <= limits.max_sampler_descriptor_buffer_range
        && limits.layout_size <= limits.resource_descriptor_address_space_size
        && limits.layout_size <= limits.sampler_descriptor_address_space_size
        && limits.layout_size <= limits.descriptor_buffer_address_space_size;
}

fn void? create_descriptor_heap(VkDeviceState* state) {
    DescriptorHeapState* heap = &state.strict_heap.descriptors;
    uint tex_capacity = state.strict_heap.texture_capacity;
    uint smp_capacity = state.strict_heap.sampler_capacity;

    if (state.strict_heap.mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
        IndexingHeapLimits limits = query_indexing_heap_limits(state.physical_device);
        bool fits = indexing_heap_fits_limits(
            tex_capacity,
            smp_capacity,
            limits,
        );
        if (!fits) {
            if (state.debug_callback == null) {
                HeapDescriptorTotals totals = descriptor_heap_totals(tex_capacity, smp_capacity);
                io::eprintfn(
                    "gpu: descriptor capacities (tex %d, smp %d) exceed a device update-after-bind limit (set %d/%d/%d, per-stage %d/%d/%d, per-stage resources %d/%d, all-pools descriptors %d/%d)",
                    tex_capacity,
                    smp_capacity,
                    limits.max_sampled_images,
                    limits.max_storage_images,
                    limits.max_samplers,
                    limits.max_per_stage_sampled_images,
                    limits.max_per_stage_storage_images,
                    limits.max_per_stage_samplers,
                    totals.per_stage_resources,
                    limits.max_per_stage_resources,
                    totals.all_pools_descriptors,
                    limits.max_all_pools_descriptors,
                );
            }
            return report_contract_failure(
                state:               state,
                operation:           "create_device",
                public_fault:        gpu::INVALID_ARGUMENT,
                resource_kind:       gpu::DebugResourceKind.DEVICE,
                has_public_identity: false,
                index:               0,
                generation:          0,
                debug_name:          null,
                rejected_field:      "DeviceDesc.texture_descriptor_capacity/sampler_descriptor_capacity",
                invariant:           "descriptor capacities must fit device update-after-bind limits",
                backend_text:        null,
            )~;
        }
    }

    heap.texture_cells          = mem::new_array(TextureDescriptorCell, tex_capacity);
    heap.texture_free           = mem::new_array(uint, tex_capacity);

    vk::DescriptorSetLayoutBinding[3] bindings = {
        {
            .binding          = HEAP_BINDING_SAMPLED,
            .descriptor_type  = vk::DescriptorType.SAMPLED_IMAGE,
            .descriptor_count = tex_capacity,
            .stage_flags      = vk::ShaderStageFlagBits.ALL,
        },
        {
            .binding          = HEAP_BINDING_STORAGE,
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
    };

    // Update-after-bind flags require features only the indexing path enables.
    uint binding_flag_bits = (uint)vk::DescriptorBindingFlagBits.UPDATE_AFTER_BIND_BIT
        | (uint)vk::DescriptorBindingFlagBits.UPDATE_UNUSED_WHILE_PENDING_BIT
        | (uint)vk::DescriptorBindingFlagBits.PARTIALLY_BOUND_BIT;
    vk::DescriptorBindingFlags[3] binding_flags = {
        (vk::DescriptorBindingFlags)binding_flag_bits,
        (vk::DescriptorBindingFlags)binding_flag_bits,
        (vk::DescriptorBindingFlags)binding_flag_bits,
    };
    vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info = vk::descriptor_set_layout_binding_flags_create_info()
        .set_binding_flags(binding_flags[..]);

    vk::DescriptorSetLayoutCreateInfo layout_info = vk::descriptor_set_layout_create_info()
        .set_bindings(bindings[..]);
    if (state.strict_heap.mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
        layout_info = layout_info
            .set_next(&flags_info)
            .set_flags(vk::DescriptorSetLayoutCreateFlagBits.UPDATE_AFTER_BIND_POOL_BIT);
    } else {
        layout_info = layout_info
            .set_flags(vk::DescriptorSetLayoutCreateFlagBits.DESCRIPTOR_BUFFER_BIT_EXT);
    }
    check_backend_result(
        state:     state,
        operation: "create_device",
        result:    vk::create_descriptor_set_layout(state.device, &layout_info, null, &heap.set_layout),
    )!;

    if (state.strict_heap.mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
        vk::DescriptorPoolSize[3] pool_sizes = {
            { .type = vk::DescriptorType.SAMPLED_IMAGE, .descriptor_count = tex_capacity },
            { .type = vk::DescriptorType.STORAGE_IMAGE, .descriptor_count = tex_capacity },
            { .type = vk::DescriptorType.SAMPLER,       .descriptor_count = smp_capacity },
        };
        vk::DescriptorPoolCreateInfo pool_info = vk::descriptor_pool_create_info()
            .set_flags(vk::DescriptorPoolCreateFlagBits.UPDATE_AFTER_BIND_BIT)
            .set_max_sets(1)
            .set_pool_sizes(pool_sizes[..]);
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
    } else {
        create_descriptor_buffer(state)!;
    }
}

<* The family slice must remain alive through Vulkan buffer creation. *>
fn vk::BufferCreateInfo make_descriptor_buffer_create_info(
    vk::DeviceSize layout_size,
    uint[] queue_families,
) @private {
    uint usage = (uint)vk::BufferUsageFlagBits.RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT
        | (uint)vk::BufferUsageFlagBits.SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT
        | (uint)vk::BufferUsageFlagBits.SHADER_DEVICE_ADDRESS_BIT;
    vk::BufferCreateInfo info = vk::buffer_create_info()
        .set_size(layout_size)
        .set_usage((vk::BufferUsageFlags)usage);
    if (queue_families.len >= 2) {
        info = info
            .set_sharing_mode(vk::SharingMode.CONCURRENT)
            .set_queue_family_indices(queue_families);
    }
    return info;
}

<* Descriptor sizes, binding offsets, and total layout size come from the
   driver; descriptors are written straight into mapped memory. *>
fn void? create_descriptor_buffer(VkDeviceState* state) @private {
    DescriptorHeapState* heap = &state.strict_heap.descriptors;

    vk::PhysicalDeviceDescriptorBufferPropertiesEXT db_props = vk::physical_device_descriptor_buffer_properties_ext();
    vk::PhysicalDeviceProperties2 props2 = vk::physical_device_properties2().set_next(&db_props);
    vk::get_physical_device_properties2(state.physical_device, &props2);
    heap.sampled_descriptor_size = (usz)db_props.sampled_image_descriptor_size;
    heap.storage_descriptor_size = (usz)db_props.storage_image_descriptor_size;
    heap.sampler_descriptor_size = (usz)db_props.sampler_descriptor_size;

    vk::DeviceSize layout_size;
    state.device_dispatch.descriptor_buffer.get_descriptor_set_layout_size(state.device, heap.set_layout, &layout_size);

    HeapLimits limits = {
        .max_resource_descriptor_buffer_range    = db_props.max_resource_descriptor_buffer_range,
        .max_sampler_descriptor_buffer_range     = db_props.max_sampler_descriptor_buffer_range,
        .resource_descriptor_address_space_size  = db_props.resource_descriptor_buffer_address_space_size,
        .sampler_descriptor_address_space_size   = db_props.sampler_descriptor_buffer_address_space_size,
        .descriptor_buffer_address_space_size    = db_props.descriptor_buffer_address_space_size,
        .layout_size                          = layout_size,
    };
    bool fits = heap_fits_limits(
        mode:         state.strict_heap.mode,
        tex_capacity: state.strict_heap.texture_capacity,
        smp_capacity: state.strict_heap.sampler_capacity,
        limits:       limits,
    );
    if (!fits) {
        if (state.debug_callback == null) {
            io::eprintfn(
                "gpu: descriptor buffer layout size %d exceeds a device descriptor-buffer bound (ranges %d/%d, address space %d/%d/%d)",
                layout_size,
                limits.max_resource_descriptor_buffer_range, limits.max_sampler_descriptor_buffer_range,
                limits.resource_descriptor_address_space_size, limits.sampler_descriptor_address_space_size,
                limits.descriptor_buffer_address_space_size);
        }
        return report_contract_failure(
            state:               state,
            operation:           "create_device",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.DEVICE,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "DeviceDesc.texture_descriptor_capacity/sampler_descriptor_capacity",
            invariant:           "descriptor buffer layout must fit device limits",
            backend_text:        null,
        )~;
    }

    state.device_dispatch.descriptor_buffer.get_descriptor_set_layout_binding_offset(
        state.device,
        heap.set_layout,
        HEAP_BINDING_SAMPLED,
        &heap.binding_offset_sampled,
    );
    state.device_dispatch.descriptor_buffer.get_descriptor_set_layout_binding_offset(
        state.device,
        heap.set_layout,
        HEAP_BINDING_STORAGE,
        &heap.binding_offset_storage,
    );
    state.device_dispatch.descriptor_buffer.get_descriptor_set_layout_binding_offset(
        state.device,
        heap.set_layout,
        HEAP_BINDING_SAMPLERS,
        &heap.binding_offset_samplers,
    );

    uint[MAX_QUEUE_FAMILIES] family_storage;
    uint family_count = strict_heap_queue_families(state, family_storage[..]);
    vk::BufferCreateInfo buffer_info = make_descriptor_buffer_create_info(layout_size, family_storage[:family_count]);
    // Descriptor bytes are written mapped (vkGetDescriptorEXT) with no
    // flush hook — require coherent memory.
    vma::AllocationCreateInfo alloc_info = {
        .usage          = vma::MemoryUsage.AUTO_PREFER_HOST,
        .flags          = { .host_access_sequential_write, .mapped },
        .required_flags = (vk::MemoryPropertyFlags)vk::MemoryPropertyFlagBits.HOST_COHERENT_BIT,
    };
    vma::AllocationInfo allocation_info;
    check_backend_result(
        state:     state,
        operation: "create_device",
        result:    state.allocator.create_buffer(
            buffer_info:    &buffer_info,
            alloc_info:     &alloc_info,
            out_buffer:     &heap.descriptor_buffer,
            out_allocation: &heap.descriptor_buffer_allocation,
            out_info:       &allocation_info,
        ),
    )!;
    heap.descriptor_buffer_cpu = allocation_info.mapped_data;

    vk::BufferDeviceAddressInfo address_info = vk::buffer_device_address_info().set_buffer(heap.descriptor_buffer);
    heap.descriptor_buffer_gpu = vk::get_buffer_device_address(state.device, &address_info);
    if (heap.descriptor_buffer_gpu == 0) {
        return report_descriptor_backend_failure(
            state:          state,
            operation:      "create_device",
            public_fault:   gpu::BACKEND_ERROR,
            backend_text:   "descriptor buffer device address is zero",
            backend_result: 0,
        );
    }
}

<* Safe on a partial or zeroed heap. *>
fn void destroy_descriptor_heap(VkDeviceState* state) {
    DescriptorHeapState* heap = &state.strict_heap.descriptors;
    if (heap.descriptor_pool != null) {
        vk::destroy_descriptor_pool(state.device, heap.descriptor_pool, null);
        heap.descriptor_pool = (vk::DescriptorPool)null;
    }
    if (heap.descriptor_buffer != null && state.allocator != null) {
        state.allocator.destroy_buffer(heap.descriptor_buffer, heap.descriptor_buffer_allocation);
        heap.descriptor_buffer = (vk::Buffer)null;
    }
    if (heap.set_layout != null) {
        vk::destroy_descriptor_set_layout(state.device, heap.set_layout, null);
        heap.set_layout = (vk::DescriptorSetLayout)null;
    }
    if (heap.texture_cells.ptr != null)          { mem::free(heap.texture_cells.ptr); heap.texture_cells = {}; }
    if (heap.texture_free.ptr != null)           { mem::free(heap.texture_free.ptr); heap.texture_free = {}; }
}

fn void report_descriptor_leaks(VkDeviceState* state) {
    DescriptorHeapState* heap = &state.strict_heap.descriptors;
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
}

fn TextureDescriptorCell*? texture_cell_get(DescriptorHeapState* heap, gpu::TextureIndex index) @private {
    uint slot = index.slot;
    if (slot >= heap.texture_cells.len || !heap.texture_cells[slot].used) return gpu::INVALID_HANDLE~;
    if (heap.texture_cells[slot].generation != index.generation) return gpu::INVALID_HANDLE~;
    return &heap.texture_cells[slot];
}

<* Sampled and/or storage descriptor per the texture's usage, same slot in
   both bindings, so one TextureIndex serves both access kinds. Descriptor
   layout follows usage: sampled-only textures are accessed in SHADER_READ,
   storage-capable ones in GENERAL. *>
fn void write_texture_descriptor(VkDeviceState* state, uint slot, TextureSlot* texture, vk::ImageView view) @private {
    DescriptorHeapState* heap = &state.strict_heap.descriptors;
    vk::DescriptorImageInfo image_info = {
        .image_view   = view,
        .image_layout = texture.usage.storage ? vk::ImageLayout.GENERAL : vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
    };

    if (state.strict_heap.mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
        vk::WriteDescriptorSet[2] writes;
        uint write_count = 0;
        if (texture.usage.sampled) {
            writes[write_count++] = vk::write_descriptor_set()
                .set_dst_set(heap.descriptor_set)
                .set_dst_binding(HEAP_BINDING_SAMPLED)
                .set_dst_array_element(slot)
                .set_descriptor_type(vk::DescriptorType.SAMPLED_IMAGE)
                .set_image_info((&image_info)[:1]);
        }
        if (texture.usage.storage) {
            writes[write_count++] = vk::write_descriptor_set()
                .set_dst_set(heap.descriptor_set)
                .set_dst_binding(HEAP_BINDING_STORAGE)
                .set_dst_array_element(slot)
                .set_descriptor_type(vk::DescriptorType.STORAGE_IMAGE)
                .set_image_info((&image_info)[:1]);
        }
        if (write_count > 0) {
            vk::update_descriptor_sets(state.device, write_count, &writes[0], 0, null);
        }
        return;
    }

    if (texture.usage.sampled) {
        vk::DescriptorGetInfoEXT info = vk::descriptor_get_info_ext();
        info.type = vk::DescriptorType.SAMPLED_IMAGE;
        info.data.p_sampled_image = &image_info;
        void* dst = (char*)heap.descriptor_buffer_cpu + (usz)heap.binding_offset_sampled + slot * heap.sampled_descriptor_size;
        state.device_dispatch.descriptor_buffer.get_descriptor(
            state.device,
            &info,
            (sz)heap.sampled_descriptor_size,
            dst,
        );
    }
    if (texture.usage.storage) {
        vk::DescriptorGetInfoEXT info = vk::descriptor_get_info_ext();
        info.type = vk::DescriptorType.STORAGE_IMAGE;
        info.data.p_storage_image = &image_info;
        void* dst = (char*)heap.descriptor_buffer_cpu + (usz)heap.binding_offset_storage + slot * heap.storage_descriptor_size;
        state.device_dispatch.descriptor_buffer.get_descriptor(
            state.device,
            &info,
            (sz)heap.storage_descriptor_size,
            dst,
        );
    }
}

fn void write_sampler_descriptor(VkDeviceState* state, uint slot, vk::Sampler sampler) @private {
    DescriptorHeapState* heap = &state.strict_heap.descriptors;

    if (state.strict_heap.mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
        vk::DescriptorImageInfo image_info = { .sampler = sampler };
        vk::WriteDescriptorSet write = vk::write_descriptor_set()
            .set_dst_set(heap.descriptor_set)
            .set_dst_binding(HEAP_BINDING_SAMPLERS)
            .set_dst_array_element(slot)
            .set_descriptor_type(vk::DescriptorType.SAMPLER)
            .set_image_info((&image_info)[:1]);
        vk::update_descriptor_sets(state.device, 1, &write, 0, null);
        return;
    }

    vk::DescriptorGetInfoEXT info = vk::descriptor_get_info_ext();
    info.type = vk::DescriptorType.SAMPLER;
    info.data.p_sampler = &sampler;
    void* dst = (char*)heap.descriptor_buffer_cpu + (usz)heap.binding_offset_samplers + slot * heap.sampler_descriptor_size;
    state.device_dispatch.descriptor_buffer.get_descriptor(
        state.device,
        &info,
        (sz)heap.sampler_descriptor_size,
        dst,
    );
}

fn void? validate_descriptor_heap_binding(VkDeviceState* state) {
    if (!state.strict_heap.enabled) return gpu::UNSUPPORTED_FEATURE~;
}

fn void? bind_descriptor_heap(VkDeviceState* state, vk::CommandBuffer cb, vk::PipelineLayout layout, vk::PipelineBindPoint bind_point) {
    validate_descriptor_heap_binding(state)!;
    if (state.strict_heap.mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
        vk::cmd_bind_descriptor_sets(
            command_buffer:        cb,
            pipeline_bind_point:   bind_point,
            layout:                layout,
            first_set:             HEAP_SET,
            descriptor_set_count:  1,
            p_descriptor_sets:     &state.strict_heap.descriptors.descriptor_set,
            dynamic_offset_count:  0,
            p_dynamic_offsets:     null,
        );
        return;
    }

    uint usage = (uint)vk::BufferUsageFlagBits.RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT
        | (uint)vk::BufferUsageFlagBits.SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT
        | (uint)vk::BufferUsageFlagBits.SHADER_DEVICE_ADDRESS_BIT;
    vk::DescriptorBufferBindingInfoEXT binding_info = vk::descriptor_buffer_binding_info_ext();
    binding_info.address = state.strict_heap.descriptors.descriptor_buffer_gpu;
    binding_info.usage   = (vk::BufferUsageFlags)usage;
    state.device_dispatch.descriptor_buffer.cmd_bind_descriptor_buffers(cb, 1, &binding_info);

    uint buffer_index = 0;
    vk::DeviceSize offset = 0;
    state.device_dispatch.descriptor_buffer.cmd_set_descriptor_buffer_offsets(
        cb,
        bind_point,
        layout,
        HEAP_SET,
        1,
        &buffer_index,
        &offset,
    );
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
            invariant:           "view must select a valid texture subresource range and format",
            backend_text:        null,
        )~;
    }
    return resolved;
}

fn gpu::TextureIndex? vk_create_texture_descriptor(gpu::Device* device, gpu::TextureHandle texture, gpu::TextureViewDesc* view) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    DescriptorHeapState* heap = &state.strict_heap.descriptors;
    TextureSlot* texture_slot = resolve_descriptor_texture(
        state:         state,
        texture:       texture,
        operation:     "create_texture_descriptor",
        texture_field: "texture",
        usage_field:   "TextureDesc.usage",
    )!;

    uint headroom = heap.texture_free_count
        + (uint)heap.texture_cells.len - heap.texture_next_new;
    if (headroom == 0) {
        return report_contract_failure(
            state:               state,
            operation:           "create_texture_descriptor",
            public_fault:        gpu::DESCRIPTOR_HEAP_FULL,
            resource_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "DeviceDesc.texture_descriptor_capacity",
            invariant:           "must exceed live texture descriptors",
            backend_text:        null,
        )~;
    }

    CachedViewCreation creation;
    vk::ImageView image_view = resolve_descriptor_texture_view(
        state:     state,
        slot:      texture_slot,
        texture:   texture,
        view:      view,
        operation: "create_texture_descriptor",
        view_field: "view",
        creation:  &creation,
    )!;

    gpu::TextureIndex index;
    commit_texture_descriptor_item(
        heap:      heap,
        texture:   texture,
        out_index: &index,
    );
    write_texture_descriptor(state, index.slot, texture_slot, image_view);
    return index;
}

<* Validate and resolve one batch item without touching descriptor allocation
   or output state. The caller owns any returned cache creation until commit. *>
fn void? prepare_texture_descriptor_item(
    VkDeviceState* state,
    gpu::TextureDescriptorDesc* desc,
    PendingTextureWrite* pending,
    CachedViewCreation* creation,
) @private {
    TextureSlot* texture_slot = resolve_descriptor_texture(
        state:         state,
        texture:       desc.texture,
        operation:     "create_texture_descriptors",
        texture_field: "descs[].texture",
        usage_field:   "TextureDesc.usage",
    )!;
    vk::ImageView image_view = resolve_descriptor_texture_view(
        state:      state,
        slot:       texture_slot,
        texture:    desc.texture,
        view:       &desc.view,
        operation:  "create_texture_descriptors",
        view_field: "descs[].view",
        creation:   creation,
    )!;
    *pending = { .texture_slot = texture_slot, .view = image_view };
}

<*
 @require heap.texture_free_count > 0 || heap.texture_next_new < heap.texture_cells.len : "descriptor headroom"
*>
fn void commit_texture_descriptor_item(DescriptorHeapState* heap, gpu::TextureHandle texture, gpu::TextureIndex* out_index) @private {
    uint slot;
    if (heap.texture_free_count > 0) {
        slot = heap.texture_free[--heap.texture_free_count];
    } else {
        slot = heap.texture_next_new++;
    }

    TextureDescriptorCell* cell = &heap.texture_cells[slot];
    cell.owner = texture;
    cell.used = true;
    cell.generation = bump_generation(cell.generation);
    *out_index = { .slot = (ushort)slot, .generation = cell.generation };
}

fn uint accumulate_texture_descriptor_write(DescriptorHeapState* heap, uint slot, TextureSlot* texture_slot, vk::ImageView view, vk::DescriptorImageInfo[] image_infos, vk::WriteDescriptorSet[] writes, uint write_count) @private {
    vk::DescriptorImageInfo image_info = {
        .image_view   = view,
        .image_layout = texture_slot.usage.storage ? vk::ImageLayout.GENERAL : vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
    };
    if (texture_slot.usage.sampled) {
        image_infos[write_count] = image_info;
        writes[write_count] = vk::write_descriptor_set()
            .set_dst_set(heap.descriptor_set)
            .set_dst_binding(HEAP_BINDING_SAMPLED)
            .set_dst_array_element(slot)
            .set_descriptor_type(vk::DescriptorType.SAMPLED_IMAGE)
            .set_image_info((&image_infos[write_count])[:1]);
        write_count++;
    }
    if (texture_slot.usage.storage) {
        image_infos[write_count] = image_info;
        writes[write_count] = vk::write_descriptor_set()
            .set_dst_set(heap.descriptor_set)
            .set_dst_binding(HEAP_BINDING_STORAGE)
            .set_dst_array_element(slot)
            .set_descriptor_type(vk::DescriptorType.STORAGE_IMAGE)
            .set_image_info((&image_infos[write_count])[:1]);
        write_count++;
    }
    return write_count;
}

<* All validation and fallible view creation precedes descriptor mutation.
   A preparation fault restores only views created by this transaction. *>
fn void? vk_create_texture_descriptors(gpu::Device* device, gpu::TextureDescriptorDesc[] descs, gpu::TextureIndex[] out_indices) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    if (out_indices.len != descs.len) {
        return report_contract_failure(
            state:               state,
            operation:           "create_texture_descriptors",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "out_indices",
            invariant:           "output count must match descriptor count",
            backend_text:        null,
        )~;
    }
    if (descs.len == 0) return;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    DescriptorHeapState* heap = &state.strict_heap.descriptors;

    uint headroom = heap.texture_free_count
        + (uint)heap.texture_cells.len - heap.texture_next_new;
    if ((uint)descs.len > headroom) {
        return report_contract_failure(
            state:               state,
            operation:           "create_texture_descriptors",
            public_fault:        gpu::DESCRIPTOR_HEAP_FULL,
            resource_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "DeviceDesc.texture_descriptor_capacity",
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
                state:    state,
                desc:     &descs[i],
                pending:  &pending[i],
                creation: &creation,
            )) {
                rollback_texture_view_creations(state, creations[:creation_count]);
                return f~;
            }
            if (creation.cell != null) creations[creation_count++] = creation;
        }

        for (uint i = 0; i < descs.len; i++) {
            commit_texture_descriptor_item(heap, descs[i].texture, &out_indices[i]);
        }

        if (state.strict_heap.mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
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
                    heap:         heap,
                    slot:         out_indices[i].slot,
                    texture_slot: item.texture_slot,
                    view:         item.view,
                    image_infos:  image_infos,
                    writes:       writes,
                    write_count:  write_count,
                );
            }
            if (write_count > 0) vk::update_descriptor_sets(state.device, write_count, &writes[0], 0, null);
        } else {
            foreach (i, &item : pending) {
                write_texture_descriptor(state, out_indices[i].slot, item.texture_slot, item.view);
            }
        }
    };
}

fn void? vk_destroy_texture_descriptor(gpu::Device* device, gpu::TextureIndex index) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if (catch f = texture_cell_get(&state.strict_heap.descriptors, index)) {
        return report_contract_failure(
            state:               state,
            operation:           "destroy_texture_descriptor",
            public_fault:        f,
            resource_kind:       gpu::DebugResourceKind.TEXTURE_DESCRIPTOR,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "index",
            invariant:           "descriptor index must identify a live generation",
            backend_text:        null,
        )~;
    }
    TextureDescriptorCell* cell = texture_cell_get(&state.strict_heap.descriptors, index)!!;
    cell.used = false;
    cell.generation = bump_generation(cell.generation);
    state.strict_heap.descriptors.texture_free[state.strict_heap.descriptors.texture_free_count++] = (uint)index.slot;
}
