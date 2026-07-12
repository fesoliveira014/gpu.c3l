module gpu::vk;

import gpu;
import vk;
import std::thread;
import vma;
import std::io;
import std::core::mem;
import std::math;

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

struct SamplerDescriptorCell {
    vk::Sampler sampler;
    ushort      generation;
    bool        used;
}

struct DescriptorRetireEntry {
    uint  slot;
    ulong timeline_value;
}

// FIFO sized to heap capacity (each slot retires at most once before reuse,
// so it cannot overflow); push order is non-decreasing in timeline value.
struct DescriptorRetireRing {
    DescriptorRetireEntry[] entries;
    uint                    head;
    uint                    count;
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

struct HeapDescriptorTotals {
    ulong per_stage_resources;
    ulong all_pools_descriptors;
}

struct DescriptorHeapState {
    TextureDescriptorCell[] texture_cells;
    SamplerDescriptorCell[] sampler_cells;
    uint                    texture_next_new;
    uint                    sampler_next_new;
    uint[]                  texture_free;
    uint                    texture_free_count;
    uint[]                  sampler_free;
    uint                    sampler_free_count;
    DescriptorRetireRing    texture_retire;
    DescriptorRetireRing    sampler_retire;
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

fn void DescriptorRetireRing.push(&self, uint slot, ulong timeline_value) {
    uint tail = (self.head + self.count) % (uint)self.entries.len;
    self.entries[tail] = { .slot = slot, .timeline_value = timeline_value };
    self.count++;
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

fn void? check_descriptor_retire_backend_result(
    VkDeviceState* state,
    ZString operation,
    vk::Result result,
) {
    if (operation == null) return check(result);
    return check_backend_result(state, operation, result);
}

fn ushort bump_generation(ushort generation) @private {
    uint next = ((uint)generation + 1) & 0xFFFF;
    return next == 0 ? (ushort)1 : (ushort)next;
}

fn HeapDescriptorTotals descriptor_heap_totals(uint tex_capacity, uint smp_capacity) @private {
    return {
        .per_stage_resources   = (ulong)tex_capacity * 2,
        .all_pools_descriptors = (ulong)tex_capacity * 2 + (ulong)smp_capacity,
    };
}

<* Texture capacity binds both SAMPLED_IMAGE and STORAGE_IMAGE, so on the
   indexing path it must satisfy both image limits. On the buffer path the
   bound quantity is the descriptor set's byte layout, not slot counts. *>
fn bool heap_fits_limits(gpu::DescriptorHeapMode mode, uint tex_capacity, uint smp_capacity, HeapLimits limits) {
    if (mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
        HeapDescriptorTotals totals = descriptor_heap_totals(tex_capacity, smp_capacity);
        // Bindings use ALL stages, so per-stage limits bind alongside the
        // set-level ones.
        return tex_capacity <= limits.max_sampled_images
            && tex_capacity <= limits.max_storage_images
            && tex_capacity <= limits.max_per_stage_sampled_images
            && tex_capacity <= limits.max_per_stage_storage_images
            && smp_capacity <= limits.max_samplers
            && smp_capacity <= limits.max_per_stage_samplers
            && totals.per_stage_resources <= (ulong)limits.max_per_stage_resources
            && totals.all_pools_descriptors <= (ulong)limits.max_all_pools_descriptors;
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
    DescriptorHeapState* heap = &state.heap;
    uint tex_capacity = state.texture_descriptor_capacity;
    uint smp_capacity = state.sampler_descriptor_capacity;

    if (state.heap_mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
        vk::PhysicalDeviceVulkan12Properties v12_props = vk::physical_device_vulkan12properties();
        vk::PhysicalDeviceProperties2 v12_props2 = vk::physical_device_properties2().set_next(&v12_props);
        vk::get_physical_device_properties2(state.physical_device, &v12_props2);
        HeapLimits limits = {
            .max_sampled_images           = v12_props.max_descriptor_set_update_after_bind_sampled_images,
            .max_storage_images           = v12_props.max_descriptor_set_update_after_bind_storage_images,
            .max_samplers                 = v12_props.max_descriptor_set_update_after_bind_samplers,
            .max_per_stage_sampled_images = v12_props.max_per_stage_descriptor_update_after_bind_sampled_images,
            .max_per_stage_storage_images = v12_props.max_per_stage_descriptor_update_after_bind_storage_images,
            .max_per_stage_samplers       = v12_props.max_per_stage_descriptor_update_after_bind_samplers,
            .max_per_stage_resources      = v12_props.max_per_stage_update_after_bind_resources,
            .max_all_pools_descriptors    = v12_props.max_update_after_bind_descriptors_in_all_pools,
        };
        bool fits = heap_fits_limits(
            mode:         state.heap_mode,
            tex_capacity: tex_capacity,
            smp_capacity: smp_capacity,
            limits:       limits,
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
    heap.sampler_cells          = mem::new_array(SamplerDescriptorCell, smp_capacity);
    heap.texture_free           = mem::new_array(uint, tex_capacity);
    heap.sampler_free           = mem::new_array(uint, smp_capacity);
    heap.texture_retire.entries = mem::new_array(DescriptorRetireEntry, tex_capacity);
    heap.sampler_retire.entries = mem::new_array(DescriptorRetireEntry, smp_capacity);

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
    if (state.heap_mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
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

    if (state.heap_mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
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
    DescriptorHeapState* heap = &state.heap;

    vk::PhysicalDeviceDescriptorBufferPropertiesEXT db_props = vk::physical_device_descriptor_buffer_properties_ext();
    vk::PhysicalDeviceProperties2 props2 = vk::physical_device_properties2().set_next(&db_props);
    vk::get_physical_device_properties2(state.physical_device, &props2);
    heap.sampled_descriptor_size = (usz)db_props.sampled_image_descriptor_size;
    heap.storage_descriptor_size = (usz)db_props.storage_image_descriptor_size;
    heap.sampler_descriptor_size = (usz)db_props.sampler_descriptor_size;

    vk::DeviceSize layout_size;
    vk::extensions.vk_get_descriptor_set_layout_size_ext(state.device, heap.set_layout, &layout_size);

    HeapLimits limits = {
        .max_resource_descriptor_buffer_range    = db_props.max_resource_descriptor_buffer_range,
        .max_sampler_descriptor_buffer_range     = db_props.max_sampler_descriptor_buffer_range,
        .resource_descriptor_address_space_size  = db_props.resource_descriptor_buffer_address_space_size,
        .sampler_descriptor_address_space_size   = db_props.sampler_descriptor_buffer_address_space_size,
        .descriptor_buffer_address_space_size    = db_props.descriptor_buffer_address_space_size,
        .layout_size                          = layout_size,
    };
    bool fits = heap_fits_limits(
        mode:         state.heap_mode,
        tex_capacity: state.texture_descriptor_capacity,
        smp_capacity: state.sampler_descriptor_capacity,
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

    vk::extensions.vk_get_descriptor_set_layout_binding_offset_ext(state.device, heap.set_layout, HEAP_BINDING_SAMPLED, &heap.binding_offset_sampled);
    vk::extensions.vk_get_descriptor_set_layout_binding_offset_ext(state.device, heap.set_layout, HEAP_BINDING_STORAGE, &heap.binding_offset_storage);
    vk::extensions.vk_get_descriptor_set_layout_binding_offset_ext(state.device, heap.set_layout, HEAP_BINDING_SAMPLERS, &heap.binding_offset_samplers);

    uint[2] family_storage;
    uint family_count = graphics_compute_queue_families(state, family_storage[..]);
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

<* Safe on a partial or zeroed heap; destroys live sampler objects, including
   retired-but-undrained ones. *>
fn void destroy_descriptor_heap(VkDeviceState* state) {
    DescriptorHeapState* heap = &state.heap;
    if (heap.sampler_cells.ptr != null) {
        foreach (&cell : heap.sampler_cells[:heap.sampler_next_new]) {
            if (cell.sampler != null) {
                vk::destroy_sampler(state.device, cell.sampler, null);
                cell.sampler = (vk::Sampler)null;
            }
        }
    }
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
    if (heap.sampler_cells.ptr != null)          { mem::free(heap.sampler_cells.ptr); heap.sampler_cells = {}; }
    if (heap.texture_free.ptr != null)           { mem::free(heap.texture_free.ptr); heap.texture_free = {}; }
    if (heap.sampler_free.ptr != null)           { mem::free(heap.sampler_free.ptr); heap.sampler_free = {}; }
    if (heap.texture_retire.entries.ptr != null) { mem::free(heap.texture_retire.entries.ptr); heap.texture_retire.entries = {}; }
    if (heap.sampler_retire.entries.ptr != null) { mem::free(heap.sampler_retire.entries.ptr); heap.sampler_retire.entries = {}; }
}

fn void report_descriptor_leaks(VkDeviceState* state) {
    DescriptorHeapState* heap = &state.heap;
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
    if (heap.sampler_cells.ptr != null) {
        foreach (i, &cell : heap.sampler_cells[:heap.sampler_next_new]) {
            if (!cell.used) continue;
            if (!report_resource_lifetime(
                state:               state,
                resource_kind:       gpu::DebugResourceKind.SAMPLER,
                has_public_identity: true,
                index:               (uint)i,
                generation:          (uint)cell.generation,
                debug_name:          null,
                backend_text:        "sampler descriptor remains live",
            )) {
                io::eprintfn("gpu: leaked sampler descriptor slot %d", i);
            }
        }
    }
}

fn uint? count_ready_texture_retires(VkDeviceState* state, ZString operation) @private {
    DescriptorRetireRing* retire = &state.heap.texture_retire;
    if (retire.count == 0) return 0;

    ulong current = 0;
    check_descriptor_retire_backend_result(
        state:     state,
        operation: operation,
        result:    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &current),
    )!;
    uint ready = 0;
    while (ready < retire.count) {
        uint index = (retire.head + ready) % (uint)retire.entries.len;
        if (retire.entries[index].timeline_value > current) break;
        ready++;
    }
    return ready;
}

fn void drain_texture_retire_prefix(VkDeviceState* state, uint ready_count) @private {
    DescriptorHeapState* heap = &state.heap;
    for (uint i = 0; i < ready_count; i++) {
        DescriptorRetireEntry* entry = &heap.texture_retire.entries[heap.texture_retire.head];
        heap.texture_free[heap.texture_free_count++] = entry.slot;
        heap.texture_retire.head = (heap.texture_retire.head + 1) % (uint)heap.texture_retire.entries.len;
        heap.texture_retire.count--;
    }
}

<* Recycles ready ring slots under the caller's lock; the ready handles land
   in `drained` for destruction after unlock. Returns the count. *>
fn uint drain_sampler_retire(VkDeviceState* state, vk::Sampler[] drained) @private {
    DescriptorHeapState* heap = &state.heap;
    if (heap.sampler_retire.count == 0) return 0;
    ulong current = 0;
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &current);
    uint drained_count = 0;
    while (heap.sampler_retire.count > 0) {
        DescriptorRetireEntry* entry = &heap.sampler_retire.entries[heap.sampler_retire.head];
        if (entry.timeline_value > current) break;
        SamplerDescriptorCell* cell = &heap.sampler_cells[entry.slot];
        if (cell.sampler != null) {
            drained[drained_count++] = cell.sampler;
            cell.sampler = (vk::Sampler)null;
        }
        heap.sampler_free[heap.sampler_free_count++] = entry.slot;
        heap.sampler_retire.head = (heap.sampler_retire.head + 1) % (uint)heap.sampler_retire.entries.len;
        heap.sampler_retire.count--;
    }
    return drained_count;
}

fn TextureDescriptorCell*? texture_cell_get(DescriptorHeapState* heap, gpu::TextureIndex index) @private {
    uint slot = index.slot;
    if (slot >= heap.texture_cells.len || !heap.texture_cells[slot].used) return gpu::INVALID_HANDLE~;
    if (heap.texture_cells[slot].generation != index.generation) return gpu::INVALID_HANDLE~;
    return &heap.texture_cells[slot];
}

fn SamplerDescriptorCell*? sampler_cell_get(DescriptorHeapState* heap, gpu::SamplerIndex index) @private {
    uint slot = index.slot;
    if (slot >= heap.sampler_cells.len || !heap.sampler_cells[slot].used) return gpu::INVALID_HANDLE~;
    if (heap.sampler_cells[slot].generation != index.generation) return gpu::INVALID_HANDLE~;
    return &heap.sampler_cells[slot];
}

<* Sampled and/or storage descriptor per the texture's usage, same slot in
   both bindings, so one TextureIndex serves both access kinds. Descriptor
   layout follows usage: sampled-only textures are accessed in SHADER_READ,
   storage-capable ones in GENERAL. *>
fn void write_texture_descriptor(VkDeviceState* state, uint slot, TextureSlot* texture, vk::ImageView view) @private {
    DescriptorHeapState* heap = &state.heap;
    vk::DescriptorImageInfo image_info = {
        .image_view   = view,
        .image_layout = texture.usage.storage ? vk::ImageLayout.GENERAL : vk::ImageLayout.SHADER_READ_ONLY_OPTIMAL,
    };

    if (state.heap_mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
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
        vk::get_descriptor_ext(state.device, &info, (sz)heap.sampled_descriptor_size, dst);
    }
    if (texture.usage.storage) {
        vk::DescriptorGetInfoEXT info = vk::descriptor_get_info_ext();
        info.type = vk::DescriptorType.STORAGE_IMAGE;
        info.data.p_storage_image = &image_info;
        void* dst = (char*)heap.descriptor_buffer_cpu + (usz)heap.binding_offset_storage + slot * heap.storage_descriptor_size;
        vk::get_descriptor_ext(state.device, &info, (sz)heap.storage_descriptor_size, dst);
    }
}

fn void write_sampler_descriptor(VkDeviceState* state, uint slot, vk::Sampler sampler) @private {
    DescriptorHeapState* heap = &state.heap;

    if (state.heap_mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
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
    vk::get_descriptor_ext(state.device, &info, (sz)heap.sampler_descriptor_size, dst);
}

fn void bind_descriptor_heap(VkDeviceState* state, vk::CommandBuffer cb, vk::PipelineLayout layout, vk::PipelineBindPoint bind_point) {
    if (state.heap_mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
        vk::cmd_bind_descriptor_sets(cb, bind_point, layout, HEAP_SET, 1, &state.heap.descriptor_set, 0, null);
        return;
    }

    uint usage = (uint)vk::BufferUsageFlagBits.RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT
        | (uint)vk::BufferUsageFlagBits.SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT
        | (uint)vk::BufferUsageFlagBits.SHADER_DEVICE_ADDRESS_BIT;
    vk::DescriptorBufferBindingInfoEXT binding_info = vk::descriptor_buffer_binding_info_ext();
    binding_info.address = state.heap.descriptor_buffer_gpu;
    binding_info.usage   = (vk::BufferUsageFlags)usage;
    vk::cmd_bind_descriptor_buffers_ext(cb, 1, &binding_info);

    uint buffer_index = 0;
    vk::DeviceSize offset = 0;
    vk::cmd_set_descriptor_buffer_offsets_ext(cb, bind_point, layout, HEAP_SET, 1, &buffer_index, &offset);
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
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    DescriptorHeapState* heap = &state.heap;
    TextureSlot* texture_slot = resolve_descriptor_texture(
        state:         state,
        texture:       texture,
        operation:     "create_texture_descriptor",
        texture_field: "texture",
        usage_field:   "TextureDesc.usage",
    )!;

    uint ready_retires = count_ready_texture_retires(state, null) ?? 0;
    uint headroom = heap.texture_free_count + ready_retires
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
            invariant:           "must exceed live and retiring texture descriptors",
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

    drain_texture_retire_prefix(state, ready_retires);
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
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
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

    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    DescriptorHeapState* heap = &state.heap;

    uint ready_retires = count_ready_texture_retires(state, "create_texture_descriptors")!;
    uint headroom = heap.texture_free_count + ready_retires
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
            invariant:           "must fit the complete descriptor batch after live and retiring descriptors",
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

        drain_texture_retire_prefix(state, ready_retires);
        for (uint i = 0; i < descs.len; i++) {
            commit_texture_descriptor_item(heap, descs[i].texture, &out_indices[i]);
        }

        if (state.heap_mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
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
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if (catch f = texture_cell_get(&state.heap, index)) {
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
    TextureDescriptorCell* cell = texture_cell_get(&state.heap, index)!!;
    cell.used = false;
    cell.generation = bump_generation(cell.generation);
    state.heap.texture_retire.push((uint)index.slot, retire_timeline_value(state));
}

fn vk::Filter filter_to_vk(gpu::Filter filter) @private {
    return filter == gpu::Filter.LINEAR ? vk::Filter.LINEAR : vk::Filter.NEAREST;
}

fn vk::SamplerAddressMode address_mode_to_vk(gpu::AddressMode mode) @private {
    switch (mode) {
        case gpu::AddressMode.REPEAT:          return vk::SamplerAddressMode.REPEAT;
        case gpu::AddressMode.MIRRORED_REPEAT: return vk::SamplerAddressMode.MIRRORED_REPEAT;
        case gpu::AddressMode.CLAMP_TO_EDGE:   return vk::SamplerAddressMode.CLAMP_TO_EDGE;
        case gpu::AddressMode.CLAMP_TO_BORDER: return vk::SamplerAddressMode.CLAMP_TO_BORDER;
        default:                               return vk::SamplerAddressMode.REPEAT;
    }
}

<* Heap mutation runs under the lock; faults DESCRIPTOR_HEAP_FULL with
   `sampler` untouched, so the caller destroys it unlocked. Drained retirees
   are destroyed after unlock on every exit path. *>
fn gpu::SamplerIndex? publish_sampler(VkDeviceState* state, vk::Sampler sampler) @private {
    DescriptorHeapState* heap = &state.heap;
    @pool() {
        vk::Sampler[] drained = mem::talloc_array(vk::Sampler, heap.sampler_retire.entries.len);
        uint drained_count = 0;
        defer {
            for (uint i = 0; i < drained_count; i++) vk::destroy_sampler(state.device, drained[i], null);
        }

        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();

        drained_count = drain_sampler_retire(state, drained);
        uint slot;
        if (heap.sampler_free_count > 0) {
            slot = heap.sampler_free[--heap.sampler_free_count];
        } else if (heap.sampler_next_new < heap.sampler_cells.len) {
            slot = heap.sampler_next_new++;
        } else {
            return gpu::DESCRIPTOR_HEAP_FULL~;
        }

        SamplerDescriptorCell* cell = &heap.sampler_cells[slot];
        cell.sampler = sampler;
        cell.used = true;
        cell.generation = bump_generation(cell.generation);

        write_sampler_descriptor(state, slot, sampler);
        return (gpu::SamplerIndex){ .slot = (ushort)slot, .generation = cell.generation };
    };
}

fn gpu::SamplerIndex? vk_create_sampler(gpu::Device* device, gpu::SamplerDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    if (!math::is_finite(desc.mip_lod_bias)) {
        return report_contract_failure(
            state:               state,
            operation:           "create_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.mip_lod_bias",
            invariant:           "must be finite",
            backend_text:        null,
        )~;
    }
    if (!math::is_finite(desc.min_lod)) {
        return report_contract_failure(
            state:               state,
            operation:           "create_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.min_lod",
            invariant:           "must be finite",
            backend_text:        null,
        )~;
    }
    if (!math::is_finite(desc.max_lod)) {
        return report_contract_failure(
            state:               state,
            operation:           "create_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.max_lod",
            invariant:           "must be finite",
            backend_text:        null,
        )~;
    }
    if (desc.min_lod > desc.max_lod) {
        return report_contract_failure(
            state:               state,
            operation:           "create_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.min_lod",
            invariant:           "must not exceed SamplerDesc.max_lod",
            backend_text:        null,
        )~;
    }
    if (desc.anisotropy_enable && !math::is_finite(desc.max_anisotropy)) {
        return report_contract_failure(
            state:               state,
            operation:           "create_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.max_anisotropy",
            invariant:           "must be finite when anisotropy is enabled",
            backend_text:        null,
        )~;
    }
    if (desc.anisotropy_enable && state.max_sampler_anisotropy <= 0) {
        return report_contract_failure(
            state:               state,
            operation:           "create_sampler",
            public_fault:        gpu::UNSUPPORTED_FEATURE,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.anisotropy_enable",
            invariant:           "device must support sampler anisotropy",
            backend_text:        null,
        )~;
    }
    if (desc.anisotropy_enable && desc.max_anisotropy < 1.0f) {
        return report_contract_failure(
            state:               state,
            operation:           "create_sampler",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "SamplerDesc.max_anisotropy",
            invariant:           "must be at least 1.0 when anisotropy is enabled",
            backend_text:        null,
        )~;
    }
    enter_tracked(state);
    defer exit_tracked(state);

    vk::SamplerCreateInfo sampler_info = vk::sampler_create_info()
        .set_mag_filter(filter_to_vk(desc.mag_filter))
        .set_min_filter(filter_to_vk(desc.min_filter))
        .set_mipmap_mode(desc.mip_filter == gpu::Filter.LINEAR
            ? vk::SamplerMipmapMode.LINEAR
            : vk::SamplerMipmapMode.NEAREST)
        .set_address_mode_u(address_mode_to_vk(desc.address_u))
        .set_address_mode_v(address_mode_to_vk(desc.address_v))
        .set_address_mode_w(address_mode_to_vk(desc.address_w))
        .set_mip_lod_bias(desc.mip_lod_bias)
        .set_anisotropy_enable(desc.anisotropy_enable ? vk::TRUE : vk::FALSE)
        .set_max_anisotropy(desc.max_anisotropy > state.max_sampler_anisotropy ? state.max_sampler_anisotropy : desc.max_anisotropy)
        .set_compare_enable(desc.compare_enable ? vk::TRUE : vk::FALSE)
        .set_compare_op(compare_op_to_vk(desc.compare))
        .set_min_lod(desc.min_lod)
        .set_max_lod(desc.max_lod);

    vk::Sampler sampler;
    check_backend_result(
        state:     state,
        operation: "create_sampler",
        result:    vk::create_sampler(state.device, &sampler_info, null, &sampler),
    )!;

    gpu::SamplerIndex? published = publish_sampler(state, sampler);
    if (try index = published) return index;
    fault publish_fault = @catch(published);
    vk::destroy_sampler(state.device, sampler, null);
    if (publish_fault != gpu::DESCRIPTOR_HEAP_FULL) return publish_fault~;
    return report_contract_failure(
        state:               state,
        operation:           "create_sampler",
        public_fault:        gpu::DESCRIPTOR_HEAP_FULL,
        resource_kind:       gpu::DebugResourceKind.SAMPLER,
        has_public_identity: false,
        index:               0,
        generation:          0,
        debug_name:          null,
        rejected_field:      "DeviceDesc.sampler_descriptor_capacity",
        invariant:           "must exceed live and retiring samplers",
        backend_text:        null,
    )~;
}

fn void? vk_destroy_sampler(gpu::Device* device, gpu::SamplerIndex index) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if (catch f = sampler_cell_get(&state.heap, index)) {
        return report_contract_failure(
            state:               state,
            operation:           "destroy_sampler",
            public_fault:        f,
            resource_kind:       gpu::DebugResourceKind.SAMPLER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          null,
            rejected_field:      "index",
            invariant:           "sampler index must identify a live generation",
            backend_text:        null,
        )~;
    }
    SamplerDescriptorCell* cell = sampler_cell_get(&state.heap, index)!!;
    cell.used = false;
    cell.generation = bump_generation(cell.generation);
    state.heap.sampler_retire.push((uint)index.slot, retire_timeline_value(state));
}
