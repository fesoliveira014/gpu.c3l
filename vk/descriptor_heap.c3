module gpu::vk;

import gpu;
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
    vk::DeviceSize max_resource_descriptor_buffer_range;
    vk::DeviceSize max_sampler_descriptor_buffer_range;
    vk::DeviceSize resource_descriptor_address_space_size;
    vk::DeviceSize sampler_descriptor_address_space_size;
    vk::DeviceSize descriptor_buffer_address_space_size;
    vk::DeviceSize layout_size;
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

fn ushort bump_generation(ushort generation) @private {
    uint next = ((uint)generation + 1) & 0xFFFF;
    return next == 0 ? (ushort)1 : (ushort)next;
}

<* Texture capacity binds both SAMPLED_IMAGE and STORAGE_IMAGE, so on the
   indexing path it must satisfy both image limits. On the buffer path the
   bound quantity is the descriptor set's byte layout, not slot counts. *>
fn bool heap_fits_limits(gpu::DescriptorHeapMode mode, uint tex_capacity, uint smp_capacity, HeapLimits limits) {
    if (mode == gpu::DescriptorHeapMode.DESCRIPTOR_INDEXING) {
        // Bindings use ALL stages, so per-stage limits bind alongside the
        // set-level ones.
        return tex_capacity <= limits.max_sampled_images
            && tex_capacity <= limits.max_storage_images
            && tex_capacity <= limits.max_per_stage_sampled_images
            && tex_capacity <= limits.max_per_stage_storage_images
            && smp_capacity <= limits.max_samplers
            && smp_capacity <= limits.max_per_stage_samplers;
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
        };
        bool fits = heap_fits_limits(
            mode:         state.heap_mode,
            tex_capacity: tex_capacity,
            smp_capacity: smp_capacity,
            limits:       limits,
        );
        if (!fits) {
            io::eprintfn(
                "gpu: descriptor capacities (tex %d, smp %d) exceed a device update-after-bind limit (set %d/%d/%d, per-stage %d/%d/%d)",
                tex_capacity, smp_capacity,
                limits.max_sampled_images, limits.max_storage_images, limits.max_samplers,
                limits.max_per_stage_sampled_images, limits.max_per_stage_storage_images, limits.max_per_stage_samplers);
            return gpu::INVALID_ARGUMENT~;
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
    check(vk::create_descriptor_set_layout(state.device, &layout_info, null, &heap.set_layout))!;

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
        check(vk::create_descriptor_pool(state.device, &pool_info, null, &heap.descriptor_pool))!;

        vk::DescriptorSetAllocateInfo alloc_info = vk::descriptor_set_allocate_info()
            .set_descriptor_pool(heap.descriptor_pool)
            .set_set_layouts((&heap.set_layout)[:1]);
        check(vk::allocate_descriptor_sets(state.device, &alloc_info, &heap.descriptor_set))!;
    } else {
        create_descriptor_buffer(state)!;
    }
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
        io::eprintfn(
            "gpu: descriptor buffer layout size %d exceeds a device descriptor-buffer bound (ranges %d/%d, address space %d/%d/%d)",
            layout_size,
            limits.max_resource_descriptor_buffer_range, limits.max_sampler_descriptor_buffer_range,
            limits.resource_descriptor_address_space_size, limits.sampler_descriptor_address_space_size,
            limits.descriptor_buffer_address_space_size);
        return gpu::INVALID_ARGUMENT~;
    }

    vk::extensions.vk_get_descriptor_set_layout_binding_offset_ext(state.device, heap.set_layout, HEAP_BINDING_SAMPLED, &heap.binding_offset_sampled);
    vk::extensions.vk_get_descriptor_set_layout_binding_offset_ext(state.device, heap.set_layout, HEAP_BINDING_STORAGE, &heap.binding_offset_storage);
    vk::extensions.vk_get_descriptor_set_layout_binding_offset_ext(state.device, heap.set_layout, HEAP_BINDING_SAMPLERS, &heap.binding_offset_samplers);

    uint usage = (uint)vk::BufferUsageFlagBits.RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT
        | (uint)vk::BufferUsageFlagBits.SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT
        | (uint)vk::BufferUsageFlagBits.SHADER_DEVICE_ADDRESS_BIT;
    vk::BufferCreateInfo buffer_info = vk::buffer_create_info()
        .set_size(layout_size)
        .set_usage((vk::BufferUsageFlags)usage);
    // Descriptor bytes are written mapped (vkGetDescriptorEXT) with no
    // flush hook — require coherent memory.
    vma::AllocationCreateInfo alloc_info = {
        .usage          = vma::MemoryUsage.AUTO_PREFER_HOST,
        .flags          = { .host_access_sequential_write, .mapped },
        .required_flags = (vk::MemoryPropertyFlags)vk::MemoryPropertyFlagBits.HOST_COHERENT_BIT,
    };
    vma::AllocationInfo allocation_info;
    check(state.allocator.create_buffer(&buffer_info, &alloc_info, &heap.descriptor_buffer, &heap.descriptor_buffer_allocation, &allocation_info))!;
    heap.descriptor_buffer_cpu = allocation_info.mapped_data;

    vk::BufferDeviceAddressInfo address_info = vk::buffer_device_address_info().set_buffer(heap.descriptor_buffer);
    heap.descriptor_buffer_gpu = vk::get_buffer_device_address(state.device, &address_info);
    if (heap.descriptor_buffer_gpu == 0) return gpu::DEVICE_LOST~;
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
            if (cell.used) {
                io::eprintfn("gpu: leaked texture descriptor slot %d (owner texture index %d)", i, cell.owner.index);
            }
        }
    }
    if (heap.sampler_cells.ptr != null) {
        foreach (i, &cell : heap.sampler_cells[:heap.sampler_next_new]) {
            if (cell.used) io::eprintfn("gpu: leaked sampler descriptor slot %d", i);
        }
    }
}

fn void drain_texture_retire(VkDeviceState* state) @private {
    DescriptorHeapState* heap = &state.heap;
    if (heap.texture_retire.count == 0) return;
    ulong current = 0;
    vk::get_semaphore_counter_value(state.device, state.frame_timeline, &current);
    while (heap.texture_retire.count > 0) {
        DescriptorRetireEntry* entry = &heap.texture_retire.entries[heap.texture_retire.head];
        if (entry.timeline_value > current) break;
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

fn gpu::TextureIndex? vk_create_texture_descriptor(gpu::Device* device, gpu::TextureHandle texture, gpu::TextureViewDesc* view) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    DescriptorHeapState* heap = &state.heap;
    TextureSlot* texture_slot = state.textures.get(texture)!;
    // Descriptors only make sense for shader-visible textures; anything else
    // would return an index whose heap slots are never written.
    if (!texture_slot.usage.sampled && !texture_slot.usage.storage) return gpu::INVALID_ARGUMENT~;
    vk::ImageView image_view = resolve_texture_view(
        state:   state,
        slot:    texture_slot,
        view:    view,
        created: null,
    )!;

    drain_texture_retire(state);
    uint slot;
    if (heap.texture_free_count > 0) {
        slot = heap.texture_free[--heap.texture_free_count];
    } else if (heap.texture_next_new < heap.texture_cells.len) {
        slot = heap.texture_next_new++;
    } else {
        return gpu::DESCRIPTOR_HEAP_FULL~;
    }

    TextureDescriptorCell* cell = &heap.texture_cells[slot];
    cell.owner = texture;
    cell.used = true;
    cell.generation = bump_generation(cell.generation);

    write_texture_descriptor(state, slot, texture_slot, image_view);
    return (gpu::TextureIndex){ .slot = (ushort)slot, .generation = cell.generation };
}

<* One batch item's validate+publish: mirrors vk_create_texture_descriptor's
   per-item steps without its own drain or lock (the caller holds both once
   for the whole batch). Records the pending write info the tail needs. *>
fn void? publish_texture_descriptor_item(VkDeviceState* state, gpu::TextureDescriptorDesc* desc, gpu::TextureIndex* out_index, PendingTextureWrite* pending) @private {
    DescriptorHeapState* heap = &state.heap;
    TextureSlot* texture_slot = state.textures.get(desc.texture)!;
    if (!texture_slot.usage.sampled && !texture_slot.usage.storage) return gpu::INVALID_ARGUMENT~;
    vk::ImageView image_view = resolve_texture_view(
        state:   state,
        slot:    texture_slot,
        view:    &desc.view,
        created: null,
    )!;

    uint slot;
    if (heap.texture_free_count > 0) {
        slot = heap.texture_free[--heap.texture_free_count];
    } else if (heap.texture_next_new < heap.texture_cells.len) {
        slot = heap.texture_next_new++;
    } else {
        return gpu::DESCRIPTOR_HEAP_FULL~;
    }

    TextureDescriptorCell* cell = &heap.texture_cells[slot];
    cell.owner = desc.texture;
    cell.used = true;
    cell.generation = bump_generation(cell.generation);

    *out_index = { .slot = (ushort)slot, .generation = cell.generation };
    *pending = { .texture_slot = texture_slot, .view = image_view };
}

<* Generation bump + used=false + free-list push per published item; no
   retire-ring involvement — the cells never became visible. *>
fn void rollback_texture_descriptors(DescriptorHeapState* heap, gpu::TextureIndex[] published) @private {
    foreach (&index : published) {
        TextureDescriptorCell* cell = &heap.texture_cells[index.slot];
        cell.used = false;
        cell.generation = bump_generation(cell.generation);
        heap.texture_free[heap.texture_free_count++] = index.slot;
    }
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

<* All-or-nothing: the pre-flight leaves the heap untouched, and descriptor
   writes run only after every item publishes, so rollback never has to
   unwrite one. *>
fn void? vk_create_texture_descriptors(gpu::Device* device, gpu::TextureDescriptorDesc[] descs, gpu::TextureIndex[] out_indices) @private {
    if (out_indices.len != descs.len) return gpu::INVALID_ARGUMENT~;
    if (descs.len == 0) return;

    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    DescriptorHeapState* heap = &state.heap;

    drain_texture_retire(state);
    uint headroom = heap.texture_free_count + (uint)heap.texture_cells.len - heap.texture_next_new;
    if ((uint)descs.len > headroom) return gpu::DESCRIPTOR_HEAP_FULL~;

    @pool() {
        PendingTextureWrite[] pending = mem::talloc_array(PendingTextureWrite, descs.len);
        for (uint i = 0; i < descs.len; i++) {
            if (catch f = publish_texture_descriptor_item(state, &descs[i], &out_indices[i], &pending[i])) {
                rollback_texture_descriptors(heap, out_indices[:i]);
                return f~;
            }
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

<* Whitebox: a copy of a texture-descriptor cell's bookkeeping state, for
   rollback-invariant tests. *>
fn TextureDescriptorCell texture_descriptor_cell(VkDeviceState* state, ushort slot) {
    return state.heap.texture_cells[slot];
}

fn void? vk_destroy_texture_descriptor(gpu::Device* device, gpu::TextureIndex index) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    TextureDescriptorCell* cell = texture_cell_get(&state.heap, index)!;
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
    if (desc.anisotropy_enable && state.max_sampler_anisotropy <= 0) return gpu::UNSUPPORTED_FEATURE~;
    // Vulkan requires 1.0 <= maxAnisotropy when enabled; enabling without a
    // valid value is a caller error, not a default request.
    if (desc.anisotropy_enable && desc.max_anisotropy < 1.0f) return gpu::INVALID_ARGUMENT~;
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
    check(vk::create_sampler(state.device, &sampler_info, null, &sampler))!;

    if (try index = publish_sampler(state, sampler)) {
        return index;
    }
    vk::destroy_sampler(state.device, sampler, null);
    return gpu::DESCRIPTOR_HEAP_FULL~;
}

fn void? vk_destroy_sampler(gpu::Device* device, gpu::SamplerIndex index) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    SamplerDescriptorCell* cell = sampler_cell_get(&state.heap, index)!;
    cell.used = false;
    cell.generation = bump_generation(cell.generation);
    state.heap.sampler_retire.push((uint)index.slot, retire_timeline_value(state));
}
