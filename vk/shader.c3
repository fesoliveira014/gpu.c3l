module gpu::vk;

import gpu;
import vk;
import vma;
import std::thread;
import spvreflect;
import std::core::mem;

// Concrete (non-generic) slot table for shaders. Monomorphized twin of
// gpu::slots to sidestep the c3c 0.8.0 generic-instantiation codegen crash.
const uint SHADER_FREE_NONE = 0;

struct ShaderSlot {
    vk::ShaderModule shader_module;
    gpu::ShaderStage stage;
    ZString          entry_point;
}

struct ShaderCell {
    ShaderSlot value;
    uint       generation;
    uint       next_free;
    bool       used;
}

struct ShaderTable {
    ShaderCell[] slots;
    uint         free_head;
    uint         next_new;
}

fn gpu::ShaderHandle? ShaderTable.alloc(&self, ShaderSlot value) {
    uint idx;
    if (self.free_head != SHADER_FREE_NONE) {
        idx = self.free_head - 1;
        self.free_head = self.slots[idx].next_free;
    } else if (self.next_new < self.slots.len) {
        idx = self.next_new++;
        self.slots[idx].generation = 0;
    } else {
        return gpu::SLOT_TABLE_FULL~;
    }
    self.slots[idx].value = value;
    self.slots[idx].generation++;
    self.slots[idx].used = true;
    return (gpu::ShaderHandle){ .index = idx, .generation = self.slots[idx].generation };
}

fn ShaderSlot*? ShaderTable.get(&self, gpu::ShaderHandle h) {
    uint idx = h.index;
    if (idx >= self.slots.len || !self.slots[idx].used) return gpu::INVALID_HANDLE~;
    if (self.slots[idx].generation != h.generation) return gpu::INVALID_HANDLE~;
    return &self.slots[idx].value;
}

fn void? ShaderTable.free(&self, gpu::ShaderHandle h) {
    self.get(h)!;
    uint idx = h.index;
    self.slots[idx].used = false;
    self.slots[idx].generation++;
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

<* Fault SHADER_INVALID on any set other than 0 or a set-0 binding that breaks
   the heap convention; buffers-only shaders pass. *>
fn void? check_shader_heap_convention(char[] spirv) @private {
    // ulong storage keeps the opaque reflect blob 8-byte aligned.
    ulong[spvreflect::SHADER_MODULE_SIZE / 8] module_storage;
    spvreflect::ShaderModule* reflect_module = (spvreflect::ShaderModule*)&module_storage;
    if (spvreflect::create_shader_module(spirv.len, spirv.ptr, reflect_module) != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return gpu::SHADER_INVALID~;
    }
    defer reflect_module.destroy();

    uint count = 0;
    if (reflect_module.enumerate_descriptor_bindings(&count, null) != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return gpu::SHADER_INVALID~;
    }
    if (count == 0) return;

    @pool() {
        spvreflect::DescriptorBinding*[] bindings = mem::talloc_array(spvreflect::DescriptorBinding*, (sz)count);
        if (reflect_module.enumerate_descriptor_bindings(&count, bindings.ptr) != spvreflect::ReflectResult.RESULT_SUCCESS) {
            return gpu::SHADER_INVALID~;
        }
        foreach (binding : bindings) {
            if (binding.set_index() != HEAP_SET) return gpu::SHADER_INVALID~;
            int descriptor_type = (int)binding.descriptor_type();
            switch (binding.binding_index()) {
                case HEAP_BINDING_SAMPLED:
                    if (descriptor_type != (int)vk::DescriptorType.SAMPLED_IMAGE) return gpu::SHADER_INVALID~;
                case HEAP_BINDING_STORAGE:
                    if (descriptor_type != (int)vk::DescriptorType.STORAGE_IMAGE) return gpu::SHADER_INVALID~;
                case HEAP_BINDING_SAMPLERS:
                    if (descriptor_type != (int)vk::DescriptorType.SAMPLER) return gpu::SHADER_INVALID~;
                default:
                    return gpu::SHADER_INVALID~;
            }
        }
    };
}

<* SPIR-V arrives with no alignment guarantee; it is copied into a
   4-byte-aligned word buffer for the uint* code pointer Vulkan requires. *>
fn gpu::ShaderHandle? vk_create_shader(gpu::Device* device, gpu::ShaderDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if (desc.spirv.len == 0 || desc.spirv.len % 4 != 0) return gpu::SHADER_INVALID~;
    check_shader_heap_convention(desc.spirv)!;

    vk::ShaderModule sh_module;
    @pool() {
        uint[] code = mem::talloc_array(uint, desc.spirv.len / 4);
        mem::copy(code.ptr, desc.spirv.ptr, desc.spirv.len);
        vk::ShaderModuleCreateInfo info = vk::shader_module_create_info()
            .set_code_size(desc.spirv.len)
            .set_code(code.ptr);
        check(vk::create_shader_module(state.device, &info, null, &sh_module))!;
    };
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.SHADER_MODULE,
        object_handle: (ulong)sh_module,
        name:          desc.debug_name,
    );

    ShaderSlot slot = {
        .shader_module = sh_module,
        .stage         = desc.stage,
        .entry_point   = desc.entry_point == null ? (ZString)"main" : desc.entry_point,
    };
    if (try handle = state.shaders.alloc(slot)) {
        return handle;
    }
    vk::destroy_shader_module(state.device, sh_module, null);
    return gpu::SLOT_TABLE_FULL~;
}

<* Frees the handle now; module release is deferred until frames referencing
   its pipelines retire. *>
fn void? vk_destroy_shader(gpu::Device* device, gpu::ShaderHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    ShaderSlot* slot = state.shaders.get(handle)!;
    vk::ShaderModule sh_module = slot.shader_module;
    ulong retire = retire_timeline_value(state);
    state.shaders.free(handle)!;
    enqueue_deferred(state, DeferredKind.SHADER_MODULE, retire, (void*)sh_module, (vma::Allocation)null);
}
