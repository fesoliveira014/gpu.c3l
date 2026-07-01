module gpu::vk;

import gpu;
import vk;
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

<* Create a shader module. SPIR-V arrives as a byte slice with no alignment
   guarantee; it is copied into a 4-byte-aligned word buffer for the uint* code
   pointer Vulkan requires. *>
fn gpu::ShaderHandle? vk_create_shader(gpu::Device* device, gpu::ShaderDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    if (desc.spirv.len == 0 || desc.spirv.len % 4 != 0) return gpu::SHADER_INVALID~;

    vk::ShaderModule sh_module;
    @pool() {
        uint[] code = mem::talloc_array(uint, desc.spirv.len / 4);
        mem::copy(code.ptr, desc.spirv.ptr, desc.spirv.len);
        vk::ShaderModuleCreateInfo info = vk::shader_module_create_info()
            .set_code_size(desc.spirv.len)
            .set_code(code.ptr);
        check(vk::create_shader_module(state.device, &info, null, &sh_module))!;
    };

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

fn void? vk_destroy_shader(gpu::Device* device, gpu::ShaderHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)device.backend_state;
    ShaderSlot* slot = state.shaders.get(handle)!;
    vk::destroy_shader_module(state.device, slot.shader_module, null);
    state.shaders.free(handle)!;
}
