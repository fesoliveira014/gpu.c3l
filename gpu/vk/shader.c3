module gpu::vk @private;

import gpu @public;
import vk;
import vma;
import std::thread;
import spvreflect;
import std::core::mem;

// Concrete (non-generic) slot table for shaders. Monomorphized twin of
// gpu::slots to sidestep the c3c 0.8.0 generic-instantiation codegen crash.
const uint SHADER_FREE_NONE = 0;
const usz SHADER_ENTRY_POINT_CAP = 64;

struct ShaderSlot {
    vk::ShaderModule shader_module;
    gpu::ShaderStage stage;
    char[SHADER_ENTRY_POINT_CAP] entry_point;
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
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation);
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
    self.slots[idx].generation = next_handle_generation(self.slots[idx].generation);
    self.slots[idx].next_free = self.free_head;
    self.free_head = idx + 1;
}

<* Fault SHADER_INVALID on any set other than 0 or a set-0 binding that breaks
   the heap convention; buffers-only shaders pass. *>
fn void? check_heap_convention(spvreflect::ShaderModule* reflect_module) @private {
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

<* Unsupported execution models fault INVALID_ARGUMENT. *>
fn gpu::ShaderStage? gpu_stage_from_spv(spvreflect::ShaderStage stage) @private {
    switch (stage) {
        case spvreflect::ShaderStage.SHADER_STAGE_COMPUTE:  return gpu::ShaderStage.COMPUTE;
        case spvreflect::ShaderStage.SHADER_STAGE_VERTEX:   return gpu::ShaderStage.VERTEX;
        case spvreflect::ShaderStage.SHADER_STAGE_FRAGMENT: return gpu::ShaderStage.FRAGMENT;
        default: return gpu::INVALID_ARGUMENT~;
    }
}

<* Push blocks must fit the stage's root-push size; zero blocks are valid. *>
fn void? check_push_constant_range(spvreflect::ShaderModule* reflect_module, gpu::ShaderStage stage) @private {
    usz expected_size = stage == gpu::ShaderStage.COMPUTE ? gpu::RootPush::size : gpu::GraphicsRootPush::size;

    uint count = 0;
    if (reflect_module.enumerate_push_constant_blocks(&count, null) != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return gpu::SHADER_INVALID~;
    }
    if (count == 0) return;

    @pool() {
        spvreflect::BlockVariable*[] blocks = mem::talloc_array(spvreflect::BlockVariable*, (sz)count);
        if (reflect_module.enumerate_push_constant_blocks(&count, blocks.ptr) != spvreflect::ReflectResult.RESULT_SUCCESS) {
            return gpu::SHADER_INVALID~;
        }
        foreach (block : blocks) {
            if ((usz)block.byte_offset() + (usz)block.byte_size() > expected_size) return gpu::INVALID_ARGUMENT~;
        }
    };
}

<* Truncating copy is unsafe for entry points (unlike debug names): a name
   that doesn't fit SHADER_ENTRY_POINT_CAP faults INVALID_ARGUMENT instead of
   being silently shortened into a different, possibly valid, symbol. Null
   defaults to "main". *>
fn void? copy_entry_point(char* dst, ZString name) @private {
    ZString source = name == null ? (ZString)"main" : name;
    usz i = 0;
    while (source[i] != 0) {
        if (i >= SHADER_ENTRY_POINT_CAP - 1) return gpu::INVALID_ARGUMENT~;
        dst[i] = source[i];
        i++;
    }
    dst[i] = 0;
}

<* SPIR-V arrives with no alignment guarantee; it is copied into a
   4-byte-aligned word buffer for the uint* code pointer Vulkan requires. *>
fn gpu::ShaderHandle? vk_create_shader(gpu::Device* device, gpu::ShaderDesc* desc) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    if (desc.spirv.len == 0 || desc.spirv.len % 4 != 0) {
        return report_contract_failure(
            state:               state,
            operation:           "create_shader",
            public_fault:        gpu::SHADER_INVALID,
            resource_kind:       gpu::DebugResourceKind.SHADER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "ShaderDesc.spirv",
            invariant:           "SPIR-V must be non-empty and four-byte sized",
            backend_text:        null,
        )~;
    }

    ShaderSlot slot = { .stage = desc.stage };
    if (catch err = copy_entry_point(&slot.entry_point[0], desc.entry_point)) {
        return report_contract_failure(
            state:               state,
            operation:           "create_shader",
            public_fault:        err,
            resource_kind:       gpu::DebugResourceKind.SHADER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "ShaderDesc.entry_point",
            invariant:           "entry point must fit the backend name capacity",
            backend_text:        null,
        )~;
    }

    // ulong storage keeps the opaque reflect blob 8-byte aligned.
    ulong[spvreflect::SHADER_MODULE_SIZE / 8] module_storage;
    spvreflect::ShaderModule* reflect_module = (spvreflect::ShaderModule*)&module_storage;
    if (spvreflect::create_shader_module(desc.spirv.len, desc.spirv.ptr, reflect_module) != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return report_contract_failure(
            state:               state,
            operation:           "create_shader",
            public_fault:        gpu::SHADER_INVALID,
            resource_kind:       gpu::DebugResourceKind.SHADER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "ShaderDesc.spirv",
            invariant:           "SPIR-V reflection must succeed",
            backend_text:        null,
        )~;
    }
    defer reflect_module.destroy();
    if (catch err = check_heap_convention(reflect_module)) {
        return report_contract_failure(
            state:               state,
            operation:           "create_shader",
            public_fault:        err,
            resource_kind:       gpu::DebugResourceKind.SHADER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "ShaderDesc.spirv",
            invariant:           "descriptor bindings must match the gpu heap convention",
            backend_text:        null,
        )~;
    }
    spvreflect::EntryPoint* entry = reflect_module.get_entry_point((ZString)&slot.entry_point[0]);
    if (entry == null) {
        return report_contract_failure(
            state:               state,
            operation:           "create_shader",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SHADER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "ShaderDesc.entry_point",
            invariant:           "entry point must exist in SPIR-V module",
            backend_text:        null,
        )~;
    }
    gpu::ShaderStage? reflected_stage_optional = gpu_stage_from_spv(entry.stage());
    if (catch reflected_fault = reflected_stage_optional) {
        return report_contract_failure(
            state:               state,
            operation:           "create_shader",
            public_fault:        reflected_fault,
            resource_kind:       gpu::DebugResourceKind.SHADER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "ShaderDesc.stage",
            invariant:           "reflected execution model must be supported",
            backend_text:        null,
        )~;
    }
    gpu::ShaderStage reflected_stage = reflected_stage_optional;
    if (reflected_stage != desc.stage) {
        return report_contract_failure(
            state:               state,
            operation:           "create_shader",
            public_fault:        gpu::INVALID_ARGUMENT,
            resource_kind:       gpu::DebugResourceKind.SHADER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "ShaderDesc.stage",
            invariant:           "stage must match the reflected entry point",
            backend_text:        null,
        )~;
    }
    if (catch err = check_push_constant_range(reflect_module, desc.stage)) {
        return report_contract_failure(
            state:               state,
            operation:           "create_shader",
            public_fault:        err,
            resource_kind:       gpu::DebugResourceKind.SHADER,
            has_public_identity: false,
            index:               0,
            generation:          0,
            debug_name:          desc.debug_name,
            rejected_field:      "ShaderDesc.spirv",
            invariant:           "push constants must fit the shader-stage root ABI",
            backend_text:        null,
        )~;
    }

    @pool() {
        uint[] code = mem::talloc_array(uint, desc.spirv.len / 4);
        mem::copy(code.ptr, desc.spirv.ptr, desc.spirv.len);
        vk::ShaderModuleCreateInfo info = vk::shader_module_create_info()
            .set_code_size(desc.spirv.len)
            .set_code(code.ptr);
        check_shader_create_device_result(
            state: state,
            result: vk::create_shader_module(
                device:          state.device,
                p_create_info:   &info,
                p_allocator:     null,
                p_shader_module: &slot.shader_module,
            ),
        )!;
    };
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.SHADER_MODULE,
        object_handle: (ulong)slot.shader_module,
        name:          desc.debug_name,
    );

    {
        state.resource_mutex.lock()!!;
        defer state.resource_mutex.unlock();
        if (try handle = state.shaders.alloc(slot)) {
            return handle;
        }
    }
    vk::destroy_shader_module(state.device, slot.shader_module, null);
    return gpu::SLOT_TABLE_FULL~;
}

<* Frees the handle now; module release is deferred until frames referencing
   its pipelines retire. *>
fn void? vk_destroy_shader(gpu::Device* device, gpu::ShaderHandle handle) @private {
    VkDeviceState* state = (VkDeviceState*)gpu::device_backend_state_ptr(device);
    enter_tracked(state);
    defer exit_tracked(state);
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    ShaderSlot* slot = state.shaders.get(handle)!;
    vk::ShaderModule sh_module = slot.shader_module;
    ulong retire = retire_timeline_value(state);
    state.shaders.free(handle)!;
    enqueue_deferred(
        state:          state,
        kind:           DeferredKind.SHADER_MODULE,
        timeline_value: retire,
        object:         (void*)sh_module,
        allocation:     (vma::Allocation)null,
    );
}
