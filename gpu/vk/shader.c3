module gpu::vk @private;

import gpu @public;
import vk;
import spvreflect;
import std::core::mem;
import std::core::mem::alloc;
import std::core::env;

typedef ShaderId = uint;

const ShaderId SHADER_ID_INVALID = (ShaderId)uint::max;
const uint SHADER_FREE_NONE = 0;
const uint SHADER_STORE_MAX_PIPELINE_CAPACITY = 0x20000000;

struct ShaderStoreEntry {
    ulong                             hash;
    char[]                            spirv;
    char[gpu::SHADER_ENTRY_POINT_CAP] entry_point;
    gpu::ShaderStage                  stage;
    uint                              refcount;
    uint                              next_free;
    uint                              next_in_bucket;
    bool                              used;
}

struct ShaderStore {
    ShaderStoreEntry[] entries;
    uint[]             bucket_heads;
    uint               free_head;
    uint               next_new;
}

struct PreparedShaderIdentity {
    ShaderId id;
    bool     retained;
}

struct PipelineIdentityStats @if(env::TESTING || $feature(PIPELINE_CACHE_STATS)) {
    ulong shader_intern_probes;
    ulong shader_intern_bytes_compared;
    ulong shader_owned_bytes_cloned;
    ulong shader_owned_bytes_freed;
    ulong pipeline_key_probes;
}

struct PipelineIdentityProbeResult @if(env::TESTING || $feature(PIPELINE_CACHE_STATS)) {
    PipelineIdentityStats interning;
    PipelineIdentityStats lookup;
}

struct PipelineShaderModule {
    vk::ShaderModule shader_module;
    char[gpu::SHADER_ENTRY_POINT_CAP] entry_point;
}

struct PipelineBatchShader {
    ShaderId             id;
    PipelineShaderModule native;
}

fn void note_shader_intern_probe(VkDeviceState* state) {
    $if env::TESTING || $feature(PIPELINE_CACHE_STATS):
        state.shader_intern_probes++;
    $endif
}

fn void note_shader_intern_bytes_compared(
    VkDeviceState* state,
    usz byte_count,
) {
    $if env::TESTING || $feature(PIPELINE_CACHE_STATS):
        state.shader_intern_bytes_compared += byte_count;
    $endif
}

fn void note_shader_owned_bytes_cloned(
    VkDeviceState* state,
    usz byte_count,
) {
    $if env::TESTING || $feature(PIPELINE_CACHE_STATS):
        state.shader_owned_bytes_cloned += byte_count;
    $endif
}

fn void note_shader_owned_bytes_freed(
    VkDeviceState* state,
    usz byte_count,
) {
    $if env::TESTING || $feature(PIPELINE_CACHE_STATS):
        state.shader_owned_bytes_freed += byte_count;
    $endif
}

fn void note_pipeline_key_probe(VkDeviceState* state) {
    $if env::TESTING || $feature(PIPELINE_CACHE_STATS):
        state.pipeline_key_probes++;
    $endif
}

fn void reset_pipeline_identity_stats(
    VkDeviceState* state,
) @if(env::TESTING || $feature(PIPELINE_CACHE_STATS)) {
    state.shader_intern_probes = 0;
    state.shader_intern_bytes_compared = 0;
    state.shader_owned_bytes_cloned = 0;
    state.shader_owned_bytes_freed = 0;
    state.pipeline_key_probes = 0;
}

fn PipelineIdentityStats pipeline_identity_stats(
    VkDeviceState* state,
) @if(env::TESTING || $feature(PIPELINE_CACHE_STATS)) {
    return {
        .shader_intern_probes          = state.shader_intern_probes,
        .shader_intern_bytes_compared  = state.shader_intern_bytes_compared,
        .shader_owned_bytes_cloned     = state.shader_owned_bytes_cloned,
        .shader_owned_bytes_freed      = state.shader_owned_bytes_freed,
        .pipeline_key_probes           = state.pipeline_key_probes,
    };
}

fn uint shader_store_bucket(ulong hash, uint bucket_count) {
    return (uint)((hash ^ (hash >> 32)) & (ulong)(bucket_count - 1));
}

fn void link_shader_store_entry(ShaderStore* store, uint index) {
    uint bucket = shader_store_bucket(
        store.entries[index].hash,
        (uint)store.bucket_heads.len,
    );
    store.entries[index].next_in_bucket = store.bucket_heads[bucket];
    store.bucket_heads[bucket] = index + 1;
}

fn bool unlink_shader_store_entry(ShaderStore* store, uint index) {
    uint bucket = shader_store_bucket(
        store.entries[index].hash,
        (uint)store.bucket_heads.len,
    );
    uint* current = &store.bucket_heads[bucket];
    while (*current != index + 1) {
        if (*current == 0) return false;
        current = &store.entries[*current - 1].next_in_bucket;
    }
    *current = store.entries[index].next_in_bucket;
    return true;
}

fn bool shader_store_entry_equals(
    VkDeviceState* state,
    ShaderStoreEntry* entry,
    gpu::ShaderCode* code,
) {
    if (entry.stage != code.stage
        || entry.spirv.len != code.spirv.len
        || ((ZString)&entry.entry_point[0]).str_view()
            != code.entry_point.str_view()) return false;
    note_shader_intern_bytes_compared(state, code.spirv.len);
    return mem::equals(entry.spirv, code.spirv);
}

fn ShaderId? retain_shader_identity_locked(
    VkDeviceState* state,
    gpu::ShaderCode* code,
) {
    if (code == null || !code.is_valid()) return gpu::INVALID_ARGUMENT~;
    ShaderStore* store = &state.shader_store;
    uint bucket = shader_store_bucket(
        code.digest,
        (uint)store.bucket_heads.len,
    );
    for (uint current = store.bucket_heads[bucket]; current != 0; ) {
        note_shader_intern_probe(state);
        ShaderStoreEntry* entry = &store.entries[current - 1];
        if (entry.hash == code.digest
            && shader_store_entry_equals(state, entry, code)) {
            if (entry.refcount == uint::max) return gpu::SLOT_TABLE_FULL~;
            entry.refcount++;
            return (ShaderId)(current - 1);
        }
        current = entry.next_in_bucket;
    }

    uint index;
    if (store.free_head != SHADER_FREE_NONE) {
        index = store.free_head - 1;
        store.free_head = store.entries[index].next_free;
    } else if (store.next_new < store.entries.len) {
        index = store.next_new++;
    } else {
        return gpu::SLOT_TABLE_FULL~;
    }

    char[] spirv = alloc::new_array(
        state.host_allocator,
        char,
        (sz)code.spirv.len,
    );
    mem::copy(spirv.ptr, code.spirv.ptr, code.spirv.len);
    note_shader_owned_bytes_cloned(state, code.spirv.len);
    ShaderStoreEntry entry = {
        .hash     = code.digest,
        .spirv    = spirv,
        .stage    = code.stage,
        .refcount = 1,
        .used     = true,
    };
    String entry_point = code.entry_point.str_view();
    mem::copy(
        &entry.entry_point[0],
        entry_point.ptr,
        entry_point.len,
    );
    entry.entry_point[entry_point.len] = 0;
    store.entries[index] = entry;
    link_shader_store_entry(store, index);
    return (ShaderId)index;
}

fn void? release_shader_identity_locked(
    VkDeviceState* state,
    ShaderId id,
) {
    uint index = (uint)id;
    if (index >= state.shader_store.next_new) return gpu::INVALID_ARGUMENT~;
    ShaderStoreEntry* entry = &state.shader_store.entries[index];
    if (!entry.used || entry.refcount == 0) return gpu::INVALID_ARGUMENT~;
    entry.refcount--;
    if (entry.refcount > 0) return;
    if (!unlink_shader_store_entry(&state.shader_store, index)) {
        entry.refcount++;
        return gpu::INVALID_ARGUMENT~;
    }
    note_shader_owned_bytes_freed(state, entry.spirv.len);
    alloc::free(state.host_allocator, entry.spirv.ptr);
    *entry = { .next_free = state.shader_store.free_head };
    state.shader_store.free_head = index + 1;
}

fn PreparedShaderIdentity? prepare_shader_identity_locked(
    VkDeviceState* state,
    gpu::ShaderCode* code,
) {
    return {
        .id       = retain_shader_identity_locked(state, code)!,
        .retained = true,
    };
}

fn void? release_prepared_shader_locked(
    VkDeviceState* state,
    PreparedShaderIdentity* prepared,
) {
    if (!prepared.retained) return;
    release_shader_identity_locked(state, prepared.id)!;
    prepared.retained = false;
}

fn void transfer_prepared_shader(PreparedShaderIdentity* prepared) {
    prepared.retained = false;
}

fn void release_prepared_shaders(
    VkDeviceState* state,
    PreparedShaderIdentity[] prepared,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    foreach (&identity : prepared) {
        release_prepared_shader_locked(state, identity)!!;
    }
}

fn void poison_unprepared_shader_identities(
    VkDeviceState* state,
    PreparedShaderIdentity[] prepared,
) @if(env::TESTING) {
    if (!state.pipeline_prepare_poison_enabled) return;
    foreach (&identity : prepared) {
        *identity = {
            .id       = state.pipeline_prepare_poison_id,
            .retained = true,
        };
    }
}

fn void set_pipeline_prepare_poison(
    VkDeviceState* state,
    ShaderId id,
) @if(env::TESTING) {
    state.pipeline_prepare_poison_id = id;
    state.pipeline_prepare_poison_enabled = true;
}

fn void clear_pipeline_prepare_poison(
    VkDeviceState* state,
) @if(env::TESTING) {
    state.pipeline_prepare_poison_enabled = false;
    state.pipeline_prepare_poison_id = SHADER_ID_INVALID;
}

fn gpu::ShaderCode shader_store_code(
    ShaderStoreEntry* entry,
    ZString debug_name,
) {
    return {
        .stage       = entry.stage,
        .spirv       = entry.spirv,
        .entry_point = (ZString)&entry.entry_point[0],
        .debug_name  = debug_name,
        .digest      = entry.hash,
    };
}

<* The caller retains `id` for the lifetime of the returned borrowed view. *>
fn gpu::ShaderCode shader_store_code_for_retained_id(
    VkDeviceState* state,
    ShaderId id,
    ZString debug_name,
) {
    return shader_store_code(&state.shader_store.entries[(uint)id], debug_name);
}

fn uint shader_store_live_count(VkDeviceState* state) {
    uint count = 0;
    foreach (&entry : state.shader_store.entries[:state.shader_store.next_new]) {
        if (entry.used) count++;
    }
    return count;
}

fn bool shader_store_index_consistent(VkDeviceState* state) {
    ShaderStore* store = &state.shader_store;
    uint linked = 0;
    foreach (bucket, head : store.bucket_heads) {
        for (uint current = head; current != 0; ) {
            ShaderStoreEntry* entry = &store.entries[current - 1];
            if (!entry.used
                || shader_store_bucket(entry.hash, (uint)store.bucket_heads.len)
                    != (uint)bucket) return false;
            linked++;
            current = entry.next_in_bucket;
        }
    }
    return linked == shader_store_live_count(state);
}

fn void? create_shader_store(VkDeviceState* state, uint pipeline_capacity) {
    if (pipeline_capacity == 0
        || pipeline_capacity > SHADER_STORE_MAX_PIPELINE_CAPACITY) {
        return gpu::INVALID_ARGUMENT~;
    }
    uint entry_capacity = pipeline_capacity * 2;
    state.shader_store.entries = mem::new_array(
        ShaderStoreEntry,
        entry_capacity,
    );
    state.shader_store.bucket_heads = mem::new_array(
        uint,
        next_pow2(entry_capacity * 2),
    );
}

fn void destroy_shader_store(VkDeviceState* state) {
    if (state.shader_store.entries.ptr != null) {
        foreach (&entry : state.shader_store.entries[:state.shader_store.next_new]) {
            if (entry.used && entry.spirv.ptr != null) {
                note_shader_owned_bytes_freed(state, entry.spirv.len);
                alloc::free(state.host_allocator, entry.spirv.ptr);
            }
        }
        mem::free(state.shader_store.entries.ptr);
        state.shader_store.entries = {};
    }
    if (state.shader_store.bucket_heads.ptr != null) {
        mem::free(state.shader_store.bucket_heads.ptr);
        state.shader_store.bucket_heads = {};
    }
    state.shader_store.free_head = 0;
    state.shader_store.next_new = 0;
}

<* Fault SHADER_INVALID on any set other than 0 or a set-0 binding that breaks
   the heap convention; buffers-only shaders pass. *>
fn void? check_heap_convention(
    spvreflect::ShaderModule* reflect_module,
    ZString entry_point,
) @private {
    uint count = 0;
    if (reflect_module.enumerate_entry_point_descriptor_bindings(
        entry_point,
        &count,
        null,
    ) != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return gpu::SHADER_INVALID~;
    }
    if (count == 0) return;

    @pool() {
        spvreflect::DescriptorBinding*[] bindings =
            mem::talloc_array(spvreflect::DescriptorBinding*, (sz)count);
        if (reflect_module.enumerate_entry_point_descriptor_bindings(
            entry_point,
            &count,
            bindings.ptr,
        )
            != spvreflect::ReflectResult.RESULT_SUCCESS) {
            return gpu::SHADER_INVALID~;
        }
        foreach (binding : bindings) {
            if (binding.set_index() != HEAP_SET) return gpu::SHADER_INVALID~;
            int descriptor_type = (int)binding.descriptor_type();
            switch (binding.binding_index()) {
                case HEAP_BINDING_SAMPLED:
                    if (descriptor_type != (int)vk::DescriptorType.SAMPLED_IMAGE) {
                        return gpu::SHADER_INVALID~;
                    }
                case HEAP_BINDING_STORAGE:
                    if (descriptor_type != (int)vk::DescriptorType.STORAGE_IMAGE) {
                        return gpu::SHADER_INVALID~;
                    }
                case HEAP_BINDING_SAMPLERS:
                    if (descriptor_type != (int)vk::DescriptorType.SAMPLER) {
                        return gpu::SHADER_INVALID~;
                    }
                default:
                    return gpu::SHADER_INVALID~;
            }
        }
    };
}

fn gpu::ShaderStage? gpu_stage_from_spv(spvreflect::ShaderStage stage) @private {
    switch (stage) {
        case spvreflect::ShaderStage.SHADER_STAGE_COMPUTE:
            return gpu::ShaderStage.COMPUTE;
        case spvreflect::ShaderStage.SHADER_STAGE_VERTEX:
            return gpu::ShaderStage.VERTEX;
        case spvreflect::ShaderStage.SHADER_STAGE_FRAGMENT:
            return gpu::ShaderStage.FRAGMENT;
        default:
            return gpu::SHADER_INVALID~;
    }
}

fn gpu::RootAbiSpec root_abi_for_stage(gpu::ShaderStage stage) @private {
    switch (stage) {
        case gpu::ShaderStage.COMPUTE:
            return gpu::ROOT_PUSH_ABI;
        case gpu::ShaderStage.VERTEX:
        case gpu::ShaderStage.FRAGMENT:
            return gpu::GRAPHICS_ROOT_PUSH_ABI;
    }
}

fn bool has_type_flag(
    spvreflect::TypeFlags flags,
    spvreflect::TypeFlags flag,
) @private {
    return ((uint)flags & (uint)flag) != 0;
}

fn bool root_member_shape_matches(
    spvreflect::BlockVariable* member,
    gpu::RootAbiMemberSpec* expected,
) @private {
    if (member == null
        || member.byte_offset() != expected.offset
        || member.byte_size() != expected.size
        || member.scalar_width() != expected.scalar_width
        || member.scalar_is_signed() != expected.scalar_signed) return false;
    spvreflect::TypeFlags flags = member.type_flags();
    bool integer = has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_INT);
    bool float_scalar = has_type_flag(
        flags,
        spvreflect::TypeFlags.TYPE_FLAG_FLOAT,
    );
    bool scalar = !has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_VECTOR)
        && !has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_MATRIX)
        && !has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_ARRAY)
        && !has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_STRUCT)
        && !has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_BOOL)
        && !has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_REF);
    return scalar
        && integer == expected.integer
        && float_scalar != expected.integer
        && member.vector_component_count() == 0
        && member.matrix_column_count() == 0
        && member.matrix_row_count() == 0
        && member.array_dimension_count() == 0
        && member.member_count() == 0;
}

<* Fault SHADER_INVALID unless the selected entry declares no push block or
   one exact schema-generated root block for its prepared shader stage. *>
fn void? check_root_push_abi(
    spvreflect::ShaderModule* reflect_module,
    ZString entry_point,
    gpu::ShaderStage stage,
) @private {
    gpu::RootAbiSpec expected = root_abi_for_stage(stage);
    uint count = 0;
    if (reflect_module.enumerate_entry_point_push_constant_blocks(
        entry_point,
        &count,
        null,
    )
        != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return gpu::SHADER_INVALID~;
    }
    if (count == 0) return;
    if (count != 1) return gpu::SHADER_INVALID~;

    spvreflect::BlockVariable*[1] blocks;
    if (reflect_module.enumerate_entry_point_push_constant_blocks(
        entry_point,
        &count,
        &blocks[0],
    )
        != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return gpu::SHADER_INVALID~;
    }
    spvreflect::BlockVariable* block = blocks[0];
    if (block == null
        || block.byte_offset() != 0
        || block.byte_size() != expected.block_size
        || block.member_count() != expected.member_count) {
        return gpu::SHADER_INVALID~;
    }
    for (uint i = 0; i < expected.member_count; i++) {
        if (!root_member_shape_matches(block.member(i), &expected.members[i])) {
            return gpu::SHADER_INVALID~;
        }
    }
}

fn void? copy_entry_point(char* dst, ZString source) @private {
    usz i = 0;
    while (source[i] != 0) {
        if (i >= gpu::SHADER_ENTRY_POINT_CAP - 1) return gpu::INVALID_ARGUMENT~;
        dst[i] = source[i];
        i++;
    }
    dst[i] = 0;
}

fn vk::Result create_pipeline_shader_module_native(
    VkDeviceState* state,
    vk::ShaderModuleCreateInfo* info,
    vk::ShaderModule* out_shader,
) {
    $if env::TESTING:
        state.pipeline_shader_create_attempts++;
        if (state.pipeline_shader_create_result != vk::Result.SUCCESS) {
            return state.pipeline_shader_create_result;
        }
    $endif
    return vk::create_shader_module(
        device:          state.device,
        p_create_info:   info,
        p_allocator:     null,
        p_shader_module: out_shader,
    );
}

fn fault report_shader_contract_failure(
    VkDeviceState* state,
    ZString operation,
    fault public_fault,
    ZString debug_name,
    ZString rejected_field,
    ZString invariant,
) @private {
    return report_contract_failure(
        state:               state,
        operation:           operation,
        public_fault:        public_fault,
        resource_kind:       gpu::DebugResourceKind.SHADER,
        has_public_identity: false,
        index:               0,
        generation:          0,
        debug_name:          debug_name,
        rejected_field:      rejected_field,
        invariant:           invariant,
        backend_text:        null,
    );
}

fn void? validate_pipeline_shader_code(
    VkDeviceState* state,
    gpu::ShaderCode* code,
    ZString operation,
    ZString rejected_field,
) {
    if (code != null && code.is_valid()) return;
    return report_shader_contract_failure(
        state:          state,
        operation:      operation,
        public_fault:   gpu::INVALID_ARGUMENT,
        debug_name:     code == null ? null : code.debug_name,
        rejected_field: rejected_field,
        invariant:      "shader code must come from prepare_shader_code",
    )~;
}

<* Validate one prepared shader and create the temporary native module used by
   a single pipeline build. *>
fn PipelineShaderModule? create_pipeline_shader_module(
    VkDeviceState* state,
    gpu::ShaderCode* code,
    gpu::ShaderStage expected_stage,
    ZString operation,
    ZString rejected_field,
) {
    validate_pipeline_shader_code(
        state:          state,
        code:           code,
        operation:      operation,
        rejected_field: rejected_field,
    )!;
    if (code.stage != expected_stage) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "shader stage must match its pipeline role",
        )~;
    }

    PipelineShaderModule shader;
    if (catch entry_fault = copy_entry_point(
        &shader.entry_point[0],
        code.entry_point,
    )) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   entry_fault,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "entry point must fit the backend name capacity",
        )~;
    }

    ulong[spvreflect::SHADER_MODULE_SIZE / 8] module_storage;
    spvreflect::ShaderModule* reflect_module =
        (spvreflect::ShaderModule*)&module_storage;
    if (spvreflect::create_shader_module(
        code.spirv.len,
        code.spirv.ptr,
        reflect_module,
    ) != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::SHADER_INVALID,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "SPIR-V reflection must succeed",
        )~;
    }
    defer reflect_module.destroy();

    $if env::TESTING:
        state.shader_reflection_validations++;
    $endif

    spvreflect::EntryPoint* entry =
        reflect_module.get_entry_point((ZString)&shader.entry_point[0]);
    if (entry == null) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::SHADER_INVALID,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "entry point must exist in SPIR-V module",
        )~;
    }
    gpu::ShaderStage? reflected_stage_optional = gpu_stage_from_spv(entry.stage());
    if (catch reflected_stage_optional) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::SHADER_INVALID,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "reflected execution model must be supported",
        )~;
    }
    gpu::ShaderStage reflected_stage = reflected_stage_optional;
    if (reflected_stage != code.stage) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::SHADER_INVALID,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "prepared stage must match the reflected entry point",
        )~;
    }
    ZString selected_entry_point = (ZString)&shader.entry_point[0];
    if (catch heap_fault = check_heap_convention(
        reflect_module,
        selected_entry_point,
    )) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   heap_fault,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "selected-entry descriptor bindings must match the gpu heap convention",
        )~;
    }
    if (catch push_fault = check_root_push_abi(
        reflect_module,
        selected_entry_point,
        code.stage,
    )) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   push_fault,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "selected-entry push constants must exactly match the root ABI",
        )~;
    }

    @pool() {
        uint[] aligned_code = mem::talloc_array(uint, code.spirv.len / uint::size);
        mem::copy(aligned_code.ptr, code.spirv.ptr, code.spirv.len);
        vk::ShaderModuleCreateInfo info = vk::shader_module_create_info()
            .set_code_size(code.spirv.len)
            .set_code(aligned_code.ptr);
        check_shader_create_device_result(
            state: state,
            result: create_pipeline_shader_module_native(
                state,
                &info,
                &shader.shader_module,
            ),
        )!;
    };
    state.pipeline_shader_modules_created++;
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.SHADER_MODULE,
        object_handle: (ulong)shader.shader_module,
        name:          code.debug_name,
    );
    return shader;
}

fn void destroy_pipeline_shader_module(
    VkDeviceState* state,
    PipelineShaderModule* shader,
) {
    if (shader == null || shader.shader_module == null) return;
    vk::destroy_shader_module(state.device, shader.shader_module, null);
    shader.shader_module = (vk::ShaderModule)null;
}

fn PipelineShaderModule*? get_pipeline_batch_shader(
    VkDeviceState* state,
    ShaderId id,
    ZString debug_name,
    gpu::ShaderStage expected_stage,
    ZString operation,
    ZString rejected_field,
    PipelineBatchShader[] shaders,
    uint* shader_count,
) {
    gpu::ShaderCode code = shader_store_code_for_retained_id(state, id, debug_name);
    if (code.stage != expected_stage) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            debug_name:     debug_name,
            rejected_field: rejected_field,
            invariant:      "shader stage must match its pipeline role",
        )~;
    }
    foreach (&shader : shaders[:*shader_count]) {
        if (shader.id == id) return &shader.native;
    }
    PipelineShaderModule native = create_pipeline_shader_module(
        state:          state,
        code:           &code,
        expected_stage: expected_stage,
        operation:      operation,
        rejected_field: rejected_field,
    )!;
    shaders[*shader_count] = {
        .id     = id,
        .native = native,
    };
    PipelineShaderModule* result = &shaders[*shader_count].native;
    (*shader_count)++;
    return result;
}

fn void destroy_pipeline_batch_shaders(
    VkDeviceState* state,
    PipelineBatchShader[] shaders,
    uint shader_count,
) {
    foreach (&shader : shaders[:shader_count]) {
        destroy_pipeline_shader_module(state, &shader.native);
    }
}
