module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import spvreflect;
import std::core::env;
import std::core::mem;
import std::core::mem::alloc;

typedef ShaderId = uint;

alias ShaderModuleCreateFn @private = fn vk::Result (
    void* user_data,
    vk::Device device,
    vk::ShaderModuleCreateInfo* create_info,
    vk::ShaderModule* shader_module,
);

const ShaderId SHADER_ID_INVALID = (ShaderId)uint::max;
const uint SHADER_FREE_NONE = 0;
const uint SHADER_STORE_MAX_PIPELINE_CAPACITY = 0x20000000;

struct ShaderIdentityInput {
    char[]                    spirv;
    uint*                     spirv_words;
    char[gpu::internal::SHADER_ENTRY_POINT_CAP] entry_point;
    ZString                   debug_name;
    ulong                     hash;
    gpu::internal::ShaderRole role;
}

struct ShaderStoreEntry {
    ulong                             hash;
    uint[]                            spirv_words;
    usz                               spirv_byte_size;
    char[gpu::internal::SHADER_ENTRY_POINT_CAP] entry_point;
    gpu::internal::ShaderRole         role;
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

struct PipelineShaderModule {
    vk::ShaderModule shader_module;
    char[gpu::internal::SHADER_ENTRY_POINT_CAP] entry_point;
}

struct ShaderModuleCreateOps @private {
    void*                   user_data;
    ShaderModuleCreateFn    create;
    bool allow_alternate_root_abi @if(env::TESTING);
}

<* Borrow the bytes of a live shader store entry.
   @require entry != null && entry.used && entry.spirv_words.ptr != null : "Entry must own live word storage." *>
fn char[] shader_entry_bytes(ShaderStoreEntry* entry) @private {
    return ((char*)entry.spirv_words.ptr)[:entry.spirv_byte_size];
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
    ShaderStoreEntry* entry,
    ShaderIdentityInput* input,
) {
    char[] entry_spirv = shader_entry_bytes(entry);
    if (entry.role != input.role
        || entry_spirv.len != input.spirv.len
        || ((ZString)&entry.entry_point[0]).str_view()
            != ((ZString)&input.entry_point[0]).str_view()) return false;
    return mem::equals(entry_spirv, input.spirv);
}

<* @require input == null || input.spirv.len % uint::size == 0 : "SPIR-V byte length must be word-aligned." *>
fn ShaderId? retain_shader_identity_locked(
    VkDeviceState* state,
    ShaderIdentityInput* input,
) {
    if (input == null) return gpu::INVALID_ARGUMENT~;
    ShaderStore* store = &state.shader_store;
    uint bucket = shader_store_bucket(
        input.hash,
        (uint)store.bucket_heads.len,
    );
    for (uint current = store.bucket_heads[bucket]; current != 0; ) {
        ShaderStoreEntry* entry = &store.entries[current - 1];
        if (entry.hash == input.hash
            && shader_store_entry_equals(entry, input)) {
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

    uint[] spirv_words = alloc::new_array(
        state.host_allocator,
        uint,
        (sz)(input.spirv.len / uint::size),
    );
    mem::copy(spirv_words.ptr, input.spirv.ptr, input.spirv.len);
    ShaderStoreEntry entry = {
        .hash            = input.hash,
        .spirv_words     = spirv_words,
        .spirv_byte_size = input.spirv.len,
        .role            = input.role,
        .refcount        = 1,
        .used            = true,
    };
    String entry_point = ((ZString)&input.entry_point[0]).str_view();
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
    alloc::free(state.host_allocator, entry.spirv_words.ptr);
    *entry = { .next_free = state.shader_store.free_head };
    state.shader_store.free_head = index + 1;
}

fn PreparedShaderIdentity? prepare_shader_identity_locked(
    VkDeviceState* state,
    ShaderIdentityInput* input,
) {
    return {
        .id       = retain_shader_identity_locked(state, input)!,
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

fn void shader_store_input(
    ShaderStoreEntry* entry,
    ZString debug_name,
    ShaderIdentityInput* out_input,
) {
    *out_input = {
        .spirv       = shader_entry_bytes(entry),
        .spirv_words = entry.spirv_words.ptr,
        .debug_name  = debug_name,
        .hash        = entry.hash,
        .role        = entry.role,
    };
    mem::copy(
        &out_input.entry_point[0],
        &entry.entry_point[0],
        out_input.entry_point.len,
    );
}

<* The caller retains `id` while using the borrowed output. *>
fn void shader_store_input_for_retained_id(
    VkDeviceState* state,
    ShaderId id,
    ZString debug_name,
    ShaderIdentityInput* out_input,
) {
    shader_store_input(
        entry:      &state.shader_store.entries[(uint)id],
        debug_name: debug_name,
        out_input:  out_input,
    );
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
            if (entry.used && entry.spirv_words.ptr != null) {
                alloc::free(state.host_allocator, entry.spirv_words.ptr);
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
        entry_point: entry_point,
        count:       &count,
        bindings:    null,
    ) != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return gpu::SHADER_INVALID~;
    }
    if (count == 0) return;

    @pool() {
        spvreflect::DescriptorBinding*[] bindings =
            mem::talloc_array(spvreflect::DescriptorBinding*, (sz)count);
        if (reflect_module.enumerate_entry_point_descriptor_bindings(
            entry_point: entry_point,
            count:       &count,
            bindings:    bindings.ptr,
        )
            != spvreflect::ReflectResult.RESULT_SUCCESS) {
            return gpu::SHADER_INVALID~;
        }
        foreach (binding : bindings) {
            if (binding.set_index() != HEAP_SET) return gpu::SHADER_INVALID~;
            int descriptor_type = (int)binding.descriptor_type();
            switch (binding.binding_index()) {
                case HEAP_BINDING_SAMPLED_2D:
                case HEAP_BINDING_SAMPLED_3D:
                    if (descriptor_type != (int)vk::DescriptorType.SAMPLED_IMAGE) {
                        return gpu::SHADER_INVALID~;
                    }
                case HEAP_BINDING_STORAGE_2D:
                case HEAP_BINDING_STORAGE_3D:
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

fn gpu::internal::ShaderRole? shader_role_from_spv(
    spvreflect::ShaderStage stage,
) @private {
    switch (stage) {
        case spvreflect::ShaderStage.SHADER_STAGE_COMPUTE:
            return gpu::internal::ShaderRole.COMPUTE;
        case spvreflect::ShaderStage.SHADER_STAGE_VERTEX:
            return gpu::internal::ShaderRole.VERTEX;
        case spvreflect::ShaderStage.SHADER_STAGE_FRAGMENT:
            return gpu::internal::ShaderRole.FRAGMENT;
        default:
            return gpu::SHADER_INVALID~;
    }
}

fn gpu::internal::RootAbiSpec root_abi_for_role(
    gpu::internal::ShaderRole role,
) @private {
    switch (role) {
        case gpu::internal::ShaderRole.COMPUTE:
            return gpu::internal::ROOT_PUSH_ABI;
        case gpu::internal::ShaderRole.VERTEX:
        case gpu::internal::ShaderRole.FRAGMENT:
            return gpu::internal::GRAPHICS_ROOT_PUSH_ABI;
    }
}

fn bool has_type_flag(
    spvreflect::TypeFlags flags,
    spvreflect::TypeFlags flag,
) @private {
    return ((uint)flags & (uint)flag) != 0;
}

enum RootAbiMismatch : int {
    NONE,
    REFLECTION,
    BLOCK_COUNT,
    BLOCK_MISSING,
    BLOCK_OFFSET,
    BLOCK_SIZE,
    MEMBER_COUNT,
    MEMBER_MISSING,
    MEMBER_OFFSET,
    MEMBER_SIZE,
    MEMBER_WIDTH,
    MEMBER_SIGNEDNESS,
    MEMBER_KIND,
    MEMBER_AGGREGATE,
}

fn RootAbiMismatch root_member_mismatch(
    spvreflect::BlockVariable* member,
    gpu::internal::RootAbiMemberSpec* expected,
    bool allow_alternate,
) @private {
    if (member == null) return RootAbiMismatch.MEMBER_MISSING;
    if (member.byte_offset() != expected.offset) {
        return RootAbiMismatch.MEMBER_OFFSET;
    }
    if (member.byte_size() != expected.size) {
        return RootAbiMismatch.MEMBER_SIZE;
    }
    spvreflect::TypeFlags flags = member.type_flags();
    bool reference = has_type_flag(
        flags,
        spvreflect::TypeFlags.TYPE_FLAG_REF,
    );
    if (allow_alternate && reference) {
        // Buffer references reflect as struct pointers, so scalar struct/member checks do not apply.
        bool incompatible_reference_shape = has_type_flag(
            flags,
            spvreflect::TypeFlags.TYPE_FLAG_VECTOR,
        )
            || has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_MATRIX)
            || has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_ARRAY)
            || has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_BOOL)
            || member.vector_component_count() != 0
            || member.matrix_column_count() != 0
            || member.matrix_row_count() != 0
            || member.array_dimension_count() != 0;
        return incompatible_reference_shape
            ? RootAbiMismatch.MEMBER_AGGREGATE
            : RootAbiMismatch.NONE;
    }
    bool aggregate = has_type_flag(
        flags,
        spvreflect::TypeFlags.TYPE_FLAG_VECTOR,
    )
        || has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_MATRIX)
        || has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_ARRAY)
        || has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_STRUCT)
        || has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_BOOL)
        || has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_REF)
        || member.vector_component_count() != 0
        || member.matrix_column_count() != 0
        || member.matrix_row_count() != 0
        || member.array_dimension_count() != 0
        || member.member_count() != 0;
    if (aggregate) return RootAbiMismatch.MEMBER_AGGREGATE;
    if (member.scalar_width() != expected.scalar_width) {
        return RootAbiMismatch.MEMBER_WIDTH;
    }
    if (!allow_alternate
        && member.scalar_is_signed() != expected.scalar_signed) {
        return RootAbiMismatch.MEMBER_SIGNEDNESS;
    }
    bool integer = has_type_flag(flags, spvreflect::TypeFlags.TYPE_FLAG_INT);
    bool float_scalar = has_type_flag(
        flags,
        spvreflect::TypeFlags.TYPE_FLAG_FLOAT,
    );
    if (integer != expected.integer
        || float_scalar == expected.integer) {
        return RootAbiMismatch.MEMBER_KIND;
    }
    return RootAbiMismatch.NONE;
}

fn ZString root_abi_mismatch_invariant(
    RootAbiMismatch mismatch,
) @private {
    switch (mismatch) {
        case RootAbiMismatch.REFLECTION:
            return "selected-entry push-constant reflection must succeed";
        case RootAbiMismatch.BLOCK_COUNT:
            return "selected-entry push block count must be zero or one";
        case RootAbiMismatch.BLOCK_MISSING:
            return "selected-entry push block must be present";
        case RootAbiMismatch.BLOCK_OFFSET:
            return "selected-entry push block offset must be zero";
        case RootAbiMismatch.BLOCK_SIZE:
            return "selected-entry push block size must match the root ABI";
        case RootAbiMismatch.MEMBER_COUNT:
            return "selected-entry push member count must match the root ABI";
        case RootAbiMismatch.MEMBER_MISSING:
            return "selected-entry push member must be present";
        case RootAbiMismatch.MEMBER_OFFSET:
            return "selected-entry push member offset must match the root ABI";
        case RootAbiMismatch.MEMBER_SIZE:
            return "selected-entry push member size must match the root ABI";
        case RootAbiMismatch.MEMBER_WIDTH:
            return "selected-entry push member scalar width must match the root ABI";
        case RootAbiMismatch.MEMBER_SIGNEDNESS:
            return "selected-entry push member signedness must match the root ABI";
        case RootAbiMismatch.MEMBER_KIND:
            return "selected-entry push member scalar kind must match the root ABI";
        case RootAbiMismatch.MEMBER_AGGREGATE:
            return "selected-entry push member must be a flat scalar";
        case RootAbiMismatch.NONE:
            return "selected-entry push constants must match the root ABI";
    }
}

fn RootAbiMismatch root_push_abi_mismatch(
    spvreflect::ShaderModule* reflect_module,
    ZString entry_point,
    gpu::internal::ShaderRole role,
    bool allow_alternate = false,
) @private {
    gpu::internal::RootAbiSpec expected = root_abi_for_role(role);
    uint count = 0;
    if (reflect_module.enumerate_entry_point_push_constant_blocks(
        entry_point,
        &count,
        null,
    )
        != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return RootAbiMismatch.REFLECTION;
    }
    if (count == 0) return RootAbiMismatch.NONE;
    if (count != 1) return RootAbiMismatch.BLOCK_COUNT;

    spvreflect::BlockVariable*[1] blocks;
    if (reflect_module.enumerate_entry_point_push_constant_blocks(
        entry_point,
        &count,
        &blocks[0],
    )
        != spvreflect::ReflectResult.RESULT_SUCCESS) {
        return RootAbiMismatch.REFLECTION;
    }
    spvreflect::BlockVariable* block = blocks[0];
    if (block == null) return RootAbiMismatch.BLOCK_MISSING;
    if (block.byte_offset() != 0) return RootAbiMismatch.BLOCK_OFFSET;
    if (block.byte_size() != expected.block_size) {
        return RootAbiMismatch.BLOCK_SIZE;
    }
    if (block.member_count() != expected.member_count) {
        return RootAbiMismatch.MEMBER_COUNT;
    }
    for (uint i = 0; i < expected.member_count; i++) {
        RootAbiMismatch mismatch =
            root_member_mismatch(
                block.member(i),
                &expected.members[i],
                allow_alternate,
            );
        if (mismatch != RootAbiMismatch.NONE) return mismatch;
    }
    return RootAbiMismatch.NONE;
}

<* Fault SHADER_INVALID unless the selected entry declares no push block or
   one exact schema-generated root block for its pipeline role. *>
fn void? check_root_push_abi(
    spvreflect::ShaderModule* reflect_module,
    ZString entry_point,
    gpu::internal::ShaderRole role,
) @private {
    if (root_push_abi_mismatch(
        reflect_module:  reflect_module,
        entry_point:     entry_point,
        role:            role,
        allow_alternate: false,
    ) != RootAbiMismatch.NONE) return gpu::SHADER_INVALID~;
}

fn void? copy_entry_point(char* dst, ZString source) @private {
    usz i = 0;
    while (i < gpu::internal::SHADER_ENTRY_POINT_CAP - 1
        && source[i] != 0) {
        dst[i] = source[i];
        i++;
    }
    if (i == 0 || source[i] != 0) return gpu::INVALID_ARGUMENT~;
    dst[i] = 0;
}

fn vk::Result create_shader_module_real(
    void* user_data,
    vk::Device device,
    vk::ShaderModuleCreateInfo* info,
    vk::ShaderModule* out_shader,
) @private {
    (void)user_data;
    return vk::create_shader_module(
        device:          device,
        p_create_info:   info,
        p_allocator:     null,
        p_shader_module: out_shader,
    );
}

fn ShaderModuleCreateOps shader_module_create_ops_real() @private {
    return { .create = &create_shader_module_real };
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

fn void? normalize_shader_input(
    VkDeviceState* state,
    gpu::ShaderDesc* desc,
    gpu::internal::ShaderRole role,
    ZString operation,
    ZString rejected_field,
    ShaderIdentityInput* out_input,
) {
    if (desc == null) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::INVALID_ARGUMENT,
            debug_name:     null,
            rejected_field: rejected_field,
            invariant:      "shader descriptor must be present",
        )~;
    }
    if (desc.spirv.ptr == null
        || desc.spirv.len < gpu::internal::SPIRV_HEADER_SIZE
        || desc.spirv.len % uint::size != 0) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::SHADER_INVALID,
            debug_name:     desc.debug_name,
            rejected_field: rejected_field,
            invariant:      "SPIR-V must contain a word-aligned header",
        )~;
    }
    uint magic;
    mem::copy(&magic, desc.spirv.ptr, uint::size);
    if (magic != gpu::internal::SPIRV_MAGIC) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::SHADER_INVALID,
            debug_name:     desc.debug_name,
            rejected_field: rejected_field,
            invariant:      "SPIR-V magic must be valid",
        )~;
    }

    *out_input = {
        .spirv      = desc.spirv,
        .debug_name = desc.debug_name,
        .hash       = gpu::internal::shader_identity_hash(desc.spirv),
        .role       = role,
    };
    ZString entry_point =
        desc.entry_point == null ? "main" : desc.entry_point;
    if (catch entry_fault = copy_entry_point(
        &out_input.entry_point[0],
        entry_point,
    )) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   entry_fault,
            debug_name:     desc.debug_name,
            rejected_field: rejected_field,
            invariant:      "entry point must be nonempty and fit the backend name capacity",
        )~;
    }
}

<* Reflect one normalized shader and create its temporary native module. *>
fn PipelineShaderModule? create_pipeline_shader_module_with_ops(
    VkDeviceState* state,
    ShaderIdentityInput* code,
    ZString operation,
    ZString rejected_field,
    ShaderModuleCreateOps* create_ops,
) {
    PipelineShaderModule shader;
    mem::copy(
        &shader.entry_point[0],
        &code.entry_point[0],
        shader.entry_point.len,
    );

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
    gpu::internal::ShaderRole? reflected_role_optional =
        shader_role_from_spv(entry.stage());
    if (catch reflected_role_optional) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::SHADER_INVALID,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "reflected execution model must be supported",
        )~;
    }
    gpu::internal::ShaderRole reflected_role = reflected_role_optional;
    if (reflected_role != code.role) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::SHADER_INVALID,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "reflected execution model must match the pipeline field",
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
    bool allow_alternate_root_abi = false;
    $if env::TESTING:
        allow_alternate_root_abi =
            create_ops.allow_alternate_root_abi;
    $endif
    RootAbiMismatch root_mismatch = root_push_abi_mismatch(
        reflect_module:  reflect_module,
        entry_point:     selected_entry_point,
        role:            code.role,
        allow_alternate: allow_alternate_root_abi,
    );
    if (root_mismatch != RootAbiMismatch.NONE) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::SHADER_INVALID,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      root_abi_mismatch_invariant(root_mismatch),
        )~;
    }

    if (code.spirv_words == null) {
        return report_shader_contract_failure(
            state:          state,
            operation:      operation,
            public_fault:   gpu::SHADER_INVALID,
            debug_name:     code.debug_name,
            rejected_field: rejected_field,
            invariant:      "SPIR-V storage must be retained and word-aligned",
        )~;
    }
    vk::ShaderModuleCreateInfo info = vk::shader_module_create_info()
        .set_code_size(code.spirv.len)
        .set_code(code.spirv_words);
    check_shader_create_device_result(
        state: state,
        result: create_ops.create(
            user_data:     create_ops.user_data,
            device:        state.device,
            create_info:   &info,
            shader_module: &shader.shader_module,
        ),
    )!;
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.SHADER_MODULE,
        object_handle: (ulong)shader.shader_module,
        name:          code.debug_name,
    );
    return shader;
}

fn PipelineShaderModule? create_pipeline_shader_module(
    VkDeviceState* state,
    ShaderIdentityInput* code,
    ZString operation,
    ZString rejected_field,
) {
    ShaderModuleCreateOps create_ops = shader_module_create_ops_real();
    return create_pipeline_shader_module_with_ops(
        state:          state,
        code:           code,
        operation:      operation,
        rejected_field: rejected_field,
        create_ops:      &create_ops,
    );
}

fn void destroy_pipeline_shader_module(
    VkDeviceState* state,
    PipelineShaderModule* shader,
) {
    if (shader == null || shader.shader_module == null) return;
    vk::destroy_shader_module(state.device, shader.shader_module, null);
    shader.shader_module = (vk::ShaderModule)null;
}
