module gpu::internal::vk @private;

import gpu::internal @public;
import gpu @public;
import vk;
import std::atomic;
import std::core::mem;

alias TimestampPoolCreateFn @private = fn vk::Result (
    void* user_data,
    vk::Device device,
    vk::QueryPoolCreateInfo* info,
    vk::QueryPool* out_pool,
);
alias TimestampPoolDestroyFn @private = fn void (
    void* user_data,
    vk::Device device,
    vk::QueryPool pool,
);
alias TimestampPoolReadFn @private = fn vk::Result (
    void* user_data,
    vk::Device device,
    vk::QueryPool pool,
    uint first,
    uint count,
    sz data_size,
    void* data,
    vk::DeviceSize stride,
    vk::QueryResultFlags flags,
);
alias TimestampResetCommandFn @private = fn void (
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::QueryPool pool,
    uint first,
    uint count,
);
alias TimestampWriteCommandFn @private = fn void (
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::PipelineStageFlags2 stage,
    vk::QueryPool pool,
    uint index,
);
alias TimestampResolveCommandFn @private = fn void (
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::QueryPool pool,
    uint first,
    uint count,
    vk::Buffer destination,
    vk::DeviceSize offset,
    vk::DeviceSize stride,
    vk::QueryResultFlags flags,
);

const uint MAX_TIMESTAMP_POOLS = 256;
const uint TIMESTAMP_POOL_FREE_NONE = 0;

struct TimestampPoolSlot {
    vk::QueryPool pool;
    uint          capacity;
    char[DEBUG_NAME_CAP] debug_name;
}

struct TimestampPoolCell {
    TimestampPoolSlot value;
    uint              generation;
    uint              next_free;
    bool              used;
}

struct TimestampPoolTable {
    TimestampPoolCell[] slots;
    ulong               owner;
    uint                free_head;
    uint                next_new;
}

struct TimestampPoolNativeOps @private {
    void*                  user_data;
    TimestampPoolCreateFn  create;
    TimestampPoolDestroyFn destroy;
    TimestampPoolReadFn    read;
}

struct TimestampCommandNativeOps @private {
    void*                     user_data;
    TimestampResetCommandFn   reset;
    TimestampWriteCommandFn   write;
    TimestampResolveCommandFn resolve;
}

struct LoweredTimestampResolve {
    TimestampPoolSlot*       pool;
    uint                     first;
    uint                     count;
    ResolvedSpan             destination;
}

fn gpu::TimestampPoolHandle? TimestampPoolTable.alloc(
    &self,
    TimestampPoolSlot value,
) {
    uint index;
    bool found;
    while (self.free_head != TIMESTAMP_POOL_FREE_NONE) {
        index = self.free_head - 1;
        self.free_head = self.slots[index].next_free;
        if (can_allocate_handle_generation(self.slots[index].generation)) {
            found = true;
            break;
        }
        self.slots[index].next_free = TIMESTAMP_POOL_FREE_NONE;
    }
    if (!found) {
        if (self.next_new >= self.slots.len) return gpu::SLOT_TABLE_FULL~;
        index = self.next_new++;
        self.slots[index].generation = 0;
    }
    self.slots[index].value = value;
    self.slots[index].generation = next_handle_generation(
        self.slots[index].generation,
    )!;
    self.slots[index].used = true;
    return {
        .owner      = self.owner,
        .index      = index,
        .generation = self.slots[index].generation,
    };
}

fn TimestampPoolSlot*? TimestampPoolTable.get(
    &self,
    gpu::TimestampPoolHandle handle,
) {
    if (handle.owner != self.owner) return gpu::INVALID_HANDLE~;
    uint index = handle.index;
    if (index >= self.slots.len || !self.slots[index].used) {
        return gpu::INVALID_HANDLE~;
    }
    if (self.slots[index].generation != handle.generation) {
        return gpu::INVALID_HANDLE~;
    }
    return &self.slots[index].value;
}

fn void? TimestampPoolTable.free(
    &self,
    gpu::TimestampPoolHandle handle,
) {
    self.get(handle)!;
    uint index = handle.index;
    self.slots[index].used = false;
    if (!should_recycle_handle_generation(self.slots[index].generation)) {
        self.slots[index].generation = HANDLE_GENERATION_MAX;
        self.slots[index].next_free = TIMESTAMP_POOL_FREE_NONE;
        return;
    }
    self.slots[index].generation = next_handle_generation(
        self.slots[index].generation,
    )!;
    self.slots[index].next_free = self.free_head;
    self.free_head = index + 1;
}

fn void? validate_timestamp_range(
    uint capacity,
    uint first,
    uint count,
) {
    if (count == 0) return gpu::INVALID_ARGUMENT~;
    if (first > capacity || count > capacity - first) {
        return gpu::INVALID_ARGUMENT~;
    }
}

fn usz? timestamp_result_bytes(uint count) {
    if (count == 0 || (usz)count > usz::max / ulong::size) {
        return gpu::INVALID_ARGUMENT~;
    }
    return (usz)count * ulong::size;
}

fn vk::PipelineStageFlags2 timestamp_stage_to_vk(
    gpu::StageMask stage,
) {
    if (stage.all) {
        return (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_ALL_COMMANDS_BIT;
    }
    if (stage.transfer) {
        return (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_ALL_TRANSFER_BIT;
    }
    if (stage.compute) {
        return (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_COMPUTE_SHADER_BIT;
    }
    if (stage.vertex_shader) {
        return (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_VERTEX_SHADER_BIT;
    }
    if (stage.fragment_shader) {
        return (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_FRAGMENT_SHADER_BIT;
    }
    if (stage.color_output) {
        return (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    if (stage.depth_output) {
        return (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    }
    if (stage.indirect) {
        return (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_DRAW_INDIRECT_BIT;
    }
    if (stage.acceleration_structure_build) {
        return (vk::PipelineStageFlags2)vk::PipelineStageFlagBits2
            .STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    }
    if (stage.ray_tracing) {
        return (vk::PipelineStageFlags2)vk::PipelineStageFlagBits2
            .STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    }
    if (stage.mesh_shader) {
        return (vk::PipelineStageFlags2)vk::PipelineStageFlagBits2
            .STAGE_2_MESH_SHADER_BIT_EXT;
    }
    if (stage.index_input) {
        return (vk::PipelineStageFlags2)
            vk::PipelineStageFlagBits2.STAGE_2_INDEX_INPUT_BIT;
    }
    return vk::PipelineStageFlagBits2.STAGE_2_NONE;
}

fn gpu::TimestampPoolHandle? create_timestamp_pool_with(
    TimestampPoolTable* table,
    vk::Device device,
    uint capacity,
    ZString debug_name,
    TimestampPoolNativeOps* ops,
) {
    if (table == null || ops == null || capacity == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    vk::QueryPoolCreateInfo info = vk::query_pool_create_info()
        .set_query_type(vk::QueryType.TIMESTAMP)
        .set_query_count(capacity);
    TimestampPoolSlot slot = { .capacity = capacity };
    check(ops.create(
        ops.user_data,
        device,
        &info,
        &slot.pool,
    ))!;
    copy_debug_name(&slot.debug_name[0], debug_name);
    if (try handle = table.alloc(slot)) return handle;
    ops.destroy(ops.user_data, device, slot.pool);
    return gpu::SLOT_TABLE_FULL~;
}

fn void? destroy_timestamp_pool_with(
    TimestampPoolTable* table,
    vk::Device device,
    gpu::TimestampPoolHandle handle,
    TimestampPoolNativeOps* ops,
) {
    if (table == null || ops == null) return gpu::INVALID_ARGUMENT~;
    TimestampPoolSlot* slot = table.get(handle)!;
    vk::QueryPool pool = slot.pool;
    table.free(handle)!;
    ops.destroy(ops.user_data, device, pool);
}

fn void? read_timestamps_with(
    TimestampPoolTable* table,
    vk::Device device,
    gpu::TimestampPoolHandle pool,
    uint first,
    uint count,
    ulong[] out,
    TimestampPoolNativeOps* ops,
) {
    if (table == null || ops == null || out.ptr == null || out.len < count) {
        return gpu::INVALID_ARGUMENT~;
    }
    TimestampPoolSlot* slot = table.get(pool)!;
    validate_timestamp_range(slot.capacity, first, count)!;
    usz data_size = timestamp_result_bytes(count)!;
    vk::Result result = ops.read(
        user_data: ops.user_data,
        device:    device,
        pool:      slot.pool,
        first:     first,
        count:     count,
        data_size: (sz)data_size,
        data:      out.ptr,
        stride:    ulong::size,
        flags:     (vk::QueryResultFlags)
            vk::QueryResultFlagBits.RESULT_64_BIT,
    );
    if (result == vk::Result.NOT_READY) return gpu::DEVICE_BUSY~;
    check(result)!;
}

fn uint destroy_timestamp_pool_table_with(
    TimestampPoolTable* table,
    vk::Device device,
    TimestampPoolNativeOps* ops,
) {
    if (table == null || ops == null) return 0;
    uint live;
    foreach (&cell : table.slots[:table.next_new]) {
        if (!cell.used) continue;
        ops.destroy(ops.user_data, device, cell.value.pool);
        cell.used = false;
        live++;
    }
    table.owner = 0;
    table.free_head = TIMESTAMP_POOL_FREE_NONE;
    table.next_new = 0;
    return live;
}

fn gpu::TimestampPoolHandle? vk_create_timestamp_pool(
    VkDeviceState* state,
    gpu::TimestampPoolDesc* desc,
) {
    if (state == null || desc == null || desc.capacity == 0) {
        return gpu::INVALID_ARGUMENT~;
    }
    vk::QueryPoolCreateInfo info = vk::query_pool_create_info()
        .set_query_type(vk::QueryType.TIMESTAMP)
        .set_query_count(desc.capacity);
    TimestampPoolSlot slot = { .capacity = desc.capacity };
    check_device_result(
        state,
        vk::create_query_pool(
            state.device,
            &info,
            null,
            &slot.pool,
        ),
    )!;
    copy_debug_name(&slot.debug_name[0], desc.debug_name);
    set_object_name(
        state:         state,
        object_type:   vk::ObjectType.QUERY_POOL,
        object_handle: (ulong)slot.pool,
        name:          desc.debug_name,
    );

    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    if (try handle = state.timestamp_pools.alloc(slot)) return handle;
    vk::destroy_query_pool(state.device, slot.pool, null);
    return gpu::SLOT_TABLE_FULL~;
}

fn void? vk_destroy_timestamp_pool(
    VkDeviceState* state,
    gpu::TimestampPoolHandle pool,
) {
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    TimestampPoolSlot* slot = state.timestamp_pools.get(pool)!;
    vk::QueryPool native_pool = slot.pool;
    state.timestamp_pools.free(pool)!;
    vk::destroy_query_pool(state.device, native_pool, null);
}

fn void? vk_read_timestamps(
    VkDeviceState* state,
    gpu::TimestampPoolHandle pool,
    uint first,
    uint count,
    ulong[] out,
) {
    if (out.ptr == null || out.len < count) return gpu::INVALID_ARGUMENT~;
    state.resource_mutex.lock()!!;
    defer state.resource_mutex.unlock();
    TimestampPoolSlot* slot = state.timestamp_pools.get(pool)!;
    validate_timestamp_range(slot.capacity, first, count)!;
    usz required = timestamp_result_bytes(count)!;
    vk::Result result = vk::get_query_pool_results(
        device:      state.device,
        query_pool:  slot.pool,
        first_query: first,
        query_count: count,
        data_size:   (sz)required,
        p_data:      out.ptr,
        stride:      ulong::size,
        flags:       (vk::QueryResultFlags)
            vk::QueryResultFlagBits.RESULT_64_BIT,
    );
    if (result == vk::Result.NOT_READY) return gpu::DEVICE_BUSY~;
    check_device_result(state, result)!;
}

fn void destroy_timestamp_pool_table(VkDeviceState* state) {
    if (state == null || state.timestamp_pools.slots.ptr == null) return;
    foreach (&cell : state.timestamp_pools.slots[
        :state.timestamp_pools.next_new
    ]) {
        if (!cell.used) continue;
        vk::destroy_query_pool(state.device, cell.value.pool, null);
        cell.used = false;
    }
    mem::free(state.timestamp_pools.slots.ptr);
    state.timestamp_pools = {};
}

fn TimestampPoolSlot*? lower_timestamp_pool_range(
    CommandRecord* record,
    gpu::TimestampPoolHandle pool,
    uint first,
    uint count,
    bool allow_render,
) {
    bool recording = record.state == gpu::internal::CommandState.RECORDING;
    if (allow_render) {
        recording = recording
            || record.state
                == gpu::internal::CommandState.RECORDING_RENDER_PASS;
    }
    if (!recording) return gpu::COMMAND_RECORDING_ERROR~;
    TimestampPoolSlot* slot =
        record.device_state.timestamp_pools.get(pool)!;
    validate_timestamp_range(slot.capacity, first, count)!;
    return slot;
}

fn LoweredTimestampResolve? lower_timestamp_resolve(
    CommandRecord* record,
    gpu::TimestampPoolHandle pool,
    uint first,
    uint count,
    gpu::GpuSpan destination,
) {
    TimestampPoolSlot* slot = lower_timestamp_pool_range(
        record,
        pool,
        first,
        count,
        false,
    )!;
    usz required = timestamp_result_bytes(count)!;
    ResolvedSpan resolved = resolve_span_slot(
        state:          record.device_state,
        span:           &destination,
        required:       required,
        operation:      "cmd_resolve_timestamps",
        rejected_field: "dst",
    )!;
    return {
        .pool             = slot,
        .first            = first,
        .count            = count,
        .destination      = resolved,
    };
}

fn void reset_timestamp_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::QueryPool pool,
    uint first,
    uint count,
) {
    (void)user_data;
    vk::cmd_reset_query_pool(
        command_buffer,
        pool,
        first,
        count,
    );
}

fn void write_timestamp_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::PipelineStageFlags2 stage,
    vk::QueryPool pool,
    uint index,
) {
    (void)user_data;
    vk::cmd_write_timestamp2(
        command_buffer,
        stage,
        pool,
        index,
    );
}

fn void resolve_timestamp_real(
    void* user_data,
    vk::CommandBuffer command_buffer,
    vk::QueryPool pool,
    uint first,
    uint count,
    vk::Buffer destination,
    vk::DeviceSize offset,
    vk::DeviceSize stride,
    vk::QueryResultFlags flags,
) {
    (void)user_data;
    vk::cmd_copy_query_pool_results(
        command_buffer: command_buffer,
        query_pool:     pool,
        first_query:    first,
        query_count:    count,
        dst_buffer:     destination,
        dst_offset:     offset,
        stride:         stride,
        flags:          flags,
    );
}

fn TimestampCommandNativeOps timestamp_command_native_ops() {
    return {
        .reset   = &reset_timestamp_real,
        .write   = &write_timestamp_real,
        .resolve = &resolve_timestamp_real,
    };
}

fn void emit_reset_timestamps_with(
    CommandRecord* record,
    TimestampPoolSlot* slot,
    uint first,
    uint count,
    TimestampCommandNativeOps* ops,
) {
    ops.reset(
        ops.user_data,
        record.command_buffer,
        slot.pool,
        first,
        count,
    );
}

fn void emit_write_timestamp_with(
    CommandRecord* record,
    TimestampPoolSlot* slot,
    uint index,
    vk::PipelineStageFlags2 stage,
    TimestampCommandNativeOps* ops,
) {
    ops.write(
        ops.user_data,
        record.command_buffer,
        stage,
        slot.pool,
        index,
    );
}

fn void emit_resolve_timestamps_with(
    CommandRecord* record,
    LoweredTimestampResolve* resolve,
    TimestampCommandNativeOps* ops,
) {
    ops.resolve(
        user_data:     ops.user_data,
        command_buffer: record.command_buffer,
        pool:          resolve.pool.pool,
        first:         resolve.first,
        count:         resolve.count,
        destination:   resolve.destination.slot.buffer,
        offset:        (vk::DeviceSize)resolve.destination.offset,
        stride:        (vk::DeviceSize)ulong::size,
        flags:         (vk::QueryResultFlags)(
            (uint)vk::QueryResultFlagBits.RESULT_64_BIT
                | (uint)vk::QueryResultFlagBits.WAIT_BIT
        ),
    );
}

fn void? vk_cmd_reset_timestamps_with(
    CommandRecord* record,
    gpu::TimestampPoolHandle pool,
    uint first,
    uint count,
    TimestampCommandNativeOps* ops,
) {
    TimestampPoolSlot* slot = lower_timestamp_pool_range(
        record,
        pool,
        first,
        count,
        false,
    )!;
    emit_reset_timestamps_with(record, slot, first, count, ops);
}

fn void? vk_cmd_write_timestamp_with(
    CommandRecord* record,
    gpu::TimestampPoolHandle pool,
    uint index,
    gpu::StageMask stage,
    TimestampCommandNativeOps* ops,
) {
    TimestampPoolSlot* slot = lower_timestamp_pool_range(
        record,
        pool,
        index,
        1,
        true,
    )!;
    emit_write_timestamp_with(
        record,
        slot,
        index,
        timestamp_stage_to_vk(stage),
        ops,
    );
}

fn void? vk_cmd_resolve_timestamps_with(
    CommandRecord* record,
    gpu::TimestampPoolHandle pool,
    uint first,
    uint count,
    gpu::GpuSpan destination,
    TimestampCommandNativeOps* ops,
) {
    LoweredTimestampResolve resolve = lower_timestamp_resolve(
        record,
        pool,
        first,
        count,
        destination,
    )!;
    emit_resolve_timestamps_with(record, &resolve, ops);
}

fn void? vk_cmd_reset_timestamps(
    CommandRecord* record,
    gpu::TimestampPoolHandle pool,
    uint first,
    uint count,
) {
    TimestampCommandNativeOps ops = timestamp_command_native_ops();
    return vk_cmd_reset_timestamps_with(record, pool, first, count, &ops);
}

fn void? vk_cmd_write_timestamp(
    CommandRecord* record,
    gpu::TimestampPoolHandle pool,
    uint index,
    gpu::StageMask stage,
) {
    TimestampCommandNativeOps ops = timestamp_command_native_ops();
    return vk_cmd_write_timestamp_with(record, pool, index, stage, &ops);
}

fn void? vk_cmd_resolve_timestamps(
    CommandRecord* record,
    gpu::TimestampPoolHandle pool,
    uint first,
    uint count,
    gpu::GpuSpan destination,
) {
    TimestampCommandNativeOps ops = timestamp_command_native_ops();
    return vk_cmd_resolve_timestamps_with(
        record,
        pool,
        first,
        count,
        destination,
        &ops,
    );
}
