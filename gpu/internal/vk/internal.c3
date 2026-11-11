module gpu::internal::vk @private;

import gpu @public;
import std::hash::fnv64a;
import std::atomic;

const uint HANDLE_GENERATION_MAX = 0xFF_FFFF;
// Pending texture initializations are grouped by access: one bucket per
// non-empty subset of the graphics, compute and transfer roles.
const uint QUEUE_ROLE_SLOTS = 3;
const uint QUEUE_ROLE_BITS = (1 << QUEUE_ROLE_SLOTS) - 1;
const uint PENDING_INIT_BUCKETS = QUEUE_ROLE_BITS;

enum IndexingHeapLimitFailure : char {
    NONE,
    SAMPLED_IMAGES,
    STORAGE_IMAGES,
    SAMPLERS,
    PER_STAGE_RESOURCES,
    ALL_POOLS_DESCRIPTORS,
}

struct IndexingHeapLimits {
    uint max_sampled_images;
    uint max_storage_images;
    uint max_samplers;
    uint max_per_stage_sampled_images;
    uint max_per_stage_storage_images;
    uint max_per_stage_samplers;
    uint max_per_stage_resources;
    uint max_all_pools_descriptors;
}

struct HeapDescriptorTotals {
    ulong sampled_images;
    ulong storage_images;
    ulong per_stage_resources;
    ulong all_pools_descriptors;
}

struct SamplerKey {
    gpu::Filter      min_filter;
    gpu::Filter      mag_filter;
    gpu::Filter      mip_filter;
    gpu::AddressMode address_u;
    gpu::AddressMode address_v;
    gpu::AddressMode address_w;
    float            mip_lod_bias;
    float            min_lod;
    float            max_lod;
    uint             anisotropy_enable;
    float            max_anisotropy;
    uint             compare_enable;
    gpu::CompareOp   compare;
}
$assert SamplerKey::size == 52;

<* Unified mode: textures awaiting their UNDEFINED -> GENERAL transition,
   bucketed by access so a submit takes only what its queue may touch. One
   array holds the buckets back to back: bucket b is
   `handles[starts[b] : starts[b + 1]]` and `starts[PENDING_INIT_BUCKETS]` is
   the total, so an insertion or removal moves at most one handle per later
   bucket. Mutated under the device's pending-init mutex. `outstanding`
   counts, per role, the textures queued or taken but not yet settled; it is
   read and settled without the mutex. *>
struct PendingTextureInits {
    gpu::TextureHandle[]           handles;
    uint[PENDING_INIT_BUCKETS + 1] starts;
    Atomic{uint}[QUEUE_ROLE_SLOTS] outstanding;
}

<* Textures one submit took from the pending list, in bucket order:
   `handles` starts with `bucket_counts[0]` handles of bucket 0, then those of
   bucket 1, and so on. The counts, not the handles or their texture slots,
   settle the take, so a texture destroyed after the take cannot hide its
   outstanding count. *>
struct PendingInitTake {
    gpu::TextureHandle[]       handles;
    uint[PENDING_INIT_BUCKETS] bucket_counts;
}

<* Queue a texture with `access` and count it as outstanding. A full list
   drops it and returns false. *>
fn bool PendingTextureInits.push(&self, gpu::TextureHandle handle, gpu::QueueRoles access) {
    if (!self.insert(handle, access)) return false;
    self.count_outstanding(access, 1, true);
    return true;
}

<* Put a taken, unsettled texture back. A full list drops and settles it. *>
fn bool PendingTextureInits.requeue(&self, gpu::TextureHandle handle, gpu::QueueRoles access) {
    if (self.insert(handle, access)) return true;
    self.count_outstanding(access, 1, false);
    return false;
}

<* Remove a queued texture and its outstanding count; false when it is not
   queued. *>
fn bool PendingTextureInits.remove(&self, gpu::TextureHandle handle, gpu::QueueRoles access) {
    uint bucket = pending_init_bucket(access);
    for (uint i = self.starts[bucket]; i < self.starts[bucket + 1]; i++) {
        if (self.handles[i] != handle) continue;
        self.remove_at(bucket, i);
        self.count_outstanding(access, 1, false);
        return true;
    }
    return false;
}

<* Queued textures a queue with `roles` may initialize. *>
fn uint PendingTextureInits.compatible_count(&self, gpu::QueueRoles roles) {
    uint count = 0;
    for (uint bucket = 0; bucket < PENDING_INIT_BUCKETS; bucket++) {
        if (!queue_roles_overlap(pending_init_bucket_roles(bucket), roles)) continue;
        count += self.starts[bucket + 1] - self.starts[bucket];
    }
    return count;
}

<* Move every queued texture a queue with `roles` may initialize into
   `out.handles`, in bucket order, shrink `out.handles` to them, count them
   per bucket in `out.bucket_counts` and return how many. They stay
   outstanding until the take is settled or requeued.
   @param [&inout] out
   @require out.handles.len >= self.compatible_count(roles)
*>
fn uint PendingTextureInits.take(&self, gpu::QueueRoles roles, PendingInitTake* out) {
    uint count = 0;
    out.bucket_counts = {};
    for (uint bucket = 0; bucket < PENDING_INIT_BUCKETS; bucket++) {
        if (!queue_roles_overlap(pending_init_bucket_roles(bucket), roles)) continue;
        while (self.starts[bucket + 1] > self.starts[bucket]) {
            uint last = self.starts[bucket + 1] - 1;
            out.handles[count++] = self.handles[last];
            out.bucket_counts[bucket]++;
            self.remove_at(bucket, last);
        }
    }
    out.handles = out.handles[:count];
    return count;
}

<* End the outstanding counts of a take once its initialization is stored for
   the other queues, or once its textures are dropped. *>
fn void PendingTextureInits.settle(&self, PendingInitTake* take) {
    for (uint bucket = 0; bucket < PENDING_INIT_BUCKETS; bucket++) {
        if (take.bucket_counts[bucket] == 0) continue;
        self.count_outstanding(
            access: pending_init_bucket_roles(bucket),
            amount: take.bucket_counts[bucket],
            add:    false,
        );
    }
}

<* Whether a texture a queue with `roles` may access is queued, or taken and
   not yet settled. Lock-free; `take_pending_texture_inits` gives the
   ordering argument. *>
fn bool PendingTextureInits.has_outstanding(&self, gpu::QueueRoles roles) {
    uint bits = (uint)roles;
    for (uint role = 0; role < QUEUE_ROLE_SLOTS; role++) {
        if ((bits & (1u << role)) == 0) continue;
        if (self.outstanding[role].load(AtomicOrdering.SEQ_CONSISTENT) > 0) return true;
    }
    return false;
}

fn uint PendingTextureInits.count(&self) => self.starts[PENDING_INIT_BUCKETS];

fn bool PendingTextureInits.insert(&self, gpu::TextureHandle handle, gpu::QueueRoles access) {
    if (self.starts[PENDING_INIT_BUCKETS] >= self.handles.len) return false;
    uint bucket = pending_init_bucket(access);
    // Each later bucket hands its first handle to the hole at its end.
    uint hole = self.starts[PENDING_INIT_BUCKETS]++;
    for (uint later = PENDING_INIT_BUCKETS - 1; later > bucket; later--) {
        self.handles[hole] = self.handles[self.starts[later]];
        hole = self.starts[later]++;
    }
    self.handles[hole] = handle;
    return true;
}

fn void PendingTextureInits.remove_at(&self, uint bucket, uint index) {
    // The bucket's last handle fills the gap; each later bucket then moves its
    // last handle into the hole before its start.
    uint hole = self.starts[bucket + 1] - 1;
    self.handles[index] = self.handles[hole];
    for (uint later = bucket + 1; later < PENDING_INIT_BUCKETS; later++) {
        uint last = self.starts[later + 1] - 1;
        self.handles[hole] = self.handles[last];
        hole = last;
        self.starts[later]--;
    }
    self.starts[PENDING_INIT_BUCKETS]--;
}

fn void PendingTextureInits.count_outstanding(&self, gpu::QueueRoles access, uint amount, bool add) {
    uint bits = (uint)pending_init_bucket_roles(pending_init_bucket(access));
    for (uint role = 0; role < QUEUE_ROLE_SLOTS; role++) {
        if ((bits & (1u << role)) == 0) continue;
        if (add) {
            self.outstanding[role].add(amount, AtomicOrdering.SEQ_CONSISTENT);
        } else {
            self.outstanding[role].sub(amount, AtomicOrdering.SEQ_CONSISTENT);
        }
    }
}

<* Union of the access buckets the take holds textures from. *>
fn gpu::QueueRoles PendingInitTake.roles(&self) {
    uint bits = 0;
    for (uint bucket = 0; bucket < PENDING_INIT_BUCKETS; bucket++) {
        if (self.bucket_counts[bucket] > 0) bits |= (uint)pending_init_bucket_roles(bucket);
    }
    return (gpu::QueueRoles)bits;
}

fn float canonical_sampler_float(float value) {
    return value == 0.0f ? 0.0f : value;
}

fn bool sampler_key_equal(SamplerKey* left, SamplerKey* right) {
    return left.min_filter == right.min_filter
        && left.mag_filter == right.mag_filter
        && left.mip_filter == right.mip_filter
        && left.address_u == right.address_u
        && left.address_v == right.address_v
        && left.address_w == right.address_w
        && left.mip_lod_bias == right.mip_lod_bias
        && left.min_lod == right.min_lod
        && left.max_lod == right.max_lod
        && left.anisotropy_enable == right.anisotropy_enable
        && left.max_anisotropy == right.max_anisotropy
        && left.compare_enable == right.compare_enable
        && left.compare == right.compare;
}

fn ulong hash_sampler_key(SamplerKey* key) {
    return fnv64a::hash(((char*)key)[:SamplerKey::size]);
}

fn SamplerKey canonical_sampler_key(gpu::SamplerDesc* desc) {
    return {
        .min_filter        = desc.min_filter,
        .mag_filter        = desc.mag_filter,
        .mip_filter        = desc.mip_filter,
        .address_u         = desc.address_u,
        .address_v         = desc.address_v,
        .address_w         = desc.address_w,
        .mip_lod_bias      = canonical_sampler_float(desc.mip_lod_bias),
        .min_lod           = canonical_sampler_float(desc.min_lod),
        .max_lod           = canonical_sampler_float(desc.max_lod),
        .anisotropy_enable = desc.anisotropy_enable ? 1 : 0,
        .max_anisotropy    = desc.anisotropy_enable
            ? canonical_sampler_float(desc.max_anisotropy)
            : 0.0f,
        .compare_enable    = desc.compare_enable ? 1 : 0,
        .compare           = desc.compare_enable ? desc.compare : {},
    };
}

fn HeapDescriptorTotals descriptor_heap_totals(uint tex_capacity, uint smp_capacity) {
    // Heap-local checks cannot reserve device-wide capacity used by other
    // update-after-bind pools or pipeline layouts.
    return {
        .sampled_images        = (ulong)tex_capacity * 3,
        .storage_images        = (ulong)tex_capacity * 2,
        .per_stage_resources   = (ulong)tex_capacity * 5,
        .all_pools_descriptors = (ulong)tex_capacity * 5
            + (ulong)smp_capacity,
    };
}

fn bool indexing_heap_fits_limits(
    uint tex_capacity,
    uint smp_capacity,
    IndexingHeapLimits limits,
) {
    return indexing_heap_limit_failure(
        tex_capacity,
        smp_capacity,
        limits,
    ) == IndexingHeapLimitFailure.NONE;
}

fn IndexingHeapLimitFailure indexing_heap_limit_failure(
    uint tex_capacity,
    uint smp_capacity,
    IndexingHeapLimits limits,
) {
    HeapDescriptorTotals totals = descriptor_heap_totals(tex_capacity, smp_capacity);
    if (totals.sampled_images > (ulong)limits.max_sampled_images
        || totals.sampled_images
            > (ulong)limits.max_per_stage_sampled_images) {
        return IndexingHeapLimitFailure.SAMPLED_IMAGES;
    }
    if (totals.storage_images > (ulong)limits.max_storage_images
        || totals.storage_images
            > (ulong)limits.max_per_stage_storage_images) {
        return IndexingHeapLimitFailure.STORAGE_IMAGES;
    }
    if (smp_capacity > limits.max_samplers
        || smp_capacity > limits.max_per_stage_samplers) {
        return IndexingHeapLimitFailure.SAMPLERS;
    }
    if (totals.per_stage_resources
        > (ulong)limits.max_per_stage_resources) {
        return IndexingHeapLimitFailure.PER_STAGE_RESOURCES;
    }
    if (totals.all_pools_descriptors
        > (ulong)limits.max_all_pools_descriptors) {
        return IndexingHeapLimitFailure.ALL_POOLS_DESCRIPTORS;
    }
    return IndexingHeapLimitFailure.NONE;
}

fn uint? next_handle_generation(uint generation) {
    if (generation >= HANDLE_GENERATION_MAX) return gpu::SLOT_TABLE_FULL~;
    return generation + 1;
}

fn bool can_allocate_handle_generation(uint generation) {
    return generation < HANDLE_GENERATION_MAX;
}

fn bool should_recycle_handle_generation(uint generation) {
    return generation < HANDLE_GENERATION_MAX - 1;
}

fn bool is_pow2(usz value) {
    return value != 0 && (value & (value - 1)) == 0;
}

fn uint? resolve_descriptor_capacity(
    uint requested,
    uint default_capacity,
) {
    uint capacity = requested == 0 ? default_capacity : requested;
    if (capacity > gpu::MAX_SHADER_HEAP_CAPACITY) return gpu::INVALID_ARGUMENT~;
    return capacity;
}

fn bool queue_roles_overlap(gpu::QueueRoles left, gpu::QueueRoles right) {
    return ((uint)left & (uint)right & QUEUE_ROLE_BITS) != 0;
}

<* Bucket of a texture's access; an empty access counts as every role. *>
fn uint pending_init_bucket(gpu::QueueRoles access) {
    uint bits = (uint)access & QUEUE_ROLE_BITS;
    return bits == 0 ? PENDING_INIT_BUCKETS - 1 : bits - 1;
}

fn gpu::QueueRoles pending_init_bucket_roles(uint bucket) {
    return (gpu::QueueRoles)(bucket + 1);
}

<* Queue identities other than `source` whose roles overlap the access of the
   textures `source` initialized: the queues whose later batches must wait
   for that initialization. Bit i stands for identity i. *>
fn uint texture_init_targets(
    gpu::QueueRoles[] identity_roles,
    uint source,
    gpu::QueueRoles initialized,
) {
    uint targets = 0;
    foreach (identity, roles : identity_roles) {
        if (identity == source || !queue_roles_overlap(roles, initialized)) continue;
        targets |= 1u << (uint)identity;
    }
    return targets;
}

<* Whether a batch must wait for another queue's texture initialization at
   `init_sequence`: not once that queue has retired it, nor when an
   application wait in the batch already gates every stage at or after it. *>
fn bool texture_init_wait_needed(
    ulong init_sequence,
    ulong retired_sequence,
    ulong covered_sequence,
) {
    return init_sequence > retired_sequence && init_sequence > covered_sequence;
}
