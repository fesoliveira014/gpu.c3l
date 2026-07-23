module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;
import std::atomic;
import std::thread;

const uint SAMPLER_INTERN_WORKER_COUNT = 8;

struct SamplerInternWorkerArgs {
    Device*       device;
    SamplerDesc*  desc;
    Atomic{uint}* ready;
    Atomic{bool}* release;
    SamplerIndex  index;
    fault         failure;
}

fn Runtime sampler_test_runtime() {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_set_strict_support(true);
    RuntimeDesc desc = {
        .backend          = BackendKind.VULKAN,
        .application_name = "sampler_identity_test",
    };
    return gpu::create_runtime(&desc)!!;
}

fn Device sampler_test_device(Runtime* runtime) {
    Adapter adapter = gpu::enumerate_adapters(runtime)!!.get(0)!!;
    DeviceRequest request = gpu::strict_device_request();
    return gpu::create_device(&adapter, &request)!!;
}

fn SamplerDesc sampler_test_desc() {
    return {
        .min_filter = Filter.NEAREST,
        .mag_filter = Filter.LINEAR,
        .mip_filter = Filter.NEAREST,
        .address_u  = AddressMode.REPEAT,
        .address_v  = AddressMode.CLAMP_TO_EDGE,
        .address_w  = AddressMode.MIRRORED_REPEAT,
        .min_lod    = 0.0f,
        .max_lod    = 4.0f,
    };
}

fn int sampler_intern_worker(void* raw_args) {
    SamplerInternWorkerArgs* args = (SamplerInternWorkerArgs*)raw_args;
    (void)args.ready.add(1, AtomicOrdering.RELEASE);
    while (!args.release.load(AtomicOrdering.ACQUIRE)) {
        thread::yield();
    }
    SamplerIndex? index = gpu::intern_sampler(args.device, args.desc);
    if (catch intern_fault = index) {
        args.failure = intern_fault;
        return 1;
    } else {
        args.index = index;
        return 0;
    }
}

fn void sampler_interning_pins_public_contract() @test {
    Runtime runtime = sampler_test_runtime();
    Device device = sampler_test_device(&runtime);
    uint destroys_before = gpu::internal::vk::stub_sampler_native_destroy_count();
    ulong children_before =
        gpu::internal::device_slots[device.index].live_children.load(AtomicOrdering.ACQUIRE);

    assert(@catch(gpu::intern_sampler(&device, null)) == gpu::INVALID_ARGUMENT);
    assert(gpu::internal::vk::stub_sampler_intern_attempt_count(&device) == 0,
        "null descriptions stop before backend mutation");

    DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(caps.max_sampler_lod_bias == 16.0f,
        "caps expose the backend-neutral sampler LOD bias limit");
    SamplerDesc invalid_bias = sampler_test_desc();
    invalid_bias.mip_lod_bias = caps.max_sampler_lod_bias + 1.0f;
    assert(@catch(gpu::intern_sampler(&device, &invalid_bias))
        == gpu::INVALID_ARGUMENT);
    assert(gpu::internal::vk::stub_sampler_intern_attempt_count(&device) == 0,
        "over-limit LOD bias stops before sampler table mutation");

    SamplerDesc first_desc = sampler_test_desc();
    first_desc.debug_name = "first_debug_name";
    SamplerIndex first = gpu::intern_sampler(&device, &first_desc)!!;
    SamplerIndex duplicate = gpu::intern_sampler(&device, &first_desc)!!;
    SamplerDesc renamed_desc = first_desc;
    renamed_desc.debug_name = "renamed_debug_name";
    SamplerIndex renamed = gpu::intern_sampler(&device, &renamed_desc)!!;
    SamplerDesc filter_desc = first_desc;
    filter_desc.min_filter = Filter.LINEAR;
    SamplerIndex different_filter = gpu::intern_sampler(&device, &filter_desc)!!;
    SamplerDesc address_desc = first_desc;
    address_desc.address_u = AddressMode.CLAMP_TO_BORDER;
    SamplerIndex different_address = gpu::intern_sampler(&device, &address_desc)!!;

    assert(first == duplicate, "identical descriptions share a sampler index");
    assert(first == renamed, "debug names do not participate in sampler interning");
    assert(first != different_filter,
        "filter changes produce distinct sampler indices");
    assert(first != different_address,
        "address changes produce distinct sampler indices");
    assert(gpu::internal::vk::stub_sampler_native_create_count(&device) == 3,
        "only semantic descriptions create backend samplers");

    assert(first.is_valid(), "interning returns a live shader index");
    ulong children_after =
        gpu::internal::device_slots[device.index].live_children.load(AtomicOrdering.ACQUIRE);
    assert(children_after == children_before,
        "interning retains no device child");

    gpu::destroy_device(&device)!!;
    assert(gpu::internal::vk::stub_sampler_native_destroy_count() == destroys_before + 3,
        "device teardown destroys each interned native sampler exactly once");
    gpu::destroy_runtime(&runtime)!!;
}

fn void sampler_anisotropy_contract_is_exact_and_transactional() @test {
    Runtime runtime = sampler_test_runtime();
    Device device = sampler_test_device(&runtime);
    DeviceCaps caps = gpu::get_device_caps(&device)!!;
    assert(caps.max_sampler_anisotropy == 16.0f,
        "the strict stub exposes its exact anisotropy capability");

    SamplerDesc exact = sampler_test_desc();
    exact.anisotropy_enable = true;
    exact.max_anisotropy = caps.max_sampler_anisotropy;
    SamplerIndex exact_index = gpu::intern_sampler(&device, &exact)!!;
    uint attempts_after_exact =
        gpu::internal::vk::stub_sampler_intern_attempt_count(&device);
    uint creates_after_exact =
        gpu::internal::vk::stub_sampler_native_create_count(&device);

    float preferred = 64.0f;
    SamplerDesc explicitly_clamped = exact;
    explicitly_clamped.max_anisotropy = preferred > caps.max_sampler_anisotropy
        ? caps.max_sampler_anisotropy
        : preferred;
    assert(gpu::intern_sampler(&device, &explicitly_clamped)!! == exact_index,
        "caller-clamped anisotropy aliases the exact device maximum");
    assert(gpu::internal::vk::stub_sampler_native_create_count(&device)
            == creates_after_exact,
        "an explicitly clamped alias creates no second native sampler");

    SamplerDesc accepted = exact;
    accepted.max_anisotropy = 8.0f;
    SamplerIndex accepted_index = gpu::intern_sampler(&device, &accepted)!!;
    assert(accepted_index != exact_index,
        "distinct accepted anisotropy values keep distinct identities");

    SamplerDesc inactive = sampler_test_desc();
    inactive.max_anisotropy = float::nan;
    *(int*)&inactive.compare = int::max;
    SamplerIndex inactive_index = gpu::intern_sampler(&device, &inactive)!!;
    SamplerDesc inactive_alias = inactive;
    inactive_alias.max_anisotropy = float::inf;
    inactive_alias.compare = CompareOp.ALWAYS;
    assert(gpu::intern_sampler(&device, &inactive_alias)!! == inactive_index,
        "inactive anisotropy and comparison values are ignored by the public operation");

    uint attempts_before_invalid =
        gpu::internal::vk::stub_sampler_intern_attempt_count(&device);
    uint creates_before_invalid =
        gpu::internal::vk::stub_sampler_native_create_count(&device);

    float[3] below_minimum = { 0.0f, -1.0f, 0.5f };
    foreach (value : below_minimum) {
        SamplerDesc invalid = exact;
        invalid.max_anisotropy = value;
        assert(@catch(gpu::intern_sampler(&device, &invalid))
                == gpu::INVALID_ARGUMENT,
            "active anisotropy below one faults INVALID_ARGUMENT");
    }
    float[3] non_finite = { float::nan, float::inf, -float::inf };
    foreach (value : non_finite) {
        SamplerDesc invalid = exact;
        invalid.max_anisotropy = value;
        assert(@catch(gpu::intern_sampler(&device, &invalid))
                == gpu::INVALID_ARGUMENT,
            "active non-finite anisotropy faults INVALID_ARGUMENT");
    }
    SamplerDesc over_limit = exact;
    over_limit.max_anisotropy = caps.max_sampler_anisotropy + 1.0f;
    assert(@catch(gpu::intern_sampler(&device, &over_limit))
            == gpu::INVALID_ARGUMENT,
        "active anisotropy above the device maximum is rejected");
    assert(gpu::internal::vk::stub_sampler_intern_attempt_count(&device)
            == attempts_before_invalid
        && gpu::internal::vk::stub_sampler_native_create_count(&device)
            == creates_before_invalid,
        "invalid anisotropy stops before backend lookup or native creation");
    assert(gpu::intern_sampler(&device, &exact)!! == exact_index,
        "an existing exact identity remains retrievable after rejection");
    assert(attempts_after_exact == 1,
        "the first accepted anisotropy request reaches the backend once");

    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&runtime)!!;

    Device unsupported =
        gpu::internal::vk::stub_create_anisotropy_disabled_device()!!;
    SamplerDesc unsupported_desc = sampler_test_desc();
    unsupported_desc.anisotropy_enable = true;
    unsupported_desc.max_anisotropy = 1.0f;
    assert(@catch(gpu::intern_sampler(&unsupported, &unsupported_desc))
            == gpu::UNSUPPORTED_FEATURE,
        "a valid anisotropy request faults when the device lacks support");
    assert(gpu::internal::vk::stub_sampler_intern_attempt_count(&unsupported)
            == 0,
        "unsupported anisotropy stops before backend dispatch");
    gpu::destroy_device(&unsupported)!!;
}

fn void concurrent_identical_sampler_interning_is_deduplicated() @test {
    Runtime runtime = sampler_test_runtime();
    Device device = sampler_test_device(&runtime);
    SamplerDesc desc = sampler_test_desc();
    Atomic{uint} ready;
    Atomic{bool} release;
    SamplerInternWorkerArgs[SAMPLER_INTERN_WORKER_COUNT] args;
    Thread[SAMPLER_INTERN_WORKER_COUNT] workers;

    for (uint i = 0; i < SAMPLER_INTERN_WORKER_COUNT; i++) {
        args[i] = {
            .device  = &device,
            .desc    = &desc,
            .ready   = &ready,
            .release = &release,
        };
        workers[i].create(&sampler_intern_worker, &args[i])!!;
    }
    while (ready.load(AtomicOrdering.ACQUIRE) < SAMPLER_INTERN_WORKER_COUNT) {
        thread::yield();
    }
    release.store(true, AtomicOrdering.RELEASE);

    bool all_joined = true;
    foreach (&worker : workers) {
        if (worker.join()!! != 0) all_joined = false;
    }
    bool all_equal = args[0].index.is_valid();
    for (uint i = 1; i < SAMPLER_INTERN_WORKER_COUNT; i++) {
        if (args[i].index != args[0].index) all_equal = false;
    }

    assert(all_joined, "every sampler interning worker succeeds");
    assert(all_equal, "concurrent identical descriptions return one identity");
    assert(gpu::internal::vk::stub_sampler_intern_attempt_count(&device)
        == SAMPLER_INTERN_WORKER_COUNT,
        "every worker reaches backend interning");
    assert(gpu::internal::vk::stub_sampler_native_create_count(&device) == 1,
        "concurrent identical descriptions create one backend sampler");

    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void sampler_keys_normalize_inactive_and_accepted_fields() @test {
    SamplerDesc base = sampler_test_desc();
    base.max_anisotropy = float::nan;
    *(int*)&base.compare = int::max;
    SamplerDesc inactive = base;
    inactive.max_anisotropy = float::inf;
    inactive.compare = CompareOp.ALWAYS;
    inactive.mip_lod_bias = -0.0f;
    gpu::internal::vk::SamplerKey base_key =
        gpu::internal::vk::canonical_sampler_key(&base);
    gpu::internal::vk::SamplerKey inactive_key = gpu::internal::vk::canonical_sampler_key(
        &inactive,
    );
    assert(gpu::internal::vk::sampler_key_equal(&base_key, &inactive_key),
        "disabled anisotropy, compare, and signed zero normalize");
    assert(gpu::internal::vk::hash_sampler_key(&base_key)
            == gpu::internal::vk::hash_sampler_key(&inactive_key),
        "canonical aliases have identical byte hashes");
    assert(base_key.anisotropy_enable == 0 && base_key.compare_enable == 0,
        "disabled sampler flags use canonical zero words");

    SamplerDesc accepted_a = sampler_test_desc();
    accepted_a.anisotropy_enable = true;
    accepted_a.max_anisotropy = 4.0f;
    SamplerDesc accepted_b = accepted_a;
    accepted_b.max_anisotropy = 8.0f;
    gpu::internal::vk::SamplerKey accepted_a_key =
        gpu::internal::vk::canonical_sampler_key(
            &accepted_a,
        );
    gpu::internal::vk::SamplerKey accepted_b_key =
        gpu::internal::vk::canonical_sampler_key(
            &accepted_b,
        );
    assert(!gpu::internal::vk::sampler_key_equal(
            &accepted_a_key,
            &accepted_b_key,
        ),
        "distinct accepted anisotropy values keep distinct keys");
    assert(accepted_a_key.max_anisotropy == 4.0f
            && accepted_b_key.max_anisotropy == 8.0f,
        "canonical keys preserve accepted anisotropy exactly");
    assert(accepted_a_key.anisotropy_enable == 1,
        "enabled sampler flags use canonical one words");
}

fn void sampler_identity_exhaustion_is_transactional() @test {
    Runtime runtime = sampler_test_runtime();
    Device device = sampler_test_device(&runtime);
    SamplerDesc desc = sampler_test_desc();
    SamplerIndex first;
    for (uint i = 0; i < 16; i++) {
        desc.mip_lod_bias = (float)i;
        SamplerIndex index = gpu::intern_sampler(&device, &desc)!!;
        if (i == 0) first = index;
    }
    uint creates_before = gpu::internal::vk::stub_sampler_native_create_count(&device);
    desc.mip_lod_bias = 16.0f;
    assert(@catch(gpu::intern_sampler(&device, &desc)) == gpu::SLOT_TABLE_FULL);
    desc.mip_lod_bias = 0.0f;
    assert(gpu::intern_sampler(&device, &desc)!! == first,
        "an existing identity survives table exhaustion");
    assert(gpu::internal::vk::stub_sampler_native_create_count(&device) == creates_before,
        "exhaustion creates no native sampler");
    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void sampler_interning_requires_strict_capability() @test {
    Device device = gpu::internal::vk::stub_create_strict_disabled_device()!!;
    SamplerDesc desc = sampler_test_desc();
    uint attempts_before = gpu::internal::vk::stub_sampler_intern_attempt_count(&device);
    assert(@catch(gpu::intern_sampler(&device, &desc))
        == gpu::UNSUPPORTED_FEATURE);
    assert(gpu::internal::vk::stub_sampler_intern_attempt_count(&device)
        == attempts_before,
        "strict-disabled interning stops before backend mutation");
    gpu::destroy_device(&device)!!;
}
