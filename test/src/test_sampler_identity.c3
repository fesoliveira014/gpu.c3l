module gpu_test;

import gpu @public;
import gpu::vk @public;
import std::atomic;
import std::thread;

const uint SAMPLER_INTERN_WORKER_COUNT = 8;

$assert Sampler::size <= ulong::size * 2;

struct SamplerInternWorkerArgs {
    Device*       device;
    SamplerDesc*  desc;
    Atomic{uint}* ready;
    Atomic{bool}* release;
    Sampler       sampler;
    fault         failure;
}

fn Runtime sampler_test_runtime() {
    gpu::vk::stub_set_adapter_count(1);
    gpu::vk::stub_set_strict_support(true);
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
    Sampler? sampler = gpu::intern_sampler(args.device, args.desc);
    if (catch intern_fault = sampler) {
        args.failure = intern_fault;
        return 1;
    } else {
        args.sampler = sampler;
        return 0;
    }
}

fn void sampler_identity_is_owner_bearing() @test {
    Sampler first = { .owner = 7, .index = 3, .generation = 1 };
    Sampler same = first;
    Sampler different_owner = first;
    Sampler different_index = first;
    Sampler different_generation = first;
    different_owner.owner++;
    different_index.index++;
    different_generation.generation++;

    assert(first.is_valid(), "owner and generation make a sampler valid");
    assert(first == same, "copied sampler identity compares equal");
    assert(first != different_owner, "owner participates in sampler identity");
    assert(first != different_index, "index participates in sampler identity");
    assert(first != different_generation,
        "generation participates in sampler identity");
    assert(!gpu::SAMPLER_INVALID.is_valid(), "the zero sampler is invalid");
}

fn void sampler_interning_pins_public_contract() @test {
    Runtime runtime = sampler_test_runtime();
    Device device = sampler_test_device(&runtime);
    ulong children_before =
        gpu::device_slots[device.index].live_children.load(AtomicOrdering.ACQUIRE);

    assert(@catch(gpu::intern_sampler(&device, null)) == gpu::INVALID_ARGUMENT);
    assert(gpu::vk::stub_sampler_intern_attempt_count(&device) == 0,
        "null descriptions stop before backend mutation");

    SamplerDesc first_desc = sampler_test_desc();
    first_desc.debug_name = "first_debug_name";
    Sampler first = gpu::intern_sampler(&device, &first_desc)!!;
    Sampler duplicate = gpu::intern_sampler(&device, &first_desc)!!;
    SamplerDesc renamed_desc = first_desc;
    renamed_desc.debug_name = "renamed_debug_name";
    Sampler renamed = gpu::intern_sampler(&device, &renamed_desc)!!;
    SamplerDesc filter_desc = first_desc;
    filter_desc.min_filter = Filter.LINEAR;
    Sampler different_filter = gpu::intern_sampler(&device, &filter_desc)!!;
    SamplerDesc address_desc = first_desc;
    address_desc.address_u = AddressMode.CLAMP_TO_BORDER;
    Sampler different_address = gpu::intern_sampler(&device, &address_desc)!!;

    assert(first == duplicate, "identical descriptions share sampler identity");
    assert(first == renamed, "debug names do not participate in sampler identity");
    assert(first != different_filter,
        "filter changes produce distinct sampler identities");
    assert(first != different_address,
        "address changes produce distinct sampler identities");
    assert(gpu::vk::stub_sampler_native_create_count(&device) == 3,
        "only semantic descriptions create backend samplers");

    SamplerIndex published = gpu::publish_sampler(&device, first)!!;
    assert(published.is_valid(), "a live sampler publishes a shader index");
    ulong children_after =
        gpu::device_slots[device.index].live_children.load(AtomicOrdering.ACQUIRE);
    assert(children_after == children_before,
        "interning and publication retain no device child");

    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void foreign_sampler_rejects_before_publication_mutation() @test {
    Runtime runtime = sampler_test_runtime();
    Device first_device = sampler_test_device(&runtime);
    Device second_device = sampler_test_device(&runtime);
    SamplerDesc desc = sampler_test_desc();
    Sampler first_sampler = gpu::intern_sampler(&first_device, &desc)!!;
    uint attempts_before =
        gpu::vk::stub_sampler_publish_attempt_count(&second_device);

    assert(@catch(gpu::publish_sampler(&second_device, first_sampler))
        == gpu::INVALID_HANDLE);
    assert(gpu::vk::stub_sampler_publish_attempt_count(&second_device)
        == attempts_before,
        "foreign samplers stop before backend publication mutation");

    gpu::destroy_device(&first_device)!!;
    gpu::destroy_device(&second_device)!!;
    gpu::destroy_runtime(&runtime)!!;
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
    bool all_equal = args[0].sampler.is_valid();
    for (uint i = 1; i < SAMPLER_INTERN_WORKER_COUNT; i++) {
        if (args[i].sampler != args[0].sampler) all_equal = false;
    }

    assert(all_joined, "every sampler interning worker succeeds");
    assert(all_equal, "concurrent identical descriptions return one identity");
    assert(gpu::vk::stub_sampler_intern_attempt_count(&device)
        == SAMPLER_INTERN_WORKER_COUNT,
        "every worker reaches backend interning");
    assert(gpu::vk::stub_sampler_native_create_count(&device) == 1,
        "concurrent identical descriptions create one backend sampler");

    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void sampler_keys_normalize_inactive_and_effective_fields() @test {
    SamplerDesc base = sampler_test_desc();
    base.max_anisotropy = 99.0f;
    base.compare = CompareOp.GREATER;
    SamplerDesc inactive = base;
    inactive.max_anisotropy = 1.0f;
    inactive.compare = CompareOp.ALWAYS;
    inactive.mip_lod_bias = -0.0f;
    gpu::vk::SamplerKey base_key = gpu::vk::canonical_sampler_key(&base, 8.0f);
    gpu::vk::SamplerKey inactive_key = gpu::vk::canonical_sampler_key(
        &inactive,
        8.0f,
    );
    assert(gpu::vk::sampler_key_equal(&base_key, &inactive_key),
        "disabled anisotropy, compare, and signed zero normalize");

    SamplerDesc clamped_a = base;
    clamped_a.anisotropy_enable = true;
    clamped_a.max_anisotropy = 16.0f;
    SamplerDesc clamped_b = clamped_a;
    clamped_b.max_anisotropy = 32.0f;
    gpu::vk::SamplerKey clamped_a_key = gpu::vk::canonical_sampler_key(
        &clamped_a,
        8.0f,
    );
    gpu::vk::SamplerKey clamped_b_key = gpu::vk::canonical_sampler_key(
        &clamped_b,
        8.0f,
    );
    assert(gpu::vk::sampler_key_equal(&clamped_a_key, &clamped_b_key),
        "requests with the same effective anisotropy share a key");
}

fn void sampler_identity_exhaustion_is_transactional() @test {
    Runtime runtime = sampler_test_runtime();
    Device device = sampler_test_device(&runtime);
    SamplerDesc desc = sampler_test_desc();
    Sampler first;
    for (uint i = 0; i < 16; i++) {
        desc.mip_lod_bias = (float)i;
        Sampler sampler = gpu::intern_sampler(&device, &desc)!!;
        if (i == 0) first = sampler;
    }
    uint creates_before = gpu::vk::stub_sampler_native_create_count(&device);
    desc.mip_lod_bias = 16.0f;
    assert(@catch(gpu::intern_sampler(&device, &desc)) == gpu::SLOT_TABLE_FULL);
    desc.mip_lod_bias = 0.0f;
    assert(gpu::intern_sampler(&device, &desc)!! == first,
        "an existing identity survives table exhaustion");
    assert(gpu::vk::stub_sampler_native_create_count(&device) == creates_before,
        "exhaustion creates no native sampler");
    gpu::destroy_device(&device)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void sampler_identity_survives_without_strict_capability() @test {
    Device device = gpu::vk::stub_create_strict_disabled_device()!!;
    SamplerDesc desc = sampler_test_desc();
    Sampler sampler = gpu::intern_sampler(&device, &desc)!!;
    uint attempts_before = gpu::vk::stub_sampler_publish_attempt_count(&device);
    assert(@catch(gpu::publish_sampler(&device, sampler))
        == gpu::UNSUPPORTED_FEATURE);
    assert(gpu::vk::stub_sampler_publish_attempt_count(&device)
        == attempts_before,
        "strict-disabled publication stops before backend mutation");
    gpu::destroy_device(&device)!!;
}
