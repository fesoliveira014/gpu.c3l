module gpu_test;

import gpu::internal @public;
import gpu @public;
import gpu::internal::vk @public;

fn RuntimeDesc runtime_desc() {
    return {
        .application_name = "gpu_runtime_test",
    };
}

fn void contract_validation_defaults_to_trusted() @test {
    RuntimeDesc desc;
    assert(
        desc.contract_validation == ContractValidation.TRUSTED,
        "the zero runtime descriptor selects trusted contract validation"
    );
    assert(gpu::internal::contract_validation_is_known(ContractValidation.TRUSTED));
    assert(gpu::internal::contract_validation_is_known(ContractValidation.FULL));
}

fn void full_validation_runtime_desc_enables_full_and_layers() @test {
    RuntimeDesc desc = gpu::full_validation_runtime_desc();
    assert(desc.contract_validation == ContractValidation.FULL);
    assert(desc.enable_vulkan_validation);
}

fn void unknown_contract_validation_is_rejected() @test {
    RuntimeDesc desc = runtime_desc();
    desc.contract_validation = bitcast(99, ContractValidation);
    assert(
        @catch(gpu::create_runtime(&desc)) == gpu::INVALID_ARGUMENT,
        "unknown contract validation ordinals are rejected before backend work"
    );
}

fn void runtime_and_adapter_tokens_are_compact() @test {
    assert(Runtime::size == ulong::size, "Runtime is one opaque 64-bit token");
    assert(Adapter::size == ulong::size, "Adapter is one opaque 64-bit token");
}

fn void null_runtime_descriptor_is_rejected() @test {
    assert(
        @catch(gpu::create_runtime((RuntimeDesc*)null)) == gpu::INVALID_ARGUMENT,
        "create_runtime rejects a null descriptor"
    );
}

fn void runtimes_enumerate_independent_borrowed_adapters() @test {
    gpu::internal::vk::stub_set_adapter_count(2);
    RuntimeDesc desc = runtime_desc();
    Runtime first = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&first)!!;
    Runtime second = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&second)!!;

    AdapterList first_list = gpu::enumerate_adapters(&first)!!;
    AdapterList repeated = gpu::enumerate_adapters(&first)!!;
    AdapterList second_list = gpu::enumerate_adapters(&second)!!;
    assert(first_list.count == 2, "the first runtime exposes its adapters");
    assert(repeated.count == first_list.count, "enumeration is stable");
    assert(second_list.count == 2, "the second runtime enumerates independently");

    Adapter first_adapter = first_list.get(0)!!;
    Adapter repeated_adapter = repeated.get(0)!!;
    Adapter second_adapter = second_list.get(0)!!;
    assert((ulong)first_adapter == (ulong)repeated_adapter, "repeated enumeration preserves identity");
    assert((ulong)first_adapter != (ulong)second_adapter, "adapter identity includes its runtime");

    AdapterInfo info = gpu::get_adapter_info(&first_adapter)!!;
    assert(info.name == "stub adapter 0", "semantic adapter name is available");
    assert(info.device_class == AdapterClass.SOFTWARE, "semantic device class is available");
    assert(info.memory.device_local_bytes == 1024, "semantic device-local memory is available");
    assert(info.queues.compute_count == 1, "semantic queue counts are available");
    assert(info.strict_supported, "strict support is reported semantically");

    AdapterDiagnostics diagnostics = gpu::get_adapter_diagnostics(&first_adapter)!!;
    assert(diagnostics.backend_name == "stub", "backend identity is diagnostic-only data");
    assert(diagnostics.driver_name == "stub driver", "driver identity is diagnostic-only data");
}

fn void runtime_can_have_no_adapters() @test {
    gpu::internal::vk::stub_set_adapter_count(0);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;

    AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    assert(adapters.count == 0, "an adapter-less runtime is valid");
    assert(@catch(adapters.get(0)) == gpu::INVALID_ARGUMENT, "an empty view rejects indexing");
}

fn void borrowed_adapters_expire_with_their_runtime() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    Runtime stale_runtime = runtime;
    AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    Adapter adapter = adapters.get(0)!!;

    gpu::destroy_runtime(&runtime)!!;
    Runtime replacement = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&replacement)!!;

    assert((ulong)runtime == 0, "destroy_runtime consumes the runtime token");
    assert(@catch(gpu::enumerate_adapters(&stale_runtime)) == gpu::INVALID_HANDLE, "a reused slot rejects the stale runtime");
    assert(@catch(adapters.get(0)) == gpu::INVALID_HANDLE, "a stale adapter view is rejected");
    assert(@catch(gpu::get_adapter_info(&adapter)) == gpu::INVALID_HANDLE, "a stale adapter is rejected");
    AdapterList replacement_adapters = gpu::enumerate_adapters(&replacement)!!;
    Adapter replacement_adapter = replacement_adapters.get(0)!!;
    assert(gpu::get_adapter_info(&replacement_adapter)!!.name.len > 0, "the replacement runtime is valid");
}

fn void reserved_runtime_and_adapter_bits_are_rejected() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    Adapter adapter = adapters.get(0)!!;

    Runtime forged_runtime = runtime;
    forged_runtime.reserved = 1;
    Adapter forged_adapter = adapter;
    forged_adapter.reserved = 1;
    assert(
        @catch(gpu::enumerate_adapters(&forged_runtime)) == gpu::INVALID_HANDLE,
        "reserved runtime bits cannot alias a live runtime"
    );
    assert(
        @catch(gpu::get_adapter_info(&forged_adapter)) == gpu::INVALID_HANDLE,
        "reserved adapter bits cannot alias a borrowed adapter"
    );
}

fn void forged_adapter_runtime_identity_is_rejected() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    defer gpu::destroy_runtime(&runtime)!!;
    AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    Adapter adapter = adapters.get(0)!!;

    Adapter wrong_generation = adapter;
    wrong_generation.runtime_generation = adapter.runtime_generation + 1;
    Adapter out_of_range_runtime = adapter;
    out_of_range_runtime.runtime_index = (ushort)gpu::internal::MAX_RUNTIMES;
    assert(
        @catch(gpu::get_adapter_info(&wrong_generation)) == gpu::INVALID_HANDLE,
        "a forged runtime generation cannot alias a live adapter"
    );
    assert(
        @catch(gpu::get_adapter_info(&out_of_range_runtime)) == gpu::INVALID_HANDLE,
        "an out-of-range runtime index cannot alias a live adapter"
    );
}

fn void runtime_is_invalid_during_backend_teardown() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_probe_destroy_visibility();
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;

    gpu::destroy_runtime(&runtime)!!;
    assert(
        gpu::internal::vk::stub_destroy_visibility_fault() == gpu::INVALID_HANDLE,
        "backend teardown cannot re-enter a live public runtime"
    );
}

fn void runtime_destruction_rejects_live_dependents() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    gpu::internal::runtime_retain_device(&runtime)!!;

    assert(
        @catch(gpu::destroy_runtime(&runtime)) == gpu::RESOURCE_IN_USE,
        "a runtime with a live device is preserved"
    );
    assert((ulong)runtime != 0, "failed destruction preserves the token");

    gpu::internal::runtime_release_device(&runtime)!!;

    gpu::internal::runtime_retain_surface(&runtime)!!;
    assert(
        @catch(gpu::destroy_runtime(&runtime)) == gpu::RESOURCE_IN_USE,
        "a runtime with a live surface is preserved"
    );
    gpu::internal::runtime_release_surface(&runtime)!!;
    gpu::destroy_runtime(&runtime)!!;
}

fn void runtime_adapter_count_matches_token_capacity() @test {
    gpu::internal::vk::stub_set_adapter_count(gpu::internal::MAX_ADAPTERS_PER_RUNTIME);
    RuntimeDesc desc = runtime_desc();
    Runtime runtime = gpu::create_runtime(&desc)!!;
    AdapterList adapters = gpu::enumerate_adapters(&runtime)!!;
    Adapter last = adapters.get((uint)ushort::max)!!;
    assert(last.adapter_index == ushort::max, "the full adapter index range is representable");
    gpu::destroy_runtime(&runtime)!!;

    gpu::internal::vk::stub_set_adapter_count(gpu::internal::MAX_ADAPTERS_PER_RUNTIME + 1);
    gpu::internal::vk::stub_reset_runtime_counts();
    assert(
        @catch(gpu::create_runtime(&desc)) == gpu::SLOT_TABLE_FULL,
        "an adapter count beyond the token range is rejected"
    );
    assert(
        gpu::internal::vk::stub_runtime_create_count() == gpu::internal::vk::stub_runtime_destroy_count(),
        "the rejected backend runtime is released"
    );
}

fn void runtime_publication_failure_is_transactional() @test {
    gpu::internal::vk::stub_set_adapter_count(1);
    gpu::internal::vk::stub_reset_runtime_counts();
    RuntimeDesc desc = runtime_desc();
    Runtime[gpu::internal::MAX_RUNTIMES] runtimes;
    for (uint i = 0; i < gpu::internal::MAX_RUNTIMES; i++) {
        runtimes[i] = gpu::create_runtime(&desc)!!;
    }

    assert(@catch(gpu::create_runtime(&desc)) == gpu::SLOT_TABLE_FULL, "a full runtime table faults cleanly");
    assert(
        gpu::internal::vk::stub_runtime_destroy_count() == 1,
        "an unpublished backend runtime is destroyed"
    );

    foreach (&runtime : runtimes) gpu::destroy_runtime(runtime)!!;
    assert(
        gpu::internal::vk::stub_runtime_create_count() == gpu::internal::vk::stub_runtime_destroy_count(),
        "every created backend runtime is released"
    );
}
