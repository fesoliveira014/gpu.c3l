module gpu;

fn char[] shader_words_bytes(uint[5]* words) {
    return ((char*)&words.[0])[:uint[5]::size];
}

fn ShaderDesc shader_desc_for(uint[5]* words) {
    return {
        .stage       = ShaderStage.COMPUTE,
        .spirv       = shader_words_bytes(words),
        .entry_point = "main",
    };
}

fn void shader_code_rejects_malformed_ir() @test {
    assert(@catch(prepare_shader_code(null)) == INVALID_ARGUMENT,
        "null shader descriptions fault INVALID_ARGUMENT");

    ShaderDesc empty = { .stage = ShaderStage.COMPUTE, .entry_point = "main" };
    assert(@catch(prepare_shader_code(&empty)) == SHADER_INVALID,
        "empty IR faults SHADER_INVALID");

    char[3] unaligned_ir;
    ShaderDesc unaligned = {
        .stage       = ShaderStage.COMPUTE,
        .spirv       = unaligned_ir[..],
        .entry_point = "main",
    };
    assert(@catch(prepare_shader_code(&unaligned)) == SHADER_INVALID,
        "non-word-sized IR faults SHADER_INVALID");

    uint[4] truncated_words = { 0x07230203, 0x00010000, 0, 1 };
    ShaderDesc truncated = {
        .stage       = ShaderStage.COMPUTE,
        .spirv       = ((char*)&truncated_words[0])[:uint[4]::size],
        .entry_point = "main",
    };
    assert(@catch(prepare_shader_code(&truncated)) == SHADER_INVALID,
        "IR shorter than the SPIR-V header faults SHADER_INVALID");

    uint[5] bad_magic_words = { 0xDEADBEEF, 0x00010000, 0, 1, 0 };
    ShaderDesc bad_magic = shader_desc_for(&bad_magic_words);
    assert(@catch(prepare_shader_code(&bad_magic)) == SHADER_INVALID,
        "non-SPIR-V input faults SHADER_INVALID");

    uint[5] valid_words = { 0x07230203, 0x00010000, 0, 1, 0 };
    ShaderDesc empty_entry = shader_desc_for(&valid_words);
    empty_entry.entry_point = "";
    assert(@catch(prepare_shader_code(&empty_entry)) == INVALID_ARGUMENT,
        "empty entry points fault INVALID_ARGUMENT");

    ShaderDesc long_entry = shader_desc_for(&valid_words);
    long_entry.entry_point = "0123456789012345678901234567890123456789012345678901234567890123";
    assert(@catch(prepare_shader_code(&long_entry)) == INVALID_ARGUMENT,
        "entry points at the storage cap fault INVALID_ARGUMENT");
}

fn void shader_code_validity_rechecks_public_structure() @test {
    uint[6] words = { 0x07230203, 0x00010000, 0, 1, 0, 0 };
    ShaderDesc desc = {
        .stage       = ShaderStage.COMPUTE,
        .spirv       = ((char*)&words[0])[:uint[6]::size],
        .entry_point = "main",
    };
    ShaderCode code = prepare_shader_code(&desc)!!;

    ShaderCode unaligned = code;
    unaligned.spirv = unaligned.spirv[:unaligned.spirv.len - 1];
    assert(!unaligned.is_valid(),
        "public shader code must remain word-sized");

    ShaderCode missing_entry = code;
    missing_entry.entry_point = null;
    assert(!missing_entry.is_valid(),
        "public shader code must retain its normalized entry point");

    words[0] = 0xDEADBEEF;
    assert(!code.is_valid(),
        "public shader code must retain SPIR-V magic");
}

fn void shader_code_identity_is_deterministic_and_borrowed() @test {
    uint[5] first_words = { 0x07230203, 0x00010000, 0, 1, 0 };
    uint[5] second_words = first_words;
    ShaderDesc first_desc = shader_desc_for(&first_words);
    ShaderDesc second_desc = shader_desc_for(&second_words);
    first_desc.debug_name = "first";
    second_desc.debug_name = "second";

    ShaderCode first = prepare_shader_code(&first_desc)!!;
    ShaderCode second = prepare_shader_code(&second_desc)!!;

    assert(first.is_valid() && second.is_valid(),
        "prepared shader code is valid");
    assert(first.spirv.ptr == first_desc.spirv.ptr,
        "prepared shader code borrows the caller's IR");
    assert(first.digest == second.digest,
        "identical bytes from different storage receive the same digest");
    assert(first == second,
        "debug names do not participate in shader identity");

    ShaderDesc default_entry_desc = shader_desc_for(&second_words);
    default_entry_desc.entry_point = null;
    ShaderCode default_entry = prepare_shader_code(&default_entry_desc)!!;
    assert(default_entry.entry_point.str_view() == "main" && first == default_entry,
        "a null entry point normalizes to main");

    ShaderDesc alternate_entry_desc = shader_desc_for(&second_words);
    alternate_entry_desc.entry_point = "alternate";
    ShaderCode alternate_entry = prepare_shader_code(&alternate_entry_desc)!!;
    assert(first != alternate_entry,
        "entry-point selection participates in shader identity");
}

fn void shader_code_collisions_compare_length_and_bytes() @test {
    uint[5] first_words = { 0x07230203, 0x00010000, 0, 1, 0 };
    uint[5] different_words = { 0x07230203, 0x00010000, 0, 2, 0 };
    ShaderDesc first_desc = shader_desc_for(&first_words);
    ShaderDesc different_desc = shader_desc_for(&different_words);
    ShaderCode first = prepare_shader_code(&first_desc)!!;
    ShaderCode different = prepare_shader_code(&different_desc)!!;

    different.digest = first.digest;
    assert(first != different,
        "equal digests still compare IR length and bytes");

    uint[6] longer_words = { 0x07230203, 0x00010000, 0, 1, 0, 0 };
    ShaderDesc longer_desc = {
        .stage       = ShaderStage.COMPUTE,
        .spirv       = ((char*)&longer_words[0])[:uint[6]::size],
        .entry_point = "main",
    };
    ShaderCode longer = prepare_shader_code(&longer_desc)!!;
    longer.digest = first.digest;
    assert(first != longer,
        "equal digests still compare IR length");
}

fn void mutating_borrowed_ir_breaks_identity() @test {
    uint[5] original_words = { 0x07230203, 0x00010000, 0, 1, 0 };
    uint[5] mutable_words = original_words;
    ShaderDesc original_desc = shader_desc_for(&original_words);
    ShaderDesc mutable_desc = shader_desc_for(&mutable_words);
    ShaderCode original = prepare_shader_code(&original_desc)!!;
    ShaderCode mutated = prepare_shader_code(&mutable_desc)!!;

    assert(original == mutated, "equal borrowed IR begins with equal identity");
    mutable_words[3] = 2;
    assert(original != mutated,
        "byte comparison prevents mutated borrowed IR from aliasing its original");
}
