module gpu_shaders;

const uint ROOT_PUSH_CAPACITY = 128;
const uint COMPUTE_ROOT_HEADER = 8;
const uint GRAPHICS_ROOT_HEADER = 16;

fn uint align_up(uint value, uint alignment) => (value + alignment - 1) / alignment * alignment;

<* Compute std430 layout for every struct in declaration order, filling field
   offsets and struct size/alignment, then register each struct as a field
   type. Faults LAYOUT_ERROR on any implicit interior or trailing padding,
   naming the exact _padN fields to insert. *>
fn void? compute_layouts(Schema* schema, TypeTable* types) {
    foreach (&decl : schema.structs) {
        compute_struct_layout(decl, types)!;
        types.entries.set(decl.name, {
            .kind = TypeKind.STRUCT_REF,
            .size = decl.size,
            .alignment = decl.alignment,
            .c3_name = decl.name,
            .glsl_name = decl.name,
        });
    }
}

fn void? compute_struct_layout(StructDecl* decl, TypeTable* types) {
    uint cursor = 0;
    if (decl.push_role != PushRole.NONE) {
        decl.payload_offset = decl.push_role == PushRole.COMPUTE ? COMPUTE_ROOT_HEADER : GRAPHICS_ROOT_HEADER;
        cursor = decl.payload_offset;
    }
    uint struct_align = 4;
    foreach (&field : decl.fields) {
        TypeInfo info = types.lookup(field.type_name, field.type_loc)!;
        if (decl.kind == DeclKind.PUSH
            && (field.array_len > 0
                || info.kind == TypeKind.MATRIX
                || info.kind == TypeKind.STRUCT_REF)) {
            diagnostic(field.loc, "push members must be scalars, vectors, or semantic types");
            return LAYOUT_ERROR~;
        }
        if (info.alignment > struct_align) struct_align = info.alignment;
        uint aligned = align_up(cursor, info.alignment);
        if (aligned != cursor) {
            diagnostic(field.loc, "implicit std430 padding before '%s' (offset %d, needs alignment %d): insert %s",
                field.name, cursor, info.alignment, pad_suggestion(aligned - cursor));
            return LAYOUT_ERROR~;
        }
        field.offset = cursor;
        uint? size = field_size(field, &info);
        if (catch err = size) {
            diagnostic(field.loc, "field '%s' makes the struct larger than 4 GiB", field.name);
            return LAYOUT_ERROR~;
        }
        if (size > uint::max - cursor) {
            diagnostic(field.loc, "field '%s' makes the struct larger than 4 GiB", field.name);
            return LAYOUT_ERROR~;
        }
        cursor += size;
    }
    uint total = align_up(cursor, struct_align);
    if (total != cursor) {
        diagnostic(decl.loc, "struct '%s' needs trailing std430 padding (size %d, alignment %d): insert %s",
            decl.name, cursor, struct_align, pad_suggestion(total - cursor));
        return LAYOUT_ERROR~;
    }
    if (decl.push_role != PushRole.NONE) {
        decl.size = total - decl.payload_offset;
        decl.alignment = struct_align;
        if (total > ROOT_PUSH_CAPACITY) {
            diagnostic(decl.loc, "push payload '%s' ends at byte %d; the push block holds %d bytes after the %d-byte header",
                decl.name, total, ROOT_PUSH_CAPACITY - decl.payload_offset, decl.payload_offset);
            return LAYOUT_ERROR~;
        }
        return;
    }
    decl.size = total;
    decl.alignment = struct_align;
}

fn String pad_suggestion(uint gap_bytes) {
    switch (gap_bytes / 4) {
        case 1:  return "'uint _padN;'";
        case 2:  return "2 x 'uint _padN;'";
        case 3:  return "3 x 'uint _padN;'";
        default: return "'uint _padN;' fields to fill the gap";
    }
}
