module gpu;

// Buffer usage flags. A bitstruct of bools, composed by field-set:
//   BufferUsage u = { .storage, .addressable, .transfer_dst };
bitstruct BufferUsage : uint {
    bool transfer_src : 0;
    bool transfer_dst : 1;
    bool uniform      : 2;
    bool storage      : 3;
    bool addressable  : 4;
    bool indirect     : 5;
    bool index        : 6;
    bool vertex       : 7;
}

struct BufferDesc {
    usz         size;
    BufferUsage usage;
    MemoryKind  memory_kind;
    ZString     debug_name;
}
