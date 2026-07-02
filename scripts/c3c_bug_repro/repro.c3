module repro;

struct Dev {
    VTable* vt;
}

struct VTable {
    BeginFn begin;
}

alias BeginFn = fn Cmd? (Dev*);

struct Cmd {
    Dev* device;
    int  x;
}

fn void use_cmd(Cmd* c) {
    (void)c;
}

fn Cmd? make(Dev* d) {
    return { .device = d, .x = 1 };
}

fn void main() {
    Dev d;
    use_cmd(null);
    Cmd? c = make(&d);
    (void)c;
}
