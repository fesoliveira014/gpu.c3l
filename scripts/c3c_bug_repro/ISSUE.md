# c3c crash with `-g`: optional-of-struct captures a deleted debug-info temporary (dangling `backend_debug_type`)

## Summary

With debug info enabled, c3c crashes (SIGSEGV on release builds, LLVM assertion
on assert-enabled builds) whenever this pattern is compiled:

1. A struct `S` whose members (transitively) reach a function-pointer type
   whose signature mentions `S?` (optional of the same struct), and
2. some function's debug info forces `S` to be generated *before* the first
   standalone use of `S?` in another function's signature.

Root cause (confirmed by tracing the exact metadata node in gdb, see
"Mechanism"): `llvm_get_debug_type_internal`'s `TYPE_OPTIONAL` case caches the
*forward temporary* of the struct while the struct is still mid-construction.
When the struct completes, `LLVMMetadataReplaceAllUsesWith(fwd, real)` deletes
the temporary, but the optional type's `backend_debug_type` still points at the
freed node. The next function whose signature uses `S?` embeds the dangling
pointer into its `DISubroutineType`, and LLVM crashes — either immediately at
IR build/verify (`cast<Ty>() argument of incompatible type`, assert builds) or
later in the DWARF finalizer
(`cast_if_present<Ty>() argument of incompatible type` in
`DwarfUnit::applySubprogramAttributes`, or a plain SIGSEGV on release builds).

Affected: c3c **0.8.0** (LLVM 22.1.5) and **0.8.1** (LLVM 22.1.7), linux-x64.
`-g0` compiles and runs fine — no real codegen problem.

## Minimal reproduction (30 lines, no dependencies)

```c3
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
```

```sh
c3c compile-only repro.c3 -g --threads 1
```

Observed:

| build                  | result                                            |
|------------------------|---------------------------------------------------|
| 0.8.1 linux-static     | SIGSEGV (exit 139), sometimes after a long stall  |
| 0.8.1 static-debug     | `Assertion failed: isa<To>(Val) && "cast<Ty>() argument of incompatible type!"` (Casting.h:572), abort |
| 0.8.0 linux            | SIGSEGV (exit 139)                                |
| any, with `-g0`        | compiles fine                                     |

Negative controls (both compile cleanly, isolating the trigger):

- Change `alias BeginFn = fn Cmd? (Dev*)` to return plain `Cmd` → no crash.
  The **optional** in the fn-ptr signature is essential.
- Swap the order of `make` and `use_cmd` (so `Cmd?` is first requested when
  `Cmd` is *not* mid-construction) → no crash. The **generation order** is
  essential — which is why large projects see this as a flaky,
  "combination-dependent" crash.

## Mechanism

All in `src/compiler/llvm_codegen_debug_info.c` (v0.8.1, git 075f481):

1. Debug gen of `use_cmd` needs `Cmd*` → `llvm_debug_structlike_type(Cmd)`
   creates a temporary forward node and caches it:
   `type->backend_debug_type = forward` (line 412–413).
2. Member recursion: `Cmd.device` → `Dev` → `VTable` → member `begin` is
   `fn Cmd? (Dev*)` → `llvm_debug_func_type` generates the return type `Cmd?`
   → `TYPE_OPTIONAL` case (line 620–621):

   ```c
   case TYPE_OPTIONAL:
       return type->backend_debug_type = llvm_get_debug_type(c, type_lowering(type));
   ```

   `type_lowering(Cmd?)` is `Cmd`, whose cache currently holds the **forward
   temporary** → the optional's `backend_debug_type` now stores a raw pointer
   to a temporary MDNode. This raw C-side pointer is invisible to LLVM's
   RAUW/uniquing machinery.
3. The struct completes: `llvm_get_debug_struct` calls
   `LLVMMetadataReplaceAllUsesWith(forward, real)` (line 81), which RAUWs
   *and then* `MDNode::deleteTemporary`s the forward node. LLVM updates all
   metadata operands that referenced the temp — but not c3c's cached raw
   pointer. `Cmd`'s own cache is fixed by the caller
   (`type->backend_debug_type = llvm_debug_structlike_type(...)`, line 661);
   `Cmd?`'s cache now **dangles**.
4. Debug gen of `make` requests `Cmd?` → cache hit on the freed node → it is
   placed into the `DISubroutineType` type array → crash (location depends on
   heap reuse: assert builds die at the next `cast<>`, release builds corrupt
   or die in the DWARF finalizer while writing the object).

Verified empirically on the originating project (gdb on the static-debug
build, `--threads 1`, ASLR off — heap addresses reproduce exactly):

- The aborting `DISubprogram` (`applySubprogramAttributes`) was
  `gpu.vk.vk_begin_commands`, signature
  `fn CommandList? (Device*, QueueKind)`.
- Operand 0 (return type) of its `DISubroutineType` type array pointed at a
  node whose `SubclassID` byte read **96** — not a valid `MetadataKind`; the
  memory had been reallocated.
- Breakpoints on `LLVMMetadataReplaceAllUsesWith` / `MDNode::deleteTemporary`
  filtered to that exact address showed the last metadata occupant being a
  forward temp created by `llvm_debug_structlike_type("gpu.CommandList")`
  and deleted from `llvm_get_debug_struct` (line 81) — precisely the chain
  above (`CommandList.device` → `Device` → `BackendVTable` →
  `alias BeginCommandsFn = fn CommandList? (Device*, QueueKind)`).

## Notes on the wider bug class

Any `backend_debug_type` cache that stores a node it did not create can
capture someone else's forward temporary; `TYPE_OPTIONAL` and `TYPE_BITSTRUCT`
(same case label) do exactly that. Related latent variant: caches that store
*uniqued* nodes built on top of a forward (e.g. `llvm_debug_pointer_type`,
`llvm_debug_func_type` results) can also dangle if RAUW re-uniques the node
into a pre-existing equivalent (LLVM deletes the re-uniqued copy). A fix that
only patches `TYPE_OPTIONAL` closes the reproducible crash; a robust fix wants
the DIBuilder "replaceable composite + replaceTemporary" pattern or
re-fetching caches after RAUW.

## Environment

- c3c 0.8.0 (d78f10d, LLVM 22.1.5) and 0.8.1 (075f481, LLVM 22.1.7)
- linux-x64 (WSL2, glibc 2.35 host — static release binaries)
- Repro file: `repro.c3` next to this document.
