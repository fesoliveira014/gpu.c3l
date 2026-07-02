# c3c codegen SIGABRT — DWARF debug-info assertion (ROOT-CAUSED, minimal repro in `repro.c3`)

> **2026-07-01 (session 3): fully root-caused.** The bad node is a *deleted
> forward temporary* captured by the `TYPE_OPTIONAL` debug-type cache while its
> struct was mid-construction (`llvm_codegen_debug_info.c:621`); RAUW at struct
> completion deletes the temp (line 81), the cache dangles, and the next
> function whose signature uses `S?` embeds the freed node. 30-line
> dependency-free repro in `repro.c3`; full mechanism, gdb evidence, and
> negative controls in `ISSUE.md`. Sections below are the investigation
> history.

## Summary

`c3c` aborts while emitting the `gpu::vk` module's object file, with no C3-level
diagnostic (exit 139 for the release build — the abort signal reaches the shell
as a core dump). The crash is **inside LLVM's DWARF debug-info finalizer**, not
in real code generation: c3c emits a malformed `DISubprogram` for a `gpu::vk`
function whose type operand is not a `DIType`, and LLVM asserts when writing the
debug info into `gpu.vk.o`.

Confirmed present in **both c3c 0.8.0 (LLVM 22.1.5) and 0.8.1 (LLVM 22.1.7)**.

## The assertion + backtrace (from the static-debug build under gdb)

```
Assertion failed: isa<X>(Val) && "cast_if_present<Ty>() argument of incompatible type!"
  (llvm/include/llvm/Support/Casting.h: cast_if_present: 686)

#4  llvm::cast_if_present<llvm::DIType, llvm::MDOperand>(...)
#5  llvm::DwarfUnit::applySubprogramAttributes(DISubprogram const*, DIE&, bool)
#6  llvm::DwarfCompileUnit::applySubprogramAttributesToDefinition(...)
#7  llvm::DwarfDebug::finishSubprogramDefinitions()
#8  llvm::DwarfDebug::finalizeModuleInfo()
#9  llvm::DwarfDebug::endModule()
#10 llvm::AsmPrinter::doFinalization(Module&)
#11 llvm::FPPassManager::doFinalization(Module&)
#12 llvm::legacy::PassManagerImpl::run(Module&)
#13 LLVMTargetMachineEmit(...)
#14 LLVMTargetMachineEmitToMemoryBuffer(...)
#15 llvm_emit_file(... "build/obj/linux-x64/gpu.vk.o" ...)   src/compiler/llvm_codegen.c:691
#16 llvm_codegen(...)                                        src/compiler/llvm_codegen.c:1129
#17 thread_compile_task_llvm(...)                            src/compiler/compiler.c:157
#18 taskqueue_thread(...)                                    src/utils/taskqueue.c:29
```

The worker thread aborts; the release build has stripped libLLVM frames (hence
the earlier "crash in a stripped thread" dead end). A **static-debug** c3c build
symbolizes it cleanly (see "How this was found").

## Why it looked like an "accumulation" bug

- The abort is in `finishSubprogramDefinitions()` — it fires only when the bad
  `DISubprogram` is actually emitted into the object. Which functions get debug
  metadata emitted depends on what is reachable per target, so:
  - Each of `vk_bootstrap`'s 6 files compiled **alone** builds fine.
  - `vk_command`, `vk_root_pointer`, `unit`, and `samples/root_pointer_compute`
    build fine **with full debug info** — they don't pull in the offending
    subprogram.
  - Only `vk_bootstrap`'s combined file set makes it reachable → abort ~100%.
- `--safe=no` "reduced frequency" only because it changed what was emitted, not
  because safety was the cause.
- The `set_stage` by-value builder was a real but **separate** issue (large
  struct copy); fixing it did not remove this abort.

## Reliable reproduction (in-project)

```sh
# from repo root
sh scripts/build_shaders.sh
cd test
c3c test vk_bootstrap -g         # -> abort during gpu.vk.o debug-info emission
```

The `-g` is required in this checkout because `vk_bootstrap` now has
`"debug-info": "none"` in `test/project.json`; plain `c3c test vk_bootstrap`
exercises the workaround and should pass.

## Workaround (applied in-tree)

Disable debug info for the affected target — `test/project.json`:

```json
"vk_bootstrap": { "debug-info": "none", ... }
```

- `debug-info: "none"` → **10/10 tests pass, safety on**, on both 0.8.0 and 0.8.1.
- `debug-info: "line-tables"` → **still aborts** (the bad subprogram is emitted
  even in line-tables mode).
- CLI equivalent: `c3c test vk_bootstrap -g0`.

This is a workaround, not a fix — the underlying malformed-metadata bug is in
c3c's debug-info generation and should be fixed upstream.

## How this was found (reproducible diagnostic recipe)

1. `-v/-vv/-vvv` did not help: the crash is in a codegen worker thread that
   prints nothing before aborting.
2. Download a **static-debug** c3c build (has symbols, no glibc dependency):
   `https://github.com/c3lang/c3c/releases/download/v0.8.1/c3-linux-static-debug.tar.gz`
   (regular Linux builds need GLIBC_2.38; this WSL host has 2.35 — use `-static`).
3. Run under gdb and dump all threads:
   ```sh
   gdb --batch -nx -ex run -ex 'thread apply all bt' \
       --args /path/to/staticdbg/c3c test vk_bootstrap
   ```
   The aborting thread's backtrace is the assertion above.

## Upstream report checklist

- c3c 0.8.0 (git d78f10d, LLVM 22.1.5) and 0.8.1 (git 075f481, LLVM 22.1.7).
- linux-x64, WSL2.
- Malformed `DISubprogram` type operand emitted for a `gpu::vk` function; LLVM
  `cast_if_present<DIType>` assert in `DwarfUnit::applySubprogramAttributes`
  during object emission of `gpu.vk.o`.
- Repro requires the `gpu.c3l` repo + a Vulkan 1.3 loader (lavapipe is fine).

## Reduction status (what was tried, why the node is not dumped)

- **Deterministic repro:** the full `vk_bootstrap` target (6 files). Reproduces
  on 0.8.0 release, 0.8.1 release, and 0.8.1 static-debug (the static-debug +
  gdb run is where the backtrace above came from).
- **Reachability, not volume:** each of the 6 files compiled with its functions
  forced live (a driver that calls them; `@test` stripped so they are callable)
  builds fine **alone**. The abort needs a *combination* co-emitted into
  `gpu.vk.o` — i.e. a debug node shared across functions that only goes bad when
  a certain set is emitted together.
- **Smallest pair found:** `test_vk_features.c3` + `test_vk_vma_allocator.c3`
  together abort on the **release** static build (each alone is clean). This
  pair is *marginal*: it does **not** trip the static-debug build, and a
  hand-written single file that merely calls `check_required_features` +
  `create_device` does **not** reproduce. So the trigger is a specific
  type/function interaction, not simply "these two entry points". Use the full
  6-file target for a dependable repro.
- **Why no textual IR of the bad module:** the malformed metadata aborts c3c
  even under `compile-only --no-obj --emit-llvm` (it asserts at IR
  finalize/verify, before any `.ll` is written), so `gpu.vk.ll` for a
  *crashing* build cannot be dumped. A near-identical *non-crashing* build
  (backend forced live via `&gpu::vk::VK_VTABLE`) emits a clean `gpu.vk.ll`
  whose `DISubroutineType` type-arrays and `DISubprogram` types all validate —
  confirming the bad node exists only in the specific crashing combination.
- **Recipe to emit IR for a non-crashing build** (deps wired on the CLI, since
  `compile-only` does not read project.json dependencies):
  ```sh
  c3c compile-only <driver.c3> <gpu sources...> vk/*.c3 \
      --libdir lib --lib vk --lib vma --no-obj --emit-llvm -g --llvm-out out
  ```

## Session 3: how the node was finally identified (in-memory, not via IR dump)

Textual IR stayed impossible (asserts before `.ll` is written), but the node is
inspectable **in memory** at abort time. Recipe (static-debug build, gdb, from
`test/`; all heap addresses reproduce across runs because gdb disables ASLR and
`--threads 1` makes allocation order deterministic):

1. `break llvm::DwarfUnit::applySubprogramAttributes` (via its mangled name —
   libLLVM has symbols but no DWARF, so `SP` isn't printable; record `$rsi`,
   the 2nd SysV argument, at each hit; last hit before SIGABRT = the bad
   `DISubprogram`). Then
   `call ((void(*)(void*))'llvm::Metadata::dump() const')($lastsp)` → printed
   `vk_begin_commands`, `fn CommandList? (Device*, QueueKind)`.
2. Same trick one level down: dump the `DISubroutineType`, then its `types`
   tuple. Dumping the tuple itself **aborts** — first hint an element is not
   valid metadata. Decoding the tuple's co-allocated header
   (`SmallNumOps = 3`) locates the operand slots; operand 0 (the return type,
   `CommandList?`) held the bad pointer.
3. The failing cast's ISRA clone
   (`cast_if_present<DIType, MDOperand>`) takes the `Metadata*` directly in
   `%rdi` — breakpoint on it, record last `$rdi`. Its `SubclassID` byte read
   **96**, not a valid `MetadataKind` (tuple=5, DISubroutineType=15, DIFile=16,
   DISubprogram=18) → freed/reallocated memory, i.e. a dangling pointer.
4. `break LLVMMetadataReplaceAllUsesWith if $rdi == <addr>` and
   `break llvm::MDNode::deleteTemporary if $rdi == <addr>` with `bt` — showed
   the last metadata occupant of that address was the forward temp of
   `gpu.CommandList` created by `llvm_debug_structlike_type` and deleted from
   `llvm_get_debug_struct` (line 81), reached via
   `CommandList.device → Device → BackendVTable → BeginCommandsFn`
   (`fn CommandList? (Device*, QueueKind)`) — the `TYPE_OPTIONAL` cache capture
   described at the top.

Predicted-and-confirmed minimal repro: `repro.c3` (30 lines, no deps).
`c3c compile-only repro.c3 -g` → SIGSEGV (release) / `cast<Ty>` assert
(static-debug) on 0.8.0 and 0.8.1; `-g0` fine; non-optional fn-ptr return or
reversed function order → no crash.

## c3c edge cases noticed while minimizing (possibly separate issues)

- All-uppercase identifiers (e.g. a type named `H`) rejected as types in local
  declarations ("Parameter names may not be all uppercase" / "Expected a type
  here"). Mixed-case name resolves it.
- A generic-instantiation `alias T = SlotTable{H, V};` used as a **local
  variable** type reports "Expected a type here", while the same instantiation
  used as a **struct field** type or written inline compiles.
