# c3c codegen SIGABRT — DWARF debug-info assertion (root cause found)

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
c3c test vk_bootstrap            # -> abort during gpu.vk.o debug-info emission
```

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
- Minimal self-contained reduction not yet isolated; still emits with
  `line-tables`, so any target that makes the offending subprogram reachable
  should reproduce. Getting the textual IR (`--emit-llvm`) to identify the exact
  malformed node needs `compile-only --no-obj --emit-llvm` with the c3l deps
  wired on the CLI (project.json deps are not auto-resolved by `compile-only`).

## c3c edge cases noticed while minimizing (possibly separate issues)

- All-uppercase identifiers (e.g. a type named `H`) rejected as types in local
  declarations ("Parameter names may not be all uppercase" / "Expected a type
  here"). Mixed-case name resolves it.
- A generic-instantiation `alias T = SlotTable{H, V};` used as a **local
  variable** type reports "Expected a type here", while the same instantiation
  used as a **struct field** type or written inline compiles.
