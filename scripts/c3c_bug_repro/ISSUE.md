# c3c crash: malformed DWARF debug-info in `gpu::vk` aborts LLVM

## Exact issue

c3c emits a **malformed `DISubprogram`** (function debug-info node) for the
`gpu::vk` module. One of its type operands is not a `DIType`. When LLVM
finalizes DWARF while writing `gpu.vk.o`, it hits an assertion and aborts. No
C3-level diagnostic; no object emitted; the process dies with a signal (the
shell sees exit 139).

**It is a debug-info bug, not real codegen** — disabling debug info compiles and
runs fine.

Assertion + backtrace (from a static-debug c3c under gdb):

```
Assertion failed: isa<X>(Val) && "cast_if_present<Ty>() argument of incompatible type!"
  (llvm/include/llvm/Support/Casting.h:686)

#4  llvm::cast_if_present<llvm::DIType, llvm::MDOperand>(...)
#5  llvm::DwarfUnit::applySubprogramAttributes(DISubprogram const*, DIE&, bool)
#6  llvm::DwarfCompileUnit::applySubprogramAttributesToDefinition(...)
#7  llvm::DwarfDebug::finishSubprogramDefinitions()
#8  llvm::DwarfDebug::finalizeModuleInfo()
#9  llvm::DwarfDebug::endModule()
#10 llvm::AsmPrinter::doFinalization(Module&)
...
#15 llvm_emit_file(... "build/obj/linux-x64/gpu.vk.o" ...)   src/compiler/llvm_codegen.c:691
```

## Key properties

- **Reachability, not volume.** The bad node is emitted only when a certain
  *combination* of `gpu::vk` functions lands in `gpu.vk.o` together. Each source
  file compiled alone is clean.
- **Debug level.** `debug-info: full` and `line-tables` both abort. `none`
  (`-g0`) builds.
- **Versions.** Reproduces on c3c 0.8.0 (LLVM 22.1.5) **and** 0.8.1 (LLVM
  22.1.7). Not fixed by upgrade.
- **Node undumpable.** c3c asserts at IR finalize/verify *before* writing any
  `.ll`, even with `compile-only --no-obj --emit-llvm` — so the offending
  `gpu.vk.ll` from a crashing build cannot be captured.

---

## Steps to reproduce (this repo)

Needs: the repo + a Vulkan 1.3 loader (lavapipe is fine). `c3c` on PATH.

### 1. Trigger the crash

Currently masked by the workaround. To see the raw crash, force debug info back
on:

```sh
sh scripts/build_shaders.sh          # generate SPIR-V the tests embed
cd test
c3c test vk_bootstrap -g             # -g overrides "debug-info": "none"
# -> aborts during gpu.vk.o emission, no object written, exit 139
```

### 2. Confirm the workaround (green)

```sh
cd test
c3c test vk_bootstrap                # debug-info: none in project.json
# -> PASSED: 10 passed, 0 failed  (safety on)
```

### 3. Get the symbolized assertion (why it aborts)

Release builds land in stripped libLLVM. Use a **static-debug** c3c:

```sh
# regular Linux builds need GLIBC_2.38; a 2.35 host must use -static
curl -fsSL -o c3-linux-static-debug.tar.gz \
  https://github.com/c3lang/c3c/releases/download/v0.8.1/c3-linux-static-debug.tar.gz
mkdir sd && tar xzf c3-linux-static-debug.tar.gz -C sd

cd test
gdb --batch -nx -ex run -ex 'thread apply all bt' \
    --args ../sd/c3/c3c test vk_bootstrap -g
# aborting thread -> the assertion above
```

### 4. Smallest reduction (release build only)

Two files from `test/src/`, functions forced live (a driver calls them; `@test`
stripped so they are callable), compiled via `compile-only` with deps wired on
the CLI:

```sh
cd test
# strip @test so the fns are callable, add a driver main that calls all of them,
# then:
c3c compile-only <driver.c3> src/test_vk_features.c3 src/test_vk_vma_allocator.c3 \
    --libdir libs --lib gpu --lib vk --lib vma -g --threads 1
# -> aborts (release static). Each file alone is clean.
```

Marginal: this pair does **not** trip the static-debug build, and a hand-written
file calling only `check_required_features` + `create_device` does not
reproduce. Use the full `vk_bootstrap` target for a dependable repro.

---

## Where it lives

- Backend module `gpu::vk` = `vk/*.c3` (`module gpu::vk;`). Compiled into one
  object `gpu.vk.o`.
- Workaround: `test/project.json` → `vk_bootstrap` target has
  `"debug-info": "none"`.
- Full investigation notes: `scripts/c3c_bug_repro/REPRO.md`.

When filing upstream at `github.com/c3lang/c3c`, attach
`scripts/c3c_bug_repro/REPRO.md` (assertion, backtrace, reduction status,
diagnostic recipe).
