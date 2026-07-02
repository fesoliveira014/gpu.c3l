# bindless_texture_compute

Headless proof of the bindless descriptor heap. No window, no descriptor sets
in application code.

Flow:

1. Create a 16x16 `sampled|storage` texture and register it in the heap once —
   one `TextureIndex` serves both access kinds.
2. Create a nearest-filter sampler (`SamplerIndex`).
3. Dispatch 1 writes a coordinate pattern into the image through
   `store_storage_texture(TextureIndex, ...)`.
4. A texture barrier orders the storage write against sampled reads
   (GENERAL layout throughout).
5. Dispatch 2 samples the image through `sample_texture_2d(TextureIndex,
   SamplerIndex, uv)` into a root-pointer buffer.
6. Readback verifies the pattern.

Shaders consume `include/shaders/descriptor_heap.glsl` — the published heap
ABI include — and receive indices inside a root struct referenced by one
64-bit push constant.

Build and run (shaders first, see `samples/README.md`):

```sh
sh scripts/build_shaders.sh
cd samples
c3c build bindless_texture_compute
./build/bindless_texture_compute
```
