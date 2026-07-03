# offscreen_triangle

Headless rasterization proof: a textured triangle rendered without a window or
swapchain, written out as `offscreen_triangle.ppm`.

Flow:

1. Upload a checkerboard texture (staging → `TRANSFER_DST` → `SHADER_READ`).
2. Transition a `color_attach|transfer_src` render target to `COLOR_ATTACHMENT`.
3. One dynamic-rendering pass: clear + a single `cmd_draw`. The vertex shader
   pulls positions/UVs through `vertex_root` (no vertex buffers); the fragment
   shader samples the checkerboard through `TextureIndex`/`SamplerIndex` in
   `fragment_root` — the descriptor heap, from a graphics stage.
4. `COLOR_ATTACHMENT → TRANSFER_SRC` barrier, copy to a readback buffer, dump
   as PPM.

Build and run (shaders first, see `samples/README.md`):

```sh
sh scripts/build_shaders.sh
cd samples
c3c build offscreen_triangle
./build/offscreen_triangle
```
