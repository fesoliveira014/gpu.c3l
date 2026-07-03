# gpu_driven_draw_sdl

Windowed GPU-driven rendering: a compute pass culls a 16×16 quad grid against
an animated spotlight and writes `DrawIndexedIndirectCommand`s (plus a draw
count where `DeviceCaps.draw_indirect_count` holds); one indirect multi-draw
renders the visible quads, with per-draw position/color pulled through
`gl_DrawID`.

Build shaders first (`scripts/build_shaders.sh`), then:

```sh
c3c run gpu_driven_draw_sdl --path samples -- --frames 300
```

`--frames N` auto-exits after N frames; `--no-vsync` selects MAILBOX.
