# memory_report

Headless memory observability: creates a spread of buffers and textures,
prints `MemoryStats` (live resource counts, per-heap VMA budgets), the
persistent-arena stats, and the VMA statistics string, then tears down
cleanly — with validation enabled, teardown prints zero leak lines.

```sh
c3c run memory_report --path samples          # summary VMA report
c3c run memory_report --path samples -- --detailed
```
