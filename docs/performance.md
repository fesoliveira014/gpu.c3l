# Performance

Benchmarks are advisory and must be compared with the same compiler, adapter,
driver, validation state, and queue topology.

## Run

After installing the native dependencies described in
[Testing](testing.md), run:

```sh
python scripts/run_benchmarks.py
```

The runner builds with C3 `-O1`, disables validation and implicit Vulkan
layers, runs the fixed suite once, and writes
`test/build/benchmark-report.md`. Individual targets perform their documented
warmups and repetitions.

## Windows baseline

Recorded 2026-07-13 with C3 0.8.0_2 on Windows 11:

```text
adapter: NVIDIA GeForce RTX 4090
driver: NVIDIA, id=4, raw version=2417000448
validation: disabled
queues: graphics=0:0 compute=0:1 transfer=1:0
```

| Area | Case | Iterations | Result |
|---|---|---:|---:|
| Allocation | 64-byte frame span | 100,000 | 10.9 ns/allocation |
| Allocation | 64-byte persistent span | 4,096 | 92.7 ns/allocation; 35.7 ns/free |
| Resource creation | Buffer / texture / shader, one worker | 300 each | 1,726 / 2,294 / 11,225 ns/op |
| Descriptor churn | Texture single / batch of 16 / sampler, one worker | 320 each | 523 / 330 / 563 ns/descriptor |
| Upload recording | 256 KiB, 32 MiB arena, one / four workers | 40 per worker | 1,199 / 4,441 uploads/s |
| Command reset | Idle / 15 recording contexts | 2,000 each | 2,208 / 118,863 ns/begin_frame |
| Command recording | Global / buffer barrier / indirect dispatch | 20,000 × 5 | 123 / 142 / 173 ns/record median |
| Submission | Graphics / graphics+compute / all queues | 2,000 each | 45,514 / 131,617 / 149,877 ns/end_frame |
| Pipeline creation | Cold / cached duplicate | 200 / 200,000 | 14,808 / 137 ns/create |
| Async overlap | Serialized / independent queues | 5 | 8.872 / 9.464 ms wall median; overlap observed |

## Usage guidance

- Use frame spans for transient data; persistent allocation is about 8.5 times
  slower in this run, while resource creation costs microseconds.
- Batch descriptor writes. Batches of 16 reduced one-worker texture descriptor
  cost by about 37%.
- Keep recording contexts idle when unused. Reset cost scales with dirty pools.
- Avoid activating queues without useful work. End-of-frame cost increased with
  each queue participating in the frame.
- Cache pipelines. A cached duplicate was about 108 times cheaper than a cold
  pipeline creation.
- Queue overlap does not guarantee lower frame time. This run observed GPU
  overlap, but the independent phase was slower in wall time.

The public API boundary and selected-device workload-limit checks pass the
stabilization review; no unresolved correctness blocker is known.
