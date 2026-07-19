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
adapter: name="NVIDIA GeForce RTX 4090" type=discrete api_version=4210991
driver: name="NVIDIA" id=4 version=2417000448
validation: enabled=false
queues: graphics=0:0 compute=0:1 compute_distinct=true transfer=1:0 transfer_distinct=true
```

| Area | Case | Iterations | Result |
|---|---|---:|---:|
| Allocation | 64-byte frame span | 100,000 | 10.9 ns/allocation |
| Allocation | 64-byte persistent span | 4,096 | 92.7 ns/allocation; 35.7 ns/free |
| Descriptor churn | Texture single / batch of 16 / sampler, one worker | 320 each | 523 / 330 / 563 ns/descriptor |
| Command reset | Idle / 15 worker pools | 2,000 each | 4,444 / 159,622 ns/begin_frame |
| Command recording | Global / buffer barrier / indirect dispatch | 20,000 × 5 | 123 / 142 / 173 ns/record median |
| Submission | Graphics / graphics+compute / all queues | 2,000 each | 45,514 / 131,617 / 149,877 ns/end_frame |
| Pipeline creation | Cold / cached duplicate | 200 / 200,000 | 14,808 / 137 ns/create |
| Async overlap | Serialized / independent queues | 5 | 8.872 / 9.464 ms wall median; overlap observed |

## Usage guidance

- Use frame spans for transient data; persistent allocation was about 8.5 times
  slower in this run.
- Batch descriptor writes. Batches of 16 reduced one-worker texture descriptor
  cost by about 37%.
- Reuse worker threads. Recording pools are cached per worker and only dirty pools reset.
- Avoid activating queues without useful work. End-of-frame cost increased with
  each queue participating in the frame.
- Cache pipelines. A cached duplicate was about 108 times cheaper than a cold
  pipeline creation.
- Queue overlap does not guarantee lower frame time. This run observed GPU
  overlap, but the independent phase was slower in wall time.
