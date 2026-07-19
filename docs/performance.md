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

## Allocation overhead

`allocation_bench` measures two explicit `CPU_WRITE` phases: allocate 4,000
independent 64-byte allocations at 16-byte alignment, then free the same 4,000
allocations. Its exact output schema is one `cpu_write allocate` result in
`ns/allocation` and one `cpu_write free` result in `ns/free`.

The runner rejects missing phases, different counts, or extra output. This
document does not publish allocation numbers until fresh results are captured
with the environment context below; measurements from the retired storage path
are not comparable.

## Explicit upload throughput

`upload_throughput_bench` measures the explicit caller-owned upload path at
1, 2, and 4 persistent workers across bounded payload sizes. Worker threads and
recording contexts are created once per combination, then one complete untimed
warmup precedes payload-specific measured counts: 2,048 at 4 KiB, 512 at
256 KiB, and 32 at 4 MiB. These fixed counts target roughly 100 ms or more per
combination. Each worker reuses one `CPU_WRITE`
allocation and one `GPU_PRIVATE` allocation only after waiting for the prior
covering `CompletionPoint`. A measured iteration writes the mapped source,
calls `flush_mapped_span`, records `cmd_copy_buffer`, batches executable lists
on the submitting thread, submits, and waits before reuse.

Run only this benchmark with:

```sh
c3c -O1 build upload_throughput_bench --path test
./test/build/upload_throughput_bench
```

The output reports numeric `uploads_per_sec` values with worker and payload
fields. Compare rates only with the same environment context emitted by
`benchmark_info`. This document records the method but does not publish upload
throughput numbers until fresh output is added to the canonical baseline.

## Windows baseline

Environment recorded 2026-07-13 with C3 0.8.0_2 on Windows 11. The
descriptor rows were refreshed on 2026-07-19:

```text
adapter: name="NVIDIA GeForce RTX 4090" type=discrete api_version=4210991
driver: name="NVIDIA" id=4 version=2417000448
validation: enabled=false
queues: graphics=0:0 compute=0:1 compute_distinct=true transfer=1:0 transfer_distinct=true
```

| Area | Case | Iterations | Result |
|---|---|---:|---:|
| Descriptor churn | Texture single / batch of 16, one worker | 320 each | 512 / 313 ns/descriptor |
| Sampler lookup | Intern + stable publication hits, one worker | 320 | 239 ns/op |
| Command recording | Global / buffer barrier / indirect dispatch | 20,000 × 5 | 123 / 142 / 173 ns/record median |
| Pipeline creation | Cold / cached duplicate | 200 / 200,000 | 14,808 / 137 ns/create |
| Async overlap | Serialized / independent queues | 5 | 8.872 / 9.464 ms wall median; overlap observed |

## Usage guidance

- Reuse caller-owned upload and destination allocations only after the covering
  completion point is complete; free them only after all covering work completes.
- Batch descriptor writes. Batches of 16 reduced one-worker texture descriptor
  cost by about 37%.
- Reuse worker threads. Recording pools are cached per worker and only dirty pools reset.
- Submit only useful work and retain the completion points that govern reuse.
- Cache pipelines. A cached duplicate was about 108 times cheaper than a cold
  pipeline creation.
- Queue overlap does not guarantee lower completion latency. This run observed
  GPU overlap, but the independent phase was slower in wall time.
