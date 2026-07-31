# Experiment Plan

## Research question

How do serving runtime and concurrency affect user-visible latency, throughput, and GPU efficiency for the same language model and prompt workload?

## MVP hypotheses

1. A simple Transformers baseline will be competitive at concurrency 1 but degrade under concurrent requests.
2. vLLM will sustain higher output-token throughput as concurrency rises.
3. Higher concurrency will increase throughput until saturation, while P95 TTFT and latency eventually worsen.

## Controlled variables

Keep these identical within an experiment:

- model and revision
- tokenizer
- GPU model and driver
- prompt dataset and order
- maximum output tokens
- temperature, top-p, and seed
- container image or dependency lock
- warm-up procedure
- runtime flags

Record the exact command, Git commit, image digest, hardware, date, and environment variables with every published result.

## Suggested first experiment

### Hardware

Use one NVIDIA GPU with enough memory for the selected model. Start with a small model to validate the system, then repeat on a 7B–8B class model when hardware permits.

### Backends

- Hugging Face Transformers baseline
- vLLM with default scheduling
- vLLM with prefix caching enabled for a repeated-prefix workload

### Workloads

Create separate datasets rather than mixing everything into one number:

1. **Short chat:** 128–256 input tokens, 64 output tokens.
2. **Long prefill:** 2K–8K input tokens, 64 output tokens.
3. **Decode-heavy:** 128 input tokens, 256 output tokens.
4. **Repeated prefix:** shared system/context prefix with varying questions.

### Load levels

Run concurrency levels `1, 2, 4, 8, 16, 32`. Use at least 100 measured requests per level after warm-up for publishable results.

### Metrics

Primary:

- P50/P95 time to first token
- P50/P95 end-to-end latency
- requests per second
- output tokens per second
- failure rate

Secondary:

- peak GPU memory
- average GPU utilization
- cache hit rate when available
- input/output token counts
- cost estimate for rented GPU time

## Procedure

1. Pin the repository commit and runtime image.
2. Start the backend and wait for health checks.
3. Send 10–20 warm-up requests that are excluded from results.
4. Run each workload and concurrency combination three times.
5. Save raw JSONL for every request.
6. Report median results across repeated runs and include error bars where useful.
7. Inspect outliers rather than deleting them silently.
8. Repeat any surprising result before drawing conclusions.

### Runner mapping

The benchmark CLI supports steps 2–6 as follows:

- it requires a successful gateway `/health` response before creating load
- `--warmup-requests` runs before every concurrency/repetition pair and is excluded from raw output
- `--repetitions` retains each trial and computes median metrics across trials
- raw rows include prompt indices and SHA-256 hashes; the manifest includes the full dataset hash
- `--metadata` records observed hardware, driver, image digest, model revision, and runtime flags

A publishable invocation should state all non-default controls explicitly:

```bash
python -m inference_lab.benchmark.runner \
  --url http://localhost:8000 \
  --dataset data/<workload>.jsonl \
  --concurrency 1,2,4,8,16,32 \
  --requests 120 \
  --repetitions 3 \
  --warmup-requests 10 \
  --max-new-tokens 64 \
  --temperature 0 \
  --top-p 1 \
  --seed 42 \
  --metadata experiment-metadata.json \
  --output results/<backend>-<workload>.jsonl
```

The generated summary records the canonical command, resolved settings, Git state, dataset digest,
client environment, backend/model identity, per-trial summaries, and aggregate medians. Do not treat
a run as publishable when the Git state is dirty, required server metadata is missing, an image uses
a floating tag, or exact token usage is unavailable for a token-throughput claim.

## MVP completion criteria

The MVP is complete when it can:

- pass CI without a GPU using the mock backend
- serve streaming text through the same gateway from at least one real model backend
- benchmark at three or more concurrency levels
- save raw and summarized results
- expose Prometheus metrics
- reproduce one documented comparison with commands and hardware details

The mock backend currently validates the service/benchmark path without claiming performance.
Real-model serving and the first pinned-hardware comparison remain unverified in this environment;
both must be demonstrated before the MVP can be called complete.

## Post-MVP milestones

### Milestone 2: Quantization study

Compare BF16/FP16, INT8, and INT4 where supported. Measure speed, VRAM, and quality on a small reasoning/coding set.

### Milestone 3: Prefix caching and workload-aware analysis

Construct repeated-prefix and no-reuse datasets. Measure cache hit behavior and TTFT changes.

### Milestone 4: Speculative decoding

Add a draft model, record acceptance rate, and identify workload regimes where the extra machinery helps or hurts.

### Milestone 5: Triton kernel

Profile the runtime, select one isolated operator or preprocessing bottleneck, implement a Triton version, validate numerical correctness, and report operator-level plus end-to-end impact.

### Milestone 6: Multi-GPU serving

Evaluate tensor parallelism, communication overhead, throughput scaling, and failure behavior on two or more GPUs.
