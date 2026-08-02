# Experiment Plan

## Research question

How do serving runtime and concurrency affect user-visible latency, throughput, and GPU efficiency for the same language model and prompt workload?

## Evaluation hypotheses

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

## Reference experiment design

### Hardware

Use one NVIDIA GPU with enough memory for the selected model. Validate the execution path with a
low-cost model before collecting the controlled 7B–8B class comparison.

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

Token ranges must be checked with the exact pinned tokenizer rather than approximated from words
or characters. `inference-lab-inspect-dataset` produces an auditable report and fails when a prompt
falls outside the workload bounds. A dataset is not ready for measured use until that validation
passes for the selected model revision.

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

Release-gate metrics for deterministic workloads:

- exact output match rate against a stable reference backend
- output stability across repetitions and target concurrency levels
- matching-prefix character ratio when plaintext evidence is explicitly retained
- SLO-compliant eligibility under a versioned performance-and-equivalence policy

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

### Performance-and-equivalence procedure

For deterministic backend qualification, collect the reference and candidate with
`--output-evidence hash` in addition to the controls above. Evaluate the resulting artifact pairs
with `inference-lab-evaluate` and a committed policy from `policies/`. A valid gate requires:

- temperature zero and identical dataset, model revision, output limit, top-p, and seed
- matching raw/summary experiment identities
- stable reference output for every prompt
- complete candidate evidence at each selected concurrency
- every configured performance and equivalence threshold to pass

Use `--output-evidence text` only when prefix-divergence analysis is required and the dataset's
handling policy permits retaining generated text. Hash-only evidence supports exact equivalence but
not semantic equivalence, task correctness, or safety claims. Gate reports certify only the
requirements encoded in their policy.

The [Qwen3-8B A100 qualification run](QUALIFICATION_RUN.md) applied a pre-registered policy to a
new controlled artifact pair. vLLM passed the concurrency-32 performance envelope, while strict
output equivalence failed across concurrency 1–32; the candidate therefore remained ineligible
under that policy. The immutable
[qualification report](../benchmarks/2026-08-02-qwen3-8b-a100-qualification/README.md) retains the
raw evidence and failed decision without retroactively changing thresholds. The earlier
performance-only artifacts remain separate and cannot satisfy the equivalence gate.

## Baseline release acceptance criteria

The baseline release must:

- pass CI without a GPU using the mock backend
- serve streaming text through the same gateway from at least one real model backend
- benchmark at three or more concurrency levels
- save raw and summarized results
- expose Prometheus metrics
- enforce a versioned performance-and-equivalence policy for deterministic workloads
- reproduce one documented comparison with commands and hardware details

The serving and measurement criteria are demonstrated by the audited performance comparison under
`benchmarks/2026-08-01-qwen3-8b-a100/`. The policy workflow is demonstrated by the subsequent
qualification under `benchmarks/2026-08-02-qwen3-8b-a100-qualification/`, which retained a valid
failed decision after performance passed but exact equivalence did not. Both result sets retain raw
rows, summaries, commands, environment records, and checksums; validation smoke runs are excluded
from reported evidence. The deterministic policy gate and published-artifact invariants are
covered in GPU-free CI. Subsequent studies expand workload coverage and add semantic correctness
and GPU-efficiency measurements.

## Evaluation roadmap

### Quantization study

Compare BF16/FP16, INT8, and INT4 where supported. Measure speed, VRAM, and quality on a small
reasoning/coding set. Require every candidate to pass a workload-specific equivalence or correctness
policy before including its performance result on the acceptable Pareto frontier.

### Prefix caching and workload-aware analysis

Construct repeated-prefix and no-reuse datasets. Measure cache hit behavior and TTFT changes.

### Speculative decoding

Add a draft model, record acceptance rate, and identify workload regimes where the extra machinery helps or hurts.

### Triton kernel

Profile the runtime, select one isolated operator or preprocessing bottleneck, implement a Triton version, validate numerical correctness, and report operator-level plus end-to-end impact.

### Multi-GPU serving

Evaluate tensor parallelism, communication overhead, throughput scaling, and failure behavior on two or more GPUs.

### Sampled-output equivalence

Add repeated sampled generations and compare output or task-score distributions. Do not apply the
deterministic exact-match gate to temperature-positive workloads.
