# Qwen3-8B backend qualification on one A100

This record applies the pre-registered `qwen3-8b-a100-short-chat-v1` policy to a serialized
Transformers reference and a vLLM candidate measured on the same GPU. The policy decision is
**FAIL**: vLLM passed the registered performance envelope, but failed strict decoded-output
equivalence and cross-run stability requirements.

The result demonstrates a fail-closed qualification workflow. It does not establish that the
candidate outputs are less correct, less useful, or less safe; hash-only evidence cannot answer
those questions.

## Policy decision

| Section | Result | Evidence |
|---|---|---|
| Shared controls | PASS | Model, revision, dtype, dataset, generation controls, hardware, driver, and Git commit match |
| Artifact consistency | PASS | Raw rows and summaries have matching experiment identities and complete request sets |
| C32 performance | PASS | All five pre-registered thresholds passed |
| Reference stability | PASS | One output digest per prompt across all reference trials |
| Candidate evidence coverage | PASS | All 2,160 candidate rows contain output digests |
| Exact output equivalence | FAIL | 917 of 2,160 candidate outputs matched the stable reference output |
| Candidate stability | FAIL | 0 of 10 prompts retained one digest across all candidate trials |
| Deployment eligibility | **FAIL** | The failed `equivalence` section makes the candidate ineligible under this policy |

The policy was committed before measurement in Git commit
`4faecdcb654f809763df60843c394ac38393a13c`. Its SHA-256 digest is
`2ed6650a3bcaf6058621763a4abe19bd08ef4c14f19d6c6fd90d359778f2afcc`.
Thresholds were not changed after the results were inspected.

## Performance envelope

The policy evaluates performance only at concurrency 32. Each observed value is the median of
three trial summaries, with 120 measured requests per trial.

| Metric at concurrency 32 | Requirement | Observed | Result |
|---|---:|---:|---|
| Failure rate | `<= 0.0` | 0.0 | PASS |
| P95 TTFT | `<= 250 ms` | 214.542 ms | PASS |
| P95 end-to-end latency | `<= 1,250 ms` | 1,119.736 ms | PASS |
| Request throughput | `>= 27 requests/s` | 29.561 requests/s | PASS |
| Output throughput | `>= 1,720 tokens/s` | 1,891.890 tokens/s | PASS |

All 2,160 reference requests and all 2,160 candidate requests completed successfully and reported
exact output-token counts.

## Deterministic-equivalence evidence

The reference produced one stable SHA-256 output digest for every prompt. The candidate exact-match
rate varied by concurrency and did not reach the registered requirement of 1.0 at any level.

| Concurrency | Exact matches | Attempted | Exact-match rate |
|---:|---:|---:|---:|
| 1 | 108 | 360 | 0.3000 |
| 2 | 180 | 360 | 0.5000 |
| 4 | 188 | 360 | 0.5222 |
| 8 | 163 | 360 | 0.4528 |
| 16 | 141 | 360 | 0.3917 |
| 32 | 137 | 360 | 0.3806 |
| **All** | **917** | **2,160** | **0.4245** |

Across the ten prompt identities, the reference had
`[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` unique digests and the candidate had
`[7, 2, 4, 2, 4, 12, 6, 10, 4, 9]`. This establishes that the tested vLLM stack did not preserve
the reference's exact deterministic outputs and that candidate outputs varied across the combined
repetition/concurrency scope. Because plaintext was intentionally not retained, the record makes
no semantic-equivalence, task-correctness, or output-quality claim.

## Controlled setup

- Measurement date: 2026-08-02 UTC
- Provider: Lambda Cloud; recorded instance rate $1.99/hour
- GPU: one NVIDIA A100-SXM4-40GB, 40,960 MiB, driver 580.105.08
- CPU/RAM: AMD EPYC 7J13 64-Core Processor, 216 GiB observed RAM
- Model: `Qwen/Qwen3-8B`
- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Dtype: BF16 for both backends
- Dataset: `data/short_chat.jsonl`, ten prompts, 132–158 tokens (mean 145.3)
- Dataset SHA-256: `520495730dea75d35688682f0fe31ded568b5789a3397187df3bf333caa5f0e0`
- Generation: 64 maximum output tokens, temperature 0, top-p 1, seed 42
- Load: concurrency 1, 2, 4, 8, 16, and 32
- Trials: 120 measured requests per level, three repetitions
- Warm-up: ten excluded requests before every repetition/level pair
- Benchmark Git commit: `4faecdcb654f809763df60843c394ac38393a13c`, clean worktree
- Output evidence: SHA-256 digest only; no generated plaintext retained

The Transformers reference used Python 3.11.15, Torch 2.13.0+cu130, Transformers 5.14.1,
and Accelerate 1.14.0. Its complete package freeze is in `transformers.environment.txt`.

The candidate used vLLM 0.26.0 and Torch 2.11.0+cu130 from
`vllm/vllm-openai@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52`.
Prefix caching was disabled. `vllm.runtime.txt` records the exact image IDs and container command;
`vllm-gateway.environment.txt` contains the gateway package freeze.

The serialized reference provides the deterministic comparison target; it is not an optimized
serving implementation. Performance requirements apply to vLLM only, and the reference timing does
not affect this policy decision.

## Commands

Both backends used the same measured command, changing only the metadata and output filenames:

```bash
python -m inference_lab.benchmark.runner \
  --url http://127.0.0.1:8000 \
  --dataset data/short_chat.jsonl \
  --concurrency 1,2,4,8,16,32 \
  --requests 120 \
  --repetitions 3 \
  --warmup-requests 10 \
  --max-new-tokens 64 \
  --temperature 0 \
  --top-p 1 \
  --seed 42 \
  --output-evidence hash \
  --metadata results/qualification/<backend>.metadata.json \
  --output results/qualification/<backend>.jsonl
```

The recorded decision was produced with the committed policy:

```bash
inference-lab-evaluate \
  --reference results/qualification/transformers.jsonl \
  --reference-summary results/qualification/transformers.summary.json \
  --candidate results/qualification/vllm.jsonl \
  --candidate-summary results/qualification/vllm.summary.json \
  --policy policies/qwen3-8b-a100-short-chat.json \
  --output results/qualification/vllm.evaluation.json \
  --report-only
```

The complete resolved commands are also embedded in the summary manifests.

## Integrity audit

- Both raw result files contain 2,160 rows, with 2,160 successful requests and no failures.
- Every raw row has a prompt identity, request identity, exact token count, and output digest.
- Both summaries contain 18 trial summaries and six median aggregate rows.
- Recomputing raw/summary consistency and the policy report locally reproduced every recorded
  control, artifact, performance, equivalence, and eligibility field.
- The transferred archive SHA-256 matched on the GPU host and local workstation:
  `973d671202412efd07668d709b4e32ca7000f7f29fce30ef7461f52409322a52`.
- `SHA256SUMS` covers every published evidence file. Verify it from this directory with
  `shasum -a 256 -c SHA256SUMS` on macOS or `sha256sum -c SHA256SUMS` on Linux.

Raw hashes of the principal decision artifacts:

- Transformers rows: `61c8f6a0fa13e8b8ea1691a8ed0fc6683ca0b6b2417dc5a0e037bae5a6255a09`
- vLLM rows: `f680e0882a0ab7c6feaccb432f2d372925da2a3b390e7ec79cc381264d704547`
- Evaluation report: `5577a71ae70fdc9b0165a66742ada772f3ecb46462497cda5a155bcade0e63a0`

## Scope and next investigation

This decision applies only to the recorded hardware, model revision, software images, dataset,
generation controls, and policy. Hash-only exact matching is intentionally strict and cannot
distinguish harmless textual variation from a material model-output change.

A follow-up investigation should retain approved plaintext on a non-sensitive diagnostic workload,
locate first-token divergence, and test runtime controls that may affect deterministic decoding.
Any revised semantic or correctness policy must be registered before collecting the follow-up
measurements; this failed record remains immutable.
