# Qwen3-8B A100 qualification run

## Status

Completed 2026-08-02 UTC; deployment eligibility **FAIL** under the registered policy.

This protocol evaluated vLLM against the serialized Transformers reference on the existing
Qwen3-8B short-chat workload. The policy and protocol were committed before hardware was rented or
measurements were collected. The immutable artifacts and full analysis are in the
[qualification report](../benchmarks/2026-08-02-qwen3-8b-a100-qualification/README.md).

The run passed shared controls, artifact consistency, reference stability, evidence coverage, and
every C32 performance threshold. It failed strict equivalence: 917 of 2,160 candidate outputs
matched the stable reference output, and none of the ten candidate prompt outputs stayed stable
across the full repetition/concurrency scope. The failed report was retained without changing the
registered thresholds. Hash-only evidence does not establish whether non-matching output was
semantically equivalent or correct.

## Decision rule

The candidate is eligible under `policies/qwen3-8b-a100-short-chat.json` only when all of the
following pass:

- reference and candidate use the same clean Git commit, hardware record, driver, model revision,
  dataset digest, and generation controls
- every measured request succeeds and reports output evidence
- C32 vLLM median performance satisfies the registered regression envelope
- all candidate outputs exactly match the stable reference output for the same prompt
- candidate output remains stable across repetitions and concurrency 1–32

The performance thresholds were registered from the audited 2026-08-01 vLLM result with explicit
regression headroom; they are requirements for the next run, not new observations.

| Metric at concurrency 32 | Audited result | Registered requirement |
|---|---:|---:|
| Failure rate | 0.0 | `<= 0.0` |
| P95 TTFT | 203.099 ms | `<= 250 ms` |
| P95 end-to-end latency | 1,094.417 ms | `<= 1,250 ms` |
| Request throughput | 29.9092 requests/s | `>= 27 requests/s` |
| Output throughput | 1,914.1919 tokens/s | `>= 1,720 tokens/s` |

See the [audited source report](../benchmarks/2026-08-01-qwen3-8b-a100/README.md). Passing this policy
supports a claim about this exact hardware/model/workload configuration. It is not a general model
quality, safety, or production-readiness certification.

## Registered controls

- GPU: one NVIDIA A100-SXM4-40GB
- Model: `Qwen/Qwen3-8B`
- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Dtype: BF16 for both backends
- Dataset: `data/short_chat.jsonl`
- Dataset SHA-256: `520495730dea75d35688682f0fe31ded568b5789a3397187df3bf333caa5f0e0`
- Generation: 64 maximum output tokens, temperature 0, top-p 1, seed 42
- Load: concurrency 1, 2, 4, 8, 16, 32
- Trials: 120 measured requests per level, three repetitions
- Warm-up: 10 excluded requests before every repetition/level pair
- vLLM image: `vllm/vllm-openai@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52`
- vLLM prefix caching: disabled
- Output evidence: SHA-256 hash; plaintext output is not retained

Use one instance for both backends. The metadata files must contain identical `hardware` objects
and `software.gpu_driver` values. Backend-specific runtime and dependency fields may differ.

## Preflight

After installing the project environment, run these commands from the repository root on the GPU
instance:

```bash
git switch main
git pull --ff-only
git status --porcelain
git rev-parse HEAD

source .venv/bin/activate
python --version
pytest
ruff check .
docker compose --profile gpu config --quiet
sha256sum data/short_chat.jsonl
```

`git status --porcelain` must print nothing. Record the full commit returned by `git rev-parse`.
The dataset digest must match the registered value above.

Set the shared model controls in `.env` and pin the registered vLLM image:

```dotenv
INFERENCE_LAB_MODEL=Qwen/Qwen3-8B
INFERENCE_LAB_MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
INFERENCE_LAB_MODEL_DTYPE=bfloat16
VLLM_IMAGE=vllm/vllm-openai@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
```

Before each measured backend run, verify `/health` reports the expected backend, model, revision,
and dtype. Do not collect reference and candidate measurements concurrently.

## Measured command

Run this command once against the Transformers gateway and once against the vLLM gateway. Change
only the metadata and output paths:

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

Do not use smoke results as qualification evidence. Do not modify raw rows or summaries.

## Evaluate

After both measured runs finish:

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

Use `--report-only` for the first inspection so an expected FAIL still produces a report and does
not interrupt artifact collection. The report is authoritative: do not loosen the registered
policy in response to a failure. Investigate and repeat only when there is evidence of an invalid
run or implementation defect, retaining the original report.

## Archive before terminating the instance

```bash
sha256sum \
  results/qualification/transformers.metadata.json \
  results/qualification/transformers.jsonl \
  results/qualification/transformers.summary.json \
  results/qualification/vllm.metadata.json \
  results/qualification/vllm.jsonl \
  results/qualification/vllm.summary.json \
  results/qualification/vllm.evaluation.json \
  > results/qualification/SHA256SUMS

tar -czf results/qwen3-8b-a100-qualification.tar.gz \
  -C results qualification

sha256sum results/qwen3-8b-a100-qualification.tar.gz
```

Copy the archive and its printed SHA-256 digest off the instance. Verify the local digest before
terminating the rental. Environment freezes and exact container/image identities must also be
captured alongside the qualification artifacts, following the audited comparison procedure.
