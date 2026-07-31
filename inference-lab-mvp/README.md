# Inference Lab

A portfolio-ready MVP for comparing LLM inference backends under reproducible concurrent load.

The project is intentionally focused on **ML systems evidence**, not another chatbot UI. It gives you a stable streaming API, pluggable backends, Prometheus instrumentation, an async load generator, raw experiment artifacts, and a roadmap toward quantization, prefix caching, speculative decoding, Triton, and multi-GPU serving.

## What the MVP demonstrates

- backend-neutral inference service design
- streaming generation and time-to-first-token measurement
- concurrent load generation with controlled workloads
- P50/P95 latency and throughput analysis
- observability with Prometheus
- Dockerized vLLM deployment
- deterministic, GPU-free CI
- clean extension points for deeper inference work

## Architecture

```mermaid
flowchart LR
    Dataset --> Benchmark[Async benchmark runner]
    Benchmark -->|HTTP/SSE| Gateway[FastAPI gateway]
    Gateway --> Adapter{Backend adapter}
    Adapter --> Mock
    Adapter --> Transformers[HF Transformers]
    Adapter --> vLLM[vLLM server]
    Gateway --> Metrics[Prometheus metrics]
    vLLM --> Metrics
    Benchmark --> Results[JSONL + summary + plots]
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for component and design details.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | Python 3.11, FastAPI, Pydantic | Typed async API with automatic OpenAPI docs |
| Streaming | Server-sent events | Simple client-observable TTFT measurement |
| Baseline runtime | Hugging Face Transformers + PyTorch | Transparent reference implementation |
| Optimized runtime | vLLM OpenAI-compatible server | Production-oriented continuous serving runtime |
| Load generation | asyncio + HTTPX | Controlled concurrency without another service |
| Metrics | Prometheus client + Prometheus | Histograms, counters, in-flight load, token counts |
| Artifacts | JSONL and JSON | Easy to inspect, version, and analyze |
| Packaging | `pyproject.toml`, Docker Compose | Reproducible local and GPU setup |
| Quality | pytest, Ruff, GitHub Actions | GPU-free validation through the mock backend |

## Repository layout

```text
src/inference_lab/
  app.py                       FastAPI gateway and SSE contract
  backends/
    base.py                    Backend interface
    mock.py                    Deterministic CI backend
    transformers_local.py      Simple local PyTorch baseline
    openai_compatible.py       vLLM adapter
  benchmark/
    runner.py                  Concurrent benchmark CLI
    stats.py                   Percentiles and throughput summaries
scripts/plot_results.py        Result plotting
data/prompts.jsonl             Starter workload
docs/                          Architecture and experiment plan
tests/                         API and statistics tests
```

## Start in mock mode

The mock backend validates the full service and benchmark path without downloading a model.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn inference_lab.app:app --reload
```

In another terminal:

```bash
python -m inference_lab.benchmark.runner \
  --url http://localhost:8000 \
  --dataset data/prompts.jsonl \
  --concurrency 1,4,8 \
  --requests 24 \
  --max-new-tokens 32 \
  --output results/mock.jsonl
```

Outputs:

- `results/mock.jsonl`: one row per request
- `results/mock.summary.json`: aggregate metrics per concurrency level

API documentation is available at `http://localhost:8000/docs`; Prometheus metrics are at `http://localhost:8000/metrics/`.

## Run the Hugging Face baseline

Install the optional PyTorch/Transformers dependencies:

```bash
pip install -e ".[dev,hf]"
cp .env.example .env
```

Set:

```dotenv
INFERENCE_LAB_BACKEND=transformers
INFERENCE_LAB_MODEL=Qwen/Qwen3-0.6B
INFERENCE_LAB_TRANSFORMERS_DEVICE=auto
INFERENCE_LAB_TRANSFORMERS_DTYPE=auto
```

Then start the gateway and run the same benchmark command. The baseline is deliberately simple so that optimized serving behavior has a clear comparison point.

## Run vLLM on an NVIDIA GPU

Prerequisites:

- Docker and Docker Compose
- NVIDIA driver
- NVIDIA Container Toolkit
- enough GPU memory for the selected model

```bash
cp .env.example .env
INFERENCE_LAB_BACKEND=openai docker compose --profile gpu up --build
```

The gateway is exposed on port `8000`, vLLM on port `8001`, and Prometheus on port `9090`.

Run the benchmark:

```bash
python -m inference_lab.benchmark.runner \
  --url http://localhost:8000 \
  --dataset data/prompts.jsonl \
  --concurrency 1,2,4,8,16,32 \
  --requests 120 \
  --max-new-tokens 64 \
  --output results/vllm.jsonl
```

For published results, replace floating `latest` image tags with an exact image tag or digest and record the GPU, driver, model revision, and command.

## Plot a summary

```bash
pip install -e ".[plots]"
python scripts/plot_results.py results/vllm.summary.json
```

The script creates separate throughput, TTFT, and latency figures under `results/plots/`.

## API examples

Non-streaming:

```bash
curl -s http://localhost:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "Explain KV caching in two paragraphs.",
    "max_new_tokens": 64,
    "temperature": 0
  }'
```

Streaming:

```bash
curl -N http://localhost:8000/v1/generate/stream \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "Explain continuous batching.",
    "max_new_tokens": 64,
    "temperature": 0
  }'
```

## Fair benchmark checklist

- use the same model revision and tokenizer
- keep prompt order and generation parameters fixed
- warm up each runtime before collecting results
- separate short-prefill, long-prefill, decode-heavy, and repeated-prefix workloads
- run each configuration multiple times
- retain raw request results
- report failures and outliers
- record hardware, image digest, Git commit, and runtime flags

See [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) for the first publishable study.

## High-value next milestones

1. Compare BF16/FP16, INT8, and INT4 for speed, memory, and quality.
2. Build repeated-prefix workloads and measure prefix-caching impact.
3. Add speculative decoding and record draft-token acceptance rates.
4. Profile with PyTorch Profiler or Nsight and implement one Triton kernel.
5. Compare single-GPU and tensor-parallel multi-GPU serving.
6. Add a small reasoning/coding quality suite so performance is never reported without correctness.

## Resume outcome to target

Do not add the project to your resume until you have real hardware results. A strong final bullet should resemble:

> Built a reproducible LLM serving benchmark across Hugging Face and vLLM, increasing output throughput from X to Y tokens/s at concurrency 16 while tracking P95 TTFT, latency, GPU memory, and correctness.

Replace every placeholder with measured results and state the model and hardware in the repository report.
