.PHONY: install dev test lint serve bench docker-mock docker-gpu

install:
	python -m pip install -e ".[dev]"

dev:
	uvicorn inference_lab.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check .

serve:
	uvicorn inference_lab.app:app --host 0.0.0.0 --port 8000

bench:
	python -m inference_lab.benchmark.runner \
		--url http://localhost:8000 \
		--dataset data/prompts.jsonl \
		--concurrency 1,4,8 \
		--requests 24 \
		--max-new-tokens 32 \
		--output results/latest.jsonl

docker-mock:
	docker compose up --build gateway prometheus

docker-gpu:
	INFERENCE_LAB_BACKEND=openai docker compose --profile gpu up --build
