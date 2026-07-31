import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from prometheus_client import make_asgi_app

from inference_lab.backends.factory import build_backend
from inference_lab.config import Settings, get_settings
from inference_lab.metrics import GENERATED_TOKENS, IN_FLIGHT, REQUEST_LATENCY, REQUESTS, TTFT
from inference_lab.schemas import GenerateRequest, GenerateResponse, StreamEvent


def _sse(event: StreamEvent) -> str:
    return f"data: {event.model_dump_json(exclude_none=True)}\n\n"


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.backend = build_backend(resolved_settings)
        yield
        await app.state.backend.close()

    app = FastAPI(
        title="Inference Lab",
        version="0.1.0",
        description="Pluggable LLM inference gateway with benchmark-friendly streaming.",
        lifespan=lifespan,
    )
    app.mount("/metrics", make_asgi_app())

    @app.get("/")
    async def root(request: Request) -> dict[str, object]:
        backend = request.app.state.backend
        return {
            "name": "inference-lab",
            "backend": backend.name,
            "model": backend.model,
            "docs": "/docs",
            "metrics": "/metrics",
        }

    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        result = await request.app.state.backend.health()
        if not result.get("ok"):
            raise HTTPException(status_code=503, detail=result)
        return result

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
        backend = request.app.state.backend
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        labels = {"backend": backend.name, "mode": "nonstream"}

        IN_FLIGHT.labels(**labels).inc()
        try:
            result = await backend.generate(payload)
            latency_s = time.perf_counter() - started
            REQUESTS.labels(**labels, status="ok").inc()
            REQUEST_LATENCY.labels(**labels).observe(latency_s)
            if result.output_tokens is not None:
                GENERATED_TOKENS.labels(backend=backend.name).inc(result.output_tokens)
            return GenerateResponse(
                request_id=request_id,
                backend=backend.name,
                model=backend.model,
                text=result.text,
                prompt_tokens=result.prompt_tokens,
                output_tokens=result.output_tokens,
                total_latency_ms=latency_s * 1000,
            )
        except Exception as exc:
            REQUESTS.labels(**labels, status="error").inc()
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            IN_FLIGHT.labels(**labels).dec()

    @app.post("/v1/generate/stream")
    async def generate_stream(payload: GenerateRequest, request: Request) -> StreamingResponse:
        backend = request.app.state.backend
        request_id = str(uuid.uuid4())

        async def events() -> AsyncIterator[str]:
            started = time.perf_counter()
            first_text_at: float | None = None
            prompt_tokens: int | None = None
            output_tokens: int | None = None
            backend_finished = False
            labels = {"backend": backend.name, "mode": "stream"}
            IN_FLIGHT.labels(**labels).inc()

            try:
                async for chunk in backend.stream(payload):
                    if chunk.text:
                        if first_text_at is None:
                            first_text_at = time.perf_counter()
                            TTFT.labels(backend=backend.name).observe(first_text_at - started)
                        yield _sse(
                            StreamEvent(
                                type="chunk",
                                request_id=request_id,
                                text=chunk.text,
                            )
                        )
                    if chunk.finished:
                        backend_finished = True
                        prompt_tokens = chunk.prompt_tokens
                        output_tokens = chunk.output_tokens

                if not backend_finished:
                    raise RuntimeError("Backend stream ended without a final chunk")

                finished = time.perf_counter()
                latency_s = finished - started
                ttft_ms = (
                    (first_text_at - started) * 1000 if first_text_at is not None else None
                )
                REQUESTS.labels(**labels, status="ok").inc()
                REQUEST_LATENCY.labels(**labels).observe(latency_s)
                if output_tokens is not None:
                    GENERATED_TOKENS.labels(backend=backend.name).inc(output_tokens)
                yield _sse(
                    StreamEvent(
                        type="done",
                        request_id=request_id,
                        backend=backend.name,
                        model=backend.model,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        ttft_ms=ttft_ms,
                        total_latency_ms=latency_s * 1000,
                    )
                )
            except Exception as exc:
                REQUESTS.labels(**labels, status="error").inc()
                yield _sse(
                    StreamEvent(
                        type="error",
                        request_id=request_id,
                        backend=backend.name,
                        model=backend.model,
                        message=str(exc),
                    )
                )
            finally:
                IN_FLIGHT.labels(**labels).dec()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
