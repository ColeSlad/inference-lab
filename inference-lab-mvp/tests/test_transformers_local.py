import asyncio
import sys
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace

import pytest

from inference_lab.backends.transformers_local import TransformersBackend
from inference_lab.schemas import BackendChunk, GenerateRequest


def test_transformers_backend_pins_tokenizer_and_model_revision(monkeypatch) -> None:
    tokenizer_calls: list[tuple[str, dict[str, object]]] = []
    model_calls: list[tuple[str, dict[str, object]]] = []

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(model: str, **kwargs: object) -> object:
            tokenizer_calls.append((model, kwargs))
            return object()

    class FakeLoadedModel:
        def eval(self) -> None:
            return None

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(model: str, **kwargs: object) -> FakeLoadedModel:
            model_calls.append((model, kwargs))
            return FakeLoadedModel()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeModelFactory
    fake_transformers.AutoTokenizer = FakeTokenizerFactory
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))

    backend = TransformersBackend(
        "test-model",
        revision="abc123",
        device="cuda",
        dtype="bfloat16",
    )

    assert tokenizer_calls == [("test-model", {"revision": "abc123"})]
    assert model_calls == [
        (
            "test-model",
            {"device_map": "cuda", "torch_dtype": "bf16", "revision": "abc123"},
        )
    ]
    assert backend.model_revision == "abc123"
    assert backend.model_dtype == "bfloat16"


@pytest.mark.asyncio
async def test_transformers_backend_serializes_generation() -> None:
    backend = TransformersBackend.__new__(TransformersBackend)
    backend._generation_lock = asyncio.Lock()
    active = 0
    maximum_active = 0

    async def fake_stream(request: GenerateRequest) -> AsyncIterator[BackendChunk]:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        yield BackendChunk(text=request.prompt)
        await asyncio.sleep(0)
        active -= 1
        yield BackendChunk(prompt_tokens=1, output_tokens=1, finished=True)

    backend._stream_serialized = fake_stream

    async def consume(prompt: str) -> None:
        request = GenerateRequest(prompt=prompt, max_new_tokens=1, temperature=1, seed=42)
        async for _ in backend.stream(request):
            await asyncio.sleep(0)

    await asyncio.gather(consume("first"), consume("second"))

    assert maximum_active == 1
