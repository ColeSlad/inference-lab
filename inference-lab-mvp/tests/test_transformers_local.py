import asyncio
from collections.abc import AsyncIterator

import pytest

from inference_lab.backends.transformers_local import TransformersBackend
from inference_lab.schemas import BackendChunk, GenerateRequest


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
