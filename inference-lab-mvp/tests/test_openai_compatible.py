import json

import httpx
import pytest

from inference_lab.backends.openai_compatible import OpenAICompatibleBackend
from inference_lab.schemas import GenerateRequest


@pytest.mark.asyncio
async def test_missing_upstream_usage_remains_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/completions" and json.loads(request.content)["stream"]:
            return httpx.Response(
                200,
                text='data: {"choices":[{"text":" generated text"}]}\n\ndata: [DONE]\n\n',
            )
        return httpx.Response(200, json={"choices": [{"text": " generated text"}]})

    backend = OpenAICompatibleBackend(
        model="test-model",
        base_url="http://test",
        api_key="test",
        timeout_s=1,
    )
    await backend._client.aclose()
    backend._client = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    request = GenerateRequest(prompt="hello", max_new_tokens=2)

    result = await backend.generate(request)
    chunks = [chunk async for chunk in backend.stream(request)]
    await backend.close()

    assert result.prompt_tokens is None
    assert result.output_tokens is None
    assert chunks[-1].finished is True
    assert chunks[-1].prompt_tokens is None
    assert chunks[-1].output_tokens is None
