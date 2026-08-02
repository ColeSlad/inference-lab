import json
from collections.abc import AsyncIterator

import httpx

from inference_lab.backends.base import InferenceBackend
from inference_lab.schemas import BackendChunk, BackendResult, GenerateRequest


class OpenAICompatibleBackend(InferenceBackend):
    """Adapter for vLLM or another OpenAI-compatible completion server."""

    name = "openai-compatible"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout_s: float,
        model_revision: str | None = None,
        model_dtype: str | None = None,
    ) -> None:
        self.model = model
        self.model_revision = model_revision
        self.model_dtype = model_dtype
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def _payload(self, request: GenerateRequest, *, stream: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": request.prompt,
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def generate(self, request: GenerateRequest) -> BackendResult:
        payload = self._payload(request, stream=False)
        response = await self._client.post("/v1/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        return BackendResult(
            text=body["choices"][0].get("text", ""),
            prompt_tokens=(
                int(usage["prompt_tokens"]) if usage.get("prompt_tokens") is not None else None
            ),
            output_tokens=(
                int(usage["completion_tokens"])
                if usage.get("completion_tokens") is not None
                else None
            ),
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[BackendChunk]:
        prompt_tokens: int | None = None
        output_tokens: int | None = None

        async with self._client.stream(
            "POST", "/v1/completions", json=self._payload(request, stream=True)
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue

                event = json.loads(data)
                usage = event.get("usage") or {}
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens or 0))
                    output_tokens = int(usage.get("completion_tokens", output_tokens or 0))

                choices = event.get("choices") or []
                if choices:
                    piece = choices[0].get("text", "")
                    if piece:
                        yield BackendChunk(text=piece)

        yield BackendChunk(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            finished=True,
        )

    async def health(self) -> dict[str, object]:
        try:
            response = await self._client.get("/v1/models")
            response.raise_for_status()
            return await super().health()
        except Exception as exc:  # health endpoint should report rather than raise
            result: dict[str, object] = {
                "ok": False,
                "backend": self.name,
                "model": self.model,
                "error": str(exc),
            }
            if self.model_revision is not None:
                result["model_revision"] = self.model_revision
            if self.model_dtype is not None:
                result["model_dtype"] = self.model_dtype
            return result

    async def close(self) -> None:
        await self._client.aclose()
