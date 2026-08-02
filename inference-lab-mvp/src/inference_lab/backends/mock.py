import asyncio

from inference_lab.backends.base import InferenceBackend
from inference_lab.schemas import BackendChunk, BackendResult, GenerateRequest


class MockBackend(InferenceBackend):
    """Deterministic backend used for local development, CI, and benchmark validation."""

    name = "mock"

    def __init__(
        self,
        model: str,
        first_token_ms: float,
        token_ms: float,
        model_revision: str | None = None,
        model_dtype: str | None = None,
    ) -> None:
        self.model = model
        self.model_revision = model_revision
        self.model_dtype = model_dtype
        self.first_token_s = first_token_ms / 1000
        self.token_s = token_ms / 1000

    def _pieces(self, request: GenerateRequest) -> list[str]:
        vocabulary = [" inference", " systems", " need", " careful", " measurement", "."]
        return [vocabulary[index % len(vocabulary)] for index in range(request.max_new_tokens)]

    async def generate(self, request: GenerateRequest) -> BackendResult:
        pieces = []
        async for chunk in self.stream(request):
            if chunk.text:
                pieces.append(chunk.text)
            if chunk.finished:
                return BackendResult(
                    text="".join(pieces),
                    prompt_tokens=chunk.prompt_tokens or 0,
                    output_tokens=chunk.output_tokens or 0,
                )
        raise RuntimeError("Mock stream ended without a final chunk")

    async def stream(self, request: GenerateRequest):
        pieces = self._pieces(request)
        await asyncio.sleep(self.first_token_s)
        for index, piece in enumerate(pieces):
            if index > 0:
                await asyncio.sleep(self.token_s)
            yield BackendChunk(text=piece)

        yield BackendChunk(
            prompt_tokens=max(1, len(request.prompt.split())),
            output_tokens=len(pieces),
            finished=True,
        )
