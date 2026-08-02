from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from inference_lab.schemas import BackendChunk, BackendResult, GenerateRequest


class InferenceBackend(ABC):
    name: str
    model: str
    model_revision: str | None = None
    model_dtype: str | None = None

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> BackendResult:
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: GenerateRequest) -> AsyncIterator[BackendChunk]:
        raise NotImplementedError

    async def health(self) -> dict[str, object]:
        result: dict[str, object] = {
            "ok": True,
            "backend": self.name,
            "model": self.model,
        }
        if self.model_revision is not None:
            result["model_revision"] = self.model_revision
        if self.model_dtype is not None:
            result["model_dtype"] = self.model_dtype
        return result

    async def close(self) -> None:
        return None
