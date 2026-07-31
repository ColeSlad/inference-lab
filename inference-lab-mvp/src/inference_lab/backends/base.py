from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from inference_lab.schemas import BackendChunk, BackendResult, GenerateRequest


class InferenceBackend(ABC):
    name: str
    model: str

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> BackendResult:
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: GenerateRequest) -> AsyncIterator[BackendChunk]:
        raise NotImplementedError

    async def health(self) -> dict[str, object]:
        return {"ok": True, "backend": self.name, "model": self.model}

    async def close(self) -> None:
        return None
