from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    max_new_tokens: int = Field(default=64, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None


class BackendResult(BaseModel):
    text: str
    prompt_tokens: int
    output_tokens: int


class BackendChunk(BaseModel):
    text: str = ""
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    finished: bool = False


class GenerateResponse(BaseModel):
    request_id: str
    backend: str
    model: str
    text: str
    prompt_tokens: int
    output_tokens: int
    total_latency_ms: float


class StreamEvent(BaseModel):
    type: Literal["chunk", "done", "error"]
    request_id: str
    text: str = ""
    backend: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None
    total_latency_ms: float | None = None
    message: str | None = None
