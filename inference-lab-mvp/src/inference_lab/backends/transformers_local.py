import asyncio
from collections.abc import AsyncIterator
from threading import Thread

from inference_lab.backends.base import InferenceBackend
from inference_lab.schemas import BackendChunk, BackendResult, GenerateRequest


class TransformersBackend(InferenceBackend):
    """Single-process Hugging Face baseline; intentionally simple for comparison."""

    name = "transformers"

    def __init__(self, model: str, device: str = "auto", dtype: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'Transformers backend requires: pip install -e ".[hf]"'
            ) from exc

        self.model = model
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model)

        model_kwargs: dict[str, object] = {"device_map": device}
        if dtype != "auto":
            torch_dtype = getattr(torch, dtype, None)
            if torch_dtype is None:
                raise ValueError(f"Unsupported torch dtype: {dtype}")
            model_kwargs["torch_dtype"] = torch_dtype
        else:
            model_kwargs["torch_dtype"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        self._model.eval()
        self._generation_lock = asyncio.Lock()

    async def generate(self, request: GenerateRequest) -> BackendResult:
        text_parts: list[str] = []
        prompt_tokens = 0
        output_tokens = 0
        async for chunk in self.stream(request):
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.finished:
                prompt_tokens = chunk.prompt_tokens or 0
                output_tokens = chunk.output_tokens or 0
        return BackendResult(
            text="".join(text_parts),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[BackendChunk]:
        async with self._generation_lock:
            async for chunk in self._stream_serialized(request):
                yield chunk

    async def _stream_serialized(
        self, request: GenerateRequest
    ) -> AsyncIterator[BackendChunk]:
        from transformers import AsyncTextIteratorStreamer

        inputs = self._tokenizer(request.prompt, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        streamer = AsyncTextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=180.0,
        )
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.temperature > 0,
            "temperature": max(request.temperature, 1e-5),
            "top_p": request.top_p,
            "use_cache": True,
        }
        if request.seed is not None:
            self._torch.manual_seed(request.seed)

        errors: list[BaseException] = []

        def run_generation() -> None:
            try:
                with self._torch.inference_mode():
                    self._model.generate(**generation_kwargs)
            except BaseException as exc:  # propagate worker-thread failures
                errors.append(exc)

        worker = Thread(target=run_generation, daemon=True)
        worker.start()

        text_parts: list[str] = []
        async for piece in streamer:
            if piece:
                text_parts.append(piece)
                yield BackendChunk(text=piece)

        await asyncio.to_thread(worker.join)
        if errors:
            raise RuntimeError("Transformers generation failed") from errors[0]

        output_text = "".join(text_parts)
        output_tokens = len(self._tokenizer.encode(output_text, add_special_tokens=False))
        yield BackendChunk(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            finished=True,
        )
