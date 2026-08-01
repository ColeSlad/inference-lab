from inference_lab.backends.base import InferenceBackend
from inference_lab.backends.mock import MockBackend
from inference_lab.backends.openai_compatible import OpenAICompatibleBackend
from inference_lab.config import Settings


def build_backend(settings: Settings) -> InferenceBackend:
    if settings.backend == "mock":
        return MockBackend(
            model=settings.model,
            first_token_ms=settings.mock_first_token_ms,
            token_ms=settings.mock_token_ms,
        )
    if settings.backend == "openai":
        return OpenAICompatibleBackend(
            model=settings.model,
            model_revision=settings.model_revision,
            model_dtype=settings.model_dtype,
            base_url=settings.upstream_url,
            api_key=settings.upstream_api_key,
            timeout_s=settings.request_timeout_s,
        )
    if settings.backend == "transformers":
        from inference_lab.backends.transformers_local import TransformersBackend

        return TransformersBackend(
            model=settings.model,
            revision=settings.model_revision,
            device=settings.transformers_device,
            dtype=settings.model_dtype,
        )
    raise ValueError(f"Unknown backend: {settings.backend}")
