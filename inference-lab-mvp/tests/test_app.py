import json

from fastapi.testclient import TestClient

from inference_lab.app import create_app
from inference_lab.config import Settings


def test_health_and_generate() -> None:
    app = create_app(
        Settings(
            backend="mock",
            model="mock-test",
            mock_first_token_ms=0,
            mock_token_ms=0,
        )
    )
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["backend"] == "mock"

        response = client.post(
            "/v1/generate",
            json={"prompt": "hello", "max_new_tokens": 3},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["output_tokens"] == 3
        assert body["model"] == "mock-test"


def test_stream_emits_chunks_and_done_event() -> None:
    app = create_app(
        Settings(
            backend="mock",
            model="mock-test",
            mock_first_token_ms=0,
            mock_token_ms=0,
        )
    )
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/generate/stream",
            json={"prompt": "hello world", "max_new_tokens": 2},
        ) as response:
            events = []
            for line in response.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line.removeprefix("data:").strip()))

    assert response.status_code == 200
    assert [event["type"] for event in events] == ["chunk", "chunk", "done"]
    assert events[-1]["output_tokens"] == 2
    assert events[-1]["prompt_tokens"] == 2
