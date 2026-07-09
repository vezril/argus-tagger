import json

import httpx
import pytest

from argus.errors import TransientFailure
from argus.hermes import HermesClient


def _client(handler: object) -> HermesClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    http = httpx.Client(transport=transport, base_url="http://hermes")
    return HermesClient("http://hermes", client=http)


def test_ensure_subscription_treats_created_and_conflict_as_success() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        # First call: created. Second: already exists (409).
        code = 201 if len(seen) == 1 else 409
        return httpx.Response(code)

    client = _client(handler)
    client.ensure_subscription("argus.media.tag", "media.tag")
    client.ensure_subscription("argus.media.tag", "media.tag")  # idempotent, no raise
    assert seen == ["/v1/subscriptions", "/v1/subscriptions"]


def test_pull_parses_messages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/subscriptions/sub/pull"
        assert json.loads(request.content) == {"max": 2}
        return httpx.Response(
            200,
            json={
                "messages": [
                    {
                        "ackId": "ack-1",
                        "payload": "{}",
                        "attributes": {"postId": "p1"},
                        "publishTime": "2026-07-09T00:00:00Z",
                    }
                ]
            },
        )

    messages = _client(handler).pull("sub", 2)
    assert len(messages) == 1
    assert messages[0].ack_id == "ack-1"
    assert messages[0].attributes == {"postId": "p1"}


def test_publish_returns_message_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/topics/media.suggestions/messages"
        return httpx.Response(202, json={"messageId": "m-7"})

    assert _client(handler).publish("media.suggestions", "{}", {"postId": "p1"}) == "m-7"


def test_ack_and_modify_skip_empty_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not fire
        raise AssertionError("should not POST for empty ack list")

    client = _client(handler)
    client.ack("sub", [])
    client.modify_ack_deadline("sub", [], 0)


def test_network_error_becomes_transient_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(TransientFailure):
        _client(handler).pull("sub", 1)


def test_unexpected_status_becomes_transient_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(TransientFailure):
        _client(handler).ack("sub", ["ack-1"])
