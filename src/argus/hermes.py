"""Argus's Python client for HermesMQ's REST pub/sub surface (hermes-consumer spec, task 4.1).

Verified against Hermes ``PubSubRoutes.scala``:

* ensure subscription  — ``POST /v1/subscriptions``                   ``{subscriptionId, topicId}``
* pull                 — ``POST /v1/subscriptions/{id}/pull``         ``{max}``
* extend lease / nack  — ``POST /v1/subscriptions/{id}/modifyAckDeadline``
                         ``{ackIds, ackDeadlineSeconds}``
* ack                  — ``POST /v1/subscriptions/{id}/ack``          ``{ackIds}``
* publish              — ``POST /v1/topics/{id}/messages``            ``{payload, attributes}``

Message bodies are opaque UTF-8 ``payload`` strings + ``attributes`` maps
(see :mod:`argus.contract`).
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from argus.errors import TransientFailure


@dataclass(frozen=True, slots=True)
class PulledMessage:
    """One leased message from a pull."""

    ack_id: str
    payload: str
    attributes: dict[str, str]
    publish_time: str


class HermesClient:
    """Thin REST wrapper over a Hermes broker. Network faults surface as
    :class:`TransientFailure` so the consumer treats them as recoverable."""

    def __init__(self, base_url: str, *, client: httpx.Client | None = None, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self._base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HermesClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ensure_subscription(self, subscription_id: str, topic_id: str) -> None:
        """Idempotently create a subscription. 201 Created and 409 Conflict
        (already exists) are both success."""
        resp = self._post(
            "/v1/subscriptions",
            json={"subscriptionId": subscription_id, "topicId": topic_id},
        )
        if resp.status_code not in (httpx.codes.CREATED, httpx.codes.CONFLICT):
            self._raise(resp, "ensure_subscription")

    def pull(self, subscription_id: str, max_messages: int) -> list[PulledMessage]:
        """Pull up to ``max_messages`` (keep this small — backpressure, task 4.2)."""
        resp = self._post(
            f"/v1/subscriptions/{subscription_id}/pull",
            json={"max": max_messages},
        )
        if resp.status_code != httpx.codes.OK:
            self._raise(resp, "pull")
        return [
            PulledMessage(
                ack_id=m["ackId"],
                payload=m["payload"],
                attributes=m.get("attributes", {}),
                publish_time=m.get("publishTime", ""),
            )
            for m in resp.json().get("messages", [])
        ]

    def modify_ack_deadline(
        self, subscription_id: str, ack_ids: list[str], seconds: int
    ) -> None:
        """Extend a lease for long inference (task 4.3), or nack by passing 0."""
        if not ack_ids:
            return
        resp = self._post(
            f"/v1/subscriptions/{subscription_id}/modifyAckDeadline",
            json={"ackIds": ack_ids, "ackDeadlineSeconds": seconds},
        )
        if resp.status_code != httpx.codes.OK:
            self._raise(resp, "modify_ack_deadline")

    def ack(self, subscription_id: str, ack_ids: list[str]) -> None:
        """Acknowledge — only AFTER suggestions are durably published (task 4.3)."""
        if not ack_ids:
            return
        resp = self._post(
            f"/v1/subscriptions/{subscription_id}/ack",
            json={"ackIds": ack_ids},
        )
        if resp.status_code != httpx.codes.OK:
            self._raise(resp, "ack")

    def publish(self, topic_id: str, payload: str, attributes: dict[str, str]) -> str:
        """Publish a message; returns the broker message id."""
        resp = self._post(
            f"/v1/topics/{topic_id}/messages",
            json={"payload": payload, "attributes": attributes},
        )
        if resp.status_code != httpx.codes.ACCEPTED:
            self._raise(resp, "publish")
        return str(resp.json().get("messageId", ""))

    def _post(self, path: str, *, json: dict[str, object]) -> httpx.Response:
        try:
            return self._client.post(path, json=json)
        except httpx.HTTPError as exc:
            raise TransientFailure(f"Hermes request failed ({path}): {exc}") from exc

    @staticmethod
    def _raise(resp: httpx.Response, op: str) -> None:
        raise TransientFailure(f"Hermes {op} returned {resp.status_code}: {resp.text[:200]}")
