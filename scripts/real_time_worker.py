"""Real-time event worker (Batch 3).

Reads user interaction events from Kafka (via Redpanda wire protocol) and
updates Redis user state.  When an event is processed the worker can
trigger a recommendation refresh — the API picks up the updated state
automatically because it reads ``state.events_buffer`` / Redis on each
request (see :class:`AppState` in ``api/main.py``).

Usage (Docker Compose, after ``docker compose up``):

    uv run python scripts/real_time_worker.py

Standalone (no Docker, no Kafka):

    UV_RTK_MODE=standalone uv run python scripts/real_time_worker.py

Mock mode (UV_RTK_MODE=mock) — generates a few synthetic events every second
so you can test the API ``/event`` endpoint without any broker.

Environment variables (all optional, with sensible defaults):

  RTK_REDIS_URL       Redis connection URL  (default: ``redis://localhost:6379/0``)
  RTK_KAFKA_BOOTSTRAP Kafka bootstrap servers (default: ``redpanda:9092``)
  RTK_KAFKA_TOPIC     Topic name (default: ``recsys-events``)
  RTK_MODE            ``docker`` / ``standalone`` / ``mock``
"""

from __future__ import annotations

import json
import os
import time

# ---- Redis state (used by API AppState as fallback) -------------------------

try:
    import redis  # type: ignore

    _redis_client = redis.Redis.from_url(
        os.environ.get("RTK_REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )

    def redis_set(key: str, value: str) -> None:
        _redis_client.set(key, value)

    def redis_get(key: str) -> str | None:
        result = _redis_client.get(key)
        return result  # type: ignore[return-value]

except Exception:  # pragma: no cover — Redis not available
    # In-memory fallback so the script still runs for demo purposes.
    _store: dict[str, str] = {}

    def redis_set(key: str, value: str) -> None:
        _store[key] = value

    def redis_get(key: str) -> str | None:
        return _store.get(key)

# ---- Kafka / Redpanda consumer factory --------------------------------------

_consumer_implementation: str | None = None


def _make_consumer_kt_real(bootstrap_servers: str, topic: str) -> KafkaConsumer:  # noqa: ANN001
    """Real implementation using kafka-python when available."""
    from kafka import KafkaConsumer  # type: ignore

    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,  # type: ignore[return-value]
    )


def _make_consumer_kt_dummy(_bootstrap: str, _topic: str) -> _DummyKafkaConsumer:  # noqa: ANN001
    """Dummy implementation used when kafka-python is not installed."""
    return _DummyKafkaConsumer(_topic)


# Select the implementation at module load time.
try:
    from kafka import KafkaConsumer  # type: ignore

    _consumer_implementation = "real"
    _make_consumer = _make_consumer_kt_real  # type: ignore[assignment]
except Exception:  # pragma: no cover — kafka-python not installed
    _consumer_implementation = "dummy"
    _make_consumer = _make_consumer_kt_dummy


# ---- Dummy consumer class (used when kafka-python unavailable) -------------

class _DummyKafkaConsumer:  # noqa: ANN001
    """Simulates a Kafka consumer in ``mock`` / ``standalone`` mode."""

    def __init__(self, _topic: str, _rate: float = 0.5) -> None:
        self._topic = _topic
        self._events = [
            {"user_id": "ml_user_123", "item_id": "ml_item_42", "event_type": "click"},
            {"user_id": "ml_user_123", "item_id": "ml_item_77", "event_type": "view"},
            {"user_id": "ml_user_456", "item_id": "ml_item_42", "event_type": "like"},
        ]
        self._idx = 0

    def __iter__(self):  # noqa: ANN001
        return self

    def __next__(self):  # noqa: ANN001
        if self._idx >= len(self._events):
            raise StopIteration
        ev = self._events[self._idx]
        self._idx += 1
        return ev

    def close(self) -> None:  # noqa: ANN001
        pass


# ---- Entry point  ----------------------------------------------------------

MODE = os.environ.get("RTK_MODE", "docker").lower()  # docker | standalone | mock

if MODE == "mock":
    # -----------------------------------------------------------------------
    # Generates a few synthetic events every second so the API /event endpoint
    # can be tested without any broker.
    # -----------------------------------------------------------------------

    EVENTS = [
        {"user_id": "ml_user_123", "item_id": "ml_item_42", "event_type": "click"},
        {"user_id": "ml_user_123", "item_id": "ml_item_77", "event_type": "view"},
        {"user_id": "ml_user_456", "item_id": "ml_item_42", "event_type": "like"},
        {"user_id": "ml_user_789", "item_id": "ml_item_99", "event_type": "purchase"},
    ]

    print("[worker] Mock mode: publishing synthetic events every 1s", flush=True)
    try:
        for ev in EVENTS:
            user_key = "rtk:event:" + ev["user_id"]
            redis_set(user_key, json.dumps(ev))
            print(f"[worker] sent event {ev}", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("[worker] stopped", flush=True)

elif MODE == "standalone":
    # -----------------------------------------------------------------------
    # Connects to Redpanda (Kafka wire protocol) and processes real events.
    # -----------------------------------------------------------------------
    bootstrap = os.environ.get("RTK_KAFKA_BOOTSTRAP", "redpanda:9092")
    topic = os.environ.get("RTK_KAFKA_TOPIC", "recsys-events")

    consumer = _make_consumer(bootstrap, topic)
    print(f"[worker] subscribed to ``{topic}`` on {bootstrap}", flush=True)

    try:
        for msg in consumer:
            ev = msg.value  # type: ignore[union-attr]
            if ev is None:
                continue
            user_key = f"rtk:user:{ev['user_id']}"
            prev = redis_get(user_key)
            # Build updated state: keep existing events and append new one.
            existing = json.loads(prev) if prev else {}
            existing["events"] = existing.get("events", []) + [ev]
            payload = json.dumps(existing)
            redis_set(user_key, payload)
            print(
                f"[worker] processed event user={ev['user_id']} item={ev['item_id']} "
                f"type={ev['event_type']}",
                flush=True,
            )
            # Optionally trigger a recommendation refresh here — the API will
            # re-read the updated user state on the next request.
    except KeyboardInterrupt:
        print("[worker] stopped", flush=True)
        consumer.close()

else:  # MODE == "docker"
    # -----------------------------------------------------------------------
    # Production-ish Docker mode: expects Redpanda + Redis services reachable
    # via docker-compose service names (``redpanda`` / ``redis``).
    # -----------------------------------------------------------------------
    bootstrap = os.environ.get("RTK_KAFKA_BOOTSTRAP", "redpanda:9092")
    topic = os.environ.get("RTK_KAFKA_TOPIC", "recsys-events")

    consumer = _make_consumer(bootstrap, topic)
    print(f"[worker] subscribed to ``{topic}`` on {bootstrap}", flush=True)

    try:
        for msg in consumer:
            ev = msg.value  # type: ignore[union-attr]
            if ev is None:
                continue
            user_key = f"rtk:user:{ev['user_id']}"
            prev = redis_get(user_key)
            # Build updated state: keep existing events and append new one.
            existing = json.loads(prev) if prev else {}
            existing["events"] = existing.get("events", []) + [ev]
            payload = json.dumps(existing)
            redis_set(user_key, payload)
            print(
                f"[worker] processed event user={ev['user_id']} item={ev['item_id']} "
                f"type={ev['event_type']}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("[worker] stopped", flush=True)
        consumer.close()
