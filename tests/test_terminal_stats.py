"""Periodic terminal stats: window rates, cache deltas, route integration and no GPU dependency."""

import asyncio
from types import SimpleNamespace

import httpx

from basal.server import http_app, parser
from basal.terminal_stats import TerminalStats
from tests.test_http_status import BODY, AppServer, Engine


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Cache:
    def __init__(self):
        self.values = {"hits": 0, "misses": 0}

    def stats(self, blocking=True):
        assert blocking is False  # logging must never wait for the cache lock
        return dict(self.values)


class Fake:
    def __init__(self):
        self.backend = SimpleNamespace(state_cache=Cache())
        self.running = self.queued = 0

    def load(self):
        return {"running": self.running, "queued": self.queued}


def test_window_rates_latency_cache_and_gpu():
    clock, server = Clock(), Fake()
    stats = TerminalStats(server, interval=10, clock=clock)
    stats.record(200, 3, 0.010)
    stats.record(200, 1, 0.020)
    stats.record(500, 0, 0.080)
    server.backend.state_cache.values.update(hits=3, misses=1)
    clock.now += 10
    line = stats.snapshot([{"index": 0, "util": 94, "mem_used": 8 * 2**30, "mem_total": 16 * 2**30}])
    assert "0.3 req/s" in line and "0.4 decisions/s" in line
    assert "p50=20.0ms p95=80.0ms" in line
    assert "running=0 waiting=0" in line and "errors=1" in line
    assert "cache_hit=75%" in line
    assert "GPU0=94% VRAM=8.0/16.0GiB" in line

    # Counters, including cache-hit rate, are interval deltas rather than lifetime averages.
    stats.record(200, 2, 0.005)
    server.backend.state_cache.values.update(hits=3, misses=2)
    clock.now += 10
    line = stats.snapshot()
    assert "0.1 req/s" in line and "0.2 decisions/s" in line
    assert "cache_hit=0%" in line and "errors=0" in line


def test_idle_is_silent_and_busy_without_completions_is_visible():
    clock, server = Clock(), Fake()
    stats = TerminalStats(server, clock=clock)
    clock.now += 10
    assert stats.snapshot() is None
    server.queued = 2
    clock.now += 10
    assert "waiting=2" in stats.snapshot()


def test_missing_cache_or_gpu_is_safe():
    clock, server = Clock(), Fake()
    server.backend.state_cache = None
    stats = TerminalStats(server, clock=clock)
    stats.record(200, 1, 0.001)
    clock.now += 10
    line = stats.snapshot([{"index": 0, "util": None, "mem_used": None, "mem_total": None}])
    assert "cache_hit" not in line and "GPU" not in line


def test_stats_flag_and_opt_out():
    assert parser().parse_args([]).stats_interval == 10.0
    assert parser().parse_args(["--stats-interval", "0"]).stats_interval == 0.0
    assert parser().parse_args(["--stats-interval", "2.5"]).stats_interval == 2.5


def test_http_collects_real_usage_and_failures_without_touching_response():
    server = AppServer(Engine())
    stats = TerminalStats(server)
    app = http_app(server, "fast", stats=stats)
    assert [r.path for r in app.routes if r.endpoint.__name__.startswith("stats_")] == [
        "/v1/systemone", "/v1/basal", "/v1/decisions"
    ]

    async def go():
        worker = asyncio.create_task(server.worker())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://basal") as c:
                one = await c.post("/v1/systemone", json=BODY)
                bad = await c.post("/v1/systemone", content=b"{not json")
                other = await c.get("/health")
                decisions = await c.post("/v1/decisions", json={
                    "input": "Invoice 1240 PLN.",
                    "questions": [{"type": "predicate", "name": "paid", "instructions": "Paid?"}],
                })
                return one, bad, other, decisions
        finally:
            worker.cancel()

    one, bad, other, decisions = asyncio.run(go())
    assert one.status_code == 200 and one.json()["answers"]["q"]["choice"] in ("a", "b")
    assert bad.status_code == 422 and bad.json()["code"] == "INVALID_REQUEST"
    assert other.status_code == 200 and decisions.status_code == 200
    assert stats.requests == 3 and stats.errors == 1
    assert stats.decisions == 2  # one System One question and one Decisions API question
    assert one.json()["usage"]["questions"] == 1
    assert len(stats.latencies) == 3


def test_stats_and_browser_ui_can_be_enabled_together():
    from basal.ui import UI

    server = AppServer(Engine())
    args = parser().parse_args(["--ui", "on"])
    ui = UI(server, args, "fast")
    stats = TerminalStats(server)
    app = http_app(server, "fast", ui=ui, stats=stats)

    async def go():
        worker = asyncio.create_task(server.worker())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://basal") as c:
                return await c.post("/v1/systemone", json=BODY)
        finally:
            worker.cancel()

    response = asyncio.run(go())
    assert response.status_code == 200
    assert stats.requests == 1 and stats.decisions == 1
    assert len(ui.ring) == 1  # the UI gets the same request, without double-counting
