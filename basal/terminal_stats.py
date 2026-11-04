"""Low-overhead vLLM-style terminal statistics for basal-serve.

One record per completed HTTP decision request; no tokenisation, response parsing,
CUDA calls or GPU synchronisation on the request path. Aggregation and optional
NVML telemetry happen outside request handlers.
"""

import asyncio
import math
import time
from collections import deque


class TerminalStats:
    """Periodic rates and latencies over completed HTTP requests."""

    MAX_LATENCIES = 16_384

    def __init__(self, server, interval=10.0, clock=time.perf_counter):
        if interval <= 0:
            raise ValueError("stats interval must be positive")
        self.server, self.interval, self.clock = server, interval, clock
        self.last = clock()
        self.requests = self.decisions = self.errors = 0
        self.latencies = deque(maxlen=self.MAX_LATENCIES)
        self.cache_last = None

    def record(self, status, questions, seconds):
        """Called once after an HTTP request completes, on the server event loop."""
        self.requests += 1
        self.errors += status >= 400
        if status < 400:
            self.decisions += questions
        self.latencies.append(seconds * 1000)

    def wrap(self, handler):
        """Keep logging off the engine thread and out of unsuccessful response bodies."""
        clock, record = self.clock, self.record

        async def recorded(request):
            start = clock()
            try:
                response = await handler(request)
            except Exception:
                record(500, 0, clock() - start)
                raise
            record(response.status_code, getattr(request.state, "basal_questions", 0), clock() - start)
            return response

        recorded.__name__ = f"stats_{handler.__name__}"
        return recorded

    @staticmethod
    def percentile(sorted_values, percentile):
        """Nearest-rank percentile, matching the UI's reporting convention."""
        if not sorted_values:
            return None
        return sorted_values[max(1, math.ceil(percentile * len(sorted_values))) - 1]

    def snapshot(self, gpus=()):
        """Format one completed interval. Return None when the server is idle."""
        now = self.clock()
        elapsed = max(now - self.last, 1e-9)
        self.last = now
        requests, decisions, errors = self.requests, self.decisions, self.errors
        self.requests = self.decisions = self.errors = 0
        latencies = sorted(self.latencies)
        self.latencies.clear()

        load = self.server.load()
        cache = getattr(self.server.backend, "state_cache", None)
        cache_part = ""
        if cache is not None:
            # Nonblocking: never delay the engine for a stats line.
            current = cache.stats(blocking=False)
            if current is not None:
                previous = self.cache_last or {"hits": 0, "misses": 0}
                hits = max(0, current["hits"] - previous["hits"])
                misses = max(0, current["misses"] - previous["misses"])
                if hits + misses:
                    cache_part = f" | cache_hit={100 * hits / (hits + misses):.0f}%"
                self.cache_last = current

        if not requests and not load["running"] and not load["queued"]:
            return None

        p50 = self.percentile(latencies, 0.50)
        p95 = self.percentile(latencies, 0.95)
        latency = f"p50={p50:.1f}ms p95={p95:.1f}ms" if p50 is not None else "p50=- p95=-"
        line = (f"basal.stats: {requests / elapsed:.1f} req/s | {decisions / elapsed:.1f} decisions/s"
                f" | {latency} | running={load['running']} waiting={load['queued']}"
                f" | errors={errors}{cache_part}")
        for gpu in gpus:
            util, used, total = gpu.get("util"), gpu.get("mem_used"), gpu.get("mem_total")
            if util is None and (used is None or total is None):
                continue
            part = f"GPU{gpu['index']}"
            if util is not None:
                part += f"={util:.0f}%"
            if used is not None and total is not None:
                part += f" VRAM={used / 2**30:.1f}/{total / 2**30:.1f}GiB"
            line += f" | {part}"
        return line

    async def run(self):
        """Sample telemetry only on the logging interval, never with torch.cuda."""
        from .ui import Telemetry  # reuses the UI's optional NVML integration

        telemetry = Telemetry()
        try:
            await asyncio.to_thread(telemetry.open)
        except Exception:  # noqa: BLE001  (telemetry must never affect serving)
            pass
        while True:
            await asyncio.sleep(self.interval)
            try:
                sample = await asyncio.to_thread(telemetry.read)
                gpus = sample.get("gpus", ())
            except Exception:  # noqa: BLE001  (missing/stopped NVML must not stop statistics)
                gpus = ()
            line = self.snapshot(gpus)
            if line is not None:
                print(line, flush=True)
