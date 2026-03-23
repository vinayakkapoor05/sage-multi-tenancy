from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from sage_mt.config import GatewaySettings
from sage_mt.models import Engine, JobRecord, JobSubmit, LatencyClass, TorchPayload, VllmPayload
from sage_mt.resource_layer import GPUResourceLayer

log = logging.getLogger(__name__)


@dataclass
class _TorchItem:
    job_id: str
    tenant_id: str
    latency_class: LatencyClass
    payload: TorchPayload
    expected_runtime_ms: float
    enqueued_at: float


@dataclass
class _VllmItem:
    job_id: str
    tenant_id: str
    latency_class: LatencyClass
    payload: VllmPayload
    expected_runtime_ms: float
    enqueued_at: float


@dataclass
class InferenceScheduler:
    settings: GatewaySettings
    jobs: dict[str, JobRecord] = field(default_factory=dict)
    _client: httpx.AsyncClient | None = None
    _task: asyncio.Task[None] | None = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event)

    _resource_layer: GPUResourceLayer | None = None

    _vllm_rt: deque[_VllmItem] = field(default_factory=deque)
    _vllm_by_tenant: dict[str, deque[_VllmItem]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    _torch_by_tenant: dict[str, deque[_TorchItem]] = field(
        default_factory=lambda: defaultdict(deque)
    )

    _torch_deadline: float | None = None

    _tenant_recent_gpu_ms: dict[str, float] = field(default_factory=dict)
    _tenant_total_gpu_ms: dict[str, float] = field(default_factory=dict)
    _tenant_deadline_miss: dict[str, int] = field(default_factory=dict)
    _last_decay_ts: float = field(default_factory=time.monotonic)
    _vllm_kv_cache_usage_ratio: float | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        self._resource_layer = GPUResourceLayer(self.settings)
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(), name="inference-scheduler")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client:
            await self._client.aclose()
            self._client = None

    async def submit(self, body: JobSubmit) -> str:
        now_ms = int(time.time() * 1000)
        deadline_at = now_ms + body.deadline_ms if body.deadline_ms else None
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = JobRecord(
            id=job_id,
            status="queued",
            tenant_id=body.tenant_id,
            latency_class=body.latency_class,
            engine=body.engine,
            submitted_at_ms=now_ms,
            deadline_at_ms=deadline_at,
        )

        expected_runtime_ms = body.expected_runtime_ms or self._default_cost_ms(body)
        enqueued_at = time.monotonic()

        if body.engine == Engine.vllm:
            if not body.vllm:
                self._fail(job_id, "vllm payload required")
                return job_id
            item = _VllmItem(
                job_id=job_id,
                tenant_id=body.tenant_id,
                latency_class=body.latency_class,
                payload=body.vllm,
                expected_runtime_ms=expected_runtime_ms,
                enqueued_at=enqueued_at,
            )
            if body.latency_class == LatencyClass.realtime:
                self._vllm_rt.append(item)
            else:
                self._vllm_by_tenant[body.tenant_id].append(item)
            return job_id

        if not body.torch or not body.torch.image_base64:
            self._fail(job_id, "torch payload with image_base64 required")
            return job_id
        self._torch_by_tenant[body.tenant_id].append(
            _TorchItem(
                job_id=job_id,
                tenant_id=body.tenant_id,
                latency_class=body.latency_class,
                payload=body.torch,
                expected_runtime_ms=expected_runtime_ms,
                enqueued_at=enqueued_at,
            )
        )
        if self._torch_deadline is None:
            self._torch_deadline = time.monotonic() + (
                self.settings.torch_batch_window_ms / 1000.0
            )
        return job_id

    def metrics(self) -> dict[str, Any]:
        self._decay_recent_usage(time.monotonic())
        resource_metrics = (
            self._resource_layer.metrics() if self._resource_layer else {"gpu": {}, "engine_inflight": {}}
        )
        return {
            "queues": {
                "vllm_realtime": len(self._vllm_rt),
                "vllm_normal_total": sum(len(q) for q in self._vllm_by_tenant.values()),
                "torch_total": sum(len(q) for q in self._torch_by_tenant.values()),
            },
            "wfq": {
                "tenant_recent_gpu_ms": self._tenant_recent_gpu_ms,
                "tenant_total_gpu_ms": self._tenant_total_gpu_ms,
            },
            "deadline_miss": self._tenant_deadline_miss,
            "vllm_kv_cache_usage_ratio": self._vllm_kv_cache_usage_ratio,
            "resource_layer": resource_metrics,
        }

    def _default_cost_ms(self, body: JobSubmit) -> float:
        if body.engine == Engine.vllm and body.vllm:
            # rough proxy: tokens ~= GPU work share
            return float(max(20, body.vllm.max_tokens))
        return self.settings.torch_estimated_runtime_ms

    def _priority_weight(self, latency_class: LatencyClass) -> float:
        if latency_class == LatencyClass.realtime:
            return self.settings.priority_realtime
        if latency_class == LatencyClass.interactive:
            return self.settings.priority_interactive
        return self.settings.priority_batch

    def _decay_recent_usage(self, now: float) -> None:
        dt = max(0.0, now - self._last_decay_ts)
        if dt == 0:
            return
        half = max(1e-3, self.settings.wfq_decay_half_life_s)
        factor = math.exp(-math.log(2) * dt / half)
        for t in list(self._tenant_recent_gpu_ms.keys()):
            self._tenant_recent_gpu_ms[t] *= factor
            if self._tenant_recent_gpu_ms[t] < 1e-6:
                self._tenant_recent_gpu_ms[t] = 0.0
        self._last_decay_ts = now

    def _score(self, tenant_id: str, base_priority: float, oldest_wait_s: float) -> float:
        usage = self._tenant_recent_gpu_ms.get(tenant_id, 0.0)
        boost = 1.0 if oldest_wait_s >= self.settings.wfq_starvation_threshold_s else 0.0
        return (base_priority * (1.0 + boost)) / (1.0 + usage)

    def _pick_tenant_wfq_vllm(self) -> str | None:
        now = time.monotonic()
        self._decay_recent_usage(now)
        best_tenant = None
        best_score = -1.0
        for tenant_id, q in self._vllm_by_tenant.items():
            if not q:
                continue
            oldest = q[0]
            score = self._score(
                tenant_id=tenant_id,
                base_priority=self._priority_weight(oldest.latency_class),
                oldest_wait_s=max(0.0, now - oldest.enqueued_at),
            )
            if score > best_score:
                best_score = score
                best_tenant = tenant_id
        return best_tenant

    def _pick_tenant_wfq_torch(self) -> str | None:
        now = time.monotonic()
        self._decay_recent_usage(now)
        best_tenant = None
        best_score = -1.0
        for tenant_id, q in self._torch_by_tenant.items():
            if not q:
                continue
            oldest = q[0]
            score = self._score(
                tenant_id=tenant_id,
                base_priority=self._priority_weight(oldest.latency_class),
                oldest_wait_s=max(0.0, now - oldest.enqueued_at),
            )
            if score > best_score:
                best_score = score
                best_tenant = tenant_id
        return best_tenant

    def _record_usage(self, tenant_id: str, gpu_time_ms: float) -> None:
        self._tenant_recent_gpu_ms[tenant_id] = (
            self._tenant_recent_gpu_ms.get(tenant_id, 0.0) + max(0.0, gpu_time_ms)
        )
        self._tenant_total_gpu_ms[tenant_id] = (
            self._tenant_total_gpu_ms.get(tenant_id, 0.0) + max(0.0, gpu_time_ms)
        )

    def _record_deadline_miss_if_any(self, rec: JobRecord) -> None:
        if rec.deadline_at_ms is None:
            return
        if int(time.time() * 1000) > rec.deadline_at_ms:
            self._tenant_deadline_miss[rec.tenant_id] = (
                self._tenant_deadline_miss.get(rec.tenant_id, 0) + 1
            )

    def _fail(self, job_id: str, message: str) -> None:
        rec = self.jobs.get(job_id)
        if rec:
            rec.status = "failed"
            rec.error = message
            self._record_deadline_miss_if_any(rec)

    async def _run_loop(self) -> None:
        assert self._client is not None
        client = self._client
        next_vllm_metrics_pull = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_vllm_metrics_pull:
                await self._pull_vllm_kv_metrics(client)
                next_vllm_metrics_pull = now + 2.0

            await self._maybe_run_torch_batch(client)
            processed = await self._maybe_run_vllm_job(client)
            if not processed:
                await asyncio.sleep(0.02)

    async def _maybe_run_vllm_job(self, client: httpx.AsyncClient) -> bool:
        assert self._resource_layer is not None
        # realtime lane first
        if self._vllm_rt:
            item = self._vllm_rt[0]
            if self._resource_layer.can_admit(Engine.vllm, item.latency_class):
                self._vllm_rt.popleft()
                await self._run_vllm(client, item)
                return True

        # then WFQ-normal lane
        tenant = self._pick_tenant_wfq_vllm()
        if not tenant:
            return False
        q = self._vllm_by_tenant[tenant]
        item = q[0]
        if not self._resource_layer.can_admit(Engine.vllm, item.latency_class):
            return False
        q.popleft()
        await self._run_vllm(client, item)
        return True

    async def _maybe_run_torch_batch(self, client: httpx.AsyncClient) -> None:
        assert self._resource_layer is not None
        if self._torch_deadline is None:
            return
        now = time.monotonic()
        total_torch_pending = sum(len(q) for q in self._torch_by_tenant.values())
        if total_torch_pending == 0:
            self._torch_deadline = None
            return

        if now < self._torch_deadline and total_torch_pending < self.settings.torch_max_batch_size:
            return

        tenant = self._pick_tenant_wfq_torch()
        if not tenant:
            return
        q = self._torch_by_tenant[tenant]
        if not q:
            return

        seed = q[0]
        labels = seed.payload.labels
        batch: list[_TorchItem] = []
        kept: deque[_TorchItem] = deque()
        while q and len(batch) < self.settings.torch_max_batch_size:
            cand = q.popleft()
            if cand.payload.labels == labels:
                batch.append(cand)
            else:
                kept.append(cand)
        while q:
            kept.append(q.popleft())
        self._torch_by_tenant[tenant] = kept

        if not batch:
            self._torch_deadline = now + (self.settings.torch_batch_window_ms / 1000.0)
            return

        if not self._resource_layer.can_admit(Engine.torch, batch[0].latency_class):
            # put batch back at front to retry later
            current = self._torch_by_tenant[tenant]
            self._torch_by_tenant[tenant] = deque(batch) + current
            return

        await self._run_torch_batch(client, batch)
        self._torch_deadline = time.monotonic() + (self.settings.torch_batch_window_ms / 1000.0)

    async def _run_vllm(self, client: httpx.AsyncClient, item: _VllmItem) -> None:
        assert self._resource_layer is not None
        rec = self.jobs[item.job_id]
        rec.status = "running"
        self._resource_layer.on_start(Engine.vllm)
        started = time.monotonic()

        url = f"{self.settings.vllm_base_url.rstrip('/')}/v1/chat/completions"
        body = {
            "model": self.settings.vllm_model,
            "messages": [{"role": "user", "content": item.payload.prompt}],
            "max_tokens": item.payload.max_tokens,
        }
        try:
            r = await client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            rec.status = "completed"
            rec.result = {"text": text, "raw": data}
        except Exception as e:
            log.exception("vllm job %s failed", item.job_id)
            rec.status = "failed"
            rec.error = str(e)
        finally:
            self._resource_layer.on_finish(Engine.vllm)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            usage_ms = max(elapsed_ms, item.expected_runtime_ms)
            self._record_usage(item.tenant_id, usage_ms)
            self._record_deadline_miss_if_any(rec)

    async def _run_torch_batch(self, client: httpx.AsyncClient, batch: list[_TorchItem]) -> None:
        assert self._resource_layer is not None
        if not batch:
            return
        for it in batch:
            self.jobs[it.job_id].status = "running"

        self._resource_layer.on_start(Engine.torch)
        started = time.monotonic()
        url = f"{self.settings.torch_worker_url.rstrip('/')}/v1/infer"
        req = {
            "labels": batch[0].payload.labels,
            "items": [{"job_id": it.job_id, "image_base64": it.payload.image_base64} for it in batch],
        }
        try:
            r = await client.post(url, json=req)
            r.raise_for_status()
            data = r.json()
            by_id: dict[str, Any] = {x["job_id"]: x for x in data.get("results", [])}
            for it in batch:
                rec = self.jobs[it.job_id]
                row = by_id.get(it.job_id)
                if row and row.get("error"):
                    rec.status = "failed"
                    rec.error = row["error"]
                elif row:
                    rec.status = "completed"
                    rec.result = {
                        "top_labels": row.get("top_labels"),
                        "logits": row.get("logits"),
                    }
                else:
                    rec.status = "failed"
                    rec.error = "missing result for job"
        except Exception as e:
            log.exception("torch batch failed")
            for it in batch:
                rec = self.jobs[it.job_id]
                rec.status = "failed"
                rec.error = str(e)
        finally:
            self._resource_layer.on_finish(Engine.torch)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            split_ms = elapsed_ms / max(1, len(batch))
            for it in batch:
                usage_ms = max(split_ms, it.expected_runtime_ms)
                self._record_usage(it.tenant_id, usage_ms)
                self._record_deadline_miss_if_any(self.jobs[it.job_id])

    async def _pull_vllm_kv_metrics(self, client: httpx.AsyncClient) -> None:
        metrics_url = f"{self.settings.vllm_base_url.rstrip('/')}/metrics"
        try:
            r = await client.get(metrics_url, timeout=2.0)
            if r.status_code != 200:
                return
            text = r.text
            # best effort: one common metric name in vLLM exports
            m = re.search(r"^vllm_gpu_cache_usage_perc\s+([0-9.]+)$", text, re.MULTILINE)
            if m:
                self._vllm_kv_cache_usage_ratio = float(m.group(1)) / 100.0
        except Exception:
            return
