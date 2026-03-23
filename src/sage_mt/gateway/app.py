from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from sage_mt.config import GatewaySettings
from sage_mt.models import JobRecord, JobSubmit
from sage_mt.scheduler import InferenceScheduler

log = logging.getLogger(__name__)
_settings = GatewaySettings()
_scheduler: InferenceScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    _scheduler = InferenceScheduler(settings=_settings)
    await _scheduler.start()
    log.info(
        "Gateway up; vLLM=%s model=%s torch=%s",
        _settings.vllm_base_url,
        _settings.vllm_model,
        _settings.torch_worker_url,
    )
    yield
    if _scheduler:
        await _scheduler.stop()
        _scheduler = None


app = FastAPI(title="Sage inference gateway", version="0.1.0", lifespan=lifespan)


def _sched() -> InferenceScheduler:
    if _scheduler is None:
        raise HTTPException(503, "scheduler not ready")
    return _scheduler


@app.post("/api/jobs", response_model=JobRecord)
async def api_submit_job(body: JobSubmit) -> JobRecord:
    sched = _sched()
    job_id = await sched.submit(body)
    return sched.jobs[job_id]


@app.get("/api/jobs/{job_id}", response_model=JobRecord)
async def api_job_status(job_id: str) -> JobRecord:
    sched = _sched()
    rec = sched.jobs.get(job_id)
    if not rec:
        raise HTTPException(404, "job not found")
    return rec


@app.get("/api/scheduler/metrics")
async def api_scheduler_metrics() -> dict:
    sched = _sched()
    return sched.metrics()


@app.get("/api/jobs", response_model=list[JobRecord])
async def api_jobs_list() -> list[JobRecord]:
    sched = _sched()
    return sorted(sched.jobs.values(), key=lambda r: r.submitted_at_ms)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "sage_mt.gateway.app:app",
        host=_settings.gateway_host,
        port=_settings.gateway_port,
        factory=False,
    )
