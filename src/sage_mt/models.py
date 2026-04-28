from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LatencyClass(str, Enum):
    realtime = "realtime"
    interactive = "interactive"
    batch = "batch"


class Engine(str, Enum):
    vllm = "vllm"
    torch = "torch"


class VllmPayload(BaseModel):
    prompt: str
    max_tokens: int = 256
    image_base64: str | None = None
    image_mime_type: str | None = None
    rtsp_url: str | None = None


class TorchPayload(BaseModel):
    """Zero-shot style labels; gateway sends base64 JPEG/PNG per job."""

    labels: list[str] = Field(min_length=1)
    image_base64: str | None = None
    rtsp_url: str | None = None


class JobSubmit(BaseModel):
    tenant_id: str = "default"
    latency_class: LatencyClass = LatencyClass.batch
    engine: Engine
    deadline_ms: int | None = None
    expected_runtime_ms: float | None = None
    vllm: VllmPayload | None = None
    torch: TorchPayload | None = None


class JobRecord(BaseModel):
    id: str
    status: str
    tenant_id: str
    latency_class: LatencyClass
    engine: Engine
    submitted_at_ms: int
    deadline_at_ms: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
