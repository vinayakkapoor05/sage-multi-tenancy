from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from sage_mt.models import JobRecord

log = logging.getLogger(__name__)


@dataclass
class NodePublisher:
    enabled: bool
    plugin: Any | None = None

    @classmethod
    def create(cls, enabled: bool) -> "NodePublisher":
        if not enabled:
            return cls(enabled=False, plugin=None)
        try:
            from waggle.plugin import Plugin

            plugin = Plugin()
            plugin.__enter__()
            log.info("pywaggle publisher enabled")
            return cls(enabled=True, plugin=plugin)
        except Exception:
            log.exception("failed to initialize pywaggle Plugin; disabling publish")
            return cls(enabled=False, plugin=None)

    def close(self) -> None:
        if self.plugin is None:
            return
        try:
            self.plugin.__exit__(None, None, None)
        except Exception:
            log.exception("failed closing pywaggle plugin")

    def publish_job(self, rec: JobRecord) -> None:
        if not self.enabled or self.plugin is None:
            return
        try:
            ts = time.time_ns()
            status_value = 1 if rec.status == "completed" else 0
            meta = {
                "job_id": rec.id,
                "tenant_id": rec.tenant_id,
                "engine": rec.engine.value,
                "latency_class": rec.latency_class.value,
                "status": rec.status,
            }
            self.plugin.publish("sage.inference.job.status", status_value, meta=meta, timestamp=ts)
            if rec.error:
                self.plugin.publish(
                    "sage.inference.job.error",
                    rec.error,
                    meta=meta,
                    timestamp=ts,
                )
            if rec.result:
                self.plugin.publish(
                    "sage.inference.job.result_json",
                    json.dumps(rec.result),
                    meta=meta,
                    timestamp=ts,
                )
        except Exception:
            log.exception("pywaggle publish failed for job_id=%s", rec.id)
