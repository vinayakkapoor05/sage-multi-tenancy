from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Sage Inference Gateway UI</title>
    <style>
      body { font-family: system-ui, sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; }
      .row { display: grid; grid-template-columns: 10rem 1fr; gap: 0.75rem; align-items: start; margin-top: 0.75rem; }
      label { font-size: 0.9rem; font-weight: 600; color: #222; }
      input, select, textarea { width: 100%; padding: 0.5rem; box-sizing: border-box; }
      textarea { min-height: 5rem; }
      button { margin-top: 1rem; padding: 0.6rem 1rem; cursor: pointer; }
      pre { background: #f6f6f6; padding: 0.75rem; border-radius: 8px; overflow: auto; }
      .muted { color: #666; font-size: 0.85rem; margin-top: 0.5rem; }
      .section { margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid #eee; }
    </style>
  </head>
  <body>
    <h1>Sage Inference Gateway</h1>
    <div class="muted">Submit vLLM and BioCLIP jobs locally, then view this page over SSH port-forwarding.</div>

    <div class="section">
      <h2>Submit job</h2>

      <div class="row"><label>tenant_id</label><input id="tenant_id" value="tenant-a" /></div>
      <div class="row"><label>latency_class</label>
        <select id="latency_class">
          <option value="realtime">realtime</option>
          <option value="interactive" selected>interactive</option>
          <option value="batch">batch</option>
        </select>
      </div>
      <div class="row"><label>engine</label>
        <select id="engine" onchange="onEngineChange()">
          <option value="vllm" selected>vllm</option>
          <option value="torch">torch</option>
        </select>
      </div>
      <div class="row"><label>deadline_ms</label><input id="deadline_ms" placeholder="optional" /></div>
      <div class="row"><label>expected_runtime_ms</label><input id="expected_runtime_ms" placeholder="optional" /></div>

      <div id="vllm_section">
        <div class="row"><label>prompt</label><textarea id="prompt" placeholder="Hello from Orin"></textarea></div>
        <div class="row"><label>max_tokens</label><input id="max_tokens" value="32" /></div>
        <div class="row"><label>image (optional for vLLM)</label><input id="vllm_image" type="file" accept="image/*" /></div>
      </div>

      <div id="torch_section" style="display:none;">
        <div class="row"><label>labels</label><textarea id="labels" placeholder="Comma-separated, e.g. forest fire, wildfire smoke, animal"></textarea></div>
        <div class="row"><label>image</label><input id="torch_image" type="file" accept="image/*" /></div>
      </div>

      <button onclick="submitJob()">Submit</button>
    </div>

    <div class="section">
      <h2>Response</h2>
      <pre id="out">Submit to see results...</pre>
    </div>

    <script>
      let vllmImageBase64 = null;
      let vllmImageMimeType = null;
      let torchImageBase64 = null;
      let torchImageMimeType = null;

      function splitDataUrl(dataUrl) {
        // data:image/jpeg;base64,/9j/...
        const parts = String(dataUrl).split(',');
        if (parts.length !== 2) return { b64: dataUrl, mime: null };
        const header = parts[0]; // data:mime;base64
        const mime = header.includes('data:') && header.includes(';base64') ? header.slice(5, header.indexOf(';base64')) : null;
        return { b64: parts[1], mime: mime };
      }
      function onEngineChange() {
        const engine = document.getElementById('engine').value;
        document.getElementById('vllm_section').style.display = engine === 'vllm' ? 'block' : 'none';
        document.getElementById('torch_section').style.display = engine === 'torch' ? 'block' : 'none';
      }

      document.getElementById('vllm_image').addEventListener('change', async (ev) => {
        const file = ev.target.files[0];
        if (!file) { vllmImageBase64 = null; vllmImageMimeType = null; return; }
        const reader = new FileReader();
        reader.onload = () => {
          const res = String(reader.result);
          const { b64, mime } = splitDataUrl(res);
          vllmImageBase64 = b64;
          vllmImageMimeType = mime;
        };
        reader.readAsDataURL(file);
      });

      document.getElementById('torch_image').addEventListener('change', async (ev) => {
        const file = ev.target.files[0];
        if (!file) { torchImageBase64 = null; torchImageMimeType = null; return; }
        const reader = new FileReader();
        reader.onload = () => {
          const res = String(reader.result);
          const { b64, mime } = splitDataUrl(res);
          torchImageBase64 = b64;
          torchImageMimeType = mime;
        };
        reader.readAsDataURL(file);
      });

      async function submitJob() {
        const tenant_id = document.getElementById('tenant_id').value.trim();
        const latency_class = document.getElementById('latency_class').value;
        const engine = document.getElementById('engine').value;
        const deadline_ms = document.getElementById('deadline_ms').value.trim();
        const expected_runtime_ms = document.getElementById('expected_runtime_ms').value.trim();
        const prompt = document.getElementById('prompt').value;
        const max_tokens = parseInt(document.getElementById('max_tokens').value, 10);
        const labelsRaw = document.getElementById('labels').value;

        let body = {
          tenant_id,
          latency_class,
          engine
        };
        if (deadline_ms) body.deadline_ms = parseInt(deadline_ms, 10);
        if (expected_runtime_ms) body.expected_runtime_ms = parseFloat(expected_runtime_ms);

        if (engine === 'vllm') {
          body.vllm = { prompt: prompt, max_tokens: max_tokens };
          if (vllmImageBase64) {
            body.vllm.image_base64 = vllmImageBase64;
            body.vllm.image_mime_type = vllmImageMimeType || 'image/jpeg';
          }
        } else {
          const labels = labelsRaw.split(',').map(s => s.trim()).filter(Boolean);
          body.torch = { labels: labels, image_base64: torchImageBase64 };
        }

        if (engine === 'torch') {
          if (!labelsRaw.trim()) { alert('Provide torch labels'); return; }
          if (!torchImageBase64) { alert('Select an image'); return; }
        }

        const out = document.getElementById('out');
        out.textContent = 'Submitting...';

        try {
          const res = await fetch('/api/jobs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
          });
          const text = await res.text();
          let data = null;
          try { data = JSON.parse(text); } catch (e) { data = { raw: text }; }
          if (!res.ok) {
            out.textContent = `Error ${res.status}: ${JSON.stringify(data, null, 2)}`;
            return;
          }

          // Poll job status until completed/failed so the presenter sees the real result.
          const jobId = data.id;
          const pollUrl = `/api/jobs/${jobId}`;
          out.textContent = JSON.stringify(data, null, 2);

          const deadline = Date.now() + 180000; // 3 minutes
          while (Date.now() < deadline) {
            const stRes = await fetch(pollUrl);
            const stText = await stRes.text();
            let stData = null;
            try { stData = JSON.parse(stText); } catch (e) { stData = { raw: stText }; }
            out.textContent = JSON.stringify(stData, null, 2);
            const status = stData.status;
            if (status === 'completed' || status === 'failed') {
              return;
            }
            await new Promise(r => setTimeout(r, 800));
          }

          out.textContent = (out.textContent || '') + "\n\nTimed out waiting for job completion.";
        } catch (e) {
          out.textContent = `Fetch failed: ${String(e)}`;
        }
      }

      onEngineChange();
    </script>
  </body>
</html>
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "sage_mt.gateway.app:app",
        host=_settings.gateway_host,
        port=_settings.gateway_port,
        factory=False,
    )
