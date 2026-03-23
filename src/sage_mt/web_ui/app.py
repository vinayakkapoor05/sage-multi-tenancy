from __future__ import annotations

import base64
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sage_mt.config import GatewaySettings

log = logging.getLogger(__name__)


class UiSettings(BaseModel):
    ui_host: str = "0.0.0.0"
    ui_port: int = 9090
    gateway_base_url: str = "http://127.0.0.1:8080"


def _settings() -> UiSettings:
    import os

    return UiSettings(
        ui_host=os.getenv("UI_HOST", "0.0.0.0"),
        ui_port=int(os.getenv("UI_PORT", "9090")),
        gateway_base_url=os.getenv("GATEWAY_BASE_URL", "http://127.0.0.1:8080"),
    )


class _Proxy:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _assert(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("proxy client not started")
        return self._client

    async def post_jobs(self, body: dict) -> dict:
        c = self._assert()
        r = await c.post(f"{self.base_url}/api/jobs", json=body)
        r.raise_for_status()
        return r.json()

    async def get_job(self, job_id: str) -> dict:
        c = self._assert()
        r = await c.get(f"{self.base_url}/api/jobs/{job_id}")
        if r.status_code == 404:
            raise HTTPException(404, "job not found")
        r.raise_for_status()
        return r.json()

    async def get_metrics(self) -> dict:
        c = self._assert()
        r = await c.get(f"{self.base_url}/api/scheduler/metrics")
        r.raise_for_status()
        return r.json()

    async def list_jobs(self) -> list[dict]:
        c = self._assert()
        r = await c.get(f"{self.base_url}/api/jobs")
        r.raise_for_status()
        return r.json()


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = _settings()
    proxy = _Proxy(s.gateway_base_url)
    await proxy.start()
    app.state.proxy = proxy
    yield
    await proxy.stop()


app = FastAPI(title="Sage gateway UI (proxy)", version="0.1.0", lifespan=lifespan)


def _proxy() -> _Proxy:
    proxy = getattr(app.state, "proxy", None)
    if proxy is None:
        raise HTTPException(503, "proxy not ready")
    return proxy


@app.post("/api/jobs-proxy")
async def api_jobs_proxy_submit(body: dict) -> dict:
    return await _proxy().post_jobs(body)


@app.get("/api/jobs-proxy/{job_id}")
async def api_jobs_proxy_status(job_id: str) -> dict:
    return await _proxy().get_job(job_id)


@app.get("/api/metrics-proxy")
async def api_metrics_proxy() -> dict:
    return await _proxy().get_metrics()


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Sage UI</title>
    <style>
      body { font-family: system-ui, sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; }
      .row { display: grid; grid-template-columns: 10rem 1fr; gap: 0.75rem; align-items: start; margin-top: 0.75rem; }
      label { font-size: 0.9rem; font-weight: 600; color: #222; }
      input, select, textarea { width: 100%; padding: 0.5rem; box-sizing: border-box; }
      textarea { min-height: 5rem; }
      button { margin-top: 1rem; padding: 0.6rem 1rem; cursor: pointer; }
      pre { background: #f6f6f6; padding: 0.75rem; border-radius: 8px; overflow: auto; }
      .muted { color: #666; font-size: 0.9rem; margin-top: 0.5rem; }
      .section { margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid #eee; }
    </style>
  </head>
  <body>
    <h1>Sage Inference Gateway</h1>
    <div class="muted">Interactive submit + polling. Backend proxies to your gateway.</div>

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
          <option value="vllm" selected>vLLM</option>
          <option value="torch">BioCLIP (torch)</option>
        </select>
      </div>

      <div class="row"><label>deadline_ms</label><input id="deadline_ms" placeholder="optional" /></div>
      <div class="row"><label>expected_runtime_ms</label><input id="expected_runtime_ms" placeholder="optional" /></div>

      <div id="vllm_section">
        <div class="row"><label>prompt</label><textarea id="prompt" placeholder="Summarize image"></textarea></div>
        <div class="row"><label>max_tokens</label><input id="max_tokens" value="32" /></div>
        <div class="row"><label>image (optional)</label><input id="vllm_image" type="file" accept="image/*" /></div>
      </div>

      <div id="torch_section" style="display:none;">
        <div class="row"><label>labels</label><textarea id="labels" placeholder="Comma-separated, e.g. forest fire, wildfire smoke"></textarea></div>
        <div class="row"><label>image</label><input id="torch_image" type="file" accept="image/*" /></div>
      </div>

      <button onclick="submitJob()">Submit</button>
    </div>

    <div class="section">
      <h2>Response</h2>
      <pre id="out">Submit to see results...</pre>
    </div>

    <div class="section">
      <h2>Scheduler metrics (live)</h2>
      <pre id="metrics">Waiting...</pre>
    </div>

    <script>
      let vllmImageBase64 = null;
      let vllmImageMimeType = null;
      let torchImageBase64 = null;
      let torchImageMimeType = null;

      function splitDataUrl(dataUrl) {
        const parts = String(dataUrl).split(',');
        if (parts.length !== 2) return { b64: dataUrl, mime: null };
        const header = parts[0];
        const mime = header.includes('data:') && header.includes(';base64') ? header.slice(5, header.indexOf(';base64')) : null;
        return { b64: parts[1], mime: mime };
      }

      function onEngineChange() {
        const engine = document.getElementById('engine').value;
        document.getElementById('vllm_section').style.display = engine === 'vllm' ? 'block' : 'none';
        document.getElementById('torch_section').style.display = engine === 'torch' ? 'block' : 'none';
      }

      const vllmImgEl = document.getElementById('vllm_image');
      vllmImgEl.addEventListener('change', async (ev) => {
        const file = ev.target.files[0];
        if (!file) { vllmImageBase64 = null; vllmImageMimeType = null; return; }
        const reader = new FileReader();
        reader.onload = () => {
          const { b64, mime } = splitDataUrl(String(reader.result));
          vllmImageBase64 = b64;
          vllmImageMimeType = mime;
        };
        reader.readAsDataURL(file);
      });

      const torchImgEl = document.getElementById('torch_image');
      torchImgEl.addEventListener('change', async (ev) => {
        const file = ev.target.files[0];
        if (!file) { torchImageBase64 = null; torchImageMimeType = null; return; }
        const reader = new FileReader();
        reader.onload = () => {
          const { b64, mime } = splitDataUrl(String(reader.result));
          torchImageBase64 = b64;
          torchImageMimeType = mime;
        };
        reader.readAsDataURL(file);
      });

      async function submitJob() {
        const out = document.getElementById('out');
        out.textContent = 'Submitting...';

        const tenant_id = document.getElementById('tenant_id').value.trim();
        const latency_class = document.getElementById('latency_class').value;
        const engine = document.getElementById('engine').value;
        const deadline_ms = document.getElementById('deadline_ms').value.trim();
        const expected_runtime_ms = document.getElementById('expected_runtime_ms').value.trim();
        const prompt = document.getElementById('prompt').value;
        const max_tokens = parseInt(document.getElementById('max_tokens').value, 10);
        const labelsRaw = document.getElementById('labels').value;

        let body = { tenant_id, latency_class, engine };
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

        try {
          const res = await fetch('/api/jobs-proxy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          });
          const data = await res.json();
          if (!res.ok) {
            out.textContent = `Error ${res.status}: ${JSON.stringify(data, null, 2)}`;
            return;
          }

          const jobId = data.id;
          out.textContent = `Submitted job_id=${jobId}. Polling...\\n\\n` + JSON.stringify(data, null, 2);

          const pollUrl = `/api/jobs-proxy/${jobId}`;
          const deadline = Date.now() + 180000;
          while (Date.now() < deadline) {
            const stRes = await fetch(pollUrl);
            const stData = await stRes.json();
            out.textContent = JSON.stringify(stData, null, 2);
            if (stData.status === 'completed' || stData.status === 'failed') return;
            await new Promise(r => setTimeout(r, 800));
          }
          out.textContent = out.textContent + \"\\n\\nTimed out waiting for completion.\";
        } catch (e) {
          out.textContent = `Submit failed: ${String(e)}`;
        }
      }

      async function refreshMetrics() {
        const el = document.getElementById('metrics');
        try {
          const res = await fetch('/api/metrics-proxy');
          const data = await res.json();
          el.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
          el.textContent = 'metrics error: ' + String(e);
        }
      }

      async function start() {
        onEngineChange();
        await refreshMetrics();
        setInterval(refreshMetrics, 2000);
      }
      start();
    </script>
  </body>
</html>
"""


def main() -> None:
    s = _settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("sage_mt.web_ui.app:app", host=s.ui_host, port=s.ui_port, factory=False)

