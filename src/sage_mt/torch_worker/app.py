from __future__ import annotations

import base64
import io
import logging
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel, Field

from sage_mt.config import TorchWorkerSettings

log = logging.getLogger(__name__)

app = FastAPI(title="Sage BioCLIP torch worker", version="0.1.0")
_settings = TorchWorkerSettings()
_model = None
_preprocess = None
_tokenizer = None
_device: torch.device | None = None


def _load_model() -> None:
    global _model, _preprocess, _tokenizer, _device
    import open_clip

    dev = torch.device(_settings.torch_device if torch.cuda.is_available() else "cpu")
    _device = dev
    log.info("Loading BioCLIP from %s on %s", _settings.bioclip_hf_id, dev)
    _model, _, _preprocess = open_clip.create_model_and_transforms(_settings.bioclip_hf_id)
    _tokenizer = open_clip.get_tokenizer(_settings.bioclip_hf_id)
    _model = _model.to(dev)
    _model.eval()


@app.on_event("startup")
def startup() -> None:
    logging.basicConfig(level=logging.INFO)
    _load_model()


class InferItem(BaseModel):
    job_id: str
    image_base64: str


class InferRequest(BaseModel):
    labels: list[str] = Field(min_length=1)
    items: list[InferItem] = Field(min_length=1)


class InferOneResult(BaseModel):
    job_id: str
    top_labels: list[str] | None = None
    logits: list[float] | None = None
    error: str | None = None


class InferResponse(BaseModel):
    results: list[InferOneResult]


def _decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.post("/v1/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    assert _model is not None and _preprocess is not None and _tokenizer is not None
    assert _device is not None

    decode_errors: dict[str, str] = {}
    tensors: list[torch.Tensor] = []
    valid_job_ids: list[str] = []

    for it in req.items:
        try:
            tensors.append(_preprocess(_decode_image(it.image_base64)))
            valid_job_ids.append(it.job_id)
        except Exception as e:
            decode_errors[it.job_id] = f"image decode: {e}"

    results_map: dict[str, InferOneResult] = {
        jid: InferOneResult(job_id=jid, error=err) for jid, err in decode_errors.items()
    }

    if tensors:
        image_tensor = torch.stack(tensors).to(_device)
        text_tokens = _tokenizer(req.labels).to(_device)
        with torch.no_grad():
            image_features = _model.encode_image(image_tensor)
            text_features = _model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = _model.logit_scale.exp()
            logits_m = logit_scale * (image_features @ text_features.T)

        for row, jid in enumerate(valid_job_ids):
            scores = logits_m[row].float().cpu().tolist()
            pairs = sorted(zip(req.labels, scores), key=lambda x: x[1], reverse=True)
            top = [p[0] for p in pairs[:5]]
            results_map[jid] = InferOneResult(
                job_id=jid,
                top_labels=top,
                logits=scores,
            )

    ordered = [results_map[it.job_id] for it in req.items]
    return InferResponse(results=ordered)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "device": str(_device)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "sage_mt.torch_worker.app:app",
        host=_settings.torch_worker_host,
        port=_settings.torch_worker_port,
        factory=False,
    )
