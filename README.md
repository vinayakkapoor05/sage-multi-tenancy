# Sage multi-tenancy (Orin milestone)

Gateway + in-process scheduler + optional BioCLIP torch worker. Submit jobs via **REST API**.

## Quick start (three terminals on Orin)

1. **vLLM** (serve the model from SSD):

   ```bash
   docker run --rm -it \
     -p 8000:8000 \
     -v /ssd/vllm-models:/models \
     vllm:r36.4.tegra-aarch64-cu126-22.04 \
     vllm serve /models/qwen2.5-vl-3b-instruct \
       --gpu-memory-utilization 0.6 \
       --max-model-len 65536
   ```

   ```bash
   curl http://127.0.0.1:8000/v1/models
   ```

2. **Torch worker** (GPU; loads [imageomics/bioclip](https://huggingface.co/imageomics/bioclip) via OpenCLIP):

   ```bash
   cd /path/to/sage-multi-tenancy
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[torch-worker]"
   export HF_HUB_ENABLE_HF_TRANSFER=1   
   sage-torch-worker
   ```

3. **Gateway** (scheduler + API). Point `VLLM_BASE_URL` at your running vLLM:

   ```bash
   source .venv/bin/activate
   pip install -e .
   export VLLM_BASE_URL=http://127.0.0.1:8000
   export VLLM_MODEL=/models/qwen2.5-vl-3b-instruct
   export TORCH_WORKER_URL=http://127.0.0.1:8001
   sage-gateway
   ```

API endpoints:
- `POST /api/jobs` to submit a job
- `GET /api/jobs/{job_id}` to fetch status/result
- `GET /api/scheduler/metrics` for fairness/resource metrics
- `GET /api/jobs`

`POST /api/jobs` supports optional scheduling hints in the JSON body:
- `deadline_ms`: deadline relative to submit time
- `expected_runtime_ms`: scheduler cost hint for WFQ accounting

See `.env.example` for all variables.

