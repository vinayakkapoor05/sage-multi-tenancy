from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8080

    vllm_base_url: str = "http://127.0.0.1:8000"
    vllm_model: str = "replace-me"
    torch_worker_url: str = "http://127.0.0.1:8001"

    torch_batch_window_ms: float = 80.0
    torch_max_batch_size: int = 8
    torch_estimated_runtime_ms: float = 120.0

    # WFQ / fairness tuning
    wfq_decay_half_life_s: float = 30.0
    wfq_starvation_threshold_s: float = 5.0
    priority_realtime: float = 3.0
    priority_interactive: float = 2.0
    priority_batch: float = 1.0

    # Basic GPU resource layer controls (Orin-friendly soft controls)
    max_non_realtime_gpu_utilization_pct: float = 92.0
    min_free_vram_mb: int = 1024
    reserve_realtime_vram_mb: int = 512
    max_inflight_vllm: int = 2
    max_inflight_torch: int = 2

    # Optional RTSP snapshot source for jobs without direct image input
    default_rtsp_url: str | None = None
    rtsp_snapshot_timeout_s: float = 8.0

    # Optional pywaggle publishing
    pywaggle_publish_enabled: bool = False


class TorchWorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    torch_worker_host: str = "0.0.0.0"
    torch_worker_port: int = 8001
    torch_device: str = "cuda"
    bioclip_hf_id: str = "hf-hub:imageomics/bioclip"
