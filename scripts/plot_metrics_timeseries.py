#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _to_float(x: str | None) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(x: str | None) -> int | None:
    v = _to_float(x)
    if v is None:
        return None
    return int(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to metrics_timeseries.csv")
    ap.add_argument("--out-dir", default="plots_from_metrics", help="Output directory for PNGs")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise SystemExit("No rows found in CSV")

    # Core series
    t = [_to_float(r.get("elapsed_s")) for r in rows]
    vllm_rt = [_to_float(r.get("vllm_realtime")) or 0.0 for r in rows]
    vllm_n = [_to_float(r.get("vllm_normal_total")) or 0.0 for r in rows]
    torch_q = [_to_float(r.get("torch_total")) or 0.0 for r in rows]

    a_recent = [_to_float(r.get("wfq_recent_ms_tenant-a")) for r in rows]
    b_recent = [_to_float(r.get("wfq_recent_ms_tenant-b")) for r in rows]
    a_total = [_to_float(r.get("wfq_total_ms_tenant-a")) for r in rows]
    b_total = [_to_float(r.get("wfq_total_ms_tenant-b")) for r in rows]

    a_miss = [_to_int(r.get("deadline_miss_tenant-a")) or 0 for r in rows]
    b_miss = [_to_int(r.get("deadline_miss_tenant-b")) or 0 for r in rows]

    util = [_to_float(r.get("gpu_util_pct")) for r in rows]

    # Derived
    total_queue = [vllm_rt[i] + vllm_n[i] + torch_q[i] for i in range(len(rows))]
    total_miss = [a_miss[i] + b_miss[i] for i in range(len(rows))]
    a_share = []
    b_share = []
    for i in range(len(rows)):
        a = a_total[i] or 0.0
        b = b_total[i] or 0.0
        denom = a + b
        if denom <= 0:
            a_share.append(0.0)
            b_share.append(0.0)
        else:
            a_share.append(a / denom)
            b_share.append(b / denom)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise SystemExit(
            "matplotlib is required to generate plots. Install with: pip install matplotlib\n"
            f"Import error: {e}"
        )

    # 1) Queue depths
    plt.figure(figsize=(10, 4.8))
    plt.plot(t, vllm_rt, label="vllm_realtime")
    plt.plot(t, vllm_n, label="vllm_normal_total")
    plt.plot(t, torch_q, label="torch_total")
    plt.plot(t, total_queue, label="total_queue", linestyle="--")
    plt.title("Queue Depth Over Time")
    plt.xlabel("Elapsed seconds")
    plt.ylabel("Jobs in queue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "01_queue_depth_over_time.png", dpi=200)
    plt.close()

    # 2) WFQ recent usage
    plt.figure(figsize=(10, 4.8))
    plt.plot(t, a_recent, label="tenant-a recent ms")
    plt.plot(t, b_recent, label="tenant-b recent ms")
    plt.title("WFQ Recent GPU Usage (Rolling)")
    plt.xlabel("Elapsed seconds")
    plt.ylabel("Recent GPU-ms (decayed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "02_wfq_recent_usage.png", dpi=200)
    plt.close()

    # 3) Cumulative usage and share
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(t, a_total, label="tenant-a total ms", color="tab:blue")
    ax1.plot(t, b_total, label="tenant-b total ms", color="tab:orange")
    ax1.set_xlabel("Elapsed seconds")
    ax1.set_ylabel("Cumulative GPU-ms")
    ax2 = ax1.twinx()
    ax2.plot(t, a_share, label="tenant-a share", color="tab:green", linestyle="--")
    ax2.plot(t, b_share, label="tenant-b share", color="tab:red", linestyle="--")
    ax2.set_ylabel("Share (0-1)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title("Cumulative GPU Usage and Tenant Share")
    plt.tight_layout()
    plt.savefig(out_dir / "03_cumulative_usage_and_share.png", dpi=200)
    plt.close()

    # 4) Deadline misses (step)
    plt.figure(figsize=(10, 4.8))
    plt.step(t, a_miss, where="post", label="tenant-a deadline_miss")
    plt.step(t, b_miss, where="post", label="tenant-b deadline_miss")
    plt.step(t, total_miss, where="post", label="total deadline_miss", linestyle="--")
    plt.title("Deadline Misses Over Time")
    plt.xlabel("Elapsed seconds")
    plt.ylabel("Miss count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "04_deadline_miss_over_time.png", dpi=200)
    plt.close()

    # 5) Queue pressure vs misses
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(t, total_queue, label="total_queue", color="tab:blue")
    ax1.set_xlabel("Elapsed seconds")
    ax1.set_ylabel("Queue depth")
    ax2 = ax1.twinx()
    ax2.step(t, total_miss, where="post", label="total_deadline_miss", color="tab:red")
    ax2.set_ylabel("Deadline miss count")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title("Queue Pressure vs Deadline Misses")
    plt.tight_layout()
    plt.savefig(out_dir / "05_queue_vs_deadline_miss.png", dpi=200)
    plt.close()

    # 6) GPU utilization (if present)
    if any(v is not None for v in util):
        plt.figure(figsize=(10, 4.2))
        util_clean = [v if v is not None else 0.0 for v in util]
        plt.plot(t, util_clean, label="gpu_util_pct")
        plt.title("GPU Utilization (source from metrics)")
        plt.xlabel("Elapsed seconds")
        plt.ylabel("Utilization %")
        plt.tight_layout()
        plt.savefig(out_dir / "06_gpu_utilization.png", dpi=200)
        plt.close()

    print(f"Generated plots in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

