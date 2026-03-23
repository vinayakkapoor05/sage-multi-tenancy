#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request


def http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway-url", default="http://127.0.0.1:8080")
    ap.add_argument("--image", required=True, help="One image used for all torch jobs.")
    ap.add_argument("--poll-interval-s", type=float, default=0.8)
    ap.add_argument("--timeout-s", type=float, default=180.0)
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")

    base = args.gateway_url.rstrip("/")
    jobs_submit_url = f"{base}/api/jobs"
    jobs_list_url = f"{base}/api/jobs"
    metrics_url = f"{base}/api/scheduler/metrics"

    labels = ["bird", "plant", "insect", "fish", "reptile"]

    jobs = []
    jobs.append(("tenant-a", "interactive", "vllm", {"prompt": "Tenant A job 1", "max_tokens": 48}, 5000, 250))
    jobs.append(("tenant-b", "interactive", "vllm", {"prompt": "Tenant B job 1", "max_tokens": 24}, 5000, 250))
    jobs.append(("tenant-a", "batch", "torch", {"labels": labels, "image_base64": img_b64}, 7000, 220))
    jobs.append(("tenant-a", "batch", "vllm", {"prompt": "Tenant A job 2 fairness", "max_tokens": 32}, 5000, 200))
    jobs.append(("tenant-b", "batch", "torch", {"labels": labels, "image_base64": img_b64}, 7000, 220))
    jobs.append(("tenant-b", "interactive", "vllm", {"prompt": "Tenant B job 2 <=10 words", "max_tokens": 32}, 5000, 200))
    jobs.append(("tenant-a", "interactive", "torch", {"labels": labels, "image_base64": img_b64}, 7000, 220))
    jobs.append(("tenant-a", "interactive", "vllm", {"prompt": "Tenant A job 3 one line only", "max_tokens": 32}, 5000, 180))
    jobs.append(("tenant-b", "interactive", "torch", {"labels": labels, "image_base64": img_b64}, 7000, 220))
    jobs.append(("tenant-b", "batch", "vllm", {"prompt": "Tenant B job 3 one sentence", "max_tokens": 32}, 5000, 180))

    submitted = []
    for tenant_id, latency_class, engine, payload, deadline_ms, expected_runtime_ms in jobs:
        body = {
            "tenant_id": tenant_id,
            "latency_class": latency_class,
            "engine": engine,
            "deadline_ms": deadline_ms,
            "expected_runtime_ms": expected_runtime_ms,
        }
        if engine == "vllm":
            body["vllm"] = payload
        else:
            body["torch"] = payload

        rec = http_post_json(jobs_submit_url, body)
        submitted.append(rec["id"])

    clear_screen()
    print(f"Submitted {len(submitted)} jobs to {base}")
    print("Waiting for completion...\n")

    deadline = time.time() + args.timeout_s
    while True:
        try:
            all_jobs = http_get_json(jobs_list_url)
        except urllib.error.URLError:
            all_jobs = []

        metrics = http_get_json(metrics_url)

        by_id = {j["id"]: j for j in all_jobs} if all_jobs else {}
        shown = [by_id.get(jid) for jid in submitted]
        shown = [j for j in shown if j is not None]

        status_counts = {}
        for j in shown:
            status_counts[j["status"]] = status_counts.get(j["status"], 0) + 1

        clear_screen()
        print(f"Live demo @ {base}")
        print(f"Status: {status_counts} (total shown: {len(shown)}/{len(submitted)})\n")

        print("Jobs (id | tenant | engine | latency | status):")
        for j in shown:
            print(
                f"  {j['id']} | {j['tenant_id']} | {j['engine']} | {j['latency_class']} | {j['status']}"
            )
            if j.get("error"):
                print(f"    error: {j['error']}")

        wfq = metrics.get("wfq", {})
        dlm = metrics.get("deadline_miss", {})
        res = metrics.get("resource_layer", {})
        gpu = res.get("gpu", {})

        print("\nWFQ summary:")
        print(f"  tenant_recent_gpu_ms: {wfq.get('tenant_recent_gpu_ms', {})}")
        print(f"  tenant_total_gpu_ms:  {wfq.get('tenant_total_gpu_ms', {})}")
        print(f"  deadline_miss:        {dlm}")
        print(f"  gpu.utilization_pct:   {gpu.get('utilization_pct')}")
        print(f"  gpu.source:            {gpu.get('source')}")

        if status_counts.get("completed", 0) + status_counts.get("failed", 0) >= len(submitted):
            clear_screen()
            print("All jobs finished.\n")
            print("Final scheduler metrics:")
            print(json.dumps(metrics, indent=2))
            return

        if time.time() > deadline:
            clear_screen()
            print("Timed out waiting for jobs.\n")
            print("Current scheduler metrics:")
            print(json.dumps(metrics, indent=2))
            return

        time.sleep(args.poll_interval_s)


if __name__ == "__main__":
    main()

