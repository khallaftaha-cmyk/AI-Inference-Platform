"""
Synthetic Traffic Generator for AI Inference Platform.

This script simulates client traffic by sending continuous ECG prediction
requests to the FastAPI inference endpoints. It serves two purposes:
1. Demonstrates that the inference API is working under real load.
2. Generates live metrics for Prometheus to scrape and Grafana to visualize
   (e.g., requests per second, p95/p99 latency, error rates, canary distribution).

Usage:
    python scripts/simulate_traffic.py
    python scripts/simulate_traffic.py --requests 100 --delay 0.2
"""

import argparse
import random
import time
import httpx
import numpy as np


def generate_synthetic_ecg(leads: int = 12, length: int = 5000) -> list[list[float]]:
    """
    Generates a realistic-looking synthetic multi-lead ECG signal.
    Combines a sine wave (simulating cardiac cycle) with small Gaussian noise.
    """
    t = np.linspace(0, 10, length)
    base_signal = np.sin(2 * np.pi * 1.2 * t)  # ~72 beats per minute
    
    leads_data = []
    for _ in range(leads):
        lead_signal = base_signal + np.random.normal(0, 0.05, size=length)
        leads_data.append(lead_signal.tolist())
    
    return leads_data


def run_traffic_simulation(base_url: str, total_requests: int, delay: float):
    print("=" * 65)
    print("  AI Inference Platform - Traffic Simulator")
    print(f"  Target: {base_url}")
    print(f"  Total Requests: {total_requests if total_requests > 0 else 'Infinite (Ctrl+C to stop)'}")
    print(f"  Delay between requests: {delay}s")
    print("=" * 65)

    endpoints = [
        "/api/v1/predict/ecg",
        "/api/v2/predict/ecg",
        "/api/health",
    ]

    client = httpx.Client(base_url=base_url, timeout=10.0)
    count = 0

    try:
        while total_requests == 0 or count < total_requests:
            count += 1
            # 80% v1 traffic, 15% v2 (canary) traffic, 5% health checks
            rand = random.random()
            if rand < 0.75:
                endpoint = "/api/v1/predict/ecg"
            elif rand < 0.95:
                endpoint = "/api/v2/predict/ecg"
            else:
                endpoint = "/api/health"

            start_time = time.perf_counter()

            if endpoint == "/api/health":
                response = client.get(endpoint)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"[{count:04d}] GET  {endpoint:<22} -> {response.status_code} ({elapsed_ms:.1f}ms)")
            else:
                # 5% chance of sending bad payload to trigger and verify error metrics (HTTP 422)
                is_faulty = random.random() < 0.05
                if is_faulty:
                    payload = {"signal": [[0.0] * 10]}  # Incorrect length
                else:
                    payload = {"signal": generate_synthetic_ecg()}

                response = client.post(endpoint, json=payload)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    cls = data.get("predicted_class", "N/A")
                    conf = data.get("confidence", 0.0)
                    model_ver = data.get("model_version", "v1")
                    print(
                        f"[{count:04d}] POST {endpoint:<22} -> {response.status_code} "
                        f"({elapsed_ms:5.1f}ms) | {model_ver} | class: {cls} (conf: {conf:.2%})"
                    )
                else:
                    print(
                        f"[{count:04d}] POST {endpoint:<22} -> {response.status_code} "
                        f"({elapsed_ms:5.1f}ms) [Expected validation error]"
                    )

            time.sleep(delay)

    except KeyboardInterrupt:
        print("\n[Stopped] Simulation interrupted by user.")
    finally:
        client.close()
        print(f"\nCompleted {count} requests.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate API traffic for testing & Prometheus metrics.")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the inference API")
    parser.add_argument("--requests", type=int, default=30, help="Number of requests to send (0 = infinite)")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay in seconds between requests")
    args = parser.parse_args()

    run_traffic_simulation(args.url, args.requests, args.delay)
