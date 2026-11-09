"""
Synthetic Traffic Generator for AI Inference Platform.

This script simulates client traffic by sending continuous ECG prediction
requests to the FastAPI inference endpoints. It serves two purposes:
1. Demonstrates that the inference API is working under real load.
2. Generates live metrics for Prometheus to scrape and Grafana to visualize.
3. Tests data drift detection: with `--drift`, it increases corruption rate to
   trigger the automated drift alert (>5%) and initiate background model retraining!

Usage:
    python scripts/simulate_traffic.py
    python scripts/simulate_traffic.py --requests 50 --delay 0.2
    python scripts/simulate_traffic.py --drift --requests 30   # triggers drift alert!
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


def generate_corrupted_ecg(leads: int = 12, length: int = 5000) -> list[list[float]]:
    """
    Generates realistic sensor failure/drift patterns:
    - Flatline lead (disconnected electrode)
    - Wrong lead count or sample count
    - Extreme noise
    """
    corrupt_type = random.choice(["flatline", "wrong_shape", "nan_noise"])

    if corrupt_type == "flatline":
        signal = generate_synthetic_ecg(leads, length)
        # Disconnect lead 2 or 3
        disconnected_lead = random.randint(0, leads - 1)
        signal[disconnected_lead] = [0.0] * length
        return signal

    elif corrupt_type == "wrong_shape":
        # Wrong length or lead count
        return [[0.0] * 50]

    else:
        # Extreme voltage out of bounds
        signal = generate_synthetic_ecg(leads, length)
        return (np.array(signal) * 50.0).tolist()


def run_traffic_simulation(base_url: str, total_requests: int, delay: float, drift_mode: bool, corrupt_rate: float):
    active_corrupt_rate = corrupt_rate if not drift_mode else 0.30

    print("=" * 65)
    print("  AI Inference Platform - Traffic Simulator")
    print(f"  Target: {base_url}")
    print(f"  Total Requests: {total_requests if total_requests > 0 else 'Infinite (Ctrl+C to stop)'}")
    print(f"  Delay between requests: {delay}s")
    print(f"  Drift Mode Active: {drift_mode} (Corrupt Rate: {active_corrupt_rate:.0%})")
    print("=" * 65)

    client = httpx.Client(base_url=base_url, timeout=10.0)
    count = 0

    try:
        while total_requests == 0 or count < total_requests:
            count += 1
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
                is_corrupted = random.random() < active_corrupt_rate
                if is_corrupted:
                    payload = {"signal": generate_corrupted_ecg(), "patient_id": f"p-corrupt-{count}"}
                else:
                    payload = {"signal": generate_synthetic_ecg(), "patient_id": f"p-normal-{count}"}

                response = client.post(endpoint, json=payload)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    cls = data.get("predicted_class", "N/A")
                    conf = data.get("confidence", 0.0)
                    model_ver = data.get("model_version", "v1")
                    warns = data.get("validation_warnings", [])
                    warn_str = f" [Warnings: {len(warns)}]" if warns else ""
                    print(
                        f"[{count:04d}] POST {endpoint:<22} -> {response.status_code} "
                        f"({elapsed_ms:5.1f}ms) | {model_ver} | class: {cls} (conf: {conf:.2%}){warn_str}"
                    )
                else:
                    print(
                        f"[{count:04d}] POST {endpoint:<22} -> {response.status_code} "
                        f"({elapsed_ms:5.1f}ms) [Validation rejected -> Drift logged]"
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
    parser.add_argument("--drift", action="store_true", help="Simulate data drift with high failure rate (>5%)")
    parser.add_argument("--corrupt-rate", type=float, default=0.04, help="Normal corruption rate (default: 4%)")
    args = parser.parse_args()

    run_traffic_simulation(args.url, args.requests, args.delay, args.drift, args.corrupt_rate)
