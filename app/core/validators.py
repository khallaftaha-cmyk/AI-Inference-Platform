"""
validators.py — Input data quality checks for ECG signals.

What is "Data Drift" and why does it matter?
============================================
Imagine your model was trained on ECG signals recorded by a specific device
in a hospital in France. That model learned patterns from THAT device's data.

Now someone sends signals from a completely different device — or someone
accidentally sends all zeros, or a signal with the wrong units.

The model will still give you an answer. It might even be a confident answer.
But it's GARBAGE. This is called "data drift" or "input shift."

As an AI Platform Engineer, your job is to CATCH this before it reaches the model.
This file does exactly that — it's a data quality gate.

What we check here:
  1. Correct shape: 12 leads × 5000 samples
  2. No NaN (Not a Number) or Inf values (numerical corruption)
  3. Reasonable amplitude range (ECG values are typically -5mV to +5mV)
  4. No flatline leads (a constant flat signal means a disconnected electrode)
  5. Signal variance: too little = noise/artifact, too much = corrupted

These checks are also tracked as Prometheus metrics so you can see in Grafana
when bad inputs start arriving — a leading indicator of production problems.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS — track input quality issues over time
#
# Counter: A metric that only goes UP (never resets).
#          Perfect for counting errors, bad inputs, etc.
# Histogram: Tracks distribution of values (e.g., how often is signal amplitude high?)
# ─────────────────────────────────────────────────────────────────────────────

# Counts total validation attempts
ecg_validation_total = Counter(
    "ecg_validation_total",
    "Total number of ECG signal validation attempts",
    ["result"],  # label: "passed" or "failed"
)

# Counts specific quality issues found
ecg_quality_issues = Counter(
    "ecg_quality_issues_total",
    "Count of specific ECG data quality issues detected",
    ["issue_type"],  # e.g., "nan_detected", "flatline_lead", "wrong_shape"
)

# Tracks the amplitude range of incoming signals (tells us if scale is drifting)
ecg_signal_amplitude = Histogram(
    "ecg_signal_amplitude_mv",
    "Histogram of peak-to-peak amplitude of incoming ECG signals (mV)",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
)

# Tracks signal-to-noise ratio proxy (std dev across the signal)
ecg_signal_std = Histogram(
    "ecg_signal_std",
    "Histogram of per-lead standard deviation of incoming ECG signals",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION RESULT — a structured object returned from validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Holds the outcome of a signal validation.

    is_valid: True if the signal passes all quality checks.
    issues:   List of human-readable problem descriptions.
    warnings: Non-blocking issues worth logging but not rejecting.
    stats:    Signal statistics for debugging and observability.

    Example:
        result = validate_ecg_signal(signal)
        if not result.is_valid:
            raise HTTPException(422, detail=result.issues)
    """
    is_valid: bool = True
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add_issue(self, msg: str):
        """Add a hard error that causes rejection."""
        self.is_valid = False
        self.issues.append(msg)

    def add_warning(self, msg: str):
        """Add a soft warning — logged but doesn't reject the signal."""
        self.warnings.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — limits and thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Expected ECG dimensions (must match model training data)
EXPECTED_LEADS = 12
EXPECTED_LENGTH = 5000

# ECG amplitude limits in millivolts.
# Real clinical ECGs rarely exceed ±5mV. Beyond ±15mV is almost certainly a bug.
MIN_AMPLITUDE_MV = -15.0
MAX_AMPLITUDE_MV = 15.0

# A "flatline" is when the entire lead has no variation.
# Standard deviation below this threshold = likely disconnected electrode.
FLATLINE_STD_THRESHOLD = 1e-4

# Leads with very high standard deviation are likely corrupted / artifact.
HIGH_NOISE_STD_THRESHOLD = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def validate_ecg_signal(
    signal: list[list[float]],
    expected_leads: int = EXPECTED_LEADS,
    expected_length: int = EXPECTED_LENGTH,
    patient_id: Optional[str] = None,
) -> ValidationResult:
    """
    Validate an ECG signal for data quality before passing it to the model.

    This function acts as a "gate" — if the signal looks suspicious,
    we return issues before wasting GPU/CPU on bad data, and before
    the model produces an unreliable prediction.

    Args:
        signal:          2D list: [num_leads][num_samples]
        expected_leads:  How many leads the model expects (default: 12)
        expected_length: How many samples per lead (default: 5000)
        patient_id:      Optional ID for logging

    Returns:
        ValidationResult with .is_valid, .issues, .warnings, .stats
    """
    result = ValidationResult()
    pid = patient_id or "unknown"

    # ── Check 1: Shape ────────────────────────────────────────────────────────
    num_leads = len(signal)
    if num_leads != expected_leads:
        result.add_issue(
            f"Wrong number of leads: got {num_leads}, expected {expected_leads}."
        )
        ecg_quality_issues.labels(issue_type="wrong_num_leads").inc()

    # Check each lead has the right length
    wrong_length_leads = [
        i for i, lead in enumerate(signal)
        if len(lead) != expected_length
    ]
    if wrong_length_leads:
        result.add_issue(
            f"Wrong signal length in leads {wrong_length_leads}: "
            f"expected {expected_length} samples."
        )
        ecg_quality_issues.labels(issue_type="wrong_signal_length").inc()

    # ── Early exit: If shape is wrong, numeric checks will be misleading ──────
    if not result.is_valid:
        ecg_validation_total.labels(result="failed").inc()
        logger.warning(f"[Validation FAILED] patient={pid} | shape errors: {result.issues}")
        return result

    # ── Convert to numpy for efficient numeric checks ─────────────────────────
    arr = np.array(signal, dtype=np.float64)  # shape: (num_leads, num_samples)

    # ── Check 2: NaN / Inf values ─────────────────────────────────────────────
    # NaN = Not a Number (e.g., 0/0). Inf = infinity. Both crash model inference.
    if np.any(np.isnan(arr)):
        result.add_issue("Signal contains NaN (Not a Number) values — likely corrupt data.")
        ecg_quality_issues.labels(issue_type="nan_detected").inc()

    if np.any(np.isinf(arr)):
        result.add_issue("Signal contains Inf (infinite) values — likely corrupt data.")
        ecg_quality_issues.labels(issue_type="inf_detected").inc()

    # Stop here if data is numerically broken — further checks would fail
    if not result.is_valid:
        ecg_validation_total.labels(result="failed").inc()
        logger.warning(f"[Validation FAILED] patient={pid} | numeric errors: {result.issues}")
        return result

    # ── Check 3: Amplitude range ──────────────────────────────────────────────
    global_min = float(arr.min())
    global_max = float(arr.max())
    peak_to_peak = global_max - global_min

    # Record amplitude to Prometheus histogram (for trend monitoring)
    ecg_signal_amplitude.observe(peak_to_peak)

    if global_min < MIN_AMPLITUDE_MV or global_max > MAX_AMPLITUDE_MV:
        result.add_warning(
            f"Signal amplitude out of typical ECG range "
            f"[{MIN_AMPLITUDE_MV}, {MAX_AMPLITUDE_MV}] mV. "
            f"Got range [{global_min:.2f}, {global_max:.2f}]. "
            f"Check if units are correct (mV expected)."
        )
        ecg_quality_issues.labels(issue_type="amplitude_out_of_range").inc()

    # ── Check 4: Flatline detection ───────────────────────────────────────────
    # A disconnected electrode produces a flat line (zero variation).
    per_lead_std = np.std(arr, axis=1)  # shape: (num_leads,)
    flatline_leads = [i for i, s in enumerate(per_lead_std) if s < FLATLINE_STD_THRESHOLD]

    if flatline_leads:
        result.add_issue(
            f"Flatline detected in leads {flatline_leads}. "
            f"These leads show no variation — likely disconnected electrodes."
        )
        ecg_quality_issues.labels(issue_type="flatline_lead").inc()

    # ── Check 5: High-noise leads ─────────────────────────────────────────────
    noisy_leads = [i for i, s in enumerate(per_lead_std) if s > HIGH_NOISE_STD_THRESHOLD]
    if noisy_leads:
        result.add_warning(
            f"High noise in leads {noisy_leads} (std > {HIGH_NOISE_STD_THRESHOLD}). "
            f"Could be muscle artifact or poor electrode contact."
        )
        ecg_quality_issues.labels(issue_type="high_noise_lead").inc()

    # Record per-lead std to Prometheus (track noise trends over time)
    for std_val in per_lead_std:
        ecg_signal_std.observe(float(std_val))

    # ── Collect summary statistics for logging/debugging ─────────────────────
    result.stats = {
        "num_leads": num_leads,
        "signal_length": expected_length,
        "global_min_mv": round(global_min, 4),
        "global_max_mv": round(global_max, 4),
        "peak_to_peak_mv": round(peak_to_peak, 4),
        "mean_lead_std": round(float(per_lead_std.mean()), 4),
    }

    # ── Final verdict ─────────────────────────────────────────────────────────
    if result.is_valid:
        ecg_validation_total.labels(result="passed").inc()
        if result.warnings:
            logger.warning(
                f"[Validation PASSED with warnings] patient={pid} | "
                f"warnings={result.warnings}"
            )
        else:
            logger.debug(f"[Validation OK] patient={pid} | stats={result.stats}")
    else:
        ecg_validation_total.labels(result="failed").inc()
        logger.warning(
            f"[Validation FAILED] patient={pid} | issues={result.issues}"
        )

    return result
