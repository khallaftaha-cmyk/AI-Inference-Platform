"""
drift.py — Real-Time Data Drift & Quality Anomaly Detector.

What is Data Drift in MLOps?
============================
Even without code bugs, an ML model degrades when real-world incoming data diverges
from the training distribution. For ECG signals, drift shows up as:
  - Spike in sensor flatlines (loose electrodes).
  - Amplitude scaling changes (different recording equipment or unit conversions).
  - Excess noise artifacts (patient movement, baseline wander).
  - Increased validation rejection rates (>5%).

This module maintains a thread-safe sliding window of recent validation observations.
When the failure rate crosses the configured threshold, it:
  1. Emits Prometheus gauge metrics (`ecg_drift_failure_rate`, `ecg_drift_alert_active`).
  2. Raises an active drift alert.
  3. Optionally invokes an automated retraining trigger callback.
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional

from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS FOR DRIFT
# ─────────────────────────────────────────────────────────────────────────────

ecg_drift_failure_rate = Gauge(
    "ecg_drift_failure_rate_pct",
    "Current sliding-window percentage of ECG validation failures",
)

ecg_drift_alert_active = Gauge(
    "ecg_drift_alert_active",
    "1 if ECG validation failure rate exceeds threshold, 0 otherwise",
)

ecg_retraining_triggers_total = Counter(
    "ecg_retraining_triggers_total",
    "Total number of times a retraining pipeline has been triggered",
    ["trigger_type"],  # e.g., "auto_drift", "manual", "webhook"
)


class DriftSeverity(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL_DRIFT = "CRITICAL_DRIFT"


@dataclass
class ValidationRecord:
    timestamp: datetime
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    patient_id: Optional[str] = None


class DriftDetector:
    """
    Monitors sliding-window data validation health and triggers automated retraining.
    """

    def __init__(
        self,
        window_size: int = 50,
        drift_threshold_pct: float = 5.0,
        min_samples: int = 10,
        on_drift_detected: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Args:
            window_size: Number of recent predictions to maintain in sliding window.
            drift_threshold_pct: Percentage of failures (0-100) that triggers a drift alert.
            min_samples: Minimum records needed in window before alerting.
            on_drift_detected: Optional callback invoked when drift is first triggered.
        """
        self.window_size = window_size
        self.drift_threshold_pct = drift_threshold_pct
        self.min_samples = min_samples
        self.on_drift_detected = on_drift_detected

        self._window: Deque[ValidationRecord] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._alert_active: bool = False
        self._last_alert_time: Optional[datetime] = None
        self._last_eval_time: Optional[datetime] = None

    def record_validation(
        self,
        is_valid: bool,
        issues: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        patient_id: Optional[str] = None,
    ) -> Dict:
        """
        Records a single validation outcome and re-evaluates drift metrics.
        Returns the current drift status summary.
        """
        record = ValidationRecord(
            timestamp=datetime.now(timezone.utc),
            is_valid=is_valid,
            issues=issues or [],
            warnings=warnings or [],
            patient_id=patient_id,
        )

        with self._lock:
            self._window.append(record)
            self._last_eval_time = datetime.now(timezone.utc)
            status = self._evaluate_drift_unlocked()

        return status

    def get_status(self) -> Dict:
        """Returns the current drift health and window statistics."""
        with self._lock:
            return self._evaluate_drift_unlocked()

    def _evaluate_drift_unlocked(self) -> Dict:
        """
        Internal evaluation of the sliding window. Must be called with self._lock held.
        """
        total = len(self._window)
        if total == 0:
            ecg_drift_failure_rate.set(0.0)
            ecg_drift_alert_active.set(0)
            return {
                "severity": DriftSeverity.NORMAL.value,
                "total_samples": 0,
                "window_size": self.window_size,
                "failed_samples": 0,
                "passed_samples": 0,
                "failure_rate_pct": 0.0,
                "threshold_pct": self.drift_threshold_pct,
                "alert_active": False,
                "last_alert_time": None,
                "last_eval_time": self._last_eval_time.isoformat() if self._last_eval_time else None,
                "issue_breakdown": {},
            }

        failed_records = [r for r in self._window if not r.is_valid]
        failed_count = len(failed_records)
        passed_count = total - failed_count
        failure_rate = (failed_count / total) * 100.0

        # Update Prometheus gauge
        ecg_drift_failure_rate.set(round(failure_rate, 2))

        # Collect breakdown of specific issues
        issue_counts: Dict[str, int] = {}
        for r in failed_records:
            for issue in r.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        # Check alert conditions
        severity = DriftSeverity.NORMAL
        was_alert_active = self._alert_active
        trigger_callback = False

        if total >= self.min_samples:
            if failure_rate >= self.drift_threshold_pct:
                severity = DriftSeverity.CRITICAL_DRIFT
                self._alert_active = True
                self._last_alert_time = datetime.now(timezone.utc)
                ecg_drift_alert_active.set(1)

                # If this is a transition to active alert, trigger callback
                if not was_alert_active and self.on_drift_detected:
                    trigger_callback = True
            elif failure_rate >= (self.drift_threshold_pct / 2.0):
                severity = DriftSeverity.WARNING
                self._alert_active = False
                ecg_drift_alert_active.set(0)
            else:
                severity = DriftSeverity.NORMAL
                self._alert_active = False
                ecg_drift_alert_active.set(0)
        else:
            self._alert_active = False
            ecg_drift_alert_active.set(0)

        summary = {
            "severity": severity.value,
            "total_samples": total,
            "window_size": self.window_size,
            "failed_samples": failed_count,
            "passed_samples": passed_count,
            "failure_rate_pct": round(failure_rate, 2),
            "threshold_pct": self.drift_threshold_pct,
            "alert_active": self._alert_active,
            "last_alert_time": self._last_alert_time.isoformat() if self._last_alert_time else None,
            "last_eval_time": self._last_eval_time.isoformat() if self._last_eval_time else None,
            "issue_breakdown": issue_counts,
        }

        # Fire callback outside the critical path if needed
        if trigger_callback:
            logger.warning(
                f"[DRIFT DETECTED] Failure rate {failure_rate:.1f}% exceeded threshold "
                f"{self.drift_threshold_pct}%. Triggering retraining callback."
            )
            # Run in separate thread so request processing is not blocked
            threading.Thread(
                target=self.on_drift_detected,
                args=(summary,),
                daemon=True,
            ).start()

        return summary

    def reset_window(self) -> None:
        """Clears the sliding window and resets alert state."""
        with self._lock:
            self._window.clear()
            self._alert_active = False
            ecg_drift_failure_rate.set(0.0)
            ecg_drift_alert_active.set(0)
            logger.info("DriftDetector sliding window reset.")
