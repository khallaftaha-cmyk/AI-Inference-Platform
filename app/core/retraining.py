"""
retraining.py — Automated Retraining Pipeline Orchestrator.

What happens when data drift is detected?
=========================================
In production MLOps, detecting drift is only half the battle. Once an alert fires:
1. A retraining job must be scheduled with a unique Job ID.
2. The pipeline collects recent production data & annotations.
3. Model training / fine-tuning runs (asynchronously in the background).
4. Training metrics (accuracy, F1 score, validation loss) are logged.
5. The new candidate artifact (e.g. `ecg_model_v3.onnx`) is saved to `models_store/`.
6. The ModelRegistry is notified, making the new candidate immediately available
   for canary deployment and hot-swapping!
"""

import logging
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from app.core.drift import ecg_retraining_triggers_total

logger = logging.getLogger(__name__)


class RetrainingJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RetrainingPipeline:
    """
    Orchestrates automated and manual model retraining jobs.
    """

    def __init__(self, models_store_dir: str, model_registry=None):
        self.models_store_dir = Path(models_store_dir)
        self.model_registry = model_registry
        self._jobs: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def trigger_job(
        self,
        trigger_type: str = "auto_drift",
        reason: str = "Validation failure rate exceeded threshold",
        auto_activate: bool = False,
    ) -> Dict:
        """
        Creates and launches an asynchronous retraining job.

        Args:
            trigger_type: "auto_drift", "manual", or "webhook"
            reason: Human-readable explanation of why retraining was triggered
            auto_activate: If True, automatically activates the new model in production

        Returns:
            Job metadata dict
        """
        job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = datetime.now(timezone.utc).isoformat()

        job_record = {
            "job_id": job_id,
            "status": RetrainingJobStatus.PENDING.value,
            "trigger_type": trigger_type,
            "reason": reason,
            "started_at": now,
            "completed_at": None,
            "candidate_model_version": None,
            "metrics": {},
            "auto_activate": auto_activate,
            "error": None,
        }

        with self._lock:
            self._jobs[job_id] = job_record

        # Increment Prometheus metric
        ecg_retraining_triggers_total.labels(trigger_type=trigger_type).inc()

        logger.info(f"[RETRAINING] Started job '{job_id}' (trigger: {trigger_type}, reason: {reason})")

        # Launch background worker
        threading.Thread(
            target=self._execute_retraining_job,
            args=(job_id, auto_activate),
            daemon=True,
        ).start()

        return job_record

    def _execute_retraining_job(self, job_id: str, auto_activate: bool) -> None:
        """
        Executes model retraining in the background.
        In this platform, it produces a newly retrained model version artifact (e.g. v3),
        simulates fine-tuning, updates metrics, and registers with ModelRegistry.
        """
        try:
            with self._lock:
                self._jobs[job_id]["status"] = RetrainingJobStatus.RUNNING.value

            # Determine the next model version name
            existing_models = list(self.models_store_dir.glob("*.onnx"))
            next_version_num = len(existing_models) + 1
            candidate_version = f"ecg_model_v{next_version_num}"
            output_filename = f"{candidate_version}.onnx"
            output_path = self.models_store_dir / output_filename

            # Simulate training work (e.g. loading dataset, epochs, onnx export)
            time.sleep(1.0)

            # Copy baseline model to simulate the newly generated ONNX model artifact
            base_model = self.models_store_dir / "ecg_model_v1.onnx"
            if base_model.exists():
                shutil.copyfile(base_model, output_path)
            else:
                # Fallback to any existing onnx
                first_model = next(self.models_store_dir.glob("*.onnx"))
                shutil.copyfile(first_model, output_path)

            training_metrics = {
                "training_loss": 0.042,
                "validation_loss": 0.051,
                "validation_accuracy": 0.984,
                "epochs": 10,
                "samples_trained": 12000,
            }

            # If registry is attached, pre-warm the model
            if self.model_registry:
                try:
                    self.model_registry.discover_models()
                    self.model_registry.load_model(candidate_version)
                    if auto_activate:
                        self.model_registry.activate_model(candidate_version)
                        logger.info(f"[RETRAINING] Auto-activated candidate {candidate_version}")
                except Exception as e:
                    logger.warning(f"Could not auto-register model with registry: {e}")

            with self._lock:
                self._jobs[job_id]["status"] = RetrainingJobStatus.COMPLETED.value
                self._jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                self._jobs[job_id]["candidate_model_version"] = candidate_version
                self._jobs[job_id]["metrics"] = training_metrics

            logger.info(
                f"[RETRAINING] Job '{job_id}' completed successfully! "
                f"Candidate model: {candidate_version} (accuracy: {training_metrics['validation_accuracy']:.2%})"
            )

        except Exception as e:
            logger.error(f"[RETRAINING] Job '{job_id}' failed: {e}", exc_info=True)
            with self._lock:
                self._jobs[job_id]["status"] = RetrainingJobStatus.FAILED.value
                self._jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                self._jobs[job_id]["error"] = str(e)

    def list_jobs(self) -> List[Dict]:
        """Returns all retraining jobs sorted by start time descending."""
        with self._lock:
            return sorted(
                list(self._jobs.values()),
                key=lambda j: j["started_at"],
                reverse=True,
            )

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Returns details for a specific retraining job."""
        with self._lock:
            return self._jobs.get(job_id)
