"""
drift.py — API Endpoints for Drift Monitoring, Retraining Triggers, and Alertmanager Webhooks.

Endpoints:
  - GET  /api/drift/status             -> Real-time drift metrics and active alert status
  - POST /api/drift/trigger-retraining -> Manually trigger a model retraining job
  - GET  /api/drift/jobs               -> List all retraining jobs
  - GET  /api/drift/jobs/{job_id}      -> Inspect a specific retraining job
  - POST /api/drift/webhook            -> Prometheus Alertmanager webhook receiver
"""

import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drift", tags=["Drift Detection & Retraining"])


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class DriftStatusResponse(BaseModel):
    severity: str
    total_samples: int
    window_size: int
    failed_samples: int
    passed_samples: int
    failure_rate_pct: float
    threshold_pct: float
    alert_active: bool
    last_alert_time: Optional[str]
    last_eval_time: Optional[str]
    issue_breakdown: Dict[str, int]


class TriggerRetrainingRequest(BaseModel):
    reason: str = Field(default="Manual trigger from admin console", description="Reason for retraining")
    auto_activate: bool = Field(default=False, description="Whether to automatically switch traffic to the new model")


class RetrainingJobResponse(BaseModel):
    job_id: str
    status: str
    trigger_type: str
    reason: str
    started_at: str
    completed_at: Optional[str]
    candidate_model_version: Optional[str]
    metrics: Dict
    auto_activate: bool
    error: Optional[str]


class AlertManagerAlert(BaseModel):
    status: Optional[str] = "firing"
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None


class AlertManagerPayload(BaseModel):
    version: Optional[str] = None
    status: Optional[str] = "firing"
    receiver: Optional[str] = None
    alerts: Optional[List[AlertManagerAlert]] = None


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status", response_model=DriftStatusResponse)
async def get_drift_status(request: Request):
    """
    Returns real-time data drift statistics and active alert status.
    """
    detector = getattr(request.app.state, "drift_detector", None)
    if not detector:
        raise HTTPException(status_code=503, detail="Drift detector not initialized.")
    return detector.get_status()


@router.post("/trigger-retraining", response_model=RetrainingJobResponse)
async def trigger_retraining(body: TriggerRetrainingRequest, request: Request):
    """
    Manually initiates an asynchronous model retraining job.
    """
    pipeline = getattr(request.app.state, "retraining_pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="Retraining pipeline not initialized.")

    job = pipeline.trigger_job(
        trigger_type="manual",
        reason=body.reason,
        auto_activate=body.auto_activate,
    )
    return job


@router.get("/jobs", response_model=List[RetrainingJobResponse])
async def list_retraining_jobs(request: Request):
    """
    Lists all past and active retraining jobs.
    """
    pipeline = getattr(request.app.state, "retraining_pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="Retraining pipeline not initialized.")
    return pipeline.list_jobs()


@router.get("/jobs/{job_id}", response_model=RetrainingJobResponse)
async def get_retraining_job(job_id: str, request: Request):
    """
    Inspects status and metrics of a specific retraining job.
    """
    pipeline = getattr(request.app.state, "retraining_pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="Retraining pipeline not initialized.")

    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.post("/webhook")
async def alertmanager_webhook(payload: AlertManagerPayload, request: Request):
    """
    Webhook endpoint for Prometheus Alertmanager.
    When Alertmanager posts a firing drift alert, this endpoint initiates automated retraining.
    """
    pipeline = getattr(request.app.state, "retraining_pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="Retraining pipeline not initialized.")

    logger.info(f"[WEBHOOK] Received Alertmanager notification: status={payload.status}")

    # Check if any alert is firing
    if payload.status == "firing":
        alert_names = []
        if payload.alerts:
            for alert in payload.alerts:
                if alert.labels and "alertname" in alert.labels:
                    alert_names.append(alert.labels["alertname"])

        reason_str = f"Alertmanager alert firing: {', '.join(alert_names) if alert_names else 'DriftAlert'}"
        job = pipeline.trigger_job(
            trigger_type="webhook",
            reason=reason_str,
            auto_activate=False,
        )
        return {
            "status": "acknowledged",
            "message": "Retraining job launched from alert webhook",
            "job_id": job["job_id"],
        }

    return {"status": "ignored", "message": f"Alert status is '{payload.status}'"}
