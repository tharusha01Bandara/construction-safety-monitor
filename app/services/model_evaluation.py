import json
import os
from datetime import datetime
from typing import Any, Dict, List

from fastapi import HTTPException

from ..config import settings
from .detector import detector


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _build_failure_cases(precision: float | None, recall: float | None, map50_95: float | None) -> List[str]:
    failure_cases: List[str] = []

    if precision is not None and precision < 0.60:
        failure_cases.append(
            "Low precision: model may produce too many false positives (unsafe alerts where PPE is actually present)."
        )

    if recall is not None and recall < 0.60:
        failure_cases.append(
            "Low recall: model may miss real safety violations, especially distant or occluded workers."
        )

    if map50_95 is not None and map50_95 < 0.50:
        failure_cases.append(
            "Low mAP50-95: localization and generalization are weak across IoU thresholds."
        )

    if not failure_cases:
        failure_cases.append(
            "No major threshold-based red flags found, but field validation is still required for new camera angles and lighting."
        )

    return failure_cases


def _cache_path() -> str:
    return os.path.join(settings.OUTPUT_DIR, "model_metrics.json")


def evaluate_model_performance(refresh: bool = False) -> Dict[str, Any]:
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    cache_file = _cache_path()

    if not refresh and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached["source"] = "cache"
        return cached

    if detector.model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check MODEL_PATH.")

    if not os.path.exists(settings.DATASET_YAML_PATH):
        return {
            "status": "unavailable",
            "metrics_available": False,
            "source": "live",
            "model_path": settings.MODEL_PATH,
            "dataset_yaml_path": settings.DATASET_YAML_PATH,
            "split": settings.EVAL_SPLIT,
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP50_95": None,
            "fitness": None,
            "failure_cases": [
                "Validation dataset config not found. Add data.yaml and dataset labels to compute precision/recall/mAP honestly."
            ],
            "notes": [
                "No metrics were fabricated. This response is intentionally marked unavailable.",
                "Run again after adding DATASET_YAML_PATH to enable real model validation."
            ],
        }

    try:
        results = detector.model.val(
            data=settings.DATASET_YAML_PATH,
            split=settings.EVAL_SPLIT,
            imgsz=settings.EVAL_IMAGE_SIZE,
            batch=settings.EVAL_BATCH_SIZE,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {exc}")

    precision = _safe_float(getattr(results.box, "mp", None))
    recall = _safe_float(getattr(results.box, "mr", None))
    map50 = _safe_float(getattr(results.box, "map50", None))
    map50_95 = _safe_float(getattr(results.box, "map", None))
    fitness = _safe_float(getattr(results.box, "fitness", None))

    payload: Dict[str, Any] = {
        "status": "ok",
        "metrics_available": True,
        "source": "live",
        "model_path": settings.MODEL_PATH,
        "dataset_yaml_path": settings.DATASET_YAML_PATH,
        "split": settings.EVAL_SPLIT,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "fitness": fitness,
        "failure_cases": _build_failure_cases(precision, recall, map50_95),
        "notes": [
            "Metrics are produced from the configured validation split and not estimated from production requests.",
            "Use the same camera domain in validation data to avoid optimistic reporting.",
            f"Evaluated at {datetime.utcnow().isoformat()}Z",
        ],
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload
