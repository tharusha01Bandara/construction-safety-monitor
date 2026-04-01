from typing import List, Optional
from pydantic import BaseModel

class WorkerResult(BaseModel):
    worker_id: int
    helmet: bool
    helmet_conf: Optional[float]
    vest: bool
    vest_conf: Optional[float]
    person_conf: float
    status: str  # "safe" or "unsafe"
    violations: List[str]
    confidence: float
    review_needed: bool

class PredictionResponse(BaseModel):
    scene_status: str
    scene_confidence: float
    workers: List[WorkerResult]
    alert_report: str

class VisualizeResponse(PredictionResponse):
    output_image_path: str

# --- TEMPORAL ANALYSIS RESPONSE MODELS ---
class VideoWorkerAlert(BaseModel):
    track_id: int
    alert_type: str
    frames: List[int]
    message: str

class VideoPredictionResponse(BaseModel):
    video_status: str
    frames_processed: int
    persistent_scene_alert: bool
    worker_alerts: List[VideoWorkerAlert]
    summary_report: str

class VideoVisualizeResponse(VideoPredictionResponse):
    output_video_path: str


class ModelPerformanceResponse(BaseModel):
    status: str
    metrics_available: bool
    source: str
    model_path: str
    dataset_yaml_path: str
    split: str
    precision: Optional[float]
    recall: Optional[float]
    mAP50: Optional[float]
    mAP50_95: Optional[float]
    fitness: Optional[float]
    failure_cases: List[str]
    notes: List[str]
