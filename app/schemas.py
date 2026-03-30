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
