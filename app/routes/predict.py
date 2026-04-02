from fastapi import APIRouter, UploadFile, File, HTTPException
from ..schemas import (
    PredictionResponse, VisualizeResponse, 
    VideoPredictionResponse, VideoVisualizeResponse, ModelPerformanceResponse
)
from ..services.detector import detector
from ..services.safety_logic import apply_safety_rules
from ..services.reporting import generate_alert_report
from ..services.visualization import draw_visualizations
from ..services.temporal_analysis import run_temporal_analysis, parse_detections
from ..services.model_evaluation import evaluate_model_performance
from ..config import settings
import numpy as np
from PIL import Image
import io
import os
import shutil
import uuid

router = APIRouter()


@router.get("/metrics/model", response_model=ModelPerformanceResponse)
async def model_metrics(refresh: bool = False):
    return evaluate_model_performance(refresh=refresh)

def process_image(file: UploadFile):
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        return image, image_np
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

def parse_detections(results):
    persons, helmets, vests = [], [], []
    names = results[0].names  # Get the dynamic class names mapping from the model
    
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0].tolist()
            c = float(box.conf)
            cls_id = int(box.cls)
            label = names[cls_id]  # e.g., "person", "helmet", "vest"
            
            if label == "person":
                persons.append((b, c))
            elif label == "helmet":
                helmets.append((b, c))
            elif label == "vest":
                vests.append((b, c))
    return persons, helmets, vests

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Must be an image")
        
    image, _ = process_image(file)
    results = detector.predict(image)
    persons, helmets, vests = parse_detections(results)
    
    workers, scene_status, scene_conf = apply_safety_rules(persons, helmets, vests)
    report = generate_alert_report(scene_status, scene_conf, workers)
    
    return {
        "scene_status": scene_status,
        "scene_confidence": scene_conf,
        "workers": workers,
        "alert_report": report
    }

@router.post("/predict/visualize", response_model=VisualizeResponse)
async def predict_visualize(file: UploadFile = File(...)):
    image, image_np = process_image(file)
    results = detector.predict(image)
    persons, helmets, vests = parse_detections(results)
    
    workers, scene_status, scene_conf = apply_safety_rules(persons, helmets, vests)
    report = generate_alert_report(scene_status, scene_conf, workers)
    
    output_path = draw_visualizations(image_np, workers, scene_status, scene_conf)

    return {
        "scene_status": scene_status,
        "scene_confidence": scene_conf,
        "workers": workers,
        "alert_report": report,
        "output_image_path": output_path
    }

# --- NEW TEMPORAL VIDEO ENDPOINTS ---

async def save_temp_video(file: UploadFile) -> str:
    """Helper to save uploaded video to a temp file on disk."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Must be a video file")
    
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    temp_path = os.path.join(settings.OUTPUT_DIR, f"temp_{uuid.uuid4().hex}.mp4")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save video file: {e}")
        
    return temp_path

@router.post("/predict/video", response_model=VideoPredictionResponse)
async def predict_video(file: UploadFile = File(...)):
    temp_path = await save_temp_video(file)
    
    try:
        result_dict, _ = run_temporal_analysis(temp_path, save_visuals=False)
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return result_dict

@router.post("/predict/video/visualize", response_model=VideoVisualizeResponse)
async def predict_video_visualize(file: UploadFile = File(...)):
    temp_path = await save_temp_video(file)
    
    try:
        result_dict, output_video_path = run_temporal_analysis(temp_path, save_visuals=True)
        result_dict["output_video_path"] = output_video_path
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return result_dict
