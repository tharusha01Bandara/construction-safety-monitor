from fastapi import APIRouter, UploadFile, File, HTTPException
from ..schemas import PredictionResponse, VisualizeResponse
from ..services.detector import detector
from ..services.safety_logic import apply_safety_rules
from ..services.reporting import generate_alert_report
from ..services.visualization import draw_visualizations
import numpy as np
from PIL import Image
import io

router = APIRouter()

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
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0].tolist()
            c = float(box.conf)
            cls = int(box.cls)
            # Assuming classes mapping: 0: person, 1: helmet, 2: vest (adjust based on your actual model)
            if cls == 0: persons.append((b, c))
            elif cls == 1: helmets.append((b, c))
            elif cls == 2: vests.append((b, c))
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
    
    output_path = draw_visualizations(image_np, workers, scene_status)
    
    return {
        "scene_status": scene_status,
        "scene_confidence": scene_conf,
        "workers": workers,
        "alert_report": report,
        "output_image_path": output_path
    }
