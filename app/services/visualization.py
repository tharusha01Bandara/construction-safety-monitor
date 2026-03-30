import cv2
import os
from ..config import settings
import uuid

def draw_visualizations(image_np, workers, scene_status):
    img = image_np.copy()
    
    for w in workers:
        box = w["person_box"]
        color = (0, 255, 0) if w["status"] == "safe" else (0, 0, 255)
        
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"W{w['worker_id']} {w['status'].upper()} ({w['confidence']:.2f})"
        cv2.putText(img, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    status_color = (0, 255, 0) if scene_status == "safe" else (0, 0, 255)
    cv2.putText(img, f"SCENE: {scene_status.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
    
    if not os.path.exists(settings.OUTPUT_DIR):
        os.makedirs(settings.OUTPUT_DIR)
        
    filename = f"{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(settings.OUTPUT_DIR, filename)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return path
