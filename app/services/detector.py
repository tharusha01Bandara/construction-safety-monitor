import os
from ultralytics import YOLO
from ..config import settings

class YOLOModel:
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(settings.MODEL_PATH):
            self.model = YOLO(settings.MODEL_PATH)
        else:
            print(f"Warning: Model not found at {settings.MODEL_PATH}")
            self.model = None
            
    def predict(self, image):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        return self.model(image)

detector = YOLOModel()
