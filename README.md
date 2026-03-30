# Construction Safety Monitor

A FastAPI-based application for detecting construction safety compliance (helmets, vests) using YOLOv8.

## Overview
This system processes input images, detects workers and their PPE (helmets and high-visibility vests) using a trained YOLOv8 model, applies safety logic, and returns a detailed JSON response or an annotated image.

### Safety Rules
- **Rule 1**: Every worker must wear a helmet.
- **Rule 2**: Every worker must wear a high-visibility vest.
- **Rule 3**: If one is missing, that worker is unsafe.
- **Rule 4**: If any worker is unsafe, the whole scene is unsafe.

### Edge-Case Handling & Confidence Scoring
- **Confidence Scoring**: Worker confidence is calculated as the average of their person detection confidence and matched PPE confidences. Scene confidence is the average across all workers.
- **Review Flagger**: If the average confidence of a worker's prediction falls below `0.6` (configurable), they are flagged as `review_needed = true`.
- **Tiny False Positives**: Small person boxes (area < 1000 px) are ignored as false positives or people too far to reasonably check.
- **Spatial Matching**: Helmets and vests are matched by checking if their bounding boxes are physically inside a person box (Intersection over Area logic).

## Setup & Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your trained YOLOv8 model (`best.pt`) in the `models/` directory. (Note: Ensure the classes learned by your model match: `0: person`, `1: helmet`, `2: vest`). You can adjust these in `app/routes/predict.py`.

## Running the Server

Start the application with Uvicorn:
```bash
uvicorn app.main:app --reload
```

## API Usage

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Predict Scene (JSON Only)
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_image.jpg"
```

### 3. Predict Scene and Visualize
```bash
curl -X POST http://localhost:8000/predict/visualize \
  -F "file=@sample_image.jpg"
```
Outputs standard JSON + an `output_image_path` linking to the annotated photo in the `outputs/` folder.

## Limitations
- Performance depends highly on the quality of `best.pt`.
- Crowded scenes may have overlapping boxes causing mismatched PPE. (Can be improved with IoU-based bipartite matching).
