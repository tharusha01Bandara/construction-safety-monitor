# 🚀 Training and Results Documentation – PPE Detection

## 1. Model Overview

The model used in this project is **YOLOv8n (nano version)** from the Ultralytics YOLOv8 family.

YOLOv8n is a lightweight and efficient object detection model, suitable for training on smaller datasets while maintaining good performance.

The model was trained to detect three classes:
- Person
- Helmet
- High-visibility vest

---

## 2. Training Configuration

The final model was trained using the following configuration:

| Parameter | Value |
|----------|------|
| Model | YOLOv8n |
| Epochs | 50 |
| Batch Size | 16 |
| Image Size | 640 |
| Optimizer | Auto |
| Learning Rate | 0.01 |
| Weight Decay | 0.0005 |
| Device | GPU (Tesla T4 - Google Colab) |

---

## 3. Data Augmentation

The following augmentation techniques were applied:

- Horizontal flipping
- Brightness and color adjustments (HSV)
- Scaling and translation
- Mosaic augmentation

These augmentations help improve model generalization under different real-world conditions.

---

## 4. Training Process

Multiple experiments were conducted to identify the best training configuration.

### Experiments included:
- Different epoch values (25, 50, 80)
- Dataset improvements and balancing
- Adjustment of training parameters

### Observations:
- 25 epochs resulted in underfitting
- 80 epochs showed slight overfitting
- 40-50 epochs provided the best balance between accuracy and generalization

---

## 5. Final Model Performance

The best model achieved the following results:

| Metric | Value |
|-------|------|
| Precision | 0.85 |
| Recall | 0.65 |
| mAP@50 | 0.725 |
| mAP@50-95 | 0.533 |

These results indicate strong detection performance for a lightweight model.

---

## 6. Class-wise Performance

| Class   | mAP@50-95 |
|--------|----------|
| Helmet | 0.614 |
| Person | 0.549 |
| Vest   | 0.435 |

### Analysis:
- The model performs best on **helmet detection**
- **Person detection** is strong and consistent
- **Vest detection** is comparatively lower but improved through dataset balancing

---

## 7. Performance Analysis

The model demonstrates strong overall performance:

- High precision (0.85) indicates accurate predictions
- Good recall (0.65) shows most objects are detected
- mAP@50-95 of 0.533 reflects reliable detection quality

However:
- Vest detection remains the most challenging class
- Some objects are missed at higher confidence thresholds
- Background confusion is observed in difficult scenarios

Despite these challenges, the model performs effectively for real-world PPE detection tasks.

---

## 8. Recall–Confidence Analysis

The recall-confidence curve shows that:

- Recall is high at low confidence thresholds
- Recall decreases as confidence increases
- The model becomes more strict at higher confidence levels

The vest class has the lowest recall across all confidence levels.

This indicates:
- insufficient vest training data
- harder detection compared to helmet/person

A confidence threshold between **0.3 and 0.5** provides a good balance between detection accuracy and recall.

---

## 9. Confusion Matrix Analysis

The confusion matrix indicates:

- Strong detection performance for person and helmet classes
- Higher misclassification and missed detections for the vest class
- Some objects are incorrectly classified as background

This suggests that:
- The model is reliable but can miss smaller or partially visible objects
- Additional data for the vest class could improve performance

---

## 10. Sample Predictions

Include 3–5 prediction images showing:

- Correct detection of helmet, vest, and person
- Multiple workers in a single image
- Challenging scenarios (occlusion, distance, lighting)

---

## 11. Model Selection Justification

The final model was selected based on:

- Highest mAP@50-95 (0.533)
- Strong balance between precision and recall
- Improved detection performance across all classes
- Efficient performance using a lightweight model (YOLOv8n)

Although larger models may provide higher accuracy, YOLOv8n was sufficient for this project due to its efficiency and strong results.

---

## 12. Screenshots (Proof)

### 📸 Figure 1: Training Results Graph
Include screenshot from:
- runs/detect/train/results.png

---

### 📸 Figure 2: Confusion Matrix
Include screenshot from:
- runs/detect/train/confusion_matrix.png

---

### 📸 Figure 3: Recall-Confidence Curve
Include screenshot from:
- runs/detect/train/R_curve.png

---

### 📸 Figure 4: Sample Predictions
Include images from:
- runs/detect/predict/

---

## 13. Files Used

- Model weights:
  - runs/detect/train/weights/best.pt

- Dataset configuration:
  - data.yaml

- Training environment:
  - Google Colab (GPU enabled)

---

## 14. Conclusion

The YOLOv8n model achieved strong performance for PPE detection, demonstrating that a lightweight model can effectively detect safety equipment in construction environments.

The model successfully identifies persons, helmets, and vests, while also detecting safety violations.

Further improvements can be achieved by:
- Increasing dataset size
- Adding more vest examples
- Improving annotation quality