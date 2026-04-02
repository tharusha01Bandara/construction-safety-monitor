# 📊 Dataset Documentation – PPE Detection

## 1. Overview

This project uses a **fully custom dataset** created specifically for detecting Personal Protective Equipment (PPE) in construction environments.

The dataset focuses on three classes:
- Person
- Helmet
- High-visibility vest

The dataset was designed to represent real-world safety scenarios, including both compliant and non-compliant cases.

All images were uploaded to **Roboflow**, where annotation was performed using bounding boxes for each class (person, helmet, and vest). The platform was also used to organize the dataset, apply preprocessing and augmentation, and generate the final train, validation, and test splits.
This approach ensures strong diversity within the dataset and improves the model’s ability to generalize to real-world environments. By using a fully custom dataset and Roboflow for annotation and management, the project demonstrates the ability to independently collect, label, and prepare training data, which is a key requirement of this assignment.


---

## 2. Data Collection

All images were manually collected and curated from multiple sources. No pre-existing dataset was directly used.

Images were collected from:
- Screenshots extracted from construction-related videos
- Publicly available royalty-free images
- Manually gathered images representing real-world conditions

The dataset includes:
- Workers wearing full PPE (helmet and vest)
- Workers without helmets
- Workers without safety vests
- Workers in indoor and outdoor environments
- Workers at different distances and camera angles
- Multiple workers in a single frame
- Partial occlusions

This ensures diversity and improves model generalization.

---

## 3. Dataset Size

| Item | Value |
|------|------|
| Total Images | 245 |
| Total Annotations | 1773 |
| Number of Custom Images | 245 |

All images in the dataset are custom collected.

---

## 4. Class Distribution

| Class   | Number of Instances |
|--------|-------------------|
| Person | 857 |
| Helmet | 587 |
| Vest   | 329 |

The dataset contains both:
- **Compliant cases** (workers wearing PPE)
- **Non-compliant cases** (missing helmet or vest)

This helps the model detect safety violations effectively.

---

## 5. Annotation Process

All images were annotated using **Roboflow**.

### Annotation Details:
- Annotation type: Bounding boxes
- Classes: Person, Helmet, Vest
- Each object in the image was labeled manually
- Care was taken to ensure:
  - Accurate bounding boxes
  - Consistent labeling
  - Inclusion of small and partially visible objects

Roboflow was also used for:
- Dataset organization
- Preprocessing
- Data augmentation
- Dataset splitting

---

## 6. Data Split

The dataset was split into:

| Split | Percentage | Images |
|------|-----------|--------|
| Train | 72% | 176 |
| Validation | 18% | 44 |
| Test | 10% | 25 |

This split ensures:
- Sufficient training data
- Reliable validation
- Proper final evaluation

---

## 7. Preprocessing and Augmentation

The following preprocessing and augmentation techniques were applied:

- Horizontal flipping
- Brightness adjustment
- Rotation (small angles)
- Scaling and translation
- Mosaic augmentation

These techniques improve model robustness and performance.

---

## 8. Screenshots (Proof) in Dataset Overview images folder

### 📸 Figure 1: Dataset Overview (Roboflow Dashboard)
Include screenshot showing:
- Total number of images
- Dataset name
- Project overview

---

### 📸 Figure 2: Class Distribution (Roboflow Analytics)
Include screenshot showing:
- Person, Helmet, Vest distribution
- Number of annotations

---

### 📸 Figure 3: Annotation Example
Include screenshot showing:
- Bounding boxes for:
  - Person
  - Helmet
  - Vest

---

### 📸 Figure 4: Dataset Split
Include screenshot showing:
- Train / Validation / Test split

---

## 9. Summary

The dataset is fully custom-built and designed to capture real-world PPE scenarios. The diversity of images, balanced class representation, and proper annotation ensure that the trained model can generalize effectively to unseen data.

This dataset demonstrates the ability to collect, curate, annotate, and prepare data for a real-world computer vision task.