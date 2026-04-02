import os
import uuid
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from ..config import settings

def draw_visualizations(image_np, workers, scene_status, scene_conf=0.0):
    img = PILImage.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i, w in enumerate(workers, start=1):
        # Convert Pydantic model to dict if needed
        worker = w.dict() if hasattr(w, "dict") else w
        
        px1, py1, px2, py2 = map(int, worker["person_box"])
        status = worker["status"]
        person_conf = worker.get("person_conf", 0.0)

        person_color = "green" if status == "safe" else "red"

        draw.rectangle([px1, py1, px2, py2], outline=person_color, width=4)

        if status == "safe":
            person_label = f"W{i}: SAFE ({person_conf:.2f})"
        else:
            person_label = f"W{i}: UNSAFE ({person_conf:.2f})"

        p_text_y = max(0, py1 - 28)
        p_bbox = draw.textbbox((px1, p_text_y), person_label, font=font)
        draw.rectangle(p_bbox, fill=person_color)
        draw.text((px1, p_text_y), person_label, fill="white", font=font)

        if worker.get("helmet") and worker.get("helmet_box") is not None:
            hx1, hy1, hx2, hy2 = map(int, worker["helmet_box"])
            helmet_conf = worker.get("helmet_conf", 0.0)

            draw.rectangle([hx1, hy1, hx2, hy2], outline="blue", width=3)

            helmet_label = f"helmet {helmet_conf:.2f}"
            h_text_y = max(0, hy1 - 22)
            h_bbox = draw.textbbox((hx1, h_text_y), helmet_label, font=font)
            draw.rectangle(h_bbox, fill="blue")
            draw.text((hx1, h_text_y), helmet_label, fill="white", font=font)

        if worker.get("vest") and worker.get("vest_box") is not None:
            vx1, vy1, vx2, vy2 = map(int, worker["vest_box"])
            vest_conf = worker.get("vest_conf", 0.0)

            draw.rectangle([vx1, vy1, vx2, vy2], outline="orange", width=3)

            vest_label = f"vest {vest_conf:.2f}"
            v_text_y = max(0, vy1 - 22)
            v_bbox = draw.textbbox((vx1, v_text_y), vest_label, font=font)
            draw.rectangle(v_bbox, fill="orange")
            draw.text((vx1, v_text_y), vest_label, fill="white", font=font)

    scene_color = "green" if scene_status == "safe" else "red"
    scene_label = f"Scene: {scene_status.upper()} ({scene_conf:.2f})"

    s_bbox = draw.textbbox((10, 10), scene_label, font=font)
    draw.rectangle(s_bbox, fill=scene_color)
    draw.text((10, 10), scene_label, fill="white", font=font)

    if not os.path.exists(settings.OUTPUT_DIR):
        os.makedirs(settings.OUTPUT_DIR)

    filename = f"{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(settings.OUTPUT_DIR, filename)
    img.save(path)

    return path
