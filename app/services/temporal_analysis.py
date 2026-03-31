import cv2
import os
import uuid
import numpy as np
from typing import Tuple, List, Dict, Any
from app.config import settings
from app.services.detector import detector
from app.services.safety_logic import apply_safety_rules

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class WorkerTracker:
    def __init__(self):
        self.next_track_id = 1
        self.tracks = {} # track_id -> dict of state
        self.alerts = []

    def update(self, workers: List[Dict], frame_idx: int) -> List[Dict]:
        """Matches un-tracked workers to existing tracks using IoU."""
        current_tracked_workers = []
        
        unmatched_new = list(workers)
        unmatched_existing = list(self.tracks.keys())
        
        for track_id in unmatched_existing:
            last_bbox = self.tracks[track_id]["last_bbox"]
            best_iou = 0
            best_match_idx = -1
            
            for idx, worker in enumerate(unmatched_new):
                if "bbox" not in worker:
                    continue
                iou = calculate_iou(last_bbox, worker["bbox"])
                if iou > best_iou and iou >= settings.WORKER_MATCH_IOU_THRESHOLD:
                    best_iou = iou
                    best_match_idx = idx
                    
            if best_match_idx != -1:
                # Matched
                matched_worker = unmatched_new.pop(best_match_idx)
                matched_worker["track_id"] = track_id
                self._update_track_state(track_id, matched_worker, frame_idx)
                current_tracked_workers.append(matched_worker)
                
        # Register new workers
        for worker in unmatched_new:
            track_id = self.next_track_id
            self.next_track_id += 1
            worker["track_id"] = track_id
            
            self.tracks[track_id] = {
                "last_bbox": worker.get("bbox", [0, 0, 0, 0]),
                "consecutive_unsafe": 0,
                "consecutive_missing_helmet": 0,
                "consecutive_missing_vest": 0,
                "consecutive_low_conf": 0,
                "alert_triggered": set(),
                "history": []
            }
            self._update_track_state(track_id, worker, frame_idx)
            current_tracked_workers.append(worker)
            
        return current_tracked_workers
        
    def _update_track_state(self, track_id: int, worker: Dict, frame_idx: int):
        state = self.tracks[track_id]
        state["last_bbox"] = worker.get("bbox", state["last_bbox"])
        state["history"].append(frame_idx)
        
        if worker["status"] == "unsafe":
            state["consecutive_unsafe"] += 1
            if not worker["helmet"]:
                state["consecutive_missing_helmet"] += 1
            else:
                state["consecutive_missing_helmet"] = 0
                
            if not worker["vest"]:
                state["consecutive_missing_vest"] += 1
            else:
                state["consecutive_missing_vest"] = 0
        else:
            state["consecutive_unsafe"] = 0
            state["consecutive_missing_helmet"] = 0
            state["consecutive_missing_vest"] = 0
            
        if worker["review_needed"]:
            state["consecutive_low_conf"] += 1
        else:
            state["consecutive_low_conf"] = 0

        self._check_and_raise_alerts(track_id, frame_idx)

    def _check_and_raise_alerts(self, track_id: int, frame_idx: int):
        state = self.tracks[track_id]
        th1 = settings.PERSISTENT_UNSAFE_WORKER_FRAMES
        th2 = settings.PERSISTENT_LOW_CONF_FRAMES
        
        if state["consecutive_missing_helmet"] >= th1 and "helmet" not in state["alert_triggered"]:
            state["alert_triggered"].add("helmet")
            self.alerts.append({
                "track_id": track_id,
                "alert_type": "Persistent Missing Helmet",
                "frames": state["history"][-th1:],
                "message": f"Worker {track_id} missing helmet for {th1} consecutive processed frames"
            })
            
        if state["consecutive_missing_vest"] >= th1 and "vest" not in state["alert_triggered"]:
            state["alert_triggered"].add("vest")
            self.alerts.append({
                "track_id": track_id,
                "alert_type": "Persistent Missing Vest",
                "frames": state["history"][-th1:],
                "message": f"Worker {track_id} missing vest for {th1} consecutive processed frames"
            })
            
        if state["consecutive_low_conf"] >= th2 and "low_conf" not in state["alert_triggered"]:
            state["alert_triggered"].add("low_conf")
            self.alerts.append({
                "track_id": track_id,
                "alert_type": "Persistent Low Confidence",
                "frames": state["history"][-th2:],
                "message": f"Worker {track_id} requiring manual review for {th2} consecutive processed frames"
            })

class TemporalSceneAnalyzer:
    def __init__(self):
        self.consecutive_unsafe_scene = 0
        self.persistent_scene_alert = False

    def update(self, scene_status: str, frame_idx: int):
        if scene_status == "unsafe":
            self.consecutive_unsafe_scene += 1
        else:
            self.consecutive_unsafe_scene = 0
            
        if self.consecutive_unsafe_scene >= settings.PERSISTENT_UNSAFE_SCENE_FRAMES:
            self.persistent_scene_alert = True

def parse_detections(results):
    persons, helmets, vests = [], [], []
    names = results[0].names
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0].tolist()
            c = float(box.conf)
            cls_id = int(box.cls)
            label = names.get(cls_id, "unknown")
            if label == "person":
                persons.append((b, c))
            elif label == "helmet":
                helmets.append((b, c))
            elif label == "vest":
                vests.append((b, c))
    return persons, helmets, vests

def run_temporal_analysis(video_path: str, save_visuals: bool = False) -> Tuple[Dict, str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_video = None
    output_path = ""
    if save_visuals:
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        filename = f"out_video_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(settings.OUTPUT_DIR, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, settings.OUTPUT_VIDEO_FPS, (width, height))

    tracker = WorkerTracker()
    scene_analyzer = TemporalSceneAnalyzer()
    
    frame_idx = 0
    processed_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % settings.VIDEO_FRAME_STRIDE == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = detector.predict(rgb_frame)
            persons, helmets, vests = parse_detections(results)
            
            workers_out, scene_status, scene_conf = apply_safety_rules(persons, helmets, vests)
            
            worker_dicts = []
            for i, w in enumerate(workers_out):
                wd = w.copy() if isinstance(w, dict) else w.dict()
                if i < len(persons):
                    wd["bbox"] = persons[i][0]
                worker_dicts.append(wd)

            tracked_workers = tracker.update(worker_dicts, frame_idx)
            scene_analyzer.update(scene_status, frame_idx)
            
            if save_visuals and out_video is not None:
                annotated = _draw_temporal_frame(frame, tracked_workers, scene_status, frame_idx, scene_analyzer.persistent_scene_alert)
                out_video.write(annotated)
                
            processed_count += 1

        frame_idx += 1

    cap.release()
    if out_video:
        out_video.release()

    video_status = "unsafe" if scene_analyzer.persistent_scene_alert else "safe"
    
    summary = []
    summary.append(f"Temporal Analysis Report")
    summary.append(f"Total Frames Analyzed: {processed_count}")
    summary.append(f"Final Video Status: {video_status.upper()}")
    summary.append(f"Persistent Scene Alert: {'YES' if scene_analyzer.persistent_scene_alert else 'NO'}")
    summary.append(f"Worker Alerts Triggered: {len(tracker.alerts)}")
    for a in tracker.alerts:
        summary.append(f" - [Frame {a['frames'][0]}-{a['frames'][-1]}] {a['message']}")
        
    result_dict = {
        "video_status": video_status,
        "frames_processed": processed_count,
        "persistent_scene_alert": scene_analyzer.persistent_scene_alert,
        "worker_alerts": tracker.alerts,
        "summary_report": "\\n".join(summary)
    }
    
    return result_dict, output_path

def _draw_temporal_frame(frame: np.ndarray, workers: List[Dict], scene_status: str, frame_idx: int, persistent_alert: bool):
    out_frame = frame.copy()
    
    color = (0, 0, 255) if scene_status == 'unsafe' else (0, 255, 0)
    cv2.putText(out_frame, f"Scene: {scene_status.upper()} (Frame {frame_idx})", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
    if persistent_alert:
        cv2.putText(out_frame, "PERSISTENT UNSAFE SCENE DETECTED", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    for w in workers:
        if "bbox" not in w:
            continue
        box = w["bbox"]
        tid = w.get("track_id", "?")
        status = w.get("status", "unknown")
        
        box_color = (0, 0, 255) if status == "unsafe" else (0, 255, 0)
        x1, y1, x2, y2 = map(int, box)
        
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), box_color, 2)
        
        label = f"ID:{tid} {status.upper()}"
        cv2.putText(out_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        violations = w.get("violations", [])
        if violations:
            v_text = ",".join(violations)
            cv2.putText(out_frame, v_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return out_frame